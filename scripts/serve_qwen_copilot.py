#!/usr/bin/env python3
"""Serve a trained PEFT adapter through the local Ollama-compatible API subset.

Run ``python serve_qwen_copilot.py`` from a prepared training bundle. The default
port is 11434; stop Ollama first, or use --port 11435 and set TERRA_QWEN_URL in the
backend environment. The backend still verifies every generated numeric claim.
No CUDA/model imports occur merely by importing this module for offline tests.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Mapping, Optional

from starlette.requests import Request

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == 'scripts' else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_NAME = 'terra-shield-qwen:latest'
MAX_CONTEXT = 8192
JSON_INSTRUCTION = 'Return one valid JSON object only, without Markdown code fences or surrounding text.'


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def reject_nonfinite_json(value):
    raise ValueError('Non-finite JSON literal: ' + value)


def adapter_metadata(adapter: Path) -> Dict[str, Any]:
    """Resolve the recorded base/revision, never silently substitute another base."""
    adapter = Path(adapter)
    path = adapter / 'adapter_config.json'
    if not path.is_file():
        raise FileNotFoundError('Missing trained adapter_config.json: ' + str(path))
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('peft_type') != 'LORA' or not config.get('base_model_name_or_path'):
        raise ValueError('Expected a LoRA adapter with a recorded base_model_name_or_path')
    weights = sorted(adapter.glob('adapter_model*.safetensors'))
    if not weights:
        raise FileNotFoundError('No adapter_model*.safetensors weights in ' + str(adapter))
    training = {}
    metadata_path = None
    # The record inside an adapter belongs to those weights. A later failed run
    # may replace the parent's run log while leaving a previous adapter valid.
    for candidate in (adapter / 'training_metadata.json', adapter / 'adapter_manifest.json',
                      adapter.parent / 'training_run.json'):
        if candidate.is_file():
            training = json.loads(candidate.read_text(encoding='utf-8'))
            metadata_path = candidate
            break
    base = config['base_model_name_or_path']
    declared = training.get('base_model')
    if declared and declared != base:
        raise ValueError('Adapter configuration and training manifest disagree on base model')
    revision = training.get('base_model_revision') or config.get('revision')
    if not isinstance(base, str) or not re.fullmatch(r'[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*', base):
        raise ValueError('Serving/export requires a recorded Hugging Face repository and immutable revision; unverified local base directories are unsupported')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Missing immutable base-model revision; refusing to silently load latest weights for this adapter')
    digest = hashlib.sha256()
    for item in [path, *weights, *([metadata_path] if metadata_path else [])]:
        digest.update(item.name.encode('utf-8'))
        with item.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    return {'base_model': base, 'base_model_revision': revision,
            'adapter_sha256': digest.hexdigest(), 'adapter_files': [p.name for p in weights]}


class HFAdapterEngine:
    """One NF4/BF16 CUDA model, native SDPA and last-token-only output projection."""
    def __init__(self, adapter: Path):
        import torch
        from scripts.hardware_preflight import cuda_preflight
        from scripts.qlora_runtime import enable_fused_sdpa
        self.hardware = cuda_preflight(require_bf16=True, check_bitsandbytes=True)
        # Keep native Flash/efficient/cuDNN SDPA eligible and prohibit a silent
        # quadratic math-attention allocation during an 8192-token prefill.
        enable_fused_sdpa(torch)
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.metadata = adapter_metadata(adapter)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        self.device = torch.device('cuda', torch.cuda.current_device())
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(self.metadata['base_model'],
            revision=self.metadata['base_model_revision'], quantization_config=quantization,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map={'': torch.cuda.current_device()}, low_cpu_mem_usage=True,
            trust_remote_code=False)
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        self.model.eval()
        self.model.config.use_cache = True
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True,
                                                       trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(json.dumps({'service': MODEL_NAME, 'hardware': self.hardware,
            'attention': 'sdpa', 'compute_dtype': 'bfloat16', 'quantization': 'nf4',
            'max_context': MAX_CONTEXT, **self.metadata}, indent=2), flush=True)

    def render(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, prompt: str) -> int:
        return len(self.tokenizer(prompt, add_special_tokens=False)['input_ids'])

    def generate(self, prompt: str, max_new_tokens: int, options: Mapping[str, Any]) -> Dict[str, Any]:
        torch = self.torch
        tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        count = int(tokens['input_ids'].shape[-1])
        kwargs = {'max_new_tokens': max_new_tokens, 'use_cache': True,
            'pad_token_id': self.tokenizer.pad_token_id, 'eos_token_id': self.tokenizer.eos_token_id,
            # Qwen2 prefill must not allocate [1, prompt_length, vocabulary] logits.
            'logits_to_keep': 1, 'do_sample': options['temperature'] > 0,
            'repetition_penalty': options['repeat_penalty']}
        if kwargs['do_sample']:
            kwargs.update(temperature=options['temperature'], top_p=options['top_p'], top_k=options['top_k'])
        if options['stop']:
            kwargs.update(stop_strings=options['stop'], tokenizer=self.tokenizer)
        started = time.perf_counter_ns()
        # The service lock encloses this function, including the optional RNG seed.
        seed_context = torch.random.fork_rng(devices=[self.device.index]) if options.get('seed') is not None else nullcontext()
        with torch.inference_mode(), seed_context:
            if options.get('seed') is not None:
                torch.manual_seed(options['seed'])
            generated = self.model.generate(**tokens, **kwargs)
            torch.cuda.synchronize()
        completion = generated[0, count:]
        text = self.tokenizer.decode(completion, skip_special_tokens=True)
        for stop in options['stop']:
            if stop in text:
                text = text.split(stop, 1)[0]
        return {'text': text, 'eval_count': int(completion.numel()),
                'eval_duration': time.perf_counter_ns() - started}


@dataclass
class ServiceConfig:
    model_name: str = MODEL_NAME
    max_context: int = MAX_CONTEXT
    max_new_tokens: int = 1536


class CopilotService:
    def __init__(self, engine: Any, config: Optional[ServiceConfig] = None):
        self.engine = engine
        self.config = config or ServiceConfig()
        if not (1 <= self.config.max_new_tokens < self.config.max_context <= MAX_CONTEXT):
            raise ValueError('Completion/context budget must fit the declared 8192-token context')
        self.lock = threading.Lock()

    def _options(self, raw: Any) -> Dict[str, Any]:
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise RequestError('options must be an object')
        supported = {'temperature', 'num_predict', 'num_ctx', 'seed', 'top_p', 'top_k', 'repeat_penalty', 'stop'}
        if set(raw) - supported:
            raise RequestError('Unsupported options: ' + ', '.join(sorted(set(raw) - supported)))
        result = {'temperature': .1, 'num_predict': self.config.max_new_tokens,
            'num_ctx': self.config.max_context, 'top_p': .9, 'top_k': 40,
            'repeat_penalty': 1., 'stop': [], 'seed': None, **raw}
        for key in ('num_predict', 'num_ctx', 'top_k'):
            if type(result[key]) is not int or result[key] <= 0:
                raise RequestError(key + ' must be a positive integer')
        if result['num_predict'] > self.config.max_new_tokens:
            raise RequestError('num_predict exceeds the configured completion budget')
        for key in ('temperature', 'top_p', 'repeat_penalty'):
            value = result[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RequestError(key + ' must be finite')
        if result['temperature'] < 0 or not 0 < result['top_p'] <= 1 or result['repeat_penalty'] <= 0:
            raise RequestError('Invalid temperature, top_p or repeat_penalty')
        if result['seed'] is not None and type(result['seed']) is not int:
            raise RequestError('seed must be an integer')
        if not isinstance(result['stop'], list) or any(not isinstance(s, str) or not s for s in result['stop']):
            raise RequestError('stop must be a list of nonempty strings')
        # A request may advertise a larger context, but no text is truncated to fit.
        result['num_ctx'] = min(result['num_ctx'], self.config.max_context)
        return result

    def respond(self, body: Dict[str, Any], chat: bool) -> Dict[str, Any]:
        if not isinstance(body, dict):
            raise RequestError('Request body must be an object')
        supported = {'model', 'messages' if chat else 'prompt', 'stream', 'format', 'options'}
        if not chat:
            supported |= {'system', 'raw'}
        if set(body) - supported:
            raise RequestError('Unsupported request fields: ' + ', '.join(sorted(set(body) - supported)))
        model = body.get('model')
        if model not in (self.config.model_name, self.config.model_name.split(':', 1)[0]):
            raise RequestError('Requested model is not loaded', 404)
        if 'stream' in body and type(body['stream']) is not bool:
            raise RequestError('stream must be a boolean')
        output_format = body.get('format')
        if output_format not in (None, '', 'json'):
            raise RequestError('This local endpoint supports format="json", not arbitrary JSON-schema decoding')
        options = self._options(body.get('options'))
        with self.lock:
            if chat:
                messages = body.get('messages')
                if not isinstance(messages, list) or not messages:
                    raise RequestError('messages must be a nonempty list')
                messages = [self._message(m) for m in messages]
                if output_format == 'json':
                    messages = [dict(m) for m in messages]
                    if messages[0]['role'] == 'system':
                        messages[0]['content'] += '\n' + JSON_INSTRUCTION
                    else:
                        messages.insert(0, {'role': 'system', 'content': JSON_INSTRUCTION})
                prompt = self.engine.render(messages)
            else:
                prompt = body.get('prompt', '')
                system = body.get('system', '')
                if not isinstance(prompt, str) or not isinstance(system, str):
                    raise RequestError('prompt and system must be strings')
                raw = body.get('raw', False)
                if type(raw) is not bool:
                    raise RequestError('raw must be a boolean')
                if raw:
                    if system:
                        raise RequestError('raw=true bypasses templating; put any system text in the raw prompt')
                else:
                    if output_format == 'json':
                        system = (system + '\n' + JSON_INSTRUCTION).strip()
                    messages = ([{'role': 'system', 'content': system}] if system else [])
                    prompt = self.engine.render([*messages, {'role': 'user', 'content': prompt}])
            prompt_count = self.engine.count_tokens(prompt)
            if prompt_count + options['num_predict'] > options['num_ctx']:
                raise RequestError('Prompt plus reserved completion exceeds the effective context; shorten evidence/history or request fewer output tokens. No input was truncated.')
            started = time.perf_counter_ns()
            output = self.engine.generate(prompt, options['num_predict'], options)
            text = output['text']
            if output_format == 'json':
                try:
                    parsed = json.loads(text, parse_constant=reject_nonfinite_json)
                    if not isinstance(parsed, dict):
                        raise ValueError('JSON object required')
                except (TypeError, ValueError) as exc:
                    raise RequestError('Model did not generate a valid JSON object; output withheld, not repaired or fabricated', 422) from exc
            result = {'model': self.config.model_name,
                'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'done': True, 'done_reason': 'length' if output['eval_count'] >= options['num_predict'] else 'stop',
                'total_duration': time.perf_counter_ns() - started, 'prompt_eval_count': prompt_count,
                'eval_count': output['eval_count'], 'eval_duration': output.get('eval_duration', 0),
                'context_window': options['num_ctx']}
            result['message' if chat else 'response'] = {'role': 'assistant', 'content': text} if chat else text
            return result

    @staticmethod
    def _message(message: Any) -> Dict[str, str]:
        if not isinstance(message, dict) or set(message) - {'role', 'content'}:
            raise RequestError('Each message requires only role and text content; images/tool calls are unsupported')
        if message.get('role') not in ('system', 'user', 'assistant', 'tool') or not isinstance(message.get('content'), str):
            raise RequestError('Invalid role or non-text message content')
        return {'role': message['role'], 'content': message['content']}


def create_app(engine: Any, config: Optional[ServiceConfig] = None):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
    from starlette.concurrency import run_in_threadpool
    service = CopilotService(engine, config)
    app = FastAPI(title='Local Qwen adapter service', version='1.0')
    app.state.copilot = service

    @app.get('/api/tags')
    def tags():
        return {'models': [{'name': service.config.model_name, 'model': service.config.model_name,
            'details': {'format': 'peft', 'family': 'qwen2', 'quantization_level': 'NF4'}}]}

    @app.get('/healthz')
    def health():
        return {'status': 'ready', 'model': service.config.model_name,
            'max_context': service.config.max_context, 'max_new_tokens': service.config.max_new_tokens,
            'hardware': getattr(engine, 'hardware', {}), 'adapter': getattr(engine, 'metadata', {})}

    async def generate_response(request: Request, chat: bool):
        try:
            body = await request.json()
            result = await run_in_threadpool(service.respond, body, chat)
        except RequestError as exc:
            return JSONResponse({'error': str(exc)}, status_code=exc.status)
        except (ValueError, UnicodeDecodeError):
            return JSONResponse({'error': 'Invalid JSON request'}, status_code=400)
        # Buffered NDJSON: one final record, deliberately not fake incremental tokens.
        if body.get('stream', True):
            return StreamingResponse(iter([json.dumps(result, ensure_ascii=False) + '\n']), media_type='application/x-ndjson')
        return JSONResponse(result)

    @app.post('/api/generate')
    async def generate(request: Request):
        return await generate_response(request, False)

    @app.post('/api/chat')
    async def chat(request: Request):
        return await generate_response(request, True)

    return app


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, default=ROOT / 'artifacts/qwen25_terra_lora/adapter')
    parser.add_argument('--host', choices=['127.0.0.1', 'localhost', '::1'], default='127.0.0.1')
    parser.add_argument('--port', type=int, default=11434)
    parser.add_argument('--max-new-tokens', type=int, default=1536)
    parser.add_argument('--auto-port', action='store_true', help='Try loopback 8001 if the requested local port is occupied')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535')
    for candidate in dict.fromkeys([args.port, 8001] if args.auto_port else [args.port]):
        try:
            with socket.socket(socket.AF_INET6 if args.host == '::1' else socket.AF_INET) as probe:
                probe.bind((args.host, candidate))
            args.port = candidate
            break
        except OSError:
            if candidate == (8001 if args.auto_port else args.port):
                parser.error('Local service ports are occupied; select an available --port. Existing services are left running.')
    config = ServiceConfig(max_new_tokens=args.max_new_tokens)
    engine = HFAdapterEngine(args.adapter)
    endpoint = ROOT / 'artifacts/qwen_copilot_endpoint.json'
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    host = '[::1]' if args.host == '::1' else args.host
    value = {'url': f'http://{host}:{args.port}', 'model': MODEL_NAME,
             'source': 'local_trained_adapter_server', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'adapter_sha256': engine.metadata['adapter_sha256']}
    temporary = endpoint.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8'); temporary.replace(endpoint)
    import uvicorn
    try:
        uvicorn.run(create_app(engine, config), host=args.host, port=args.port, workers=1, limit_concurrency=8)
    finally:
        if endpoint.exists() and json.loads(endpoint.read_text()).get('url') == value['url']:
            endpoint.unlink()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
