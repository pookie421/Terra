"""Offline protocol/export tests; no CUDA model downloads or GPU claims."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from scripts.serve_qwen_copilot import (CopilotService, MODEL_NAME, RequestError,
    ServiceConfig, adapter_metadata, create_app)
from scripts.export_qwen_ollama import (export_ollama, gguf_metadata, modelfile,
    require_merge_ram)


class FakeEngine:
    def __init__(self, text='{"markdown":"Conditional evidence only","claims":{}}', count=20):
        self.text = text
        self.count = count
        self.calls = []
        self.active = self.maximum_active = 0
        self.counter = threading.Lock()

    def render(self, messages):
        self.messages = messages
        return 'CHATML:' + json.dumps(messages)

    def count_tokens(self, prompt):
        return self.count

    def generate(self, prompt, max_new_tokens, options):
        with self.counter:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(.01)
        self.calls.append((prompt, max_new_tokens, options))
        with self.counter:
            self.active -= 1
        return {'text': self.text, 'eval_count': 10, 'eval_duration': 1000}


class ServingProtocol(unittest.TestCase):
    def body(self, **extra):
        return {'model': MODEL_NAME, 'messages': [{'role': 'system', 'content': 'Use ACTIVE_EVIDENCE'},
            {'role': 'user', 'content': 'unchanged evidence'}], 'stream': False, 'format': 'json', **extra}

    def test_backend_chat_payload_and_full_evidence_preserved(self):
        engine = FakeEngine()
        result = CopilotService(engine).respond(self.body(options={'temperature': .1, 'num_ctx': 16384}), True)
        self.assertEqual(result['message']['content'], engine.text)
        self.assertEqual(result['context_window'], 8192)
        self.assertEqual(engine.messages[0]['role'], 'system')
        self.assertIn('Use ACTIVE_EVIDENCE', engine.messages[0]['content'])
        self.assertEqual(sum(m['role'] == 'system' for m in engine.messages), 1)
        self.assertEqual(engine.messages[1]['content'], 'unchanged evidence')
        self.assertFalse('VERIFIED_LLM_PROSE' in result)

    def test_raw_generate_is_not_double_templated(self):
        engine = FakeEngine(text='raw answer')
        result = CopilotService(engine).respond({'model': 'terra-shield-qwen', 'prompt': '<|im_start|>user\nraw', 'raw': True, 'stream': False}, False)
        self.assertEqual(engine.calls[0][0], '<|im_start|>user\nraw')
        self.assertEqual(result['response'], 'raw answer')
        self.assertFalse(hasattr(engine, 'messages'))

    def test_context_overflow_is_rejected_without_generation_or_truncation(self):
        engine = FakeEngine(count=8190)
        with self.assertRaisesRegex(RequestError, 'No input was truncated'):
            CopilotService(engine).respond(self.body(), True)
        self.assertEqual(engine.calls, [])

    def test_invalid_or_nonfinite_generated_json_is_withheld(self):
        for text in ('not JSON', '[1,2]', '{"score":NaN}'):
            with self.subTest(text=text):
                with self.assertRaises(RequestError) as caught:
                    CopilotService(FakeEngine(text)).respond(self.body(), True)
                self.assertEqual(caught.exception.status, 422)

    def test_unsupported_schema_tools_and_unbounded_output_are_explicit(self):
        service = CopilotService(FakeEngine())
        for body in (self.body(format={'type': 'object'}), self.body(tools=[]),
                     self.body(options={'num_predict': -1}), self.body(options={'num_predict': 1537}),
                     self.body(options={'temperature': float('nan')})):
            with self.subTest(body=body), self.assertRaises(RequestError):
                service.respond(body, True)

    def test_gpu_generation_is_serialized(self):
        engine = FakeEngine()
        service = CopilotService(engine)
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: service.respond(self.body(), True), range(3)))
        self.assertEqual(len(engine.calls), 3)
        self.assertEqual(engine.maximum_active, 1)


class ServingHTTP(unittest.IsolatedAsyncioTestCase):
    async def test_tags_chat_generate_and_ndjson(self):
        app = create_app(FakeEngine())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            tags = await client.get('/api/tags')
            self.assertEqual(tags.json()['models'][0]['name'], MODEL_NAME)
            chat = await client.post('/api/chat', json={'model': MODEL_NAME, 'messages': [{'role': 'user', 'content': 'inspect'}], 'format': 'json', 'stream': False})
            self.assertEqual(chat.status_code, 200, chat.text)
            self.assertTrue(chat.json()['done'])
            generated = await client.post('/api/generate', json={'model': MODEL_NAME, 'prompt': 'inspect', 'stream': True})
            self.assertEqual(generated.status_code, 200, generated.text)
            self.assertIn('application/x-ndjson', generated.headers['content-type'])
            lines = generated.text.strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertTrue(json.loads(lines[0])['done'])
            missing = await client.post('/api/chat', json={'model': 'other', 'messages': []})
            self.assertEqual(missing.status_code, 404)


def write_gguf(path, architecture='qwen2', template='<|im_start|>user\n{{ content }}<|im_end|>'):
    entries = [('general.architecture', architecture), ('tokenizer.chat_template', template)]
    def string(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw
    body = b'GGUF' + struct.pack('<IQQ', 3, 0, len(entries))
    for key, value in entries:
        body += string(key) + struct.pack('<I', 8) + string(value)
    path.write_bytes(body)


class ExportProtocol(unittest.TestCase):
    def test_adapter_metadata_requires_matching_base_and_actual_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'adapter_config.json').write_text(json.dumps({'peft_type': 'LORA', 'base_model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct'}))
            with self.assertRaises(FileNotFoundError):
                adapter_metadata(root)
            (root / 'adapter_model.safetensors').write_bytes(b'test-only-placeholder')
            (root / 'adapter_manifest.json').write_text(json.dumps({'base_model': 'wrong-base'}))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                adapter_metadata(root)

    def test_adapter_uses_pinned_metadata_inside_saved_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / 'adapter'
            adapter.mkdir()
            base = 'Qwen/Qwen2.5-7B-Instruct'
            (adapter / 'adapter_config.json').write_text(json.dumps({'peft_type': 'LORA', 'base_model_name_or_path': base}))
            (adapter / 'adapter_model.safetensors').write_bytes(b'not-a-loaded-model')
            with self.assertRaisesRegex(ValueError, 'immutable'):
                adapter_metadata(adapter)
            (adapter / 'training_metadata.json').write_text(json.dumps({'base_model': base, 'base_model_revision': 'a'*40}))
            (root / 'training_run.json').write_text(json.dumps({'base_model': 'new-failed-run', 'base_model_revision': 'b'*40}))
            actual = adapter_metadata(adapter)
            self.assertEqual(actual['base_model_revision'], 'a'*40)
            first = actual['adapter_sha256']
            (adapter / 'training_metadata.json').write_text(json.dumps({'base_model': base, 'base_model_revision': 'c'*40}))
            self.assertNotEqual(adapter_metadata(adapter)['adapter_sha256'], first)

    def test_gguf_architecture_and_chat_template_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.gguf'
            write_gguf(path)
            self.assertEqual(gguf_metadata(path)['general.architecture'], 'qwen2')
            write_gguf(path, architecture='llama')
            with self.assertRaisesRegex(ValueError, 'unrelated'):
                gguf_metadata(path)
            write_gguf(path, template='missing ChatML')
            with self.assertRaisesRegex(ValueError, 'ChatML'):
                gguf_metadata(path)

    def test_cpu_merge_ram_guard_does_not_try_loading_model(self):
        with patch('scripts.export_qwen_ollama.available_ram_bytes', return_value=16 * 1024**3):
            with self.assertRaisesRegex(RuntimeError, 'no GPU merge'):
                require_merge_ram()

    def test_existing_gguf_is_not_claimed_adapter_verified_and_old_export_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.gguf'
            output = root / 'output'
            output.mkdir()
            write_gguf(source)
            target = output / 'terra-shield-qwen-q4_k_m.gguf'
            target.write_bytes(b'previous output preserved')
            manifest = export_ollama(output, gguf=source)
            self.assertFalse(manifest['adapter_binding_verified'])
            self.assertEqual(manifest['ollama_create'], 'not_requested')
            self.assertEqual((output / 'previous_exports' / target.name).read_bytes(), b'previous output preserved')
            self.assertEqual(target.read_bytes(), source.read_bytes())
            text = (output / 'Modelfile').read_text()
            self.assertIn('PARAMETER num_ctx 8192', text)
            self.assertIn('.Messages', text)
            self.assertIn('.Prompt', text)
            self.assertNotIn('ADAPTER ', text)
            self.assertNotIn('SYSTEM ', text)

    def test_failed_local_import_keeps_validated_gguf_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.gguf'
            write_gguf(source)
            with patch('scripts.export_qwen_ollama.subprocess.run', side_effect=OSError('Ollama not installed')):
                with self.assertRaisesRegex(RuntimeError, 'remain available'):
                    export_ollama(root / 'output', gguf=source, create=True)
            manifest = json.loads((root / 'output/ollama_export_manifest.json').read_text())
            self.assertEqual(manifest['ollama_create'], 'failed')
            self.assertTrue((root / 'output/terra-shield-qwen-q4_k_m.gguf').exists())

    def test_modelfile_cannot_inject_directives_or_expand_context(self):
        for name in ('../model.gguf', 'x\nSYSTEM fake'):
            with self.assertRaises(ValueError):
                modelfile(name)
        with self.assertRaises(ValueError):
            modelfile('model.gguf', 16384)


if __name__ == '__main__':
    unittest.main()
