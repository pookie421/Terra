#!/usr/bin/env python3
"""Optional CPU merge + llama.cpp GGUF export, followed by local Ollama import.

The direct serve_qwen_copilot.py path needs no conversion. This helper never
merges full BF16 weights on the 16 GB GPU and never assumes Qwen Safetensors LoRA
ADAPTER import is supported by a particular Ollama version. Install llama.cpp's
own conversion requirements separately when choosing conversion.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == 'scripts' else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Qwen2.5 ChatML structure, based on the official model's message/Prompt paths:
# https://ollama.com/library/qwen2.5/blobs/eb4402837c78
# The backend supplies its own system/evidence, once; none is baked in here.
CHAT_TEMPLATE = '''{{- if .Messages }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}{{ else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}{{ end }}{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def available_ram_bytes() -> Optional[int]:
    """Native Windows availability, with no Linux-only or optional hard dependency."""
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                ('total_phys', ctypes.c_ulonglong), ('avail_phys', ctypes.c_ulonglong),
                ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                ('avail_extended', ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_phys)
        return None
    try:
        return int(os.sysconf('SC_AVPHYS_PAGES')) * int(os.sysconf('SC_PAGE_SIZE'))
    except (ValueError, OSError, AttributeError):
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except ImportError:
            return None


def require_merge_ram(minimum_gib: float = 24.) -> int:
    if minimum_gib < 24:
        raise ValueError('BF16 7B CPU merge requires at least a 24 GiB available-RAM guard')
    available = available_ram_bytes()
    if available is None or available < minimum_gib * 1024 ** 3:
        raise RuntimeError('CPU BF16 merge requires at least %.1f GiB available system RAM. '
            'Close other applications or use another machine / --merged-dir. '
            'The adapter remains usable by the NF4 direct server; no GPU merge was attempted.' % minimum_gib)
    return available


def gguf_metadata(path: Path) -> Dict[str, Any]:
    """Read selected GGUF metadata without reading multi-GB tensor payloads."""
    wanted = {'general.architecture', 'tokenizer.chat_template', 'general.file_type',
              'qwen2.context_length', 'qwen2.embedding_length'}
    scalar = {0: ('B', 1), 1: ('b', 1), 2: ('H', 2), 3: ('h', 2),
              4: ('I', 4), 5: ('i', 4), 6: ('f', 4), 7: ('?', 1),
              10: ('Q', 8), 11: ('q', 8), 12: ('d', 8)}
    size = Path(path).stat().st_size
    result = {}
    with Path(path).open('rb') as handle:
        def read(count):
            data = handle.read(count)
            if len(data) != count:
                raise ValueError('Truncated GGUF metadata')
            return data
        def number(fmt):
            return struct.unpack('<' + fmt, read(struct.calcsize(fmt)))[0]
        def string(keep):
            count = number('Q')
            if count > 128 * 1024 ** 2 or handle.tell() + count > size:
                raise ValueError('Invalid GGUF string length')
            if keep:
                return read(count).decode('utf-8')
            handle.seek(count, 1)
        def value(kind, keep=False, array=False):
            if kind in scalar:
                fmt, count = scalar[kind]
                if keep:
                    return struct.unpack('<' + fmt, read(count))[0]
                read(count)
            elif kind == 8:
                return string(keep)
            elif kind == 9 and not array:
                element, count = number('I'), number('Q')
                if count > 10_000_000:
                    raise ValueError('Invalid GGUF array length')
                for _ in range(count):
                    value(element, False, True)
            else:
                raise ValueError('Unsupported or invalid GGUF metadata value type')
        if read(4) != b'GGUF' or number('I') not in (2, 3):
            raise ValueError('Expected a GGUF v2/v3 model')
        number('Q')  # tensor count; tensors are not needed for the metadata check.
        count = number('Q')
        if count > 4096:
            raise ValueError('Unexpected GGUF metadata count')
        for _ in range(count):
            key = string(True)
            item = value(number('I'), key in wanted)
            if key in wanted:
                result[key] = item
    if result.get('general.architecture') != 'qwen2':
        raise ValueError('Expected Qwen2/Qwen2.5 GGUF architecture, not an unrelated model')
    template = result.get('tokenizer.chat_template', '')
    if not isinstance(template, str) or '<|im_start|>' not in template or '<|im_end|>' not in template:
        raise ValueError('GGUF lacks the expected Qwen ChatML tokenizer template; preserve the trained tokenizer during conversion')
    return result


def preserve(path: Path, directory: Path) -> Optional[str]:
    if not path.exists():
        return None
    previous = directory / 'previous_exports'
    previous.mkdir(parents=True, exist_ok=True)
    target = previous / path.name
    counter = 0
    while target.exists():
        counter += 1
        target = previous / (path.name + '.previous-' + str(counter))
    path.rename(target)
    return str(target.relative_to(directory))


def modelfile(gguf_name: str, max_context: int = 8192) -> str:
    if not 1 <= max_context <= 8192:
        raise ValueError('Exported context must not exceed the trained 8192-token budget')
    # Paths are generated by this helper, never arbitrary Modelfile directives.
    if Path(gguf_name).name != gguf_name or any(c in gguf_name for c in '\n\r"'):
        raise ValueError('Invalid GGUF filename')
    return ('FROM "./' + gguf_name + '"\n'
            'PARAMETER num_ctx ' + str(max_context) + '\n'
            'PARAMETER num_predict 512\nPARAMETER temperature 0.1\n'
            'PARAMETER stop "<|im_end|>"\nPARAMETER stop "<|endoftext|>"\n'
            'TEMPLATE """' + CHAT_TEMPLATE + '"""\n')


def merge_cpu(adapter: Path, merged: Path, minimum_gib: float = 24.) -> Dict[str, Any]:
    from scripts.serve_qwen_copilot import adapter_metadata
    available = require_merge_ram(minimum_gib)
    metadata = adapter_metadata(adapter)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(metadata['base_model'],
        revision=metadata['base_model_revision'], torch_dtype=torch.bfloat16,
        device_map={'': 'cpu'}, low_cpu_mem_usage=True, trust_remote_code=False)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    if any(p.device.type != 'cpu' for p in model.parameters()):
        raise RuntimeError('CPU export unexpectedly placed parameters on another device')
    merged_model = model.merge_and_unload(safe_merge=True)
    merged_model.save_pretrained(str(merged), safe_serialization=True, max_shard_size='2GB')
    AutoTokenizer.from_pretrained(str(adapter), local_files_only=True,
        trust_remote_code=False).save_pretrained(str(merged))
    if adapter_metadata(adapter)['adapter_sha256'] != metadata['adapter_sha256']:
        raise ValueError('Adapter changed during CPU export')
    return {**metadata, 'available_system_ram_bytes': available, 'merge_device': 'cpu',
            'merge_dtype': 'bfloat16'}


def find_quantizer(llama_cpp: Path, explicit: Optional[Path] = None) -> Path:
    if explicit:
        if not Path(explicit).is_file():
            raise FileNotFoundError('llama-quantize executable not found: ' + str(explicit))
        return Path(explicit).resolve()
    for relative in ('build/bin/Release/llama-quantize.exe', 'build/bin/llama-quantize.exe',
                     'build/bin/llama-quantize', 'llama-quantize.exe', 'llama-quantize'):
        candidate = Path(llama_cpp) / relative
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError('Install/build llama-quantize from llama.cpp or pass --quantizer. No source compilation is triggered automatically.')


def export_ollama(output_dir: Path, *, adapter: Optional[Path] = None,
                  gguf: Optional[Path] = None, merged_dir: Optional[Path] = None,
                  llama_cpp: Optional[Path] = None, quantizer: Optional[Path] = None,
                  merge: bool = False, create: bool = False,
                  minimum_ram_gib: float = 24., model_name: str = 'terra-shield-qwen') -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/:-]*', model_name):
        raise ValueError('Invalid Ollama model name')
    if gguf and (merge or merged_dir):
        raise ValueError('Choose an existing GGUF or the merge/conversion path')
    if merge and merged_dir:
        raise ValueError('Choose --merge-cpu or --merged-dir, not both')
    if merge and not adapter:
        raise ValueError('--merge-cpu requires a trained --adapter')
    manifest = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'preparing', 'model_name': model_name, 'max_context': 8192,
        'adapter_binding_verified': False, 'ollama_create': 'not_requested',
        'scope': 'Local conversion/import evidence; backend prose verification and target semantic evaluation remain separate'}
    target = output_dir / 'terra-shield-qwen-q4_k_m.gguf'
    with tempfile.TemporaryDirectory(prefix='ollama_export_', dir=str(output_dir)) as temporary:
        stage = Path(temporary)
        candidate = Path(gguf) if gguf else stage / target.name
        if gguf:
            manifest['input_kind'] = 'user_supplied_gguf'
        else:
            if not llama_cpp:
                raise ValueError('Conversion requires --llama-cpp; use the direct adapter server when conversion tools are unavailable')
            converter = Path(llama_cpp) / 'convert_hf_to_gguf.py'
            if not converter.is_file():
                raise FileNotFoundError('Missing llama.cpp convert_hf_to_gguf.py')
            executable = find_quantizer(Path(llama_cpp), quantizer)
            if merge:
                merged_dir = stage / 'merged_bf16'
                manifest.update(merge_cpu(Path(adapter), merged_dir, minimum_ram_gib))
                manifest['adapter_binding_verified'] = True
            elif not merged_dir:
                raise ValueError('Supply --merged-dir or explicitly request --merge-cpu')
            merged_dir = Path(merged_dir)
            if not (merged_dir / 'config.json').is_file() or not list(merged_dir.glob('*.safetensors')):
                raise ValueError('Merged directory must contain full-model config.json and Safetensors weights, not an adapter alone')
            intermediate = stage / 'model-bf16.gguf'
            subprocess.run([sys.executable, str(converter.resolve()), str(merged_dir.resolve()),
                '--outfile', str(intermediate), '--outtype', 'bf16'], check=True)
            subprocess.run([str(executable), str(intermediate), str(candidate), 'Q4_K_M'], check=True)
            manifest.update(input_kind='cpu_merge_and_conversion' if merge else 'user_supplied_merged_model',
                converter_sha256=sha256(converter), quantizer_sha256=sha256(executable))
        metadata = gguf_metadata(candidate)
        candidate_digest = sha256(candidate)
        # A source equal to the existing target is already staged; never rename it out from under itself.
        if candidate.resolve() != target.resolve():
            staged_copy = stage / 'validated-model.gguf'
            if candidate.resolve() != staged_copy.resolve():
                shutil.copyfile(candidate, staged_copy)
            previous = preserve(target, output_dir)
            if previous:
                manifest.setdefault('previous_exports_preserved', {})['gguf'] = previous
            staged_copy.replace(target)
        for name in ('Modelfile', 'ollama_export_manifest.json'):
            previous = preserve(output_dir / name, output_dir)
            if previous:
                manifest.setdefault('previous_exports_preserved', {})[name] = previous
        (output_dir / 'Modelfile').write_text(modelfile(target.name), encoding='utf-8')
        manifest.update(status='exported', gguf=target.name, gguf_sha256=candidate_digest,
                        gguf_bytes=target.stat().st_size,
                        architecture=metadata['general.architecture'],
                        qwen_chatml_template_present=True)
        path = output_dir / 'ollama_export_manifest.json'
        path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        if create:
            try:
                subprocess.run(['ollama', 'create', model_name, '-f', str((output_dir / 'Modelfile').resolve())],
                               cwd=str(output_dir), check=True)
                manifest['ollama_create'] = 'passed'
            except (OSError, subprocess.CalledProcessError) as exc:
                manifest.update(ollama_create='failed', import_error_type=type(exc).__name__)
                path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
                raise RuntimeError('Ollama import failed; validated GGUF and adapter remain available') from exc
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, default=ROOT / 'artifacts/qwen2_5_7b_lora/adapter')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/ollama_qwen')
    parser.add_argument('--gguf', type=Path, help='Import an existing trained Qwen GGUF; adapter inclusion cannot be inferred from this file')
    parser.add_argument('--merged-dir', type=Path, help='Previously merged full BF16 model, including the trained tokenizer')
    parser.add_argument('--merge-cpu', action='store_true', help='Merge the trained adapter into the exact full base on CPU (requires >=24 GiB available RAM)')
    parser.add_argument('--llama-cpp', type=Path, help='User-installed llama.cpp checkout with conversion requirements installed')
    parser.add_argument('--quantizer', type=Path, help='User-installed llama-quantize executable')
    parser.add_argument('--minimum-ram-gib', type=float, default=24.)
    parser.add_argument('--create', action='store_true', help='Run local ollama create after successful export; does not push/upload')
    parser.add_argument('--model-name', default='terra-shield-qwen')
    args = parser.parse_args()
    manifest = export_ollama(args.output_dir, adapter=args.adapter, gguf=args.gguf,
        merged_dir=args.merged_dir, llama_cpp=args.llama_cpp, quantizer=args.quantizer,
        merge=args.merge_cpu, create=args.create, minimum_ram_gib=args.minimum_ram_gib,
        model_name=args.model_name)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
