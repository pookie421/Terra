"""Native Windows BF16/NF4 Qwen2.5 QLoRA with a measured 16-GB VRAM gate.

Full runs always test the longest complete trajectory before training. The
--smoke-test mode performs that test and exits without fitting the corpus.
No Torch, model download or CUDA initialization occurs on module import.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import platform
import random
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == 'scripts' else HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.qlora_runtime import (MAX_SEQUENCE_LENGTH, LOSS_CHUNK_TOKENS, VRAM_LIMIT_BYTES,
    GRADIENT_ACCUMULATION, DynamicCausalCollator, backbone_loss, enable_fused_sdpa,
    restore_frozen_embedding_dtypes, restore_sdpa, tokenize_trajectories, training_policy)


def preserve_previous(path: Path, output_dir: Path) -> str:
    archive = output_dir / 'previous_exports'
    archive.mkdir(parents=True, exist_ok=True)
    destination = archive / path.name
    counter = 0
    while destination.exists():
        counter += 1
        destination = archive / (path.name + '.previous-' + str(counter))
    path.rename(destination)
    return str(destination.relative_to(output_dir))


def validate_chatml(path: Path, expected_rows=None) -> int:
    if not path.exists():
        raise FileNotFoundError('source-linked ChatML trajectory file is missing: {}'.format(path))
    seen = set()
    for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        identity = str(row.get('source_event_id') or '')
        messages = row.get('messages')
        if not identity or not isinstance(messages, list) or not messages:
            raise ValueError('ChatML row {} lacks a source ID or messages'.format(number))
        if identity in seen:
            raise ValueError('duplicate source-linked ChatML trajectory: {}'.format(identity))
        if not any(message.get('role') == 'tool' for message in messages):
            raise ValueError('ChatML row {} does not contain a source observation tool result'.format(number))
        seen.add(identity)
    if not seen:
        raise ValueError('Empty instruction corpus')
    if expected_rows is not None and len(seen) != expected_rows:
        raise ValueError('expected {} distinct trajectories, found {}'.format(expected_rows, len(seen)))
    return len(seen)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def corpus_provenance(dataset_path, dataset_sha256, row_count):
    """Bind retained SFT to its actual fusion model on every trainer entry path."""
    dataset_path = Path(dataset_path)
    artifacts = dataset_path.parent.parent / 'artifacts'
    report_path = artifacts / 'sft_trajectory_report.json'
    if report_path.exists():
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        if report.get('sha256') == dataset_sha256:
            fusion = artifacts / 'physics_informed_models.pkl'
            if not fusion.exists():
                raise ValueError('SFT provenance requires its recorded fusion artifact')
            fusion_sha = hashlib.sha256(fusion.read_bytes()).hexdigest()
            if report.get('fusion_model_sha256') != fusion_sha or report.get('rows') != row_count:
                raise ValueError('SFT provenance differs from the actual fusion model or trajectory count; regenerate the corpus')
            return {'fusion_binding_status': 'verified_against_corpus_report',
                    'fusion_model_sha256': fusion_sha,
                    'sft_trajectory_report_sha256': hashlib.sha256(report_bytes).hexdigest()}
    if dataset_path.name == 'sft_context_trajectories.jsonl':
        raise ValueError('Retained SFT corpus is missing a matching provenance report; regenerate and verify it before training')
    # A custom corpus can train without a fusion claim. Directory proximity
    # alone must never make its adapter claim the neighboring model's hash.
    return {'fusion_binding_status': 'unverified_custom_corpus'}


def export_trained_model(model: Any, tokenizer: Any, output_dir: Path, gguf_path: Path,
                         metadata=None) -> Dict[str, Any]:
    """Save the adapter first; CPU GGUF conversion is a separate explicit helper.

    A 16-GB GPU is not used to materialize a merged 7B model. Existing merged or
    GGUF exports are left intact and never relabeled fresh successful exports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = output_dir / 'adapter'
    preserved = {}
    for previous in (adapter, output_dir / 'export_manifest.json'):
        if previous.exists():
            preserved[previous.name] = preserve_previous(previous, output_dir)
    model.save_pretrained(str(adapter), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter))
    if metadata is not None:
        atomic_json(adapter / 'training_metadata.json', metadata)
    hashes = {str(p.relative_to(adapter)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(adapter.rglob('*')) if p.is_file()}
    manifest = {'adapter': {'status': 'saved', 'path': 'adapter', 'sha256': hashes},
                'merged_16bit': {'status': 'not_run_requires_export_helper'},
                'gguf': {'status': 'not_run_requires_export_helper', 'requested_path': str(gguf_path)},
                'direct_serving': 'python serve_qwen_copilot.py --adapter "{}"'.format(adapter),
                'ollama_export': 'python export_qwen_ollama.py --help',
                'previous_exports_preserved': preserved}
    atomic_json(output_dir / 'export_manifest.json', manifest)
    return manifest


def _trainer_class():
    from transformers import Trainer

    class BoundedCausalTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # We normalize each batch over shifted, nonpadding targets ourselves.
            # Otherwise Qwen's **kwargs makes HF skip accumulation division.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch
            loss = backbone_loss(model, inputs)
            if not torch.isfinite(loss).item():
                raise FloatingPointError('Non-finite training loss; refusing an invalid adapter export')
            return (loss, {'loss': loss}) if return_outputs else loss

    return BoundedCausalTrainer


def _verify_attention(torch, backend):
    if backend == 'sdpa':
        return {'implementation': 'sdpa', 'math_fallback': False}
    if backend != 'flash_attention_2':
        raise ValueError('Unknown attention backend')
    try:
        from flash_attn import flash_attn_func
        q = torch.randn((1, 32, 4, 64), device='cuda', dtype=torch.bfloat16, requires_grad=True)
        flash_attn_func(q, q, q, dropout_p=0., causal=True).float().square().mean().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(q.grad).all().item():
            raise RuntimeError('Non-finite FlashAttention gradient')
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError('Optional FlashAttention-2 failed its active-GPU kernel check; use --attention-backend sdpa. '
                           'No source compilation is attempted.') from exc
    return {'implementation': backend, 'kernel_backward': 'passed'}


def _load_model(base_model, revision, attention_backend, local_files_only):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=['lm_head'])
    model = AutoModelForCausalLM.from_pretrained(base_model, revision=revision,
        local_files_only=local_files_only, trust_remote_code=False, torch_dtype=torch.bfloat16,
        quantization_config=quantization, device_map={'': torch.cuda.current_device()},
        attn_implementation=attention_backend, low_cpu_mem_usage=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    restore_frozen_embedding_dtypes(model, torch)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0., bias='none',
        task_type='CAUSAL_LM', revision=revision,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n for n, _ in trainable):
        raise RuntimeError('Only decoder LoRA parameters may be trainable in the bounded-head contract')
    return model


def _memory_result(torch, device, minimum_free):
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    limit = min(VRAM_LIMIT_BYTES, total - 1_000_000_000)
    return {'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved,
            'peak_allocated_gb': allocated / 1e9, 'peak_allocated_gib': allocated / 1024**3,
            'minimum_sampled_free_bytes': min(minimum_free, free), 'total_device_bytes': total,
            'budget_bytes': limit, 'budget_units': 'decimal GB (14.5e9 bytes maximum)',
            'passed': allocated < limit and reserved < limit and min(minimum_free, free) >= 1_000_000_000,
            'scope': 'Torch peaks plus sampled device free memory; paging does not protect activations or guarantee freedom from OOM'}


def longest_trajectory_smoke(model, dataset, collator, report_path, optimizer_factory=None):
    """Two real optimizer steps with the full accumulation policy, then restore.

    The longest example is repeated only for this gate, not added to the fit.
    Adapter parameters and Python/NumPy/Torch RNG are restored even on failure.
    """
    import numpy as np
    import torch
    if optimizer_factory is None:
        import bitsandbytes as bnb
        optimizer_factory = bnb.optim.PagedAdamW8bit
    device = torch.cuda.current_device()
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    saved = {n: p.detach().cpu().clone() for n, p in parameters.items()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state(device))
    report = {'status': 'running', 'requested_optimizer_steps': 2, 'optimizer_steps': 0, 'forward_backward_passes': 0, 'gradient_accumulation_steps': GRADIENT_ACCUMULATION,
              'trajectory_tokens': dataset.lengths[dataset.longest_index],
              'trajectory_source_id': dataset.identities[dataset.longest_index],
              'full_training_performed': False, 'truncation': False}
    report_path = Path(report_path)
    if report_path.exists():
        preserve_previous(report_path, report_path.parent)
    optimizer = None
    was_training = model.training
    losses = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    minimum_free = torch.cuda.mem_get_info(device)[0]
    try:
        if not _memory_result(torch, device, minimum_free)['passed']:
            raise RuntimeError('VRAM preflight model baseline exceeds the memory budget; training refused')
        batch = {k: v.to(device) for k, v in collator([dataset[dataset.longest_index]]).items()}
        optimizer = optimizer_factory(list(parameters.values()), lr=2e-4, weight_decay=.01)
        model.train()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(GRADIENT_ACCUMULATION):
                loss = backbone_loss(model, batch)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Non-finite longest-trajectory loss')
                (loss / GRADIENT_ACCUMULATION).backward()
                losses.append(float(loss.detach().item()))
                torch.cuda.synchronize(device)
                minimum_free = min(minimum_free, torch.cuda.mem_get_info(device)[0])
                report['forward_backward_passes'] += 1
                if not _memory_result(torch, device, minimum_free)['passed']:
                    raise RuntimeError('VRAM preflight exceeded its memory budget; full training refused')
            if not any(p.grad is not None for p in parameters.values()):
                raise RuntimeError('No adapter gradients reached the decoder')
            torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            if any(not torch.isfinite(p).all().item() for p in parameters.values()):
                raise FloatingPointError('Non-finite adapter parameters after smoke optimizer step')
            minimum_free = min(minimum_free, torch.cuda.mem_get_info(device)[0])
            report['optimizer_steps'] += 1
            if not _memory_result(torch, device, minimum_free)['passed']:
                raise RuntimeError('VRAM preflight optimizer step exceeded its memory budget; full training refused')
            print('Longest-trajectory preflight optimizer step {}/2 complete'.format(step + 1), flush=True)
        if not any(not torch.equal(p.detach().cpu(), saved[n]) for n, p in parameters.items()):
            raise RuntimeError('No adapter update observed in the real optimizer smoke')
        report.update(_memory_result(torch, device, minimum_free), losses=losses, adapter_update_observed=True)
        if not report['passed']:
            raise RuntimeError('VRAM preflight exceeded the conservative allocation/reserve/free-memory budget; full training refused')
        report['status'] = 'passed'
    except BaseException as exc:
        try:
            report.update(_memory_result(torch, device, minimum_free))
        except BaseException as measurement_error:
            report['memory_measurement_error'] = type(measurement_error).__name__ + ': ' + str(measurement_error)
        report.update(status='failed', passed=False, error_type=type(exc).__name__, error=str(exc), losses=losses)
        raise
    finally:
        # Release paged state before restoring weights, including on CUDA OOM.
        if optimizer is not None:
            try:
                optimizer.zero_grad(set_to_none=True)
            except BaseException as cleanup_error:
                report['optimizer_cleanup_error'] = type(cleanup_error).__name__ + ': ' + str(cleanup_error)
        optimizer = None
        gc.collect()
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(saved[name])
            model.zero_grad(set_to_none=True)
            model.train(was_training)
            random.setstate(rng[0]); np.random.set_state(rng[1])
            torch.set_rng_state(rng[2]); torch.cuda.set_rng_state(rng[3], device)
            report['adapter_and_rng_restored'] = True
        except BaseException as exc:
            report.update(status='failed', passed=False, adapter_and_rng_restored=False,
                          restoration_error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            atomic_json(report_path, report)
            del saved
            gc.collect(); torch.cuda.empty_cache()
    return report


def train_qwen25(dataset_path: Path, output_dir: Path, gguf_path: Path, smoke_test=False,
                 smoke_only=False, base_model='Qwen/Qwen2.5-7B-Instruct', base_revision=None,
                 attention_backend='sdpa', local_files_only=False) -> Dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Qwen2.5 QLoRA requires a visible CUDA GPU; CPU execution is not target verification')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    row_count = validate_chatml(dataset_path)
    provenance = corpus_provenance(dataset_path, digest, row_count)
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(base_model, revision=base_revision, local_files_only=local_files_only,
                                       trust_remote_code=False)
    revision = getattr(config, '_commit_hash', None)
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
        raise ValueError('Base model must resolve to an immutable Hugging Face commit before training')
    if config.model_type != 'qwen2':
        raise ValueError('Use a Qwen2/Qwen2.5 base for the bounded-head implementation')
    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision, local_files_only=local_files_only,
                                             trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    dataset = tokenize_trajectories(dataset_path, tokenizer)
    if len(dataset) != row_count or hashlib.sha256(dataset_path.read_bytes()).hexdigest() != digest:
        raise ValueError('Instruction corpus changed during tokenizer preflight')
    from scripts.hardware_preflight import cuda_preflight
    hardware = cuda_preflight(require_bf16=True, check_bitsandbytes=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    attention = _verify_attention(torch, attention_backend)
    metadata = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'preflight',
        'dataset': dataset_path.name, 'dataset_rows': row_count, 'dataset_sha256': digest,
        'base_model': base_model, 'base_model_revision': revision, 'hardware': hardware,
        'attention': attention, 'max_seq_length': MAX_SEQUENCE_LENGTH,
        'maximum_example_tokens': max(dataset.lengths), 'padding': 'dynamic longest in each batch; no static padding',
        'loss': 'shifted token-mean CE; frozen BF16 head; forward/backward chunks recomputed without full logits',
        'loss_chunk_tokens': LOSS_CHUNK_TOKENS, 'vocabulary_size': config.vocab_size,
        'maximum_fp32_logits_chunk_bytes': LOSS_CHUNK_TOKENS * config.vocab_size * 4,
        'training_policy': training_policy(), 'python': platform.python_version(), 'platform': platform.system(),
        'packages': {}, 'scope': 'Target execution evidence only, not independent assistant quality or warning validation'}
    for package in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'safetensors'):
        try: metadata['packages'][package] = version(package)
        except PackageNotFoundError: metadata['packages'][package] = None
    metadata.update(provenance)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = output_dir / ('smoke_run.json' if smoke_test or smoke_only else 'training_run.json')
    if record.exists():
        preserve_previous(record, output_dir)
    atomic_json(record, metadata)
    model = trainer = None
    previous_sdpa = enable_fused_sdpa(torch)
    try:
        from transformers import TrainingArguments, set_seed
        set_seed(3407)
        model = _load_model(base_model, revision, attention_backend, local_files_only)
        collator = DynamicCausalCollator(tokenizer.pad_token_id)
        metadata['smoke'] = longest_trajectory_smoke(model, dataset, collator, output_dir / 'vram_smoke_test.json')
        if smoke_test or smoke_only:
            metadata.update(status='smoke_passed', finished_at_utc=datetime.now(timezone.utc).isoformat())
            atomic_json(record, metadata)
            return metadata
        # Previous checkpoints are local history, not rotation candidates for a new fit.
        for previous in [*output_dir.glob('checkpoint-*'), output_dir / 'trainer_state.json']:
            if previous.exists():
                preserve_previous(previous, output_dir)
        args = TrainingArguments(output_dir=str(output_dir), **training_policy(), warmup_steps=10,
            num_train_epochs=2, learning_rate=2e-4, logging_steps=10, weight_decay=.01,
            lr_scheduler_type='linear', seed=3407, report_to='none', save_strategy='epoch',
            save_total_limit=2, label_names=['labels'])
        trainer = _trainer_class()(model=model, processing_class=tokenizer, train_dataset=dataset,
                                   data_collator=collator, args=args)
        metadata['status'] = 'training'
        atomic_json(record, metadata)
        result = trainer.train()
        if any(not torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
            raise FloatingPointError('Non-finite trained adapter parameters; export refused')
        trainer.save_state()
        metadata.update(status='trained', global_steps=trainer.state.global_step,
                        training_metrics=result.metrics,
                        training_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        training_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        finished_at_utc=datetime.now(timezone.utc).isoformat())
        metadata['exports'] = export_trained_model(model, tokenizer, output_dir, gguf_path,
            metadata={k: v for k, v in metadata.items() if k != 'exports'})
        atomic_json(record, metadata)
        return metadata['exports']
    except BaseException as exc:
        metadata.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        atomic_json(record, metadata)
        raise
    finally:
        restore_sdpa(torch, previous_sdpa)
        del trainer, model
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=str(ROOT / 'data/sft_context_trajectories.jsonl'))
    parser.add_argument('--output-dir', default=str(ROOT / 'artifacts/qwen2_5_7b_lora'))
    parser.add_argument('--gguf', default=str(ROOT / 'artifacts/terra-shield-qwen-q4_k_m.gguf'))
    parser.add_argument('--base-model', default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--base-revision', help='Optional requested revision; the resolved immutable commit is recorded')
    parser.add_argument('--attention-backend', choices=('sdpa', 'flash_attention_2'), default='sdpa')
    parser.add_argument('--smoke-test', '--smoke-only', dest='smoke_test', action='store_true',
                        help='Run the longest-trajectory two-step VRAM test, restore state, then exit')
    parser.add_argument('--local-files-only', action='store_true', help='Require predownloaded base/tokenizer files')
    args = parser.parse_args()
    train_qwen25(Path(args.dataset), Path(args.output_dir), Path(args.gguf), smoke_test=args.smoke_test,
        base_model=args.base_model, base_revision=args.base_revision,
        attention_backend=args.attention_backend, local_files_only=args.local_files_only)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
