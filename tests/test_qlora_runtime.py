"""Offline policy checks and real CPU loss/gradient equivalence; no model downloads."""
import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scripts.qlora_runtime import (training_policy, TokenizedTraces, tokenize_trajectories,
    DynamicCausalCollator, chunked_causal_loss, backbone_loss, VRAM_LIMIT_BYTES)
from scripts.train_qwen25_agent import _memory_result, _trainer_class, longest_trajectory_smoke

TORCH = importlib.util.find_spec('torch') is not None
HF = TORCH and importlib.util.find_spec('transformers') is not None and importlib.util.find_spec('peft') is not None


class Policies(unittest.TestCase):
    def test_windows_memory_policy(self):
        p = training_policy()
        self.assertEqual(p['per_device_train_batch_size'], 1)
        self.assertEqual(p['gradient_accumulation_steps'], 8)
        self.assertTrue(p['bf16']); self.assertFalse(p['fp16']); self.assertTrue(p['tf32'])
        self.assertTrue(p['group_by_length']); self.assertTrue(p['gradient_checkpointing'])
        self.assertEqual(p['gradient_checkpointing_kwargs'], {'use_reentrant': False})
        self.assertEqual(p['optim'], 'paged_adamw_8bit')
        self.assertEqual(p['dataloader_num_workers'], 0); self.assertTrue(p['dataloader_pin_memory'])
        self.assertEqual(VRAM_LIMIT_BYTES, 14_500_000_000)

    def test_tokenization_is_complete_and_rejects_oversize(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                assert kwargs == {'tokenize': True, 'add_generation_prompt': False}
                return list(range(int(messages[0]['content'])))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rows.jsonl'
            path.write_text('\n'.join(json.dumps({'source_event_id': str(n), 'messages': [{'content': str(n)}]}) for n in [11, 8192]))
            dataset = tokenize_trajectories(path, Tokenizer())
            self.assertEqual(dataset.lengths, [11, 8192]); self.assertEqual(dataset.longest_index, 1)
            path.write_text(json.dumps({'source_event_id': 'long', 'messages': [{'content': '8193'}]}))
            with self.assertRaisesRegex(ValueError, 'refusing truncation'):
                tokenize_trajectories(path, Tokenizer())

    def test_reserved_memory_and_decimal_budget_are_enforced(self):
        cuda = SimpleNamespace(mem_get_info=lambda _: (2_000_000_000, 16_000_000_000),
            max_memory_allocated=lambda _: 14_400_000_000, max_memory_reserved=lambda _: 14_490_000_000)
        self.assertTrue(_memory_result(SimpleNamespace(cuda=cuda), 0, 2_000_000_000)['passed'])
        cuda.max_memory_reserved = lambda _: 14_500_000_000
        self.assertFalse(_memory_result(SimpleNamespace(cuda=cuda), 0, 2_000_000_000)['passed'])
        cuda.max_memory_reserved = lambda _: 14_400_000_000
        self.assertFalse(_memory_result(SimpleNamespace(cuda=cuda), 0, 999_999_999)['passed'])

    def test_backend_endpoint_override_is_explicit_and_keeps_default(self):
        from backend.llm.qwen_agent import TerraShieldQwenAgent
        with patch.dict('os.environ', {'TERRA_QWEN_URL': 'http://127.0.0.1:11435/'}):
            self.assertEqual(TerraShieldQwenAgent(scan_service=object()).ollama_url, 'http://127.0.0.1:11435')
            self.assertEqual(TerraShieldQwenAgent(scan_service=object(), ollama_url='http://localhost:11434').ollama_url, 'http://localhost:11434')


@unittest.skipUnless(TORCH, 'Requires optional CPU PyTorch test environment, not CUDA')
class ChunkedLoss(unittest.TestCase):
    def test_dense_value_and_gradient_with_shift_padding_and_chunk_boundaries(self):
        import torch
        torch.manual_seed(19)
        hidden = torch.randn(2, 11, 7, requires_grad=True)
        head = torch.nn.Linear(7, 23, bias=True).requires_grad_(False)
        labels = torch.randint(0, 23, (2, 11)); labels[0, 5] = -100; labels[1, -4:] = -100
        actual = chunked_causal_loss(hidden, head, labels, 3); actual.backward()
        gradient = hidden.grad.clone(); hidden.grad = None
        expected = torch.nn.functional.cross_entropy(head(hidden[:, :-1]).float().reshape(-1, 23), labels[:, 1:].reshape(-1), ignore_index=-100)
        expected.backward()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(gradient, hidden.grad, rtol=2e-5, atol=1e-7)
        self.assertTrue(torch.equal(gradient[:, -1], torch.zeros_like(gradient[:, -1])))

    def test_no_linear_projection_exceeds_chunk_size_and_no_retained_logits(self):
        import torch
        import torch.nn.functional as functional
        hidden = torch.randn(1, 19, 4, requires_grad=True)
        head = torch.nn.Linear(4, 29, bias=False).requires_grad_(False)
        labels = torch.randint(0, 29, (1, 19))
        original = functional.linear
        sizes, saved_shapes = [], []
        def linear(value, *args, **kwargs):
            sizes.append(value.shape[0]); return original(value, *args, **kwargs)
        def pack(tensor):
            saved_shapes.append(tuple(tensor.shape)); return tensor
        with patch('torch.nn.functional.linear', side_effect=linear), torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            loss = chunked_causal_loss(hidden, head, labels, 4)
        # Only hidden states, frozen head, labels and optional bias survive forward.
        self.assertNotIn((18, 29), saved_shapes)
        with patch('torch.nn.functional.linear', side_effect=linear):
            loss.backward()
        self.assertEqual(max(sizes), 4)
        self.assertGreater(len(sizes), 5)  # Recomputed backward chunks, not retained CE graphs.

    def test_dynamic_padding_masks_only_padding_and_preserves_real_eos(self):
        import torch
        batch = DynamicCausalCollator(9)([{'input_ids': [1, 9, 2]}, {'input_ids': [3, 9]}])
        self.assertEqual(tuple(batch['input_ids'].shape), (2, 3))
        self.assertEqual(batch['labels'].tolist(), [[1, 9, 2], [3, 9, -100]])
        self.assertEqual(batch['attention_mask'].tolist(), [[1, 1, 1], [1, 1, 0]])
        with self.assertRaisesRegex(ValueError, 'never truncates'):
            DynamicCausalCollator(9, max_length=2)([{'input_ids': [1, 2, 3]}])

    def test_trainable_head_and_all_ignored_targets_are_rejected(self):
        import torch
        hidden = torch.randn(1, 3, 4, requires_grad=True); head = torch.nn.Linear(4, 7)
        with self.assertRaisesRegex(ValueError, 'frozen lm_head'):
            chunked_causal_loss(hidden, head, torch.ones((1, 3), dtype=torch.long))
        head.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'No unmasked'):
            chunked_causal_loss(hidden, head, torch.full((1, 3), -100, dtype=torch.long))

    def test_smoke_restores_updates_rng_and_training_mode(self):
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.lora_weight = torch.nn.Parameter(torch.ones(2, 2))
        model = Model().eval(); before = model.lora_weight.detach().clone(); torch_rng = torch.get_rng_state()
        calls = []
        def loss(m, _):
            calls.append(1); random.random(); np.random.rand(); torch.rand(1)
            return m.lora_weight.square().mean()
        dataset = TokenizedTraces([{'input_ids': [1, 2], 'labels': [1, 2]}], ['fixture'])
        functions = {'current_device': lambda: 'cpu', 'empty_cache': lambda: None,
            'reset_peak_memory_stats': lambda _: None, 'mem_get_info': lambda _: (10_000_000_000, 16_000_000_000),
            'max_memory_allocated': lambda _: 1_000_000_000, 'max_memory_reserved': lambda _: 2_000_000_000,
            'get_rng_state': lambda _: torch.zeros(1, dtype=torch.uint8), 'set_rng_state': lambda *args: None,
            'synchronize': lambda _: None}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name, value in functions.items():stack.enter_context(patch.object(torch.cuda, name, value))
            stack.enter_context(patch('scripts.train_qwen25_agent.backbone_loss', side_effect=loss))
            report = longest_trajectory_smoke(model, dataset, DynamicCausalCollator(0), Path(directory)/'smoke.json', optimizer_factory=torch.optim.SGD)
            self.assertEqual(len(calls), 16)
            class BrokenFinalStep(torch.optim.SGD):
                steps = 0
                def step(self, *args, **kwargs):
                    result = super().step(*args, **kwargs)
                    self.steps += 1
                    if self.steps == 2:
                        with torch.no_grad():model.lora_weight.fill_(float('nan'))
                    return result
            with self.assertRaisesRegex(FloatingPointError, 'after smoke optimizer step'):
                longest_trajectory_smoke(model, dataset, DynamicCausalCollator(0), Path(directory)/'smoke.json', optimizer_factory=BrokenFinalStep)
            failure = json.loads((Path(directory)/'smoke.json').read_text())
            self.assertEqual(failure['status'], 'failed')
            self.assertFalse(failure['passed']); self.assertTrue(failure['adapter_and_rng_restored'])
            self.assertEqual(len(list((Path(directory)/'previous_exports').glob('smoke.json*'))), 1)
        self.assertEqual(len(calls), 32)
        self.assertTrue(report['adapter_update_observed']); self.assertTrue(report['adapter_and_rng_restored'])
        torch.testing.assert_close(model.lora_weight, before, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state())); self.assertFalse(model.training)


@unittest.skipUnless(HF, 'Requires optional pinned Transformers/PEFT CPU test environment')
class RealAdapter(unittest.TestCase):
    @staticmethod
    def model():
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from peft import LoraConfig, get_peft_model
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=31, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, use_cache=False, attention_dropout=0.))
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=2, lora_dropout=0., task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj']))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        return model.train()

    def test_backbone_preserves_peft_gradients_without_causal_lm_forward(self):
        import torch
        torch.manual_seed(7); model = self.model()
        batch = {'input_ids': torch.randint(0, 31, (1, 7)), 'attention_mask': torch.ones((1, 7), dtype=torch.long)}
        batch['labels'] = batch['input_ids'].clone()
        dense = model(**batch).loss; dense.backward()
        expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        causal = model.get_base_model()
        with patch.object(causal, 'forward', side_effect=AssertionError('Full CausalLM logits are forbidden')):
            bounded = backbone_loss(model, batch); bounded.backward()
        torch.testing.assert_close(bounded, dense, rtol=1e-5, atol=1e-6)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:torch.testing.assert_close(parameter.grad, expected[name], rtol=2e-4, atol=1e-6)
        self.assertTrue(any(v.abs().sum().item() for v in expected.values()))

    def test_real_trainer_accumulation_matches_dense_reference(self):
        import torch
        from transformers import TrainingArguments
        torch.manual_seed(9); actual = self.model(); expected = copy.deepcopy(actual)
        ids = [1, 2, 3, 4, 5]
        row = {'input_ids': ids, 'attention_mask': [1]*5, 'labels': ids.copy()}
        dataset = TokenizedTraces([copy.deepcopy(row) for _ in range(8)], [str(i) for i in range(8)])
        collator = DynamicCausalCollator(0); batch = collator([row])
        reference_optimizer = torch.optim.SGD([p for p in expected.parameters() if p.requires_grad], lr=.1)
        for _ in range(8):(expected(**batch).loss / 8).backward()
        reference_optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, max_steps=1,
                per_device_train_batch_size=1, gradient_accumulation_steps=8, group_by_length=True,
                max_grad_norm=0., learning_rate=.1, report_to='none', save_strategy='no',
                disable_tqdm=True, dataloader_num_workers=0, remove_unused_columns=False)
            trainer = _trainer_class()(model=actual, args=args, train_dataset=dataset,
                data_collator=collator, optimizers=(torch.optim.SGD([p for p in actual.parameters() if p.requires_grad], lr=.1), None))
            self.assertFalse(trainer.model_accepts_loss_kwargs)
            trainer.train()
        for (name, got), (_, want) in zip(actual.named_parameters(), expected.named_parameters()):
            if got.requires_grad:torch.testing.assert_close(got, want, rtol=3e-4, atol=1e-6, msg=name)


if __name__ == '__main__':unittest.main()
