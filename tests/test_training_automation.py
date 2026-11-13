"""Turnkey pipeline: real failure categories, complete fallback and exact snapshots."""
import json
import os
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from scripts.train_gpu import train_copilot
from scripts.train_qwen25_agent import download_base_snapshot, download_tokenizer_snapshot


class Automation(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Executes native Windows batch parsing in Windows CI')
    def test_conflicting_launcher_flags_fail_before_environment_setup_in_either_order(self):
        script=Path(__file__).resolve().parents[1]/'scripts/run_training.bat'
        for first,second in (('--llm-only','--xgboost-only'),('--smoke-test','--xgboost-only'),
                             ('--llm-only','--smoke-only'),('--smoke-test','--smoke-only'),('--llm-only','--smoke-test')):
            for flags in ((first,second),(second,first)):
                with self.subTest(flags=flags):
                    result=subprocess.run(['cmd.exe','/d','/c',str(script),*flags],capture_output=True,text=True,timeout=10)
                    self.assertEqual(result.returncode,2,result.stdout+result.stderr)
                    self.assertIn('Choose one execution mode',result.stdout)
    def options(self, **values):
        return SimpleNamespace(base_model='Qwen/Qwen2.5-7B-Instruct', fallback_base_model='Qwen/Qwen2.5-3B-Instruct',
            no_model_fallback=False, smoke_test=True, serve_copilot=False, copilot_port=11434, **values)

    def test_memory_failure_retests_the_same_corpus_and_records_explicit_model_change(self):
        with tempfile.TemporaryDirectory() as directory, patch('scripts.train_gpu.ROOT', Path(directory)):
            trainer = Mock(side_effect=[RuntimeError('VRAM preflight exceeded budget'), {'status': 'smoke_passed'}])
            train_copilot(self.options(), trainer)
            self.assertEqual(trainer.call_count, 2)
            first, second = trainer.call_args_list
            self.assertEqual(first.args, second.args)
            self.assertEqual(first.kwargs['base_model'], 'Qwen/Qwen2.5-7B-Instruct')
            self.assertEqual(second.kwargs['base_model'], 'Qwen/Qwen2.5-3B-Instruct')
            self.assertTrue(second.kwargs['smoke_test'])
            self.assertEqual(second.kwargs['run_context']['fallback_from'], first.kwargs['base_model'])
            history = json.loads((Path(directory)/'artifacts/qwen25_terra_lora/training_attempts.json').read_text())
            self.assertEqual([r['status'] for r in history], ['failed', 'smoke_passed'])

    def test_provider_or_data_error_never_silently_changes_the_model(self):
        with tempfile.TemporaryDirectory() as directory, patch('scripts.train_gpu.ROOT', Path(directory)):
            trainer = Mock(side_effect=ValueError('SFT provenance mismatch'))
            with self.assertRaisesRegex(ValueError, 'provenance'):train_copilot(self.options(), trainer)
            self.assertEqual(trainer.call_count, 1)

    def test_snapshot_resumes_only_immutable_safetensors_and_rejects_missing_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'model-1.safetensors').write_bytes(b'fixture')
            (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': 'model-1.safetensors'}}))
            hub = SimpleNamespace(snapshot_download=Mock(return_value=str(root)))
            with patch.dict('sys.modules', {'huggingface_hub': hub}):
                self.assertEqual(download_tokenizer_snapshot('Qwen/Qwen2.5-7B-Instruct', 'a'*40, True), str(root))
                self.assertTrue(hub.snapshot_download.call_args.kwargs['local_files_only'])
                self.assertNotIn('*.safetensors', hub.snapshot_download.call_args.kwargs['allow_patterns'])
                self.assertEqual(download_base_snapshot('Qwen/Qwen2.5-7B-Instruct', 'a'*40), str(root))
                self.assertEqual(hub.snapshot_download.call_args.kwargs['revision'], 'a'*40)
                self.assertTrue(hub.snapshot_download.call_args.kwargs['resume_download'])
                (root/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x': '../missing.safetensors'}}))
                with self.assertRaisesRegex(ValueError, 'weight shard'):
                    download_base_snapshot('Qwen/Qwen2.5-7B-Instruct', 'a'*40)


if __name__ == '__main__':unittest.main()
