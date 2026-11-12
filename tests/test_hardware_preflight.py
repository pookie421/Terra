"""Offline target-policy, exact XGBoost serialization and manifest checks."""
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scripts.hardware_preflight import cuda_preflight
from scripts import hardware_preflight, train_gpu
from scripts.train_gpu import smoke_test, verify_training_artifacts
from scripts.train_physics_informed import verify_reload_parity
from backend.ml.physics_informed import KINDS, NAMES, VERSION
from backend.ml.validation_suite import ranking_metrics


def fake_torch(capability=(12, 0), runtime='12.8', bf16=True, available=True, architectures=None):
    cuda = SimpleNamespace(is_available=lambda: available, current_device=lambda: 0,
        get_device_capability=lambda index: capability,
        get_arch_list=lambda: architectures if architectures is not None else ['sm_80', 'sm_120'],
        is_bf16_supported=lambda **kwargs: bf16,
        get_device_properties=lambda index: SimpleNamespace(total_memory=16*1024**3),
        mem_get_info=lambda index: (15*1024**3, 16*1024**3),
        get_device_name=lambda index: 'NVIDIA GeForce RTX 5070 Ti')
    return SimpleNamespace(cuda=cuda, version=SimpleNamespace(cuda=runtime), __version__='2.7.1+cu128')


class HardwarePolicy(unittest.TestCase):
    def test_blackwell_reports_actual_fields_and_requires_kernel_execution(self):
        with patch('scripts.hardware_preflight._probe_kernels', return_value={'float32_matmul_backward': 'passed', 'fused_sdpa_backends':['FLASH_ATTENTION']}) as probe:
            result = cuda_preflight(fake_torch(), require_bf16=True)
        probe.assert_called_once()
        self.assertEqual(result['compute_capability'], [12, 0])
        self.assertEqual(result['total_vram_bytes'], 16*1024**3)
        self.assertTrue(result['bf16_supported'])
        self.assertEqual(result['fused_sdpa_backends'], ['FLASH_ATTENTION'])

    def test_old_cuda_and_missing_sm120_fail_before_kernels(self):
        for torch, message in ((fake_torch(runtime='12.4'), 'CUDA 12.8'),
                               (fake_torch(architectures=['sm_80', 'sm_90']), 'compatible sm_120')):
            with self.subTest(message=message), patch('scripts.hardware_preflight._probe_kernels') as probe:
                with self.assertRaisesRegex(RuntimeError, message):cuda_preflight(torch)
                probe.assert_not_called()

    def test_cpu_torch_is_not_silently_accepted(self):
        with self.assertRaisesRegex(RuntimeError, 'cannot execute CUDA'):
            cuda_preflight(fake_torch(available=False))

    def test_bf16_requirement_has_no_fp16_fallback(self):
        with self.assertRaisesRegex(RuntimeError, 'FP16 fallback is disabled'):
            cuda_preflight(fake_torch(bf16=False), require_bf16=True)

    def test_other_working_cuda_gpu_is_allowed_for_xgboost(self):
        with patch('scripts.hardware_preflight._probe_kernels', return_value={}):
            report = cuda_preflight(fake_torch(capability=(8, 6), runtime='12.6', architectures=['sm_86']))
            self.assertEqual(report['compute_capability'], [8, 6])
            with self.assertRaisesRegex(RuntimeError, 'requires compute capability'):
                cuda_preflight(fake_torch(capability=(8, 6)), require_blackwell=True)

    def test_old_bf16_api_cannot_certify_emulation_on_turing(self):
        torch = fake_torch(capability=(7, 5), runtime='12.6', architectures=['sm_75'])
        torch.cuda.is_bf16_supported = lambda: True
        with self.assertRaisesRegex(RuntimeError, 'BF16 CUDA support'):
            cuda_preflight(torch, require_bf16=True)

    def test_nf4_probe_is_explicit_and_errors_propagate(self):
        with patch('scripts.hardware_preflight._probe_kernels', return_value={}), patch('scripts.hardware_preflight._probe_bitsandbytes', side_effect=RuntimeError('native wheel kernel failure')) as probe:
            cuda_preflight(fake_torch())
            probe.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'native wheel kernel failure'):
                cuda_preflight(fake_torch(), check_bitsandbytes=True)


class ExactSerialization(unittest.TestCase):
    def test_cpu_smoke_uses_exact_save_reload_predictions(self):
        result = smoke_test('cpu')
        self.assertEqual(result['same_device_prediction_parity'], 'exact')
        self.assertEqual(result['tree_method'], 'hist')
        self.assertEqual(result['xgboost_device'], 'cpu')
        self.assertEqual(len(result['prediction_sha256']), 64)
        self.assertEqual(result['cross_device_inference']['status'], 'not_applicable_cpu_run')

    def test_ensemble_parity_rejects_even_tiny_prediction_change(self):
        def model(value):return SimpleNamespace(folds=[], predict_proba=lambda x: np.full(len(x),value))
        before = {kind:model(.25) for kind in KINDS}
        after = {kind:model(.25) for kind in KINDS}
        x = np.zeros((2,len(NAMES)))
        proof = verify_reload_parity(before,after,x,'cpu')
        self.assertEqual(proof['models']['Physics-Informed ML']['maximum_absolute_difference'],0.)
        after['Physics-Informed ML']=model(.25+1e-12)
        with self.assertRaises(AssertionError):verify_reload_parity(before,after,x,'cpu')


class TrainedArtifactManifest(unittest.TestCase):
    def create_fixture(self,root):
        art=root/'artifacts';data=root/'data';art.mkdir();data.mkdir()
        model=art/'physics_informed_models.pkl';model.write_bytes(b'fixture serialization')
        rows=[{'features':[i]+[0.]*(len(NAMES)-1),'label':i} for i in (0,1)]
        frame=data/'physics_informed_training.json';frame.write_text(json.dumps({'version':VERSION,'train':[],'holdout':rows}))
        holdout=data/'event_holdout_manifest.json';holdout.write_text('{"selected_ids":["frozen"]}')
        sft=data/'sft_context_trajectories.jsonl';sft.write_text('{"fixture":true}\n')
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        predict=lambda x:.2+.6*x[:,0]
        models={kind:SimpleNamespace(version=VERSION,predict_proba=predict) for kind in KINDS}
        metrics={kind:ranking_metrics([0,1],predict(np.asarray([r['features'] for r in rows]))) for kind in KINDS}
        report={'version':VERSION,'features':list(NAMES),'model_sha256':sha(model),'dataset_sha256':sha(frame),
                'holdout_manifest_sha256':sha(holdout),'environment':{'device':'cpu'},
                'save_reload_prediction_parity':{'status':'passed'},'development_holdout':metrics}
        (art/'physics_informed_ablation_report.json').write_text(json.dumps(report))
        (art/'scientific_validation_report.json').write_text(json.dumps(report))
        (art/'global_context_benchmark.json').write_text(json.dumps({'version':VERSION,'holdout_membership_changed':False}))
        (art/'sft_trajectory_report.json').write_text(json.dumps({'fusion_model_sha256':sha(model),'sha256':sha(sft)}))
        return models

    def test_manifest_hashes_artifacts_and_frozen_data_after_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);models=self.create_fixture(root)
            selected_devices=[]
            for model in models.values():
                model.folds=[(SimpleNamespace(set_params=lambda **kwargs:selected_devices.append(kwargs['device'])),None)]
            with patch('joblib.load',return_value=models):manifest=verify_training_artifacts(root,'cpu',{'fixture_smoke':'passed'})
            self.assertEqual(selected_devices,['cpu']*len(KINDS))
            self.assertEqual(len(manifest['sha256']),8)
            self.assertEqual(manifest['saved_model_development_metrics'],'exact')
            self.assertEqual(json.loads((root/'artifacts/training_artifact_manifest.json').read_text()),manifest)
            for relative,digest in manifest['sha256'].items():
                self.assertEqual(hashlib.sha256((root/relative).read_bytes()).hexdigest(),digest)

    def test_changed_holdout_prevents_fresh_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.create_fixture(root)
            (root/'data/event_holdout_manifest.json').write_text('{"selected_ids":["changed"]}')
            with self.assertRaisesRegex(RuntimeError,'Holdout membership changed'):
                verify_training_artifacts(root,'cpu',{})
            self.assertFalse((root/'artifacts/training_artifact_manifest.json').exists())

    def test_stale_sft_cannot_inherit_new_model_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);models=self.create_fixture(root)
            (root/'data/sft_context_trajectories.jsonl').write_text('{"changed":true}\n')
            with patch('joblib.load',return_value=models), self.assertRaisesRegex(RuntimeError,'Instruction corpus'):
                verify_training_artifacts(root,'cpu',{})
            self.assertFalse((root/'artifacts/training_artifact_manifest.json').exists())


class CurrentCheckEvidence(unittest.TestCase):
    def test_failed_hardware_cli_replaces_pass_and_preserves_exact_previous_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'hardware_preflight.json'
            previous=b'{"status":"passed","gpu":"old fixture GPU","scope":"historical fixture"}\n'
            output.write_bytes(previous)
            with patch('sys.argv',['hardware_preflight.py','--output',str(output)]), \
                 patch('scripts.hardware_preflight.cuda_preflight',side_effect=RuntimeError('current driver failure')), \
                 redirect_stdout(io.StringIO()):
                code=hardware_preflight.main()
            self.assertEqual(code,1)
            current=json.loads(output.read_text())
            self.assertEqual(current['status'],'failed')
            self.assertEqual(current['error'],'current driver failure')
            self.assertNotIn('gpu',current)
            self.assertTrue(current['scope'])
            for key in ('started_at_utc','finished_at_utc'):
                self.assertIsNotNone(datetime.fromisoformat(current[key]).utcoffset())
            self.assertEqual((output.parent/current['previous_report_preserved']).read_bytes(),previous)
            self.assertFalse(output.with_suffix('.json.tmp').exists())

    def test_failed_training_cli_records_current_failure_before_propagating(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'artifacts/gpu_smoke_test.json'
            output.parent.mkdir();previous=b'{"status":"passed","xgboost_device":"cuda"}\n'
            output.write_bytes(previous)
            with patch('sys.argv',['train.py','--device','cpu','--smoke-only']), \
                 patch('scripts.train_gpu.ROOT',root), \
                 patch('scripts.train_gpu.smoke_test',side_effect=RuntimeError('current serialization failure')), \
                 redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'current serialization failure'):train_gpu.main()
            current=json.loads(output.read_text())
            self.assertEqual(current['status'],'failed')
            self.assertEqual(current['requested_device'],'cpu')
            self.assertNotIn('xgboost_device',current)
            self.assertEqual((output.parent/current['previous_report_preserved']).read_bytes(),previous)
            self.assertTrue(current['started_at_utc'])
            self.assertTrue(current['finished_at_utc'])

    def test_success_also_preserves_a_corrupt_previous_report_without_reusing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'check.json';previous=b'corrupt previous report'
            output.write_bytes(previous)
            current=hardware_preflight.write_check_report(output,{'status':'passed','scope':'new fixture only'})
            self.assertEqual(json.loads(output.read_text()),current)
            self.assertEqual((output.parent/current['previous_report_preserved']).read_bytes(),previous)
            self.assertNotIn('error',current)


if __name__=='__main__':unittest.main()
