"""Current-bundle selection, actual-byte verification and export preservation."""
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
from scripts.sync_training_bundle import ALIASES,copy_file,inventory,main
from scripts.train_qwen25_agent import export_trained_model,train_qwen25


class BundleIntegrity(unittest.TestCase):
    def test_current_sources_and_observations_survive_legacy_cache_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for source,_ in ALIASES:
                path=root/source;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
            data={
                'data/real_global_landslide_observations.parquet':b'observation fixture',
                'data/benchmark_history/v12/event_holdout_manifest.json':b'{"selected_ids":["original"]}',
                'data/source_catalogs/terrain_metric_v1/corrected.json':b'corrected terrain fixture',
                'data/source_catalogs/seismic_archive/event.json':b'catalog fixture',
                'data/source_catalogs/terrain_v13_old.json':b'previous derived terrain fixture',
            }
            for relative,content in data.items():
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
            for name in ('training','windows_5070ti_trainer'):
                old=root/name/'data/source_catalogs/terrain_v13_old.json';old.parent.mkdir(parents=True,exist_ok=True);old.write_bytes(b'older bundle terrain')
                (root/name/'bundle_manifest.json').write_text('{"old":"manifest"}')
            files,excluded=inventory(root)
            self.assertEqual(excluded,['data/source_catalogs/terrain_v13_old.json'])
            self.assertIn('data/source_catalogs/seismic_archive/event.json',files)
            with redirect_stdout(io.StringIO()):main(root)
            manifests=[]
            for name in ('training','windows_5070ti_trainer'):
                target=root/name;manifest=json.loads((target/'bundle_manifest.json').read_text());manifests.append(manifest)
                self.assertNotIn('data/source_catalogs/terrain_v13_old.json',manifest)
                self.assertEqual((target/'archive_snapshots/legacy_terrain_v13/terrain_v13_old.json').read_bytes(),b'older bundle terrain')
                self.assertTrue((target/'archive_snapshots/bundle_manifest.previous.json').exists())
                for relative,content in data.items():
                    if 'terrain_v13_' in relative:continue
                    self.assertEqual((target/relative).read_bytes(),content)
                    self.assertEqual(manifest[relative],hashlib.sha256(content).hexdigest())
            self.assertEqual(manifests[0],manifests[1])
            self.assertEqual((root/'data/source_catalogs/terrain_v13_old.json').read_bytes(),data['data/source_catalogs/terrain_v13_old.json'])

    def test_unreadable_source_cannot_be_replaced_by_stale_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';destination=Path(directory)/'destination'
            destination.write_bytes(b'previous data')
            with patch('scripts.sync_training_bundle.subprocess.run',side_effect=subprocess.TimeoutExpired('read',30)):
                with self.assertRaisesRegex(RuntimeError,'Timed out reading canonical source'):copy_file(source,destination)
            self.assertEqual(destination.read_bytes(),b'previous data')

    def test_export_preserves_previous_adapter_before_optional_conversion_failure(self):
        class Model:
            def save_pretrained(self,path):
                target=Path(path);target.mkdir(parents=True,exist_ok=True);(target/'adapter.safetensors').write_bytes(b'new adapter')
            def save_pretrained_merged(self,*args,**kwargs):raise RuntimeError('conversion fixture')
            def save_pretrained_gguf(self,*args,**kwargs):raise RuntimeError('conversion fixture')
        class Tokenizer:
            def save_pretrained(self,path):(Path(path)/'tokenizer.json').write_text('{}')
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'model';old=output/'adapter';old.mkdir(parents=True);(old/'adapter.safetensors').write_bytes(b'previous adapter')
            with redirect_stdout(io.StringIO()):
                report=export_trained_model(Model(),Tokenizer(),output,Path(directory)/'model.gguf')
            self.assertEqual((output/'adapter/adapter.safetensors').read_bytes(),b'new adapter')
            self.assertEqual((output/report['previous_exports_preserved']['adapter']/'adapter.safetensors').read_bytes(),b'previous adapter')
            self.assertEqual(report['adapter']['status'],'saved')
            self.assertEqual(report['gguf']['status'],'failed')

    def test_8192_context_preflight_rejects_oversized_trace_before_training(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return 'rendered fixture'
            def __call__(self,*args,**kwargs):return {'input_ids':list(range(8193))}
        class Dataset:
            def map(self,render,**kwargs):return {'text':[render({'messages':[]})['text']]}
        class FastLanguageModel:
            @staticmethod
            def from_pretrained(**kwargs):
                if kwargs['max_seq_length']!=8192:raise AssertionError('Context limit changed')
                return object(),Tokenizer()
            @staticmethod
            def get_peft_model(model,**kwargs):return model
        def premature_training(**kwargs):raise AssertionError('Oversized input reached the trainer')
        modules={'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True)),
            'unsloth':SimpleNamespace(FastLanguageModel=FastLanguageModel),
            'datasets':SimpleNamespace(load_dataset=lambda *args,**kwargs:Dataset()),
            'trl':SimpleNamespace(SFTConfig=premature_training,SFTTrainer=premature_training)}
        with tempfile.TemporaryDirectory() as directory:
            dataset=Path(directory)/'corpus.jsonl'
            dataset.write_text(json.dumps({'source_event_id':'fixture','messages':[{'role':'tool','content':'evidence'}]})+'\n')
            output=Path(directory)/'output'
            with patch.dict('sys.modules',modules):
                with self.assertRaisesRegex(ValueError,'exceeds the declared 8192-token context'):
                    train_qwen25(dataset,output,Path(directory)/'model.gguf')
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
