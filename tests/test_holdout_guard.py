"""An incomplete source refresh must never train on held-out event rows."""
import argparse
import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
from scripts import run_mac_baseline_and_verify as runner

class HoldoutGuard(unittest.TestCase):
    def test_incomplete_enrichment_does_not_fit_or_replace_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'data').mkdir();(root/'artifacts').mkdir()
            pq.write_table(pa.Table.from_pylist([{'label':None}]),root/'data/observations.parquet')
            (root/'data/observations.metadata.json').write_text(json.dumps({'event_group_holdout':{'status':'pending_audit_refresh'}}))
            (root/'artifacts/readiness.json').write_text(json.dumps({'engineering_ready':True,'pretraining_ready':True}))
            model=root/'artifacts/global_multi_causal_xgb.pkl';model.write_bytes(b'previous trusted model')
            args=argparse.Namespace(dataset='data/observations.parquet',skip_dataset_build=True)
            with patch.object(runner,'ROOT',root),patch.object(runner,'_fit_real_model') as fit,contextlib.redirect_stdout(io.StringIO()):
                code=asyncio.run(runner.run(args))
            self.assertEqual(code,2);fit.assert_not_called()
            self.assertEqual(model.read_bytes(),b'previous trusted model')
            report=json.loads((root/'artifacts/scientific_validation_report.json').read_text())
            self.assertFalse(report['pretraining_ready'])
            self.assertFalse(report['training_model_fitted'])
            readiness=json.loads((root/'artifacts/readiness.json').read_text())
            self.assertFalse(readiness['pretraining_ready'])
            self.assertFalse(readiness['engineering_ready'])

if __name__=='__main__':unittest.main()
