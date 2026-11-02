#!/usr/bin/env python3
"""Export exact numbered source, with hashes, for external mathematical review."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=['backend/realtime/dem_processor.py','backend/realtime/global_ingestor.py','backend/realtime/ground_context.py','backend/api/scan_service.py','training/train.py','training/train_qwen25_agent.py','backend/physics/multi_causal_engine.py','backend/physics/mechanisms.py',
       'backend/physics/alpine_weather.py','backend/physics/kinematics.py',
       'backend/physics/causal_attribution.py','backend/ml/global_xgb.py',
       'backend/ml/conformal_calibration.py','backend/physics/cascading.py',
       'backend/llm/qwen_agent.py','backend/physics/temporal_forecaster.py',
       'backend/realtime/forecast_weather.py','backend/ml/build_real_labeled_dataset.py',
       'scripts/terra_cli.py','backend/api/live_input_builder.py','backend/realtime/cache.py']
def main():
    parts=['# Exact core source for external audit','',
           'Line-number prefixes are added; text after each prefix is verbatim source. '
           'Inclusion is not a claim that every equation is literature-verified. '
           'See docs/physics_sources.md for proxy and domain disclosures.','']
    for name in FILES:
        path=ROOT/name;raw=path.read_bytes()
        parts += ['## '+name,'','SHA-256: `'+hashlib.sha256(raw).hexdigest()+'`','',
                  '```python',*['{:04d}{}'.format(i,' '+line if line else '') for i,line in enumerate(raw.decode().splitlines(),1)],'```','']
    (ROOT/'artifacts/core_code_audit.md').write_text('\n'.join(parts))
    print('Exported exact numbered source for {} files.'.format(len(FILES)))
if __name__=='__main__':main()
