"""Portable diagnostic serialization shared by reports and release preparation."""
import re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
HOME_PREFIX = chr(47) + 'Users' + chr(47)

def portable_text(text):
    text = text.replace(str(ROOT) + '/', '').replace(str(ROOT), '.')
    text = re.sub(re.escape(HOME_PREFIX) + r'[^\s\"\'<>]+', 'USER_HOME_PATH', text)
    text = re.sub(r'(?i)[A-Z]:\\Users\\[^\s\"<>]+', 'USER_HOME_PATH', text)
    return text.replace(HOME_PREFIX, 'USER_HOME_PREFIX')

def portable_payload(value):
    if isinstance(value, str):
        return portable_text(value)
    if isinstance(value, dict):
        return {k: portable_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable_payload(v) for v in value]
    return value
def invalidate_readiness(artifacts, reason):
    """Clear copied audit approval before any model rebuild or artifact replacement."""
    import json
    from pathlib import Path
    from datetime import datetime, timezone
    directory = Path(artifacts)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('readiness.json', 'scientific_validation_report.json'):
        path = directory / name
        try:
            payload = json.loads(path.read_text()) if path.exists() else {}
        except (ValueError, OSError):
            payload = {}
        payload.update(engineering_ready=False, pretraining_ready=False,
                       invalidated_at_utc=datetime.now(timezone.utc).isoformat(),
                       invalidation_reason=reason)
        payload.pop('combined_audit', None)
        path.write_text(json.dumps(payload, indent=2)+'\n')

