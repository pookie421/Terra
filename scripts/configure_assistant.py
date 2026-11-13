#!/usr/bin/env python3
"""Configure one deployment's assistant without putting a key in shell history."""
import getpass
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.llm.model_client import assistant_config, AssistantConfigurationError


def save_settings(path, settings):
    """Preserve other providers' settings; validate before any environment write."""
    config=assistant_config(settings)
    if not config.configured:raise AssistantConfigurationError('Hosted setup requires an endpoint, model and API key')
    path=Path(path)
    previous=path.read_text(encoding='utf-8') if path.exists() else ''
    # These values are read by python-dotenv, never executed as shell commands.
    lines=previous.splitlines()
    for name,value in settings.items():
        if not name.startswith('TERRA_LLM_'):raise ValueError('Only assistant settings can be changed')
        if any(c in value for c in '\r\n\x00'):raise ValueError('Settings must be single-line values')
        rendered=name+"='"+value.replace('\\','\\\\').replace("'","\\'")+"'"
        pattern=re.compile(r'^\s*(?:export\s+)?'+re.escape(name)+r'\s*=')
        lines=[line for line in lines if not pattern.match(line)]
        lines.append(rendered)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.assistant-tmp')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as handle:
            handle.write('\n'.join(lines)+'\n')
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        if temporary.exists():temporary.unlink()


def main():
    print('Assistant connection for this deployment. Shared users do not need personal keys.')
    selection=input('Provider: [1] Local Ollama  [2] Hosted compatible API (default 2): ').strip() or '2'
    if selection=='1':
        settings={'TERRA_LLM_PROVIDER':'ollama','TERRA_LLM_MODEL':input('Local model name (blank selects available Qwen): ').strip(),
                  'TERRA_LLM_BASE_URL':'','TERRA_LLM_API_KEY':''}
    elif selection=='2':
        settings={'TERRA_LLM_PROVIDER':'openai_compatible',
                  'TERRA_LLM_BASE_URL':input('Base URL from your provider console (ending in /v1): ').strip(),
                  'TERRA_LLM_MODEL':input('JSON-capable, non-thinking model ID: ').strip(),
                  'TERRA_LLM_API_KEY':getpass.getpass('API key (hidden): ').strip()}
    else:raise ValueError('Choose provider 1 or 2')
    save_settings(ROOT/'.env',settings)
    print('Saved local .env. No provider request was sent. Restart the backend with --env-file .env.')


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyboardInterrupt,EOFError) as exc:
        print('Setup did not complete: '+(str(exc) if isinstance(exc,ValueError) else type(exc).__name__),file=sys.stderr)
        raise SystemExit(1)
