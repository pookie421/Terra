"""Server-owned assistant configuration and a small compatible chat transport.

The language model explains computed evidence. It never replaces the physical
or landslide classifiers, and its output still passes the state co-pilot guards.
"""
from dataclasses import dataclass, field
import ipaddress
import os
from typing import Mapping, Optional
from urllib.parse import urlsplit


class AssistantConfigurationError(ValueError):
    """Safe, value-free configuration error suitable for public diagnostics."""


class AssistantServiceError(RuntimeError):
    """Provider failure without upstream response bodies or credentials."""


@dataclass(frozen=True)
class AssistantConfig:
    provider: str = 'ollama'
    base_url: str = ''
    model: str = ''
    api_key: str = field(default='',repr=False)

    @property
    def loopback(self):
        host=urlsplit(self.base_url).hostname
        if host=='localhost':return True
        try:return ipaddress.ip_address(host).is_loopback
        except (TypeError,ValueError):return False

    @property
    def configured(self):
        return self.provider=='ollama' or bool(self.base_url and self.model and (self.api_key or self.loopback))


def assistant_config(environ: Optional[Mapping[str,str]]=None):
    values=os.environ if environ is None else environ
    provider=values.get('TERRA_LLM_PROVIDER','ollama').strip().lower()
    if provider not in ('ollama','openai_compatible'):
        raise AssistantConfigurationError('Unsupported assistant provider')
    model=values.get('TERRA_LLM_MODEL','').strip()
    if len(model)>256 or any(c in model for c in '\r\n\x00'):
        raise AssistantConfigurationError('Invalid assistant model identifier')
    # An unrelated key in the environment must never enable a paid provider.
    if provider=='ollama':return AssistantConfig(provider=provider,model=model)
    base=values.get('TERRA_LLM_BASE_URL','').strip().rstrip('/')
    key=values.get('TERRA_LLM_API_KEY','').strip()
    if len(key)>8192 or any(c in key for c in '\r\n\x00'):
        raise AssistantConfigurationError('Invalid assistant credential format')
    if base:
        try:
            parts=urlsplit(base)
            host=parts.hostname
            _=parts.port  # Validate the port without exposing the configured URL.
            local=host=='localhost'
            if not local:
                try:local=ipaddress.ip_address(host).is_loopback
                except (TypeError,ValueError):pass
            if (not host or parts.username is not None or parts.password is not None or
                parts.query or parts.fragment or any(c.isspace() for c in base) or
                (parts.scheme!='https' and not (parts.scheme=='http' and local))):
                raise ValueError
        except (TypeError,ValueError):
            raise AssistantConfigurationError('Assistant base URL requires HTTPS or loopback HTTP, without credentials or query parameters') from None
    return AssistantConfig(provider,base,model,key)


def connection_summary(environ: Optional[Mapping[str,str]]=None):
    """Configuration only: no key, endpoint URL, network probe or readiness claim."""
    try:
        config=assistant_config(environ)
    except AssistantConfigurationError:
        return {'provider':'unconfigured','model':None,'configured':False,
                'mode':'unconfigured','requires_personal_key':False,
                'configuration_error':'Invalid assistant configuration','availability':'not_checked'}
    return {'provider':config.provider,'model':config.model or None,
            'configured':config.configured,
            'mode':'local' if config.provider=='ollama' or config.loopback else 'hosted',
            'requires_personal_key':False,'availability':'not_checked'}


async def chat_json_content(client, model, messages, ollama_url):
    """Return prose JSON only; private reasoning and generic tool calls are ignored."""
    config=assistant_config()
    if config.provider=='ollama':
        local_url=ollama_url() if callable(ollama_url) else ollama_url
        response=await client.post(local_url+'/api/chat',json={
            'model':model,'messages':messages,'stream':False,'format':'json',
            'options':{'temperature':.1,'num_ctx':8192,'num_predict':1536}})
        response.raise_for_status()
        content=response.json()['message']['content']
    else:
        if not config.configured:raise AssistantServiceError('HOSTED_ASSISTANT_NOT_CONFIGURED')
        headers={'Authorization':'Bearer '+config.api_key} if config.api_key else {}
        payload={
            'model':model,'messages':messages,'stream':False,'temperature':.1,
            'max_tokens':1536,'response_format':{'type':'json_object'}}
        # DashScope's hybrid Qwen models require non-thinking mode for this
        # bounded non-streaming JSON protocol. Thinking-only models are not a
        # compatible selection; provider errors remain visible as safe codes.
        host=urlsplit(config.base_url).hostname or ''
        name=model.lower()
        hybrid_qwen=name.startswith(('qwen3','qwen-plus','qwen-flash','qwen-turbo'))
        if host.endswith('.aliyuncs.com') and hybrid_qwen:
            payload['enable_thinking']=False
        if host=='openrouter.ai':
            payload['provider']={'require_parameters':True}
        response=await client.post(config.base_url+'/chat/completions',headers=headers,json=payload)
        if not response.is_success:
            raise AssistantServiceError('ASSISTANT_PROVIDER_HTTP_'+str(response.status_code))
        try:
            message=response.json()['choices'][0]['message']
            if response.json()['choices'][0].get('finish_reason')=='length':
                raise ValueError
            if message.get('tool_calls'):raise ValueError
            content=message['content']
        except (ValueError,KeyError,IndexError,TypeError):
            raise ValueError('Invalid compatible assistant response') from None
    if not isinstance(content,str) or len(content)>65536:
        raise ValueError('Assistant prose must be a bounded JSON string')
    return content
