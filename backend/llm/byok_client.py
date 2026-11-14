"""Request-scoped personal-key transport for the public co-pilot.

No environment credentials, local model discovery, credential cache or disk
configuration participates in this path. The legacy CLI transport is separate.
"""
import asyncio
from dataclasses import dataclass, field
import ipaddress
import json
import socket
from urllib.parse import urlsplit, urlunsplit

from .model_client import AssistantConfigurationError, AssistantServiceError


PROVIDERS = ('openai', 'gemini', 'anthropic', 'openrouter', 'groq', 'custom_openai')
MAX_RESPONSE_BYTES = 256 * 1024
ENDPOINTS = {
    'openai': 'https://api.openai.com/v1/chat/completions',
    'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    'anthropic': 'https://api.anthropic.com/v1/messages',
    'openrouter': 'https://openrouter.ai/api/v1/chat/completions',
    'groq': 'https://api.groq.com/openai/v1/chat/completions',
}


def _public_address(value):
    address = ipaddress.ip_address(value)
    return address.is_global and not (address.is_multicast or address.is_unspecified or
        address.is_reserved or address.is_loopback or address.is_link_local)


@dataclass(frozen=True)
class RequestAssistantConfig:
    provider: str = 'openai'
    model: str = ''
    base_url: str = field(default='', repr=False)
    api_key: str = field(default='', repr=False)

    @property
    def configured(self):
        return bool(self.api_key and self.model)


def request_assistant_config(provider='openai', model='', base_url=None, api_key=None):
    """Validate values without exposing the credential or endpoint in errors."""
    if provider not in PROVIDERS:
        raise AssistantConfigurationError('Unsupported personal assistant provider')
    if not isinstance(model, str) or len(model) > 256 or any(c in model for c in '\r\n\x00'):
        raise AssistantConfigurationError('Invalid assistant model identifier')
    model = model.strip()
    key = '' if api_key is None else api_key
    if not isinstance(key, str) or len(key) > 8192 or any(c in key for c in '\r\n\x00'):
        raise AssistantConfigurationError('Invalid personal assistant credential format')
    key = key.strip()
    if key and not model:
        raise AssistantConfigurationError('Choose an assistant model before using a personal key')
    base = ''
    if provider == 'custom_openai' and (key or base_url):
        if not isinstance(base_url, str) or len(base_url) > 2048:
            raise AssistantConfigurationError('Custom assistant endpoint requires a public HTTPS URL')
        try:
            parts = urlsplit(base_url.strip().rstrip('/'))
            host = parts.hostname
            if (parts.scheme != 'https' or not host or parts.port not in (None, 443) or
                    parts.username is not None or parts.password is not None or
                    parts.query or parts.fragment or any(c.isspace() for c in base_url)):
                raise ValueError
            host = host.encode('idna').decode('ascii')
            # Numeric private endpoints are rejected before any DNS operation.
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                if host.lower().rstrip('.') in ('localhost', 'localhost.localdomain'):
                    raise AssistantConfigurationError('Custom assistant endpoint must use a public network address')
            else:
                if not _public_address(address):
                    raise AssistantConfigurationError('Custom assistant endpoint must use a public network address')
            authority = '[' + host + ']' if ':' in host else host
            base = urlunsplit(('https', authority, parts.path.rstrip('/'), '', ''))
        except (TypeError, ValueError, UnicodeError):
            raise AssistantConfigurationError('Custom assistant endpoint requires public HTTPS on port 443, without credentials or query parameters') from None
    return RequestAssistantConfig(provider, model, base, key)


def byok_connection_summary():
    return {'provider': None, 'model': None, 'configured': False, 'mode': 'byok',
            'requires_personal_key': True, 'availability': 'not_checked',
            'providers': list(PROVIDERS), 'credential_scope': 'request',
            'server_credentials_enabled': False}


def request_provenance(config, guard, model_error=None):
    availability = ('verified_response' if guard == 'VERIFIED_LLM_PROSE' else
                    'unavailable' if model_error else 'not_checked')
    return {'assistant_provider': config.provider if config.configured else None,
            'assistant_mode': 'byok' if config.configured else 'deterministic',
            'model_provenance': {'provider': config.provider if config.configured else None,
                'model': config.model if config.configured else None,
                'mode': 'byok' if config.configured else 'deterministic',
                'credential_source': 'request' if config.api_key else 'none',
                'availability': availability}}


async def _public_destination(url):
    """Pin custom requests to a validated public IP, retaining TLS SNI and Host.

    Checking DNS and then letting HTTPX resolve again would permit DNS rebinding.
    Every candidate address must be public; redirects are disabled by the caller.
    """
    parts = urlsplit(url)
    host = parts.hostname
    try:
        addresses = await asyncio.wait_for(asyncio.get_running_loop().getaddrinfo(
            host, 443, type=socket.SOCK_STREAM), timeout=5.)
    except (asyncio.TimeoutError, OSError, UnicodeError):
        raise AssistantServiceError('ASSISTANT_ENDPOINT_DNS_FAILED') from None
    candidates = list(dict.fromkeys(row[4][0] for row in addresses))
    if not candidates or any(not _public_address(value) for value in candidates):
        raise AssistantServiceError('ASSISTANT_ENDPOINT_NOT_PUBLIC')
    address = candidates[0]
    authority = '[' + address + ']' if ':' in address else address
    pinned = urlunsplit((parts.scheme, authority, parts.path, '', ''))
    return pinned, host


async def request_chat_json_content(client, config, messages):
    if not config.configured:
        raise AssistantServiceError('PERSONAL_API_KEY_REQUIRED')
    endpoint = ENDPOINTS.get(config.provider)
    # Explicit identity encoding permits a bound on bytes before JSON decoding,
    # including for an arbitrary user-selected custom public endpoint.
    headers = {'Accept-Encoding': 'identity'}
    extensions = None
    if config.provider == 'custom_openai':
        endpoint = config.base_url
        if not endpoint.endswith('/chat/completions'):
            endpoint += '/chat/completions'
        endpoint, hostname = await _public_destination(endpoint)
        headers['Host'] = '[' + hostname + ']' if ':' in hostname else hostname
        extensions = {'sni_hostname': hostname}
    if config.provider == 'anthropic':
        headers.update({'x-api-key': config.api_key, 'anthropic-version': '2023-06-01'})
        system = '\n\n'.join(row['content'] for row in messages if row['role'] == 'system')
        payload = {'model': config.model, 'system': system,
                   'messages': [row for row in messages if row['role'] != 'system'],
                   'max_tokens': 1536, 'temperature': .1, 'stream': False}
    else:
        headers['Authorization'] = 'Bearer ' + config.api_key
        payload = {'model': config.model, 'messages': messages, 'stream': False,
                   'temperature': .1, 'max_tokens': 1536,
                   'response_format': {'type': 'json_object'}}
        if config.provider == 'openrouter':
            payload['provider'] = {'require_parameters': True}
        # Qwen's hybrid hosted models need non-thinking mode for this bounded,
        # non-streaming structured-prose protocol, as on the retained transport.
        host = urlsplit(config.base_url).hostname or ''
        if host.endswith('.aliyuncs.com') and config.model.lower().startswith(
                ('qwen3', 'qwen-plus', 'qwen-flash', 'qwen-turbo')):
            payload['enable_thinking'] = False
    async with client.stream('POST', endpoint, headers=headers, json=payload,
                             extensions=extensions, follow_redirects=False) as response:
        if not response.is_success:
            raise AssistantServiceError('ASSISTANT_PROVIDER_HTTP_' + str(response.status_code))
        if response.headers.get('Content-Encoding', '').lower().strip() not in ('', 'identity'):
            raise AssistantServiceError('ASSISTANT_RESPONSE_ENCODING_UNSUPPORTED')
        declared = response.headers.get('Content-Length', '')
        if declared.isdigit() and int(declared) > MAX_RESPONSE_BYTES:
            raise AssistantServiceError('ASSISTANT_RESPONSE_TOO_LARGE')
        chunks = []; size = 0
        async for chunk in response.aiter_bytes(chunk_size=16384):
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise AssistantServiceError('ASSISTANT_RESPONSE_TOO_LARGE')
            chunks.append(chunk)
        raw = b''.join(chunks)
    try:
        result = json.loads(raw)
        if config.provider == 'anthropic':
            if result.get('stop_reason') == 'max_tokens':
                raise ValueError
            blocks = result['content']
            if any(block.get('type') == 'tool_use' for block in blocks):
                raise ValueError
            content = ''.join(block['text'] for block in blocks if block.get('type') == 'text')
        else:
            choice = result['choices'][0]
            if choice.get('finish_reason') == 'length' or choice['message'].get('tool_calls'):
                raise ValueError
            content = choice['message']['content']
    except (ValueError, KeyError, IndexError, TypeError):
        raise ValueError('Invalid personal assistant response') from None
    if not isinstance(content, str) or not content or len(content) > 65536:
        raise ValueError('Assistant prose must be a bounded JSON string')
    return content
