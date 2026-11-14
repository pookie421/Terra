# Assistant models and personal API keys

Open **Assistant → AI Model & API Key**, or **Options → AI Model & API Key**. Choose a provider, enter its model ID, paste your API key, and save. The model ID is editable so you can use a supported model available in your own account.

| Provider | Connection |
|---|---|
| OpenAI | OpenAI chat completions |
| Google Gemini | Gemini's OpenAI-compatible endpoint |
| Anthropic Claude | Anthropic Messages API |
| Groq | Groq's OpenAI-compatible endpoint |
| OpenRouter | OpenRouter chat completions; use its model identifiers |
| Custom endpoint | A public HTTPS OpenAI-compatible base URL, including supported hosted Qwen services |

The selected provider must support structured JSON responses with the requested model. Provider/model availability, quotas and usage charges belong to your account. A saved key means configuration is present; a successful verified reply establishes that the selected service responded.

## Personal keys only

Generated assistant prose uses the key supplied with that request. Public chat endpoints do not fall back to an operator's paid key, local Ollama service or default server model. No shared AI subscription or GPU server is required.

The masked key is kept in this tab's browser session storage and sent over HTTPS to this deployment's backend for the selected provider request. It is excluded from response bodies, validation errors, URLs, exports and application logs, and is not saved to a server file. **Clear key** removes it from the active settings. Provider/model preferences are independent of the landslide prediction selector.

Without a key, **deterministic mode** still explains verified telemetry and executes supported map commands. It is visibly marked as a verified template. No paid language-model request is made.

## What the assistant receives

The selected provider receives your message, bounded conversation history and the active analysis evidence: scan/state/cell identifiers, the selected prediction model, applicable NASA daily product, source summaries, conditional physical results and forecast/runout context. Your API key is an authentication header to the provider, not prompt content.

The server checks numeric bindings, state identity and the user's permitted action plan before rendering generated prose or map commands. Short public agent steps describe evidence and intended actions; they are not private model deliberation. A model change, cell change, forecast change or NASA product change cancels stale replies/actions.

**VERIFIED_LLM_PROSE** identifies generated prose that passed those checks. **VERIFIED_TEMPLATE** identifies the deterministic path. Provider failures and rejected prose remain distinct; neither verification label validates the geophysical model itself.

## Custom endpoints

Enter the provider's public HTTPS base URL, without appending chat/completions. Private addresses, local network endpoints, embedded credentials, query-string secrets and redirects are rejected by the public backend. DNS resolution is checked and the connection is pinned to the verified address while preserving TLS hostname verification.

For Qwen, use a compatible hosted endpoint and a JSON-capable model from the same region as your API key, or select a Qwen model through OpenRouter. Existing local/CLI research transports remain separate from the public website's personal-key policy. The retired [Windows training workflow](legacy/local_llm_training.md) is not a website requirement.

## Prediction models and satellite access

**Terra In House Model**, **NASA LHASA** and **Rainfall baseline** select landslide analyses; they do not select the assistant provider. NASA's official Today/Tomorrow products retain their native support and do not acquire Terra's 72-hour forecast capability.

Public weather, terrain and LHASA services do not need visitors' AI keys. Authenticated measured NISAR extraction uses a separately configured server-side Earthdata token. The radar panel shows whether extraction is processing, measured pixels are available, authentication is needed or the area has no valid product pixels. A discovered granule alone is not a radar measurement. See [NASA and radar methods](nasa_radar_comparison.md).
