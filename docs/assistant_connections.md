# Assistant connections and API keys

People using a shared deployment do **not** need personal API keys. The deployment owner configures credentials once on the backend. Public weather, terrain and NASA LHASA requests do not require visitors to supply a key. Earthdata authentication is needed for supported authenticated NISAR data access; local subsets can also be used.

The **Options → Connections** view reports deployment configuration. It does not expose keys, edit global secrets, or claim a provider is healthy merely because configuration exists. It links here for the installation owner.

## Configure an assistant

From the repository root, run:

```bash
python scripts/configure_assistant.py
```

Choose local Ollama or a hosted compatible API. The helper prompts for the endpoint, model and a hidden API key, validates the configuration and saves it to the ignored `.env`. It preserves the Earthdata token and other provider settings. It makes no billable provider call.

Restart the backend with the environment file:

```bash
python -m uvicorn backend.api.main:app --env-file .env --host 127.0.0.1 --port 8000
```

Hosted configuration uses these server-side variables:

```dotenv
TERRA_LLM_PROVIDER=openai_compatible
TERRA_LLM_BASE_URL=https://YOUR_PROVIDER_ENDPOINT/v1
TERRA_LLM_MODEL=YOUR_JSON_CAPABLE_MODEL
TERRA_LLM_API_KEY=YOUR_PROVIDER_KEY
```

Use the base URL and model identifier from your provider console. Do not append `/chat/completions`. Qwen's current endpoints can depend on workspace and region; use an API key for the same region. [Alibaba compatible API guide](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope). [OpenRouter](https://openrouter.ai/developers) also offers a compatible endpoint for multiple model families.

Select a model supporting bounded, non-streaming JSON output. On Alibaba Qwen endpoints the client disables thinking mode for this protocol; thinking-only models need a different protocol and are not supported here. OpenRouter requests require endpoint support for the supplied parameters. JSON syntax alone does not establish evidence correctness: the server still verifies numeric claims and authorized map actions. [Qwen structured-output guidance](https://help.aliyun.com/en/model-studio/qwen-structured-output).

Local inference remains available with `TERRA_LLM_PROVIDER=ollama`. Leave `TERRA_LLM_MODEL` empty to discover the existing custom Qwen/Qwen2.5 service, or set an installed local model explicitly. Existing `TERRA_QWEN_URL` and the trained-adapter endpoint discovery continue to work. The Windows trainer is optional and has not been deleted.

## What the hosted model does

The assistant receives the selected scan's bounded physical column, source summaries, conversation context and authorized display plan. It explains that evidence and may propose only the actions authorized by the user's request. The physical solver, rainfall/terrain classifiers, NASA comparison and NISAR processing remain on the backend.

A hosted API avoids the prerequisite of GPU fine-tuning for chat. It introduces usage charges, network latency, quotas and external processing of the selected evidence and messages. Evaluate verified answer quality and action completion on held-out prompts before choosing a model or deciding that fine-tuning adds value. No claim of improved landslide prediction follows from changing the chat model.

Provider failures use the existing verified template or withheld-prose outcome. `VERIFIED_LLM_PROSE` means the returned prose passed the project's evidence checks; it does not certify global predictive skill. Keys are not sent to the browser, stored in browser storage, included in URLs, or returned through `/api/v1/connections`. Protect a public deployment with authentication, request limits and a spending budget before offering shared paid chat access.

## Three primary prediction choices

The main interface offers **Terra In House Model**, **NASA LHASA**, and **Rainfall baseline**. Physics-only and weather/terrain results remain in advanced diagnostics. These are landslide prediction choices, independent of the assistant's language-model provider. The existing benchmark still favors rainfall-only on aggregate; simplifying the interface does not remove that result.
