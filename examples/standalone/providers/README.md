# Native OpenAI, Claude, and Gemini samples

These are expected configurations based on current provider documentation and local LiteLLM SDK protocol fixtures. They have **not** been tested against live paid APIs: no paid-provider keys were available. Model access, parameters, output limits, and availability must be qualified with your account. Unlike the OpenRouter-only preset, these examples include paid models.

## Select a provider

| Config | Small worker | Strong final | Credential |
|---|---|---|---|
| `prism-openai.json` | `openai/gpt-6-luna` | `openai/gpt-6-astra` | `OPENAI_API_KEY` |
| `prism-claude.json` | `anthropic/claude-haiku-4-5` | `anthropic/claude-opus-5-5` | `ANTHROPIC_API_KEY` |
| `prism-gemini.json` | `gemini/gemini-3.5-flash-lite` | `gemini/gemini-3.8-flash` | `GEMINI_API_KEY` |

The shared `models.json` defines each physical model once. The combined `prism.json` serves all profiles. `prism init --preset providers --out-dir NEW_DIRECTORY` generates that combined configuration from the installed wheel. Per-provider config files restrict virtual profiles; shared physical models still get direct aliases, and `doctor --probe-backends` checks the whole registry. Use `capacity --model SELECTED_ALIAS` to qualify only your selected model/profile, or trim the registry for a single-provider deployment.

```sh
export PRISM_API_KEY='your-prism-client-secret'
export OPENAI_API_KEY='your-openai-key'
prism validate --config examples/standalone/providers/prism-openai.json
prism capacity --config examples/standalone/providers/prism-openai.json --model prism-openai
prism serve --config examples/standalone/providers/prism-openai.json
```

For Claude, set `ANTHROPIC_API_KEY` and select `prism-claude.json`; for Gemini, set `GEMINI_API_KEY` and select `prism-gemini.json`. Keys come only from environment variables; no sample embeds credentials. Prism client auth uses `PRISM_API_KEY` or an explicit serving `--no-auth` flag.

## Workflow variants

Replace `VENDOR` with `openai`, `claude`, or `gemini`:

| Alias | Stages | Keys used |
|---|---|---|
| `prism-VENDOR` | Automatic direct / source-backed evidence; small worker → strong final | Native vendor |
| `prism-VENDOR-reviewed` | Small draft → strong review → strong final | Native vendor |
| `prism-VENDOR-vision` | Small image interpretation → strong final | Native vendor |
| `nano-VENDOR` | Free Nano plain-text preparation → strong final | OpenRouter + native vendor |
| `omni-VENDOR` | Free Nano image interpretation → strong final | OpenRouter + native vendor |

All final models are explicitly assigned for JSON. Free Nano retains no JSON declaration. Mixed free/native profiles require `OPENROUTER_API_KEY` as well as the native key. Native-only aliases do not need OpenRouter. Worker observations are lossy; use source-backed evidence policies when quote provenance matters. Image understanding produces text, not generated images or audio/video output.

`prism-gemini-pro` optionally uses `gemini/gemini-3.1-pro-preview`; qualify preview availability separately. Context limits are conservatively set to 131072 tokens and output to 16384; profiles cap public/worker generation at 4096, with 300-second graph deadlines. No speculative native prices or power ratings are configured. Set verified operator prices if you want dollar ceilings or cost optimization.

## Qualification

1. Validate the configuration and inspect `prism models` credential readiness.
2. Run `prism capacity --model PHYSICAL_ID` to test actual model behavior independently of declarations.
3. Run `prism capacity --model PROFILE_ALIAS` to test the real graph's final JSON, pixels, and reasoning outputs.
4. Send your own task fixtures, including JSON Schema, streaming, and images where applicable. A single passing challenge is not a general guarantee.

References checked October 4, 2026: [OpenAI Astra](https://developers.openai.com/api/docs/models/gpt-6-astra), [Claude model updates](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5), [Gemini models](https://ai.google.dev/gemini-api/docs/models), [LiteLLM Anthropic](https://docs.litellm.ai/docs/providers/anthropic), [LiteLLM Gemini](https://docs.litellm.ai/docs/providers/gemini), and [LiteLLM JSON behavior](https://docs.litellm.ai/docs/completion/json_mode). The native SDK translates the wire protocol; Prism validates the final output. There are no automatic provider fallbacks or silent parameter drops.
