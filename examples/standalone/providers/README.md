# OpenAI, Claude, and Gemini

Use the individual model definitions in the repository's [models/](../../../models) and combinations in [profiles/](../../../profiles). The same definitions are bundled in the package for `prism init`. Provider connections, model names, and credentials are in model files; profile files reference their IDs.

## Get a first response

After installing this checkout, use a fresh deployment directory:

```sh
mkdir prism-openai-demo && cd prism-openai-demo
prism init --preset openai
export OPENAI_API_KEY='your-openai-key'
export PRISM_API_KEY='your-prism-client-secret'
prism start --profile prism-openai-direct --show-cost
```

Send Chat Completions with `model: "prism-openai-direct"`. For Claude, initialize with `--preset claude`, set `ANTHROPIC_API_KEY`, and start `prism-claude-direct`. For Gemini, use `--preset gemini`, `GEMINI_API_KEY`, and `prism-gemini-direct`. `PRISM_API_KEY` always protects the Prism client connection unless `--no-auth` is explicitly supplied. Keys come from the environment.

| Model IDs | Physical models | Native transport | Credential |
|---|---|---|---|
| `openai-small`, `openai-strong` | `gpt-6-luna`, `gpt-6-astra` | `openai` at `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `claude-small`, `claude-strong` | `anthropic/claude-haiku-4-5`, `anthropic/claude-opus-5-5` | `anthropic` at `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |
| `gemini-small`, `gemini-strong` | `gemini/gemini-3.5-flash-lite`, `gemini/gemini-3.8-flash` | `gemini` at `https://generativelanguage.googleapis.com` | `GEMINI_API_KEY` |
| `gemini-pro` | `gemini/gemini-3.1-pro-preview` | `gemini` | `GEMINI_API_KEY` |

Model IDs were checked against official docs on October 5, 2026: [OpenAI Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [OpenAI Astra](https://developers.openai.com/api/docs/models/gpt-6-astra), [Claude models](https://platform.claude.com/docs/en/models/overview), and [Gemini models](https://ai.google.dev/gemini-api/docs/models). Context/output bounds in the files are conservative operator limits, rather than full vendor maxima. Model access and behavior require qualification with your account. These examples have configuration and local LiteLLM protocol-fixture coverage; no live paid-cloud qualification is claimed.

## Combine models

Stop the current server and start a different profile; use that profile's `id` in your request. Replace `VENDOR` with `openai`, `claude`, or `gemini`:

| Profile | Workflow |
|---|---|
| `prism-VENDOR-direct` | One small-model call |
| `prism-VENDOR` | Automatic direct / source-backed small worker → strong final |
| `prism-VENDOR-reviewed` | Small draft → strong review → strong final |
| `prism-VENDOR-vision` | Small image interpretation → strong text answer |
| `nano-VENDOR` | Free Nano plain-text preparation → native final |
| `omni-VENDOR` | Free Nano image observations → native final |

`prism init --preset providers` includes all of these combinations. A selected native-only profile needs only that vendor's key. Mixed Nano/native profiles also need `OPENROUTER_API_KEY`; loading other sample files does not require their credentials. `prism-gemini-pro` selects the preview model; qualify preview availability separately.

Free Nano has no JSON declaration. Final models are assigned explicitly for caller JSON requirements. Worker observations can lose information; source-backed evidence policies validate quote provenance separately. Vision means image understanding and text output.

## Prices and qualification

Set both token prices in `models/MODEL_ID.json`, using numbers or strings such as `"$5/m"`, then add `--show-cost` at startup. The Haiku sample includes the documented standard $1 input / $5 output rates per million tokens; other native prices are left unknown for the operator to set. Use your actual rates, including zero for an applicable free tier. [Cost calculation and limits](../../../docs/cost-tracking.md).

```sh
prism validate --profile prism-openai
prism models --profile prism-openai
prism capacity --profile prism-openai
prism capacity --profile prism-openai --model openai-small
```

Capacity makes real inference calls and may be billed. Follow those small behavior checks with your own text, structured output, image, and streaming fixtures. [LiteLLM Anthropic](https://docs.litellm.ai/docs/providers/anthropic) and [LiteLLM Gemini](https://docs.litellm.ai/docs/providers/gemini) describe native protocol conversion; Prism independently validates final JSON. It performs no hidden provider fallback or silent parameter dropping.

## Older multi-alias examples

The JSON files beside this README are retained **legacy combined configurations**. They use `prism serve --config examples/standalone/providers/prism-openai.json` (or the Claude/Gemini equivalent) and expose raw registry aliases as well as virtual profiles. Their shared registry includes all providers, so serving it requires all configured upstream keys. Prefer the standalone profile commands above for a single-provider deployment. New `init` output has no combined `prism.json` or `models.json`.
