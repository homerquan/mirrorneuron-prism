# Changelog

## 0.4.0 — 2026-10-05

- Check that the release version is new on PyPI before building; keep package/runtime versions synchronized and serialize release runs. Update CI/release actions to Node.js 24 versions.

- Turn `--show-cost` into a live full-screen terminal dashboard, with per-model usage, uptime, safe server events, and a final summary; retain JSON reports for scripts.
- Restore `prism` to the free local Gemma/Spark pair; name the OpenRouter automatic workflow `prism-openrouter`. Add `prism-mock-cost` with real OpenRouter Super/Ultra calls and explicitly hypothetical rates for savings demonstrations.
- Remove the older top-level combined configs; standalone workflows are in `profiles/`, with bundled multi-alias examples retained for experiments.

- Separate reusable connection definitions in `models/` from individual workflows in `profiles/`; add `prism start --profile NAME` with only the selected models and credentials.
- Generate matching local, OpenRouter, OpenAI, Claude, Gemini, and mixed-provider samples from installed presets; preserve legacy `serve --config` configurations.
- Accept numeric USD token rates and shorthand such as `"$5/m"`; add `--show-cost` and authenticated `/v1/prism/costs` for process-lifetime input/output token spend and explicit estimated savings.
- Keep unpriced/unreported usage, excluded comparisons, zero baselines, and negative savings visible.
- Rewrite the first-answer quick start, provider/configuration/cost guides, and Docker command; remove the dummy Spark credential from the retained local registry.

## 0.3.1 — 2026-10-04

- Fix the PyPI description's prism image and documentation links with absolute public URLs.
- Render the model-routing diagram as a PNG supported by both GitHub and PyPI, retaining its Mermaid source for editing.
- Update installation instructions and release records for the published package.

Runtime behavior and model profiles are unchanged from 0.3.0.

## 0.3.0 — 2026-10-04

- Add bounded `vision_synthesis` and `text_synthesis` workflows with explicit observation coverage, upfront reservations, and validated final output.
- Add free OpenRouter Nano Omni / Super / Ultra combinations, JSON-capable final-model selection, and native OpenAI / Claude / Gemini example profiles using environment credentials.
- Evaluate actual physical-model and end-to-end profile JSON, JSON Schema, image, and reasoning behavior. Preserve unknown upstream outcomes separately from failed checks.
- Add explicit `prism serve --no-auth` client-auth opt-out; upstream authentication remains independent.
- Improve CLI help and terminal output using Rich, with JSON for scripts, model/profile inventories, and installed presets.
- Fix compressed-response handling and provider errors inside HTTP 200 envelopes in the guarded LiteLLM transport.
- Add Docker/Compose deployment, package resources, coverage checks, usage guides, and a reproducible live evaluation with qualified marketing narrative.

Existing local registries and profiles remain supported. Observation preparation is lossy; it does not replace source-provenance validation. Paid provider samples were checked locally and require live qualification with the user's own keys. Published to [PyPI](https://pypi.org/project/mirrorneuron-prism/0.3.0/) through the manually dispatched Trusted Publishing workflow.
