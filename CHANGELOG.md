# Changelog

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
