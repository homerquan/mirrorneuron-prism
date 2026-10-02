# Implementation status — 2026-10-01

The requested delivery is the specification's P0–P2 standalone CPU measurement slice. P3 and later integrations remain unavailable. The current checkout predates the reviewed Prism revision: its policy loader was permissive, its engine returned empty results, and its provider returned a dummy completion. Those paths now validate strictly or fail closed. The existing local `.DS_Store` change was preserved; JEV-CPU source was not edited.

| Milestone | Changed areas | Verification |
| --- | --- | --- |
| P0 | Thin CLI facade, parser/handlers, project context, redacted output, strict policy/project schemas, atomic init, root/module launchers, diagnostics, LiteLLM config/process ownership, fail-closed stubs | Offline CLI/config/process regression tests; both option placements; physical model/callback/auth preservation and child status/temp cleanup |
| P1 | Private pinned SemIf vendor, immutable artifact lock, explicit fetch/verify, CPU loader, typed decision contracts, bounded spawned executor, direct/batch classification, controller meter/telemetry | Real cached Qwen CPU inference; upstream direct parity; no implicit download, byte/token limits, timeout/cancel/reap/restart and state staleness tests; installed-wheel inference |
| P2 | Pinned local authored/shape preparation, scorer/evaluator allowlist boundary, grouped splits, immutable plans/manifests, flushed run evidence, rules/direct/shared/compact baselines, calibration export/fit/gates, compare/report and run inspection/export | All 144 authored rows exercised across calibration/validation/test; paired 48-case test arms; 42-criterion direct/shared shape smoke; calibrated gate abstained; comparison withheld parity; artifacts and unknown measurements verified |

Primary implementation lives in `src/litellm_multicall/commands/`, `classifier/`, `benchmarks/`, `project_config.py`, `config.py`, `storage.py`, `meter.py`, and `telemetry.py`. Packaging includes generated policy/project/decision/result/run schemas, prompts, typing marker, and exact upstream license/hash provenance. Existing public distribution, CLI, provider/hook paths, and Python 3.11 support are preserved. Schema-v1 policies do not invoke CPU controllers.

Validation commands and outcomes:

- `.venv/bin/python -m pytest -q`: 89 passed, 3 opt-in tests deselected, Python 3.12.
- `/private/tmp/prism-cpu-test-env/bin/python -m pytest -q`: 89 passed, 3 opt-in tests deselected, Python 3.11.
- `/private/tmp/prism-cpu-test-env/bin/python -m ruff check src tests`: passed.
- `python -m pytest -q -m real_model tests/integration/test_cpu_model.py` with explicit local snapshot/revision and offline Hugging Face flags: passed; direct/shared/compact parity and token rejection tested. The installed wheel was also tested outside the checkout.
- `python -m build --wheel --sdist`: succeeded. Archive inspection verified prompts/schemas/typing marker/vendor notices and exact vendored scorer SHA-256 hashes. No model weights or caches are packaged. A real two-row batch from the delivery wheel outside the checkout preserved both IDs and loaded the model once.
- `git diff --check`: passed.

Real-model tests require `PRISM_TEST_MODEL_SNAPSHOT` and `PRISM_TEST_MODEL_REVISION`; default tests do not download models or call endpoints. Native Apple Silicon constraints are tested; the Linux CPU recipe is explicitly unverified on Linux hardware. Generative LiteLLM compatibility and external endpoint tests were not run because generative execution is unfinished.

Local measurements are saved under `.prism/validation-report.md` and `.prism/runs/`, with every denominator and uncertainty limitation. The pinned Qwen model scored 25/48 on the diagnostic held-out native split. No 5% selective-risk gate or quality non-inferiority claim was established. Shared-prefix mode helped the long systems fixture and regressed scoring CPU time on the short authored fixture; no whole-system savings are claimed. Run wall timing excludes CLI preflight, artifact hashing and manifest creation, whose overhead remains unmeasured.

Next milestone: P3 BFCL-derived decision probes and an actual bounded engine/backend/provider baseline using the existing LiteLLM Router, replacing fail-closed stubs only with tested behavior. Policy v2 CPU shadow/enforce, explicit LLM fallback, full BFCL/τ/SWE-bench, resume/sweeps, context filtering, resource sensors, and CPU/GPU overlap remain future work. Config migration and doctor network probes are not registered. No recurring jobs or external-agent tools were created.
