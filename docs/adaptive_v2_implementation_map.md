# adaptive_v2 implementation map (Step 01 baseline)

Status: baseline survey of the CURRENT post-hardening checkout. No `adaptive_v2`
behavior exists yet; this map locks where each requirement lives today so later
steps extend existing modules instead of inventing duplicates.

## Tested LiteLLM version

- Declared range in `pyproject.toml`: `litellm>=1,<2` (and `litellm[proxy]>=1,<2`
  for the `proxy` extra).
- Currently tested installed version: **litellm 1.101.0** (see
  `tests/unit/test_legacy_regression.py::test_litellm_version_within_declared_range`).
- `CustomLLM` contract verified against the installed release in
  `tests/unit/test_baseline.py::test_provider_matches_installed_custom_llm_contract`
  (`acompletion` / `astreaming` / `completion` / `streaming` on the bound
  instance `provider.multicall_provider`).

## Module ownership (actual current symbols)

| Requirement (SPEC section) | Current owner | Status |
|---|---|---|
| Strict policy schema, loading, reference checks (§5, §12) | `src/litellm_multicall/config.py`: `PrismConfig`, `PolicySpec`, `CandidateSpec`, `SelectionSpec`, `BudgetSpec`, `AdaptiveOptions`, `FallbackSpec`, `ApiSpec`, `RuntimeConfig`; `load_policy_config`, `load_litellm_config`, `validate_references`, `validate_files`, `prism_config_json_schema`; `resources/policy.schema.json` | Implemented, strictly tested (`tests/unit/test_config_strict.py`) |
| Legacy strategies `best_of_n` / `adaptive` (§7.1, §7.2) | `config.PolicySpec.strategy` (`Literal["best_of_n", "adaptive"]`); `config.AdaptiveOptions` | Schema only; execution is a stub (see engine) |
| Per-request execution (§6, §7) | `src/litellm_multicall/engine.py`: `ExecutionEngine(backend)`, `ExecutionEngine.run(request_context, policy)` | Stub: returns empty `ExecutionResult`, performs no calls |
| Backend abstraction, single Router instance (§6.1) | `src/litellm_multicall/backend.py`: `CompletionBackend` protocol (`complete(model_group, messages, parameters, context)`), `ProxyRouterBackend(router)` | Stub: `complete` raises `NotImplementedError` |
| Budgets / reservations (§11.1–11.2) | `src/litellm_multicall/budgets.py`: `BudgetTracker(max_model_calls)`, `BudgetTracker.reserve()` | Minimal counter only; no token/deadline/attempt accounting yet |
| Validation / verifier protocol (§8.2, §9.4) | `src/litellm_multicall/validation.py`: `validate_output(output)` | Stub: truthiness check only |
| Judge / deterministic selection (§7.1, §8.1) | `src/litellm_multicall/selection.py`: `select_best(candidates)`; prompt `resources/judge.txt` | Stub: returns first candidate |
| Streaming, buffered-final (§10) | `src/litellm_multicall/streaming.py`: `convert_to_chunks(response)`; `provider.MulticallProvider.astreaming/streaming` | Stub: returns `[]` / raises `NotImplementedError` |
| Usage / tracing (§14) | `src/litellm_multicall/telemetry.py`: `log_event(event)`; `src/litellm_multicall/meter.py`: `RequestMeter`, `ModelUsage`, `estimate_cost`, `get_price_estimate` | `log_event` is a no-op; `RequestMeter` aggregates offline only |
| Version-sensitive LiteLLM integration (§3.3, §15, §20) | `src/litellm_multicall/compat/proxy_runtime.py`, `compat/request_context.py`, `compat/accounting.py`, `compat/errors.py`, `compat/versions.py` | Stubs (comment-only); all version-sensitive logic MUST land here |
| Provider entry point (§3.1) | `src/litellm_multicall/provider.py`: `MulticallProvider(CustomLLM)`, instance `multicall_provider` | Fail-closed: `acompletion`/`astreaming` raise `NotImplementedError` |
| Hooks / request context (§3.1, §6.3) | `src/litellm_multicall/hooks.py`: `multicall_hooks` | Dummy placeholder object |
| Typed records (§6.2) | `src/litellm_multicall/types.py`: `RequestContext`, `ChildContext`, `Candidate`, `ExecutionResult` | Minimal dataclasses |
| CLI (§4.4) | `src/litellm_multicall/cli.py`: `main()` with `validate` / `doctor` / `init` / `serve` / `proxy` subcommands | `validate` is strict and offline; `doctor` is a placeholder print |
| Example configs (§5) | `examples/litellm.yaml`, `examples/multicall.yaml` (`mn_prism init` regenerates an equivalent pair) | Load and cross-validate cleanly |
| Benchmarks (§18.4) | `benchmarks/run.py`, `benchmarks/README.md` | Stubs |
| Legacy regression lock (this step) | `tests/unit/test_legacy_regression.py`, `tests/unit/test_baseline.py` | Must keep passing unchanged |

## Notes for later steps

- `adaptive_v2` is opt-in and does not exist yet. `PolicySpec.strategy` rejects
  anything outside `best_of_n` / `adaptive`; no existing policy may gain new
  calls when `adaptive_v2` code lands (locked by
  `test_existing_policies_gain_no_new_calls` and `test_no_adaptive_v2_symbols_in_package`).
- Ordinary (non-`multicall/*`) LiteLLM aliases must never route through
  `MulticallProvider` (locked by `test_ordinary_aliases_untouched`).
- Do not add a second gateway, router, provider SDK layer, auth, DB, queue, or
  UI. All model-role calls go through `backend.CompletionBackend` and the same
  injected router instance.
