# MirrorNeuron Prism

Prism is a Python 3.11+ package for bounded inference through LiteLLM. This release implements the dependable CLI and a standalone, local CPU decision experiment using JEV-CPU's pinned SemIf scorer.

**Generative multicall execution is still planned.** The former dummy completion and first-candidate selector now fail closed. Installing a controller does not activate CPU assistance in schema-v1 policies.

| Capability | Status |
| --- | --- |
| Strict project/policy validation, safe initialization, diagnostics | Implemented |
| Explicit model preparation, CPU classification, bounded worker | Experimental, implemented |
| Native authored/shape fixtures, immutable plans, reports, calibration | Experimental, implemented |
| Shared-prefix and constrained one-token generation treatments | Experimental; parity tested on pinned Qwen3-0.6B |
| Generative engine, CPU/LLM fallback, full agent benchmarks | Planned; unavailable commands are not registered |
| Legacy buffered demo forwarder | Deprecated, loopback only; no streaming |

Install from this checkout:

```bash
python -m pip install -e '.[dev]'
mn_prism init --template cpu-bench --out-dir ./cpu-experiment
mn_prism --config ./cpu-experiment/prism.yaml config validate
mn_prism --config ./cpu-experiment/prism.yaml doctor --format json
```

Initialization never downloads weights and refuses existing files unless `--force` is supplied. CPU dependencies are optional; see [HOW_TO_USE.md](HOW_TO_USE.md) for tested constraints, explicit preparation, experiments, and calibration.

`mn_prism`, `python -m litellm_multicall`, and `litellm_multicall.cli:main(argv)` share one CLI. Running without arguments retains version-only behavior. The root launcher delegates; its old `python mn_prism.py --file ...` grammar selects the deprecated demo.

Help, version, configuration inspection, and benchmark listing never import Torch or load a model. Machine output is versioned JSON/JSONL; child/library logs go to stderr. Configuration and exported metadata redact secrets.

The LiteLLM launcher preserves complete physical model definitions, callbacks, authentication, and child exit status. It refuses configurations requesting unfinished multicall inference. It never creates another Router or silently supplies a test master key. Bind defaults changed to `127.0.0.1`; non-loopback LiteLLM serving requires genuine authentication.

JEV-CPU is an independent SemIf CPU adaptation, not TypeSafe's Jev. Only three reviewed scoring modules are vendored unchanged, with [license and provenance](src/litellm_multicall/_vendor/semif/UPSTREAM.json). Model weights remain separately prepared local artifacts and retain their upstream terms.

Run offline tests:

```bash
python -m pytest -q
python -m ruff check src tests
```

Real model and external endpoint tests are opt-in. The legacy endpoint smoke tests do not validate Prism's unfinished generative engine. See [implementation status](docs/implementation-status.md) for scope and limitations. [SPEC.md](SPEC.md) remains an architecture specification, not a list of working features.
