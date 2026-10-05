# Release validation: 0.3.0

Checked and published October 4, 2026 (America/New_York). [Release to PyPI #1](https://github.com/homerquan/mirrorneuron-prism/actions/runs/37257249826) succeeded for commit `b79aab4e85056ab1992e8c3cac56dc3007de70c0`, publishing the wheel and source distribution to [PyPI](https://pypi.org/project/mirrorneuron-prism/0.3.0/) using Trusted Publishing. No Git tag was created.

| Check | Result |
|---|---|
| Ruff and whitespace checks | Passed |
| Deterministic tests against the installed wheel | 347 passed; 12 integration tests excluded |
| Localhost HTTP / SDK / curl integration tests against that wheel | 12 passed |
| Combined statement/branch coverage | 87.58%, displayed as 88%; 80% CI floor |
| CLI / terminal presentation coverage | CLI 89%; UI 99% |
| Capacity / evaluation coverage | Capacity 96%; evaluation 98% |
| Build | Wheel and sdist built successfully; wheel built from sdist |
| Twine metadata and README validation | Wheel and sdist passed |
| Fresh virtual environment | Wheel installed with normal runtime dependencies; import resolves outside checkout |
| Installed resources | Local, free OpenRouter, native-provider presets; six benchmark cases; typing marker; MIT license |
| Installed CLI | Version, preset initialization/validation, JSON output, and benchmark help passed outside checkout |
| Docker | Built `mirrorneuron-prism:0.3.0`; non-root UID 10001; real CPU checkpoint startup; healthy |
| Docker auth behavior | Default missing client key exits 2; explicit `--no-auth` serves health and model inventory |
| Docker completion | Live free Super JSON Schema answer and trace confirmed JSON-aware routing |
| Published PyPI installation | Clean virtual environment, CLI version, dependency consistency, and all three installed presets passed outside the checkout |

The fresh environment used Python 3.12.14 on macOS ARM64, LiteLLM 1.104.0, Laya 0.3.27, FastAPI 0.142.2, Pydantic 2.13.5, Rich 15.0.0, and rich-argparse 1.8.0. Docker uses Python 3.12 on Linux ARM64 and CPU torch. Python 3.11–3.13 are configured in CI; other platforms/versions were not executed locally. Upstream Pydantic/LiteLLM warnings did not fail the tests. Checkpoint preparation was verified separately from fixture-based test coverage.

## Live provider limits

The OpenRouter preset contains only explicit `:free` models. Actual image/JSON/reasoning challenges were run. Nano → Super image synthesis and the profile's end-to-end JSON, JSON Schema, image, and reasoning checks passed. Two Nano → Ultra smoke attempts failed at Nano's upstream worker stage; both are retained. Nano's direct JSON capability remains deliberately disabled, while JSON-required profiles choose an explicit capable final model. Native OpenAI, Claude, and Gemini samples passed local configuration and SDK protocol checks but have no live paid-provider qualification.

The [live benchmark](evaluations/2026-10-04-openrouter/benchmark-results.md) reports 56.3% lower hypothetical premium token cost across nine matched completed pairs. It also reports lower all-attempt acceptance (9/12 → 6/12) and higher mean latency (4.83s → 19.96s). Missing usage is never priced as zero. The [marketing narrative](evaluations/2026-10-04-openrouter/marketing-narrative.md) keeps those qualifications adjacent to the claim.

## Published artifacts

- `dist/mirrorneuron_prism-0.3.0-py3-none-any.whl`
- `dist/mirrorneuron_prism-0.3.0.tar.gz`
- `dist/SHA256SUMS-0.3.0.txt`

Both published files were downloaded and their hashes matched PyPI's metadata and the prepared artifacts. All 38 package source/resource files in the published wheel matched the reviewed checkout; version and MIT license metadata passed verification.

| Published file | SHA-256 |
|---|---|
| `mirrorneuron_prism-0.3.0-py3-none-any.whl` | `78346dab83db29896b7f59f419c87faee36fa6ec44295c2f89142ac36be71c40` |
| `mirrorneuron_prism-0.3.0.tar.gz` | `ea80f5d5d023ce847d2409c9f45705dd706f18efb77276dd5426874abda76074` |

These are post-publication records; the immutable 0.3.0 distributions contain the preparation documentation from the release commit. Follow [manual release instructions](releasing.md) for subsequent releases.
