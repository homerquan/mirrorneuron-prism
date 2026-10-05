# Releasing mirrorneuron-prism

This checkout prepares **0.4.0** with standalone profiles and the cost dashboard. The previous published release is [0.3.1](https://pypi.org/project/mirrorneuron-prism/0.3.1/). Distribution: `mirrorneuron-prism`; import and CLI: `prism`. Publication uses the manual workflow and does not happen during a local build.

[Release to PyPI #3](https://github.com/homerquan/mirrorneuron-prism/actions/runs/37384898815) and [#4](https://github.com/homerquan/mirrorneuron-prism/actions/runs/37386906770) both used commit `8a9112c`, which still built `0.3.1`. Run #4 passed the build and tests, then PyPI returned `400 File already exists` for `mirrorneuron_prism-0.3.1-py3-none-any.whl`. PyPI rejects reused artifact names. Commit and push the new version, then start a **new workflow run** on the updated branch. GitHub's **Re-run jobs** reuses the original commit and cannot apply a version fix.

The Node.js deprecation annotations concern GitHub's JavaScript helper actions (checkout, Python setup, and artifact transfer). They do not add Node.js to Prism's runtime dependencies. Those actions succeeded in run #4; the PyPI upload error caused the failure.

The workflow checks that package/runtime versions agree and that the release is absent from PyPI before installing dependencies. A PyPI connectivity/service error also stops the check. Release runs are serialized, and actions use Node.js 24. The early check does not override PyPI's final upload validation.

## Build and verify

Use Python 3.11+ (CI covers 3.11–3.13). Install development dependencies and run deterministic tests, then localhost socket/curl integration checks:

```sh
python scripts/check_release.py
python -m pip install '.[dev]'
ruff check src tests examples scripts
python -m pytest -q --cov=prism --cov-report=term --cov-fail-under=80
python -m pytest tests/standalone -q -m integration -o addopts=''
python -m build
python -m twine check dist/mirrorneuron_prism-0.4.0*
```

`pyproject.toml` and `src/prism/__init__.py` share the version; API metadata imports it. The wheel ships only `prism`, bundled local/free/provider presets, the benchmark JSONL suite, `py.typed`, metadata, and the MIT license. The sdist includes source, tests, examples, Docker assets, the changelog, and documentation, including the curated synthetic evaluation evidence. Credentials, caches, virtual environments, and arbitrary benchmark output directories are excluded.

PyPI does not render Mermaid code blocks and does not resolve repository-relative images or links. Keep README URLs absolute and publish the rendered diagram in `docs/assets/prism-flow.png`, alongside its editable `prism-flow.mmd` source. Regenerate it with Mermaid CLI after changing the diagram, and visually check the rendered description and both image URLs before publishing. Twine validates metadata syntax; it does not check image availability or diagram rendering.

Install the exact wheel in a clean environment outside the checkout and exercise all installed presets:

```sh
python -m venv /tmp/prism-release-check
/tmp/prism-release-check/bin/python -m pip install dist/mirrorneuron_prism-0.4.0-py3-none-any.whl
cd /tmp
/tmp/prism-release-check/bin/prism --version
/tmp/prism-release-check/bin/prism init --preset openrouter --out-dir /tmp/prism-free-check
/tmp/prism-release-check/bin/prism validate --profile /tmp/prism-free-check/profiles/prism-balanced.json
/tmp/prism-release-check/bin/prism init --preset providers --out-dir /tmp/prism-providers-check
/tmp/prism-release-check/bin/prism validate --profile /tmp/prism-providers-check/profiles/prism-openai.json
/tmp/prism-release-check/bin/prism benchmark run --help
```

Use new output directories each time; `init` never overwrites. Also unpack the sdist and build its wheel to verify a source installation. A real Laya checkpoint startup and live provider evaluation are separate from fixture checks; the required CPU checkpoint must prepare before readiness. Native cloud samples are not live-qualified.

## Docker verification

```sh
docker build -t mirrorneuron-prism:0.4.0 .
docker run --rm mirrorneuron-prism:0.4.0 --version
docker run --rm -p 127.0.0.1:8080:8080 \
  -e OPENROUTER_API_KEY -e PRISM_API_KEY \
  -v prism-huggingface:/home/prism/.cache/huggingface \
  mirrorneuron-prism:0.4.0
```

Check `GET /health`, `/v1/models` with authentication, and a selected completion. Verify missing default client credentials prevent startup, and the explicit `start --profile ... --no-auth` command works without a Prism key. The image runs as a non-root user and uses CPU torch. Secrets are passed at runtime, never baked into the image.

## Publish manually

After reviewing the release revision and artifacts, choose one publication path:

- Configure PyPI Trusted Publishing for repository `homerquan/mirrorneuron-prism`, workflow `release.yml`, environment `pypi`. Add your desired human approval protections and manually dispatch the workflow on the reviewed revision. The publish job uses short-lived OIDC credentials.
- Authenticate locally with your own publishing credentials, then run `python -m twine upload dist/mirrorneuron_prism-0.4.0*`. This matches only this release's wheel and sdist; it does not upload older artifacts.

Optional TestPyPI publication is also a user-controlled step. Do not embed publishing credentials in source. Build output alone never publishes anything. Update the version for a later release rather than attempting to replace immutable PyPI artifacts.
