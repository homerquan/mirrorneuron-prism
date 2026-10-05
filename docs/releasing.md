# Releasing mirrorneuron-prism

Version **0.3.0** is prepared for manual publication. Distribution: `mirrorneuron-prism`; import and CLI: `prism`. This preparation does not create a tag, upload to a registry, or dispatch the publishing workflow. Verify that you own the PyPI project before publication.

## Build and verify

Use Python 3.11+ (CI covers 3.11–3.13). Install development dependencies and run deterministic tests, then localhost socket/curl integration checks:

```sh
python -m pip install '.[dev]'
ruff check src tests examples
python -m pytest -q --cov=prism --cov-report=term --cov-fail-under=80
python -m pytest tests/standalone -q -m integration -o addopts=''
python -m build
python -m twine check dist/mirrorneuron_prism-0.3.0*
```

`pyproject.toml` and `src/prism/__init__.py` share the version; API metadata imports it. The wheel ships only `prism`, bundled local/free/provider presets, the benchmark JSONL suite, `py.typed`, metadata, and the MIT license. The sdist includes source, tests, examples, Docker assets, the changelog, and documentation, including the curated synthetic evaluation evidence. Credentials, caches, virtual environments, and arbitrary benchmark output directories are excluded.

Install the exact wheel in a clean environment outside the checkout and exercise all installed presets:

```sh
python -m venv /tmp/prism-release-check
/tmp/prism-release-check/bin/python -m pip install dist/mirrorneuron_prism-0.3.0-py3-none-any.whl
cd /tmp
/tmp/prism-release-check/bin/prism --version
/tmp/prism-release-check/bin/prism init --preset openrouter --out-dir /tmp/prism-free-check
/tmp/prism-release-check/bin/prism validate --config /tmp/prism-free-check/prism.json
/tmp/prism-release-check/bin/prism init --preset providers --out-dir /tmp/prism-providers-check
/tmp/prism-release-check/bin/prism validate --config /tmp/prism-providers-check/prism.json
/tmp/prism-release-check/bin/prism benchmark run --help
```

Use new output directories each time; `init` never overwrites. Also unpack the sdist and build its wheel to verify a source installation. A real Laya checkpoint startup and live provider evaluation are separate from fixture checks; the required CPU checkpoint must prepare before readiness. Native cloud samples are not live-qualified.

## Docker verification

```sh
docker build -t mirrorneuron-prism:0.3.0 .
docker run --rm mirrorneuron-prism:0.3.0 --version
docker run --rm -p 127.0.0.1:8080:8080 \
  -e OPENROUTER_API_KEY -e PRISM_API_KEY \
  -v prism-huggingface:/home/prism/.cache/huggingface \
  mirrorneuron-prism:0.3.0
```

Check `GET /health`, `/v1/models` with authentication, and a selected completion. Verify missing default client credentials prevent startup, and the explicit `serve ... --no-auth` command works without a Prism key. The image runs as a non-root user and uses CPU torch. Secrets are passed at runtime, never baked into the image.

## Publish manually

After reviewing the release revision and artifacts, choose one publication path:

- Configure PyPI Trusted Publishing for repository `homerquan/mirrorneuron-prism`, workflow `release.yml`, environment `pypi`. Add your desired human approval protections and manually dispatch the workflow on the reviewed revision. The publish job uses short-lived OIDC credentials.
- Authenticate locally with your own publishing credentials, then run `python -m twine upload dist/mirrorneuron_prism-0.3.0*`. This matches only this release's wheel and sdist; it does not upload older artifacts.

Optional TestPyPI publication is also a user-controlled step. Do not embed publishing credentials in source. Build output alone never publishes anything. Update the version for a later release rather than attempting to replace immutable PyPI artifacts.
