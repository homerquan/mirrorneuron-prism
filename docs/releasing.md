# Releasing mirrorneuron-prism

The distribution remains `mirrorneuron-prism` to preserve the existing project identity. The standalone public import and command are `prism`. Before a first release, verify ownership/availability of that PyPI project; this repository does not establish registry ownership.

1. Update the version consistently in `pyproject.toml`, `src/prism/__init__.py`, and the API metadata. Document the implemented contract and migration.
2. Run `python -m pip install '.[dev,legacy]'`, `ruff check src/prism tests/standalone`, and `python -m pytest -q`. Real backend and checkpoint evaluation is separate from deterministic offline conformance checks. Measure the intended workload before advertising quality/cost/latency results.
3. Build with `python -m build` and check the artifacts with `python -m twine check dist/*`. Install the wheel in a clean environment, run `prism --version`, `prism init`, and `prism validate`. Check wheel contents for JSON resources, `py.typed`, and licenses.
4. Configure PyPI Trusted Publishing for repository `homerquan/mirrorneuron-prism`, workflow `release.yml`, and environment `pypi`. Protect that GitHub environment with your desired human approval rules and restrict release dispatch to reviewed revisions. The workflow intentionally requires manual dispatch.
5. Run the release workflow on the reviewed release revision. It tests/builds and transfers the artifacts to the publish job. The publish job uses short-lived OIDC credentials through `pypa/gh-action-pypi-publish`; do not put registry secrets in source files.

For a local release instead, use `python -m twine upload dist/mirrorneuron_prism-VERSION*` after authenticating with your own publishing credentials. Build output alone does not publish anything.

Normal installation excludes Torch and LiteLLM. The `laya` extra installs the decision runtime from PyPI. The `legacy` extra adds LiteLLM/PyYAML for the preserved `mn_prism` tooling; the `proxy` and `classifier-jevcpu` extras retain the old integration options. Do not combine the JEV pinned dependency set with Laya casually; validate the resulting Torch/Transformers versions for the selected model.
