# Contributing to Prism

Thank you for helping improve Prism. Useful contributions include reproducible bug reports, clearer documentation, provider protocol fixtures, routing fixes, and bounded workflow improvements. Keep discussions respectful and focused on the work.

## Development setup

Fork and clone the repository, then create a Python 3.11+ environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
prism --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. The deterministic tests use HTTP and decision fixtures; they do not need provider keys or model-weight downloads. Real server startup prepares the required Laya checkpoint separately.

## Before submitting a change

Keep a pull request focused on one problem. Explain the resulting behavior, update relevant documentation, and add meaningful regression coverage for changed behavior. Configuration examples should use environment variables for credentials.

```sh
ruff check src tests examples
python -m pytest -q --cov=prism --cov-report=term --cov-fail-under=80
python -m pytest tests/standalone -q -m integration -o addopts=''
```

The integration suite starts temporary localhost servers and exercises real HTTP, SDK, and curl requests. Install `curl` to run those checks. CI runs Python 3.11–3.13 and enforces the coverage floor. For packaging changes, also run `python -m build` and `python -m twine check dist/*`; use the current project version for later releases.

For provider or model changes, distinguish local protocol fixtures from live qualification. Report which checks actually ran, retain failed/inconclusive outcomes, and explain any pricing assumptions. Catalog metadata alone is not proof of JSON, image, or reasoning behavior. Live calls can consume upstream quota or incur charges.

## Issues and pull requests

Use the issue templates for bug reports and feature requests. Include the Prism/Python versions, a minimal configuration or request, steps to reproduce, and expected versus observed behavior. Remove credentials and private content from shared examples.

In a pull request, describe the problem, the change, and the validation results. Mention tests that were not run and why. Avoid unrelated formatting or dependency changes. Maintainers may request a smaller scope or additional evidence before merging; release publication remains a separate manual step.

For suspected vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of posting exploit details publicly. Contributions are made under this project's [MIT License](LICENSE).
