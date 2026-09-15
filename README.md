# mirrorneuron-prism

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-3388FF.svg)](https://github.com/astral-tools/ruff)

Bounded multi-call inference as a transparent LiteLLM Proxy extension.  
Transform any OpenAI-compatible client call into several bounded LLM calls, select or fuse the results, and return one ordinary completion — with zero client changes.

## Features

- **Transparent** – drop-in OpenAI compatible proxy, no SDK changes
- **Best-of-N / Adaptive / Consensus** strategies
- **Budgeting & deadlines** – token, call and time limits per logical request
- **LLM Judge & Verifiers** – pluggable selection with safety checks
- **Metering & cost estimation** – per-model usage tracking offline
- **Zero config** – just add a custom provider and a policy file

## Quick Start

```bash
# install
pip install "mirrorneuron-prism[proxy]"

# validate
mn_prism validate --policy-config multicall.yaml --litellm-config litellm.yaml
```

Client code stays unchanged:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:4000/v1", api_key="sk-test")
resp = client.chat.completions.create(
    model="smart-local",
    messages=[{"role":"user","content":"Explain DAGs"}]
)
print(resp.choices[0].message.content)
```

## Installation

```bash
pip install "mirrorneuron-prism[proxy]"
```

Development:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[proxy,dev]"
```

## Project Layout

```
src/litellm_multicall/
  provider.py  hooks.py  config.py  types.py  engine.py  backend.py
  budgets.py   selection.py  validation.py  streaming.py  telemetry.py
  compat/      resources/
```

## Docs

- [SPEC.md](SPEC.md) – full specification
- [HOW_TO_USE.md](HOW_TO_USE.md) – operator guide
- `examples/` – sample LiteLLM and MultiCall configs

## License

MIT – see [LICENSE](LICENSE)
