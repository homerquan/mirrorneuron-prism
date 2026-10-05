# Free preparation, premium-priced synthesis: a live Prism pilot

Run date: October 4, 2026 (America/New_York). All model calls used OpenRouter `:free` Nemotron models. Actual configured model token cost: $0. The dollar figures below are hypothetical costs obtained by pricing Super's measured tokens like a premium model; Nano is priced at $0.

## Method

Six synthetic tasks, 2 paired repetitions per task, sequential requests with alternating route order. Baseline: Super direct. Candidate: Nano plain-text preparation → Super final synthesis. Temperature 0; output cap 4096 tokens; no hidden retries. Five tasks include roughly 30 KB of synthetic office-history distractors; one is short microcopy. This deliberately examines sparse, long-context tasks and a small-input counterexample. It is a pilot, not a representative production workload or a benchmark of OpenAI/Anthropic model quality.

Coding is graded by functional tests in a restricted, timed interpreter. Copy is checked for required facts, JSON shape, prohibited claims, and length. Support/summaries use exact reference fields. These checks do not measure aesthetic quality, full security, or general reasoning ability. Worker notes are unverified observations: reducing input can lose facts.

## Hypothetical pricing

| Scenario | Input / million | Output / million | Reference |
|---|---:|---:|---|
| astra_standard | $10.00 | $50.00 | [Official pricing](https://developers.openai.com/api/docs/models/gpt-6-astra) |
| claude_opus_5_5 | $4.00 | $20.00 | [Official pricing](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5) |

Rates checked October 4, 2026. Standard uncached rates; no caching, batch discount, tool charges, local compute cost, or long-context premium. Nemotron tokenization, reasoning, quality, and speed are not equivalent to the priced reference models. Provider completion-token totals include reasoning tokens when reported; Prism logical byte counters are never used for these estimates.

## Results

Acceptance and latency include every attempt. Cost comparisons use only matched, completed pairs with known physical usage on both routes; a quality failure remains in the cost comparison if it completed. Failed or unknown-usage requests never count as zero cost. Premium costs below use the Astra scenario and are means per matched request.

| Task | Pass baseline / mix | Mean latency baseline / mix | Cost pairs | Premium cost baseline / mix | Savings |
|---|---:|---:|---:|---:|---:|
| coding-stock-fix | 0/2 / 1/2 | 2.66s / 4.82s | 1/2 | $0.07714 / $0.01361 | 82.4% |
| coding-input-validation | 1/2 / 1/2 | 6.82s / 27.28s | 2/2 | $0.11836 / $0.04890 | 58.7% |
| content-launch-email | 2/2 / 2/2 | 11.16s / 52.69s | 2/2 | $0.15041 / $0.10606 | 29.5% |
| support-refund-policy | 2/2 / 1/2 | 2.29s / 11.68s | 2/2 | $0.07392 / $0.01659 | 77.5% |
| summarization-incident | 2/2 / 0/2 | 1.56s / 7.32s | 1/2 | $0.07041 / $0.01067 | 84.8% |
| content-short-microcopy | 2/2 / 1/2 | 4.46s / 15.97s | 1/2 | $0.05339 / $0.02023 | 62.1% |

Task acceptance: baseline 9/12, mixed 6/12 (-25.0 percentage points). Mean end-to-end latency across all attempts: 4.83s → 19.96s (+313.5%). Matched completed pairs with known usage: 9/12. Scores below are observed checks, not confidence intervals.

| Cost scenario | Baseline paired total | Mixed paired total | Estimated savings | At 100,000 completed requests with the observed matched mix |
|---|---:|---:|---:|---:|
| astra_standard | $0.88632 | $0.38761 | 56.3% | $5,541.22 saved (linear illustration) |
| claude_opus_5_5 | $0.35453 | $0.15504 | 56.3% | $2,216.49 saved (linear illustration) |

| Matched premium-model work | Super direct | Nano → Super |
|---|---:|---:|
| Input tokens (same Super tokenizer) | 49,597 | 2,711 |
| Completion tokens, including reported reasoning | 7,807 | 7,210 |
| Physical calls, including the free worker | 9 | 18 |

Premium input fell 94.5% on the matched completed requests. The smaller reduction in total estimated cost reflects completion/reasoning work, priced at a higher rate. Nano's work adds a second call; free token pricing does not eliminate latency or compute. Different models' token counts are not combined into a purported universal token total.

Unknown-usage requests: 3. The paired cost estimate excludes 3 incomplete/unknown pairs and does not estimate a whole-run bill. Excluding failed pairs changes the task mix; the 100,000-request projection applies only to the observed completed mix and is not a production forecast. Latency includes upstream load, reasoning, networking, and Prism overhead. An aggregate can conceal a slower route or a quality regression on an individual task.

JSON-object mode guarantees neither a requested field layout nor its types. Some completed responses failed coding tests or returned mismatched JSON fields/types. Applications requiring a shape should request JSON Schema and retain task-level checks. Upstream failures are included as failed task attempts; no retries conceal them.

## Reproduction and evidence

Start `prism serve --config prism-openrouter.json --no-auth`, then run:

```sh
python examples/standalone/openrouter_evaluation.py --no-auth --repeats 2 --out-dir docs/evaluations/NEW-RUN
```

`cases.jsonl` contains prompts and references; `requests.jsonl` retains every response, sanitized trace, grader result, physical token count and latency. `manifest.json` records versions, hashes, pricing assumptions and settings. No provider credentials are saved.
