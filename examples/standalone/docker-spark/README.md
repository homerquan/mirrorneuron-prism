# Local/Spark Gemma4 and Spark Nemotron live benchmark

The runner connects local Gemma4, Spark Gemma4, and Spark Nemotron 3.5 Lightning through a localhost SSH tunnel. It does not change the root deployment configuration. Check `docker model list` locally and `ssh spark 'docker model list'` before use; adjust model names to your installed models.

## Repeatable runner

From the repository root, use the installed Prism environment:

```sh
.venv/bin/python examples/standalone/docker-spark/benchmark.py --spark spark --repeats 3
```

The runner discovers installed models, validates actual JSON inference readiness, reads explicit Docker runtime context configuration, opens its own SSH tunnel, chooses temporary local ports, generates the fixtures and isolated configuration, starts Prism and resident Laya, executes the suites sequentially, writes reports, and stops only its own server and tunnel. It does not pull models or change the deployment files. No API key setup is needed: a random request credential is passed only to the temporary server process.

For a quick direct comparison:

```sh
.venv/bin/python examples/standalone/docker-spark/benchmark.py \
  --suites direct --repeats 1 --warmup 0
```

For automatic divide-and-conquer qualification:

```sh
.venv/bin/python examples/standalone/docker-spark/benchmark.py \
  --suites large-gemma,large-nemotron --repeats 3
```

The runner includes direct baselines, automatic routing, and an explicit policy matrix. Default: all suites, one measured repetition, one warmup pair for the two short source suites. Large/review suites have no warmup. Review measures the first two cases by default; `--review-max-cases 6` runs all six. `--max-cases` limits any suite. The large fixture places the same policy facts at the start, middle, and end of two 6KB documents and verifies the final fields independently. The repeated irrelevant records expose extraction recall and completeness errors.

`--output-tokens` changes the public budget (maximum 8192); `--nemotron-worker-tokens` and `--gemma-worker-tokens` change intermediate reserves (both default to 1024). `--spark-gemma-model` selects the remote Gemma endpoint. `--reduction structured|rolling|none` selects structured state/tree/lookup (default), legacy rolling memory, or no reduction. The remaining context and intermediate settings are in the adjacent JSON template. `--local-model` and `--remote-model` accept exact endpoint IDs if automatic discovery is ambiguous. Use `--local-url` or `--remote-port` for nonstandard Docker endpoints. `--out-dir` must name a new directory. Every run retains partial artifacts on interruption and writes `report.md`, `analysis.json`, raw responses, per-suite summaries/manifests, generated cases, discovery metadata, and service logs.

For actual dollar estimates, `--metadata prices.json` accepts an object keyed by `gemma`, `spark-gemma`, and `nemotron`; each value may supply `input_cost_per_million`, `output_cost_per_million`, and optional `power_rating`. No rates are invented. Metadata is copied into the saved registry; cost/power optimization is not enabled by this benchmark template. If prices are absent, the report uses physical token consumption, including failed calls, as the cost proxy.

Quality failures do not interrupt a benchmark or make its process fail: completion of scheduled attempts and quality are separate. Inspect pass rates, errors, coverage, and routing decisions in the report. This template disables optional thinking to avoid spending the intermediate budget before producing usable artifacts. Caller reasoning controls still take precedence on direct requests. The validation runs below contain failures and partial recall; these scripts do not promise that splitting improves quality or speed.

## Manual service and benchmark commands

```sh
ssh -N -L 127.0.0.1:12435:127.0.0.1:12434 spark
```

In another terminal, set `PRISM_API_KEY` to a secret of your choice, then start:

```sh
prism serve --config examples/standalone/docker-spark/prism.json
```

Use the same key in the benchmark terminal:

```sh
prism benchmark run --config examples/standalone/docker-spark/prism.json \
  --base-url http://127.0.0.1:18080/v1 \
  --baseline gemma-direct --candidate nemotron-direct \
  --output-tokens 1024 --repeats 1 --warmup 1

prism benchmark run --config examples/standalone/docker-spark/prism.json \
  --base-url http://127.0.0.1:18080/v1 \
  --baseline nemotron-direct --candidate mixed-evidence \
  --output-tokens 1024 --repeats 1 --warmup 1
```

`mixed-review` is a Gemma draft → Nemotron critique → Nemotron synthesis profile, requiring source-free requests. `mixed-auto-small` uses Gemma extraction and Nemotron synthesis; `reverse-auto-small` in generated runs uses Nemotron for extraction and synthesis. Hard review/synthesis stages use Nemotron; helper reduction uses local Gemma. Spark Gemma has its own direct, automatic, and fixed-policy suites. Both can select direct, evidence mapping, batching, or verification automatically and partition explicitly bounded sources without dropping bytes. Generated runs cap Nemotron at Spark's configured 8192 runtime context. Gemma full aliases use conservative 32768 admission and small aliases use 8192. Model metadata maximum context is not the running server's slot limit. Runtime configuration and readiness snapshots are saved with each run.

The 2026-10-02 large-request fixture and complete run artifacts are under `benchmark-results/` in the testing workspace; see `docs/live-benchmark-20261002.md`. Supply your own JSONL cases with `<prism-source>` blocks using `--cases` to repeat the large-request test. Inspect decision disposition, selected graph, required/validated coverage, stage timing, and final reference quality separately. Successful coverage validation does not guarantee semantic recall. Do not treat a direct admission rejection's very short latency as successful generation speed.

Prices and power ratings are intentionally absent: this live test reports token consumption as the cost proxy. Automatic policy routing is active; cost/power model assignment requires operator metadata and an optimization allowlist. See the separate optimization example for that interface. Explicit zero API prices would still exclude hardware and electricity.

## Every execution policy

Force each policy independently so automatic routing cannot funnel every experiment into batching:

```sh
.venv/bin/python examples/standalone/docker-spark/benchmark.py \
  --suites policies --repeats 3
```

For a short qualification of each policy:

```sh
.venv/bin/python examples/standalone/docker-spark/benchmark.py \
  --suites policies --max-cases 1 --repeats 1
```

| Policy | Gemma-side suite | Nemotron-side suite | Expected graph / coverage |
|---|---|---|---|
| `evidence_map` | `large-evidence-gemma` | `large-evidence-nemotron` | One extraction per partition → synthesis; full coverage |
| `batched_map` | `large-batched-gemma` | `large-batched-nemotron` | Feasible groups of partitions → synthesis; full coverage |
| `verified_map` | `large-verified-gemma` | `large-verified-nemotron` | Extraction → verification per partition → synthesis; full coverage |
| `retrieve_read` | `large-retrieve-gemma` | `large-retrieve-nemotron` | Focused original spans → one final call; focused coverage |
| `draft_review` | `review` | Same mixed suite | Gemma draft → Nemotron JSON critique → Nemotron synthesis; source-free |

For the three map policies, Gemma-side means Gemma extraction with Nemotron synthesis; Nemotron-side means Nemotron extraction with Nemotron synthesis. Spark Gemma extraction also uses Nemotron synthesis. Verification uses Nemotron in both directions. For retrieval, the suite suffix identifies the final model; retrieval has no extraction or verifier calls. It uses a separate large focused-lookup fixture with independently checked numeric answers and start/middle/end placements. Its successful reference score does not establish exhaustive source coverage.

`--suites all` includes the fixed-policy matrix for all three backend families plus direct and automatic comparisons. `--suites policies` retains the original two-family matrix; `--suites reliability` covers all three families with direct, auto, review, and fixed policies. Automatic full-coverage profiles now allow verification as well as evidence mapping and batching. Retrieval is tested with focused permissions separately; draft/review uses the source-free fixture. Fixed policy profiles allow only their requested policy and cannot silently fall back to batching.

The report and `analysis.json` include a policy audit: expected policy, actual selection mismatches, valid planned operator graphs, completed graphs, unexpected completed graphs, and missing traces. Per-case tables show full/focused coverage and verified/required partitions. A failure before verification or synthesis is visible as an incomplete execution; it does not count as testing a successfully completed downstream stage. Verification profiles have larger graph output budgets to admit both extraction and verification reserves. `--gemma-worker-tokens` (default 1024, maximum 4096) and `--nemotron-worker-tokens` (default 1024) control the respective large-test intermediate budgets.

## Structured reduction and repeatable comparison

The default runner now enables bounded `ReasoningState`, a recursive reduce tree, and up to two internal evidence lookup rounds before Nemotron synthesis. Every map observation retains an original evidence ID. Reducers receive observations and pointers; quotes live in request-local temporary Markdown files. Those files are removed after lookup, including on cancellation. No database or persistent storage is needed. See [configuration and limits](../../../docs/structured-reduction.md).

```sh
.venv/bin/python examples/standalone/docker-spark/benchmark.py \
  --suites reliability --reduction structured --max-cases 1 --repeats 1 --warmup 0

.venv/bin/python examples/standalone/docker-spark/benchmark.py \
  --suites large-gemma,large-verified-spark-gemma,large-batched-gemma \
  --reduction rolling --repeats 3 --warmup 0
```

Use identical suites/cases/budgets for structured versus rolling comparisons. Reports include completion and completed-answer quality separately, recovery attempts, invalid artifacts, reduce/lookup timing and tokens, tree levels, maximum state sizes, evidence retrieval and cleanup. The default structured fixture uses a 512-byte conservative state cap to qualify the 8K setup; operators can raise `state_max_tokens` up to 4096 when stage contexts permit. Strict context and graph budgets still bound the total supported workload; this is not an unlimited-call guarantee.

## Native context qualification

The original `large_cases()` fixtures test byte-bound admission and partitioning. The separate `native_context.py` runner generates seeded cases at actual native prompt lengths using the installed llama.cpp `/apply-template` and `/tokenize` endpoints. It calibrates two JSON-mode prompts against provider `prompt_tokens` and checks the declared runtime context against native `/props`; mismatched or unavailable tokenization stops qualification instead of substituting a byte estimate. It uses disabled thinking and text-only JSON output, without tools or multimodal templates.

Use the actual runtime slot size from `docker model configure show`, rather than the model's advertised maximum. If Docker does not expose tokenizer HTTP routes, its installed llama.cpp server can be accessed through its Unix socket. Verify the socket belongs to the model being benchmarked; the runner does not switch models, pull weights, change runtime contexts, or restart deployments.

For an already-running Spark Nemotron model with an 8K slot, open a temporary loopback tunnel in another terminal:

```sh
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:57999:127.0.0.1:12434 spark
```

Then run a six-family 16K qualification:

```sh
.venv/bin/python examples/standalone/docker-spark/native_context.py \
  --base-url http://127.0.0.1:57999/engines/v1 \
  --model docker.io/ai/nemotron-3.5-lightning:latest \
  --context-window 8192 \
  --tokenizer-command 'ssh -o BatchMode=yes spark docker exec -i docker-model-runner curl --unix-socket /app/inference-runner-0.sock' \
  --sizes 16384 --seeds 11 --positions end --native-direct
```

Use `--tokenizer-url` instead when the model's native llama.cpp HTTP server is directly accessible. Socket paths are deployment-specific. The command transport accepts an argv prefix and passes request JSON through stdin, without putting source text into command arguments. Cancel the owned tunnel afterward.

For broader qualification, use `--sizes 4096 16384 32768 65536 --seeds 11 29 47 --positions start middle end --repeats 3`. This can require substantial inference time. Task families cover keyword distractors, distributed facts, exceptions, reference chains, effective policy revisions, and unspecified answers. References and complete gold byte spans stay outside model packets. Split seeds for development and held-out qualification; repetitions of a case are not independent examples.

Every Prism stage uses the same physical model and conservative admission. The runner calls the real execution engine and HTTP backend in process; it does not measure proxy startup or learned routing. `--native-direct` additionally sends unchanged requests to the backend, detects actual context errors and possible silent prompt truncation, and grades fitting direct answers. These controls and calibration usage are saved separately from Prism-stage work.

Adaptive final synthesis enforces `json_object` with an object-only schema when the backend supports JSON Schema; it does not supply field names or answer values. Native direct controls preserve the original JSON-mode request. This output-contract repair is part of the candidate implementation, so fitting-window quality differences cannot be attributed solely to partitioning. Oversized native direct rejection is the control for the runtime-window claim.

Artifacts include original cases, target/measured native lengths, gold source spans, responses/traces, physical work, per-family accuracy and P50/P95 latency, and source/runner fingerprints. Span recall separately measures gold evidence present in mapped quotes, lookup excerpts, and final state references/excerpts. It does not establish semantic entailment or exhaustive mapper recall. Failed jobs score zero. Reports distinguish actual backend rejection from Prism's conservative byte admission. There is no automatic claim of unlimited context or a qualified length based solely on completion.

The [2026-10-03 validation record](../../../docs/native-context-validation-20261003.md) records the measured results and limitations of this implementation.
