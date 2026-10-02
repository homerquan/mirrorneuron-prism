# CPU decisions and reproducible native experiments

Prism preserves Python 3.11, `mn_prism`, and its LiteLLM provider/hook import paths. This guide covers implemented P0–P2 functionality. Full generative inference and application-tool execution are not available in Prism.

## Installation

The native Apple Silicon CPU path was tested with Python 3.11.16, Torch 2.10.0, Transformers 5.17.0, and float32. Use an isolated environment:

```bash
python -m pip install -e '.[dev,classifier-jevcpu]' -c constraints/macos-arm64-py311-classifier.txt
```

For Linux, deliberately install CPU Torch wheels first, then the extra. This installation recipe is a candidate configuration, not a claim of Linux platform verification:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev,classifier-jevcpu]' -c constraints/linux-cpu-py311-candidate.txt
```

Apple Silicon inference is explicitly CPU-only, including buffers. Neither CUDA nor MPS is auto-selected. Loading takes separate startup time and several GB of RAM; classification still performs tokenization and model prefill.

## Create and validate a project

```bash
mn_prism init --template cpu-bench --out-dir ./cpu-experiment
mn_prism --config ./cpu-experiment/prism.yaml config validate
mn_prism --config ./cpu-experiment/prism.yaml config show --format json
mn_prism --config ./cpu-experiment/prism.yaml doctor --format json
mn_prism config schema project --format json
```

The template creates strict `prism.yaml`, `multicall.yaml`, `litellm.yaml`, a decision request, and a descriptive experiment YAML. The experiment YAML documents treatment choices; there is no sweep executor yet.

Project selection is `--config`, then `PRISM_CONFIG`, then `prism.yaml` in the invocation directory. Workdir precedence is `--workdir`, `PRISM_WORKDIR`, configured workdir, then the invocation directory. Explicit paths are relative to the caller; paths inside project YAML are relative to that file. Prism never changes the process working directory.

An optional personal shell setting can select your preferred workspace:

```bash
export PRISM_WORKDIR=/Users/homer/mirrorneuron-prism
```

Use your own path. Package source and tests do not require that location. Runtime datasets, plans, and runs belong beneath the selected workdir's `.prism/`; the artifact lock path is resolved from the project file.

Diagnostics distinguish valid configuration, dependency presence, prepared artifacts, and unavailable generative runtime. An offline check is not proof that a model actually loaded; use `models verify quick --load` for an explicit load check.

## Prepare model artifacts explicitly

Only this command may resolve/download remote weights:

```bash
mn_prism --config ./cpu-experiment/prism.yaml models fetch quick --resolve-revision
mn_prism --config ./cpu-experiment/prism.yaml models verify quick
```

It records an immutable model/tokenizer commit, file sizes and SHA-256 hashes, and the model-license reference. Review the referenced model terms separately. The JEV source commit is not the weight commit.

An existing complete local snapshot may be registered without network access:

```bash
mn_prism --config ./cpu-experiment/prism.yaml models fetch quick \
  --snapshot /path/to/prepared/snapshot --revision IMMUTABLE_40_HEX_MODEL_COMMIT --offline
```

The path and commit above are placeholders. A lock reference is immutable: changing its files/revision requires a new `artifact_ref`. Runtime verifies the snapshot, uses safe tensor weights, refuses incomplete loading diagnostics, and sets `local_files_only=True` and `trust_remote_code=False`. Missing weights never cause an implicit download.

## Score decisions

```bash
mn_prism --config ./cpu-experiment/prism.yaml classify run \
  --controller quick --input ./cpu-experiment/examples/decision.json --offline --format json

mn_prism --config ./cpu-experiment/prism.yaml classify run \
  --controller rules --input ./cpu-experiment/examples/decision.json --format json
```

Requests require nonempty finite JSON state, a state version, unique criterion/option IDs, and 2–16 options. The six primitives are `route`, `relevance`, `sufficient`, `retry`, `escalate`, and `stop`. The full prompt is token-limited, with no silent truncation.

Raw conditional option scores return `disposition: scored`, `confidence: null`, and no accepted action. A high option score is not a correctness guarantee. The rules baseline handles one explicit structured read-retry case and abstains on unstructured native fixtures; its low coverage is reported.

Batch input is JSONL. One worker/model is loaded per invocation:

```bash
mn_prism --config ./cpu-experiment/prism.yaml classify batch \
  --controller quick --input ./requests.jsonl --output ./results.jsonl \
  --continue-on-error --offline --format jsonl
```

`requests.jsonl` must contain structured requests. Corrupt rows produce explicit error records; processing stops unless `--continue-on-error` is given. A final summary reports counts. JSON output mode requires an output file for per-row JSONL. Output files are create-only.

There is one in-flight batch, a bounded queue, configured CPU threads, separate startup/inference limits, and a bounded restart allowance. Queue wait is included in delivered latency and `--timeout` deadlines. Hard timeout/cancellation terminates and reaps the worker. Services can turn unavailable results into typed abstentions through `DecisionPolicy.unavailable`; standalone commands surface missing preparation/dependency failures with nonzero status.

## Prepare native fixtures and plan an experiment

Use your local JEV-CPU checkout; no benchmark download flag or external harness is implemented yet:

```bash
mn_prism --config ./cpu-experiment/prism.yaml benchmark prepare semif-authored \
  --source /path/to/JEV-CPU --revision b49b5bf5776af78495fa4900996042726f8e7c10 --offline

mn_prism --config ./cpu-experiment/prism.yaml benchmark plan semif-authored \
  --treatment cpu_direct --selection smoke --split test --repetitions 1 \
  --out ./cpu-experiment/smoke-plan.json

mn_prism --config ./cpu-experiment/prism.yaml benchmark run \
  --plan ./cpu-experiment/smoke-plan.json --offline --format json
```

Replace the source path with your checkout. Preparation reads the exact Git object, not dirty working-tree data. It freezes source/file hashes, rights/notices, all task IDs and upstream split metadata. Scorer inputs exclude labels, rationales, target distributions, and provenance. Gold option IDs stay in an evaluator-only file and never enter classifier IPC.

Authored source groups are divided into three named, derived Prism splits: calibration, validation, and test. Related variants stay together. Smoke selects two complete groups; full selects every group in the named split. `--split all` is explicitly diagnostic. Authored annotations are model-reviewed synthetic examples, not human-adjudicated production ground truth.

Implemented treatments: `rules`, `cpu_direct`, `cpu_shared`, and `cpu_compact_generation`. The compact baseline uses the same native prompt and constrains generation to one declared answer-slot token. It separates the generation/readout path without demanding a long explanation.

For shape timing, prepare `semif-shape` in the same way and plan `--selection smoke`. This fixture has no gold labels. Its smoke selection exceeds the default 1024-token limit: deliberately raise the controller limit to 2048 in YAML before planning, or retain the limit and inspect the recorded failures. Do not increase limits silently. Shared batches contain exact matching state and have bounded criteria/padding. Batch accounting records prefix, suffix, padding, replication duration, and two physical forwards; it never charges two forwards for every criterion.

Plans freeze code, project, dataset, model, task selection, trial policy, resource metric, and a quality margin. A changed identity requires a new plan. Warmup is explicit and included in resource records, with load CPU time and startup wall time separate. Runs use private atomic files, per-run locks, and flushed JSONL evidence. Failed or cancelled tasks remain in the planned denominator; unknown usage stays unknown.

## Calibration

First create full `cpu_direct` plans for `--split calibration` and `--split validation`, then run both. Use their returned IDs:

```bash
mn_prism --config ./cpu-experiment/prism.yaml benchmark calibration-input \
  CALIBRATION_RUN_ID VALIDATION_RUN_ID --output ./cpu-experiment/calibration-input.json

mn_prism classify calibrate --input ./cpu-experiment/calibration-input.json \
  --output ./cpu-experiment/calibration.json --risk-target 0.05 --min-support 20

mn_prism --config ./cpu-experiment/prism.yaml benchmark plan semif-authored \
  --treatment cpu_direct --selection full --split test \
  --calibration ./cpu-experiment/calibration.json --out ./cpu-experiment/gated-test-plan.json
```

Replace run IDs with actual results. The evaluator-side export uses only first-trial logits and distinct source groups; test predictions are excluded. Temperature fitting uses calibration predictions. Threshold selection uses validation predictions with a conservative grouped uncertainty bound and minimum independent support.

This tiny fixture has only 12 groups per derived split. The default 20-group/5% risk requirement can legitimately produce `threshold: null`, so every gate abstains. Do not lower the target after seeing test errors and call that a preregistered result. Artifacts bind model/tokenizer hashes, library versions, prompt, dtype, scoring mode, state-builder version, task families, primitives, and option counts. The native benchmark supplies its declared family binding; an unbound standalone request abstains at a calibrated gate. A change invalidates calibration.

## Inspect and compare runs

```bash
mn_prism --config ./cpu-experiment/prism.yaml runs list --format json
mn_prism --config ./cpu-experiment/prism.yaml runs show RUN_ID --format json
mn_prism --config ./cpu-experiment/prism.yaml runs events RUN_ID --format jsonl
mn_prism --config ./cpu-experiment/prism.yaml benchmark report RUN_ID --output ./report.md
mn_prism --config ./cpu-experiment/prism.yaml benchmark compare BASELINE_RUN_ID CANDIDATE_RUN_ID \
  --quality-margin 0.02 --format json
mn_prism --config ./cpu-experiment/prism.yaml runs export RUN_ID --output ./run-metadata.json
```

Comparisons require the same tasks, groups, evaluator/data, code, environment, and controlled CPU settings. A different margin or `--exploratory` comparison cannot claim qualified parity. Paired source-group uncertainty stays nonzero for tiny all-success samples. `--require-parity` exits 8 when the declared criterion is not established.

Reports distinguish native prediction accuracy, accepted coverage, selective risk, queue/startup/forward/IPC timing, and CPU work. Controller CPU seconds are scoring/warmup process time; model-load CPU time is a separate metric. Summed durations are not end-to-end wall time. Run wall time covers worker execution/startup/coordination after preflight; CLI preparation, artifact hashing and manifest construction were not timed. Those overheads require further measurement before whole-system claims. No GPU active time, energy, pricing, worker savings, or agent success is fabricated. Export contains redacted manifest/settings/metrics, not weights, datasets, or private traces.

Run resume, sweeps, config migration, BFCL projections, official BFCL, τ, SWE-bench, context filtering, hybrid fallback, and CPU/GPU overlap remain unimplemented. Planned suites appear honestly in `benchmark list`; preparing them exits 3. No background execution is scheduled.

## CLI status codes

0 means command execution completed, even if accuracy is poor or every gate abstains. Codes 2/3/4/5/6/7/8 mean invalid input, unavailable capability/artifact, missing optional dependency, external failure, deadline/budget exhaustion, corrupt/incompatible evidence, and an explicitly failed quality gate. Code 1 is an internal error and 130 is interruption. LiteLLM child exits are propagated; negative signal exits map to `128 + signal`.

`--offline` forbids remote preparation and public inference endpoints. Explicit local/private HTTP endpoints are still possible; it is not an OS-enforced air gap or proof about a proxy's downstream locality.
