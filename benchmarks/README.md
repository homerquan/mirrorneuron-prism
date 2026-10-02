# Native CPU measurement

Use the installed `mn_prism benchmark` commands, documented in [HOW_TO_USE.md](../HOW_TO_USE.md). Implemented suites are `semif-authored` (model-reviewed synthetic labels) and `semif-shape` (systems fixture without labels).

Preparation reads a pinned local JEV-CPU Git revision. Execution uses immutable local plans/artifacts and the supervised CPU adapter; it does not invoke upstream GPU runners. Gold labels and annotation prose are excluded from scorer inputs.

Rules, direct logits, shared-prefix logits, and constrained one-token generation are operational treatments. Reports preserve denominators and missing measurements. Full BFCL, τ, SWE-bench, hybrid execution, sweeps, and resume are future milestones.
