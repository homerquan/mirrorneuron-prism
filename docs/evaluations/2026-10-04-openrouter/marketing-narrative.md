# Marketing narrative: spend premium tokens on the answer

Across 9 matched completed pairs in a six-task synthetic pilot, Prism's free Nano preparation followed by Super synthesis reduced the hypothetical premium-model token bill by 56.3%. All-attempt task acceptance was 6/12 for the mix and 9/12 for Super direct. All-attempt mean latency changed from 4.83s to 19.96s. Incomplete or unknown-usage pairs are excluded from the cost estimate, never priced at zero.

The product story: a free model reads the background and prepares the useful context; a larger model spends its work on the final response. Developers keep one OpenAI-compatible endpoint, with explicit stage limits and structured-output routing.

The mechanism in this pilot: premium input tokens fell 49,597 → 2,711 (94.5%), while premium completion/reasoning tokens changed 7,807 → 7,210. Show both counters: input compression alone overstates total cost savings.

Suggested campaign line: **Give the larger model a focused brief. Give your application a smaller token bill.**

Useful demo: show the same code repair or launch email from a long brief, then display premium input tokens, total completion tokens, elapsed time, and the reference checks side by side. Include short microcopy and failed requests so the demo shows where an extra stage adds delay. The pilot's lower acceptance and slower responses make this a workload-selection story, not a launch claim of equivalent quality.

Use the numeric claim only with the pilot qualification and the pricing assumption adjacent to it. The runs used free Nemotron models, not GPT-6 Astra or Claude; the estimate reuses their token rates without establishing frontier-model quality, latency, or actual invoices. Actual free-tier token charges were zero under the configured model prices. Free-tier quotas, failures, local serving costs, and missing usage still matter.

Avoid claiming production-wide savings, identical quality, faster responses, or replacing a frontier model. This report measures a narrow mechanism and identifies workloads for a larger follow-up evaluation.

[Detailed results](benchmark-results.md)
