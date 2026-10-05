# Security policy

## Reporting a vulnerability

If the repository has GitHub private vulnerability reporting enabled, use **Security → Report a vulnerability** on [the repository's Security page](https://github.com/homerquan/mirrorneuron-prism/security). Otherwise, contact a repository maintainer through their GitHub profile to arrange a private reporting channel. If no private contact method is available, open an issue requesting one without disclosing the vulnerability or exploit.

Include the affected version, a minimal reproduction, the expected security boundary, and the observed impact. Do not include real API keys or private prompts. Public bug reports are appropriate for ordinary functional problems; keep vulnerability details private until maintainers have assessed them and arranged disclosure.

## Project status and scope

Prism is alpha software. Security fixes are prioritized for the current development revision; no long-term support window or response-time guarantee is established. Relevant boundaries include client authentication, credential isolation, trace access, endpoint admission, request/resource limits, and handling of untrusted model artifacts.

Client authentication is enabled by default. Explicit `--no-auth` serving shares anonymous trace access and is intended for trusted environments. Upstream credentials remain independent. See the [usage guide](docs/usage.md) and [implemented contract](docs/standalone-contract.md) for deployment behavior and limitations.
