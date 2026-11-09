# AGENT.md — experiments/studies/gemmini_cert

Status: frozen

Archive only. `FINDINGS.md` records historical conformance observations; do not rewrite it to
describe current code or treat it as qualification of a relocated implementation.

Executable owners, the example configuration and tests now live in the Gemmini support provider
vendored at `examples/gemmini/support/` (`gemmini_conformance/`, `examples/conformance/` and
`tests/`). With `MERLIN_TARGET_PATH` unset it is the selected support; no executable compatibility
wrapper remains here. Consult its README for missing-facts/schema limitations and test prerequisites.
