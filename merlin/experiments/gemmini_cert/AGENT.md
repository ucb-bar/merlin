# AGENT.md — merlin/experiments/gemmini_cert

Status: frozen — the conformance sweep's results are in `FINDINGS.md`; `docs/guides/gemmini_experiment.md` and `docs/guides/reproducibility.md` cite `run.py`.

Gemmini certification experiment: drives `merlin.targetgen.eval` over the contract corpus and records
findings (`run.py`, `experiment.yaml`, `FINDINGS.md`). Consumes merlin; generated agent output goes to
`agent_generated/` (gitignored) and runs to `runs/`. See `FINDINGS.md`.
