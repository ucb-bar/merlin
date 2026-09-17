# CI workflows

| Workflow | Trigger | What it gates |
|---|---|---|
| [`pr-fast.yml`](pr-fast.yml) | push / PR | Fast gate: structure + artifact-layout + docs anti-drift + a fast test subset (blocking); ruff lint/format (advisory). |
| [`docs.yml`](docs.yml) | push (docs/schemas/pkg) / PR | Documentation anti-drift: stale generated docs, invalid front-matter, retired-path references. |

Heavier test buckets (`rvv/`, `gemmini/`, `runtime/`) need hardware or RTL sims (spike, verilator,
FireSim, boards) and run out-of-band, not in these hosted-runner workflows.

## Gates that deliberately run nowhere

A gate nobody runs is worse than no gate, because it reads as coverage. Measured 2026-09-16, 15 of
the 34 `build_tools/scripts/check_*.py` ran in no hook, no Stop-hook and no workflow — among them
`check_no_holdout_names`, which was *failing*: a tracked ratchet named two held-out capsules
outright, and a holdout's name is an answer key. Seven of the fifteen now run in `pr-fast.yml`.

The rest stay out, each for a stated reason rather than by omission:

| Script | Why it is not wired |
|---|---|
| `check_standalone_install.py` | Builds and installs a wheel in a clean venv; > 4 min. Out-of-band. |
| `check_claim_set_disjointness.py` | Walks every run directory; > 4 min on a populated `out/`. Out-of-band. |
| `check_conformance_coverage.py` | Same — reads the whole capsule corpus; > 4 min. Out-of-band. |
| `check_isa_matches_rtl.py` | 171 s over all targets, and needs RTL fact bundles a hosted runner does not have. Currently FAILS (`saturn_opu_rvv` compares nothing), which is real debt, not a wiring gap. |
| `check_inert_capabilities.py` | ~60 s and currently reports debt against its own ratchet; wire it once that ledger is honest. |
| `check_generated_target.py` | Takes a path to one generated target repo — a per-target tool, not a repo-wide property. |
| `check_repro_env.py` | A preflight that reports which repro capabilities THIS machine has. Its answer is environment-dependent by design, so failing CI on it would be meaningless. |
| `check_cert_affordability.py` | Wired, but advisory: ~60 s, and it prices a cohort rather than asserting a property of the diff under review. |

Adding a `check_*.py` means adding it here too — as a row in this table or as a step in a workflow.
