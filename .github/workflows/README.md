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
| ~~`check_claim_set_disjointness.py`~~ | **Wired 2026-09-22** into `pr-fast.yml` with `--fail-on-circular`. The ">4 min on a populated `out/`" reason was measured locally; a CI runner clones fresh with no `out/`, and the full six-target walk times at 2m30s here. |
| `check_conformance_coverage.py` | **Partly wired since 2026-09-22.** The `--write` / re-derivation path reads every capture and rebuilds the requirement (> 4 min), and stays out-of-band. Comparing the corpus against the TRACKED spec (`--spec … --ratchet … --fail-on-uncovered`) skips all of that — measured **4.8 s** for gemmini — and runs in `pr-fast.yml`, because what a diff can break is the corpus, not the requirement. |
| `check_isa_matches_rtl.py` | 171 s over all targets, and needs RTL fact bundles a hosted runner does not have. Currently FAILS (`saturn_opu_rvv` compares nothing), which is real debt, not a wiring gap. |
| `check_inert_capabilities.py` | ~60 s and currently reports debt against its own ratchet; wire it once that ledger is honest. |
| `check_generated_target.py` | Takes a path to one generated target repo — a per-target tool, not a repo-wide property. |
| `check_repro_env.py` | A preflight that reports which repro capabilities THIS machine has. Its answer is environment-dependent by design, so failing CI on it would be meaningless. |
| `check_cert_affordability.py` | Wired, but advisory: ~60 s, and it prices a cohort rather than asserting a property of the diff under review. |
| `report_generalization_claim.py` | Not a `check_*`, but it is the script that produces the generalization number, so its absence belongs here too. Its denominator is the target's DECLARED roster and its evidence is `out/artifacts/recaptures/` — 130 GB a hosted runner does not have. With zero captures present it would report 0/0 on every PR, which reads as coverage while measuring nothing; `--require-roster` would fail every PR for the same reason. Run out-of-band. Measured 2026-09-22 for gemmini, `--require-roster` clean: across 9 captures of the declared roster (`resnet50`, `tiny_llama`), 45 of 188 Core ATen operators appear, 12278/14132 classifiable regions route to the accelerator (86.9%), accounting for 99.8% of the roster's loop-nest work. |

Adding a `check_*.py` means adding it here too — as a row in this table or as a step in a workflow.
