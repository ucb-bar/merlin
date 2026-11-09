# CI workflows

| Workflow | Trigger | What it gates |
|---|---|---|
| [`ci.yml`](ci.yml) | push / PR / merge queue | Existing source, corpus, documentation and test gates; changed-file format gate; waits for hosted integration. |
| [`integration.yml`](integration.yml) | called by CI / weekly / manual | Wheel and source builds, distribution ownership, exact core-wheel installation outside the checkout, hosted runtime and vector-provider boundaries. |
| [`release.yml`](release.yml) | manual | Reuses integration and its retained distributions as a release qualification run. Package publication and hardware deployment are not configured. |

The primary workflow runs structure and documentation checking once. The visible
`pr-fast` workflow name and `fast`, `ruff` and `docs-freshness` check names are
retained. `fast` always reports the combined source/format/integration outcome;
`docs-freshness` reports its shared source gate's outcome without rerunning it.
The current-document freshness ratchet remains mandatory. The capsule-corpus
tests retain their early step and exclusion from the later broad pytest command.

The comparison job resolves one real revision for both format and ratchet checks.
For a new branch push, whose previous revision is all zeros, it uses the merge
base with the fetched default branch. An unavailable revision fails the job.

Hosted boundary tests execute C generation and configuration/provider contracts;
they do not qualify accelerator execution. Their JUnit results retain skips and
failures alongside built distributions. The OOT matrix selects six exact commits
in `build_tools/upstreams/backend-boundaries.json`, builds their wheels and runs
the selected cases outside source checkouts against the same main wheel. It uses
separate xDSL 0.68 and 0.65 environments; selected cases must execute without skips.
Native toolchain and hardware lanes remain separate qualification work.

The wiring gate runs after the pinned provider matrix, so real OOT consumers
participate in its source graph. It rechecks the current pins, clean Git heads
and exact tracked Python payloads; ignored outputs cannot manufacture a caller.
The separate `source_consumers` pins select exact support call-chain files in a
persistent sparse checkout. Those files are read only for this static source
graph; they are never imported, built, installed or exposed to candidates. They
add no provider test lane or simulator/runtime qualification.
Closed worker loaders and byte-matching editable compiler-source inputs are
source dependencies, distinct from host execution. Newly changed bodies gain no
credit from an older published inventory. The gate does not grant source access
or establish numerical, execution or hardware qualification.

The selected OOT commits must be available remotely before hosted CI can fetch
them. During local review, `check_backend_boundaries.py --source ID=PATH` accepts
only a clean checkout at the exact selected commit. Local commits and passing
local results do not establish remote CI success. FireSim setup and infrastructure
launch remain external to Merlin.

All external actions use full Git commit pins and read-only repository
permissions; checkout credentials are not persisted. The same local packaging
gate used by integration accepts `--wheel PATH --require-tools`, so an unavailable
installer cannot count as a passing required installation check. Inspect the
retained artifacts from the particular qualification run before publication.
The aggregate follows GitHub's documented
[job dependency and failure behavior](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-jobs);
action references follow its
[full-commit pinning guidance](https://docs.github.com/en/actions/reference/security/secure-use).

## Gates that deliberately run nowhere

A gate nobody runs is worse than no gate, because it reads as coverage. Measured 2026-09-16, 15 of
the 34 `build_tools/scripts/check_*.py` ran in no hook, no Stop-hook and no workflow — among them
`check_no_holdout_names`, which was *failing*: a tracked ratchet named two held-out capsules
outright, and a holdout's name is an answer key. Seven of the fifteen now run in `ci.yml`.

The rest stay out, each for a stated reason rather than by omission:

| Script | Why it is not wired |
|---|---|
| `check_standalone_install.py` | Wired in `integration.yml`: installs the exact built core wheel in a clean external venv; missing installer tools fail this required check. |
| ~~`check_claim_set_disjointness.py`~~ | **Wired 2026-09-22** into the primary gate, now `ci.yml`, with `--fail-on-circular`. The ">4 min on a populated `out/`" reason was measured locally; a CI runner clones fresh with no `out/`, and the full six-target walk times at 2m30s here. |
| `check_conformance_coverage.py` | **Partly wired since 2026-09-22.** The `--write` / re-derivation path reads every capture and rebuilds the requirement (> 4 min), and stays out-of-band. Comparing the corpus against the TRACKED spec (`--spec … --ratchet … --fail-on-uncovered`) skips all of that — measured **4.8 s** for gemmini — and runs in `ci.yml`, because what a diff can break is the corpus, not the requirement. |
| `check_isa_matches_rtl.py` | 171 s over all targets, and needs RTL fact bundles a hosted runner does not have. Currently FAILS (`saturn_opu_rvv` compares nothing), which is real debt, not a wiring gap. |
| `check_inert_capabilities.py` | ~60 s and currently reports debt against its own ratchet; wire it once that ledger is honest. |
| `check_generated_target.py` | Takes a path to one generated target repo — a per-target tool, not a repo-wide property. |
| `check_opset_coverage.py` | Requires an explicitly selected public op/form request through `--input`. Run it for that intake; a repository-wide invocation has no selected denominator. It must not discover a corpus or create debt implicitly. |
| `check_reference_readiness.py` | An [operator reference-coverage audit](../../docs/guides/kernel_reconstruction.md) requiring the selected `--workspace`, `--facts`, `--contract`, `--target`, one or more `--reference` machine artifacts and a fresh `--output` directory. It has no repository-wide reference denominator; run it explicitly for those artifacts. Missing coverage exits 2, and static readiness does not establish execution or performance qualification. |
| `check_repro_env.py` | A preflight that reports which repro capabilities THIS machine has. Its answer is environment-dependent by design, so failing CI on it would be meaningless. |
| `check_cert_affordability.py` | Wired, but advisory: ~60 s, and it prices a cohort rather than asserting a property of the diff under review. |
| `report_generalization_claim.py` | Not a `check_*`, but it is the script that produces the generalization number, so its absence belongs here too. Its denominator is the target's DECLARED roster and its evidence is `out/artifacts/recaptures/` — 130 GB a hosted runner does not have. With zero captures present it would report 0/0 on every PR, which reads as coverage while measuring nothing; `--require-roster` would fail every PR for the same reason. Run out-of-band. Measured 2026-09-22 for gemmini, `--require-roster` clean: across 9 captures of the declared roster (`resnet50`, `tiny_llama`), 45 of 188 Core ATen operators appear, 12278/14132 classifiable regions route to the accelerator (86.9%), accounting for 99.8% of the roster's loop-nest work. |

Adding a `check_*.py` means adding it here too — as a row in this table or as a step in a workflow.
