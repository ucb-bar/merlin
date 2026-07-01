---
name: artifact-layout
description: >-
  Where to write generated output in oscar-merlin — run dirs, results, plots, figures,
  caches, presentations, mined knowledge, dse analysis, any artifact. Use whenever you
  save output, create a run directory, emit results/figures, or wonder where generated
  files belong. Three roots only: runs/, artifacts/, build/.
---

# Artifact & run layout (MANDATORY convention)

Generated output in this repo lives in **exactly three top-level roots**. Never write
generated files anywhere else (no `output/`, `results/`, `selfcheck_out/`, `docs/presentation/`,
or per-experiment `runs/`/`reports/` — those are retired). A PreToolUse hook
(`.claude/hooks/guard_artifact_writes.py`) blocks violations; `build_tools/scripts/check_artifact_layout.py`
lints them.

| Root | Holds | How to create |
|------|-------|---------------|
| `runs/<target>/<suite>/<run-id>/` | aet experiment runs (logs/ metrics/ artifacts/ generated/ contracts/ + run_record.json) | `merlin.common.artifacts.start_run(...)` |
| `artifacts/<topic>/<target>/v<ver>/<leaf>/` | versioned products (+ manifest.yaml, `latest` symlink) | `merlin.common.artifacts.new_product(...)` |
| `artifacts/cache/<ns>/` | large regenerable caches (PURGEABLE) | `merlin.common.artifacts.cache_dir(ns)` |
| `artifacts/recaptures/` | 130 GB model recaptures (PURGEABLE) | `merlin.common.artifacts.recaptures_dir()` |
| `build/` | compiled / CMake / codegen scaffolds | build system |

**Concern-first.** `artifacts/` is organized by **tool/concern**, and EACH concern uses ITS OWN
natural axis — do not force a hardware target where it doesn't apply:

| Concern | Axis | Tools |
|---|---|---|
| `dse-guidance/<workload>/` | workload/model | `merlin-dse-guidance` |
| `dse/<workload>/<feature>/` | workload | `merlin-dse` |
| `design-pressure/<workload>/` | workload | `merlin-design-pressure` |
| `kernel-mining/<target>/<op>/` | **target** backend | `merlin-rvv-mine/-autotune/-report` |
| `kernel-index/<framework>/` | source framework | `kernel-index/-extract/-audit` |
| `ceiling/` | cross-framework | `kernel-bench` |
| `compare/<ts>/` | config×workload | `merlin-compare` |
| `measurements/<substrate>/<model>/<exp>_v<ver>_<TS>_<sha>/` | substrate→model→experiment | `scripts/k1_*`, firesim/zephyr/baremetal sweeps |
| `recaptures/<model>_<dtype>/` | model+dtype (PURGEABLE) | capture harness |
| `perf-bench/<target>/`, `capsule-bench/<target>/` | target | experiment reports |
| `targets/<target>/<package_id>/` | **target** backend | `merlin-rvv-mine/-autotune`, `merlin-targetgen` (codegen packages) |
| `presentation/`, `cache/`, `selfcheck/` | topic / ns / target | misc |

`runs/<target>/<suite>/<run-id>/` is for aet experiment runs (target-centric). `build/` for compiler output.

**Target at folder level (where target IS the axis).** Pass `target=` to `start_run`/`new_product`
so the target becomes a folder component. Keep **inner file names identical across the axis** (always
`perf_results.json`, `findings.csv`, `manifest.yaml`) so A-vs-B diffs are trivial. DSE is NOT
target-axis — it keys by workload/model.

## Naming convention ("sortable + provenance")

- Canonical timestamp token: `utc_stamp()` → `YYYYMMDDTHHMMSSZ` (UTC, no `:`, sortable).
- Runs: `<TS>_<method>_seed<NNN>_<sha7>` (timestamp-first → chronological `ls`).
- Products: `<topic>_v<ver>_<TS>_<sha7>` (topic+version are the parent dirs; leaf is self-describing).
- The folder name is a convenience; `run_record.json` / `manifest.yaml` (git_sha, timestamp,
  version, ...) is the source of truth.

## Usage

```python
from merlin.common.artifacts import start_run, finish_run, new_product, cache_dir

h = start_run(suite="gemmini-perf-bench", method="perf0001", target="gemmini")
(h.run_dir / "perf_results.json").write_text(...)      # write into h.run_dir / h.paths.*
h.store.record(h.run_dir / "perf_results.json", origin=...)   # ArtifactStore (content-addressed)
finish_run(h, "completed", summary={"n_kernels": 12})

prod = new_product("dse", version=1, target="bitvla")  # artifacts/dse/v1/dse_v1_<TS>_<sha7>/
out = prod.add_artifact("findings.csv"); out.write_text(...)
prod.write_manifest()

tmp = cache_dir("kernel_cache")                        # artifacts/cache/kernel_cache/

# hardware measurements: substrate -> model -> experiment (versioned + timestamped)
m = new_measurement("k1_spacemit", "bitvla", "cross_framework")   # also: firesim_<bitstream>,
out = m.add_artifact("cross_framework_k1.jsonl"); out.write_text(...)  # baremetal_<verilator-design>,
m.write_manifest()                                                #   zephyr_<design>, spike_<config>
```

## Deprecation: `output/` is going away

The legacy `output/` root is **deprecated**. It now holds ONLY the regenerable model recaptures
(accessed via `recaptures_dir()`); everything else has moved to `artifacts/`. **Never write new
generated content to `output/`** — the guard hook blocks it. Recaptures are PURGEABLE (regenerate
via the m2m exporter), so `output/` can be deleted once readers migrate to `recaptures_dir()`.

## Escape hatch

For a genuine one-off outside the roots: `export MERLIN_ALLOW_ARTIFACT_WRITE=1`, or add a path
prefix to `.claude/hooks/artifact_allowlist.txt`. Query past runs with `aet runs` / `aet show`.
