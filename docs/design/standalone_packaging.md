---
title: Design audit: standalone merlin wheel
kind: design
status: current
owner: core
last_verified: 2026-07-07
related: [repo_structure]
code_refs: [merlin/python, pyproject.toml]
---

# Design audit: making the `merlin` wheel standalone

**Status: audit only — no refactor has been applied.** This enumerates *every* change needed for
`pip install merlin` to work in a fresh environment **outside** the repo tree, with a phased,
independently-shippable execution plan for later approval.

All file:line references below were verified against the tree at the time of writing (grep the same
patterns to refresh). Paths are relative to `merlin/python/merlin/` unless noted.

## TL;DR

The wheel is **not** standalone today. Two independent facts cause it:

1. **No data is packaged.** `pyproject.toml` sets `package-dir = {"" = "merlin/python"}` +
   `packages.find where=["merlin/python"]` and declares **no** `package-data` / `include-package-data`
   / `MANIFEST.in`. So only `.py` files ship.
2. **Shipped code reads repo-tree data by walking up from `__file__`.** ~30 call sites resolve
   `Path(__file__).resolve().parents[N]` (N = 3/4/5 depending on file depth) or `repo_root()` to reach
   **sibling** trees of the package root — `merlin/{schemas,prompts,benchmarks,targets,runtime,contract}`.
   Installed under `site-packages/`, those parents point at unrelated directories → `FileNotFoundError`.

**Crucial scoping distinction** (keeps the refactor small):

- **External/board/toolchain deps are already env-gated and are NOT blockers.** chipyard, spike,
  verilator, riscv-gcc, VCS, the K1 board, XNNPACK/OpenBLAS/Saturn checkouts, m2m, clang,
  mlir-translate, Muon (VCS/cyclotron), ModelBlaster — every one reads a `MERLIN_*` env var with a
  default (see `runtime/backends/{gemmini,spike,muon,zephyr_model,vcs}.py`, `llvmlower/toolchain.py`,
  `rvvgen/k1.py`, `kernels/build_asm.py`). They fail gracefully at *use* time if unset, so a
  standalone install imports fine and the CLIs that don't touch them work. Leave these as-is;
  document them as optional extras.
- **Output dirs are a separate, smaller problem.** Many sites write to `repo_root()/artifacts|tmp|build`
  (e.g. `common/artifacts.py`, `dse/*.py`, `dse_guidance/cli.py`). `repo_root()` already honors
  `MERLIN_REPO_ROOT`, so an installed user can redirect outputs by setting it — but the *name* is
  misleading for an install. Addressed in Phase P4 (a distinct work-dir concept), not a hard blocker.

So the standalone work reduces to: **bundle the internal read-only data and resolve it via
`importlib.resources` with a repo-tree fallback.**

## 1. Repo-tree escapes (verified)

`repo_root()` = `MERLIN_REPO_ROOT` env else `Path(__file__).resolve().parents[4]`
(`common/paths.py:19,22`). The N varies only because files sit at different depths; every one
ultimately reaches a sibling of the package root, so **any file move silently rebreaks them** — an
argument for a single central resolver.

### 1a. Internal DATA reads — these BLOCK standalone

| Data class | Representative call sites (file:line) | Reaches | Env override? |
|---|---|---|---|
| **schemas** | `common/schemas.py:33` (`parents[3]/"schemas"`); `common/paths.py:32` (`schemas_dir`) | `merlin/schemas/*.schema.yaml` | ✅ `MERLIN_SCHEMAS_DIR` |
| **prompts** | `kernels/agent_mine.py:26`; `rvvgen/tuning_agent.py:29` (`parents[3]/"prompts"`) | `merlin/prompts/*.md` | ❌ none |
| **benchmarks** | `kernels/validate.py:26`; `design_pressure/cli.py:36`; `dse_guidance/{loader,study,case_study,cli,…}.py` (many, via `merlin_dir()/"benchmarks"`) | `merlin/benchmarks/**` | ⚠️ partial — `MERLIN_BENCH_DIR` only in `kernels/validate.py:23`; dse_guidance/design_pressure ignore it |
| **targets/contracts** | `targetgen/synthesize/{dialect_plan.py:72,target_contract.py:100}` (`targets_dir()`); `xdsl_dialects/lowering/{pipeline.py:34,target_lowering.py:52}` (`parents[5]`) | `merlin/targets/<t>/contracts/**` | ❌ none (`targets_dir()` has no env) |
| **contract ABI** | `targetgen/contract/{schemas.py:16,31; toolchain.py:17}`; `targetgen/oot_starterkit/cmdbuf.py:13` | `merlin/contract/**` | ✅ `MERLIN_CONTRACT_DIR` |
| **runtime C / ABI / baremetal** | `llvmlower/codegen.py:21`; `runtime/backends/spike_model.py:40,44,106`; `spike.py:57`; `zephyr_model.py:483-484`; `kernels/ceiling_drivers/multishape_compare.py:216-217` | `merlin/runtime/{c,abi,baremetal}/**` (`.c`/`.h`/`.S`) | ❌ none |
| **RTL facts** (target-cert) | `targetgen/circt_gate.py:21`; `rtl_checks.py:52`; `rtl/gen_{isa_module,numeric_facts,rtl_digest,rocc_replay}.py`; `rtl/circt_introspect.py:37` | `merlin/targets/<t>/contracts/rtl_facts/**` | ❌ none |

### 1b. External/board reads — already env-gated, NOT blockers (leave as-is)

`llvmlower/toolchain.py:17,22,29,36` (m2m/clang/mlir-translate); `runtime/backends/gemmini.py:46-76`,
`spike.py:39-50`, `muon.py:68-106`, `zephyr_model.py:100,619`, `vcs.py:21`;
`kernels/build_asm.py:173-219`; `rvvgen/k1.py:27-65`; `cost_model/calibrate.py:39`;
`targetgen/rtl/muon_introspect.py:30-34`. All have `MERLIN_*` defaults; document as optional extras.

### 1c. Output writes — Phase P4, not a hard blocker

`common/artifacts.py:238,314,334,346`; `dse/{cli,experiment,calibrate_npu}.py`;
`dse_guidance/cli.py`; `design_pressure/{cli,synthesize}.py`; `plotting/plot_paper_style.py:21`;
`rvvgen/autotune.py:98,153` — all under `repo_root()/{artifacts,tmp,build}`. Redirectable today via
`MERLIN_REPO_ROOT`; give it a clearer work-dir story in P4.

## 2. The packaging wrinkle: data lives OUTSIDE the package root

`package-dir = {"" = "merlin/python"}` makes `merlin/python/merlin/` the importable root. But every
data tree above is a **sibling of `merlin/python`** (`merlin/schemas`, `merlin/prompts`, …), i.e.
outside the package. setuptools `package-data` can only include files **inside** package dirs, so the
data cannot be bundled where it currently sits. Options:

- **(A) Move data under the package** → `merlin/python/merlin/_data/{schemas,prompts,contract,…}` and
  ship via `package-data`. Cleanest for `importlib.resources`, but a large move that also touches
  non-Python consumers of `merlin/runtime/*` (CMake/board build scripts) and the huge, untracked
  `rtl_facts`. **Not recommended wholesale** — only for the small pure-data (schemas, prompts).
- **(B) Build-time copy** of selected trees into `merlin/python/merlin/_data/` via a setuptools build
  hook, keeping the canonical copy in place for in-repo dev. Adds build complexity.
- **(C) Keep repo-coupled, formalize it**: require `MERLIN_REPO_ROOT` to point at a checkout;
  document `merlin` as a repo-resident dev tool. Zero refactor; not a true wheel.

**Recommendation:** hybrid — **(A) for schemas + prompts** (tiny, pure data, high import-frequency),
**(C)-with-clear-errors for the heavy/board-specific trees** (targets/rtl_facts, runtime C, benchmarks
bulk) behind extras + actionable "set `MERLIN_*` / install the `[board]` extra" messages. This makes
the *core* SDK (parse IR, load schemas, DSE on bundled benchmark specs, kernel-index) install-clean,
while the RTL-cert / board / simulator paths remain explicitly repo/extra-gated (they need external
checkouts anyway).

## 3. `pyproject.toml` changes

- Add `include-package-data = true` (or `[tool.setuptools.package-data] merlin = ["_data/**/*"]`).
- Introduce `merlin/python/merlin/_data/` and populate `schemas/` + `prompts/` there (Phase P0). Keep
  the canonical `merlin/{schemas,prompts}` for in-repo tooling; the resolver checks both.
- New optional extras to make the coupling explicit:
  `board = [...]` (spike/verilator/K1/chipyard usage — doc only, no PyPI deps),
  `targets = [...]` (RTL-cert inputs), reuse existing `dev`/`xdsl`/`kernels-*`.
- Do **not** bundle: `artifacts/`, `runs/`, `build/`, `tmp/`, `merlin/benchmarks/**/recaptures*`
  (the 130 GB recaptures — always `recaptures_dir()`/`MERLIN_BENCH_DIR`), or per-target `rtl_facts/`.

## 4. Central resolver + call-site migration

Add one helper to `merlin/common/paths.py`:

```python
def data_path(*parts: str) -> Path:
    """Resolve bundled read-only package data. Tries importlib.resources (installed wheel),
    falls back to the in-repo layout (repo_root()/merlin/...). Honors per-class env overrides."""
```

Then migrate the **1a** call sites to `data_path(...)` (or the class-specific `schemas_dir()` /
`prompts_dir()` / `contract_dir()` wrappers that call it). Grouped work:

- **schemas** (2 sites): `common/schemas.py:33`, keep `MERLIN_SCHEMAS_DIR` first.
- **prompts** (2): `kernels/agent_mine.py:26`, `rvvgen/tuning_agent.py:29` → new `prompts_dir()`.
- **benchmarks** (~15, mostly `dse_guidance/*`): route all through one `bench_dir()` that honors
  `MERLIN_BENCH_DIR` (fixes the partial-override gap), then `data_path`/repo fallback.
- **contract** (3): already has `MERLIN_CONTRACT_DIR`; add `data_path` fallback.
- **targets/contracts** (4) + **runtime C** (6) + **rtl_facts** (7): gate behind `data_path` with a
  clear "requires the `[targets]`/`[board]` data or a repo checkout" error.

Retire the ad-hoc `parents[N]` / private `_REPO` / `_repo_root()` duplicates
(`compare/{driver,empirical,structural}.py`, `rvvgen/{mine,autotune,k1}.py`, `targetgen/rtl/*`,
`kernels/decode/objdump.py`, …) in favor of the single resolver so future moves can't rebreak them.

## 5. The hard part — runtime C compiled at use time

`llvmlower/codegen.py`, `runtime/backends/{spike_model,zephyr_model}.py`, `rvvgen/k1.py`, and
`kernels/ceiling_drivers/multishape_compare.py` **compile `merlin/runtime/{c,abi}/*.c` at runtime**.
Bundling the sources as package data is necessary but not sufficient — the compile step reads them by
path. Approach: resolve via `data_path("runtime","c",...)`, and when running from a wheel, copy the
bundled sources to a temp build dir (`importlib.resources.as_file`) before invoking the toolchain.
These paths also need an external compiler (already env-gated), so they belong to the `[board]` extra;
the smoke test (P?) only asserts the *sources resolve*, not that a cross-compile runs.

## 6. Verification — standalone-install smoke test

Add `build_tools/scripts/check_standalone_install.py` (wired into CI once the refactor lands):

1. `python -m build` → wheel; `pip install` it into a fresh venv in a tempdir **with no repo on the
   path** and `MERLIN_REPO_ROOT` unset.
2. `--help` all 12 `[project.scripts]` (import-clean check).
3. One real invocation per bundled data class: load a schema, load an RVV-mining prompt, load a
   benchmark spec, load a reference target contract — assert **zero** `FileNotFoundError`.
4. Assert board/simulator CLIs fail with the *actionable* "set `MERLIN_*` / install `[board]`" message,
   not a raw traceback.

## 7. Phased execution plan (each phase independently shippable + verifiable)

- **P0 — package the pure data.** Create `_data/{schemas,prompts}`, add `include-package-data`,
  add the `data_path` resolver (schemas/prompts only). ⇒ `import merlin`, schema validation, and
  prompt loading work from a wheel. Lowest risk, highest payoff.
- **P1 — unify benchmarks.** One `bench_dir()` honoring `MERLIN_BENCH_DIR` across all dse_guidance /
  design_pressure sites; bundle the small benchmark **specs** (not recaptures).
- **P2 — contract + reference targets.** Bundle `merlin/contract` + reference `targets/<t>/contracts`
  (specs only, not `rtl_facts`); migrate the 4+ sites.
- **P3 — runtime C strategy.** `data_path` + `as_file` temp-copy for the compiled shims; put behind
  the `[board]` extra.
- **P4 — output work-dir.** Introduce an explicit writable work root (e.g. `MERLIN_WORK_DIR`,
  defaulting to `repo_root()` in-repo) so installed runs don't write under `site-packages`.
- **P5 — smoke-test gate.** Land `check_standalone_install.py`; add to `check_structure` / CI.
- **P6 — retire `parents[N]` duplicates.** Collapse the remaining ad-hoc resolvers into `data_path`.

P0–P1 deliver a genuinely useful core SDK; P2–P3 extend to targetgen/board flows; P4–P6 are hardening.

## Notes

- `repo_root()`'s existing `MERLIN_REPO_ROOT` override is the reason a repo-coupled install already
  "works" if you set it — P-phases remove that requirement class by class rather than all at once.
- Nothing here changes in-repo behavior: `data_path` falls back to today's layout when a checkout is
  present, so dev workflows and the existing gates are unaffected at every phase.
