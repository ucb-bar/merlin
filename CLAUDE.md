# Repository conventions

This file documents the conventions this repo enforces — the generated-output layout, the test
layout, and the documentation model. Read it (and the directory-local `AGENT.md` files) before making
changes; the same gates run in pre-commit and CI.

> Working-tree note: this repo is frequently developed by multiple coding agents against a single
> working tree. Commit on the currently checked-out branch and avoid switching branches mid-task; use
> a separate `git worktree` if you need isolation.

# Target-agnostic convention — derive, never hardcode (the cardinal rule)

The whole point of this repo is to plug in *any* hardware target (RTL repo) and have the compiler,
grader, and tooling work. So **library code must never bake in facts about a specific target.** A fact
about a target is *extracted* from that target's own sources (RTL via mlc/CIRCT, the capability
manifest, the ISA definition, the descriptor) at run time — it is never a literal in the code. Three
hard prohibitions, each gate-enforced (pre-commit + CI); do not add allowlist entries to route around
them without a written, reviewed rationale:

1. **No target-name literals in library code.** No `"gemmini"` / `"atlas"` / `"radiance"` (or any
   target string) in `merlin/python/merlin/**` or `build_tools/scripts/**`. The target is a *parameter*
   threaded from the descriptor/manifest; functions take `target=`, they do not assume one. A target
   name may appear only at a genuine edge where that target is legitimately the subject (a per-target
   data dir, a target-specific test, a caller that is *about* that one target) — never in a shared code
   path. Enforced by `build_tools/scripts/check_no_target_name.py` (scan roots
   `merlin/python/merlin` + `build_tools/scripts`).
2. **No regex.** Do not `import re` in core library code. Regex line-matching is brittle by
   construction — a too-narrow pattern silently drops valid-but-differently-spelled input (this has bitten
   the RoCC trace decoder repeatedly: numeric-only SSA ids, `"r,r"`-only constraints, one op spelling —
   each silently mis-measured a conformant backend). Parse **structurally** instead: real parsers,
   `str.split`/`partition`, explicit tokenizers, the xDSL/MLIR IR. Enforced by
   `build_tools/scripts/check_no_regex.py` (ratcheted allowlist; do not grow it).
3. **No assumed opcodes / encodings / ISA constants.** Never hardcode an opcode (`0x7b`), a funct/func3
   value, a mesh dimension, a memory base, an address layout, a register field, or a "the instruction
   always starts with …" assumption. Every such value is *derived* from the target's RTL facts
   (`rtl/facts.load_facts(target)` — e.g. `funct_decode_table.custom_opcode`) or its capability manifest,
   and *compared as data* (parse the field to an int and compare to the derived fact — never string-match
   a literal). Fields that vary per-instruction (e.g. the RoCC xd/xs1/xs2 `func3`) must NOT be treated as
   identity constraints. When a value cannot be derived, **fail closed** (record `UNKNOWN`, surface it) —
   never silently drop the input or substitute a baked default.

Rule of thumb: if adding a second target would require editing shared code (not just adding a
descriptor + data dir), the code is overfit — lift the fact into derivation. See memories
`derive-dont-overfit-hw-agnostic`, `abi-is-derivable-not-irreducible`, `no-regex-sweep`, and the OV/*
de-overfitting task series.

# Hardware-provenance convention — pin it, verify it, record it

A result that claims a hardware verdict must record **which hardware revision it came from**. This is not
bookkeeping: a session certified a microkernel 31/31 against the only `saturn` revision containing the
outer-product unit while the revision named for the tapeout does not contain that unit at all, and nothing
in the artifact recorded which one the numbers belonged to. A result attributed to the wrong device is
worse than no result, because it gets cited.

- **One registry**: `merlin/contract/hardware_pins.yaml` — tracked and reviewed, full quoted 40-char shas.
  Each pin carries `requires_paths` (what the work needs present) and may carry `forbids_paths` (whose
  *absence* is the point). Verify by CONTENT, not by branch name: branches move and forks share them.
- **API**: `merlin.common.provenance` — `verify()` (declared vs actual), `record()` (the block to embed),
  `require()` (the raising form), `source_digest()` (the bytes actually READ, which a dirty tree changes
  while the commit still looks right). It never mutates a checkout — other sessions work in those trees.
- **Binaries carry their own stamp** and the report compares it, so a stale ELF is detectable
  (`kernels.opu_cert.provenance_stamp`).
- **Enforced** by `build_tools/scripts/check_provenance.py` (pre-commit `--staged`, session Stop hook
  `--stop-hook`, `--verify-pins` for live checkouts). It scans untracked reports under `out/artifacts/`
  too, since that is where reports live. `provenance_ratchet.txt` lists pre-existing debt and MAY ONLY
  SHRINK — never add to it; regenerate the artifact with provenance instead.

See `.claude/skills/hardware-pins/SKILL.md`.

# Generated-output convention — one root (`out/`), three subdirs

All generated/produced output lives under a **single top-level `out/` root**, with exactly three
subdirs (`out/runs/`, `out/artifacts/`, `out/build/`). Nothing generated goes anywhere else (the old
top-level `runs/`/`artifacts/`/`build/`, plus `output/`, `results/`, `selfcheck_out/`,
`docs/presentation/`, and per-experiment `runs/`/`reports/` locations are retired and gitignored).

Root names come from `merlin.common.paths` — `out_dir()` / `runs_dir()` / `artifacts_dir()` /
`build_dir()` (honoring `MERLIN_OUT_ROOT`). Never hard-code the literal strings; call the helpers.

- **`out/runs/<target>/<suite>/<run-id>/`** — aet-managed experiment runs. Create via
  `merlin.common.artifacts.start_run(..., target=...)` (never hand-build a run path); it passes
  `project_root=out_dir()` so aet lays runs under `out/runs/`. Query with `aet runs` — point it at
  the `out/` root (it reads `<project_root>/runs`).
- **`out/artifacts/`** — every other generated product, organized **concern-first** (each tool/concern
  owns a subtree and uses ITS OWN axis — target for compiler/mining/experiments, workload for the
  three DSE tools, model for recaptures/measurements, framework for kernel-index, cross-cutting for
  ceiling/compare). Concerns: `dse-guidance/`, `dse/`, `design-pressure/`, `kernel-mining/<target>/`,
  `kernel-index/<framework>/`, `optimization-surface/<target>/`, `ceiling/`, `compare/`,
  `measurements/<substrate>/<model>/<exp>_v<ver>_<TS>_<sha>/` (substrate = `k1_spacemit` /
  `firesim_<bitstream>` / `baremetal_<verilator-design>` / `zephyr_<design>` / `spike_<config>`,
  via `new_measurement(...)`), `recaptures/`, `perf-bench/<target>/`, `capsule-bench/<target>/`,
  `targets/<target>/`, `presentation/`, `cache/`, `selfcheck/`.

  **`out/artifacts/targets/<target>/<package_id>/`** is the codegen-package home (schedules/knobs/dialects
  minted by `merlin-rvv-mine` / `merlin-rvv-autotune` / `merlin-targetgen`). It **replaces the retired
  top-level `generated_targets/`** — all references point at `out/artifacts/targets/` directly (no compat
  symlink; paths are repo-root-relative). Packages are **tool-generated**: only the hand-authored
  reference baselines + promoted champions are tracked (`rvv/hand_v0`, `rvv/hand_v0_int8`,
  `rvv/impr_tuned_*`, via `.gitignore` negations); forks and `mlir_oot/build/` trees stay ignored.
  Full buildable OOT repos live under `out/build/generated/`, not here.

  **`output/` is DEPRECATED and retired** — model recaptures now live at `out/artifacts/recaptures/`
  (via `recaptures_dir()`); never write new generated content to `output/` (the guard hook blocks it).
  - versioned products at `out/artifacts/<concern>/<axis>/v<ver>/<concern>_<axis>_v<ver>_<TS>_<sha7>/`
    (+ `manifest.yaml`, relative `latest` symlink) via `new_product(..., target=...)`;
  - regenerable caches under `out/artifacts/cache/<ns>/` via `cache_dir(...)` (PURGEABLE);
  - the 130 GB model recaptures under `out/artifacts/recaptures/` (PURGEABLE) via `recaptures_dir()`.
- **`out/build/`** — compiled / CMake output, baseline toolchains, and buildable OOT codegen repos
  (`out/build/generated/`).

**Naming ("sortable + provenance")**: timestamp token `YYYYMMDDTHHMMSSZ` (UTC, no `:`); runs are
timestamp-first (`<TS>_<method>_seed<NNN>_<sha7>`), products are topic-first
(`<topic>_<target>_v<ver>_<TS>_<sha7>`). The folder name is convenience; `run_record.json`/`manifest.yaml`
(git_sha, timestamp, version) is the source of truth.

**Target at folder level**: pass `target=` so it becomes a folder component
(`out/runs/<target>/<suite>/...`, `out/artifacts/<topic>/<target>/...`) and everything for a target
groups together. Keep inner file names identical across targets (e.g. `perf_results.json`, `findings.csv`,
`manifest.yaml`) so target-vs-target diffs are trivial.

**Enforcement** (do not bypass without cause): a PreToolUse hook
(`.claude/hooks/guard_artifact_writes.py`) blocks generated writes outside the `out/` root;
`build_tools/scripts/check_artifact_layout.py` lints tracked-file violations (pre-commit / Stop hook).
Helper API and examples: `.claude/skills/artifact-layout/SKILL.md` and `merlin/python/merlin/common/artifacts.py`.
Escape hatch for a genuine one-off: `export MERLIN_ALLOW_ARTIFACT_WRITE=1` or add a prefix to
`.claude/hooks/artifact_allowlist.txt`.

# Test layout — one suite, organized by subsystem

All tests live in **`merlin/tests/`** (the sole pytest `testpaths`), organized into **subsystem
buckets**: `kernels/ rvv/ dse/ gemmini/ targetgen/ ir/ runtime/ infra/`. Rules (enforced by
`build_tools/scripts/check_structure.py` "test layout"; see `.claude/skills/test-layout`):

- A test file is `merlin/tests/<bucket>/test_<area>.py` — **never at the `merlin/tests/` root**, and
  `<bucket>` must be one of the eight above. Place a new test in the subsystem it exercises.
- Shared inputs live in `merlin/tests/fixtures/` and `merlin/tests/data/`.
- Resolve repo paths via `merlin.common.paths.repo_root()` / `merlin_dir()` — **never** `Path(__file__).parents[N]`
  (so tests are location-independent and survive moves).
- Run the suite: `.venv/bin/python -m pytest merlin/tests`.

# Documentation convention — durable docs vs reports

Start at the generated hub **`docs/README.md`**. Durable documentation lives in **`docs/`** under
three kind-subdirs — **`reference/`** (code-derived facts), **`guides/`** (how-tos), **`design/`**
(rationale) — each file carrying YAML front-matter (`title, kind, status, owner, last_verified,
related, code_refs`). **Point-in-time reports** (results/findings/status/presentations) are NOT docs
— they live under `artifacts/` (concern-first; see "Generated-output convention"). Rules (see
`.claude/skills/docs-layout`; enforced by `check_structure.py` + the `check_docs.py` Stop/pre-commit/CI gates):

- Generated docs are regenerated, never hand-edited: `gen_cli_docs.py` (`reference/cli.md`),
  `gen_package_docs.py` (`reference/module_index.md`), `gen_schema_docs.py` (`reference/schemas.md`),
  `gen_docs_index.py` (`README.md` hub). Run them after touching CLIs/schemas/packages/docs.
- Front-matter must be schema-valid; `check_doc_paths.py` blocks retired paths; the root README/AGENT.md
  may not carry scaffold-era phrasing.
- **Semantic drift** (a doc whose `code_refs` moved past its `last_verified`) is surfaced by
  `check_docs_freshness.py --json` and fixed by the **`docs-doctor`** skill (reconcile, then bump the date).
- Enable the pre-commit gate per clone: `python build_tools/scripts/install_git_hooks.py`.

# Commit message convention

Commit messages describe the **change to the code/files** and read as if a human developer wrote them.
A reader of the history should understand *what changed and why* without any knowledge of how the work
was produced. Enforced by the `commit-msg` hook (`build_tools/git-hooks/`, installed by
`install_git_hooks.py`); the same gates run in CI.

- **Format:** `type(scope): summary` — `type` ∈ `feat|fix|docs|chore|refactor|test|perf|build`;
  `summary` in the imperative, **≤ 72 chars**. Optional body (wrapped) explains the *why* and the
  behavior/file impact, not the process. Optional footer for issue refs.
- **Describe the change, never the agent's process.** NO plan/task-step framing in the subject — no
  `Phase 4`, `P2+P3`, `iteration-2`, `PART2`, `step 3`, `round 4`, `WIP`, `FINALIZE`, `checkpoint`,
  `snapshot`. A commit is a unit of file change; name it by that (e.g. `fix(gemmini): int32_t harness
  output buffer`), not by where it fell in a workstream.
- **No decoration / no attribution noise.** No emoji, no `Co-Authored-By`, no "Generated with …", no
  AI/tool attribution. (See also: never add co-authorship trailers.)
- **One coherent change per commit**, with a message that stands on its own. Avoid run-on subjects that
  pack a whole results narrative onto the first line — put nuance in the body.
- Merge commits are the exception the hook skips; avoid auto-named throwaway branches
  (`worktree-agent-<hash>`) leaking into `main`'s merge subjects — squash or rename before merging.
