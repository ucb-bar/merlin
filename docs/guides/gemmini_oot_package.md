---
title: Running the certified gemmini OOT backend on a fresh machine
kind: guide
status: current
owner: compiler
last_verified: 2026-09-02
related: [whole_model_on_accelerator, reproducing_whole_model_on_rtl, gemmini_experiment, reproducibility, adding_a_target]
code_refs: [merlin/python/merlin/targetgen/capsule_grade.py, merlin/python/merlin/targetgen/capsule_runner.py, merlin/python/merlin/targetgen/rtl_engine_policy.py, merlin/python/merlin/targetgen/contract/compile.py, merlin/python/merlin/perf/hw_counters.py, merlin/contract/hardware_pins.yaml]
---

# Running the certified gemmini OOT backend on a fresh machine

There is a gemmini out-of-tree MLIR backend that certified **33/33 capsules at L3** (elaborated RTL) on
2026-09-02. This page is what a second person needs to re-run it somewhere else and read the result
correctly. Concepts for the *other* whole-model path live in
[whole_model_on_accelerator](whole_model_on_accelerator.md); its runbook is
[reproducing_whole_model_on_rtl](reproducing_whole_model_on_rtl.md). **This is a different artifact and
a different flow** — an agent-generated OOT backend graded through `capsule_grade`, not
`compile_model`.

## Read this before quoting the number

`33/33` is over a **34-capsule cohort, not the corpus**. From
`merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml`:

```yaml
expected_cohort: {source_capsules: 48, admitted_capsules: 34}
capability_exclude_capsules: [GC1…GC6, GF1…GF5]     # 11 bf16 ops this RTL cannot execute
resource_bound:
  policy: representative_l3_capstones_v1
  exclude_capsules: [GX0_interop_rvv_lane, M0_small_llama_gemmini, M1_lstmnetvit_gemmini]
  required_admitted_models: [M2_microvit_gemmini, M3_host_island_seam_gemmini]
```

48 − 11 − 3 = 34.

| | |
|---|---|
| **Established** | 13 ISA capsules + 11 layers + 8 model slices + 1 model capsule, certified on elaborated RTL |
| **Established** | the host/device seam: `M3_host_island_seam` (GEMM → host LayerNorm → GEMM) with an ordered dispatch ledger and per-tile mesh verification |
| **Not established** | whole-model compilation — the three capstones that would show it are excluded by `resource_bound`, and are neither passed nor refuted |
| **Not established** | anything about `M2_microvit`, which ended `budget_exhausted` |

So: **do not tell anyone that 33/33 means a full model compiles.** If that is the question, the
capsule to grade is `M0_small_llama_gemmini` (embed / RMSNorm / RoPE / attention / SwiGLU / lm_head),
and it has not been graded. See [Grading the excluded capstones](#grading-the-excluded-capstones).

## The package

Tracked at `out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0/`, with per-file checksums:

```bash
cd out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0
sha256sum -c SHA256SUMS --quiet && echo OK
```

`PROVENANCE.md` beside it records the run, the candidate, and what the verdict covers. ~1,100 lines of
original code; the rest is vendored `xdsl`/`typing_extensions`/`immutabledict`/`ordered_set`, kept as-is
so the directory *is* the graded artifact.

Its contract is one tool with four commands (`manifest.yaml`):

| command | argv |
|---|---|
| `parse` | `gemmini-opt --verify-diagnostics <in.mlir>` |
| `lower_interface_to_target` | `gemmini-opt --convert-iface-to-gemmini <in.mlir>` |
| `emit_command_buffer` | `gemmini-opt --convert-iface-to-gemmini --emit-command-buffer=<out.json> <in.mlir>` |
| `lower_target_to_llvm` | `gemmini-opt --convert-iface-to-gemmini --emit-target-artifact <in.mlir>` |

## ⚠️ Getting the code that can grade it

**This is the part that does not work yet, and it is the first thing to fix.** The harness that produced
and can reproduce `33/33` is:

- branch **`arm4/functional-launch-v4`** at **`b860f1c5`**, which has **no upstream — it is not pushed**;
- plus **107 modified tracked files** and **77 untracked files** in that working tree.

Nine of those modifications are in the capsule corpus itself (`GP0`/`GP1`/`GP2` `capsule.yaml` and
`capsule.interface.mlir`, `FT00`/`FT01` `capsule.yaml`, `profiles/gemmini.yaml`). Some of them relaxed
what the capsule demands. **A coworker grading against the committed corpus will not get 33/33**, and the
difference is not the compiler.

Until that branch is committed and pushed, a second machine cannot reproduce this. Do that first:

```bash
# in the arm4 worktree, with no grade running against it
git add -- <the source paths>            # explicit pathspecs; the tree is shared
git commit
git push -u origin arm4/functional-launch-v4
```

**Do not try to grade this package on `main` or on `feat/target-generalization`.** Their reference
interpreter and command-buffer schema are older and lack opcodes this backend emits:

| | `main` lineage | `arm4/functional-launch-v4` |
|---|---|---|
| `runtime/reference.py` | 168 lines | 335 lines |
| `MODELED_OPCODES` | 6 | 10 (`+ATTENTION_PV, ATTENTION_QK, CONV2D, MOVEMENT`) |
| `command_buffer.schema.json` opcodes | `ATTENTION_QK` only | `+CONV2D, MOVEMENT` |

Measured: grading this package on the older lineage fails **6 capsules** — `A1_mvin_mvout`, `FT00`,
`B3`, `B4`, `GP2`, `C7` — on schema vocabulary alone, with no compiler defect involved.

## Prerequisites

| | what | how it is found |
|---|---|---|
| Python | 3.13 venv at `.venv` (uv) | `.venv/bin/python`; plain `python` is not on PATH |
| `.env` | external-path map | copy `.env.example`; without it collection dies with `external path 'chipyard' unset` |
| `third_party/llvm-install` | the LLVM 23 host toolchain | **see the trap below** — a fresh checkout does not have it |
| RISC-V toolchain | for the bare-metal harness link | chipyard's `esp-tools`/`riscv-tools` env |
| chipyard + gemmini RTL | the L2/L3 oracles | `MERLIN_CHIPYARD`; **verify the revision** against `gemmini_rtl` in `merlin/contract/hardware_pins.yaml` |
| L2 oracle | spike + `libgemmini` | built in chipyard |
| L3 oracle | one of VCS / **GSIM** / Verilator | see below |

### Choosing the L3 engine

`merlin/python/merlin/targetgen/rtl_engine_policy.py` selects by availability in cost order
(`vcs`, `gsim`, `verilator`) and records every engine it passed over. All three produce an
`elaborated_rtl` verdict — the tier is a **fidelity, not a simulator**.

- **GSIM** is the practical choice, and is what certified 32 of the 33. Point `MERLIN_GEMMINI_GSIM_EMU`
  at the emulator binary.
- **Verilator** is the *citable* instrument for cycle counts. GSIM runs a slightly different elaborated
  design (harness clock ratios collapsed, one pad cell removed from a clock path), and measured
  same-ELF it reports **0 to +2 cycles per kernel invocation, ≤ +9 per capsule window** against
  Verilator. Fine for a functional verdict; **never mix the two in one comparison.**

## Grade it

```bash
export TMPDIR=/path/with/space                       # not /tmp; whole-model builds need room
export PYTHONPATH=$PWD/merlin/python                 # pin it — a shared venv can shadow the tree
export MERLIN_GEMMINI_GSIM_EMU=/path/to/emu_gemmini_gsim   # or leave unset for Verilator

.venv/bin/python -m merlin.targetgen.capsule_grade \
    --package out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0 \
    --target gemmini --labels public \
    --runs-root out/runs/gemmini/verify --score out/runs/gemmini/verify/score.json \
    --timeout 21600 --workers 6
```

Omit `--capsules`: the default is the target's own graded roots, which is what applies the cohort
policy above. A hand-passed root is how a package ends up graded against the wrong corpus.

Cycle counts are concurrency-invariant, so `--workers` does not change them. **Wall times are not
comparable across different `--workers` values.**

### Read the result correctly

Never quote a bare score. Per capsule, read `tiers.<T>.status`, `cycles`, `derived_from_rtl`,
`cycle_accurate` and `evidence`:

```json
"L3": {"status": "pass", "cycles": 19658, "derived_from_rtl": true,
       "cycle_accurate": true, "evidence": "rtl_gsim_console.log"}
```

A `pass` whose `tiers` is empty, or whose `derived_from_rtl` is `false`, is not an RTL result. `status`
values that are **not** the compiler getting it wrong: `gated`, `budget_exhausted`,
`not_gradeable_no_oracle`, `incomplete` (`NOT_RUN_IS_NOT_PASS`). Do not count those as failures *or*
passes.

## Host ⇄ device: what the seam now establishes

This is the part that improved, and it is worth being exact about.

`M3_host_island_seam_gemmini` is a GEMM → host LayerNorm → GEMM capsule where the host island is the
*subject*, not a by-product. It passes at L3 with:

- an **ordered dispatch ledger** — `{on_mesh: 2, scalar_rvv_lane: 16}`, so both lanes provably carried
  work;
- **per-tile mesh verification** (`mesh_tile_verification.per_tile`), 2 of 2 tiles certified;
- `boundary_execution: A->H->A/pass`, matching an independently derived expectation;
- **`model_execution_check: pass`** with zero violations — the first recorded verdict from that gate,
  which had been computed and discarded on every prior run.

Its L3 evidence is therefore `mesh_execution.dispatch_ledger + mesh_tile_verification.per_tile` rather
than a console log, and that is a stronger claim, not a weaker one: the tiles were verified
individually.

Two honest caveats:

- **The host half executes on x86.** The lane is served by `lower_model(targets=("host",))` +
  `build_host_shared`, run through ctypes. `scalar_rvv_lane` is a *routing label*; the seam is proven as
  a **partition**, not yet as RVV execution on the device.
- Set **`MERLIN_MESH_VERIFY=1`** to get the per-tile verification. It defaults **off**, and a
  whole-model pass taken without it carries `derived_from_rtl: false`, no ledger and `evidence: null` —
  which is a vacuous pass by this repo's own standard.

## Per-unit activity, optionally

The RTL counts the cycles each combination of its three engines (`EX`/`LD`/`ST`) was busy, and the
harness can bracket a kernel with those counters:

```bash
export MERLIN_HW_COUNTERS=1
```

Off by default and deliberately so — a change that altered every run would make one round's verdicts
incomparable with the rounds before it. With it on, each RTL tier record gains a
`timing_observations` block (`busy_cycles.<engine>.in_program`, `overlap_cycles.observed`,
`idle_cycles.no_unit_busy`) plus a `timing_capability` record. The block is refused on functional
tiers: a model that runs the program without modelling the engines returned per-engine busy totals in
the *thousands* for a 52-cycle window.

Measured on the K-sweep with this package, Verilator: 20–41% of each window has **no** engine busy, and
`min(T_compute, T_movement) − realised_overlap` leaves 70–83 cycles of overlap still available.

## Grading the excluded capstones

If the actual question is "does a full model compile", grade the three the cohort excludes. They are
excluded by policy, not by capability, so you must stage them explicitly:

```bash
# stage the admitted cohort PLUS the resource-excluded capstones into one flat root
.venv/bin/python -m merlin.targetgen.capsule_grade \
    --package out/artifacts/targets/gemmini/gemmini_xdsl_rtl_v0 \
    --capsules <flat root with M0/M1/GX0 re-admitted> \
    --target gemmini --labels public \
    --runs-root out/runs/gemmini/capstones --score .../score.json \
    --timeout 43200 --workers 4
```

Set `MERLIN_MODEL_BUDGET_S` generously — `M2` ends `budget_exhausted` at the default. And grade the
operator capsules **in the same invocation**: a whole-model capsule is deferred unless the operator
pass fraction clears its declared `after_op_pass_fraction`, and grading the model directory alone
computes that fraction as `0.00` and gates everything.

## Traps

- **A missing host toolchain looks like 43 compiler failures.** A fresh checkout has no
  `third_party/llvm-install`, and every capsule then fails as
  `spike / tool_crash: [Errno 2] No such file or directory: .../bin/clang-23`. Measured: 43 graded,
  43 failed, which reads exactly like a broken backend and is not one. Link or install it first:
  `ln -s <llvm-23-install> third_party/llvm-install`, then confirm `third_party/llvm-install/bin/clang-23`
  resolves. With it in place the same package graded 17/17 on the same commit.
- **A shared venv can shadow the tree.** `import merlin` may resolve to a *different* worktree than
  your `cwd`. Always pin `PYTHONPATH`, and check `merlin.__file__` if a result surprises you.
- **`TMPDIR=/tmp` is not enough space** for whole-model builds, and an empty `TMPDIR` fails oddly.
- **Hardware provenance.** A cycle count without the RTL revision it came from is not citable. Check
  `gemmini_rtl` in `hardware_pins.yaml` by content, not by branch name — branches move and forks share
  them.
- **The 11 bf16 capsules are excluded because this RTL cannot execute them.** That is a capability
  fact, not a gap in the backend.
