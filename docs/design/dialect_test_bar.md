---
title: The cross-target dialect test bar
kind: design
status: current
owner: core
last_verified: 2026-07-25
related: [compiler_plane, target_publishing]
code_refs: [merlin/contract/capsules, merlin/python/merlin/targetgen/capsule_golden.py, merlin/python/merlin/targetgen/capsule_runner.py, merlin/contract/schemas/capsule.schema.json]
---

# The cross-target dialect test bar

How do we define a *correct* test for a hardware dialect we just brought up — one that grades a backend
honestly, generalizes across accelerators, and never rewards a wrong answer? This note states the bar the
capsule-bench uses and why, and records the concrete case that forced it: a float MXU (atlas) cannot be
graded against an integer accelerator's (gemmini) answer key.

## Three principles

### (a) Numeric goldens: chosen by the op AND the datapath, from an INDEPENDENT oracle
The comparison policy is a property of the *datapath*, not a global default:

- **Integer datapaths → exact.** Gemmini's i8×i8→i32 systolic array is bit-deterministic; the golden is
  `exact_int` and any mismatch is a real bug. (`numeric_policy.compare: exact_int`.)
- **Float datapaths → ULP / atol-rtol.** Atlas's fp8-e4m3 × fp8-e4m3 → bf16 MXU accumulates in bf16 with
  per-step rounding; bf16 reassociation is a *legal* backend choice, so the golden is `tolerance_float`
  with a tolerance derived from the format (here rtol = 2e-2, the atlas spec's own `mxu0`
  `effect.precision.tol`; atol = 0.25, ~one bf16 ULP at the tile magnitudes). An `exact_int` bar on a float
  MXU is not "strict", it is *wrong*: the accelerator physically cannot hit an integer answer.

The golden must come from an **independent oracle**, never the target's own RTL (that is circular — it
proves the RTL equals itself). For integer capsules the oracle is merlin's dependency-free `Tensor`
engine (structurally independent of the emitted command buffer). For atlas's float capsules the oracle is
**specir's reference model** (`specir.oracle.dtypes` fp8/bf16 codecs + `refmodel.fp_reduce`) — the same
primitives `specc testbench --gen atlas-npu --op op.matmul_mxu0` uses to emit the E4M3FMA-cell golden,
here composed cell→tile. A second engine (torch fp8/bf16) *corroborates* when present; it never silently
becomes the golden.

### (b) Coverage as a DERIVED percentage, not a vibe
"Enough tests" is a measured number with two components:

1. **Instruction / mode coverage** — the capsule declares the ISA instruction classes and datapath modes it
   must exercise (`expected.instruction_classes`, `expected.modes`); the trace gate checks the emitted
   stream actually issues them (for command-ISA targets with a decoder plugin).
2. **`specc tb-coverage`** — what fraction of the *target's own shipped testbench suite* the spec's op set
   reaches. For atlas: 40/41 (97%) of the shipped Scala TBs map to a spec op; the merlin corpus then
   covers the `op.matmul_mxu0` fp8-systolic family (single-tile, K-accum, resident-reuse, scale, relu,
   attention/MLP), leaving the VPU transcendental/`mxu1`-tree lane to a follow-on corpus. The gap is
   *named*, not hand-waved. (Ledger: `out/artifacts/compare/atlas/tb_coverage/`.)

### (c) Oracle-fidelity climb to RTL
Passing the math floor is necessary, not sufficient. The tier ladder climbs oracle fidelity:
`L0` independent numeric golden → `L1` reference==simulate → trace gate → `L2/L3` a cycle-accurate RTL
oracle (spike/verilator, or the mlc arc model / program-oracle for a self-hosted-ISA target like atlas).
A mandatory tier that cannot run leaves the capsule **`incomplete`, never `pass`** (`not_run_is_not_pass`).
The bar is: an independent golden the backend can't see, in the datapath's own arithmetic, corroborated,
then confirmed against RTL.

## The case that forced this: atlas float MXU vs the gemmini integer corpus
The atlas 4-arm evaluation originally pointed `capsule_corpus` at the shared gemmini corpus
(`merlin/contract/capsules/isa`, i8×i8→i32, `exact_int`). That grades a float MXU against integer answers —
unwinnable by construction. The fix is an atlas-correct corpus (`merlin/contract/capsules/atlas/`): fp8-e4m3
operands, bf16 output, `tolerance_float`, goldens from the specir refmodel's fp8/bf16 datapath.

**Honest limitation recorded here:** the current runner recomputes the L0 golden via
`capsule_golden.golden()` on the integer-only `Tensor` engine and does **not** read `golden.yaml` (that file
is the withheld answer key + provenance). So end-to-end *grading* of a float atlas capsule still needs
`capsule_golden`/`Tensor` extended to fp8-decode + bf16 matmul (or the runner taught to consume the
independent `golden.yaml`). Until then the atlas corpus is validated as *well-formed + datatype-correct +
independently-oracled* (schema-valid load, bf16 float goldens, `tolerance_float` honored by `compare()`),
not yet graded through the full RTL ladder. We flag this rather than fake a green run.

## The friend's question: "Vortex in Verilator — how do we define the tests correctly?"
A colleague bringing up **Vortex (a SIMT GPU) in Verilator** asked exactly this: what makes a *correct*
test suite for a freshly-elaborated dialect? The three principles above are the answer — pick the numeric
policy from the datapath (a GPU's fp lanes want ULP/atol-rtol, not exact-int), derive coverage from the
core's own TB + the issued-instruction set, and climb the oracle ladder to RTL — but the honest part is
what happened on **radiance**, our own SIMT target:

> There was **no SIMT-specific capsule suite**. Radiance reused the **op-neutral matmul capsules** (a
> matmul is the same math on a systolic array or a warp of threads — only the lowering and the oracle
> differ) plus a **perf benchmark** (gflops / % of fp peak) for the throughput story the functional
> capsules don't capture. The routing distinction (`arc_available=False`, cyclotron oracle instead of the
> arc command-buffer path) is a *plumbing* difference, not a different test philosophy.

So the honest radiance answer to "how do we test a SIMT core correctly" was: **reuse the op-neutral
functional corpus with a SIMT-appropriate oracle + policy, and add a perf bench** — not author a bespoke
SIMT suite. The same recipe applies to Vortex-in-Verilator: op-neutral functional capsules graded with a
float (ULP/atol-rtol) policy from an independent fp oracle, coverage derived from Vortex's own TB via a
`repo_tb.yaml`, and Verilator as the RTL tier — no SIMT-specific test language required.
