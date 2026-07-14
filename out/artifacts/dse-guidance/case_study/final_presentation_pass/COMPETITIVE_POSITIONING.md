# Merlin — Competitive positioning

*What our infrastructure covers that others don't — and, just as importantly, what it does **not** yet
cover.* Companion to the OSCAR Workshop deck (`Merlin-9.pdf`): grounds the bringup flow (p16), the
per-backend support matrix (p17), and the AI-for-compilers related-work slide (p15).

---

## 1. Purpose & honesty contract

This document positions Merlin against the projects it is most often compared with: **Triton
(GPU + CPU), OpenXLA/XLA, IREE, ExecuTorch** (whole-model compilers) and **XNNPACK, OpenBLAS, EXO**
(kernel libraries / exocompiler). Every comparison here was taken from the **actual source of those
projects checked out on this machine**, not from reputation — the exact paths and `du -sh` sizes are
recorded in §3–§4 so any claim is re-checkable. Two further projects — **microTVM** and **Halide** — are
*not* checked out here; we reference them only where unavoidable and tag them `OFF-MACHINE` (public
knowledge, not source-verified).

**Evidence discipline.** Each Merlin claim below carries a strength tag:

| Tag | Meaning |
|---|---|
| **SOLID** | wired in code today; a named file backs it; safe to say on a slide as-is |
| **PARTIAL** | the mechanism exists but is narrower than the headline; say it with the stated caveat |
| **ROADMAP** | designed / scaffolded but **not yet wired or not yet measured**; say it only as future work |

**Guardrails (things we must NOT say).** These are pinned to source so we don't drift:

- ❌ "zero-dependency runtime" — the Merlin-*owned* runtime is small, but the full image still uses
  Zephyr + the MLIR runtime functions + picolibc.
- ❌ "HW spec automatically generates the dialect/passes" — RTL is *located and cited*, **not interpreted**
  (`ingest/scala_chisel.py`: "we make no attempt to interpret RTL"; non-toy synthesis is flagged
  `requires_human_review`).
- ⚠️ "CIRCT-grounded feedback makes the agent cheaper / faster" — **measured N=1** (capsule-bench A/B:
  `merlincirct_abc9` converged 20/20 in one round at lowest cost/tokens of four arms). Say "in our A/B
  (N=1)"; do NOT state a stable multiplier until the N>1 ablation runs (see §9, Novelty 1).
- ❌ any **system / model-level** speedup — our only credited performance number is an **isolated-kernel**
  one, and the deck's own thesis (p13) is *kernel speed-up ≠ system speed-up*.
- ❌ "we beat / are lighter than IREE/Triton/EXO/ExecuTorch" as a blanket statement — we differentiate on
  **target class and workflow**, not on a benchmark we have not run against them. In particular ExecuTorch's
  *core runtime is also genuinely lean* (§3); do not claim "lighter runtime" as the differentiator.
- ❌ crediting the on-disk **IREE checkout** with Zephyr / Gemmini / RISC-V support. That checkout
  (`/path/to/merlin-iree/third_party/iree_bar`) is a **UCB-BAR / copparihollmann fork**, and its
  `IREE_PLATFORM_ZEPHYR` plumbing + `merlin_gemmini_counter.h` are **our own commits** (`8c4ee4ad08`
  "Bare-metal / Zephyr platform support", `05e56f2bd1`) — **not stock IREE**. Stock upstream IREE has no
  Zephyr/Gemmini path; never present our fork work as a competitor feature (it would *under*-credit us).

---

## 2. Two buckets, two different comparisons

The eight on-disk projects are not the same kind of thing, so one matrix would be misleading. We split them:

- **Whole-model compilers** — Triton, Triton-CPU, IREE, XLA, **ExecuTorch**. These take a higher-level
  program and *generate* an artifact you run. The interesting axes are *what they generate*, *how heavy the
  runtime is*, and *which targets they reach*. **Caveat on the label:** Triton/Triton-CPU are really
  *kernel DSLs* (you bring the kernel, not the model — kept here for comparison), and **ExecuTorch is
  whole-model but PyTorch-native, *not* MLIR** (it lowers `torch.export`→EXIR→`.pte`, no MLIR in its
  pipeline). So within this bucket only IREE, XLA and Merlin are genuinely MLIR-class.
- **Kernel libraries / exocompiler** — XNNPACK, OpenBLAS, EXO. These provide (or generate) individual
  kernels, not a whole-model compile+run path. The interesting axes are *codegen vs hand-tuning*, *ISA
  reach*, and *whether we can mine them into our compiler*.

Merlin sits in the **MLIR-class compiler** bucket but deliberately overlaps the kernel-library world from
the other side: it *consumes* those libraries as mining input (§9, Novelty 2).

**The two closest honest neighbours** are **ExecuTorch** (whole-model + lean runtime + a real, if WIP,
Zephyr path — the nearest competitor on the *embedded-whole-model* axis) and **EXO** (lightweight,
C-emitting, co-design-by-design — the nearest competitor on *philosophy*). `OFF-MACHINE` peers worth
naming for completeness: **microTVM** (the obvious C-emitting, bare-metal, whole-model peer — its absence
from this machine is the one real gap in our source-grounded comparison) and **Halide** (a schedule-DSL
peer to EXO). We tag both as public-knowledge, not source-verified.

---

## At a glance — the axes that matter (fair Yes / Partial / No)

The single table to reuse. **Yes / Partial / No** with a ≤3-word reason; every verdict is from the on-disk
source (paths in §3–§4 / §1). **Partial** = the capability exists but is narrower than a clean yes. The
**bottom three rows are deliberately the ones Merlin does *not* lead** — a fair table has to show them.

| Capability | Merlin | Triton | Triton-CPU | IREE | XLA | ExecuTorch | XNNPACK | OpenBLAS | EXO |
|---|---|---|---|---|---|---|---|---|---|
| Whole-model compile | Yes (PyTorch→IR) | No (kernel DSL) | No (kernel DSL) | Yes | Yes (via framework) | Yes (torch.export) | No (op lib) | No (BLAS only) | No (per-kernel) |
| Emits C (first-class) | Yes (default) | No (PTX) | No (.so via LLVM) | Partial (emitc VM-C) | No (LLVM IR) | No (.pte binary) | No (links lib) | No (links lib) | Yes (generates C) |
| Embedded reach (bare-metal/RTOS) | Yes (Zephyr+spike) | No | No | Partial (custom HAL) | No | Partial (Zephyr WIP) | No (mobile/server) | Partial (Cortex-M community) | Partial (portable C) |
| New-target bringup tooling | Yes (contract+cert) | No (per-ISA) | No (LLVM target) | Partial (HAL plugin) | No | Partial (delegate API) | No (hand kernels) | No (hand kernels) | Partial (exocompilation) |
| RISC-V/RVV + accel (Gemmini) | Yes (RVV+Gemmini) | No | No | Partial (RISC-V CPU) | No | Partial (delegated RVV) | Partial (generic RVV) | Partial (tuned RVV) | Partial (RVV/Gemmini models) |
| AI inside compiler flow | Yes (gated agents) | No | No | No | No | No | No | No | No |
| GPU target | No (embedded focus) | Yes (NVIDIA/AMD) | No (CPU only) | Yes (CUDA/Vulkan) | Yes (GPU/TPU) | Partial (GPU delegates) | No (CPU) | No (CPU) | No |
| Production maturity (at scale) | No (research) | Yes | Partial (WIP) | Yes | Yes | Yes | Yes | Yes | Partial (academic) |
| ISA / target breadth | Partial (RV/Gemmini focus) | Partial (GPU vendors) | Partial (x86/ARM) | Yes (wide) | Partial (GPU/CPU/TPU) | Yes (10+ backends) | Yes (wide) | Yes (very wide) | Partial (user-defined) |

*Fairness, source-checked: IREE's C path is real (`Dialect/VM/Target/C/CModuleTarget.cpp`) and it has
RISC-V platform support, so both are Partial not No; EXO ships `platforms/{rvv,gemmini}.py`; ExecuTorch has
a `backends/cortex_m/` + `arm_zephyr.cpp`; OpenBLAS has explicit Cortex-M (community). The top six rows are
where Merlin leads; the bottom three are where it does not — **only Merlin is Yes across all six leaders at
once**, but it is No/Partial on GPU, maturity, and breadth.*

---

## 3. Capability matrix — whole-model compilers

Legend: ✅ yes · ⚠️ partial/conditional · ❌ no/not-in-source · 🔑 our differentiator.
Sizes are real `du -sh` of the on-disk checkout (paths in §1 guardrail / below).

| Capability | **Merlin** | Triton (GPU) | Triton-CPU | IREE | XLA | ExecuTorch |
|---|---|---|---|---|---|---|
| Checkout size (`du -sh`) | this repo | 726 M | 126 M | 5.3 G † | 144 M | 1.9 G |
| MLIR-based | ✅ (xDSL dialects) | ✅ | ✅ | ✅ | ✅ (+ legacy HLO) | ❌ (PyTorch EXIR) |
| Whole-model frontend (model→IR) | ✅ PyTorch/TorchAO→MLIR | ❌ (kernel DSL) | ❌ (kernel DSL) | ✅ | ✅ (framework-bound) | ✅ `torch.export`→EXIR |
| Emits **C** as a first-class target | 🔑 ✅ | ❌ (PTX/cubin) | ❌ (`.so` via LLVM) | ⚠️ (HAL/native, no C src) | ❌ (LLVM IR) | ❌ (`.pte` flatbuffer) |
| **Bare-metal** target | 🔑 ✅ spike (~365 LOC) | ❌ | ❌ | ⚠️ via custom HAL | ❌ | ⚠️ WIP (cortex_m, "not for production") |
| **RTOS / Zephyr** runtime | 🔑 ✅ whole-model on Zephyr SMP | ❌ | ❌ | ❌ (stock: none; fork-only ‡) | ❌ | ⚠️ real module, WIP (`arm_zephyr.cpp`) |
| Runtime weight | 🔑 lean (~210 LOC core C) | heavy (Py+CUDA/ROCm) | heavy | heavy tree † (runtime subset thinner) | heavy (framework) | lean (~50 KB core, advertised) |
| Primary target class | embedded / custom accel / RISC-V | datacenter GPU | server CPU (WIP) | datacenter↔edge | datacenter GPU/CPU | mobile / edge / MCU |
| New-target **bringup tooling** | 🔑 ✅ `targetgen/` (~70+ mods) + cert ladder | ❌ | ❌ | ⚠️ HAL plugin | ❌ | ⚠️ backend-delegate API |
| RISC-V / RVV / custom-accel (Gemmini) | 🔑 ✅ first-class | ❌ | ❌ | ⚠️ RISC-V plumbing (fork ‡) | ❌ | ❌ (CPU via XNNPACK delegate) |
| HW-fact ingestion (RTL/CIRCT) | ⚠️ extract-only (see §7) | ❌ | ❌ | ❌ | ❌ | ❌ |
| AI used *inside the compiler flow* | 🔑 ✅ gated (see §8) | ❌ | ❌ | ❌ | ❌ | ❌ |

† IREE's 5.3 G is the **fork tree including bundled third_party (LLVM, torch-mlir)**; its runtime subtree
is far smaller — do not quote 5.3 G as "the runtime." ‡ The on-disk IREE is a UCB-BAR fork; its Zephyr /
Gemmini / RISC-V bits are **our commits**, not stock IREE (§1 guardrail).

**Read this honestly:**
- **IREE** is the most serious MLIR-class overlap — it *also* scales toward edge and has a pluggable HAL,
  and its runtime is "thin" *relative to a datacenter stack*. But **stock** IREE has no Zephyr path, and
  reaching bare-metal means writing a custom HAL driver. Our differentiation is **degree and default**:
  C-emission + RTOS/bare-metal + RISC-V/accelerator are the *default* path, not a porting exercise.
- **ExecuTorch** is the most serious *embedded-whole-model* overlap — genuinely lean runtime, a real (WIP)
  Zephyr module, and a backend-delegate API. We differ on four verifiable axes, **not** runtime weight:
  it is **not MLIR-based**, **does not emit C** (binary `.pte`), has **no RISC-V/Gemmini backend** (CPU via
  XNNPACK delegate), and has **no AI inside the compiler**. It is also the cleanest reminder that "lean
  runtime" alone is not our moat.
- **Triton / Triton-CPU** are not whole-model compilers — they compile hand-written kernels in a Python
  DSL (Triton-CPU emits a `.so` via LLVM, not C source). Different problem.
- **XLA** cannot be used standalone; it is invoked *through* TensorFlow / JAX / PyTorch.

---

## 4. Capability matrix — kernel libraries / exocompiler

| Capability | **Merlin** | XNNPACK | OpenBLAS | EXO |
|---|---|---|---|---|
| Whole-model compile+run | 🔑 ✅ | ❌ (op primitives) | ❌ (BLAS only) | ❌ (per-kernel) |
| Produces code by | MLIR lowering + codegen | hand-tuned intrinsics/asm | hand-written asm | user-scheduled → C codegen |
| Emits portable C | ✅ | ❌ (links a lib) | ❌ (links a lib) | ✅ |
| Embedded / bare-metal reach | ✅ (RTOS + spike) | ⚠️ no explicit | ⚠️ **Cortex-M (community)** | ✅ (C output is portable) |
| ISA breadth today | RISC-V/RVV + Gemmini focus | wide (ARM/x86/RVV/Hexagon) | very wide (incl. RISC-V RVV) | user-defined per target |
| ISA / HW **co-design** posture | bringup-oriented (see §6–7) | per-ISA hand tuning | per-ISA hand tuning | 🔑 exocompilation by design |
| We can **mine it into our compiler** | — | ✅ ingested | ✅ ingested | ✅ ingested |
| Checkout size (`du -sh`) | this repo | 362 M | 272 M | 69 M (pure Python) |

**Source-verified ISA depth (these ground the "hand-tune per ISA" and "mineable expert RVV" claims):**
- **OpenBLAS** has the deepest RVV story examined: `kernel/riscv64/` holds **95 RVV kernels + 22 vector-
  length-specific variants** (ZVL128B / ZVL256B, plus SiFive **C910V**-specific GEMMs), several
  *auto-generated* by `kernel/riscv64/generate_kernel.py` carrying explicit tile/LMUL metadata — i.e. exactly
  the structured `vfmacc`/tiling decisions we decode in Novelty 2 (§9).
- **XNNPACK** RVV is a single dynamically-vectorized template (`src/f32-gemm/MRxNRv-rvv.c.in`), not
  per-VLEN hand-tuned — mineable but thinner than OpenBLAS.
- **EXO** is **pure Python, no LLVM/MLIR**, and ships hardware models as Python *platform modules*
  (`src/exo/platforms/{x86,neon,sve_*,rvv,gemmini}.py`); its `rvv.py` is f32-only and minimal, and no
  `apps/` example uses RVV — so EXO's RVV is a *framework hook*, not a tuned library.

**EXO is the closest competitor to our pillars — be precise.** EXO is the lightest project examined
(~69 MB, pure Python, no LLVM/MLIR), it emits C, and its *whole reason for existing* is letting hardware
vendors model a new target externally without forking a compiler. That is genuinely adjacent to our
"lightweight + co-design" story, so the honest line is narrow:

- **What we add over EXO:** a model→MLIR **whole-model frontend** (EXO has none — you bring kernels);
  a **whole-model runtime** on RTOS/bare-metal (EXO emits a kernel, not a model-running image);
  **automated kernel mining** that abstracts experts into compiler capabilities (§9); and an **RTL-grounded
  bringup + certification ladder** (§6).
- **What we must NOT claim:** that EXO can't reach embedded (its C output can), or that EXO isn't
  co-design-oriented (it is, by design). We differ in *scope* (kernel-scheduling DSL vs whole-model
  compiler+runtime), not in "they ignore embedded."

---

## 5. "Lightweight compiler infrastructure" — what it means for us, and how we get there

**Definition (for us):** a path from a real model to a small, certifiable C/RTOS image, with a runtime we
own end to end — not a port of a datacenter stack scaled down.

**How we get there (all SOLID, file-grounded):**

1. **Model → MLIR.** PyTorch/TorchAO captured into our xDSL dialects (the deck's "Model to MLIR" block, p16).
2. **Dialect + passes → C codegen.** Three independent C emitters:
   - `merlin/python/merlin/runtime/backends/gemmini_codegen.py` — command-buffer → bare-metal C using
     libgemmini intrinsics (explicit weight-stationary sequence, "NOT tiled_matmul_auto", certifiable).
   - `merlin/python/merlin/runtime/backends/rvv_codegen.py` — command-buffer → bare-metal C `main.c`
     around the RVV asm kernel, epilogues in scalar C matching `runtime.tensor` semantics exactly.
   - `merlin/python/merlin/xdsl_dialects/lowering/schedule_dispatch.py::emit_schedule_c` — partitioned
     schedule emitted as a C header the multicore runtime consumes.
3. **Lean runtime.** `merlin/runtime/c/*` + `merlin/runtime/abi/mlir_runtime.c` (**~210 LOC** core, measured)
   for host; `merlin/runtime/baremetal/spike/` (**~365 LOC** of bringup glue: `crt.S`, `htif`, `merlin_malloc`,
   `libc_min`, `model_main`, linker scripts — excluding the AGENT note and the standalone RVV kernel asm) for
   bare-metal; numpy-only `dispatch_runtime.py` as the reference executor.
4. **RTOS image.** `merlin/python/merlin/runtime/backends/zephyr_model.py` runs a whole model on **Zephyr
   SMP** (spike today, FireSim 2-tile board target), with worker-thread pinning to the RVV tile and
   architecture-aware flags (rv64gcv for the vector tile, rv64gc for scalar).

**Caveat (guardrail):** "lightweight" is about the **Merlin-owned** code. The full image still depends on
Zephyr, the MLIR runtime functions, and picolibc. Say "lean, owned runtime," not "no dependencies."

---

## 6. Pillar 2 — bringup & hardware development (SOLID, with caveats)

**Requirement:** bringing up a *new* target should be a tractable, auditable workflow — not a compiler
fork. Concretely, three things must hold: **(1)** a *declarative target contract* the compiler reads
(capabilities, capacities, oracle ladder) instead of hard-coded target logic; **(2)** an *out-of-tree
boundary* so a vendor can plug a target in **without Merlin source access** and without us reading their
internals; **(3)** a *fail-closed certification ladder* where "not run" is never "pass." This is where
most of our differentiation against both buckets actually lives, and it is real today.

**How we construct the SW for a new target (file-grounded):**

- A **target contract** (`merlin/targets/<t>/contracts/target_contract.yaml`, schema in
  `merlin/schemas/target_contract.schema.yaml`) declares capabilities, obligations, capacities (with
  provenance tags) and the oracle ladder. A **dialect plan** maps the hardware to dialect ops.
- `merlin/python/merlin/targetgen/` (~72 modules: `ingest / synthesize / generate / validate / evidence`)
  orchestrates the flow and records provenance for every decision.
- `targetgen/contract/oot_runner.py` hooks **any** contract-satisfying third-party package into Merlin
  across a subprocess + file boundary, with manifest validation and integrity scanning (forbids importing
  Merlin internals / reading reference outputs) — vendors can plug in without source access.
- `merlin/python/merlin/targetgen/capsule_runner.py` runs an **L0–L5 certification ladder** (L0 numeric
  golden → L1 simulate → trace decode/check → L2 spike → L3 verilator → L4 VCS → L5 FireSim), fail-closed
  at every gate.

**Caveats (guardrails):**
- The **RTL-derived cheap-checks layer** (`merlin/python/merlin/targetgen/rtl_checks.py`) is *self-labeled*
  **"Phase 0, ADVISORY / UN-WIRED"** — "imported by NOTHING in the frozen runner/grader/schema path … changes
  no pass/fail verdict." Say "in-progress RTL-derived pre-screen," not "RTL checks gate the flow."
- **L4/L5** (VCS/FireSim) are **config-gated adapters** — available when the environment provides them, not
  guaranteed. Say "6-tier ladder with optional RTL-verified L4/L5."

**Differentiation:** none of Triton/XLA expose new-target bringup; IREE offers a HAL-plugin path but not a
contract + certification-ladder workflow; the kernel libraries bring up a target by **hand-writing kernels
per ISA**. Our contract+oot+ladder workflow is the distinctive piece.

---

## 7. Pillar 3 — HW-SW co-design (PARTIAL today + clearly-labeled ROADMAP)

**Requirement:** the hardware description and the compiler should inform each other. For even the
*one-directional* half to hold, two things must be true: **(1)** HW facts must be **extractable
deterministically and provenance-tagged** (we must be able to say *where* each number came from), and
**(2)** those facts must **actually constrain lowering** (a capacity in the contract must bound what the
compiler may emit, not just sit in a report). The bidirectional half (compiler→HW feedback) is the part
that carries the **highest overclaim risk**, so we split it cleanly.

**How we "construct the HW" — the honest version.** We do **not** author hardware. The accelerators we
target (Gemmini, Saturn, Radiance) are existing UCB Chisel/RTL; our flow **locates, cites, and extracts
structural facts** from that RTL (`ingest/scala_chisel.py` + `rtl/{introspect,circt_introspect}.py`), and a
**human** then authors the target contract / dialect-plan from it. So the SW is generated against the HW;
the HW is *ingested*, not produced. Saying otherwise is the guardrail below.

**What is real today (PARTIAL, file-grounded):**
- **HW sources are ingested and cited** — `targetgen/ingest/source_manifest.py` records `source_dirs /
  files / urls / scala_roots`; `ingest/scala_chisel.py` discovers Scala/Chisel RTL so the evidence report
  can cite it.
- **Structural RTL facts are extracted** via `targetgen/rtl/introspect.py` (v1: grep over firtool FIRRTL +
  hierarchy JSON) and `targetgen/rtl/circt_introspect.py` (v2: CIRCT HW-dialect; mesh rows/cols, scratchpad
  bytes, datapath widths, accumulator ports, funct decode table). Deterministic, provenance-tagged.
- **The contract constrains lowering** — capacities like resident-storage-bytes flow from the contract into
  `xdsl_dialects/lowering/` and bound what the compiler may emit.

**What is NOT true (guardrails — do not claim):**
- The compiler does **not interpret RTL** to synthesize a dialect. `ingest/scala_chisel.py` states it
  outright: "we make no attempt to interpret RTL"; non-toy synthesis is flagged `requires_human_review`.
  A human reads the RTL, authors the contract/dialect-plan YAML, and the tooling then validates it.
- The flow is **one-directional**: HW facts → compiler constraints. There is **no compiler→HW feedback
  loop** in the code.

**Roadmap (future work, say as such):** auto-derive dialect-op candidates from extracted RTL facts; close
the compiler→HW feedback edge. The deck's "HW Spec + Golden Kernels → Dialect & Passes Gen" arrow (p16) is
the *target* of this pillar; today the arrow is human-mediated.

**Honest framing:** call this **"RTL-grounded specification and bringup,"** not "HW-driven dialect
generation."

---

## 8. AI integration into the compiler

**Principle: the LLM proposes, a deterministic gate disposes — no unchecked AI output reaches the
compiler, and no agent ever emits a credited number.**

**This is a genuine differentiator, now triangulated against source, not reputation:** grepping all eight
on-disk competitors (`llm` / `openai` / `anthropic` / `claude` / `gpt` / `agent`) found **no AI anywhere
in their compiler flows** — the only hit, Triton-CPU's `driver.py`, is a joke comment ("Thanks ChatGPT!").
Several of them *run* LLMs as models (ExecuTorch, IREE); none *use* an LLM to build or tune the compiler.
Every Merlin touchpoint, grounded:

| Touchpoint | File | The gate |
|---|---|---|
| Gemmini codegen agent | `merlin/targetgen/agent/kernel_slot.py` | cheat-token scan (no reading golden/reference) → functional correctness on visible rungs → RTL cert on held-out |
| Target-spec planning agent | `merlin/targetgen/agent/claude_cli.py` | headless Claude CLI subprocess; output parsed into structured slots, then schema-validated |
| RVV knob proposer | `merlin/python/merlin/rvvgen/tuning_agent.py` | optional; proposals clamped to `_KNOWN_OVERRIDE_KEYS`, unknown knobs dropped; deterministic gap-router fallback if no LLM |
| Kernel-mining judgment | `merlin/python/merlin/kernels/agent_mine.py` | optional; deterministic features (markers/regimes/motifs) are the base, LLM only adds nuance |
| DSE-guidance critic | `merlin/python/merlin/dse_guidance/agent/` | citation-gated devil's-advocate; must quote ≥8-char verbatim substrings of real artifacts |

This is the differentiator versus "an LLM in the loop": correctness/cheat/citation gates are structural,
and the deterministic path always exists without the LLM.

---

## 9. The two novelty claims

### Novelty 2 — Mining LLM-generated **and** expert kernels into the compiler infrastructure (SOLID)

We ingest expert kernels (XNNPACK, OpenBLAS, Autocomp/Gemmini, Exo, Triton) **plus our own compiler
output**, decode them to a Common Compute Abstraction **deterministically** ("zero LLM calls over
thousands of kernels"), cross-check each lift (asm-lifted vs clang-AST-lifted must agree per field or the
kernel is quarantined), and promote findings up a ladder
`Observation → Motif → Policy → Validated → L6 dialect requirement`.

- Ingest adapters: `merlin/python/merlin/kernels/ingest/`; method: `docs/kernel_mining.md`,
  `docs/rvv_kernel_mining_methodology.md`.
- The extracted decisions become compiler features/knobs (`llvmlower/impr_features.py`) explored by a
  certified beam search — **not** hand-kernels and **not** shape-overfit constants (project principle
  `abstract-into-compiler-not-overfit`).
- **One credited measurement anchor:** forming a real `vector.contract` so the asm carries `vfmacc`
  instead of `vfmul+vfadd` was **MEASURED ~7.9× on the isolated kernel** (64³ f32, N=5, cos=1.0, vs the
  frozen baseline), evidence kernels openblas/xnnpack RVV gemm/gemv/trmm
  (`docs/rvv_mining_report.md`, `merlin/python/merlin/kernels/action_catalog.py`).
  - **Guardrail:** this is an **isolated-kernel** number — exactly the deck's p13 distinction
    (*kernel speed-up ≠ system speed-up*). Do not promote it to a model/system speedup. Nothing that fails
    the cosine gate can ever produce a credited speedup (`speedup = None` otherwise).
  - **Scope guardrail:** today LLM-generated kernels are not yet a *distinct mining source* — the LLM is an
    optional, clamped beam proposer; the **mining itself is deterministic over expert libraries.** Say it
    that way.

### Novelty 1 — CIRCT-compiled HW info → LLM → faster iteration / lower token usage (MEASURED N=1)

The intent: feed the agent facts/checks that were *compiled* through another MLIR flow (CIRCT: Chisel/FIRRTL
→ firtool → hw/seq/comb MLIR → extractor → `facts.json` → advisory `rtl_checks`) so it brings up a backend in
fewer rounds/tokens. This is now **measured (N=1)** in the `capsule_bench_v0` A/B — see
`CIRCT_RTL_ARTIFACT_JOURNEY.md` §11 and `scripts/gen_trajectory_v2.py::ARMS`:

| Arm (run) | Converged /20 | Rounds | Cost | Tokens | Tool-calls |
|---|---|---|---|---|---|
| raw C++ `rb_abc11` | ✅ 20 | 5 | $147.34 | 82.0M | 442 |
| C++ + scaffold `rbinfra_abc11` | ❌ 17 | 5 | $159.48 | 84.3M | 677 |
| Merlin Python `merlin_abc9` | ❌ 19 | 10 | $86.43 | 45.8M | 352 |
| **Merlin + CIRCT `merlincirct_abc9`** | ✅ **20** | **1** | **$52.73** | **29.2M** | **137** |

The CIRCT-grounded arm converged **20/20 in one round** at the **lowest** cost / tokens / tool-calls of all
four — i.e. the "fewer tokens / faster iteration" effect is observed, not just designed.

**Honest caveats (keep all three):**
- **N=1 per arm** — magnitudes are directional until N>1; say "in this A/B," not a stable multiplier.
- A *separate*, more formal track is still unrun: `targetgen-evals/reports/gemmini/ablation_table.md` =
  "*(no real baseline runs yet)*" — that controlled ablation (vs without CIRCT facts) hasn't been executed.
- The committed `agent_spec` target packages did **not** use CIRCT as a *generation* input
  (`artifacts/targets/gemmini/*/inputs/rtl_facts.yaml`: `provenance_class: verilator-oracle-only`); the
  measured win above is CIRCT-as-advisory-feedback in the bring-up loop, not CIRCT-conditioned codegen.

**Say this exactly:** "In our capsule-bench A/B (N=1), the CIRCT-grounded arm reached full convergence in a
single round at the lowest cost/tokens of four arms; a controlled N>1 ablation is the next step to state a
stable magnitude."

### Related-work differentiation (deck p15)

The p15 prior work all applies AI to an **existing compiler optimizing for an existing target**:

| Work | What it generates | Correctness signal |
|---|---|---|
| Magellan (AlphaEvolve) | C++ pass heuristics | compiler legality + bench reward |
| Compiler-R1 (RL) | LLVM pass sequences | tool reward + IR instruction count |
| LLM-VeriOpt | LLVM IR peephole rewrites | Alive2 semantic equivalence |

**Our difference (two axes):**
1. **Where the AI acts** — not tuning passes for a fixed target, but **bringing up a new target** (dialect,
   passes, runtime adapter) and **mining experts into general compiler capabilities**.
2. **The gate** — outputs are gated by **functional + RTL certification and citation/cheat scans** (§8),
   not only by IR-instruction-count proxies or single-tool equivalence.

The p15 works are *research systems* that apply AI to a compiler; the production stacks we benchmark
against (Triton/XLA/IREE/ExecuTorch + the kernel libs) ship **no AI in the flow at all** (§8). So our
"AI inside a real bringup compiler, structurally gated" position is distinct on both the research axis
(p15) and the production axis — and that claim is now backed by grepping eight real codebases.

(The deck's p15 honest note — "Undecided on how much value this adds" — stays honest here: Novelty 1's
*value* is unmeasured until the ablation runs.)

---

## 10. Claims ledger (overclaim firewall + speaker notes)

| # | Claim | Strength | Evidence | Safe phrasing | Avoid |
|---|---|---|---|---|---|
| 1 | Emits C as first-class output | SOLID | `gemmini_codegen.py`, `rvv_codegen.py`, `schedule_dispatch.py` | "three independent C emitters" | "only one who emits C" |
| 2 | Zephyr/RTOS whole-model runtime | SOLID | `zephyr_model.py` | "first-class Zephyr SMP runtime (spike today, FireSim target)" | "runs on any RTOS" |
| 3 | Bare-metal + lean runtime | SOLID | `runtime/baremetal/spike/`, `runtime/c/*` | "lean, owned runtime (~210 core + ~365 bringup LOC, measured)" | "zero-dependency" / "lighter than ExecuTorch" |
| 4 | New-target bringup tooling | SOLID | `targetgen/` (~72 mods), `oot_runner.py`, `capsule_runner.py` | "contract + oot + L0–L5 cert ladder (~70+ modules)" | "fully automated bringup" |
| 5 | RTL-derived checks | PARTIAL | `rtl_checks.py` ("ADVISORY/UN-WIRED") | "in-progress RTL pre-screen" | "RTL checks gate the flow" |
| 6 | L4/L5 RTL-accurate cert | PARTIAL | `capsule_runner.py` | "optional config-gated L4/L5" | "always cycle-accurate certified" |
| 7 | RTL-grounded HW specification | PARTIAL | `ingest/scala_chisel.py`, `rtl/circt_introspect.py` | "RTL facts extracted + cited; contract human-authored" | "HW spec auto-generates the dialect" |
| 8 | HW-SW co-design (auto + feedback) | ROADMAP | (none wired) | "designed; human-mediated today" | "compiler and HW co-optimize" |
| 9 | Mine experts into compiler | SOLID | `kernels/ingest/`, `kernel_mining.md`, promotion ladder | "deterministic mining of expert libs into compiler features" | "auto-mines LLM kernels as a source" |
| 10 | 7.9× from `vfmacc` fusion | SOLID (isolated) | `rvv_mining_report.md`, `action_catalog.py` | "~7.9× on the isolated kernel, cos-gated" | "7.9× system/model speedup" |
| 11 | CIRCT-grounded feedback → cheaper/faster agent | MEASURED (N=1) | `merlincirct_abc9` vs 3 arms (`gen_trajectory_v2.py`); `targetgen-evals` ablation still unrun | "in our A/B (N=1), 20/20 in one round at lowest cost/tokens" | a stable multiplier; "CIRCT-conditioned codegen" (it was advisory feedback) |
| 12 | AI inside the compiler | SOLID | `kernel_slot.py`, `claude_cli.py`, `tuning_agent.py`, dse_guidance critic | "AI proposes, deterministic gate disposes; none of the 8 competitors do this" | "the LLM writes the compiler" |
| 13 | vs ExecuTorch (closest embedded-whole-model peer) | SOLID | ExecuTorch source: `runtime/` (~50 KB core), `arm_zephyr.cpp`, no MLIR, `.pte` only, no Gemmini/RISC-V backend | "we differ on MLIR-class + C-default + RISC-V/accel + cert ladder" | "we're lighter than ExecuTorch" / "ExecuTorch can't do embedded" |
| 14 | on-disk IREE = our fork, not stock | SOLID | `git -C iree_bar log` → commits `8c4ee4ad08`, `05e56f2bd1` (ours) | "stock IREE has no Zephyr/Gemmini; the fork's are our commits" | crediting IREE with our Zephyr/Gemmini work |

---

## 11. The defensible thesis (one paragraph)

Merlin is a **lightweight, MLIR-class compiler whose default output is C on a lean runtime it owns**,
aimed at **embedded / RTOS / bare-metal and custom-accelerator targets** that the heavyweight MLIR stacks
(Triton, XLA, IREE) do not treat as first-class, and that the kernel libraries (XNNPACK, OpenBLAS) only
reach by hand-writing per-ISA kernels. Its distinctive workflow is **new-target bringup** — a contract +
out-of-tree-package boundary + an L0–L5 certification ladder — paired with **deterministic mining of expert
kernel libraries into general, certified compiler capabilities** (one credited anchor: ~7.9× on an isolated
RVV kernel, explicitly *not* a system speedup). The two closest honest competitors are **EXO** (lightweight,
C-emitting, co-design-oriented — closest on *philosophy*) and **ExecuTorch** (whole-model + lean runtime +
a real WIP Zephyr path — closest on *embedded-whole-model*); we differ from both not by "caring about
embedded" or by a lighter runtime, but by being **MLIR-class with C as the default output, with first-class
RISC-V/RVV + custom-accelerator bringup (contract + out-of-tree boundary + L0–L5 certification ladder) and
AI structurally gated inside the flow** — a combination none of the eight source-verified projects has. On
the AI side, CIRCT-grounded advisory feedback is **measured (N=1)** to give the cheapest, single-round
bring-up of four arms (`merlincirct_abc9`); what remains **roadmap** is the N>1 ablation to fix a stable
magnitude, and automated, bidirectional HW-SW co-design. We claim the former as a one-run result and the
latter as future work — never beyond what the data supports.
