# Phase 2 decomposition — gemmini, 2026-09-08

Phase 2 optimizes the **compiler**, for a fleet of workloads. Every row below names the instrument
that surfaced the opportunity and the measurement that decided it. `decomposition.json` beside this
file carries the machine-readable form, including the per-attempt deltas.

**48 attempts · 29 helped · 11 refuted · 7 blocked (5 live) · 1 unmeasured · 0 integrity problems**
**48 distinct instruments credited.** Scopes touched: global 18, build 11, encoding 7, host_lane 4,
local 3, frontend 3, inter_layer 2.

---

## The whole-model binaries

| model | object | frame (before → after) | const blob | image | correctness gate |
|---|---|---|---|---|---|
| **resnet50** | builds | **23,406,400 → 784** | 26,563,328 B (byte-identical to the bundle that ran on FireSim) | 0.25 GiB | all-1000-logit exact |
| **smolvla flow_denoise** | builds, 599,256 B | **99,897,984 → 496** | 505,299,072 B, **809/809 unpadded tensors byte-exact vs the capture** | **0.689 GiB, linked** | 10-step trajectory, contract-derived, digest-verified |
| **tiny_llama** | builds, 1,768,840 B | **48,976,448 → 9,152** | 1,298,638,208 B planned, 359/359 sources resolvable | 2.237 GiB → **needs the far-blob layout** | **blocked** — see below |

The SmolVLA ELF is `smolvla_flow_denoise_warm1_measure1.elf`, 505,790,296 bytes: 1,163 pointer
arguments, a 10-step recurrent session with three carried states, no unresolved symbols. It is the
first non-ResNet whole-model gemmini ELF in this tree.

---

## What the instruments found, and what each cost

The five findings this round, each traced to the instrument that produced it rather than to a guess:

### 1. A recurrent session's carried state was placed in read-only memory

*Found by* `bundle_pack.plan` checked against SmolVLA's own `session_contract.yaml`.

The emitted command buffer marks a carried state input `read` — and it is right to, because within
**one** invocation it genuinely is a read. Only the capture's session contract knows the loop writes
the output back into it every step. So all three of SmolVLA's carries (`prefix_kv_cache`
2,314,240 B, `flow_state` 6,400 B, `timestep` 64 B) landed in the const blob, and a 10-step session
would have written to `.rodata`.

Each carried input now keeps a `seed` row in the const blob and gains a mutable working copy that
the ABI pointer targets. Const bytes are unchanged; the mutable arena grew 132,144,576 → 134,465,280.

### 2. A latent pointer-ordering bug, exposed by fixing (1)

*Found by* the reclassification making two orders disagree.

The harness built its argument list as `const + mutable`. That equals the ABI's declared order only
while every read argument precedes every write one. Moving one carried input breaks it, and every
pointer after the first carry would have reached the wrong parameter. The plan now records
`abi_order`; all 1,163 pointers were re-derived independently from the command buffer and agree.

### 3. `reference_sha256` was in the schema and verified by nothing

*Found by* the contract's own `correctness` block, which nothing was reading.

A tolerance and a reference chosen by the grader is not a gate. The contract already declares
reference kind, keyed array, output index, step count and a digest. All eight declared digests in
the tree now verify.

> **A correction worth recording.** My first digest subject was the `.npz` file, and it reported all
> four captures as corrupt. That was the measurement, not the tree: an `.npz` is a zip whose framing
> carries timestamps. The subject is the keyed array's contiguous float32 bytes, established by
> agreement with all eight declared digests rather than chosen.

### 4. Trajectory grading needed a per-step reference slice

A gate reading `merlin_reference[0..n)` for every step passes a **stalled carry**; one reading only
the last step misses a divergence that starts mid-session. The rendered harness was compiled and run
against four mutations — a divergence from step 2, a stalled carry, a NaN, and every step graded
against step 0 — and fails each with no metric published.

Correctness is established on the warm session's retained per-step outputs and graded **after** the
counter window closes, because the gate prints per step and console output inside a bracketed window
is charged to the program. The warm and measured invocations remain the same body, and the re-seed
between them is what makes that true.

### 5. The stack frame — which refutes this plan's own "Step 3c is already done"

*Found by* the target's own kernel stack-frame preflight rejecting the emitted object.

The host lane hoists one `alloca` per intermediate with no reuse, so the frame scales with the
model -- and the fleet measurement below is what settles whether that is one model's problem:

**Every model in the fleet was over budget, and every one now fits.** Each `before` is
`clang -fstack-usage` on the emitted object; each `after` is the same measurement on the rebuilt one.

| model | allocas bound | frame before | frame after | over budget by | arena (`.bss`) |
|---|---:|---:|---:|---:|---:|
| resnet50 | 66 | 23,406,400 B | **784 B** | 357x | 23,408,768 B |
| lstmnetvit | 449 | 25,956,928 B | **384 B** | 396x | 25,965,952 B |
| tiny_llama | 1,636 | 48,976,448 B | **9,152 B** | 747x | 48,976,384 B |
| smolvla | 1,534 | 99,897,984 B | **496 B** | 1,524x | 99,898,560 B |

Zero allocations were refused on any of the four. The annotated `merlin.host_storage_bytes` values
account for 100.0% of SmolVLA's demand and 99.95% of tiny_llama's, so the attribution is complete
and the declared sum is the frame.

> **This overturns the plan's own "Step 3c is already done", and the correction matters.** That
> conclusion rested on a receipt reporting a 816-byte ResNet-50 frame, read as evidence the
> host-lane alloca hoisting had been fixed and the 23.4 MiB frame was a historical, bf62-only
> defect. Compiling the ResNet-50 emission in this cache measures **23,406,400 bytes**. The two
> numbers are not a regression -- they describe *different emissions of the same model*. A receipt
> is evidence about the bytes it was measured on, and citing it for another emission is citing it
> for another program.
>
> The practical consequence is the opposite of what the plan concluded: this is a **fleet-wide
> compiler defect**, not a property of the largest model, and it was blocking every whole-model
> binary rather than one. It was only ever going to be found by measuring, because the declared
> annotation sum is an upper bound and reading it would have proved nothing.

The arena runs only as a repair on a **measured** failure and re-measures the rebuilt object, so an
already-fitting build takes the same path it did before and is byte-identical — which is what keeps
the one bundle known to have run correctly on hardware a valid acceptance test.

---

## Where NOT to optimize — the refuted branches, kept so they are not re-walked

Eleven rows carry a `refuted` verdict. The four from this round:

- **Byte reuse will not fix the stack frame.** The obvious remedy is to colour the allocations and
  share bytes, as `arena_bind` does for the heap. On **every** model the largest *single* allocation
  already exceeds the whole 65,536-byte budget (smolvla 489,000; tiny_llama 1,024,000; lstmnetvit
  3,179,520; resnet50 3,326,976), so no colouring however good makes any of the frames legal.
  And an `alloca` has no `free` (measured: 0 `llvm.lifetime`, 0 `llvm.stacksave` in the emitted
  module), so no two live ranges are provably disjoint. Moving the storage is the fix; sharing it is
  a separate and, on this IR, unprovable one.

- **An image's medany span cannot be read off the symbol table.** `medany_span` reported a FAULT on
  SmolVLA's 0.689 GiB image. TLS symbols carry offsets *within* the thread block, so `buf.2` at
  `0x40` made the span look like 2.689 GiB. The span is the `PT_LOAD` virtual footprint.

- **GSIM calibration cannot narrow the cost band** (earlier round). Width is `peak / slowest_rate`,
  a per-class constant, and `rates_for` keeps the minimum — a new measurement can only widen it. The
  declared compute-class split did the work instead: median band width 2816× → 211.5×.

---

## Blocked outside this lane

1. **Attention quantization.** All 66 off-unit contractions are f32 attention `batch_matmul`, and
   the captures carry no attention quantization parameters at all. Quantizing them in the compiler
   would mean inventing scales. Fix the export.
2. **`ATTENTION_PV` pricing.** Zero programs emit the opcode; only a compiler change can price it.

   A caveat recorded rather than left implicit: ResNet-50's *shipped* bundle -- the one that ran
   correctly on FireSim -- was built by a path whose receipt reports 816 bytes. The arena repair
   fires only on a measured over-budget frame, so that path is untouched and its bundle still
   reproduces byte-identically. Two emissions of one model, both accounted for.
3. **Epilogue attach + readout narrowing.** Every readout in all four whole-model buffers is
   `output_dtype: i32`, the full-width path, and the RoCC model writes the raw accumulator there —
   discarding the scale and activation it computed. Another session's active lane, and their own
   refusal ledger names the blocker (`accumulator_scale_is_not_proven_identity`).
4. **tiny_llama's independent golden.** `make_w8a8_independent_golden.py` refuses because that
   loader is session-shaped (prefill+decode) while the capture on disk is a single forward. The gate
   correctly refuses both available references by name rather than grading against the wrong one.
5. **One FireSim run with the full 8-counter bracket.** Everything downstream is built and tested;
   `attribution` refuses the existing 3-of-7 reading by design. The FPGA is in use.

---

## The tiny_llama layout blocker, quantified

| region | bytes | GiB |
|---|---:|---:|
| const blob | 1,298,638,208 | 1.209 |
| mutable arena | 1,052,254,208 | 0.980 |
| static arena (`.bss`) | 48,976,384 | 0.046 |
| code + other rodata (allowance) | 2,097,152 | 0.002 |
| **total** | **2,401,965,952** | **2.237** |

Past the ±2 GiB PC-relative reach — a **FAULT**, not a warning: the image would link and
mis-address. The remedy is the layout the repo has already built twice (`model_link.ld`, validated
on spike; `firesim_baremetal.ld`): the blob at a fixed absolute address reached by a compile-time
literal, never a relocation. With it, only the near region (mutable + static arena + code, 1.03 GiB)
has to be reachable.

`PackPlan.projected_image_bytes` now computes this **before** a link, and the harness renderer
refuses an unreachable layout naming what to do about it.

> Corroboration, not coincidence: `model_link.ld` reserves 256 KiB of stack because *"the lowering
> promotes static intermediates to stack allocas for this target, so the demand scales with the
> model rather than with the harness."* An independent part of the repo documented the same defect
> as a sizing problem and worked around it. The static arena removes the cause.
