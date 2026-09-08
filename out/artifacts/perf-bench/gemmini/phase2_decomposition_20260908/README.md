# Phase 2 decomposition — gemmini, 2026-09-08

> The campaign-level analysis — where the budget went, what each
> instrument is licensed to conclude, and what none of it can decide — is at
> `../phase2_analysis_20260908/README.md`. This file is the change-level view.

Phase 2 optimizes the **compiler**, for a fleet of workloads. Every row below names the instrument
that surfaced the opportunity and the measurement that decided it. `decomposition.json` beside this
file carries the machine-readable form, including the per-attempt deltas.

**56 attempts · 35 helped · 13 refuted · 7 blocked (5 live) · 1 unmeasured · 0 integrity problems**
**56 distinct instruments credited.** Scopes touched: global 22, build 15, encoding 7, host_lane 4,
local 3, frontend 3, inter_layer 2.

---

## The whole-model binaries

**All three link.** Each `frame` pair is `clang -fstack-usage` before and after the static arena;
each blob was verified byte-for-byte against its capture, not merely sized.

| model | ELF | args | frame (before → after) | const blob | layout | gate |
|---|---:|---:|---:|---:|---|---|
| **resnet50** | (shipped bundle, ran on FireSim) | 393 | **23,406,400 → 784** | 26,563,328 B, byte-identical to that bundle | 0.25 GiB near | all-1000-logit exact |
| **smolvla flow_denoise** | **505,790,424 B** | 1,163 | **99,897,984 → 496** | 505,299,072 B, **809/809 unpadded byte-exact** + 3 pitch-padded row-exact | 0.689 GiB near | 10-step trajectory, contract-derived, digest-verified, **tolerance derived** |
| **tiny_llama** | **1,300,782,520 B** | 825 | **48,976,448 → 9,152** | 1,298,638,208 B, **358/358 unpadded byte-exact** + 1 pitch-padded | 1.028 GiB near **+ 1.209 GiB blob at 0x200000000** | per-token top-1 over 8 rows, execution-scoped |

SmolVLA's is the first non-ResNet whole-model gemmini ELF in this tree; tiny_llama's is the first
whose const blob could not be linked beside the code at all.

**tiny_llama's far blob, verified rather than assumed:** the linker placed it at exactly the address
the harness compiled as a literal, there are **zero relocations against the blob symbol**, and
`_init` sits 9,598 bytes from `_start` — inside the ±1 MiB JAL reach that the first link attempt
overran.

---

## What the instruments found, and what each cost

Nine findings, each traced to the instrument that produced it rather than to a guess.

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

### 6. A baremetal console has no float conversions

*Found by* the SmolVLA ELF's own output on Spike.

The target's `vprintfmt` implements exactly `c s d u x l`. A `%.9g` there does not degrade
gracefully: it prints the specifier **literally** and then mis-consumes the varargs, so every later
field on the line is garbage. All 16,000 value lines read `MERLIN_OUT 0 %.9g`, and the header claimed
`elements=-350469331`. The verdict itself was sound — the comparison is C arithmetic and
`bad`/`argmax`/`digest` use integer conversions — but no magnitude could be recovered, so the run had
to be repeated. Values now travel as IEEE bit patterns with the step on every line, and a
render-time scanner refuses any float conversion. It scans structurally: `"%.9g"` does not contain
`"%g"`, so a substring check passes exactly the specifier that shipped.

### 7. My own tolerance was arbitrary — the discipline was followed and the number was not derived

*Found by* the SmolVLA gate failing `bad=1600/1600` on every step.

`atol=rtol=1e-4`, declared before the run, which is the right discipline. Comparing the capture's
**own** two references to each other, **15,740 of 16,000 elements (98.4%) fail that tolerance** and
the worst disagreement is 5.36e-2. So the run's verdict said nothing about the device: a gate that
tight fails for any conforming datapath. This is the cos-0.484 incident in subtler form — there the
wrong *reference* was chosen; here the right reference with an impossible threshold.

`reference_spread` now measures that floor from the capture and refuses a tighter tolerance, naming
the number to use. Every pre-existing test moved onto a derived tolerance rather than having the rule
switched off.

> One thing the failed run did prove: `MERLIN_GATE_EXPECT` returned 8× `argmax=1141` and 2× `832`,
> exactly matching the reference's own argmax split at step 8. The per-step reference indexing is
> correct.

### 8. A language model's ranking is per token

One argmax over TinyLlama's (1, 8, 32000) = 256,000 logits agrees with the reference whenever the
single largest logit *anywhere* lands in the same place, and says nothing about the other seven
positions. Each declared row is now ranked separately. A compiled-and-run mutation that moves row 0's
peak while leaving the **global** maximum untouched fails with `disagreeing=1`; the old check passed
it.

### 9. A JAL scaling limit that link order decides

*Found by* the tiny_llama link failing `R_RISCV_JAL truncated to fit`.

`crt.S` reaches `_init` with a JAL (±1 MiB), and tiny_llama's kernel `.text` is **1,061,906 bytes on
its own** — 13 KB past the reach. So `_init` linked after the kernel is unreachable. Support objects
go first now. This happened to work for every smaller model, and it is a hard link error rather than
a silent one, which is the good outcome.

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
4. **tiny_llama's INDEPENDENT golden.** `make_w8a8_independent_golden.py` refuses because that
   loader is session-shaped (prefill+decode) while the capture on disk is a single forward, so the
   arithmetic itself cannot be decided from this tree. What *was* producible is the **execution**
   reference (`make_w8a8_golden.py`, no model repo needed): 256,000 elements, cos 0.9946 against the
   weight-only golden — and that gap is precisely the activation-quantization error the cos-0.484
   incident was about. tiny_llama's gate therefore answers "did the device reproduce the host
   compiler", with that scope declared in the gate itself, and its tolerance is tight *because* both
   sides are the same program. That is a real question and it is not evidence about the arithmetic.
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
