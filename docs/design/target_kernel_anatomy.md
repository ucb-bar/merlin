---
title: "Design: a target's kernel anatomy, and the slot that was missing"
kind: design
status: draft
owner: targetgen
last_verified: 2026-08-11
related: [lowering_pipeline, target_agnostic_core, target_onboarding, triton_frontend]
code_refs:
  - merlin/python/merlin/runtime/backends/base.py
  - merlin/python/merlin/runtime/backends/muon.py
  - merlin/python/merlin/runtime/backends/gemmini_codegen_mlir.py
  - merlin/python/merlin/targetgen/rtl/muon_introspect.py
  - merlin/tests/targetgen/test_radiance_backend.py
---

# A target's kernel anatomy, and the slot that was missing

## The question

One accelerator target descended the staged pipeline all the way to certified RTL execution. Another
descended to a command buffer and stopped. The pipeline, the dialect layer and the oracles were the
same in both cases, so "what exactly is missing for the second one" was not answerable by reading
either target's Merlin package — the difference lay outside them.

The way in was to read how each hardware project's **own kernel corpus** is organized, on the
assumption that a corpus written by the hardware's authors is the most honest available statement of
what programming that hardware requires. The systolic corpus was read at its upstream pinned commit
(`gemmini-rocc-tests` `7c540b3`, verified stock — it is an ancestor of the local checkout, which
carries later MX work on top); the SIMT corpus is `radiance-kernels`.

## Eight slots

Both corpora decompose the same way. The slots are the invariant; only their content differs.

| # | Slot | Systolic corpus | SIMT corpus |
|---|---|---|---|
| 1 | config facts | `gemmini_params.h`: `DIM`, `ADDR_LEN`, `BANK_NUM`, `BANK_ROWS`, `ACC_ROWS`, `elem_t`/`acc_t` | `VX_config.h`, the `RadianceSharedMemKey` the RTL is elaborated from |
| 2 | op vocabulary | `k_MVIN`/`k_PRELOAD`/`k_COMPUTE_PRELOADED` + one macro per instruction over `ROCC_INSTRUCTION_RS1_RS2` | `mu_intrinsics.h` / `vx_intrinsics.h`, one inline-asm helper per custom instruction |
| 3 | **dispatch** | one hart issues a blocking stream: flush → config → mvin → preload → compute → mvout → fence | `mu_schedule(entry, args, num_warps)` spawns a warp grid; the body is a function of the warp/thread id |
| 4 | staging | `mvin` into the scratchpad address space, accumulator tagged in the high address bits | `sw.shared` / `__shared`, into a banked scratchpad |
| 5 | tiling | `tiled_matmul_auto` with `tiled_matmul_total_spad_rows`/`_acc_rows` sizing tiles **from slot 1** | `mxgemm_lib.hpp` plus per-warp chunking |
| 6 | reference | `matmul_cpu`/`conv_cpu`, in the same header as the kernels | committed `expected/` arrays |
| 7 | pass protocol | `gemmini_testutils.h` compare + cycle counters | `kernel_verify.h`: hart-0 verify then a `tohost` code |
| 8 | build | `Makefrag` + a `tests =` list, baremetal and linux flavors | per-kernel build to `.mu.o` → device ELF + fused SoC ELF |

Merlin already implemented seven of the eight for the SIMT target, all target-agnostically: config
facts through the package's own `derive_facts.py`; the op vocabulary through the RTL-derived
`IsaModel` and `isa_transcode`; dispatch through `muon.render_simt_runtime`, which reads `wspawn`,
`tmc` and the warp-id CSR out of the derived runtime ABI and emits them as `.insn` forms; boot and
link fork-free through `targetgen/fixed_format/{boot,link}.py`; reference, protocol and oracles through
`reference_outputs`, the arc readback, cyclotron and VCS.

The missing slot was the **tile body** — the per-warp computation itself. `render_simt_runtime` had
always taken it as a `worker_body` string, and the only strings ever passed were hand-written test
fixtures. Nothing turned a command buffer into one.

## Where the body belongs

`runtime/backends/base.py` already resolves a backend from the target contract's `plugin.backend`,
loads it by file path, and lets it self-register. So the body's emitter belongs in the **target's own
package**, and the core needs no edit to gain it — which is also the seam the in-core
`gemmini_codegen_mlir.py` should eventually be evicted onto, so filling this slot in the package
moves that eviction forward instead of adding a second in-core emitter.

The emitter consumes **Merlin's abstract opcodes** (`MATMUL`, `COMMIT`, `VECTOR_MAP`), never a
target-dialect op name. For this target that is load-bearing rather than stylistic: its dialect is
hand-authored and prototype-grade, so an emitter written against those op names would be certifying
names a person chose. The opcode set is Merlin's own and is what a generated dialect would lower to
as well, so the emitter survives the dialect being replaced.

## The line not to cross

Slot 3 is the one that must not be generalized from the systolic target. Its kernel *is* a single
hart's sequential instruction stream; the SIMT kernel's body is a function of the warp id over a
spawned grid. Generalizing the systolic emitter's shape would bake "one instruction stream" into the
abstraction, and `must_map_to_warps` could then never be discharged by anything.

Slot 5 is the transferable one. "Size the tile from the derived capacity facts" is the same idea for
both; only the fact differs (`BANK_NUM * BANK_ROWS` there, scratchpad capacity here).

## A capacity fact that was an address window

Reading slot 1 side by side surfaced a live defect. The SIMT package derived
`resident_storage_bytes` as `1 << SMEM_LOG_SIZE` from the kernel headers — but the same header uses
that value as the base of the *next* aperture (`IO_BASE_ADDR = SMEM_BASE_ADDR + (1 <<
SMEM_LOG_SIZE)`). It is the address window reserved for shared memory, four times the memory that
exists. The systolic parameter header keeps the two apart by construction — capacity in
`BANK_NUM`/`BANK_ROWS`, address width in `ADDR_LEN` — and its tiling helpers size against capacity.

The capacity is now derived from the RTL config the hardware is elaborated from, following the chain
in source: the config class → its cluster's `smemConfig` → that key's `size`/`numBanks`/`numWords`.
That yields **131072 bytes** (4 banks × 64 B/row × 512 rows), corroborated independently by the
target's own `MU_SMEM_SIZE_BYTES`; a disagreement between the two is refused rather than resolved by
preference. The 512 KiB value is retained under its true name, `smem_aperture_bytes`.

Nothing had consumed the wrong number yet, because the slot that would have tiled against it did not
exist. It would have over-allocated the scratchpad by 4× the first time it did.

## The blocker: the oracle cannot reliably witness a kernel's final stores

The first diagnosis was that `isa_transcode` rejected MISC_MEM, so no fork-free kernel could contain a
`fence` — and since this hardware's own runtime fences at every synchronisation point (`mu_fence()` is
literally `asm volatile("fence")`), that looked like the whole story. The transcoder now re-maps
MISC_MEM (a two-line change: a standard RISC-V opcode value joining the existing I-type set, still
gated by each target's derived opcode table). It was necessary and it was not sufficient.

Measured on the arc oracle, one kernel shape, varying only what follows the store:

| kernel | result |
|---|---|
| body store, then a tail store, no fence | **both land** |
| body store + `fence`, then tail store + `fence` | tail store **lost** |
| body store + `fence`, no tail | even the store **before** the fence lost |
| body store alone, no fence, no tail | **lost** |

Two things are visible there, and neither belongs to the emitter: **a store is unreliably recovered
when little or no execution follows it**, and **nothing after a fence executes at all**. Whether the
fixed-format encoding of MISC_MEM's ordering bits is not what this decoder expects, or the model does
not implement the instruction, is not decidable from the compiler side.

What *is* established on RTL: the contraction is bit-exact. Reading the accumulator back gave
`[192, 192, 64, 64, 192, 192, 64, 64]`, matching the independent integer reference exactly, with the
operands correctly in device memory. The arithmetic and the operand plumbing are right. What is not
reachable is proof that the kernel *finished* — which is a statement about the oracle, not the kernel.

The obvious workaround is refused: padding the tail with filler stores until the real one happens to
become visible would make the grade pass by exploiting the very race that makes the grade meaningless.

The end-to-end test is therefore a **strict expected failure** rather than a skip. A skip would claim
"the oracle is unavailable", which is false — the oracle runs, and grades the accumulator correctly.
A strict xfail states a named boundary and fails loudly the moment it moves.

### A note on mechanism, and on two of my own wrong diagnoses

This section previously asserted that the SIMT scaffold *loses the spawning warp's tile at full width*,
on the strength of three bodies all returning element 0 as zero at 8 warps and all being exact at 4.
That measurement stands. The mechanism does not: element 0 belongs to the manager warp, which is
precisely the warp whose store is followed by almost no execution — so the same table is explained by
the visibility behaviour above, with no spawn defect at all. Two candidate causes fit the evidence and
separating them is work on the model.

Recorded because it is the more useful lesson: two successive mechanisms were inferred from correct
measurements and both were wrong. The measurements are reported as measurements, the boundary is named,
and the guess is left out of the places a reader would trust it.

## What the SIMT arm claims

The backend emits the tile body, builds it fork-free, and grades it by reading the output buffers
back from device memory on the target's RTL-derived arc model. Reading memory rather than a console is
not a preference: a print from a SIMT kernel interleaves across the lanes of a warp.

Claimed today: **the emitter produces a buildable per-warp kernel from a command buffer, and the
contraction it emits is bit-exact on the target's own RTL-derived model.** The whole-kernel grade is
blocked on the fence gap above, and is encoded as a strict expected failure rather than asserted.

Not claimed, and not to be written up as such:

- **not a tensor-core result.** The computation runs on the cluster's base integer ISA. The MX/tensor
  datapath's op encodings are not derived, and the command-buffer ABI is integer-only regardless.
- **not lane-level SIMT.** Work is partitioned across warps, not across the lanes within a warp;
  the thread-mask and lane-id ops are not derived here. The `must_map_to_warps` obligation is
  discharged; SIMD width is left on the table.
- **not a certification of the package's dialect.** That dialect is hand-authored, `status:
  prototype`, `requires_human_review: true`. This result is about the command buffer.

## Refusals, and why each one is a refusal

Each of these has a plausible approximation that would build, run, and grade green while doing
something other than what the command buffer said. Each raises instead, naming the missing
derivation.

- **`RES_PACK` / `MATMUL_RESIDENT`.** Staging an operand into shared memory needs the scratchpad's
  base address, and the derived runtime ABI has no such aperture — `IsaModel.aperture` raises rather
  than offering one. A "resident" pack realized in global memory has made nothing resident.
- **Requantization.** Merlin's integer requant rounding is not derived here, and an approximation
  differs from the reference in the low bit while looking right.
- **Float element types.** The command-buffer ABI is integer-only; a float path is a runtime-tier
  gap, recorded elsewhere.
- **Bias epilogues.** The bias operand is not in the command buffer.

## Two emission choices that are correctness, not style

**Operands are `volatile`.** With constant inputs, constant bounds and no aliasing, a compiler is
entitled to fold the whole contraction at build time and store the answer. That kernel grades green
having executed no arithmetic — the vacuity failure this repo has hit before. `volatile` forces the
loads and the multiply-accumulate into the image.

**Every command partitions work identically.** The first version strided the matmul by output row and
the commit by flat index, so a warp read accumulator entries another warp was still writing — a race
that can pass by luck. No barrier op is derived for this target, so the emitter cannot repair that
after the fact; the only safe schedule is one where each command's per-warp ownership is the same.
`test_every_command_partitions_work_the_same_way` is the regression.

**The LHS row is hoisted into a non-volatile local.** `volatile` is what stops the contraction being
folded at build time, but it also stops the compiler hoisting a loop-invariant operand — so the naive
form costs `M*N*K` volatile loads where `M*K` suffice, a 16× inflation on a 16-wide tile. Copying the
row into a local once keeps the anti-folding guarantee (the values still entered through a volatile
read, so they are still unknown at compile time) and removes the re-reads.

## How it failed first, and what that changed

The first RTL run returned an all-zero output buffer. Reading the intermediate values back located it
precisely: the operands were in memory correctly, the **accumulator was bit-exact**, and only the
committed output was empty. The cycle budget had run out between the contraction and the commit.

A partially written buffer compared against a reference is indistinguishable from a miscompile, so the
kernel now records its own completion — a sentinel written by warp 0 *after* it has waited for every
other warp to park, checked before any output is graded. A starved run now says it ran out of budget
instead of reporting a wrong answer. Two smaller facts came out of the same investigation: the model's
readback recovers whole cache lines (an exact-extent or unaligned span raises inside it, which would
have broken any output smaller than a line), and the model advances roughly a thousand cycles per
second, which is why the full `tl.dot`-floored 16×32×16 tile is a separately gated test rather than
the default one.

## A defect in the shared SIMT scaffold, found by using it at full width

`muon.render_simt_runtime` has warp 0 issue the spawn and then compute its own share inline. **When the
spawn count equals the hardware's full declared warp-slot count, that share never lands.**

Measured on the RTL-arc model with three unrelated bodies — a constant store, a 32-bit load from
initialized `.data`, an 8-bit load — the result was identical in each case: at 8 warps, element 0 (the
element warp 0 owns) came back zero while elements 1–7 were correct; at 4 warps all three were exact.
The declared count is right, and is properly derived — the RTL config sets `WithSIMTConfig(numWarps =
8)`, agreeing with the perf-model config — so this is not a bad fact. It is the scaffold, or its
interpretation of the spawn count, and it was invisible because its only other caller passes 4.

The backend therefore caps its spawn count and **reports both numbers** (`warps_used` alongside
`warps_declared`), so a run on half the machine cannot be read as a run on all of it. The cap is a
default rather than a ceiling: an explicit count still reproduces the defect for whoever fixes it.

Two smaller lessons from the same investigation. The scaffold's spawned warps start with a single
active thread — the vendor's own worker entry begins with a `tmc(-1)` to enable the rest, which the
scaffold does not do — so lane-level work needs that added before it can mean anything. And the
completion sentinel, added for cycle starvation, caught this too: it refused to grade rather than
reporting an all-zero buffer as a miscompile. Its message names both causes, because they are
genuinely indistinguishable from the outside.

## What remains

In dependency order, each item unblocking the next:

1. **Derive the scratchpad aperture** (base address) from the RTL config — that alone lifts the
   residency refusal, and residency is what the whole staged pipeline is built around.
2. **Derive the shared-address-space store**: the encoding fact already carries `address_spaces:
   {global: 0, shared: 1}` and the selector field, so a shared store is expressible — but a
   stock-compiled C kernel cannot spell one, so it needs the same treatment the SIMT control ops got.
3. **Lane-level mapping**, once the thread-mask and lane-id ops are derived.
4. **The tensor-core datapath**, which needs its op encodings and a non-integer command-buffer ABI.
5. **Evict the in-core systolic emitter** onto `plugin.backend`, so no accelerator has an in-core
   emitter. This touches a certified path and needs a re-certification run.

Found by this work, none of it this package's to fix, in the order it blocks things:

0. **`isa_transcode` could not encode MISC_MEM** — *fixed*, so a fence now builds for any target whose
   derived decoder declares that opcode.
1. **The oracle does not reliably recover a kernel's final stores, and a fence stops execution.** This
   is now the blocker for the end-to-end grade, it is a model-side problem, and it should come first.
2. **Full width (8 warps) does not produce correct results where 4 does** — measured with three
   unrelated bodies. Capped in the package, visibly, with the mechanism left open (see above).
3. **`render_simt_runtime`'s worker entry omits the `tmc(-1)`** the vendor's own worker begins with, so
   spawned warps run with a single active thread. Harmless for warp-level work; must be fixed before
   lane-level mapping means anything.
