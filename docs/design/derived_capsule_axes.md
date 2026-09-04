---
title: "Design note: deriving the axes a conformance cell cannot express"
kind: design
status: current
owner: core
last_verified: 2026-09-03
related: [capsule-bench, conformance]
code_refs: [merlin/python/merlin/targetgen/conformance.py, merlin/python/merlin/targetgen/corpus_synth.py, merlin/python/merlin/targetgen/store_probe.py]
---

# Deriving the axes a conformance cell cannot express

The capsule corpus is derived from each target's own facts: its RTL, its capability manifest, and the
models it is for. The requirement is written in **conformance cells** — `(semantic_family, dtype,
tile_alignment)` — and for a long time a target scoring 12/12 on those cells was taken to mean the
corpus exercised what the compiler had to get right.

It did not. This note records what the cell vocabulary is blind to, how each blind spot was found,
and what now derives instead. Every number here is measured; where something is still unknown the
note says so rather than rounding it into a claim.

## The shape of the problem: a requirement cannot ask what its vocabulary cannot say

Four axes were added, and all four failed the same way before they existed: the corpus reported full
coverage of a requirement that never asked the question.

### 1. A dtype constrains which extents can exist at all

A block-scaled (microscaling) format carries one E8M0 exponent per whole run of K, and sub-byte codes
are addressed in nibble pairs. On a target whose tile edge is 16, that makes the datapath's own
granularity 32 — *larger than the tile*. Every legal extent is then a whole multiple of it, so
`partial` ("ragged by one") and `sub_tile` ("barely occupied") name shapes the hardware cannot
execute.

The requirement demanded them anyway, and 15 capsules failed to generate against cells nothing could
ever cover:

```
SY_contraction_mxfp4_sub_tile: MX golden needs K to be a whole multiple of the 32-element
block-scale group; got K=4 ... the scale stream would cover only 0 of 4 K elements.
```

`corpus_spec.shape_quantum` now derives the granularity — the reduction quantum from the target's own
declared scale group, the row quantum from the packer's nibble pairing — and the alignment axis
narrows per dtype. A format that cannot spell a ragged extent is no longer asked for one, and the
narrowing is recorded in `diagnostics.alignment_axis_narrowed_by_dtype` rather than being silent.

### 2. `partial` was unreachable for every family that does not contract

The ragged class ragged one axis: the output column `N`. A unary op's operand is `M x K`, so `N`
never reached it. Measured: `SY_elementwise_map_f32_partial` came out `tensor<16x32xf32>` — byte for
byte the *aligned* capsule under a partial name. Nine cells across `elementwise_map`,
`normalization`, `reduction` and `softmax` were required, emitted, built, and classified `aligned` by
the cover. The class was uncoverable for every family that is not a contraction.

`extents_for` now rags `K` as well. `K` is the one axis every family's primary operand carries.

### 3. A broadcast axis decided the whole capsule's occupancy class

An rmsnorm declares a `1 x K` gain vector. The alignment classifier read every extent of every
declared tensor, so that `1` — a broadcast parameter nothing tiles — dragged a 16×32 problem into
`sub_tile`. `normalization/bf16/aligned` was required, built, and reported uncovered while the
capsule covering it sat on disk. Extents of 1 are now excluded: there is no tile for a degenerate
axis to partially occupy.

### 4. Aspect ratio — the axis with the largest gap

A cell puts a 448:1 tall-skinny convolution and a 16×16 projection in the same box. Alignment is
measured *relative to the tile edge*, which quietly makes every capsule tile-sized, and a tile-sized
capsule is square. Running the contraction census over a real capture:

```
wide_skinny     M=128  K=2304 N=256000  x1632   99.5% of all contraction MACs
tall_skinny     M=1024 K=256  N=128     x 373
squareish_gemm  M=196  K=1024 N=256     x  33
odd_tail_heavy  M=196  K=256  N=768     x   8
gemv_like       M=1    K=256  N=1000    x   1
```

Five classes carry the work; every synthesized capsule was `projection_like`, which is not among
them. The largest convolution capsule in the corpus wrote 1,024 output elements against a real
patch-embed's 150,528.

`geometry_axis` now derives the classes a target's captures present and emits one capsule per
occupied class at the **highest-MAC-mass real shape** in it.

## Why this is affordable: a class is a ratio, not a size

Certification cost is measured and it tracks **written output**, not work:

```
seconds = 0.20509 * output_elements ^ 1.0782      (log-log r2 0.9976)
```

A ViT patch-embed (196×768) is 21.8 hours cycle-accurate. So a geometry capsule cannot simply be the
real shape. But the thing that defines the class is the ratio, so the representative is scaled down
by the smallest common divisor on all three extents that fits what the target's own operand store
holds, and then **re-classified** — a capsule is emitted only if the scaled shape is still in the
class it was drawn from. `K` moves with the rest, and every extent has a floor of
`min(original, tile_edge)` so the result stays non-degenerate.

Two things that fell out of getting this wrong, both worth keeping:

**Bound the largest tensor, not the written output.** Certification cost follows the output, so
reduction depth is nearly free to certify — but the *golden* has to synthesize and write every
operand element, and a skewed ratio has a colossal operand behind a small output. Bounding the output
alone admitted an `11 x 2304` by `2304 x 23272` capsule whose output is 255,992 elements and whose
weight is **53.6 million**: the generator held 14.7 GB and had produced nothing after 27 minutes.
`cert_cost.MEASURED_MAX_OPERAND_ELEMENTS` already said this in as many words; this axis was the
caller that did not listen.

**A ceiling without a floor produces a capsule that proves nothing.** Searching only downward, the
1:2000 class scaled to `1 x 1 x 31` — still `wide_skinny` by the taxonomy, and a contraction with no
reduction and one output row.

Where no scaled shape is both representative and buildable, the class is reported by name. On every
target measured, `wide_skinny` is unreachable and says so.

## Tier follows size, and the tier must exist

A capsule too large to certify is capped to the loop tier and names the certified sibling it rests
on, so a large perf-facing shape is an *extension* of a functional guarantee rather than a substitute
for it. The cap was to the literal `"L2"`, which broke on a target that declares only `[L3]`:

```
'SY_geometry_gemv_like' caps its oracle tier at 'L2', which is not among the tiers this
target declares (['L3']); a cap onto a tier that does not exist would silently leave the
capsule demanding everything
```

The loop tier is now derived from the target's own adapters, and a target with no cheaper tier gets
the capsule uncapped with the reason stated.

## The rank axis, and the rung the evidence ladder did not have

A contract declared `contraction ranks: [2, 4]`. `is_eligible` therefore answered *"rank 3 not in
contraction legal ranks"* for every batch matmul — while the rewrite already took `(B,M,N,K)`, the
device builder already produced the `(M,N,K)` kernel, and the emitted shim already looped over `B`.
The execution machinery existed and the contract forbade its use.

A rank the contract omits removes those regions from the **ARR denominator**, which *raises* recall —
the direction that flatters us. On one capture they are 16 of 106 contractions and **8.5% of all
contraction MAC work**, which is the same 8.5% an earlier investigation recorded as "attention
generics the matcher never saw".

The four rungs of the evidence ladder all read what the *hardware declares* — ISA roles, instruction
classes, extracted RTL, unit intent. None could see a shape the *lowering* already handles.
`capability_derive._from_lowering` is the fifth: `device_shim` emits a 3-D entry for a `(B,M,N,K)`
signature on any device with a derivable tile edge — *"a batch is a LOOP, not a third tile axis…the
same kernel called B times with the leading offset advanced"* — so rank 3 asks nothing of the device
that rank 2 does not.

The rung supplies evidence and does not edit the contract; `_axis_findings` then reports the omission
as `missing_axis`, which is a mechanism that already existed and had nothing to fire on.

Making the capsule buildable took the other half: the only batched golden was block-scaled, so
`gemv_batched` was refused at every non-MX dtype. The integer, specir and SIMT engines each grew a
batched branch — a batched contraction is B independent ones, and every engine already knew how to do
one. Each slice draws its own salted operand, so a kernel that computes one slice and broadcasts it
cannot match.

One trap on the way: `op_for_shape` at rank ≥ 3 returned the first batched op in the pool regardless
of the family asked for, so a batched *contraction* was emitted to evidence `elementwise_map.batched`
and `movement.batched`. Invisible while every such probe was a reported hole; giving the op a golden
turned the hole into a wrong capsule, which is worse.

## The operand store, where the facts run out

The memory-mapping axis needs a target's on-chip operand store. Where the RTL facts carry a memory
list it is derived. Two targets do not have that.

**A SIMT device has no compute array at all.** Its facts carry a `simt` block instead, and its store
is reached one warp at a time: a coalesced access moves `lanes_per_warp` elements whatever their
width — exactly the role the array's column edge plays on a spatial device. Reading only `arrays`
reported "no row width" for a target whose own facts state it, and the axis reported 0/0 required
regimes over a 128 KiB shared memory. Deriving the width from `simt.lanes_per_warp` took that target
from **0/0 to 3/3**.

Capacity on such a store depends on the element type, so `operand_store` takes a dtype and the
callers resolve per capsule and per region. A subtlety worth recording: an *unlinked* element width
is not the same as a *lane-granular* store. Both lack `element_bits`, and keying the fallback on that
answered "16 elements per row" for a store whose row is genuinely unknown. Stores now carry an
explicit `lane_granular` flag that only the SIMT path sets.

**The other target has no memory list at all.** Its discovery finds 39 SRAMs and declines to classify
any of them. The tempting move — pick the plausible one — has already been made and refuted: the
matrix register file (65,536 elements) is not it, because a layer needing 73,728 runs unblocked.

`store_probe` asks the question the descriptor prescribes: *measure the extent at which the backend
actually declines, and declare the store that predicts it.* It grows one contraction's working set
against the target's own capsule path and brackets the boundary.

Its first verdict was wrong in an instructive way. It called any non-passing point a decline and
"refuted" both candidate capacities. The point had come back:

```
WRONG m32 k512 n512   278,528 elements  1431s   tier L2: does not compute the declared
                                                operation within tolerance
```

The device **accepted the layer, ran it, and answered wrongly.** That is a correctness boundary and
it is not evidence about capacity. Points now carry `pass` / `wrong` / `declined`, and only
`declined` brackets a store. What the probe actually established: the store holds **at least 73,728
fp8 elements**, the backend **miscomputes a 32×512×512 fp8 matmul**, and both VMEM candidates remain
**undecided rather than falsely refuted**. That target's memory-regime axis stays honestly 0/0.

## A change that was reverted, and why

The composition axis reported `UNDETERMINABLE` for every capsule on two targets, because
`device_build` cannot emit a linkable call for their transport. Relaxing that — on the argument that
the boundary is a DRAM address contract the harness honours, so the crossings are derivable — moved
composition from 1/5 to 5/5 and 3/6 to 6/6.

It was wrong and it is reverted. Being able to **classify** which side a region falls on is not
evidence the seam is **exercised**. This axis exists to say whether the corpus can prove a compiler
assembles work correctly *across* the boundary; on a target whose seam no path can emit, a capsule
labelled `H->A->H` is a claim nothing verified. The numbers would have improved by argument alone.
The honest figures are 1/5 and 3/6, and the remedy is to build the transport.

## Two defects this work surfaced outside the corpus

**The sandbox stopped being able to launch.** One `--ro-bind /dev/null` mask per answer surface is
right, but at 1,098 goldens the wrapped command string reached 159,158 bytes, and `execve` refuses
any single argument over 128 KiB. Every sandboxed agent invocation would have died with
`OSError: [Errno 7] Argument list too long: 'bash'` — an error that names `bash` and says nothing
about masks. `bwrap --args FD` moves the arguments into a 0600 file beside the workspace (not inside
it, so the agent cannot read the list that isolates it); the string drops to 1,210 bytes and stays
there however large the corpus grows. Isolation suite: 42 passed, same masks.

**A dropped entry was recorded where nothing wrote it.** The gradeability filter removed entries and
put them in a key the profile writer never emitted. Seven batched regions on one target were required
by the spec, counted in `n_entries`, removed, and then absent from the tracked artifact with nothing
anywhere saying so. Drops now land in the provenance the profile actually writes, alongside
`n_written`.

## Where each target stands

| axis | gemmini | radiance | atlas |
|---|---|---|---|
| conformance cells | 12/12 | 72/72 | 18/18 |
| shape geometry | 5/5 | 5/5 | 5/5 |
| memory regime | 3/3 | 3/3 | **0/0** |
| composition | 6/6 | **1/5** | **3/6** |
| host-only lane | 1/1 | undeterminable | 2/2 |

The three bold entries are real gaps with named causes: one target's RTL facts carry no memory list,
and two targets' host/device seam cannot be emitted by any path in this repo. None of them is
rounded away, and none is closed by argument.

## The rule underneath all of it

Every failure in this note has the same shape: **a check that cannot see what it checks reports
success.** A requirement whose vocabulary cannot express aspect ratio reports full coverage of a
corpus that is entirely square. A classifier that reads a broadcast axis reports a capsule as
uncovered while it sits on disk. A probe that cannot tell a miscomputation from a refusal reports a
capacity it never measured.

The remedy is always the same and it is never "add the number": make the derivation able to ask the
question, and make the absence of an answer look different from the answer.
