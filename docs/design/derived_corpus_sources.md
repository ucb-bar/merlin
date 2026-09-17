---
title: "Design: deriving the corpus from the spec, the RTL facts and the workloads"
kind: design
status: current
owner: core
last_verified: 2026-09-05
related: [capsule_phase_split, perf_corpus_scope_gap, derived_capsule_axes]
code_refs:
  - merlin/python/merlin/targetgen/conformance.py
  - merlin/python/merlin/targetgen/conv_geometry.py
  - merlin/python/merlin/targetgen/applications.py
  - merlin/python/merlin/targetgen/corpus_synth.py
  - merlin/contract/claim_models.yaml
---

# Deriving the corpus from the spec, the RTL facts and the workloads

The corpus is meant to be derived: a target's obligations come from its own sources — the ISA/spec,
the RTL facts, the capability manifest, and the workloads we declare relevant — and a synthesizer
turns them into capsules. Half of that was true. The requirement was derived from two of the four
sources (manifest ∩ captures); the spec contributed individual goldens and nothing to the
requirement; and no workload reached the corpus at all.

This note records what closed, what it measured, and the three ways the old arrangement was wrong in
a direction that read as success.

## The claim set had to shrink before anything could be derived

`claim_models.yaml` holds models out of derivation so the corpus is not built from the model it is
then said to generalize to. All four models were held out — and that left the requirement with no
model-shaped evidence at all. A corpus derived from nothing model-shaped cannot contain a layer, a
chunk, or a model, which is exactly what was measured: `workload_spec.applications` was undeclared on
every target, every spec on disk read `declared_applications: 0`, and the corpus-wide
`generalization_axis` census showed `application: 0`.

The set is now the two models the experiment is **graded** on — `resnet50_v1_5` and `tiny_llama` —
and `smolvla` and `lstmnetvit` are what the corpus is **built from**. Narrowing the claim is the
honest direction: it makes the claim smaller and the evidence real, rather than leaving a wide claim
resting on a corpus that had never seen a model.

The guard is in `check_conformance_coverage._applications`, and it **raises** rather than filtering. A
target declaring five applications and deriving from four hides the difference in the one place a
reader would never look.

## What the workloads contribute: shapes the corpus was nowhere near

The application axis groups a capture's regions by what the compiler must do with them — family,
dtype, alignment, memory regime, rank, geometry class — and emits, per occupied class, an L3 capsule
at a size a certification affords plus an L2 capsule at the application's true shape naming its
`extends` sibling.

Turned on, it yields **46 required capsules on gemmini and 28 on radiance**, at shapes like
`(1024,64,1024)`, `(8,72,345)` and `(96,64,512)` — against a corpus whose members were 16×16×16.

Twelve classes are **refused**, each naming its class and the limit it exceeds (operands past the
65,536-element calibration). A named refusal is the correct output: an unaffordable behaviour must
look different from an absent one.

**Atlas yields zero, and the reason cascades further than expected.** No certification history means
no cost model, which means `size_class` can show no size affordable, which means every class is
refused. Atlas's problem is not an undecided phase split; it is that one L3 sweep of its covering set
gates its whole derivation. The refusal says so.

## What the windows contribute: the shape between two half-covering members

No capsule on any target declared a non-zero padding, so a lowering that loses the padding identity
was wrong only in border rows nothing computed.

The geometry cannot be read off the captured op. torch-mlir emits **im2col**, so a captured
convolution is a gather and a matmul and there is no convolution op and no `padding`/`stride`/
`dilation` attribute anywhere. `targetgen/conv_geometry.py` recovers it structurally instead — stride
and dilation from the affine coefficients of the gather's input map, the kernel from the iteration
space, padding from the producer `tensor.insert_slice` offsets — and yields **ten distinct windows**,
including a transposed convolution with asymmetric padding that no hand-written entry anticipated.

The finding that justifies the axis: the corpus presented `k2x2/s2x2/pad0`, `k3x3/s1x1/pad0`,
`k3x3/s1x1/pad1` and `k3x3/s2x2/pad0` — **a padded convolution and a strided convolution, and never a
strided-AND-padded one**, while the captures demand `k3x3/s2x2/pad1`. The hand-authored corpus has
`GC7_conv2d_pad_i8` and `GC8_conv2d_stride2_i8` as separate entries. The defect lived in the gap
between two members that each covered half of it, which is what a per-attribute corpus misses by
construction and a per-CLASS corpus cannot.

### Two ways this recovery was wrong before it was right

Both were caught by running it, and both produced a plausible-looking class list.

**The verification identity was the stride-1 special case.** Checking
`padded == (out-1)*stride + (kernel-1)*dilation + 1` drops every strided convolution: at stride 2 the
window legitimately fails to reach the last padded row. The corpus would have gained unstrided members
and still had no strided one — the gap the module exists to close, reintroduced by its own check.

**"A kernel is smaller than its output" is false.** A real capture holds a 4×4 kernel producing a 2×3
output; the heuristic read it as a 2×3 kernel with dilation 4, a different convolution from the one
the model contains. The affine map is symmetric under swapping the pair and the output identity does
**not** separate them — both assignments satisfy it. What separates them is the shape the gather is
reshaped into: im2col produces `[Cin·prod(kernel), N·prod(output)]`, so `K` names the kernel side
unambiguously. A gather whose reshape chain does not reach a rank-2 value yields nothing rather than a
guess.

**An unreadable padding producer is UNKNOWN, never zero.** One capture's first convolution is padded
by an `aten.index.Tensor` gather — a reflection pad, whose offsets are nowhere to read. Reporting
`pad0` would claim there is no padding where there is some, *and* claim the zero identity for one that
reflects. The class carries `padUNKNOWN`, and its synthesized member says in `source_reference` that
it tests the window and its stepping and asserts nothing about a padding identity that is not zero.

## What the spec contributes, and how it failed open

The spec was a golden source for individual capsules and contributed nothing to the requirement —
3 of 536 capsules, with zero occurrences of `specir` in `conformance.py`. Before it can be a
derivation source it has to stop failing open, because deriving from a source that fails open is
worse than not deriving from it.

**The op token was accepted and never read.** `gemmini:op.matmul`, `gemmini:op.isa_flush` and
`gemmini:op.TOTALLY_BOGUS` returned a byte-identical command buffer, golden and coverage goal. A
typo'd or renamed ref silently produced a matmul capsule under whatever name the ref carried. The
token is now checked against the gen's own declared `spec.op` nodes, which makes the emitter's
indifference unreachable rather than pretending it is fixed.

**Coverage goals were not op-scoped**, so a matmul capsule carried a transcendental's test intent —
the wrong oracle and the wrong tolerance, stated with the same confidence as the right ones. The
`covers` linkage does resolve, through `refs`; what had disabled it was a guard reading
`isinstance(table, dict)`, and xDSL's `DictionaryAttr` holds an `immutabledict`, which is a Mapping
and **not** a dict subclass. A guard that cannot pass is the same defect as a check that cannot fail.
Scoped, `op.matmul` goes from six coverage declarations to two.

## Axes that were derived, recorded, and measured by nothing

`conformance.uncovered()` measured cells, composition, geometry, scope, host_only and memory. It did
not measure `host_lane` — 12 obligations on one target — nor `epilogue` — four. Both were emitted into
every spec and read back nowhere. A requirement that is derived, written down, and never checked is
indistinguishable from one that is met.

Both are measured now. The host-lane reader is deliberately strict about what counts as evidence: the
witnessing capsule's own family and dtype must be the pair and its whole program must land on the host
lane, because every routing-shaped capsule *contains* a host stretch and would otherwise witness
everything.

**It immediately found a spurious obligation on all three targets.** A whole model is entered with a
token-id vector — i64 — and computes in none of it; it declares its arithmetic separately at
`operation.attributes.dtype`. Reading `inputs[]` as operand dtypes minted a `contraction/i64`
host-lane obligation for work no capsule performs and no hardware was ever asked to take. Measured: 20
of the 24 capsules declaring a compute dtype disagreed with their own `inputs[]`, and every one was
`kind: model`. A declared compute dtype now wins outright.

## Two sidecar lines that made a whole path dead on arrival

`verify.counterexamples` writes `profiles/<target>.smt.yaml`. `load_profile` read exactly three
filenames and that was not one of them, so every counterexample the solver found went to a file
nothing opened — consistent with no `*.smt.yaml` ever having been committed. The module's own
docstring asserted the opposite, that `load_profile` "already merges `profiles/<target>.*.yaml`
sidecars". There is no glob; it is a hardcoded chain, and the docstring now says so.

Beside it, a live and wider bug: `profile_targets()` globbed `*.yaml` and returned `path.stem`, which
strips only the **last** suffix — so `gemmini.synth.yaml` yielded the target `gemmini.synth`. Six real
targets came back as twelve, and `main()` uses that list as the default when `--target` is absent, so
a bare run generated phantom corpora from profile fragments. Excluding `.hidden` by filename was the
same bug avoided one case at a time; the rule is now structural.

## Where this leaves the split

The functional corpus is the foundation and is unchanged in kind: small members, one certified anchor
per obligation. What changed is that phase 2 now has something to be derived *from*. The application
axis emits its members in L3/L2 pairs where the L2 member names its `extends` sibling, so a large
member resting on a guarantee is distinguishable from one resting on nothing — and
`check_cert_affordability` verifies that `extends` against the sibling's own results rather than
treating a non-empty field as a remedy.

Not yet done, and not to be read as passing: atlas's certification sweep (which gates its entire
derivation), the claim-model build repairs, and the chunk capsule kind — there is still nothing
between a one-op `model_slice` and a whole `model`, so "layers 3–7 of this model" remains
inexpressible.
