---
title: Functional native placement
kind: reference
status: current
owner: core
last_verified: 2026-10-19
code_refs: [src/merlin/targetgen/core_aten_admission.py, src/merlin/targetgen/native_payload_carriers.py, src/merlin/targetgen/native_quant_boundary.py, build_tools/scripts/make_bounded_phase0.py]
---

# Functional native placement

Phase 1 routes every case admitted by a declared primitive family and its
constraints to the device when the provider supplies a general proof and the
native construction detectors accept the source and every observed preparation
stage. Runtime example values cannot narrow the proof domain. Unknown admission,
excluded constructions and descriptor-only queries retain the original host
call and a precise refusal reason. No cost estimate or performance policy
participates in this decision.

Numeric preparation preserves integer widths, exact accumulator readouts,
quantize/dequantize boundaries and IEEE host observation contracts. Readouts
use plain exact rational arithmetic with one host ties-to-even rounding;
requant-A certificates, threshold tables and multiply-shift specializations
are absent. The independent convolution reference uses scalar Fraction MACs.

`make_bounded_phase0.py` retains every selected case and assigns visibility
independently of placement. During capsule construction, `lane: admission`
replays ordinary numeric admission: admitted cases require device instructions;
other cases become unscored host guards and retain the raw call ABI. The private
packaging ledger retains each complete admission decision. There is no support
operator roster or exception list. Existing explicit lane declarations retain
their prior meaning.

The optional uniform-fill-copy and contiguous-suffix-copy performance rewrites
are removed. Literal-zero and uniform numeric folding needed by component
representation proofs remain, along with ordinary upstream bufferization and
functional copy expansion. Their retention grants no performance qualification.

Carrier provenance follows every initial/backedge value of structured loops,
all branch yields and all CFG predecessor arguments. Unknown carriers refuse,
including when known arithmetic consumes their result. Source result tables
retain storage-bit identity across movement. Known quantization boundaries
are authenticated by their ordinary generic lowering and preserve table
provenance from data while excluding parameter tables from that data role.
Unsupported boundary names, metadata, arity and incoming carriers remain unknown.

The readout-offset detector accepts scalar, splat and dense tensor literals
and follows their numeric dependencies to rounding/narrowing observations.
Zero offsets and affine offsets without rounding remain distinct. Declared
`exact_rational_host` readout requires `requant_rounding: half_even` (RNE);
non-RNE or missing declarations refuse before numeric admission.
