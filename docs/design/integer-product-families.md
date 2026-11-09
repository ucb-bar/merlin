---
title: Complete integer product families
kind: design
status: current
owner: compiler
last_verified: 2026-11-09
related: []
code_refs:
  - src/merlin/llvmlower/integer_product_family.py
  - src/merlin/llvmlower/source_product_family.py
  - src/merlin/llvmlower/source_attention_frontier.py
---

# Complete integer product families

`IntegerProductFamily` describes multiple exact integer matrix outputs that
share immutable input planes. Each output explicitly lists its operand plane
pairs and a bound covering every signed i32 accumulation prefix. Arbitrary
plane counts, output groups, repeated pairs, matrix tails and signed byte inputs
are supported. Output planes have an explicit stride and disjoint complete
storage. The contract contains no target resources or command encoding.

`family_from_radix_plan` revalidates the existing canonical radix proof. It
does not infer a floating source accuracy policy from an integer family.
`reference_outputs` uses independent Python integer sums. The emitted checked
callback validates complete pointer extents and output/input nonaliasing. A
callback returning success still needs an implementation, full write, domain,
lifetime and synchronous completion proof.

## Source reconstruction interface

`emit_source_attention_frontier(..., source_product_family=contract)` is an
explicit specialization of the owned complete fused readout seam. Default
emission and the degree callback API remain unchanged. The opt in callback has
a distinct type: its last argument is the output plane stride, not a degree.
Callers must compile a matching physical bridge; reinterpretation of an old
degree callback is unsupported.

The existing source plan derives every QK/PV shape and canonical integer bound.
One callback writes all five original degree results containing all nine radix
pair products. The provider retains all readouts through the unchanged exact
integer reconstruction and binary64 conversion. The private readout array is
flattened with identical size and alignment, permitting defined C pointer
arithmetic across its planes. Each reconstruction pointer addresses the same
original plane and reads only its complete matrix prefix. Plane gaps are not
read or written.

`SourceProductFamilyContract` separately requires exact products, complete
output planes, immutable inputs, private disjoint outputs, synchronous final
drain and preserved host FENV. Existing prepared RHS owners, point witnesses,
source scale and norm computations, interval arithmetic, rounding observations,
refinement, source fallback and numerical policy are retained. This interface
grants no approximation, zero-product elimination or workload selection.

The complete callback adapters check only the shapes derived from the actual
plan. A backend owns the physical kernel ABI, live storage reservations,
primitive instruction sequence and lowering. Full group costs include input
preparation, transfers, all output planes, reconstruction, certification and
drain. A faster product capsule alone does not establish a faster whole model.

## Verification and automatic optimization

Generic tests cover integer prefix bounds, duplicate/nonradix groups, malformed
domains, spans, private holes, callback refusal, different source dimensions,
prepared owner epochs and four borrowed consumers. Candidate and ordinary
source compositions preserve complete output words and callback failure before
public output publication. An opt in family produces one callback for each
original five degree calls.

Qualification artifacts separately retain original whole output validation,
all target integer readouts, target FENV and complete cost receipts. Neither
Python/Numpy correctness stand-ins nor functional Spike instruction counts are
FPGA cycle measurements.

The family contract exposes a useful compiler choice for automatic loops:
select bounded common operand storage and multiple live destinations, while
retaining complete mathematical outputs. Phase 0 should close source, domains,
output ownership and accuracy before performance work. Phase 1 can search legal
family tiling/layouts under backend resource facts and model transfer cost.
Phase 2 must measure complete source composition, including host preparation
and consumers, and retain negative whole or complete cost results. These choices
are driven by semantic contracts and resource facts, never benchmark identities
or sampled golden outputs.

`integer_product_family` owns the mathematical family and complete bounds;
`source_product_family.source_product_families` derives the actual source
shapes, and `prepare_source_product_family` materializes the distinct callback.
That callback is offered to a selected physical backend; its availability is not
a placement decision or a target-resource reservation.
