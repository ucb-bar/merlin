# Spatial regions from current source SSA

`merlin.llvmlower.connected_spatial.extract_connected_spatial_region` extracts
one explicitly selected convolution/pool call pair from an ordinary
`rewrite_groups_to_device` result. It first runs `group_connected_spatial` on
the complete current module. It does not construct a graph from two component
entries or infer a connection from equal dimensions.

The extraction clones the actual calls and their fresh tensor destinations.
Original free SSA values become region inputs in first-use order; every member
result with an external observer becomes a region output. The source matcher
continues to refuse intermediate observers, changed program statements, shared
callees and intervening effects. The complete original module, including its
public outputs, is unchanged. `inputs` and `outputs` retain the original SSA
objects so a source-owned runner can bind the actual frontier values.

The returned rewrite retains the original entries, programs, access declarations
and logical buffers for the two selected calls. Its region takes already bound
values rather than the complete model's argument array, so the original model's
blob argument plan is not applied to the region. The full source capture and
weight/prepack records must remain in the enclosing custody record.

The extracted region goes through the existing `group_connected_spatial`,
`whole_program.group_interface`, `prepare_connected_spatial_host` and closed
device-helper selection APIs. In particular, the original Q1 and Q2 readouts
remain separate. The source plan checker still compares their coefficients,
signed formats and complete convolution/pool geometry.

Extraction and successful object/link emission establish no Phase 0 source
closure, device effects, DMA visibility, numerical or native execution result.
Borrowed-buffer effects, completion and workspace contracts remain explicit
independent premises. Those obligations and the ordinary frozen-source checks
must be satisfied before the region is executed.

## Frozen original host selection

The versioned `merlin.owned_spatial_source.v2` selection extends the existing
source/plan pins with two mandatory direct capsule-local documents:
`source_graph` contains the extracted current SSA and `source_rewrite` contains
its original ordinary group sidecar. Public cohort relocation preserves all
four size-limited, hash-bound documents. The ordinary host compiler reparses the
graph, restores the selected group sidecar, reruns source matching and requires
the exact complete region and its ordinary interface export. It does not call
the explicit component-graph constructor used by version 1 witnesses. Both
versions retain the source-owned allocation, private helper selection and
independent completion and object admission checks.

Original frontier operands can use the existing input-only linear palette
representation. A complete scalar sequence preserves the exact original bytes;
the enclosing preparation must establish that sequence from the source SSA and
retained weights, rather than generating replacements. Independent reference
values stay outside author construction actions. A source reference evaluated
from the original captured SSA can be compared exactly with the ordinary
declared-operation reference before freezing the prospective source member.

## Concrete internal-width proof

`operation_numerics.integer_matrix_partial_sum_bound` supplies a tighter
sufficient bound when global operand maxima would combine values that never
multiply. It computes `max(abs(A) @ abs(B)) + max_abs(initial)` over the complete
concrete matrices. This bounds every partial sum in every reduction order and
does not depend on cancellation or a small final result. The selected operand
and MAC widths remain unchanged; unknown widths, invalid operands, insufficient
host proof budget and potentially overflowing proof arithmetic refuse. The
ordinary integer writer uses this fallback only after its existing global bound
reports possible overflow, with convolution patches formed by the existing
declared convolution geometry.
