# Integer convolution composition

`prepare_integer_convolution_calls` recovers an unpadded NHWC/HWIO convolution
from exact gather/reshape/transpose maps, a zero-seeded signed-i8 modular-i32
product, channel bias, and the output layout. Optional scale-one signed-i8
readout absorbs only an independently proved closed scalar clamp. Extra raw
observers, unknown effects, changed arithmetic or unsupported geometry retain
the original source. Recognition never reads provenance labels or tensor data.

`lower_integer_convolution_program` composes a caller-selected backend emitter
with ordinary host lowering and the existing borrowed-buffer lifetime checker.
The existing logical ABI bridge checks complete dense descriptors, bounded
spans and disjoint output and input buffers. Unsupported strides or overlap trap before
calling the device. Source tensors and target implementation qualifications
remain separate from these supplied synchronous borrowing premises.

The initial experimental adapter reuses the immutable V26 scalar DMA-gather
implementation with modular bias seeding and native saturation. It emits no
hardware-loop instructions. Full captured conv56 output checks pass; frozen
annotated Spike predicts 92,111,906 whole-invocation cycles against the starting
117,926,419 (1.2803×). This is an unqualified estimate pending paired hardware.
The 1,806,336-cycle compute-only floor is a partial bound, not a complete
attainable roofline. Other convolution geometries require their own cycle
comparison before selecting this route.

The affine-run option splits a spatial tile at exact output-row boundaries.
Each fragment becomes one constant-stride DMA rectangle; index setup is hoisted
outside channel/tap loops. The shared `ConvolutionGatherRuns` plan owns bounds
and splitting, while the backend owns DMA fields and scratch addresses. Its
explicit `coalesce_gathers=True` option keeps scalar commands and adds no FSM.
Captured complete-output checks pass for all three convolutions. The frozen
model predicts 17,421,535 / 17,134,279 / 45,116,103 cycles for output sides
56 / 28 / 14, respectively: 6.7690× / 3.3123× / 1.1176× against the starting
checkpoint. These are estimates awaiting matched hardware validation.

The resident-RHS option requires an explicit private device-row epoch. The
shared planner reserves one activation tile, then chooses the largest complete
reduction panel that fits the supplied operand and separate accumulator stores.
Its reduction-major RHS layout is injective for every active column tail; the
backend fences before reuse and retains each accumulator through its full
reduction. A single affine activation gather serves every column block in the
panel. Lack of complete-panel capacity falls back to ordinary coalesced gathers.
No hardware-loop instruction or numerical reassociation is introduced.

With the selected 16-element tile and 256-KiB operand store, the three weight
sizes cross the complete-residency boundary: 36 and 144 KiB fit, while 576 KiB
uses seven column blocks per bounded panel. Complete outputs pass for all three;
the same frozen model predicts 11,841,653 / 10,422,061 / 10,494,233 cycles for
output sides 56 / 28 / 14: 9.9586× / 5.4455× / 4.8046× against the starting
checkpoint. Conv14's compute-only fraction is 17.21%; a complete attainable
roofline and matched hardware validation remain pending. Tests cover layout capacity,
private ownership, tails, raw-i32/native-i8 readout, modular extreme bias,
strided/dilated windows, byte canaries, input preservation and live connected
convolution intermediates.
