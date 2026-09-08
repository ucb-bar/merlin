# Final combined Phase 2 compiler artifact

This sealed isolated child composes the exact native epilogue, exact second-tensor residual
fusion, and target-neutral global encoding work on top of the canonical affine-im2col compiler.
It contains no model, layer, or fixed-shape dispatch and does not modify the shared source tree.

The compatibility integration is deliberately strict. Native epilogues preserve the physical
order `bias initialization -> accumulator scale -> activation`; duplicate or reordered stages
do not collapse silently. The native selector combines semantic, dtype, layout, geometry,
overflow, capacity, and reduction-residency guards. No-bias native convolutions load an internal
read-only zero vector so warm and measured invocations are reentrant. Real bias is propagated
through the plan, command buffer, scheduler, and hazard reads. Encoding facts are produced from
the post-fusion boundary, including its true dtype, live consumers, fanout, lifetime, and exact
remaining refusal reasons.

Qualification is complete: 65 focused tests pass; both strict same-process warm/measured Spike
witnesses are exact (56 cycles for the exact epilogue and 102 cycles for the two-convolution
encoded chain); ResNet-50, TinyLLaMA, LSTMNetViT, and SmolVLA compile sequentially; and the
TinyLLaMA dynamic-weight bridge still compiles only under its explicit opt-in contract.

The final ResNet target is byte-identical to the sealed residual child at
`c29ba22408242dcad9aa215c33a4bbca8ac45d12f87352360afeb16ee5167883`, so the exact full-model
same-process warm/measured result transfers without another simulation: 492,147,976 cycles,
1,000/1,000 exact logits, checksum `c6e777c3fe0aae90`. Structurally it retains 15 residual
formations, 105 tasks (51 host and 54 accelerator), 389 ABI pointers, and 82,739,456 physical
intermediate bytes. The global encoding ledger inventories all 53 convolution boundaries and
fails closed on the frozen model because its non-residual narrow epilogues are not exact; it
exposes a conditional 33,341,952-byte boundary reduction rather than claiming it was realized.

No L3, FireSim, queue action, commit, or hardware performance claim was made. Run
`PYTHONDONTWRITEBYTECODE=1 python verify_final_combined.py` from this directory to verify the seal.
