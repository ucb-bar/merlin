---
title: ReLU numerical contract
kind: reference
status: current
owner: runtime
last_verified: 2026-11-19
related: [contracts, host_device_compilation]
code_refs:
  - src/merlin/compiler/frontend/patterns/narrow_float.py
  - src/merlin/compiler/lower/c/host_ops/relu.py
  - tests/ir/test_host_cnn.py
  - tests/ir/test_host_cnn.py
---

# ReLU numerical contract

The producer emits standard arithmetic. A target policy cannot redefine it.
Merlin recognizes the scalar body and records its semantics before placement.

`arith.cmpf olt, x, +0` followed by `arith.select cmp, +0, x` retains x
on ties and unordered comparisons. For BF16, `-0 -> -0`, `+0 -> +0`,
negative finite values (including subnormals) and negative infinity become +0;
positive values and positive infinity survive. NaN inputs select x. The host
implementation preserves every NaN sign and payload, including signaling
encodings, by operating on bytes without floating arithmetic. This scalar IR
contract does not model floating exception flags. Its internal contract is
`bf16-ordered-lt-positive-zero-select-v1`, domain `all-bf16-bit-patterns`.
No source attributes or runtime domain promises are required.

`arith.maximumf(x, +0)` differs at negative zero: it returns +0. It propagates
NaN, but its NaN payload is not fixed by that spelling. Merlin has no source
route for a sign-clamp declaration; such a declaration is refused. Do not
manufacture a narrower input-domain premise while importing a model.

Atlas `vrelu.bf16` uses a sign clamp. In npu_model b97b2f5's RTL-generated
lane table, a sign-set input becomes +0 and other encodings survive. Thus -0
becomes +0, positive NaNs survive, and negative NaNs become +0. It is not a
valid implementation of the ordered select over all BF16 encodings. A
negative-zero permission alone does not discharge the NaN obligation.

The default lowering for the ordered BF16 select is the exact host byte loop.
No zero-observability analysis or target rewrite is implemented by this change.
Using vrelu instead requires an approved numerical contract permitting every
observable difference, or a proof from the program's declared semantics that
the differences cannot be observed. Atlas's placement/precision policy alone
cannot supply that authority. For example:

* Under declared IEEE addition, adding either signed zero to the same finite
  nonzero operand yields that operand exactly, including its sign. If every
  use of the select result is such an add, the addend is proven nonzero, and
  no observer sees the intermediate bits, the zero-sign difference vanishes.
  An unknown addend or a zero addend does not establish this proof.
* Under a declared quantizer Q with `Q(-0) = Q(+0)` as an encoding equality,
  both zeros become the same output bytes. This proves the zero-sign issue
  only if every use passes through Q before observation; the name "quantizer"
  alone proves nothing.

Both examples still require a separate NaN argument (for example a proven
NaN-free input domain), plus checking subnormals, infinities, and all other
possible target differences. Returning or bitcasting the result, reciprocal,
or sign-sensitive consumers can expose negative zero. Preserve its sign or
keep the operation on the host when those obligations are unresolved.
