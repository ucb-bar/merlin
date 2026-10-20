# `prepare-fix-bool-fptosi` verification

## Claim and scope

The frontend intent is Python/PyTorch truth conversion: `bool(x)` is false exactly when `x == 0`,
and true otherwise, including NaN. The independently stated contract is unordered-or-not-equal to
positive zero (`cmpf une`). Thus +0 and -0 are false; finite nonzero values, both infinities,
and NaNs are true. Fractional magnitudes do not get truncated first. The independently authored A2
specification is `MerlinVerify/Spec/FrontendIntent.lean`, brought in as commit `4d5965876` from
`origin/pv/specs`; its `f32ToBool` definition classifies f32 bit patterns without calling the pass
or MLIR comparison semantics. `Rules/FrontendBoolFptosi.lean` proves the checked repair equals that
definition for every f32 bit pattern and every float model satisfying the explicit `UneCorrect`
comparison contract. The generic semantics and `SPEC_LOCK` are unchanged.

The specification was derived independently from the pass: PyTorch's
[`Tensor.bool()` documentation](https://docs.pytorch.org/docs/stable/tensors.html) identifies it
with conversion to `torch.bool`; PyTorch's
[`TypeCast.h`](https://github.com/pytorch/pytorch/blob/main/c10/util/TypeCast.h) specializes that
conversion to C++ `bool`; and MLIR's
[`CmpFPredicate::UNE` implementation](https://mlir.llvm.org/doxygen/ArithOps_8cpp_source.html)
returns true for every comparison other than equal, including unordered NaNs. The edge table was
checked against PyTorch independently of Merlin's pass. No human reviewer approval is represented
by this record. The Lean theorem quantifies over the target's comparison interpretation; the host
differential test below checks that interpretation on IEEE f32 edge cases.

`Rules/FrontendBoolFptosi.lean` contains the executable `check` function over extracted scalar
syntax facts and `check_sound`. Its explicit `UneCorrect` premise is necessary because frozen
`FloatModel.cmp` has no laws: an arbitrary model may report the wrong result for `une`. The theorem
has no admitted axiom; `#print axioms check_sound` reports only `propext`, `Classical.choice`, and
`Quot.sound`. The Python boundary test checks the corresponding facts in the actual xDSL module.
The current integration does not export an arbitrary MLIR module directly into a `Candidate`;
that binding is checked by the Python guard. A2 marks its specification **pending human review**;
this report records an independent source review and executable oracle, not human sign-off.

## Captured boundary

The checked-in `SY_model_smolvla/capsule.interface.mlir` contains one `arith.fptosi` to `i1`, in
`forward` (the remaining float-to-integer casts target wider integer types and are outside this
rewrite). The regression parses and verifies the actual capture, binds the matching op, checks the
pass rewrites exactly one occurrence, and verifies the result module.

## Mutation and differential coverage

The mutation regression constructs valid MLIR modules for retaining the signed cast, reversing the
comparison to unordered equality (`ueq`), using ordered-not-equal (`one`, which mishandles NaN),
and truncating to integer before converting to i1. Each is parsed and verified, then rejected by
the output-shape guard. That guard checks operand identity, the `i1` result, empty fast-math flags,
and a same-type positive-zero constant in addition to the `une` predicate.

The native differential test compiles the fixed module and compares its f32 output against
A2's independent bit-pattern transcription and PyTorch `to(torch.bool)` through a separate
interpreter, for NaN,
positive and negative infinity, +0, -0, positive/negative fractions, and ordinary positive and
negative values. The unmodified form is checked only for the presence of the poison-producing
cast; poison outputs are never assigned numerical expectations. On this checkout the native tests
ran with the model2MLIR Python environment, the local LLVM `llc`, system `cc`, and a PyTorch 2.14
oracle: all nine tests in `test_bool_cast_poison.py` passed. The actual SmolVLA capture test observed one float-to-i1
boundary and verified the rewritten full module.

## Limitations

The Lean theorem is a compositional f32 scalar rule conditioned on `UneCorrect`; the native
differential probes its edge cases but is not a universal proof of that premise. It does not
establish completeness of the generic MLIR model for the entire SmolVLA module. The existing
structural test continues to check that the pass handles f16, bf16, f32, and f64, but their
bit-level frontend intent is not yet specified by A2. A2's pinned PyTorch 2.10 oracle was not
available on this machine; the installed PyTorch 2.14 oracle was used for the native differential.
