# Quant-inner propagation and lifting review packet

Status: provisional, pending independent human review. No approval is asserted by a passing test,
checker acceptance, proof audit, or an `abstracted` receipt. Nothing in SPEC_LOCK is changed.

## The claim proposed for review

For an accepted complete boundary, the prepared source interpreted with explicit named runtime
bindings has exactly the same execution outcome and complete ordered result list as the lifted
program supplied with the original arguments followed by the checked named tensors. The argument
count must match the original entry block. The theorem quantifies over the float model, index
width, unchanged computations, external-call implementations and byte decoder. All functions,
nested regions, captured SSA uses, function signatures and outputs are checked. Internal calls to
the signature-changing `@forward` are rejected; declarations keep the same external interpretation.

The source binding interpretation is explicit: a well-formed tagged static `tensor.empty`
produces the value named by its key. This is the meaning implemented by the host interpreter when
its qinner dictionary is supplied. It is **not** the frozen poison-producing `tensor.empty`
meaning, and is not presented as an approved original-source equivalence theorem.

The strongest combined theorem is `MerlinVerify.Check.QinnerRuntime.program_equal`. Its premises
are acceptance of the complete transformation, acceptance of the full runtime byte certificate,
a changed boundary with a defined `@forward`, and original entry argument arity. Its result is
exact equality of complete program outcomes under the named-binding interpretation. The byte
decoder remains an explicit parameter. No assumption about every unchanged model computation's
successful execution is needed or introduced.

## What propagation changes

`_propagate_quant_inner` only assigns `prov.quant_inner` on `tensor.empty` producers. The checker
reconstructs consumer operand indices, follows supported view wrappers to their producers, and
compares the entire expected output. It also compares the complete AST after erasing only that
one producer attribute. The value-preservation theorem
`MerlinVerify.Check.QinnerPropagation.check_sound` uses the frozen semantics and covers every
function, region, output and call, including unsuccessful outcomes.

This attribute is semantics-free **in the frozen tensor-value semantics**, under the existing
reviewed annotation policy. It is not irrelevant to execution infrastructure: the interpreter
uses it to choose real arrays, and the outliner prevents cloning tagged empties. Consequently,
metadata-only change is not by itself proof of correct runtime binding.

The production walk follows operand zero through any operation for up to eight owners, and its
return count counts requests, including requests that find no producer. The verification domain
admits only the declared view wrappers and requires the empty to be found within eight owners.
Unsupported walks and out-of-range indices are rejected. Repeated consistent requests for the
same empty are allowed. Conflicting requests and one archive key on distinct empties are refused.
The latter is deliberately narrower than the production pass's behavior.

The ordinary xDSL custom printer drops producer attributes on `tensor.empty`. Evidence captures
the in-memory result with the generic printer and independently checks MLIR export and Lean
round trips. Retained models have no tags, so their historical printed boundaries are unchanged.
`qinner.lift` re-runs propagation itself.

## Argument and byte mapping

The source census is a full module-order traversal, including nested blocks. One trailing entry
argument is paired with each distinct accepted tagged empty, in that order. Its complete MLIR
static tensor type, shape, element type and encoding must match. Every use must follow that
pairing. The checker rejects removed/reordered arguments, changed uses, changed original inputs,
changed result lists/types and unrelated operations or metadata changes.

The runtime certificate contains complete little-endian, row-major byte vectors, declared types
and keys for both the archive entries and the delivered trailing arguments. Lean checks unique
keys, full byte equality in ABI order, exact types/shapes and storage lengths, including the
one-byte host storage convention for i1. A wrong tensor with the same shape and dtype fails.
`QinnerRuntime.values_equal` proves that any common decoder gives precisely the ordered values
required by the SSA/whole-program theorem. Hashes bind files in receipts; they do not replace
full tensor comparison in that checker.

The Python qualification runner uses the actual production `plan`, `lift` and `resolve`, checks
that plan and appended argument records agree, and re-runs the actual pass against the complete
retained output. Source, in-memory propagation and output exports are round-tripped through
MLIR and Lean. Replay re-exports, re-executes both passes, re-checks signatures, bodies and full
runtime certificates, and refuses changed input, dependency or compiler bytes.

## Runtime assumptions requiring explicit review

1. A capture-owned `qinner::<key>` names the elided tensor before any intervening view. The archive
   bytes, dtype and shape must be authentic capture data. The proof cannot establish archive
   authenticity or fix an incorrect upstream annotation.
2. The actual runtime must deliver the certified tensor bytes after the original entry arguments,
   in the checked order. For the C path, `c_runtime.generate` derives qinner arguments from
   `model.mlir`, resolves them, appends blob rows in order, and places quant-hoist rows afterwards.
   That implementation was inspected, but its generated C descriptors, allocator, device ABI and
   eventual deployment are not proved here.
3. Byte decoding and host storage must match the runtime's MLIR tensor representation, including
   endianness, element widths and bool storage. The theorem exposes the decoder rather than
   silently assuming that correspondence. Each delivered value must be typed and initialized
   when a successful, poison-free execution claim is desired.
4. External declarations use the same implementation/contracts in both executions. There is no
   additional assumption that arbitrary unchanged model computations succeed or avoid poison.

## Proved versus measured

Proved: metadata value preservation under frozen semantics; complete propagation reconstruction;
SSA substitution through nested regions and calls; complete-program equality under the explicit
binding interpretation; exact signature/type/order checks; full-byte certificate soundness and
its connection to program arguments. Proofs use only propext, Classical.choice and Quot.sound.
The isolated driver imports a positive `QinnerExecution.accepted` theorem and proves successful,
poison-free execution on an explicit inhabited initialized domain for that example.

Measured: actual production-pass fixtures cover f16/f32/f64, i1/i8/i16/i32/i64, several ranks,
singletons and empty dimensions. Native upstream MLIR comparisons cover all outputs of direct
and nested/calling fixtures against separately initialized valid references; inputs and raw
result bits are retained. These observations are evidence about tooling, not a proof that every
native compiler or deployment implements the Lean model.

Neither retained SmolVLA nor TinyLlama boundary contains quant-inner tags. Their receipts are
labeled `unchanged`, with zero lifted sites. Positive lifting receipts are separately labeled
synthetic fixtures. No claim that these models exercise qinner lifting is made.

## Retained refusals and limitations

`../counterexamples/` retains conflicting consumer tags, duplicate keys, custom-printer metadata
loss, and the rank-zero runtime array rejection. `resolve` calls NumPy's contiguous conversion,
which promotes a scalar array from shape () to (1,), so a rank-zero plan is rejected. Production
code is unchanged. The local semantic theorem does not turn that runtime refusal into success.

Typing, successful execution and absence of poison remain separate. Program correspondence may
relate two stuck or UB outcomes, and typed tensors can contain poison. This work does not assert
whole-capture definedness. The earlier scalar-square numerical counterexamples are preserved;
the refuted power-to-multiplication law remains refuted and no multiplication policy is approved.

## Reproduce and inspect

Run `.venv/bin/python build_tools/scripts/verify_qinner.py --compiler-python <model2MLIR-python>`.
Supply `--stage LABEL <0004_propagation-directory> <0005_lift-directory>` for each retained model.
The exact original paths are retained in the run summary and the capture stage pointers
can be recovered from the scalar-square receipts. Use the same compiler interpreter path, not
its resolved base Python executable.

Run `.venv/bin/python -m pytest -m 'not slow' merlin/tests/ir/test_lean_qinner.py` and then
`.venv/bin/python -m pytest -m slow merlin/tests/ir/test_lean_qinner.py`. The latter replays every
retained receipt. `../summary.json`, `../FINDINGS.md`, `../manifest.json` and the per-run proof audit
record the final qualification. The audit includes printed theorem statements and checks all
imported proofs for unapproved axioms and prohibited proof constructs.

Review should decide whether the named-binding intent and the explicit runtime assumptions are
the intended paper contract. It should not convert provisional evidence into an unconditional
original-MLIR, whole-model definedness, or target-runtime claim. The proofs and receipt format
remain outside shared Main/CLI integration until that integration is coordinated separately.
