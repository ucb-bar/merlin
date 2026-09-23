# Atlas: whole-model entrypoints

Whole-model inspection and accelerator deployment are different operations.
Start with an existing [model2MLIR capture](../../../docs/guides/model2mlir.md)
and its real weights/manifest, then follow the shared
[lowering and tensor-inspection guide](../../../docs/guides/model_lowering.md).
`merlin lower` consumes that capture and writes invocation-owned IR stages and
audit records beneath a fresh configured output destination. It invokes native
lowering tools; it does not by itself run or certify an accelerator compiler.

The [accelerator workflow](../../../docs/guides/whole_model_on_accelerator.md)
has separate provider, toolchain, model and execution prerequisites. Select the
exact [target descriptor](../target/descriptor.yaml) and independently qualified
compiler/support inputs; do not substitute another target's historical result.
A Phase 1 capsule certificate is not whole-model numerical or timing evidence.

Keep captures, weights, binary tensor payloads, lowered programs and execution
receipts outside examples. Preserve capture hashes and link each new result to
the exact compiler and target evidence. Compact IR views are inspection artifacts,
not executable replacements. No independently qualified Atlas whole-model
deployment is supplied here.
