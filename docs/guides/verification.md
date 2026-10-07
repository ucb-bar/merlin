---
title: Verify a compiler transformation
kind: guide
status: current
owner: verification
last_verified: 2026-10-06
related: [phase0_specification, model_lowering, simulator_selection]
code_refs:
  - src/merlin/verify/receipts.py
  - src/merlin/verify/refine.py
  - src/merlin/verify/linalg_semantics.py
  - src/merlin/verify/merlin_iface_semantics.py
  - src/merlin/verify/smt_semantics.py
  - src/merlin/verify/cb_semantics.py
  - src/merlin/verify/model_coverage.py
---

# Verify a compiler transformation

Merlin can check two concrete transformation boundaries: a `linalg` source module against its emitted
`interface` module, and an `interface` module against its emitted command buffer. The verifier encodes
both sides over the same symbolic inputs, asks Z3 whether any output can differ, and records the
answer. `unsat` verifies equivalence for **all input bit patterns at that concrete shape** under the
encoded semantics. `sat` supplies a counterexample. `unknown`, unsupported syntax, and missing tools
do not verify anything.

## Produce and replay a receipt

For saved in-tree `interface` MLIR and its emitted command buffer, the installed
entry point writes a new receipt directly:

```sh
merlin-verify compile-receipt --interface /generated/interface.mlir \
  --command-buffer /generated/command_buffer.json \
  --translator /selected/mlir-translate \
  --translator-sha256 "$MERLIN_VERIFY_TRANSLATOR_SHA256" \
  --output /generated/transform-receipt.json
```

Exit 0 means verified, 1 refuted, and 2 unsupported, unavailable or unknown.
The installed command currently accepts only an in-tree `interface` module. The
Python API can also check a capsule compiler's actual `merlin_iface` text as the
target of `linalg_to_interface`, using the same saved text passed to the OOT
compiler:

```python
receipt = verify_transformation(
    "linalg_to_interface", source_module, emitted_merlin_iface_text,
    translator=translator, expected_translator_sha256=expected_sha,
)
assert qualify_receipt(
    receipt, source_module, emitted_merlin_iface_text, translator=translator
)
```

Here `source_module` must be the actual before-pass xDSL module, not a recreated
lookalike, and the target must be the emitted UTF-8 text. Replay binds both byte
identities. This path currently covers only the narrow integer grammar below; it
does not prove a complete capsule compiler or an accelerator executable.
`merlin-verify capture-coverage /generated/model.mlir` separately prints a
conservative textual inventory of the current SMT source subset; eligibility
there is not a proof.

Install Merlin with its `verify` extra, provide a compatible LLVM `mlir-translate`, and select that
binary explicitly. The toolchain is an external input. Set `MERLIN_VERIFY_TRANSLATOR` to its absolute
path and `MERLIN_VERIFY_TRANSLATOR_SHA256` to the approved SHA-256 digest. For example:

```python
import json
import os
from pathlib import Path

from merlin.verify.receipts import TransformReceipt, qualify_receipt, verify_transformation

translator = os.environ["MERLIN_VERIFY_TRANSLATOR"]
expected_sha = os.environ["MERLIN_VERIFY_TRANSLATOR_SHA256"]

# `source_module` and `interface_module` are the actual before/after xDSL modules
# retained from the compiler invocation being checked.
receipt = verify_transformation(
    "linalg_to_interface",
    source_module,
    interface_module,
    translator=translator,
    expected_translator_sha256=expected_sha,
    timeout_ms=60_000,
)
path = Path("out/artifacts/verification/transform-receipt.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(receipt.to_dict(), indent=2) + "\n", encoding="utf-8")

loaded = TransformReceipt.from_dict(json.loads(path.read_text(encoding="utf-8")))
assert qualify_receipt(loaded, source_module, interface_module, translator=translator)
```

For the downstream boundary, use `"interface_to_command_buffer"` with the actual interface module
and emitted command-buffer dictionary. Replay runs the semantic check again and requires the same
source, target, query, verifier implementation, translator binary, solver version, and verdict. An
IR parser or syntax verifier can establish well-formedness but cannot produce a semantic receipt.
Generated receipts should be kept with run artifacts, not substituted for the input artifacts.

Phase 1 whole-model capsule grading enables the exact transform audit automatically. For a
standalone whole-model dispatch run, set `MERLIN_MODEL_TRANSFORM_AUDIT=exact` before invoking
the compiler. The result's `mesh_execution.transform_audit_index` points to an invocation-local
`index.json` and exact `captured-model`, `normalized-model`, and `outlined-model` MLIR files.
Each stage, the normalization recipe, and the original capture are hash-bound and rechecked when the audit closes. The
coverage certificate separately joins the captured operation inventory to the runtime's
outlined dispatch symbols and completed-call ledger; missing, extra, or unexecuted operations
remain incomplete. An integer contraction rewrite must account for both its contraction and
requantization children; the former must execute on the accelerator, while a completed host-side
requantization is not misclassified as fallback. These artifacts make the transformations inspectable
and provide the exact
inputs for a validator. Matching hashes, types, and dispatch identities do **not** establish
value equivalence. A semantic claim still needs a replay-qualified receipt for a supported
boundary, or an explicitly bounded independent numerical check; unsupported boundaries abstain.
To recheck an archived model outline locally, call
`merlin.runtime.dispatch_runtime.qualify_model_transform_audit(Path(index_path))`.
It verifies all three exact stage hashes and MLIR modules, reruns the declared normalization
sequence from the saved capture, reruns outlining from the resulting normalized module, and
requires byte-identical output at both boundaries. A custom Python operation-selection callback
cannot be serialized as a recipe and is reported as `unsupported_custom_selection`; target-model
grading refuses that status. Older archives without a recipe report `not_recorded`. The reported
semantic-equivalence status remains `not_proven`: replay checks determinism and provenance, not
whether a transformation preserves values or whether target execution matches the source.

Each receipt records SHA-256 identities of the generic xDSL IR text (or, for an emitted
`merlin_iface` target, its exact UTF-8 text) and canonical command-buffer JSON, both sides' typed
signatures, the verifier's source digest (the `merlin.verify` semantic modules plus the
`merlin_iface` generic-form bridge), translator path/version/hash,
xDSL and Z3 versions, timeout, assumptions, solver-query digest, outcome, and any counterexample.
The receipt's `status` is one of
`verified`, `refuted`, `unknown`, `unsupported`, `unavailable`, or `error`. Only `verified` passes the
qualification gate. A counterexample is a diagnostic input and should also be replayed through the
independent numerical oracle or simulator before attributing the defect to a particular pass.

## What the proof covers

The currently encoded source subset is one function with rank-2, positive, concrete-shape integer
tensors (`i8`, `i16`, `i32`, `i64` where width constraints permit), `arith.constant`, `tensor.empty`
used as the destination of a defining `linalg.fill`, `linalg.fill` with an integer constant,
`linalg.matmul`/`linalg.quantized_matmul`, and `func.return`. Quantized zero points must be resolvable
integer constants. The interface subset covers resident packing as value preservation, matmul,
commit without epilogue stages, eviction, and return. Its return values must be exactly its commits
in order. The command-buffer encoder supports its declared opcode subset and abstains on unknown
opcodes or numerical behavior. Shapes, widths, leaf binding, output count, and supported operations
are checked before an `unsat` result can be reported.

The source interpreter refuses a contraction whose `outs` operand is bare `tensor.empty`. [MLIR says
its contents are unspecified](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorempty-tensoremptyop),
while [a linalg contraction][mlir-matmul] reads its output accumulator.
An explicit integer `linalg.fill` defines the init; zero fill makes the common `A @ B` case eligible
for an unconditional source-to-target value proof. The older repeated-RHS example currently uses a
bare `tensor.empty` init and therefore receives `unsupported`, even when its emitted target happens
to agree with a zero-init interpretation. Integer arithmetic uses signed bitvectors and modular
accumulation at the declared accumulator width. A verified receipt establishes equality in this
modeled value domain,
not physical packing, memory behavior, actual RTL execution, performance, or independent correctness
of the semantics encoder. The encoder and command-buffer meaning require separate review and
conformance checks against an independent implementation.

The OOT-text bridge additionally recognizes static signed `i8` rank-2
`linalg.generic` matmul with `i32` accumulation when its indexing maps, region,
and zero initialization match the encoded contraction. That structural check
(`merlin.frontends.linalg_patterns`) is shared with the semantic compiler's Linalg
reader, so a provenance label or `library_call` name never stands in for the
scalar body. Its
`merlin_iface` side accepts only explicit `argN` tensor bindings, each source
argument declared exactly once, value-preserving resident pack, matmul, an
empty-epilogue `i32` commit, and eviction, with exactly one committed output.
Unknown operations or attributes abstain. A checked 16×16 instance proves equality for all input
bit patterns **at that shape**, under this value model. It does not prove the
physical packed layout, DMA, target RTL, or a mixed host/device program.
The SMT source reader uses the shared parsed-body signed-`i8`/`i32` contraction
recognizer, not provenance tags or the mere presence of `linalg.generic`;
different indexing, arithmetic or yielded values still abstain.

[mlir-matmul]: https://mlir.llvm.org/docs/Dialects/Linalg/#linalgmatmul-linalgmatmulop

Floating-point reassociation, dynamic shapes, arbitrary PyTorch operators, symbolic zero points,
nonempty epilogues, multi-function modules, host effects, and unencoded target opcodes currently
abstain. These need explicit semantics and a sound correspondence relation before they can be called
formally verified. A finite shape sweep gives one theorem per checked shape, not one theorem for all
shapes. The proof does not by itself establish that Phase 1 handles every operator or model named in
Phase 0; that broader claim needs coverage accounting and executable model-level evidence.

Adding BF16 requires an exact floating-point semantics for each operation and conversion, including
rounding mode, reduction order, signed zero, NaNs, infinities, and subnormal policy. An IEEE BF16
format can be represented with an 8-bit exponent and 8-bit significand in an SMT floating-point
theory, but the target's actual flush/saturation behavior must be modeled separately. FP8 variants
such as finite-only E4M3 need a format-specific bitvector or circuit semantics. A tolerance claim
requires a specified relational error bound and a sound bound proof, including accumulation and
nonlinear approximations. Sampled tolerance tests are useful numerical evidence; they are not that
proof. Until those semantics and their independent conformance checks exist, the receipt returns
`unsupported` for these programs.

To inventory the current captured models against the source-side subset, run:

```sh
python -m merlin.verify.model_coverage \
  merlin/contract/capsules/model/SY_model_tiny_llama/capsule.interface.mlir \
  merlin/contract/capsules/model/SY_model_smolvla/capsule.interface.mlir \
  merlin/contract/capsules/model/SY_model_resnet50/capsule.interface.mlir
```

The JSON includes each capture's SHA-256, operation counts, observed tensor ranks/dtypes, and
abstention reasons. It inventories printed MLIR; it does not parse or prove the model. The final
qualification still requires a real transformation receipt for each eligible compiler boundary.

## Connect this to Phase 0 and Phase 2

Phase 0 should freeze the software-visible operation signatures and numerical rules, workload
provenance, legal shape/layout/dtype domain, host placement, and test selection criteria. For every
admitted operation family, record which compiler boundary and semantic assumptions its proof uses.
An unsupported operation remains an explicit coverage gap; an `unsat` result on a different shape or
dtype cannot fill it. Keep a holdout set of model/operator signatures outside the corpus used to
write the compiler and encoder. Phase 1 evidence should pair each lowered program with its source,
target, receipt, and independent numerical differential test. Phase 2 consumes the frozen Phase 0
oracle definitions and holdouts, then adds actual target execution and simulator/RTL checks. This
separates a proved local transformation from the distinct questions of workload coverage, oracle
validity, and target conformance.

A minimal Phase 0 capsule set is a finite *witness basis*: it can cover the
recorded source-operation and typed-edge rows, but it cannot by itself
guarantee a functional compiler for arbitrary PyTorch models. A release claim
must name a bounded supported domain (operations, shapes, dtypes, layouts,
aliasing, control flow and numerical tolerances), show every captured operation
is either supported or deliberately routed to a verified host path, and close
the transformation, boundary-transfer, target-conformance and end-to-end
execution obligations for the exact compiler/package bytes. Any unknown,
unavailable verifier, uncovered row or unexecuted path keeps that claim open.
