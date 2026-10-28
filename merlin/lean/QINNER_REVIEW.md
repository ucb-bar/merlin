# Qinner binding and concrete byte decoder review

Status: provisional, pending independent human review. Checker acceptance, passing tests, receipts
and a clean proof audit are not human approval or approved full verification. No frozen semantics,
SPEC_LOCK, production pass, shared Main/CLI, W8A8 or boolean work is changed.

## Proposed claim

An accepted Qinner boundary has exactly equal complete program outcomes under two explicit
interpretations: the source's tagged empties denote the named archive tensors, while the target
receives the same tensors as ordered trailing entry arguments. The specialized claim now uses a
concrete little-endian, row-major byte decoder. Its accepted values are proved correctly typed,
initialized and bit-exact. The original entry argument count must match. Calls, nested regions,
captured SSA values, complete outputs and signatures participate in checking.

The strongest specialization is `MerlinVerify.Check.QinnerDecoded.program_equal`. It has no
arbitrary decoder parameter. It requires the complete transformation checker, the concrete
runtime-certificate checker, an existing changed `@forward`, and original entry arity. Its source
binding function is `QinnerRuntime.binding archive QinnerDecoded.interpret`. The target receives
`args ++ delivered.map QinnerDecoded.interpret`. The theorem quantifies over the float model,
index width, unchanged operation meanings and common external-call implementation.

The binding interpretation is explicit. It gives a tagged empty the named runtime value; the
frozen ordinary `tensor.empty` interpretation produces poison. The lift claim is not promoted to
literal original-MLIR source equivalence. Neither typing nor decoder initialization repairs the
refuted scalar-square numerical law or supplies multiplication-policy approval.

## Actual archive and C representation

The production `qinner.resolve` loads `extra.npz` entries named `qinner::<key>`, applies
`np.ascontiguousarray`, and checks exact shape and NumPy dtype spelling. Supported archive dtypes
are bool, int8/int16/int32/int64 and float16/float32/float64. C generation invokes that resolver,
then appends `np.ascontiguousarray(arr).tobytes()` to its weights blob without an endian conversion.
Its argument rows carry shape, rank, element byte size and blob offset, with qinner rows after the
original arguments and before quant-hoist rows. `DT_BYTES["i1"]` is 1; the C storage is a signed
char. This is byte-per-bool host storage, not packed DenseElements i1 storage.

Consequently this qualification requires a little-endian host and native little-endian array
dtypes. The certificate producer rejects unsupported endian representations rather than
byte-swapping them into a different representation. Fortran arrays are accepted through the
production resolver's actual conversion to contiguous C order. The emitted bytes are compared
against generated C blob slices and the complete rank/shape/element-size rows.

Canonical i1 bytes are exactly 00 and 01. NumPy can carry bool arrays whose raw storage contains
02, 80 or ff, and the production resolver checks dtype and shape without canonicalizing those
bytes. The concrete checker rejects them. This is an explicit restriction on the verified input
domain; production code is untouched and a raw archive counterexample is retained.

The formal decoder can describe a static rank-zero tensor's bytes. That does not make the current
archive runtime accept a scalar array: `ascontiguousarray` promotes its shape from () to (1,), so
`resolve` rejects it. The earlier rank-zero counterexample is preserved.

## Concrete decoder contract

`Rules/QinnerDecoder.lean` supports signless MLIR i1/i8/i16/i32/i64 and f16/f32/f64 tensors with
static dimensions and no encoding. Signed host integers map to signless bit vectors preserving
their two's-complement storage. bf16, index, unusual integer widths, encoded/dynamic tensors and
nontensor payloads are rejected. A matching byte count does not make an unsupported type valid,
including when the tensor has zero elements.

The decoder requires `byte_order = "little"` and `layout = "row_major"`, an exact byte count of
`shapeSize(shape) * elementStorageBytes`, and canonical bool bytes. At in-bounds coordinate `i`,
it reads exactly the element-sized slice beginning at
`linearIndex(shape, i) * elementStorageBytes`. That slice is interpreted as
`b[0] + 256*b[1] + 256^2*b[2] + ...`. Integer and float scalars are constructed directly from the
corresponding bit vector. There is no float evaluation, conversion, rounding or NaN normalization.

The proofs establish:

- `decode_typed_initialized`: every accepted payload successfully decodes to a tensor with its
  declared runtime type and nonpoison elements at every in-bounds coordinate.
- `complete_chunk`: every observed cell has its full slice inside the payload. Success does not
  rely on missing bytes being padded or reads falling outside the tensor.
- `exact_bits`: no high bits are truncated, including i1's full canonical host byte. The resulting
  scalar word is exactly the little-endian value of the corresponding row-major slice.
- `exact_byte` and `decoded_cells`: extracting any stored byte from that word recovers the original
  byte. This covers every bit of signed boundaries, negative zero, infinities and both quiet and
  signaling NaNs, including sign and payload bits.
- `expectedCheck_sound`: standalone acceptance additionally matches the caller's complete expected
  type and shape, and establishes typing/initialization for that expected type.

Empty dimensions require zero payload bytes. They yield correctly typed, initialized empty
tensors; their vacuous element obligations do not weaken type, format or length checks.

Raw bytes cannot reveal the endianness their producer intended. The decoder rejects an explicit
big-endian label, the archive intake rejects nonnative/big-endian dtypes, and byte-swapped delivered
payloads fail archive equality. If someone changes both the bytes and their purported provenance
while keeping a little-endian label, they have supplied different tensor data; this theorem does
not authenticate that data or infer the intended original values.

## Connection to binding and complete programs

`Check/QinnerDecoded.lean` requires the existing full runtime-binding check and concrete payload
checks for archive and delivered records. The binding check requires unique keys, exact full
byte-vector equality, exact types/shapes and ABI order. Digests only bind receipt files; they do
not substitute for full payload comparison.

`delivered_sound` proves typing and initialization of accepted delivered values.
`binding_typed_initialized` proves `BindingTyped` and `BindingClean` for all named source bindings,
so no accepted slot uses the missing-key poison default. These facts concern the added tensors,
not the successful execution of every unchanged model operation.

`program_equal` specializes the existing whole-program proof to this concrete interpretation.
The original SSA proof checks every use and output and composes through nested regions and calls.
Only `@forward` gains arguments; calls to that affected entry are refused. All unchanged external
symbols keep the same implementation/contracts. No premise asserts source/target program equality
or correct results for an inventory of unchanged model computations.

## Which assumptions this removes

The previous theorem permitted an arbitrary common decoder. The specialization removes that
parameter and the need to assume the decoded added values have the declared types or contain no
poison. Little-endian interpretation, row-major indexing, widths, exact lengths, canonical bool
storage and exact scalar bits are now implemented and proved in Lean. No numerical axiom or new
assumption is introduced; audited proofs use only propext, Classical.choice and Quot.sound.

The following obligations remain explicit:

1. The archive bytes and `qinner::<key>` annotations must authentically identify the captured
   tensors before any intervening view. This work does not prove capture/archive authenticity.
2. The host loader, NumPy representation, generated C, native compiler, descriptors and eventual
   runtime must implement the stated byte interpretation and deliver the certified ordered bytes.
   Their agreement is tested concretely, not proved by the Lean decoder. Hardware and deployment
   correspondence are outside this local claim.
3. Unchanged external-call implementations/contracts must be common to both executions.
4. A whole-model success or poison-free-output claim still needs obligations for unchanged
   computations and original inputs. Initializing added tensors does not prove those obligations.

## Propagation and lift scope retained

`_propagate_quant_inner` changes only `prov.quant_inner` on producer `tensor.empty` operations.
The complete propagation checker reconstructs consumer indices and supported view chains, checks
its expected complete output and proves complete value preservation under frozen semantics.
That metadata is not irrelevant to runtime infrastructure: it selects interpreter bindings and
prevents the outliner from cloning those empties.

The verification domain admits supported view wrappers and at most eight producer owners. It
refuses unsupported operand-zero walks, missing producers, out-of-range indices, conflicting tags
and duplicate keys on distinct empties. Repeated consistent requests for one empty are allowed.
The production request count includes failed walks; covered-site counts are independently derived.
The generic printer preserves producer attributes for evidence; ordinary custom-printer loss is
retained as an earlier counterexample.

## Evidence, replay and preservation

The current qualification is under `out/artifacts/verification/merlin-passes/frontend/qinner-decoder/`.
`summary.json` identifies the exact run; `FINDINGS.md`, `qualification/`, `manifest.json` and per-run
proof-audit statements record results and limitations. Receipts stay abstracted and unapproved.

Runtime receipts use `merlin.lean.qinner.v2` and concrete runtime certificates use
`merlin.qinner.runtime.le.v2`. Replay checks dependency/compiler/input freshness, reruns actual
propagation/lifting, round-trips complete MLIR exports and checks the concrete runtime payloads.
Standalone decoder receipts bind full requests/results and archives; replay reloads those archives,
reruns production `resolve`, reconstructs the exact payload bytes and reruns Lean decoding.

Archive/native observations include all supported types, signed boundaries, signed zero, infinities,
subnormals, quiet/signaling NaNs with payloads, and empty tensors. They compare every decoded bit
with native MLIR results and every generated C blob byte with resolved archive data. Tests also
cover singleton ranks, Fortran-order archives, wrong expected types/shapes, wrong lengths/order,
unsupported empty tensor types and invalid bool encodings. These are empirical host-correspondence
checks, not proofs of host or deployment correctness.

SmolVLA and TinyLlama remain unchanged zero-lift boundaries. Positive lifting receipts are separately
labeled synthetic cases with actual rewrites and real archive bindings. No positive model coverage
is inferred from unchanged receipts.

The prior Qinner contract is committed as `f15b41efe`. Its original evidence and counterexamples
remain byte-identical under the earlier `frontend/qinner/` directory. Old receipts remain bound to
the older checker sources and should be replayed with that version, not treated as current decoder
receipts. Decoder qualification uses a separate output directory. Prior scalar-square/Overrank
checkpoints and numerical counterexamples remain preserved.

## Qualification of this version

The Lean audit checks 2,587 imported theorems with zero violations and prints 24 claim statements.
All 161 relevant fast tests pass. Two independent slow tests replay all 11 program and 16 decoder
receipts successfully. Sixteen archive/native cases (32 samples) match every output bit. Nine
positive program boundaries perform 18 lifts; both retained model boundaries remain unchanged.

Source-layout/membership tests, Ruff and documentation freshness pass. The repository structure
gate retains the unrelated pre-existing `zephyr_model.py` ceiling failure (3,557 versus 3,537
lines); that file is outside this change. Generated evidence remains local under the directory
above and is reproduced by the committed qualification script.

## Reproduce

Run `.venv/bin/python build_tools/scripts/verify_qinner_decoder.py --compiler-python <model2MLIR-python>`.
Add `--stage LABEL <0004-propagation-directory> <0005-lift-directory>` for each retained model.
Use the environment interpreter path rather than its resolved base executable.

Run `.venv/bin/python -m pytest -m 'not slow' merlin/tests/ir/test_lean_qinner.py
merlin/tests/ir/test_lean_qinner_decoder.py merlin/tests/ir/test_qinner_binding.py`, then run the two
Qinner test modules with `-m slow`. The slow tests replay program-binding and decoder receipts.
An alternative evidence directory is selected with `MERLIN_QINNER_RECEIPT_ROOT`.

The standalone driver supports propagation, lifting, concrete runtime `binding`, payload `decode`
and MLIR `print` modes. Shared Main/CLI integration remains separate. Independent human review
should approve the named-binding intent and stated remaining assumptions before any paper claim;
it must not convert these results into unconditional original-MLIR or whole-model verification.
