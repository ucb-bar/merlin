# Graph input-admission repair — source review packet

D2-G01/G02 are repaired in the **new host source versions** in this packet.
The frozen SDK-free regression run passed **81/81 tests, zero skips**. This is
parser/admission/actual Python producer evidence, **not Lean acceptance, native
MLIR verification, graph qualification, or approval of semantic premises**.
No old seal, expected pin, receipt, protected semantics or original4499 source
was changed. Coordinator alone integrates and publishes.

## Exact changes and shared API

`src/merlin/verify/lean_graph_input_admission.py` is the single implementation;
the corrective producer contains an identical copy. `SOURCES.json` binds both.

- `admit_text(raw)`: uses the selected xDSL **lexer**, before parsing, to refuse
  file metadata. Quoted strings, escaped quotes and line comments are tokenized
  correctly. Finding a metadata opener is a conservative domain refusal, not a
  claim that its unparsed payload is well formed.
- `admit_resources(op)`: recursively visits actual attributes, properties,
  types, locations and all nested regions. DenseResourceAttr is refused. Both
  UnregisteredAttr and builtin OpaqueAttr are refused because their embedded
  text is not an inspected attribute tree. Unknown data representations fail
  closed. This producer-only check makes no claim about call semantics.
- `admit_module(module, calls='external-only')`: additionally refuses
  `func.constant`, `func.call_indirect`, function-value types, nested functions,
  unregistered function operations, and **every SymbolRefAttr outside a
  registered direct call's callee field**. Direct callees must be flat, declared,
  and have exact operand/result signatures. Default mode requires declarations
  without bodies. This deliberately excludes external symbolic metadata too.
- Explicit `calls='shape-direct'` permits registered direct calls to defined
  functions. It does **not** prove their bodies or call-depth behavior. W's
  finite-depth shape/memory adapter must independently check that correspondence.
  Shape uses this mode; reuse/memory default to external-only and expose an
  explicit keyword/CLI opt-in. Unknown opaque operations get no hidden symbol
  lookup exception.
- `parse_admitted(raw, parser, calls=...)` combines raw admission, the actual
  parser/verifier, and structural admission. InputAdmissionError carries `code`
  and `path`. Parser/verification exceptions remain distinct. The integer CLI
  now labels ParseError/VerifyException `malformed`, rather than `unsupported`.

Nine capture/export leaves use the helper, including all original/supplied
before/middle/after paths in scalar, pipeline, integer, shape, reuse, memory-SCF,
segmentation and unrolling. Public parsed-module exports are checked too.
`operation_tree` independently decodes actual index IntegerAttr values; it no
longer imports production normalization merely to read a literal. This allows
W's primary-owned export without mixed producer namespaces.

The corrective producer patch adds raw/resource admission to certificates
`parse_ir` and resource admission to `ir_text`. It does not edit Q's gate or
claim that parsing/printing proves local-call equivalence. Already parsed IR
cannot recover unused file metadata lost by some earlier external parser;
original text remains mandatory at raw intake. Structural resource values are
still refused even when the raw file dictionary was already dropped.

## Evidence and exact scope

`../input-admission-regression-v1/RESULT.json` lists all81 tests. Its `work/`
retains actual original inputs and outputs at the paths named in the new
capture records; `cases/` also retains child stdout/stderr/status. Twelve
positive capture/supplied controls span all six required families. The producer
positives include **actual changed CSE, exact-integer and shape normalization**.
Twenty-four actual CLI negatives cover capture-only and supplied outputs.
Parsed-public entry tests cover nested symbols, resources and indirect calls.
Other cases cover same-process resource state, fresh-process refusal, malformed
input, comments/escapes, opaque attributes, exact direct-call signatures,
internal shape calls and explicit memory opt-in. All are actual Python parser
or producer calls, not mocked proof results.

Reviewer fixtures are copied byte-for-byte from review commit
`5a7f872db33d378d6b0c4a294e4099e80efd4956`; their source paths/hashes are in
`merlin/tests/fixtures/graph_input_admission/PROVENANCE.json`. Changed/dropped
payload and returned-resource cases are retained without relying on global
resource-handle spellings. Unsupported program mutations are not advertised
as compiler-MLIR-valid Lean refusals: native checking remains held.

This run used **Python3.14.7 / xDSL0.73.0**, with actual loaded source identities
in `EXECUTED-SOURCE-CLOSURE.json`. These are new parser identities, not the
reviewer's historical parser qualification. Captures select unchanged producer
4499; separate tests exercise corrected producer certificates. The isolated
runner denies named dynamic-library loads, SDK/archive opens and unapproved
child commands. Python's process handle, Python execution, OS, source transport,
lexer/parser/projection and test harness are trusted. No formal sandbox claim.
Optional z3 import diagnostics from the deliberately blocked named library
remain raw output and confer no numerical evidence.

## Source delivery / closures

- `primary.patch`: exact owned primary changes from HEAD58dbc7c00;25 source/test/
  fixture files, with `source-snapshot/` and identities in `SOURCES.json`.
- `corrective-producer.patch`: ONLY certificates.py and the identical helper,
  from4499, with `corrective-producer-snapshot/`. Apply via coordinator's targeted
  producer intake, never wholesale experimental merge.
- Capture records bind `input_admission_source`. New dynamic scalar/pipeline/
  integer/segment/unroll/reuse source lists include the helper. Existing pinned
  shape/memory selectors were **not** refreshed; they must refuse changed source.
  W's new wrapper owns its new source closure and consumes the explicit mode.
  Q's corrected gate consumes this API; no gate implementation is included here.
- The loaded-source closure records what these tests actually executed. It is
  not a claim that every lazy import in every old qualifier was exercised.

Ruff and no-regex pass. The repository structure check fails only on unchanged
foreign `src/merlin/runtime/backends/zephyr_model.py`3557>3537 lines. No allowance
was changed. Development profilesv1–v6 retain first failures and prior source
snapshots: initial test guard denied Python process initialization/spawn; v2
asserted the wrong CLI refusal channel; v4 used a memref.cast custom syntax not
supported by this parser. Later profiles correct those test errors without
changing old results or promoting failed runtime contexts.

Replay in a **new** output directory by copying regression-v1/run.py there and
running it with the explicit primary path and Python environment. The runner
requires `-I -S -B`, selects the recorded xDSL site directory and unchanged
original producer (override MERLIN_ADMISSION_ORIGINAL if relocated). Ordinary
pytest collection skips this opt-in module; the retained isolated run skipped
zero cases. Old absolute Linux capture references are historical on a laptop.
No portability or original-bound native replay is claimed.

## Paper wording / outstanding obligations

Proposed narrow statement: “The revised trusted host admission rejects file
resource metadata, resource-valued attributes and unmodeled function-value or
symbol-reference routes before translating the supported complete programs.
The scalar route permits only signature-matched declared external calls. The
shape route additionally permits direct internal calls checked by its separate
finite-depth correspondence. Adversarial parser/producer tests exercise these
restrictions.” Do not state that Lean proves this Python admission policy.

Still required: independent review of this patch and exact parser/projection
assumptions; root-approved native context; new exact-source Lean qualification
and compiler-valid mutation checking; Q's enforced gate worked example; W's
actual-after mapping. Primitive/control/effect/allocator/alias/completion
premises and declared D5 remain separate. No numerical, hardware, full-model,
all-graph or human-review approval is supplied.
