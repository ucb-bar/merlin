# src/merlin/compiler

Public request, artifact, backend and pass contracts live here. Keep them
independent of optional experiments, framework imports and target-specific
implementations. Backend implementation claims do not establish independent
numerical verification or hardware qualification. Preserve compatibility of
existing caller paths while moving their actual implementations to these owners.

`backend/interface.py` describes a compiler's input domain, implementation
families, fusion premises and target pipelines. Discovery consumes optional
OOT `describe_compiler()` declarations and checks them against implementation
patterns and lowering recipes. Declarations remain claims; preserve independent
admission, source custody, required coverage and explicit legacy status.
`backend/boundary.py` prepares the actual canonical Interface program before an
Interface backend sees a request. `lower/interface.py` owns the shared source
conversion and dialect parser. `planning/interface_candidates.py` discovers
source regions in Merlin and queries target implementations only after that
conversion. Source fusion and global planning stay here; Interface-to-target
conversion, instruction fusion and local physical strategies stay OOT.

Backend authors import the public records from `backend/interface.py` and the
canonical parser from `ir/interface_contract.py`. SDK30 exposes that reviewed
dependency set without the host driver or evaluator. `lower/interface_readout.py`
owns exact source fusion matching; `ir/readout.py` defines the numerical record.
`build/completion.py` discharges proposed device wrapper requirements only through
an original-caller platform validator; a backend cannot approve its own completion.

`diagnostics.py` owns compile-debugging requests, pipeline stage discovery, dumps and
stop reporting. Its lazy driver imports discover stages declared beside actual
execution; lower-level passes and IR contracts must not import this inspection
owner. Recording and occurrence identities remain in `ir/compile_trace.py`.

Start at `compilation.py` for request/artifact records and compile orchestration,
then `system.py` for system-level planning. `backend/` owns provider discovery,
source proposals, package admission and callable boundaries; it does not own the
target implementation. Scalar lowering and reusable LLVM finalization live in
`lower/native/`, and device lowering in the selected OOT repo.

Shared algorithms live in `transforms/` and `schedule/`. The schedule library
groups loop schedules, panels, shared operands, storage and spatial execution.
Composition discovery is in `schedule/api_catalog.py`; native invocation is in
`lower/native/invocation.py`. Keep hardware resources, command
effects and encoding in their declared providers rather than embedding a target
choice in a generic algorithm. Backend proposals do not authorize numerical changes.

Backends use the same selected Merlin base. Keep backend-specific transforms
and local alternatives to shared passes in the OOT package's `transforms/`
and `transforms/overrides/`. Select them explicitly in its lowering pipeline;
do not mutate a shared function globally or carry a private edited SDK copy.
Reusable API hooks belong in Merlin once, with existing semantic checks intact.

`lower/model/` orchestrates saved-model lowering, profiling and toolchain/codegen
policy. `lower/source/` owns typed source-family transactions, while `lower/host/`
owns shared host graph and storage lowering. `build/` owns artifact and image
construction; `build/model_preparation.py` is the shared preparation owner used
by execution adapters. Runtime must not become a second compiler implementation.

Shared source preprocessing is composed in `frontend/preprocess.py`; the exact
historical `llvmlower.passes_xdsl` module alias preserves selected-stage patches.
Its real algorithms live in `quantization/linalg.py`, `transforms/canonicalization.py`
and `ir/provenance.py`. Literal legacy
syntax repair lives in `frontend/repairs.py`. Keep numerical and effect contracts
there; preprocessing observations are source accounting, not semantic proof.
`frontend.preprocess.implementation_paths()` pins the complete former source
envelope and actual selected stage callbacks for receipts. SDK26 exports those
actual owners; earlier releases must use their immutable source archives.

`ir/` also holds the IR helpers shared across the compiler and its tools.
`ir/xdsl_io.py` adapts xDSL dense float IO without converting hex bit patterns through
Python floats; its printer retains exact dense bytes, including NaN payloads and signed
zeros, and its `IR_LOCK` serializes in-process MLIR parsing (xDSL's parser is not
thread-safe). Grammar and shape checks remain xDSL-owned; this serialization boundary grants
no numerical equivalence or arithmetic rewrite. `ir/mlir_query.py` answers structural
queries on linalg MLIR. `ir/lowering_audit.py` owns opt-in exact-byte named-stage inspection
records. Lowering callers own stage serialization and explicit destination and sidecar
selection; these observations are not compiler certificates or a replacement for
frozen-source seals. Caller-produced compact stage views are explicitly non-executable and
bind the exact parent content hash. Large xDSL dense inspection payloads are stored as exact
raw bytes under content-addressed `tensors/` files within one audit. Stage descriptors bind
producer element types and shapes; identical bytes share storage across stages. Completion
rechecks generated tensor hashes. This is inspection storage, not safetensors conversion or
an executable reconstruction ABI. `ir/compile_trace.py` is the user-facing debugging switch
over the same recording points: a request (in `MERLIN_COMPILE_TRACE`, so child processes see
it) names stages to dump and one to stop after. Pipelines declare their stage names beside
the code that records them; never list them here. Only the session's own process raises the
stop (`StopAfterStage`, a `BaseException`), so a stop is never swallowed as a per-group
failure.
