# AGENT.md — src/merlin/targetgen

## Purpose

TargetGen pipeline: ingest -> evidence -> synthesize -> generate -> validate. Turns a
target's local docs/examples/source into human-reviewable plans and a generated
`merlin-target-<name>/` repo skeleton.

This directory also owns shared target contracts, corpus derivation primitives and compiler/runtime
interfaces. The same import namespace is extended by `merlin-experiments`, which owns agent
launching, capsule grading, golden generation, sandboxing and certification. Their source lives in
`packages/merlin-experiments/src/merlin/targetgen/`, not beneath the legacy core symlink.

## What belongs here

- `artifact_bundle.py` supplies explicit profile-bound structural checks for a
  draft artifact composition. File identities and logical-buffer ordering grant
  no artifact ABI, physical alias, capability, synchronization or execution
  proof. ABI `0.2` has no normal executor/certification adapter. Default package
  invocation refuses that version; typed profile permission admits structural
  emission only. ABI `0.1` and its existing text-entry aliases remain unchanged.
- `package_runtime.scoped_package_executor` accepts explicit caller-owned build
  and entrypoint execution services. It never discovers an optional evaluator;
  invocation without the scoped owner during a guarded run refuses, including
  threads that did not inherit the execution context. Normal unguarded calls
  retain their existing package contract and runtime behavior.

- `cli.py` / `pipeline.py` — the deterministic, LLM-free entry points.
- `ingest/` — record inputs as a SourceManifest (no crawling, no vendoring).
- `evidence/` — deterministic file discovery + keyword concept detection.
- `synthesize/` — emit the five plans (target_contract / dialect / runtime_adapter / zephyr / llvm).
- `generate/` — write the target-repo skeleton + xDSL / MLIR / runtime adapter / Zephyr / LLVM.
- `validate/` — schema + structural checks and the validation report.
- `oracle_policy.py` / `program_engine_policy.py` — read-only engine and tier metadata. Advertised
  availability is not successful evaluator construction or a certification result.
- `elf_lanes.py` — shared execution-artifact/evidence vocabulary and static inspection. Required
  execution and negative static evidence remain distinct.
- `host_linkage_contract.py` structurally validates closed draft or reviewed contracts for a
  caller-selected math-archive digest and symbol roster. Only reviewed admission records a pending
  requirement; only an independently selected actual-link trace can prove the defining supplier.
  Neither source-call routing nor numerical
  equivalence follows from that trace or from `numerical_contract` metadata.
- Opt-in integer-reduction `host_capabilities` source-body admission reuses the
  closed frontend sum/cumsum/paired-min recognizers. It requires an explicit
  caller-owned selected-index compiler observation and every exact parsed source
  occurrence; absent context is unknown. Structural validation accepts drafts,
  while only reviewed, pinned declarations can admit host placement. The private
  source/build witness remains mandatory and numerical equivalence unproved.
- Opt-in prepared f32 maximum source-body screening accepts only an exact
  `aten.amax.default`/`linalg.reduce` typed source roster, selected index-width
  observation and the closed unflagged `arith.maximumf` body. Structural
  validation does not review a declaration; the independent private linked
  witness remains mandatory. In particular, signed-zero and reduction-order
  equivalence to PyTorch are not established.
- Opt-in prepared tensor-index host source-body declarations name one exact
  Boolean extension, mask-sum or literal-indexed-extract role. Admission is
  conditional on a caller-reproved closed source/trace record, selected index
  width and every parsed typed ordinal. A separate private source/build witness
  must reprove the captured bytes; this is no original/compiled numerical grant.
- `group_capsule_entries.py` — deterministic capture-group restatement used by compiler diagnostics.
  Corpus writes, promotion, grading and measured capacity probes belong to experiments-owned
  `group_capsules` and `store_probe`, which keep their stable import names.
- `golden_provenance.py` — host-only metadata projection for prompt/evaluator agreement, not golden
  evaluation or a public answer loader. Its access identity stays withheld even in core-only installs.
- `conformance.py` — requirement derivation from explicitly supplied constructed-tier evidence;
  optional observers run at the historical observation point, never through default discovery.
  Evaluated conformance/default corpus selection belong to `merlin_experiments.corpus.admission`.
  Import evaluated operations from that experiments owner; core has no lazy workflow exports.
- `capsule_inputs.py` — host-only canonical input bytes, values and deterministic leaf materialization.
  The evaluator reexports these implementations; numerical answers and derived MX intermediates stay
  in experiments. Input access retains the golden grader mask and grading-source commitment.
- `corpora.py` reads its registry through the shared package-data resolver: checkout resources in
  development, bundled resources in a wheel. The registry is metadata, not the external corpus itself;
  missing or malformed metadata must raise rather than imply an empty workload.
- `TargetExperiment.resource_path()` selects descriptor-owned task, harness and bundle
  resources. Use it in parsed-descriptor consumers instead of reconstructing sibling paths.
  Installed callers pass `source_root` to `load_target_experiment`; the parsed descriptor
  retains that owner for corpus, contract, grant and host-lane path resolution. Omission
  preserves legacy checkout discovery, not portable installed execution.
  It rejects absolute member names and parent traversal, but deliberately preserves aliases;
  sandbox grant admission still owns symlink/access checks. Optional `resources_root`
  selects a repository-relative or absolute authored-resource directory. Without it,
  resources remain descriptor siblings. Optional `task_root` separately selects the
  authored prompt directory for `task` and its descendants only; other resources keep
  their shared root. Optional `contracts_root` similarly selects `contracts` and its
  descendants, including a declared curated harness, independently of generated bundles.
  All ownership fields use the same lexical validation and never
  probe filesystem existence. Early startup uses `descriptor_resources_root()`;
  release preparation must remove source ownership pointers after staging self-contained resources.
  `resolve_resource_path()` is the shared lexical resolver for parsed descriptors and
  early operator-input inventory; curated harness bytes must be frozen even when they
  are provided as toolchain bindings rather than ordinary bundle grants.
- ISA/RTL cross-check evidence follows an explicit `hardware_spec.hwbringup_set` before
  the legacy resource-root contracts directory. Missing explicitly selected evidence
  must not silently fall back to an old resource directory. Both selected bring-up
  sets and legacy parent directories may contain shipped green-card documents.
- `publish.py` exposes help and inspection without loading the optional JSON-Schema validator.
  Actual manifest validation requires the declared `targetgen` extra and never skips validation
  when it is absent.
- `rtl_checks.py` owns common reports and explicit selected-support dispatch. The
  backend's `rocc_semantics.rtl_checks` owns protocol screening, fact projection,
  TRACE assertion compilation and matching rendering. RoCC transport alone never
  selects another accelerator's ordering or geometry. Missing or malformed support
  refuses; the advisory CIRCT wrapper records unavailable evidence and still runs
  the scientific oracle. The runner selects one owner for all three operations;
  compiled assertions are not cached by mutable capsule names or object identities.
- `rtl_engine_policy.gsim_runtime_slot` admits synchronous native runs under one per-user
  mutex from the union of kernel slot reservations and an upper bound on same-user native
  plusarg processes. It discounts a duplicate only when a unique kernel FLOCK owner has a
  matching open slot FD and one stable same-UID direct native child with complete plusargs,
  verified before and after the join; ambiguous or legacy holders without that proof count
  separately. A single PID holding multiple slots lacks a child-to-slot pairing proof and may
  therefore use fewer than five workers. The caller's pending reservation always counts. A
  verified same-user live PID whose argv remains empty after bounded stabilization counts
  as one potential native;
  missing, unreadable or malformed `/proc`
  evidence refuses admission. This is conservative: unrelated empty-argv processes or a
  different simulator using the same plusargs can consume capacity, and uncoordinated
  launches after a census cannot be prevented. Do not describe the advisory protocol as
  global enforcement for processes that do not participate.
- `program_oracle.emit_bundle(target=...)` resolves `runner.program_emitter` from an
  explicitly selected OOT support provider when named-program assembly or tensor layout
  is requested. Its `path` is provider-contained; optional string `args` are target-owned
  policy. Core forwards the existing `--program`/`--inputs`/`--out` JSON protocol in the
  model environment, never importing the helper. No in-tree fallback or target-specific
  encoding flag remains. Verdict-cache source inventories include helper dependencies
  and the declaring contract; this is not complete frozen subprocess qualification.

`compile_only_execution` runs the ordinary scoped package lowering, LLVM/object
compiler and linker with a static original ABI and explicit build/ELF admission
services. It retains the linked entry and reopens exact source/tool selections,
package/contract membership, products and actual invocations. It accepts no
capsule, tensor values, golden or execution service. Semantic, index, resource
and complete-output obligations remain UNKNOWN; independent source authority
and the required grading denominator belong to the phase owners. Transport
reports cannot issue those authorities. A product overwritten after an observed
invocation remains unavailable rather than gaining an evidence exception.

## What does not belong here

- LLM/API dependencies — TargetGen is deterministic.
- New experiment orchestration or grader implementations. Use their owning optional distribution;
  do not copy them back into core to satisfy an import. Existing upward dependencies are tracked
  migration work, not a pattern to extend.
- Claims of automatic correctness. Outputs are human-reviewable; non-toy synthesis is
  flagged `requires_human_review: true`.
- The Merlin core dialects or runtime ABI (those live elsewhere in the repo).

## Interfaces

`input_palette.py` describes explicit deterministic input-only stress patterns
along static tensor axes. Values and numerical landmarks must be exactly
representable in selected storage; it never computes answers or changes oracle
order/tolerances. Normal MLIR builders and builtin PyTorch sources consume it.
Connected builtin attention/residual/norm, MLP/residual and conv/residual/pool
sources remain closed programs through ordinary sealed Model2MLIR conversion;
they do not admit arbitrary loaders, models or captures.
`component_sources` owns those closed compositions and the producer→RNE/clamp
quantizer observer. Its source scales are finite exactly represented f32 values,
with a power-of-two quantizer scale; signed i8 bounds come from the format registry.
`capsule_results` checks every multiple-result ABI against the actual parsed
program signature and complete result values before ordinary publication. The
observer publishes code values as f32; exact source witnesses do not grant a
different candidate comparison, signed-zero policy or per-output integer policy.

`compiler_library.py` binds explicit host-reviewed public leaf APIs, dependency
members and resource bytes for independent compiler packages. Candidate metadata
cannot grant access. Direct-import closure is not dynamic/interpreter isolation;
the owning experiment must independently qualify its minimal runtime boundary.

- Produces artifacts that validate against `merlin/schemas/*.schema.yaml`.
- `pipeline.build(...)` returns a `BuildResult`; `cli.py` exposes `build` and `inspect`.
- Consumes `merlin.common` (paths/io/yaml/artifacts/schemas) and `merlin.validation`.

## Invariants

The opt-in literal bucketize host source-body contract admits only an exact
typed prepared count form with a selected index-width premise. The private
capture trace and linked-build witness must independently agree with it;
this selected placement contract does not establish PyTorch numerical equivalence.

- `plugins.resolve_support()` is the shared explicit-selection check for runtime
  plugins and `load_declared()` target tools. Reference or generated-home metadata
  alone cannot authorize executable support. `load_module(root, ...)` remains the
  lower-level explicitly supplied-root interface, with provider containment checks.
- `target_registry.load_matrix_contract()` reads only the selected support
  provider's declared, contained `matrix_contract` resource. Missing selection,
  declarations or malformed metadata never use a checkout-global fallback.
  Corpus geometry and instruction classes use that target's `matrix_lowering`
  plugin; the support identity must not be guessed from a unit name.
- Deterministic: same inputs -> byte-identical YAML artifacts.
- Targets implement adapters; TargetGen never generates an independent runtime model.
- Do not generate dialect ops directly from instruction names — synthesis stays conservative.
- Every subdirectory contains an AGENT.md.

## Two caches, and why they are different things

Whole-model capsule export requires the captured external weights file and its
argument manifest before creating or modifying capsule outputs. Both remain separate
sidecars; export copies bytes without dtype conversion. Op capsules do not acquire
weights through this path. File-presence checks are not safetensors validation,
numerical qualification, or an atomic publication guarantee under concurrent writes.

Framework capture has a separate lifecycle: `capsule_source.PytorchRefSource` executes
the capture, while `capture_cache` owns source observations, per-slot locking and atomic
completion-pointer publication. Use that owner directly; the former private cache helpers
in `capsule_source` are not compatibility exports. Importing `capture_cache` is inert and
does not import model frameworks or normalization implementations.
Shared capture slots use a per-slot process lock and unique retained attempt directories;
only the parent publishes a completion pointer after normalization and artifact reads.
Worker metadata is not a cache commit. Old entries without that pointer are misses,
and unavailable locking bypasses the shared slot using a distinct retained build directory.
These measures prevent concurrent attempts from mixing outputs. Versioned structured
keys bind direct Merlin worker, parser and normalization source bytes and are rechecked
before publishing or returning a shared-cache result. Unreadable owners disable cache
use; changed owners refuse the transaction. Framework/importer installations, transitive
dependencies and arbitrary loader input data still lack complete byte attribution.
Observed loader Python sources are rehashed before cache reuse and parent publication;
missing/malformed legacy observations cannot authorize a cache hit. These observations
are not a complete loader closure and do not prove which bytes Python executed.
Do not claim complete source identity or numerical qualification from these checks.

A grade reuses work at two levels, and conflating them is the mistake to avoid.

- `build_cache.py` — **a BUILD cache.** An ELF is a pure function of the emitted program, the command
  buffer, the operands, the recipe, the toolchain and the code that compiles them; all of that is in
  the key. A restored build is still executed and still judged, so the worst a wrong hit can do is run
  the wrong program. It restores the whole generated directory, not just the executable, because the
  agent reads what is in it.
- `tier_cache.py` — **a VERDICT cache.** It carries a tier's *result* for bytes already certified, so a
  wrong hit asserts an unearned verdict. Every rule in it is written in the direction of re-running,
  and it must stay that way.

Measured on `merlincirct_g4p1_biasabi_20260906` (819 screen-tier executions): the screen spent 4.06 s
median building against 0.153 s simulating — 27x more compiling than simulating, 3,981 s of build in
one run. That is why the build cache exists, and why relaxing the verdict cache to chase the same
saving would have been the wrong trade (carrying a screen saves 0.153 s).

Provenance is scoped, not cached: `merlin.common.provenance.observation_scope`, opened by `run_suite`,
makes one grade see one revision per checkout. That is a correctness property first — two capsules in
one grade must not be attributed to different hardware revisions — and a large speedup second.

## Testing expectations

- `merlin/tests/targetgen/test_targetgen_toy.py` must pass (toy_npu build + inspect + simulate).
- Core metadata tests must run without importing optional evaluators. Moves retain registered
  access identities, and installed-package checks must not accidentally resolve checkout sources.

## Notes for future agents

`frontend_typed_add` binds original captured/public/registered schemas, actual
default observations, and every operand/result SSA type before selecting the
narrow unit-alpha, signed-integer, rank-two equal-shape form. It preserves the
result dtype; int8 addition never inherits a contraction's int32 accumulator.
The fixed native `torch_schema_defaults_observer` observes defaults only.
These source premises grant no numerical domain, purity or target support.

- `toy_npu` is the bundled reference target. Generated skeletons and support plugins are not
  qualified compiler submissions; consult target contracts and qualification records for status.

`capsule_builtin_source.py` renders builtin PyTorch templates with typed shape,
scalar and input-palette validation; `capsule_source.build_loader_src` supplies
its existing template registry. `corpus_builder.py` enforces common declaration
preservation across operation builders: palettes, epilogues, stimulus ranges,
oracle applicability and semantic coverage. These owners contain no target
schedule or workload-specific optimization.

`torch_zero_return_observer` observes the selected public native dispatcher's
zero-result schema and its actual empty-stack Python bridge. Admission also
requires fresh, complete original FX result metadata explicitly observing None;
missing metadata, container alternatives and incomplete historical slots refuse.
`frontend_operator_effects` preserves every original result slot and joins this
optional relation without granting purity, exception behavior or operation
ownership. The observer never invokes the operator or constructs a compiler.
