---
title: Construct fresh connected public development sources
kind: guide
status: current
owner: experiments
last_verified: 2026-10-19
related: [phase2_corpus_search, phase0_phase1_setup]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase0/development_sources.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/workload_roster.py
  - src/merlin/targetgen/capsule_source.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/development_cohort.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/development_capture.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/development_capture_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/development_reference_supplement.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/development_shape_supplement.py
  - packages/merlin-experiments/src/merlin_experiments/capture_execution/native_integer.py
  - packages/merlin-experiments/src/merlin_experiments/capture_execution/readonly_runtime.py
---

# Construct fresh connected public development sources

`development_sources` materializes parameterized PyTorch programs from explicitly
public development declarations. The checked-in [small source roster](../../experiments/reference-data/phase0/public-development-sources.json)
contains independent decode and prefill geometries. It contains no validation
model shapes, weights, captures or outputs. These are connected pattern tests;
they do not establish reduced full-model compilation or a complete shape spectrum.

The first factory constructs grouped-head integer mixing. Each call appends new
K/V tensors to shared prefix state, maps several query heads to each KV head,
computes QK, applies the declared readout, then computes PV. The next call consumes
the updated cache. Every wide score, mixed result and intermediate cache remains
an observable output. The input domain is deterministic, signed and nonconstant.

The explicit arithmetic is signed i8 multiplication with exact i32 accumulation,
round-half-up division by a caller-selected power of two, then saturation to i8
between QK and PV. Overflow bounds cover arbitrary signed i8 inputs, including
the rounding offset. This does not select or search a quantization recipe. It
does not include softmax, projection or normalization and must not be represented
as a full attention implementation. An incompatible selected software contract
must refuse admission rather than reinterpret the source arithmetic.

## Ordinary source and capture connection

For a fresh destination with an existing parent:

```python
import json
from pathlib import Path
from merlin_experiments.phase0 import development_sources

declarations = json.loads(Path(public_roster).read_text())["declarations"]
record = development_sources.stage(declarations, Path(fresh_destination))
```

`stage` writes self-contained loaders and a workload roster, then calls the
existing `workload_roster.stage` owner. Source and declaration digests are retained.
Its `performance_applications` roots are ordinary inputs for Phase 0 capture
selection, sealed execution and attestation. Those independent owners still
must issue their original grants; staging grants none.

For a bounded diagnostic using the selected installed capture runtime,
`PytorchRefSource.capture_loader(loader, "int8", already_quantized=True)` executes
the fresh loader and converts it through Model2MLIR. The ordinary writer consumes
`development_sources.capsule_entry(declaration, loader)` and the selected binding.
The entry explicitly selects `output_numeric_policy: captured_per_output_v1` so
each captured i32 result and i8 cache gets its own existing exact-integer policy.
The default historical mixed-output refusal remains unchanged. Tolerances and
captured dtypes are never widened. The integration test checks the complete
fresh capture against a separately implemented NumPy integer reference before
the ordinary writer publishes any capsule.

These are source/reference products. Private exclusion, exact selected software
and hardware correspondence, required accelerator execution, complete compiler
outputs and qualified timing remain separate gates. No performance objective is
issued by the factory, and no validation workload is executed.

## Tests and current limits

Run `python -m pytest packages/merlin-experiments/tests/test_development_sources.py`.
Torch is optional for source-only checks. The last test uses the explicitly
selected Model2MLIR runtime when available; it starts no simulator.

The selected CPU Torch build returns incorrect native `_int_mm` results for the
retained `head_dim=1` regression. That case remains an expected failure against
the independent reference and is not numerically qualified. A new environment
that fixes it produces a strict unexpected pass for review. No input padding,
golden substitution or tolerance change hides the failure.

The V1 grouped-head factory retains its original four-call, 32-query-head limits.
V2 additionally constructs plain/transposed-RHS GEMMs, padded and strided NHWC
convolutions, spatial-to-pointwise transitions, multiple contractions sharing an
RHS, prefix reuse with recurrent action state, and a closed vision-to-prefix-to-
action chain. Spatial sources explicitly pack windows before integer GEMM;
native accelerator convolution support remains unqualified. These are connected
patterns, not complete reduced-depth versions of the headline models. Softmax,
normalization, physical residency and qualified accelerator timing remain gaps.

## Freeze a public architecture-informed development spectrum

The checked-in [public architecture inputs](../../experiments/reference-data/phase0/public-architectures/)
contain only public configuration bytes, upstream identities and a disclosed
operator envelope. No weights, validation captures or protected answers select
members. Generated loaders are fresh PyTorch sources informed by architecture
widths and layouts; they are not intercepted model graphs. Having inspected
these architectures makes their use development, not unseen-model evaluation.

`development_envelopes.derive` selects joint tuples using the existing spectrum
planner, both RHS layouts, independently declared convolution geometry and
grouped-head/state patterns. `development_cohort.capacity_requests` reuses the
existing capacity-family owner. It preserves the complete original request and
refuses numerical/epilogue forms without an exact source factory. The wrapper
does not weaken a frozen selected numerical contract or admit those sources.

An explicit bounded command prepares sources and independent NumPy references:

```bash
python -m merlin_experiments.phase0.development_cohort_cli \
  --public-architectures experiments/reference-data/phase0/public-architectures \
  --facts "$FACTS" --target "$TARGET" \
  --capacity-policy experiments/templates/phase0/performance.yaml \
  --capacity-sweep PX --budgets "$BUDGETS_JSON" --output "$FRESH_OUTPUT"
```

The budget document requires `max_members`, `max_source_bytes`,
`max_reference_elements`, `max_reference_work`, `max_total_reference_work` and
`max_total_reference_elements`, all positive integers. Work means exact logical
scalar products, not simulator walltime or target cycles. References are refused
before allocation when ceilings would be exceeded. Source-selection denials,
unsupported arithmetic and expensive reference members stay in `manifest.json`
and the per-member `coverage.json` matrix. Complete sources remain available
where only the reference cost was denied. The output is made read-only and pins
producer code, configs, facts, original capacity policy, each ABI, arithmetic,
source/reference bytes and complete-call performance boundaries. Store the
printed manifest SHA independently; `development_cohort.verify` requires it.

The resource report prices logical payloads against the original facts-derived
capacity basis. It supplies a conditional useful-compute component and leaves
the complete roofline UNKNOWN. It does not infer physical transfers, lifetimes,
packing costs or legal overlap from summed tensor volumes. Accelerator-feeding
packing, layout, transfer, issue, readout and state belong to the complete call;
Merlin host arithmetic must also be measured and reported separately.

Selected bounded members connect to the ordinary capture and capsule writer:

```bash
python -m merlin_experiments.phase0.development_capture \
  --sources "$FROZEN_SOURCES" --manifest-sha256 "$MANIFEST_SHA256" \
  --ordinals 0 1 --max-work 20000000 --max-elements 2000000 \
  --output "$FRESH_CAPTURE_OUTPUT"
```

This diagnostic checks every captured input, output and output dtype against the
independent exact reference, requires all contractions and zero opaque nodes,
and uses the writer's existing per-output numeric policy. It records refusals
without issuing sealed capture, private exclusion, software/hardware admission,
baseline compiler correctness, timing qualification or a search objective. Large
logical cases remain available for separately budgeted capture/compilation;
freezing their loaders alone is not execution coverage. The ordinary diagnostic
capture cache may be reused; the report explicitly leaves capture freshness
UNKNOWN. Recomputed independent references are fresh, while this command cannot
issue the sealed capture owner's freshness grant.

To reuse complete frozen reference products, explicitly add `--frozen-references`
and one `--reference-supplement INDEX@SHA256` per independently pinned supplement.
The intake reopens the unchanged source manifest and every complete NPY output,
then checks captured inputs, outputs and types against that retained evidence.
The report labels reference reuse rather than fresh reference execution. It
retains a cost decision for every original member, including unavailable sources
and references, numerical refusals and unselected cases. `--max-seed-bytes`
screens the unchanged eager seed expression's allocation lower bound;
`--max-contractions` bounds logical graph expansion. Neither is an upper bound
on process memory, emitted instructions, compilation time or executable size.
Compiler and source-sealing gates still apply independently.

The ordinary sealed CPU runner accepts the explicit versioned worker option
`{"source_mode": "native_integer_v1"}` for checkpoint-free integer source. It
captures the original frontend and unchanged materialized graph; it selects no
recipe, precision conversion, staging or tolerance. All captured intermediate
types must be integral, every input and output retains its signed i8/i32 ABI,
and every output receives an exact integer comparison contract. The existing
preselection, isolated execution, byte inventory, independent replay and
attestation owners still run. Legacy float and recipe-based commands retain
their original behavior. The mode grants source closure, not reviewed target
arithmetic, compiler correctness, private exclusion or performance admission.
`SealedCaptureSource` selects this route through its `source_mode` config and
requires `already_quantized=True` to preserve the caller's own integer source.

An operator can explicitly select `readonly_runtime.select(...)` against an
existing CAS entry and independently pinned marker, then pass its record as
`runtime_reuse` to `capture_selection.select` or `SealedCaptureSource`. This
cache-hit-only route mounts verified runtime bytes read-only. It creates no
source-store lock, hard link or environment copy; all guest placeholders and
scratch belong to the current task. A complete inventory mismatch refuses the
hit despite an existing marker. The optional V2 composition defaults to sixteen
absent leaves and 1 MiB. Explicit serialized `overlay_limits` may select at most
64 leaves and 16 MiB from independently pinned direct files. The source CAS and
effective selected-runtime identities are recorded separately; different source
identities require this explicit composition and complete effective equality.
It reconstructs flat leaf directories using empty task-owned placeholders and
read-only file mounts, and verifies the **complete effective namespace** against
the original selected runtime trees and library identities. An absent flat leaf
directory may be created only beneath an existing CAS ancestor, using its pinned
source-parent mode; the complete namespace check still applies. It never repairs
the source CAS, replaces an existing source member, drops missing dependencies,
or treats marker presence as sufficient evidence. Original writable-CAS
selection remains the default.

The retained infrastructure check exercised the original sealed protocol on one
public pointwise/state transition with three live-outs. Its fresh capture and
independent replay passed, and all complete outputs matched the frozen independent
reference exactly. The original writer accepted its sealed source attestation,
then refused target numerical admission because no selected internal-arithmetic
proof was supplied for its accelerator groups. This is a source-mode connection,
not complete cohort admission, a model qualification or a performance result.

Run `python -m pytest packages/merlin-experiments/tests/test_public_development_cohort.py`
for V2 eager/reference, refusal, resource-accounting and ordinary capture checks.
The three installed-toolchain cases cover a padded/strided spatial transition,
the connected vision/prefix/action chain and a shared-RHS sequence. No simulator
or validation model runs in these tests.

## Complete attention semantics and the remaining numeric gate

`--include-complete-attention` explicitly adds ten public semantic attention
members to the preparation command. Their separate schema is
`merlin.phase0.public_attention_semantic_source.v1`. They express QK, scaling by
the head dimension, causal or explicit Boolean masking, FP32 softmax, and PV,
including GQA head mapping and prefill followed by two decode calls with growing
KV state. The public envelope includes context and vision regimes and an upper
control; expensive references remain budget-denied members. This is attention
semantics suitable for testing a future FlashAttention-style transformation,
not a FlashAttention implementation or selected full-model numerical recipe.

The source supplies real signed-i8/i32 QK work. It explicitly uses FP32 for
softmax and PV in a semantic diagnostic. The selected target's probability/value
conversion, floating equivalence policy and execution contract have not been
provided to this factory. It does not copy the separate integer-softmax model,
invent probability quantization, select a new tolerance or assert full Gemmini
PV offload. Those missing numerical gates are recorded per member and in the
coverage matrix. Existing source-only integer patterns retain their arithmetic.

Only final attention outputs and carried KV state are required live-outs. Score
and probability matrices remain intermediates: making them observable would
force their materialization and prevent the optimization under test. Coverage
reports logical materialized score bytes and potential score elimination,
online softmax, QK/PV tiling, GQA layout, KV reuse and lifetime mechanisms. No
online-softmax reassociation is admitted merely because the source exists.

The ordinary capture diagnostic retains complete source, interface, typed ABI,
inputs, independent reference and floating-error observations in a
`pending-capsule.json` closure. Exact carried integer state must match. It
explicitly refuses the ordinary capsule writer while the floating policy is
UNKNOWN; a small observed error cannot issue a tolerance or equivalence grant.
Source tests cover an exact dyadic uniform-softmax case without a tolerance,
future-token exclusion and independent GQA head-group effects. Run
`python -m pytest packages/merlin-experiments/tests/test_development_attention.py`.

## Supplement expensive exact references without changing the source cohort

`development_reference_supplement` can close a source's old reference-cost
denial under a separately stated memory/work admission. It leaves the original
manifest, source files, seeds, input/output ABI, numeric policy and objective
untouched. Every old member remains in its index, including unselected and
unsupported members. Select explicit original ordinals and a fresh destination:

```bash
python -m merlin_experiments.phase0.development_reference_supplement \
  --sources "$FROZEN_SOURCES" --manifest-sha256 "$MANIFEST_SHA256" \
  --ordinals 24 25 110 84 --budgets "$SUPPLEMENT_BUDGETS" \
  --output "$FRESH_SUPPLEMENT"
```

The budget document requires positive integral `max_workspace_bytes`,
`max_output_bytes`, `max_total_output_bytes`, `max_member_work`, `max_total_work`,
`max_dense_source_bytes`, `member_timeout_s` and `total_timeout_s`. Work/output
reservations are charged before execution and retained on failure. Each member
uses an existing shared CPU lease, a bounded walltime and complete output NPY
maps. Original exact i8/i32 matmuls, NK RHS, spatial/pointwise transitions,
shared-RHS sequences and grouped QK/PV/KV are supported. Mixed-numeric complete
attention remains outside this exact-integer path.

The implementation generates original seed coordinates in panels and reuses
`merlin.model_check.kernel_reference._int_matmul`. That owner uses float64 BLAS
only when its integer bound proves products and sums exact, otherwise integer
arithmetic. This is a reference speedup, not a new floating numerical model.
Blocked convolution derives padded spatial coordinates without a full im2col
allocation. Readout, each complete wide result, narrowed state and required KV
live-outs remain exact. Recorded computed scalar products must equal the
original complete program; output sampling cannot close a denial.

Store the file-computed `index.json` SHA independently.
`development_reference_supplement.verify` checks that pin, the unchanged full
source manifest, all member bindings and every complete output's bytes, shape
and dtype. Reopening prior evidence does not make it a new measurement.

The reference panel bound excludes Python/BLAS overhead and mapped output
residency; measured process RSS and those components are reported separately.
The unchanged eager Torch loader can still exceed its separate memory budget:
its i64 seed expression alone requires two full temporary arrays. A cheap
blocked reference does not qualify that eager capture, an accelerator schedule,
the compiler, a complete-call floor or timing.

## Add missing public vision, prefix and expert projection regimes

`development_shape_supplement.derive(public_inputs, base_manifest_sha256=pin)`
returns an additive audit and fifteen ordinary source requests. Feed its
`requests` to `development_cohort.freeze` in a separate destination, with the
audit as provenance. It neither regenerates the old member choices nor changes
their objective. Large references can use the separate blocked supplement.

The public config omits vision depth and intermediate width. Their values come
from byte-preserved selected Transformers configuration and modeling source in
`public-architectures/defaults-sources`, with version and hashes in
`defaults-provenance.json`. The parser checks the constructor route, default
assignments, four vision projections, MLP widths and connector equations.
Unknown defaults or unsupported source forms cannot silently select familiar
widths. These are defaults of the selected public implementation, not proof that
every historical Transformers release interprets a checkpoint identically.

The selected connector reduces 1024 patch tokens to 64 through pixel shuffle by
four and projects width 12288 to 960. The postconnector image-plus-max-text
envelope is therefore 112 tokens; extra state/special-token contributions remain
UNKNOWN here. Language shapes at 1024 and 1072 rows are retained as disclosed
preconnector upper controls, without invented occurrence counts. Additional
sources cover vision width 768 and its source-proved MLP, above-upper vision M,
and expert width 720. Prefix encoding is counted once when reused; ten denoising
invocations are distinct from 32 decoder calls. Expert depth/MLP remain UNKNOWN
without the expert implementation. Structural occurrence proxies guide sampling
only, and omit unknown or control-only counts.

These projection sources deliberately omit residual, bias, activation and
normalization context. They supplement the connected patterns; they do not
establish the smallest complete representative models or full-model coverage.
