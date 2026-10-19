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

Run `python -m pytest packages/merlin-experiments/tests/test_public_development_cohort.py`
for V2 eager/reference, refusal, resource-accounting and ordinary capture checks.
The three installed-toolchain cases cover a padded/strided spatial transition,
the connected vision/prefix/action chain and a shared-RHS sequence. No simulator
or validation model runs in these tests.
