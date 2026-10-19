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

The factory does not yet provide convolution transitions, complete reduced-depth
decoder/vision/action models, data-dependent action recurrence, source-spectrum
selection, physical residency or accelerator timing. Its static source depth is
bounded to four calls and 32 query heads. Extend the public envelope and source
patterns explicitly, retaining refused/budget-denied members separately from
the admitted performance objective.
