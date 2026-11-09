---
title: Explicit operation and form coverage
kind: guide
status: current
owner: targetgen
last_verified: 2026-11-09
related: [targetgen]
code_refs:
  - src/merlin/targetgen/opset_contract.py
  - src/merlin/targetgen/op_form_regimes.py
  - src/merlin/targetgen/opset_coverage_input.py
  - src/merlin/experiments/phase0/op_cell_synth.py
  - build_tools/scripts/check_opset_coverage.py
  - build_tools/scripts/synth_op_cell_capsules.py
---

# Explicit operation and form coverage

These tools distinguish an operation with no declared home from a declared host,
accelerator, structural or undetermined operation. They compare public capsule
metadata against independent form-axis obligations. This is a coverage diagnostic,
not a compiler, hardware or numerical qualification.

The core owns the vocabulary, capability check, form classifier and coverage join.
The experiments namespace owns builder execution, independent integer
golden falsifiability and recipe proposals. Importing the core does not import the
optional generator or load its source files.

## Explicit input and output

`check_opset_coverage.py --input REQUEST.json` accepts a
`merlin.opset_coverage_input.v1` document containing:

- `target`: the explicitly selected target identity;
- `contract`: the selected capability contract, including compute units;
- `facts`: its source-derived address-space facts;
- `capsules`: only the admitted public capsule metadata (each has `label: public`);
- optional `excluded`: existing public member names deliberately excluded;
- optional `dtype`, `windowed_ops` and `epilogue_stages`: declared source/readout
  inputs, not inferred from names or sample outputs.

The report records the input file SHA-256. The caller remains responsible for the
source and target identity of supplied facts; a metadata hash is not hardware
provenance. Empty/malformed schemas, hidden or duplicate members, stale exclusions,
unknown homes and uncovered cells cannot silently pass. `--ratchet FILE` permits
explicit existing debt, but stale entries also fail; the tool never writes debt.

For proposals, install `merlin[experiments]` and run:

```sh
python build_tools/scripts/synth_op_cell_capsules.py \
  --input REQUEST.json --descriptor target_experiment.yaml \
  --output out/artifacts/probes/opcells.json --max-capsules 24
```

The request additionally supplies an explicit `datapath` numeric policy for
`corpus_spec.derive_binding`. The output must be new and disjoint from retained or
descriptor-selected source corpora. Input and descriptor bytes are rechecked before
publication. No profile siblings, holdout files or canonical output paths are
implicitly discovered. This tool does not create run identities or execute targets;
normal Phase 0 generation owns materialization, source closure and run identities.

Every proposal is built, checked for primitive capability and classified from its
actual emitted metadata. A renamed or fixed-geometry builder cannot earn coverage
for the requested name/shape. Unknown compound-region admission, unavailable
writers, unfalsifiable integer goldens and unplaceable cells remain visible debt.
The explicit capsule budget raises instead of silently truncating. An optional
`--max-output-elements` certification ceiling uses the built interface's output
extent; it caps expensive proposals at L2 rather than claiming certification.

## Conservative geometry

The tile-tail classes are a diagnostic taxonomy. A single square-array mapping
cannot prove rectangular-array occupancy. Plain matrix geometry uses named/unique
operand roles and matching reduction dimensions; windowed, transposed or ambiguous
views do not inherit it. General reductions need an explicit supported axis.

Capacity and transfer classifications require `form_evidence` from layout analysis:
`operand_rows`, `accumulator_rows`, and `max_contiguous_bytes`, each with a numeric
`value` and `source` description. No input-size-as-accumulator or largest-dimension-as-
DMA assumption is made. These metadata records are not executable proofs. Without
such evidence the axis is `UNDERIVABLE`; synthesis does not fabricate it.

Duplicate cell signatures are reported, never deleted automatically. Sharing a
form signature does not make numerical, whole-model or independent-source tests
redundant. Defaults and existing corpus generation remain unchanged.

The current distribution is `merlin`; install the `experiments` extra for the
optional generator dependencies. `targetgen.opset_coverage_input` reads the
explicit metadata request, while `merlin.experiments.phase0.op_cell_synth` owns
actual proposal construction. Namespace ownership remains distinct even though
these sources share one wheel.
