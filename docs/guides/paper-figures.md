---
title: Generate RVV paper figures
kind: guide
status: current
owner: plotting
last_verified: 2026-08-31
related:
  - docs/reference/plot_style.md
  - merlin/benchmarks/rvv_paper/study_v2.yaml
code_refs:
  - merlin/python/merlin/plotting/rvv_paper_figures.py
  - merlin/python/merlin/compare/paper_full_model_ablation.py
  - merlin/python/merlin/plotting/cpu_host_experiment_figures.py
  - merlin/python/merlin/plotting/cpu_host_beam_figures.py
---

# Generate RVV paper figures

Export each receipt fingerprint immediately after its live controller run and notarize the mapping
outside the same-user mutable run tree:

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.compare.paper_measurement_controller \
  issuance-fingerprint out/<paper-run>/cells/<cell>/receipt.yaml
```

The explicitly transported notary has this closed form (one row per result run ID):

```yaml
schema_version: 1
kind: paper_external_issuance_notary_v1
study_sha256: <canonical-frozen-study-sha256>
fingerprints:
  <run-id>: <issuance-fingerprint>
```

Re-derive the structured report in a fresh process, then render figures. Both production CLIs
require the external notary; there is no default local-trust file:

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.compare.paper_report \
  --study out/<paper-run>/study.frozen.yaml \
  --results out/<paper-run>/results.yaml \
  --issuance-notary /externally/retained/issuance-notary.yaml \
  --output-dir out/<paper-run>

merlin-paper-figures \
  --report out/<paper-run>/paper-results.yaml \
  --study out/<paper-run>/study.frozen.yaml \
  --results out/<paper-run>/results.yaml \
  --issuance-notary /externally/retained/issuance-notary.yaml
```

The default output is a non-overwriting, timestamped directory under
`out/artifacts/paper-figures/k1/`. Every chart is exported as PNG and SVG, and `manifest.json` binds
the figures to the exact report, controller-rooted results seal, and frozen-study file/canonical
hashes. The results seal is version 3: each cell must point to a receipt issued by the trusted
measurement controller and a fresh process must match the separately notarized issuance
fingerprint. The controller executes and
replays the exact artifact-consuming command, derives samples from per-iteration monotonic spans,
captures stdout/stderr, and runs the source-pinned CSR/sysfs board probe. It also creates and
validates a native finalized AET child lifecycle
(`run_record.json` plus `logs/events.jsonl`). Non-agentic benchmark runs record token accounting as
not applicable; they do not manufacture zero-token agent telemetry. A hash
over editable `results.yaml` content alone—or a hand-written receipt with refreshed hashes—is
deliberately insufficient. For a board-local standalone session command, the canonical producer is:

The manifest also retains the report's exact kernel-swap candidate/eligible/routed table. A passing
XNNPACK or OpenBLAS swap requires `routed == eligible > 0`; the candidate count remains visible so
the paper does not imply that operations outside the frozen backend classifier were replaced.

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.compare.paper_measurement_controller \
  produce-result CONTRACT.yaml NEW_CELL_OUTPUT_DIR RESULT.yaml
```

For `target: k1`, this command must run locally on the RISC-V board (stage it over SSH); invoking a
K1 contract on an x86 host fails closed. The contract separately binds the exact executable and
frozen artifact. A distinct artifact must appear exactly once as `{artifact}` in the predeclared
argv. The command emits one closed semantic-session receipt, while latency, board facts, normalized
provenance, and the final PaperResult come only from the controller.
Before plotting, the tool re-derives the complete report from the retained `results.yaml`; a missing
seal or any edited latency, ratio, label, coverage row, or attribution fails closed. Rendering also
rejects a draft, an unresolved frozen study, or a report whose study hash does not match the supplied
study bytes.

Causal why/how is stricter than the descriptive comparison. An XNNPACK, OpenBLAS, hand-kernel, or
ExecuTorch result is a comparator, not an ablation control. A production claim therefore requires a
schema-v2 `paper_full_model_causal_evidence_manifest_v2` containing a separately frozen
`paper_full_model_ablation_pair_contract_v1` for the same model, precision, and core count. The two
arms are both Merlin: `merlin_ablation_control` disables the predeclared compiler-policy treatment,
and `merlin_frozen` enables the exact policy and package used by the paper matrix.

Each contract pins the capture, continuous-session identity, compiler/runtime, control and treatment
policies, packages, executable digests, build receipts, and a closed
`paper_compiler_transform_delta_v1`. The delta may name only the generic categories tiling/dataflow,
fusion/layout, register residency, instruction selection, and runtime/synchronization. Both arms
must have the same compiler source, runtime, and normalized build configuration; only the declared
policy components and their derived package/binary may differ.

Measure at least three pairs, with the frozen repeated complete-session sample count in each
controller issuance and a balanced alternating AB/BA order. Each pair compares the two issued-block
medians. Every run must retain a schema-v6 measurement-controller receipt
and its detached issuance fingerprint. The verifier replays every receipt and checks the exact
model/input/session identity, package and binary provenance, passing correctness, identical
functional output, board identity, CSR-derived VLEN, locked performance frequency, bounded thermal
drift, chronological AB/BA order, and a treatment improvement in both the paired median and a
majority of pairs. Verify the completed, still-pre-freeze manifest with:

```bash
PYTHONPATH=merlin/python .venv/bin/python -m merlin.compare.paper_full_model_ablation \
  --study STUDY.yaml --manifest FULL_MODEL_CAUSAL_MANIFEST.yaml
```

The top-level manifest is deliberately small and closed; each comparator record binds the ordinary
head-to-head identity and points to one pair contract/evidence bundle (the same verified pair may be
referenced by several comparator records for that exact model/precision/core cell):

```yaml
schema_version: 2
kind: paper_full_model_causal_evidence_manifest_v2
status: frozen
study_identity_sha256: <study identity excluding lifecycle/manifest self-reference>
records:
  - model: <model>
    precision: <precision>
    core_count: <cores>
    comparator: <backend>
    binding: <exact output of paper_attribution.expected_binding>
    binding_sha256: <canonical binding digest>
    pair_contract: {path: pair-contract.yaml, sha256: <retained-file digest>}
    pair_evidence: {path: pair-evidence.yaml, sha256: <retained-file digest>}
```

`pair-evidence.yaml` contains exactly the frozen number of pair rows. Each row has its declared
`pair_index` and AB/BA `order`, then closed `control` and `treatment` references to a normalized
PaperResult, controller receipt, and detached issuance fingerprint. Its summary is derived from
those repeated-session blocks; authored summary numbers cannot override them.

Then declare that manifest as `reporting.causal_attribution.path` before freezing the final study.
Its digest becomes part of the study. Schema-v1 synthetic micro-pairs remain accepted only by
`target: unit-test`; they have no K1 paper-claim authority. A comparator win is reported only when
the winning matrix result uses the exact treatment binary and all ordinary comparator/result gates
also pass. Missing, partial, duplicate, modified, incorrectly ordered, or non-improving evidence
leaves latency/parity/loss reporting available but labels an apparent advantage
`advantage_not_claimable`; no WHY/HOW card is produced.

The generator reads only `primary_end_to_end`; stage diagnostics can never enter a model-level
performance chart. Missing, failed, and not-run cells remain visible as marked cells. It produces
slide-style absolute latency with observed p05–p95 whiskers, head-to-head speedup, and per-backend
1-to-8-core scaling for each available precision/core configuration. Each claim-eligible win also
gets a paginated WHY/HOW card sourced from the re-derived sealed causal record; an unsealed narrative
cannot enter a figure. Backend colors and the cream/navy block style come from the shared plotting
module.

After the separate 16-cell CPU-host campaign is complete, render its full-fidelity resource-cost
and generic compiler-outcome views with:

```bash
merlin-cpu-host-figures \
  --campaign out/<cpu-host-campaign>/experiment.completed.yaml
```

The command emits two PNG/SVG figure pairs. `arm1_4_resource_cost` reports the median and observed
min–max across all four scheduled Williams blocks for provider input+output tokens, driver wall time,
and reconciled tool calls. `arm1_4_compiler_outcomes` reports L0–L3 pass/fail/not-reached counts,
sealed compiler-package size, deterministic selected-policy action count, and the final accepted
action's controller-measured marginal train/validation ratio against its parent policy. That ratio is
not multiplied into a synthetic whole-policy speedup. A failed or inapplicable search stays visibly
unavailable rather than becoming zero or 1.0.

The outcome reader reopens only the grader, compiler package, and trusted-search paths named by the
completed campaign record and verifies every retained digest before creating the output directory.
It validates selected actions against the frozen optimization-space catalogue and requires the
accepted sequence to be a nested width-one path ending at the selected policy. Its scope is the
generic development corpus only; it never opens a paper holdout. `manifest.json` retains every-cell
resource and outcome value plus the exact source paths and digests. The experiment uses
`subscription_notional`, so the renderer explicitly marks monetary cost unavailable and never
converts tokens or time into a fabricated dollar amount.

Generate the dedicated Arm3/Arm4 beam-search progression after the same campaign freezes:

```bash
merlin-cpu-host-beam-figures \
  --campaign out/<cpu-host-campaign>/experiment.completed.yaml
```

This command reconstructs every legal one-action child from the frozen generic action catalogue and
checks the screen, confirmation, promotion, winner, and independently empty convergence sweep against
the digest-bound trusted broker ledger. Every evaluation must have an exact passing terminal
request/receipt association. It emits `arm3_4_beam_coverage` plus one PNG/SVG tree per verified cell;
a typed treatment search failure is visibly unavailable and never becomes an empty or baseline tree.
A passing search with a missing child evaluation, extra evaluation, altered ranking, incomplete
receipt association, or mismatched observation digest fails before the output directory is created.

The beam reader aggregates only generic-family speedups. It does not dereference the policy/capsule
paths retained in broker requests, and neither controller-private capsule identities nor public
capsule aliases are copied into the plot manifest. Node labels contain only public frozen action IDs,
candidate hashes, lifecycle state, and aggregate trusted Spike/K1 ratios. As with the outcome figure,
the search never reads or reports a paper-network holdout.

Use `--output-dir` only when a run controller already allocated a unique destination. Existing
directories are rejected to prevent accidental replacement of paper evidence.
