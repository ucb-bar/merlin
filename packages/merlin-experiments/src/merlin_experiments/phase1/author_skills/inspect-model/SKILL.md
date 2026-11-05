---
name: inspect-model
description: Inspect a public model or whole-program test case before compiling it. It covers the case's declaration (inputs, shapes, layouts, dtypes, quantization), its connected slices, the operator cases that verify each device group, which operations dominate the work (shape times count), which operations are placed on the accelerator and which on the host, and what the grader runs for it. Open it when you start on a model case, when a model check fails and you need the operator case behind a device group, when you want the dominant work to prioritise, or when an offload census or host-compute veto names host work. Includes a script that summarises one case and ranks its operator cases by multiply-accumulate count. Public files only.
---

# Inspect a model case

`SKILL` below is this skill's directory. Run commands from the workspace root. File formats and
fields are in `reference/model-case-format.md`.

## 1. Find the public cases

- **`model/CASE/`** holds whole-program cases: `capsule.yaml`, `capsule.interface.mlir` (linalg-on-tensors), `capsule.pytorch.py`, `frontend-trace.json`, `pytorch-opset.json`, and sometimes `profile.json`.
- **`model_slices/CASE/`** holds connected model slices.
- **`layers/`, `isa/`, `projections/`** hold operator cases. A case derived from a model group names the model in `source_reference`.
- **`public_sources/`** holds linalg sources and `suite.json`. **`public_declarations/software-spec.yaml`** holds the admitted numerics, operations and quantization methodology.
- **The graded whole-model programs** are the keys `MODEL/PROGRAM` of `qa/verdict.json` `model_gate.programs`.

Everything staged is public. Goldens, expected command buffers, expected coverage and weights are
filtered out before staging, and hidden cases are never staged. Do not look for them elsewhere or
reconstruct them from digests. The summary lists named-but-withheld files under `withheld`.

## 2. Summarise one case

```
python SKILL/scripts/model_summary.py model/CASE
python SKILL/scripts/model_summary.py model/CASE --json > $TMPDIR/CASE.summary.json
python SKILL/scripts/model_summary.py model_slices/CASE        # any case directory works
```
It prints four things:
- **Declaration:** inputs with shape, dtype and role; the attributes; the numeric policy and tiers.
- **Quantization:** scheme, qparam count, granularities, accumulator dtypes, and remaining dequantize ops.
- **Placement:** the reviewed `software_screen` per operation: accelerator, host or `unresolved`, with counts.
- **Operator cases:** `model_qualification.group_capsules`, ranked by `count x output elements x reduction extent`, with each row's share.

The top rows are the work that matters most. A large `count` with a small `share` is per-call
overhead; batch or fuse it.

## 3. Read the program

- `capsule.interface.mlir` is what your compiler receives. `merlin.targetgen.contract.linalg_iface.parse_linalg_mlir` (see `inspect-ir`) gives each op's `kind`, `extents`, `reduction_dims` and `prov`.
- `frontend-trace.json` `graphs.*.by_target` counts framework ops at each stage.
- Layouts and `stride`/`padding`/`dilation` are in each operator case's `operation.attributes`.

## 4. See what you offload

```
python agent_selfcheck.py --offload-census --wait-budget 5     # then --attach RID
```
The census uses no simulator and no golden. It reads only the command buffers your package emits
for every public program, and reports for each program:
- `outcome` (offloaded, host_only, declined, silent), with `routed_macs`, `contraction_offload_fraction` and `host_regions`;
- `host_compute` rows (`verdict`, `why`, `symbol`, `dominant_block`): arithmetic found in host code;
- `host_compute_vetoed`: host compute inside an accelerator group. The grader refuses it, and it shows as failure plane `host_compute`;
- `instruction_use.unused`: declared instructions you never emit. These are opportunities, not defects.

Self-check responses can carry the same block as `offload_census`. `qa/verdict.json`
`model_gate.programs[...]` adds `census` and `accelerator_coverage` (`host_reason_counts`,
`eligible_but_host`, `host_time_cycles`).

## 5. Run what the grader runs

```
python agent_model_check.py check --model MODEL [--program PROGRAM]
python agent_model_check.py replay-kernel --model MODEL --program PROGRAM --group REGION_ID --engine spike
python agent_model_check.py wait --job-id ID [--full-json]
```
`check` builds and censuses `submission/`, replays its kernels, and compares the model's numerics.
`replay-kernel` reruns one device group by its `source_region_id`. The compact output has `gates`,
`programs[]` and `first_distinct_failure`; the full report is at `full_artifact.path`. Exit codes:
0 pass, 1 not pass, 2 pending.

Fix the operator case behind a failing group first. It is far cheaper: self-check it with
`--capsules NAME`, where NAME is the last component of the summary's `capsule` column.
