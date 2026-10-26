# Model case files and fields

All of these are public files staged in the workspace. Field names were read from the staged cases
and from the code that writes the reports (`merlin.targetgen.offload_census`,
`merlin_experiments.phase1.feedback.visible_model_gate`, `merlin_experiments.phase1.brokers.model_check`).

## `capsule.yaml` of a model case (`kind: model`)

- **Top level.** `name`, `kind`, `label: public`, `interface_mlir`, and `inputs[]` with `name`,
  `role`, `shape` and `dtype`. A model input is usually a float tensor that the program quantizes
  itself.
- **`operation.attributes`.**
  - `model`, `dtype`, `compile_dtype`, `out` and `arg_order`, which is the entry-argument order.
  - `quant_scheme` and `quant_recipe_agreement`, which holds the recipe-vs-float agreement
    statistics.
  - `integerization_receipt`:
    - `schema`, `qparams[]`, `precision_decisions[]`, `remaining_dequant_count`, plus per-kind
      counts such as `conv2d_seen` and `conv2d_integerized`.
    - Each `qparams` row has `activation` and `weight`, each with `granularity`, `scale`,
      `zero_point`, `quant_min` and `quant_max`. The row also has `reduction_k` and
      `accumulator_dtype`.
  - `weights` and `weights_manifest` name files that are withheld and not staged.
- **`numeric_policy`.** `compare` is `tolerance_float` with `atol` and `rtol` for a model, or
  `exact_int` for integer operator cases.
- **`required_oracle_tiers`, `gate`.** `gate.after_op_pass_fraction` is the operator pass fraction
  a model needs before it is executed.
- **`semantic`.** `semantic_family` and `must_accelerate`.
- **`model_qualification`.**
  - `status`, `device_groups`, `covered_groups` and `scope`.
  - `group_capsules[]`: `groups`, the device-group indices; `count`; `capsule`, a workspace-relative
    operator case; `arithmetic_qualification`; and `partial_sum_bound`. Each row also carries a
    digest of withheld data, which the summary script drops.
- **`software_screen`.**
  - `status`, `reason`, `placements`, which totals the `accelerator` and `host` placements, and
    `decisions[]`.
  - Each decision has `operation`, `mlir_operation`, `frontend_op`, `count`, `classification`,
    `placement` (`accelerator` or `host`), `status` and `reason`. `classification` is one of
    `accelerator_candidate`, `host_required` or `unresolved`.
- **`frontend_trace`, `materialized_capture`.** These carry provenance, including the capture's
  operation count.

## Other files in a model case

- **`frontend-trace.json`.**
  - `graphs.original|prepared|quantized.by_target` maps each framework op to a count.
  - `mlir.operations[]` lists the program's ops with `operation`, `operand_types`, `result_types`
    and `origin_node_ids`.
- **`profile.json`.** The workload's architecture parameters.
- **`capsule.pytorch.py`.** The model definition the program was traced from.
- **`pytorch-opset.json`.** The framework ops the model uses.

## Operator cases (`layers/`, `isa/`, `model_slices/`, `projections/`)

- **`capsule.yaml`.**
  - `operation.op` is the operation. `operation.attributes` holds `lhs` / `ifm`, `weight`, `out`,
    `epilogue` and `output_dtype`. A convolution adds `ci`, `kh`, `kw`, `stride`, `padding`,
    `dilation` and `layout`.
  - `source_reference` says which model group the case was isolated from.
  - `integer_partial_sum_bound.reduction_extent` is the K of a contraction.
  - `expected.instruction_classes` and `modes` are public coverage requirements.
- **`capsule.interface.mlir`.** The `merlin_iface` program. The op whose `name` is the output
  carries the result type, which gives the output shape.
- **The work estimate** in `model_summary.py` is output elements times reduction extent, scaled by
  the group `count`. It is an estimate for ranking, not a cycle count.

## Placement reports

- **`--offload-census`** produces schema `offload_census_v1`.
  - Each `programs[]` row has `program`, `commands`, `routed_macs`, `routed_is_lower_bound`,
    `unit_regions`, `host_regions`, `contraction_offload_fraction`, `placement_declared`,
    `declined` and `outcome`.
  - Top-level fields: `outcomes`, `routed_macs`, `silent`, `placement_undeclared` and
    `all_accounted`.
  - `instruction_use` holds `declared_count`, `unused[]` and `undeclared_emitted[]`.
  - Each `host_compute[]` row has `verdict`, `why`, `symbol`, `dominant_block` and
    `input_preparation`. `host_compute_vetoed[]` lists the vetoed ones.
  - The census exits 0 only when everything is accounted for and nothing is vetoed.
- **Host compute.** The verdict `host_compute_in_accelerator_group` means a host loop computes the
  tensor elements of a program that claims to run on the unit. The program passes numerics but
  spends its time on the host. Issuing commands and computing addresses is fine.
- **`qa/verdict.json` `model_gate`.**
  - Top-level fields: `source_sha256`, `split`, `mode`, `complete` and `passed`.
  - Each `programs["MODEL/PROGRAM"]` entry has `status`, `build`, `census`, `metrics`, `cycles`,
    `first_bad_group` and `accelerator_coverage`.
  - `accelerator_coverage` has `scope`, `host_reason_counts`, `eligible_but_host`,
    `boundary_bytes`, `host_time_cycles` and `remaining_evidence`.

## Model check (`agent_model_check.py`)

- **Actions.** `check --model M [--program P]`,
  `replay-kernel --model M --program P --group REGION_ID --engine spike` and `wait --job-id ID`.
- **Flags.** Common flags are `--out FILE` (a path relative to the workspace, holding the full
  JSON), `--full-json` and `--wait-budget 0..540`. The only submission is `submission/`.
- **Compact output** (`merlin.model_check.compact_feedback.v1`). It holds `status`, `gates`,
  `programs[]` with `build`, `census` and replay counts, `failure_count`, `first_distinct_failure`
  and `full_artifact` (path and sha256).
- **Full report** (`merlin.model_check.report.v1`).
  - `gates` holds `build`, `kernel_exact` and `model_numerics`.
  - Each `programs[]` entry has `kernels[]`, with `symbol`, `source_region_id(s)`, `form`, `cases`,
    `exact`, `mismatches`, `cycles` and `status`.
  - It also has `model_metrics` and `failures`.
  - Sorting `kernels[]` by `cycles.max` gives the hot kernels of a whole model.
- **Channel.** Requests go through `.model_check_channel/` (`req_ID.json` / `resp_ID.json`). Exit
  0 is pass, 1 is not pass and 2 is pending, in which case run `wait --job-id ID`.
