# AGENT.md — merlin/python/merlin/targetgen/synthesize

## Purpose

Synthesize the five plan artifacts from collected evidence: target_contract, dialect_plan,
runtime_adapter_plan, zephyr_plan, llvm_extension_plan.

## What belongs here

- One module per plan, each returning a schema-shaped `dict`.
- toy_npu concrete branches; conservative keyword-seeded branches for real targets.

## What does not belong here

- File writing (that is `generate/`) or schema definitions (those are `merlin/schemas/`).
- Over-claiming: real-target output must be flagged `confidence` + `requires_human_review`.

## Interfaces

- Inputs: an `Evidence` object and (for plan synthesis) the synthesized target_contract.
- Outputs validate against the matching `merlin/schemas/*.schema.yaml`.

## Invariants

- Targets only ever *implement* the Merlin runtime ABI; never invent a runtime model.
- Do not derive dialect ops directly from instruction names.

## Testing expectations

- `test_targetgen_toy.py` asserts toy_npu plans validate and stay consistent with the
  in-tree `merlin/targets/toy_npu/contracts/`.

## Notes for future agents

- Keep the toy_npu contract consistent with `merlin/targets/toy_npu/contracts/target_contract.yaml`.
