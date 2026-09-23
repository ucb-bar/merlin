# AGENT.md — merlin/python/merlin/sched/contract

## Purpose

Numerics contracts: how an integer accumulator becomes the stored output. A contract fixes the scale
granularity (`tensor` / `row` / `column` / `rank1`), the rounding, the clamp bounds and the admitted
activations. Goldens are generated under a contract, and grading compares the contract digest.

## Modules

- `registry.py` — `NumericsContract` (with `readout()` and `digest()`), `from_readout_facts()` and the
  readout-level registry (`per_tensor_readout_v1`, `per_row_readout_v1`, `per_column_readout_v1`).

## Invariants

- **Facts in, never literals.** `from_readout_facts` consumes the dict a target's header-verified
  readout contract returns (schema `scalar_narrow_readout_contract_v1`). The rounding is declared
  `half_even` only because that schema is produced after the target's scale and rounding macros were
  matched against the round-half-to-even construction.
- **The arithmetic is the C macro's, bit for bit**: one f32 multiply, `rint`, clamp, optional relu.
  `merlin/tests/ir/test_sched_contract.py` holds it to `runtime.tensor.Tensor.requant_acc_scale`.
- **Readout-level only.** Bias seeding, residual add and pooling are further stages composed by
  model-level contracts. They are not smuggled in here.
