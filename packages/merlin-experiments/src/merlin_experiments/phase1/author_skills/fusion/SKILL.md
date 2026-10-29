---
name: fusion
description: "Open when epilogue work (bias, scale/requantize, activation, residual add, pooling) or a chain of elementwise ops runs on the host or as separate kernels: a host_compute_in_accelerator_group veto, an epilogue_store_path gap, a busy execute resource with fusion in suggested_mechanisms, or many small commands per layer. Covers finding the opportunity, choosing the library route, the numeric contract to keep, verifying it and reading refusals."
---

# Fusion: absorb epilogues and elementwise chains

Every library route keeps the unfused original as a fallback and refuses with a stated reason.
The routes, refusal codes and full contracts are in `reference/routes.md`. Pipeline placement is
covered by the `compose-pass-pipeline` skill. A convolution whose requantized output feeds a pool
without a full intermediate image: `spatial_chain_recipe.py` in the `heavy-pass-recipes` skill.

## 1. Find the opportunity
- Grade feedback (see the `qa-feedback-files` skill):
  - `host_compute_in_accelerator_group`: the host computes every element of a program that is
    placed on the unit.
  - An `epilogue_store_path` `gap`: the target licenses a stage the command did not request.
  - Fusion in `suggested_mechanisms`: the execute resource is the busiest.
- In your own IR, page through every source opportunity. The rows are proposals, not legality.

```python
from merlin.compile import fusion_workflow as F

def fusion_opportunities(module, capability_contract, numerical_policy_sha256):
    rows, cursor = [], None
    while True:
        page = F.discover(module, capability_contract=capability_contract,
                          numerical_policy_sha256=numerical_policy_sha256, cursor=cursor)
        rows += page["candidates"]
        cursor = page["next_cursor"]
        if cursor is None:
            return rows
```

The `route` and `entrypoint` fields of a row name the realizer for that opportunity.

## 2. State what the hardware applies, from your target's facts
- Build an `output_path_v1` mapping from your workspace's capability facts:
  - From a derived output-epilogue record, use `merlin.targetgen.output_path.from_record`.
  - For a mapping you build yourself, check it with `merlin.targetgen.output_path.validate`.
  - Omit any field you cannot establish. The pass then refuses whatever needs that field, which is
    correct.
- To pick the readout for a set of stages, use
  `merlin.verify.epilogue_applicability.selectors_applying`. An empty result means fail closed;
  never choose by dtype width.

## 3. Decide the cut per contraction
Cuts are tried longest first: pool, requant, accumulator, bare.

```python
from merlin.targetgen import output_path as OP
from merlin.xdsl_dialects.lowering import output_fusion as OF

def epilogue_cut(root, declared_path, extents, values=None):
    values = values or OF.Values()
    decision = OF.decide(OF.source_chain(root, values), OP.validate(declared_path), values, extents)
    return decision.cut, decision.stages, [(r["cut"], r["code"]) for r in decision.refusals]
```

`extents` maps `id(root)` to `(m, k, n, rank)`. If every cut says "reduction extent is not
static", the extent reader is missing from your tree: pass your own extents. For a whole graph,
use the two-pass `fused_groups` in the reference.

## 4. Keep the numeric contract
- Keep the source's order of scales and activation, its half-even rounding, its clamp range and
  its zero point. `decide` proves equality over the whole accumulator range.
- Fold a scale chain only through `merlin.llvmlower.requantization.prove_scale`, which raises with
  a witness accumulator value. Derive integer bias preloads with
  `merlin.llvmlower.requantization.synthesize_bias`; a channel it returns as `None` stays on the
  host.
- `merlin.compile.scheduling.fusion.plan_fusion` admits only total integer scalar IR on adjacent
  synchronous calls with equal element counts and a private, unobserved intermediate.

## 5. Realize, then verify
1. Call the route's realizer and keep its `unfused` alternative.
   `merlin.compile.scheduling.schedule_pipeline.compile_schedule(fusion_capabilities)` runs fusion
   before storage and falls back to the unfused schedule if allocation fails.
2. Check numerics: run `agent_selfcheck.py` on the affected cases. In your own checks, use
   stimulus that drives accumulators negative and past the clamp, because a discarded relu
   still passes on non-negative data.
3. Check stages: `merlin.verify.epilogue_applicability.assess(command_buffer, readouts)` must
   return `applied`. Treat `discarded` and `unknown` as refusals.
4. Check cycles with the `ab-library-pass` skill against the unfused build. Confirm that
   `routed_macs` held and that the host regions shrank.
