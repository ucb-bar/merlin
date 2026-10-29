# Fusion routes, contracts and refusals

Every symbol below is resolved against the library by this skill's test, so a renamed function
fails the test rather than misleading you. Line numbers are left out because your tree may differ.
Open the function itself for the exact code.

## Routes

| What fuses | `discover` route | Realizer | Works on |
|---|---|---|---|
| Contraction + epilogue chain (convert, scale, bias, residual, relu, pool, quantize) | `epilogue` | `merlin.xdsl_dialects.lowering.output_fusion.decide`, `merlin.xdsl_dialects.lowering.output_fusion.decisions`, consumed by `merlin.xdsl_dialects.lowering.compute_groups.form_groups(fusion)` | linalg on tensors |
| Two adjacent elementwise calls | `adjacent_elementwise` | `merlin.compile.fusion_workflow.realize_adjacent`, which calls `merlin.compile.scheduling.fusion.plan_fusion`; or `merlin.compile.scheduling.schedule_pipeline.compile_schedule(fusion_capabilities)` | schedule `Kernel` of `Call`s |
| One pointwise producer feeding two pointwise consumers (two live outputs) | `multioutput` | `merlin.compile.fusion_workflow.realize_multioutput`, which calls `merlin.compile.scheduling.multioutput_fusion.compile_source_multioutput` | linalg, then guarded images |
| Sibling contractions sharing one input | `shared_input` | `merlin.compile.fusion_workflow.realize_shared_input`, which calls `merlin.compile.scheduling.shared_input_schedule.schedule_shared_input` | logical schedule |
| Connected pointwise/mixed region | `connected_residency` | `merlin.compile.connected_regions.plan_connected_region`, then `merlin.compile.connected_regions.compile_connected_pointwise` | sequential stages, not cross-stage fusion |
| Reuse of immutable prepared operands | `preparation_reuse` | `merlin.compile.compiler_observations.install_prepared_observed` | host preparation |
| Row normalization islands | (none) | `merlin.transforms.row_normalization.fuse_row_normalizations(module, numerical_policy)` | host tensor IR |
| Softmax rows | (none) | `merlin.compile.scheduling.row_softmax.compile_source_rows` | schedule |
| Pointwise source through the scheduler | (none) | `merlin.compile.scheduling.pointwise_source.compile_source_pointwise(fusion_emitters)` | linalg to schedule |

Supporting calls:
- `merlin.compile.fusion_workflow.realize_candidate(module, candidate, capability_contract, numerical_policy_sha256)`
  revalidates one row from `discover` against the current source, then dispatches it to its realizer.
  It returns a `Realization`: `artifact` is the product, `unfused` is the original, and `receipt` is
  the status with its reason.
- `merlin.compile.fusion_workflow.evaluate_fusions(source, alternatives, emit)` runs the unfused
  build plus up to 15 `FusionAlternative`s. Each runs on its own clone through
  `merlin.transforms.pipeline_policy.evaluate_variants`, and `emit` must be your ordinary emission.

## Whole-graph epilogue grouping (two passes over the same module)

```python
from merlin.targetgen import output_path as OP
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import output_fusion as OF

def fused_groups(module, target, declared_path, oracle=None, values=None):
    values = values or OF.Values()
    first = CG.form_groups(module, target, oracle=oracle)
    decided = OF.decisions(first, module, OP.validate(declared_path), values)
    return CG.form_groups(module, target, oracle=oracle, fusion=decided)
```

`decisions` keys each decision by `id(root)`, so do not mutate or re-parse the module between
the two calls. By default `form_groups` asks `merlin.xdsl_dialects.lowering.group_target_admission.TargetOracle`,
which reads your selected target contract. If your workspace states those facts differently,
pass an `oracle` that answers the same questions (`unit_for`, `absorbs`, `ask`, `readout`). Each
group's `fusion` record holds the cut, the stages, every refusal and the proof. A refusal does
not fall back to a bare device contraction with the quantize left out: if the device form for
the chosen cut is unavailable, the whole group stays on the host.

## The output-path proof (`decide`)

- The chain is the run of single-use consumers after the contraction, up to a quantize. A chain
  that does not end in a quantize after a convert is `source_unread`.
- `requant` cut: needs the `requant` exit. A source relu needs an `activation` readout stage. The
  zero point must be 0, and the narrowed dtype must equal the source quantize dtype. The device
  scale must be able to vary the way the source scale varies (`granularity`, `columns_per_scale`).
- `accumulator` cut: the bias becomes an integer `preload` and the residual an `accumulate`; the
  host applies the rest of the chain as written. If the residual cannot be absorbed, it stays on
  the host and the bias can still be preloaded.
- `pool` cut: the path pools a window max. A padded source pool needs the path's padding value.
- The accumulator bound is K times max|w| times max|x|, read from stored weights when constant,
  otherwise from the operand formats. For every channel and output level, the source and device
  thresholds must coincide over that whole range.

| Refusal code | Meaning | What to do |
|---|---|---|
| `stage_undeclared` | the path performs no such stage | keep it on the host; the hardware cannot do it |
| `exit_undeclared` | no exit at that cut | try a shorter cut |
| `format_unmodeled` | a format or rounding has no exact model here | extend the model in your tree, with a proof |
| `scale_granularity` | the device scale cannot vary as the source does | partition columns or keep the scale on the host |
| `inexact` | no integer offset aligns every output level | not fusable as written |
| `source_unread` | the chain, axis or reduction extent could not be read | supply extents or canonicalize the source |
| `pool_padding_unknown` | padded pool, device padding not derived | derive the padding fact first |

## Schedule-level fusion (`plan_fusion`)

A pair `(producer, consumer)` fuses only if every condition below holds; otherwise it raises
`merlin.compile.scheduling.fusion_semantics.UnsupportedElementwise` and the pair is kept.
- Both calls are synchronous, qualified elementwise effects (not exact copies) and are adjacent
  in a finite `Call` body. Counted loops are kept unchanged, never unrolled.
- The element counts are equal: no broadcast.
- The intermediate is newly initialized private storage. Its lifetime ends at the consumer and
  it has no other view, writer or observer. The final output has a distinct backing.
- The consumer reads the complete typed producer result with the same byte order. In-place
  aliases must be exact: no shifted or streaming alias.
- One `FusionCapability` readout applies all the ordered stages of both calls. Its emitted call
  must decode to the same scalar expression: no change of operation order, conversion, rounding
  or saturation.
- The scalar IR is a single function over signless integers of 1 to 256 bits, with byte-sized
  memory elements. Shifts must be by a constant amount within range. Floats are refused.

`merlin.compile.scheduling.fusion.FusionCapability(readout, emit, configuration_identity, numerical_policy_identity, qualification_identity)`
needs identities that match the `StorageAdapter`. Starting from declarations:
`merlin.compile.scheduling.fusion_contracts.resolve_fusion_contract(declaration, facts, isa, storage)`
then `merlin.compile.scheduling.fusion_contracts.SelectedFusionContract.bind(adapter, emitters, qualification_identity, max_operations, max_ir_bytes)`
return the capabilities together with the reasons for every one left out
(`missing_execution_qualification`, `missing_physical_emitter`, `unknown_RTL_bound`).
`merlin.compile.scheduling.schedule_pipeline.ScheduleBudgets(max_fusions)` set to 0 turns fusion off, which gives you the A/B baseline. The report's
`storage.fusion.status` is either `realized` or `allocation_fallback`.

## Grade signals

- `host_compute_in_accelerator_group`: a host loop loads, computes and stores every element of a
  program placed on the unit. Fuse that epilogue, or declare it as a host placement.
- `epilogue_store_path` (`ABSORBABLE_WORK_LEFT_ON_HOST`): only a `gap` is a decided failure. An
  `incomplete` row blocks nothing, for example when activation scales are computed at run time.
- The offload census reports `routed_macs`, `host_regions`, `contraction_offload_fraction` and
  `placement_undeclared`. A pass count can rise while `routed_macs` falls.
- `suggested_mechanisms`: when the execute resource is busy, or the compute floor binds, the
  list includes fusion.
