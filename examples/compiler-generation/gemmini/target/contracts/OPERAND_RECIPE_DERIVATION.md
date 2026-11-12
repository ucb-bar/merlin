# Operand recipe derivation: CONFIG_EX bundles and field kinds

This note justifies the `ConfigExRs1`/`ConfigExRs2` bundle selections, the
`class_operand_bundles` entry and the `field_kinds` entries in
`operand_selection_recipe.yaml`. Every value is either re-derived or verified by
`python -m merlin.development.targetgen.rocc.operand_selection_recipe` (`main`) against the
FIRRTL pinned in the selected facts:

- FIRRTL: `firechip.chip.FireSim.FireSimGemminiRocketConfig.fir`, sha256 `b7d9826b…5bf0`
- Scala: `generators/gemmini/src/main/scala/gemmini/*.scala`

The FIRRTL line numbers below refer to that file. If the FIRRTL changes, the tool
re-derives everything or refuses; it never falls back to these numbers.
Observed values from binaries were not used for any entry.

## CONFIG_EX operand bundles

`ExecuteController.scala` 542-543:
`config_ex_rs1 = rs1s(0).asTypeOf(new ConfigExRs1(acc_scale_t_bits))` and
`config_ex_rs2 = rs2s(0).asTypeOf(new ConfigExRs2)`. The roster follows the
`ConfigExRs1`/`ConfigExRs2` bundle layouts in the selected facts. Each field's
offset and width is re-derived from the FIRRTL `bits(...)` extraction, not taken
from the Scala source:

| bundle.field | bits | FIRRTL lines |
|---|---|---|
| ConfigExRs1.cmd_type | 1:0 | 321133-321135 |
| ConfigExRs1.dataflow | 2 | 321136-321137 |
| ConfigExRs1.activation | 4:3 | 321138-321139 |
| ConfigExRs1.set_only_strides | 7 | 321142-321143 |
| ConfigExRs1.a_transpose | 8 | 321144-321145 |
| ConfigExRs1.b_transpose | 9 | 321146-321147 |
| ConfigExRs1.a_stride | 31:16 | 321150-321151 |
| ConfigExRs1.acc_scale | 63:32 | 321152-321153 |
| ConfigExRs2.in_shift | 31:0 | 321157-321159 |
| ConfigExRs2.relu6_shift | 47:32 | 321160-321161 |
| ConfigExRs2.c_stride | 63:48 | 321162-321163 |

The `_config_ex_rs{1,2}_WIRE` drivers are conditional. `packing_context.guards`
lists the exact enclosing FIRRTL conditions:

- `_T_102`: `control_state === waiting_for_cmd`, line 532
- `cmd_q.io.deq.valid[0]`: line 540
- `_T_107`: the `DoConfig && ...` condition, line 541

The tool refuses a driver whose guards differ from this list. These guards only
state the packing premise. They do not prove the guards hold or that the command
is accepted.

`class_operand_bundles: CONFIG_EX: {rs1: ConfigExRs1, rs2: ConfigExRs2}` binds the
semantic class to these bundles. The declared roles in `target_contract.yaml`
give CONFIG_EX only a constant subtype. The tool projects every derived field
under its RTL name, so the layout roster reports `complete` with 41 of 41 fields.

## Field kinds

Each `field_kinds` entry is accepted only if the selected FIRRTL contains a
`connect <sink>, <bundle wire>.<field>`. The wire must be exactly the selected
bundle field. The sink must be a `reg`, `regreset` or `output` declared in the
module that owns the wire. The kind is the reviewed reading of that consumer
state element; the decoded value is never used. The sidecar records each
citation under `operand_field_kind_sources`.

| label | kind | consumer (module: sink) | connect line | sink declaration |
|---|---|---|---|---|
| CONFIG_EX.a_stride | stride | ExecuteController: a_addr_stride | 321181 (scala 564) | reg, 319970 |
| CONFIG_EX.c_stride | stride | ExecuteController: c_addr_stride | 321182 (scala 565) | reg, 319971 |
| CONFIG_EX.activation | activation | ExecuteController: activation | 321169 (scala 552) | reg, 318686 |
| CONFIG_LD.scale | scale | LoadController: scales[state_id] | 229345 (scala 148) | reg, 229110 |
| CONFIG_LD.rs1_stride | stride | LoadController: block_strides[state_id] | 229347 (scala 150) | reg, 229112 |
| CONFIG_ST.out_stride_bytes | stride | StoreController: stride | 230368 (scala 234) | regreset, 229587 |
| CONFIG_ST.acc_act | activation | StoreController: activation | 230369 (scala 236) | regreset, 229588 |
| MVIN.cols | extent | LoadController: io.dma.req.bits.cols | 229263 (scala 104) | output, 229106 |
| MVIN2.cols | extent | LoadController: io.dma.req.bits.cols | 229263 | output, 229106 |
| MVIN3.cols | extent | LoadController: io.dma.req.bits.cols | 229263 | output, 229106 |

Why each consumer supports its kind:

- `a_addr_stride` and `c_addr_stride` are the address-stride state of the A and C
  operand streams.
- `scales` is the per-load-state scale register.
- `block_strides` is the per-load-state block stride.
- StoreController `stride` is the DRAM write stride.
- The two `activation` registers select the epilogue activation.
- `io.dma.req.bits.cols` is the DMA request's column count, which is the
  transfer extent.

MVIN, MVIN2 and MVIN3 all read the same `mvin_rs2` bundle in LoadController.

## Deliberately left undeclared (kind stays unknown)

- **CONFIG_EX.acc_scale.** Its consumer reads `rs1s(0)(63,32)` directly
  (scala 555), not the bundle field, so no bundle-field connect exists.
- **CONFIG_EX.in_shift and relu6_shift.** No declared kind applies to a shift.
  relu6_shift also has no consumer.
- **CONFIG_ST.acc_scale.** It has no direct bundle-field connect.
- **CONFIG_LD.stride.** This is the raw rs2 field; it has no bundle field wire.
- **Pooling fields.** They are window parameters, not memory strides or extents.
- **All rows fields.** They drive counters, not a single state element.
- **Readout dtype flags.** They have no bundle-field consumer.

These can be added only with a verifiable consumer.
