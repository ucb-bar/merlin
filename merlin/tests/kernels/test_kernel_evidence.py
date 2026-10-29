"""Conservative region matching and bounded structural schedule recovery."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from merlin.kernels.kernel_evidence import (
    ArtifactPin,
    EvidenceFact,
    KernelEvidence,
    MeasurementBoundary,
    ScheduleModel,
    SemanticRegion,
    analyze_committed_trace,
    analyze_counted_kernel,
    analyze_disassembly,
    analyze_ir_schedule,
    analyze_native_commands,
    analyze_normalized_commands,
    cca_projection,
    compare_regions,
    semantic_region_from_ir,
)


def _numeric(scale):
    return dict(accumulator_dtype="i32", bias="none", scale=scale, activation="none",
                rounding="nearest_even", saturation="signed_i8")


def _region_record():
    return dict(version=1, region_id="expert-projections", complete=True,
                tensors=[dict(key=key, shape=[7, 13], dtype="i8", layout="logical")
                         for key in ("input", "wq", "wk", "wv", "q", "k", "v")],
                operations=[dict(key=f"projection:{i}", op="projection", inputs=["input", weight],
                                 outputs=[output], numeric=_numeric(scale))
                            for i, (weight, output, scale) in enumerate((('wq', 'q', 0.25),
                                                                         ('wk', 'k', 0.5),
                                                                         ('wv', 'v', 0.75)))],
                inputs=["input", "wq", "wk", "wv"], outputs=["q", "k", "v"],
                external_uses={"q": 1, "k": 1, "v": 1},
                boundary=dict(scope="complete_region", packing="included", preparation="included",
                              conversions="included", synchronization="included", readout="included",
                              input_identity="inputs:a", runtime_identity="runtime:a"))


def _evidence(kind="native_issue", *, complete=True):
    return KernelEvidence("evidence:a", (ArtifactPin("capture", "capture.json", "a" * 64),
                                          ArtifactPin("facts", "facts.json", "b" * 64)),
                          capture_kind=kind, complete=complete)


def _commands():
    commands = []
    for iteration in range(5):
        commands.extend([
            dict(index=iteration * 2, identity="fill", roles=["load"],
                 accesses=[dict(space="private", buffer="operand", offset=iteration * 16,
                                extent=16, mode="write", unit="byte")]),
            dict(index=iteration * 2 + 1, identity="consume", roles=["compute"],
                 accesses=[dict(space="private", buffer="operand", offset=iteration * 16,
                                extent=16, mode="read", unit="byte")]),
        ])
    return commands


def test_evidence_roundtrip_and_fact_authority():
    evidence = replace(_evidence(), facts=(EvidenceFact("operand_reuse", True, "inferred", ("capture.json#4",)),),
                       missing=("historical_raw_trace",))
    assert KernelEvidence.from_record(json.loads(json.dumps(evidence.to_dict()))) == evidence
    assert evidence.identity == KernelEvidence.from_record(evidence.to_dict()).identity
    with pytest.raises(ValueError, match="exact evidence"):
        EvidenceFact("operand_reuse", True, "observed")
    with pytest.raises(ValueError, match="SHA256"):
        ArtifactPin("capture", "missing", "guess")


def test_alpha_renaming_preserves_full_fused_semantics():
    reference = SemanticRegion.from_record(_region_record())
    record = _region_record()
    record["region_id"] = "compiler-projections"
    renaming = {tensor["key"]: f"ssa:{index}" for index, tensor in enumerate(record["tensors"])}
    for tensor in record["tensors"]:
        tensor["key"] = renaming[tensor["key"]]
    for index, operation in enumerate(record["operations"]):
        operation["key"] = f"candidate:{index}"
        for key in ("inputs", "outputs"):
            operation[key] = [renaming[value] for value in operation[key]]
    for key in ("inputs", "outputs"):
        record[key] = [renaming[value] for value in record[key]]
    record["external_uses"] = {renaming[key]: count for key, count in record["external_uses"].items()}
    candidate = SemanticRegion.from_record(record)
    result = compare_regions(reference, candidate)
    assert result.status == result.measurement_status == "equivalent"
    assert result.comparable
    assert SemanticRegion.from_record(json.loads(json.dumps(candidate.to_dict()))) == candidate


@pytest.mark.parametrize(
    "mutation", ["quantizer", "shared_input", "precision", "layout", "observer", "alias", "fusion"],
)
def test_semantic_mismatches_cannot_be_ranked(mutation):
    reference = SemanticRegion.from_record(_region_record())
    record = _region_record()
    if mutation == "quantizer":
        record["operations"][1]["numeric"]["scale"] = 0.25
    elif mutation == "shared_input":
        record["operations"][1]["inputs"][0] = "wq"
    elif mutation == "precision":
        record["tensors"][-1]["dtype"] = "i32"
    elif mutation == "layout":
        record["tensors"][0]["layout"] = "transposed"
    elif mutation == "observer":
        record["external_uses"]["q"] = 2
    elif mutation == "alias":
        record["tensors"][0]["alias_group"] = record["tensors"][1]["alias_group"] = "shared"
    else:
        record["operations"].pop()
        record["tensors"].pop()
        record["outputs"].pop()
        del record["external_uses"]["v"]
        record["inputs"].remove("wv")
        record["tensors"] = [tensor for tensor in record["tensors"] if tensor["key"] != "wv"]
    result = compare_regions(reference, SemanticRegion.from_record(record))
    assert result.status == "mismatch"
    assert not result.comparable
    assert any(difference.status == "mismatch" for difference in result.differences)


def test_unknown_numerics_and_incomplete_fusion_never_match():
    record = _region_record()
    record["operations"][0]["numeric"].pop("scale")
    unknown = SemanticRegion.from_record(record)
    assert compare_regions(unknown, unknown).status == "unknown"
    incomplete = replace(SemanticRegion.from_record(_region_record()), complete=False)
    assert compare_regions(incomplete, incomplete).status == "unknown"


def test_roi_and_input_runtime_match_are_independent_required_performance_gates():
    reference = SemanticRegion.from_record(_region_record())
    record = _region_record()
    record["boundary"]["packing"] = "excluded"
    mismatch = compare_regions(reference, SemanticRegion.from_record(record))
    assert mismatch.status == "equivalent"
    assert mismatch.measurement_status == "mismatch"
    assert not mismatch.comparable
    unknown = replace(reference, boundary=MeasurementBoundary())
    assert compare_regions(reference, unknown).measurement_status == "unknown"
    assert not compare_regions(reference, unknown).comparable


def test_graph_refuses_undefined_inputs_and_redefinitions():
    record = _region_record()
    record["operations"][0]["inputs"] = ["missing"]
    with pytest.raises(ValueError, match="topological SSA"):
        SemanticRegion.from_record(record)
    record = _region_record()
    record["operations"][0]["outputs"] = ["input"]
    with pytest.raises(ValueError, match="topological SSA"):
        SemanticRegion.from_record(record)


def test_counted_observed_recurrence_retains_effects_and_unknown_completion():
    model = analyze_normalized_commands(_commands(), complete=True, evidence_id="capture:a")
    assert model.command_count == 10 and model.complete
    assert len(model.stages) == 2
    assert len(model.loops) == 1 and model.loops[0].extent == 5
    assert model.loops[0].strides == {"0:0": 16, "1:0": 16}
    assert model.role_counts == {"load": 5, "compute": 5}
    assert model.storage[0].offset_min == 0 and model.storage[0].offset_end == 80
    assert model.storage[0].reads == model.storage[0].writes == 5
    assert model.storage[0].completion == "unknown"
    assert any(dependency.kind == "storage_RAW" and dependency.completion == "unknown"
               for dependency in model.dependencies)
    assert "hardware_completion_and_timing_not_established" in model.unknowns
    assert ScheduleModel.from_record(json.loads(json.dumps(model.to_dict()))) == model


def test_ragged_tail_is_preserved_and_search_bounds_are_explicit():
    commands = _commands()
    commands[-1]["accesses"][0]["extent"] = 3
    model = analyze_normalized_commands(commands, complete=True)
    assert model.command_count == 10 and len(model.stages) > 2
    bounded = analyze_normalized_commands(_commands(), complete=True, max_commands=3)
    assert bounded.truncated and not bounded.complete and bounded.command_count == 3
    assert "command_limit_reached" in bounded.unknowns
    exhausted = analyze_normalized_commands(_commands(), max_compaction_steps=1)
    assert "recurrence_search_budget_exhausted" in exhausted.unknowns


def test_explicit_completion_and_configuration_are_scoped_separately():
    commands = [dict(index=0, identity="configure", configuration={"state": "a"}),
                dict(index=1, identity="produce", produces=["ready"]),
                dict(index=2, identity="consume", waits_for=["ready"]),
                dict(index=3, identity="configure", configuration={"state": "b"})]
    model = analyze_normalized_commands(commands)
    event = next(item for item in model.dependencies if item.kind == "event_wait")
    assert event.completion == "declared_completion"
    assert model.configuration_scopes[0] == dict(key="state", value="a", first_issue=0,
                                                last_issue=2, status="observed", completion="unknown")
    assert not any(item.completion == "hardware_completion" for item in model.dependencies)


def test_native_capture_uses_selected_decoder_and_never_claims_timing():
    isa = dict(FUNCT_CLASS={3: "LOAD"}, OPERAND_FIELDS={})
    records = [dict(funct=3, rs1=100, rs2=200), dict(kind="fence")]
    model = analyze_native_commands(records, isa=isa, evidence=_evidence())
    assert [stage.identity for stage in model.stages] == ["LOAD", "FENCE"]
    assert "hardware_completion_and_timing_not_established" in model.unknowns
    assert cca_projection(model, op="projection").dispatch.n_dispatches == 1
    with pytest.raises(ValueError, match="actual nonnegative"):
        analyze_native_commands([dict(funct=3, rs1=None, rs2=1)], isa=isa, evidence=_evidence())


def test_reduction_partition_and_fence_comparison_require_matching_boundaries():
    from merlin.kernels.cca_compare import compare_evidence

    model = analyze_normalized_commands([], complete=False)
    def project(depths, fences):
        return cca_projection(model, op="contraction", facts=(
            EvidenceFact("compute.reduction_partition", depths, "observed", ("capture#output-block",)),
            EvidenceFact("dispatch.n_fences", fences, "observed", ("capture#entry-return",)),
        ))

    expert, ours = project((8, 3), 1), project((4, 4, 3), 5)
    axes = ("compute.reduction_partition", "dispatch.n_fences")
    assert not set(axes) & set(compare_evidence(expert, ours).compared_axes)
    for cca in (expert, ours):
        cca.provenance["axis_evidence"][axes[0]]["scope"] = "one complete contraction output block"
        cca.provenance["axis_evidence"][axes[1]]["scope"] = "one kernel entry through return"
    result = compare_evidence(expert, ours)
    assert set(axes) <= set(result.compared_axes)
    assert {row.axis for row in result.divergences} == set(axes)
    ours.provenance["axis_evidence"][axes[0]]["scope"] = "partial output block"
    assert axes[0] not in compare_evidence(expert, ours).compared_axes


def test_static_decode_uses_exact_selected_funct_table():
    opcode, funct = 35, 11
    word = (funct << 25) | opcode
    evidence = replace(_evidence(), capture_kind="static_disassembly",
                       artifacts=(ArtifactPin("disassembly", "function.dis", "c" * 64),
                                  ArtifactPin("facts", "facts.json", "b" * 64)))
    model = analyze_disassembly(f"0 <f>:\n  0: {word:08x}\t<unknown>\n", evidence=evidence,
                                table=dict(custom_opcode=opcode, names={str(funct): "selected"},
                                           legal_funct=[funct]), roles_of=lambda name: ("compute",))
    assert model.command_count == 1 and model.stages[0].identity == "selected"
    assert model.stages[0].roles == ("compute",)
    assert cca_projection(model, op="projection").dispatch is None


def test_static_unclaimed_custom_instruction_is_retained_and_incomplete(monkeypatch):
    from merlin.compile.analysis import rocc_commands
    from merlin.compile.analysis.disassembly import tokenize_text

    opcode, claimed, unclaimed = 35, 11, 12
    table = dict(custom_opcode=opcode, names={str(claimed): "selected"}, legal_funct=[claimed])
    text = (f"0 <f>:\n  0: {(claimed << 25) | opcode:08x}\t<unknown>\n"
            f"  4: {(unclaimed << 25) | opcode:08x}\t<unknown>\n")
    instructions = tokenize_text(text)
    monkeypatch.setattr(rocc_commands, "funct_table_for", lambda target: table)
    audit = rocc_commands.audit_decoded(rocc_commands.decode_stream(instructions, table), table=table)
    assert rocc_commands.audit(instructions, "selected-target") == audit
    assert audit.endpoint_insns == 1 and len(audit.unaccounted) == 1
    evidence = replace(_evidence(), capture_kind="static_disassembly",
                       artifacts=(ArtifactPin("disassembly", "function.dis", "c" * 64),
                                  ArtifactPin("facts", "facts.json", "b" * 64)))
    model = analyze_disassembly(text, evidence=evidence, table=table)
    assert model.command_count == 2 and len(model.stages) == 2
    assert not model.complete and "unaccounted_instruction_decode" in model.unknowns
    witness = model.stages[1]
    assert witness.parameters["unaccounted"] and witness.parameters["fields"]["funct"] == unclaimed
    assert not witness.roles


def test_cca_projection_preserves_existing_axes_and_per_axis_evidence():
    model = analyze_normalized_commands(_commands(), complete=True)
    facts = (EvidenceFact("memory.onchip_resident", "a", "inferred", ("source:c#storage",)),
             EvidenceFact("compute.register_block", (2, 3), "observed", ("source:c#loops",)),
             EvidenceFact("dispatch.dma_overlap", None, "unknown", reason="issue order is insufficient"))
    cca = cca_projection(model, op="projection", facts=facts)
    assert cca.memory.onchip_resident == "a" and cca.compute.register_block == (2, 3)
    assert cca.dispatch.dma_overlap is None
    assert cca.provenance["axis_evidence"]["memory.onchip_resident"]["status"] == "inferred"
    assert cca.provenance["axis_evidence"]["dispatch.dma_overlap"]["status"] == "unknown"
    with pytest.raises(ValueError, match="not an existing CCA"):
        cca_projection(model, op="projection", facts=(EvidenceFact("invented.optimization", True,
                                                                  "observed", ("source:c",)),))
    with pytest.raises(ValueError, match="contradictory evidence"):
        cca_projection(model, op="projection", facts=(EvidenceFact("dispatch.n_dispatches", 999,
                                                                  "observed", ("source:c",)),))
    with pytest.raises(ValueError, match="incompatible"):
        cca_projection(model, op="projection", facts=(EvidenceFact("compute.accumulator_resident", "yes",
                                                                  "observed", ("source:c",)),))


def test_partial_committed_trace_refuses_register_replay():
    with pytest.raises(ValueError, match="complete original capture"):
        analyze_committed_trace("", isa={}, evidence=_evidence("committed_trace", complete=False))


def test_complete_committed_trace_reuses_register_replay():
    opcode, funct = 35, 7
    custom = (funct << 25) | (2 << 20) | (1 << 15) | opcode
    trace = ("core 0: 3 0x1000 (0x00100093) x1 0x00000010\n"
             "core 0: 3 0x1004 (0x00200113) x2 0x00000020\n"
             f"core 0: 3 0x1008 (0x{custom:08x})\n")
    model = analyze_committed_trace(trace, isa=dict(CUSTOM_OPCODE=opcode, FUNCT_CLASS={funct: "SELECTED"},
                                                   OPERAND_FIELDS={}), evidence=_evidence("committed_trace"))
    assert model.command_count == 1 and model.complete
    assert model.stages[0].identity == "SELECTED"


def test_scheduling_dialect_is_decisions_until_consumed():
    pytest.importorskip("xdsl")
    from xdsl.dialects.builtin import ModuleOp, StringAttr
    from xdsl.ir import Block, Region

    from merlin.xdsl_dialects.schedule import BindOp, HandleType, PlanOp

    binding = BindOp.create(properties={"target": StringAttr("source:a")}, result_types=[HandleType()])
    plan = PlanOp.create(properties={"sym_name": StringAttr("test")}, regions=[Region(Block([binding]))])
    model = analyze_ir_schedule(ModuleOp([plan]))
    assert [decision["operation"] for decision in model.decisions] == ["schedule.plan", "schedule.bind"]
    assert all(decision["realized"] is False for decision in model.decisions)
    assert model.command_count == 0 and not model.stages


def test_counted_payload_is_compact_and_preserves_token_recurrences():
    from merlin.sched.ir import Call, Kernel, Loop, Stage

    kernel = Kernel("payload", (), (Loop("tile", 10000, (
        Call("fill", (), produces="ready", stages=(Stage("scratch", 0, 4, writes=True),)),
        Call("consume", (), awaits=("ready",), stages=(Stage("scratch", 0, 4),)),
    )),))
    model = analyze_counted_kernel(kernel)
    assert model.command_count == 20000
    assert len(model.stages) == 2 and model.loops[0].extent == 10000
    assert model.loops[0].status == "observed"
    assert any(dependency.kind == "event_wait" for dependency in model.dependencies)


def _frontend():
    pytest.importorskip("xdsl")
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import ModuleOp, i32
    from xdsl.ir import Block, Region

    body = Block(arg_types=[i32, i32, i32])
    a, b, c = body.args
    first, second = arith.MuliOp(a, b), arith.MuliOp(a, c)
    consume = arith.AddiOp(first, second)
    body.add_ops([first, second, consume, func.ReturnOp(first, consume)])
    function = func.FuncOp("source", ([i32, i32, i32], [i32, i32]), Region(body))
    return ModuleOp([function]), (first, second), consume


def test_automatic_frontend_extraction_retains_shared_ssa_and_external_users():
    module, projections, consume = _frontend()
    extracted = semantic_region_from_ir(module, operations=projections,
                                        numeric_projector=lambda op: {**_numeric(1), "saturation": "none"})
    assert len(extracted.inputs) == 3 and len(extracted.outputs) == 2
    assert extracted.operations[0].inputs[0] == extracted.operations[1].inputs[0]
    assert extracted.external_uses[extracted.operations[0].outputs[0]] == 2
    assert compare_regions(extracted, extracted).status == "equivalent"
    unknown = semantic_region_from_ir(module, operations=projections)
    assert compare_regions(unknown, unknown).status == "unknown"
    whole = semantic_region_from_ir(module, numeric_projector=lambda op: {"complete": True})
    assert len(whole.operations) == 3 and len(whole.outputs) == 2


def test_frontend_numeric_projector_is_not_allowed_to_guess_untyped_contracts():
    module, projections, _ = _frontend()
    with pytest.raises(ValueError, match="contract mappings"):
        semantic_region_from_ir(module, operations=projections, numeric_projector=lambda op: "inferred")


def test_automatic_frontend_extraction_keeps_opaque_quantization_in_the_actual_ssa_path():
    from merlin.transforms.certificates import parse_ir

    module = parse_ir('''builtin.module { func.func @source(%x: f32) -> i32 {
      %q = "quant_ext.quantize_per_tensor"(%x) {quant_min = -128 : i64, quant_max = 127 : i64}
          : (f32) -> i8
      %result = arith.extsi %q : i8 to i32
      func.return %result : i32
    }}''')
    region = semantic_region_from_ir(module)
    assert len(region.operations) == 2 and len(region.inputs) == len(region.outputs) == 1
    quantize, extend = region.operations
    assert quantize.inputs == region.inputs and extend.inputs == quantize.outputs
    assert extend.outputs == region.outputs
    assert "quant_ext.quantize_per_tensor" in str(quantize.attributes)
    assert not any(operation.numeric_complete for operation in region.operations)
    assert compare_regions(region, region).status == "unknown"


def test_automatic_frontend_extraction_refuses_an_opaque_resultless_effect():
    from merlin.transforms.certificates import parse_ir

    module = parse_ir('''builtin.module { func.func @source(%x: i32) -> i32 {
      %one = arith.constant 1 : i32
      %sum = arith.addi %x, %one : i32
      "opaque.observable_effect"(%sum) : (i32) -> ()
      func.return %sum : i32
    }}''')
    with pytest.raises(ValueError, match="result-producing roots"):
        semantic_region_from_ir(module)


def test_frontend_nested_regions_keep_implicit_captures_and_scalar_semantics():
    pytest.importorskip("xdsl")
    from xdsl.dialects import arith, func, scf
    from xdsl.dialects.builtin import ModuleOp, i1, i32
    from xdsl.ir import Block, Region

    body = Block(arg_types=[i32, i32, i32, i1])
    a, b, c, condition = body.args
    product = arith.MuliOp(a, b)
    add = arith.AddiOp(product, c)
    branch = scf.IfOp(condition, [i32], [add, scf.YieldOp(add)], [scf.YieldOp(product)])
    body.add_ops([product, branch, func.ReturnOp(branch)])
    module = ModuleOp([func.FuncOp("nested", ([i32, i32, i32, i1], [i32]), Region(body))])
    region = semantic_region_from_ir(module, numeric_projector=lambda op: {"complete": True})
    assert len(region.inputs) == 4 and len(region.operations) == 2
    first, second = region.operations
    assert first.outputs[0] in second.inputs
    assert len(second.inputs) == 3  # condition, produced value, implicit capture c
    assert "unresolved_capture" not in json.dumps(second.attributes)
    assert second.attributes["regions"][0][0]["operations"][0]["op"] == "arith.addi"

