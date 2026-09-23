"""Route QUALITY: a stated refusal the target contradicts, and host compute inside an A region.

The pair that matters is the second one. A veto that fires on legal addressing is unsatisfiable,
and an unsatisfiable rule is one everybody routes around -- so ``test_addressing_arithmetic_clears``
is what licenses ``test_tensor_arithmetic_in_accepted_region_fires`` to exist at all.
"""

import pytest
from xdsl.dialects.llvm import LLVM

from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
from merlin.perf import gate_phase
from merlin.runtime import route_quality as rq
from merlin.targetgen import readout_facet as rf

# ------------------------------------------------------------------------------------------------
# Fixtures: a minimal command buffer and a minimal derived facet, both stated here so a test says
# what capability it means to check rather than depending on a checkout's extraction toolchain.
# ------------------------------------------------------------------------------------------------


def _facet(*, readouts=(), operand_sum=None, operand_sum_absent=None, accumulator_kind=None):
    facet = rf.ReadoutFacet(target="synthetic", unit="unit")
    facet.readouts = tuple(readouts)
    facet.operand_sum = operand_sum
    facet.operand_sum_absent = operand_sum_absent
    facet.accumulator_kind = accumulator_kind
    if accumulator_kind is None:
        facet.unknown["accumulator_kind"] = "no rung produced it"
    return facet


def _placement_buffer(rows):
    mesh = [r["region"] for r in rows if r["lane"] == "on_unit"]
    host = [r["region"] for r in rows if r["lane"] != "on_unit"]
    return {
        "params": {"lane_placement": rows, "mesh_regions": mesh, "host_lane_regions": host},
        "commands": [{"opcode": "CONTRACT", "operands": {}}],
    }


def _evidence():
    rows, cause = rq.host_refusal_evidence()
    assert cause is None, cause
    return rows


# ------------------------------------------------------------------------------------------------
# Tier 1 -- the declaration check
# ------------------------------------------------------------------------------------------------


def test_contract_declares_an_evidence_row_for_every_adjudicated_family():
    """The map is contract DATA. A checker carrying its own table is the overfitting this forbids."""
    rows = _evidence()
    assert set(rows) >= {"quantize", "minmax", "pool", "elementwise"}
    for family, spec in rows.items():
        assert ("epilogue_stage" in spec) != ("facet_field" in spec), family


def test_refusal_contradicted_by_the_declared_readout_is_reported():
    buffer = _placement_buffer(
        [
            {
                "region": "q0",
                "family": "quantize",
                "op": "quantize_per_tensor",
                "lane": "host_lane",
                "reason": "family 'quantize' has no datapath on this unit",
            },
        ]
    )
    facet = _facet(
        readouts=[
            {
                "selector": "i8",
                "applies": ["acc_scale", "relu", "maxpool"],
                "evidence": "the narrowing readout applies the accumulator scale",
            }
        ]
    )
    report = rq.declaration_quality(buffer, facets=[facet], evidence=_evidence())
    assert report.status == rq.STATUS_REPORTED
    assert len(report.contradicted) == 1
    verdict = report.contradicted[0]
    assert verdict.region == "q0" and verdict.rung == "backend_declared"
    assert "acc_scale" in verdict.why


def test_refusal_the_capability_confirms_is_ok():
    buffer = _placement_buffer(
        [
            {
                "region": "q0",
                "family": "quantize",
                "op": "quantize_per_tensor",
                "lane": "host_lane",
                "reason": "family 'quantize' has no datapath on this unit",
            },
        ]
    )
    # A complete readout census that applies nothing: the target really does refuse this stage.
    facet = _facet(
        readouts=[
            {"selector": "i32", "applies": [], "evidence": "raw accumulator drain"},
            {"selector": "i8", "applies": ["relu"], "evidence": "activation only"},
        ]
    )
    report = rq.declaration_quality(buffer, facets=[facet], evidence=_evidence())
    assert report.status == rq.STATUS_OK, report.causes
    assert len(report.confirmed) == 1
    assert "none applies the 'acc_scale' stage" in report.confirmed[0].why


def test_family_the_capability_cannot_decide_is_unknown_never_a_pass():
    """Residual add on a target whose accumulator kind is underived. UNKNOWN, with the reason."""
    buffer = _placement_buffer(
        [
            {
                "region": "add_0",
                "family": "elementwise",
                "op": "add",
                "lane": "host_lane",
                "reason": "family 'elementwise' has no datapath on this unit",
            },
        ]
    )
    facet = _facet(
        readouts=[{"selector": "i8", "applies": ["acc_scale"], "evidence": "scale"}],
        operand_sum_absent="the load multiplies, but the accumulator is not derived",
        accumulator_kind=None,
    )
    report = rq.declaration_quality(buffer, facets=[facet], evidence=_evidence())
    assert report.status == rq.STATUS_INCOMPLETE
    assert not report.contradicted and not report.confirmed
    assert len(report.unknown) == 1
    why = report.unknown[0].why
    assert "accumulator_kind" in why and "no rung produced it" in why


def test_residual_add_is_decided_once_the_accumulator_rung_speaks():
    """The broader derived source DOES cover residual add -- when its dependency is derived."""
    buffer = _placement_buffer(
        [
            {
                "region": "add_0",
                "family": "elementwise",
                "op": "add",
                "lane": "host_lane",
                "reason": "family 'elementwise' has no datapath on this unit",
            },
        ]
    )
    licensed = _facet(
        operand_sum={"operands": 2, "operand_dtype": "i8", "scale_dtype": "f32"},
        accumulator_kind="addressable",
    )
    assert rq.declaration_quality(buffer, facets=[licensed], evidence=_evidence()).contradicted

    refused = _facet(
        operand_sum_absent="the accumulator is in_datapath, and separately loaded operands can only "
        "be summed in an addressable one",
        accumulator_kind="in_datapath",
    )
    confirmed = rq.declaration_quality(buffer, facets=[refused], evidence=_evidence())
    assert confirmed.status == rq.STATUS_OK and len(confirmed.confirmed) == 1


def test_a_host_route_without_a_reason_is_reported():
    buffer = _placement_buffer(
        [{"region": "q0", "family": "quantize", "op": "quantize_per_tensor", "lane": "host_lane"}]
    )
    report = rq.declaration_quality(buffer, facets=[_facet()], evidence=_evidence())
    assert report.status == rq.STATUS_REPORTED
    assert rq.HOST_ROUTE_WITHOUT_REASON in report.contradicted[0].why


def test_a_family_the_contract_does_not_name_is_unknown():
    buffer = _placement_buffer(
        [{"region": "x0", "family": "attention", "op": "sdpa", "lane": "host_lane", "reason": "no datapath"}]
    )
    report = rq.declaration_quality(buffer, facets=[_facet()], evidence=_evidence())
    assert report.status == rq.STATUS_INCOMPLETE
    assert "names no evidence source for family 'attention'" in report.unknown[0].why


def test_an_accelerator_routed_region_is_not_adjudicated_by_tier_one():
    buffer = _placement_buffer(
        [
            {
                "region": "conv_1",
                "family": "contraction",
                "op": "conv2d",
                "lane": "on_unit",
                "reason": "contraction at the mesh operand dtype",
            }
        ]
    )
    report = rq.declaration_quality(buffer, facets=[_facet()], evidence=_evidence())
    assert report.verdicts == ()
    assert report.status == rq.STATUS_OK


def test_absent_placement_is_incomplete_never_ok():
    report = rq.declaration_quality({"params": {}}, facets=[_facet()], evidence=_evidence())
    assert report.status == rq.STATUS_INCOMPLETE
    assert any("lane_placement" in c for c in report.causes)


# ------------------------------------------------------------------------------------------------
# Tier 2 -- host compute inside an accepted region
# ------------------------------------------------------------------------------------------------

_ARGS = "%t: !llvm.ptr, %n: i64"


def _host_ir(body):
    context = make_context()
    context.load_dialect(LLVM)
    module = parse_mlir_text("builtin.module { llvm.func @kernel(" + _ARGS + ") {" + body + "} }", context)
    return next(op for op in module.body.block.ops if op.name == "llvm.func")


def _buffer(task_kinds):
    return {
        "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "arg0", "access": "read"}]},
        "params": {
            "global_program_plan": {
                "schema": "mixed_program_plan_v1",
                "tasks": [{"task_index": i, "kind": k} for i, k in enumerate(task_kinds)],
            }
        },
        "commands": [{"opcode": "CONTRACT", "operands": {}}],
    }


#: One arithmetic operation on a value LOADED FROM A TENSOR BUFFER. The defect.
_ON_TENSOR = """
    %v = llvm.load %t {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
    %one = llvm.mlir.constant(1 : i64) : i64
    %s = llvm.add %v, %one {merlin.global_task = 0 : i64} : i64
    llvm.store %s, %t {merlin.global_task = 0 : i64} : i64, !llvm.ptr
    llvm.return
"""

#: The SAME chain, rewritten so the arithmetic derives only from an induction variable. Legal
#: addressing. If this does not clear, the discriminator is wrong.
_ON_ADDRESSING = """
    %one = llvm.mlir.constant(1 : i64) : i64
    %s = llvm.add %n, %one {merlin.global_task = 0 : i64} : i64
    %p = llvm.getelementptr %t[%s] {merlin.global_task = 0 : i64} : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %v = llvm.load %p {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
    llvm.store %v, %t {merlin.global_task = 0 : i64} : i64, !llvm.ptr
    llvm.return
"""


def test_tensor_arithmetic_in_accepted_region_fires():
    report = rq.host_compute(_buffer(["convolution"]), function=_host_ir(_ON_TENSOR))
    assert report.status == rq.STATUS_REPORTED, report.causes
    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.route == "A" and finding.on_tensor == 1 and finding.witnesses


def test_addressing_arithmetic_clears():
    """A veto that fires on legal addressing is unsatisfiable. This test is the licence for the one above."""
    report = rq.host_compute(_buffer(["convolution"]), function=_host_ir(_ON_ADDRESSING))
    assert report.status == rq.STATUS_OK, (report.causes, [t.to_dict() for t in report.tasks])
    assert report.findings == ()
    assert sum(t.on_addressing for t in report.tasks) == 1


def test_identical_arithmetic_in_a_host_region_is_coverage_not_a_defect():
    report = rq.host_compute(_buffer(["host"]), function=_host_ir(_ON_TENSOR))
    assert report.status == rq.STATUS_OK, report.causes
    assert report.findings == ()
    assert report.coverage == 1


def test_absent_host_ir_is_incomplete_never_ok():
    report = rq.host_compute(_buffer(["convolution"]), function=None)
    assert report.status == rq.STATUS_INCOMPLETE
    assert any("no parsed host IR" in c for c in report.causes)
    assert report.status != rq.STATUS_OK


def test_absent_program_plan_is_incomplete():
    report = rq.host_compute({"params": {}}, function=_host_ir(_ON_TENSOR))
    assert report.status == rq.STATUS_INCOMPLETE
    assert any("global_program_plan" in c for c in report.causes)


def test_tensor_data_round_tripping_the_stack_is_still_tensor_data():
    body = """
    %one = llvm.mlir.constant(1 : i64) : i64
    %slot = llvm.alloca %one x i64 : (i64) -> !llvm.ptr
    %v = llvm.load %t {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
    llvm.store %v, %slot {merlin.global_task = 0 : i64} : i64, !llvm.ptr
    %w = llvm.load %slot {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
    %s = llvm.add %w, %one {merlin.global_task = 0 : i64} : i64
    llvm.store %s, %t {merlin.global_task = 0 : i64} : i64, !llvm.ptr
    llvm.return
    """
    report = rq.host_compute(_buffer(["convolution"]), function=_host_ir(body))
    assert report.status == rq.STATUS_REPORTED
    assert report.findings[0].on_tensor == 1


def test_a_module_level_buffer_is_a_tensor_buffer():
    """Staging buffers are module globals, not kernel arguments; a walk that stops at `arg:` misses them."""
    context = make_context()
    context.load_dialect(LLVM)
    module = parse_mlir_text(
        """builtin.module {
      llvm.mlir.global internal @stage() {addr_space = 0 : i32} : !llvm.array<64 x i8>
      llvm.func @kernel(%t: !llvm.ptr, %n: i64) {
        %g = llvm.mlir.addressof @stage : !llvm.ptr
        %v = llvm.load %g {merlin.global_task = 0 : i64} : !llvm.ptr -> i64
        %one = llvm.mlir.constant(1 : i64) : i64
        %s = llvm.add %v, %one {merlin.global_task = 0 : i64} : i64
        llvm.store %s, %t {merlin.global_task = 0 : i64} : i64, !llvm.ptr
        llvm.return
      }
    }""",
        context,
    )
    function = next(op for op in module.body.block.ops if op.name == "llvm.func")
    report = rq.host_compute(_buffer(["convolution"]), function=function)
    assert report.status == rq.STATUS_REPORTED
    assert report.findings[0].on_tensor == 1


# ------------------------------------------------------------------------------------------------
# Phase wiring -- both tiers, and `incomplete` is never a pass at either
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("gate", [rq.GATE_DECLARATION, rq.GATE_HOST_COMPUTE])
def test_both_gates_declare_a_phase(gate):
    assert gate in gate_phase.declared_gates()
    assert gate_phase.configured_phase(gate) == gate_phase.PHASE_REPORT


def test_a_reported_finding_blocks_only_at_fail_and_incomplete_never_blocks():
    fires = rq.host_compute(_buffer(["convolution"]), function=_host_ir(_ON_TENSOR))
    assert not fires.blocks(gate_phase.PHASE_REPORT)
    assert fires.blocks(gate_phase.PHASE_FAIL)

    absent = rq.host_compute(_buffer(["convolution"]), function=None)
    assert absent.status == gate_phase.STATUS_INCOMPLETE
    assert not absent.blocks(gate_phase.PHASE_REPORT)
    assert not absent.blocks(gate_phase.PHASE_FAIL)


def test_declared_phase_is_read_from_the_declaration_not_a_default():
    report = rq.host_compute(_buffer(["convolution"]), function=_host_ir(_ON_TENSOR))
    assert report.blocks() is gate_phase.blocks(
        gate_phase.configured_phase(rq.GATE_HOST_COMPUTE), report.status, failing=(rq.STATUS_REPORTED,)
    )
