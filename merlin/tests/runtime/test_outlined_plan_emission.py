"""Full captured graphs undergo global IR fusion without any model execution."""
from dataclasses import replace

import pytest

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects.lowering.dispatch_program import lower_model_to_dispatch_program
from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation
from merlin.xdsl_dialects.lowering.global_plan_emission import emit_global_plan
from merlin.xdsl_dialects.lowering.outlined_plan_emission import (
    OutlinedGlobalPlanEmitter, plan_dispatch_fusion,
)


# These are captured model inputs, not shortened timing capsules or synthetic operator chains.
@pytest.fixture(params=["M0_small_llama_gemmini", "M1_lstmnetvit_gemmini",
                       "M2_microvit_gemmini", "M3_host_island_seam_gemmini"])
def captured_graph(request):
    source = merlin_dir() / "contract" / "capsules" / "model" / request.param / "capsule.interface.mlir"
    module = parse_mlir_text(source.read_text())
    return lower_model_to_dispatch_program(module, prune=False)


def _plan(graph):
    # Chunking is only an emitter stress test: it crosses real fan-outs, residual boundaries,
    # reshapes, scalar glue and dispatches. It is not a target fusion profitability heuristic.
    groups = [tuple(range(i, min(i + 16, len(graph.nodes))))
              for i in range(0, len(graph.nodes), 16) if len(graph.nodes) - i > 1]
    return plan_dispatch_fusion(graph, groups, placement="compiler_function",
                                representation=lambda b: ValueRepresentation(
                                    "tensor_ssa", "logical", graph.buffers[b].dtype))


def test_real_full_graph_fusion_preserves_computation_without_simulation(captured_graph) -> None:
    outlined, graph = captured_graph
    before = str(outlined.module)
    plan = _plan(graph)
    emitter = OutlinedGlobalPlanEmitter(outlined)

    emission = emit_global_plan(graph, plan, emitter)

    assert str(outlined.module) == before
    assert emitter.module is not None
    emitter.module.verify()
    assert emitter.proof["computation"] == "expanded_driver_structurally_equivalent"
    assert emitter.proof["full_model_simulated"] is False
    assert emission.dispatch.n_dispatches < graph.n_dispatches
    assert emission.dispatch.args == graph.args
    assert emission.dispatch.results == graph.results
    assert not plan.cycles.resolved
    assert emitter.proof["timing"] == "UNKNOWN"
    assert emitter.proof["physical_movement"] == "UNKNOWN"
    assert emitter.proof["original_module_sha256"] != emitter.proof["emitted_module_sha256"]


def test_full_graph_fusion_rejects_a_silently_replaced_capture(captured_graph) -> None:
    outlined, graph = captured_graph
    plan = _plan(graph)
    changed = replace(graph, entry="different_entry")
    with pytest.raises(ValueError, match="exact unpruned"):
        OutlinedGlobalPlanEmitter(outlined).emit_global_plan(changed, plan)


def test_fusion_rejects_an_overlapping_or_nonconvex_region(captured_graph) -> None:
    _, graph = captured_graph
    for groups in ([(0, 1), (1, 2)], [(0, 2)]):
        with pytest.raises(ValueError, match="overlap|consecutive"):
            plan_dispatch_fusion(graph, groups, placement="compiler_function",
                                 representation=lambda b: ValueRepresentation(
                                     "tensor_ssa", "logical", graph.buffers[b].dtype))


def test_logical_ir_fusion_cannot_claim_a_physical_encoding_change(captured_graph) -> None:
    outlined, graph = captured_graph
    plan = _plan(graph)
    selected = list(plan.selected)
    region = next(item for item in selected if item.inputs)
    changed = replace(region.inputs[0], representation=replace(
        region.inputs[0].representation, encoding="physical_packed"))
    selected[selected.index(region)] = replace(region, inputs=(changed, *region.inputs[1:]))
    plan = replace(plan, selected=tuple(selected))
    with pytest.raises(ValueError, match="physical encodings require"):
        OutlinedGlobalPlanEmitter(outlined).emit_global_plan(graph, plan)
