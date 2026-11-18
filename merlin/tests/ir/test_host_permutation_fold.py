"""Host permutations fold into their single host contraction reader only."""

from dataclasses import replace

import pytest

from merlin.compile.application_plan import application_plan, fold_host_permutations
from merlin.compile.c_bf16_linear import emit_c
from merlin.compile.semantic_regions import extract_semantic_region, outline_semantic_regions, semantic_dispatch
from merlin.tests.runtime.test_bf16_linear_c import permuted_head_source
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation,
    CycleInterval,
    GlobalPlan,
    RegionAlternative,
    ValueRepresentation,
)


def host_plan(source, groups, placements=None):
    program = semantic_dispatch(source, entry="application")
    index = {node.outputs[0]: i for i, node in enumerate(program.nodes)}
    rows = []
    for number, keys in enumerate(groups):
        region = extract_semantic_region(source, keys)
        ports = lambda values: tuple(
            BufferRepresentation(key, ValueRepresentation("host_memory", "dense", source.node(key).type.dtype))
            for key in values)
        rows.append(RegionAlternative(
            f"region_{number}", tuple(sorted(index[key] for key in keys)), f"kernel_{number}",
            (placements or {}).get(number, "host"), CycleInterval.unknown("uncalibrated"),
            ports(node.id for node in region.nodes if node.effect in {"input", "constant"}), ports(region.outputs),
            metadata=(("kind", f"kind_{number}"),)))
    return GlobalPlan(tuple(rows), (), CycleInterval.unknown("uncalibrated"))


HEAD = (("wt",), ("xf", "wf", "dot", "out"))


def test_single_use_weight_permutation_folds_into_host_contraction():
    source = permuted_head_source(2, 5, 3)
    plan = host_plan(source, HEAD)

    class Provider:
        def plan(self, source):
            return plan

    folded = application_plan(source, Provider())
    assert folded == fold_host_permutations(source, plan)
    (region,) = folded.selected
    original = plan.selected[1]
    assert (region.id, region.implementation, region.placement, region.metadata, region.outputs) == (
        original.id, original.implementation, original.placement, original.metadata, original.outputs)
    assert [row.buffer for row in region.inputs] == ["x", "w", "seed"]
    assert folded.notes == ("host permutation wt folded into contraction region region_1",)
    outlined = outline_semantic_regions(source, entry="application", plan=folded)
    assert [(node.inputs, node.outputs) for node in outlined.emission.dispatch.nodes] == [(["x", "w", "seed"], ["out"])]
    assert "wt" not in outlined.emission.dispatch.buffers
    record, _ = emit_c(outlined.regions[0][1], "head")
    assert record["inputs"] == ["x", "w", "seed"]


def test_fold_refuses_shared_published_broadcast_and_noncontraction_permutations():
    shared = permuted_head_source(2, 5, 3, extra_reader=True)
    published = permuted_head_source(2, 5, 3, published=True)
    broadcast = permuted_head_source(2, 5, 3, broadcast=True)
    cases = [
        (shared, host_plan(shared, (*HEAD, ("other",)))),
        (shared, host_plan(shared, (("wt",), ("xf", "wf", "dot", "out", "other")))),
        (published, host_plan(published, HEAD)),
        (broadcast, host_plan(broadcast, HEAD)),
        # A widening alone is not a contraction.
        (permuted_head_source(2, 5, 3), host_plan(permuted_head_source(2, 5, 3), (("wt",), ("wf",), ("xf", "dot", "out")))),
        # Device placement on either side keeps the materialized layout.
        (permuted_head_source(2, 5, 3), host_plan(permuted_head_source(2, 5, 3), HEAD, {1: "accelerator"})),
        (permuted_head_source(2, 5, 3), host_plan(permuted_head_source(2, 5, 3), HEAD, {0: "accelerator"})),
    ]
    for source, plan in cases:
        assert fold_host_permutations(source, plan) == plan


@pytest.mark.parametrize("lhs_view,rhs_view,widened,groups", [
    (True, False, True, (("xt",), ("xf", "wf", "dot", "out"))),
    (False, True, False, (("wt",), ("out",))),
    (True, False, False, (("xt",), ("out",))),
    (True, True, True, (("xt",), ("wt",), ("xf", "wf", "dot", "out"))),
    (True, True, False, (("xt",), ("wt",), ("out",))),
])
def test_left_plain_and_two_sided_permutations_fold(lhs_view, rhs_view, widened, groups):
    source = permuted_head_source(2, 5, 3, lhs_view=lhs_view, rhs_view=rhs_view, widened=widened)
    plan = host_plan(source, groups)
    folded = fold_host_permutations(source, plan)
    (region,) = folded.selected
    consumer = plan.selected[-1]
    assert (region.id, region.outputs) == (consumer.id, consumer.outputs)
    assert [row.buffer for row in region.inputs] == ["x", "w", "seed"]
    views = [view for view, present in (("xt", lhs_view), ("wt", rhs_view)) if present]
    assert folded.notes == tuple(f"host permutation {view} folded into contraction region {consumer.id}"
                                 for view in views)
    outlined = outline_semantic_regions(source, entry="application", plan=folded)
    assert [(node.inputs, node.outputs) for node in outlined.emission.dispatch.nodes] == [(["x", "w", "seed"], ["out"])]
    record, _ = emit_c(outlined.regions[0][1], "head")
    assert [entry["permutation"] for entry in record["folded_operand_permutations"]] == views


def with_representation(plan, region_index, buffer, **changes):
    rows = list(plan.selected)
    region = rows[region_index]
    rows[region_index] = replace(region, inputs=tuple(
        replace(row, representation=replace(row.representation, **changes)) if row.buffer == buffer else row
        for row in region.inputs))
    return replace(plan, selected=tuple(rows))


@pytest.mark.parametrize("changes", [
    {"layout": "tiled"}, {"encoding": "packed"}, {"quantization": "int8-per-channel"},
    {"attributes": (("tile", "8x8"),)}, {"placement": "device_memory"}, {"dtype": "f16"},
])
def test_fold_refuses_base_representation_that_differs_from_the_view_input(changes):
    source = permuted_head_source(2, 5, 3)
    plan = host_plan(source, HEAD)
    # The base is read row-major in place of the dense view input; any other
    # representation of either buffer keeps the materialized permutation.
    for changed in (with_representation(plan, 0, "w", **changes), with_representation(plan, 1, "wt", **changes)):
        assert fold_host_permutations(source, changed) == changed
    # Matching representations on both sides still fold.
    assert fold_host_permutations(source, plan).notes


def test_fold_requires_the_contraction_without_the_permutation_too():
    source = permuted_head_source(2, 5, 3)
    plan = host_plan(source, HEAD)
    # A region that is not itself a complete contraction does not grow.
    split = host_plan(source, (("wt",), ("xf", "wf", "dot"), ("out",)))
    assert fold_host_permutations(source, split) == split
    assert fold_host_permutations(source, replace(plan, notes=("kept",))).notes == (
        "kept", "host permutation wt folded into contraction region region_1")
