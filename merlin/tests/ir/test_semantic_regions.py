"""Source slices, fusion boundaries and the executable application dispatch."""

from dataclasses import replace

import pytest

from merlin.compile.semantic_regions import extract_semantic_region, outline_semantic_regions, semantic_dispatch
from merlin.semantic_compiler.model import (
    ConstantBinding,
    KernelRequest,
    SemanticNode,
    TensorType,
    pointwise_identity_maps,
)
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation,
    CycleInterval,
    GlobalPlan,
    RegionAlternative,
    ValueRepresentation,
)
from merlin.xdsl_dialects.lowering.global_plan_emission import verify_global_plan_emission


def test_silu_fusion_match_retains_source_and_rejects_escaping_values():
    from merlin.semantic_compiler.silu_source import recognize_silu

    type_ = TensorType((3, 5), "f32", "raw-bit-pattern")
    x = SemanticNode("x", "input", (), type_, effect="input")
    y = replace(x, id="y")
    sigmoid = SemanticNode("sigmoid", "source_f32_composite_unqualified", (x.id,), type_,
                           tuple(sorted({"literal_bits": "0x3f800000",
                                         "numeric_semantics": "source-f32-composite-unqualified-v1",
                                         "operation": "sigmoid_f32"}.items())),
                           index_maps=pointwise_identity_maps(2, 1))
    product = SemanticNode("out", "f32_mul", (x.id, sigmoid.id), type_,
                           (("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                           index_maps=pointwise_identity_maps(2, 2))
    source = KernelRequest((x, y, sigmoid, product), (product.id,), ("external",),
                           ((x.id, "external"), (y.id, "external")), "synthetic")
    digest = source.digest()
    assert recognize_silu(source, product.id) == (sigmoid.id, product.id)
    assert source.digest() == digest
    reverse = replace(source, nodes=(x, y, sigmoid, replace(product, inputs=(sigmoid.id, x.id))))
    assert recognize_silu(reverse, product.id) == (sigmoid.id, product.id)
    for mutation in (
        replace(source, outputs=(product.id, sigmoid.id), output_storages=("external",) * 2),
        replace(source, nodes=(x, y, sigmoid, replace(product, inputs=(y.id, sigmoid.id)))),
        replace(source, nodes=(x, y, replace(sigmoid, attrs=()), product)),
        replace(source, nodes=(x, y, sigmoid, replace(product, index_maps=()))),
        replace(source, nodes=(x, y, sigmoid, product, replace(product, id="other"))),
    ):
        with pytest.raises(ValueError):
            recognize_silu(mutation, product.id)
    with pytest.raises(ValueError, match="stateful computation"):
        replace(sigmoid, effect="state")


def graph():
    type_ = TensorType((4,), "i32", "signed-i32")
    return KernelRequest(
        (
            SemanticNode("x", "input", (), type_, effect="input"),
            SemanticNode("z", "input", (), type_, effect="input"),
            SemanticNode("one", "constant", (), type_, effect="constant"),
            SemanticNode("a", "add", ("x", "one"), type_),
            SemanticNode("b", "multiply", ("a", "a"), type_),
            SemanticNode("c", "add", ("z", "one"), type_),
            SemanticNode("d", "add", ("b", "c"), type_),
        ),
        ("d", "b", "d", "x"),
        ("external",) * 4,
        (("z", "external"), ("x", "external")),
        "synthetic",
        constants=(ConstantBinding("one", "i32-le", "01000000" * 4),),
    )


def plan_for(source, groups):
    pure = [node for node in source.nodes if node.effect == "pure"]
    rep = ValueRepresentation("memory", "dense", pure[0].type.dtype)
    alternatives = []
    for index, indices in enumerate(groups):
        region = extract_semantic_region(source, tuple(pure[i].id for i in indices))
        inputs = tuple(node.id for node in region.nodes if node.effect in {"input", "constant"})
        alternatives.append(
            RegionAlternative(
                f"region_{index}",
                tuple(indices),
                f"kernel_{index}",
                "device",
                CycleInterval.unknown("uncalibrated"),
                tuple(BufferRepresentation(key, replace(rep, dtype=source.node(key).type.dtype)) for key in inputs),
                tuple(
                    BufferRepresentation(key, replace(rep, dtype=source.node(key).type.dtype)) for key in region.outputs
                ),
            )
        )
    return GlobalPlan(tuple(alternatives), (), CycleInterval.unknown("uncalibrated"))


def test_noncontiguous_fusion_preserves_constants_ports_and_result_order():
    source = graph()
    plan = plan_for(source, ((0, 2), (1,), (3,)))
    outlined = outline_semantic_regions(source, entry="forward", plan=plan)
    assert not verify_global_plan_emission(outlined.logical, plan, outlined.emission)
    assert outlined.emission.dispatch.results == ["d", "b", "d", "x"]
    assert outlined.emission.dispatch.nodes[0].outputs == ["a", "c"]
    assert outlined.logical.buffers["z"].arg_index == 0 and outlined.logical.buffers["x"].arg_index == 1
    region = dict(outlined.regions)["region_0"]
    assert region.constants == source.constants
    assert region.node("a") == source.node("a") and region.node("c") == source.node("c")
    assert region.outputs == ("a", "c")
    assert region.target_identity == source.target_identity and region.lowering_policy == source.lowering_policy
    assert KernelRequest.from_record(region.record()) == region
    assert (
        outline_semantic_regions(source, entry="forward", plan=plan).emission.receipt() == outlined.emission.receipt()
    )


def test_region_calls_are_ordered_by_dependencies_not_first_source_position():
    source = graph()
    # d is fused with a, so the region cannot execute before the outside b.
    plan = plan_for(source, ((0, 3), (1,), (2,)))
    with pytest.raises(ValueError, match="cyclic"):
        outline_semantic_regions(source, entry="forward", plan=plan)
    plan = plan_for(source, ((3,), (1,), (0, 2)))
    outlined = outline_semantic_regions(source, entry="forward", plan=plan)
    assert [key for key, _ in outlined.regions] == ["region_2", "region_1", "region_0"]


@pytest.mark.parametrize("fault", ["missing", "overlap", "input", "output", "dtype", "encoding", "constant_kind"])
def test_incomplete_or_incompatible_plans_are_rejected(fault):
    source = graph()
    plan = plan_for(source, ((0,), (1,), (2,), (3,)))
    rows = list(plan.selected)
    if fault == "missing":
        rows.pop()
    elif fault == "overlap":
        rows[1] = replace(rows[1], nodes=(0, 1))
    elif fault in {"input", "output"}:
        rows[0] = replace(rows[0], **{fault + "s": ()})
    elif fault in {"dtype", "encoding"}:
        reps = list(rows[1].inputs)
        reps[0] = replace(reps[0], representation=replace(reps[0].representation, **{fault: "different"}))
        rows[1] = replace(rows[1], inputs=tuple(reps))
    else:
        outlined = outline_semantic_regions(source, entry="forward", plan=plan)
        outlined.emission.dispatch.buffers["one"] = replace(outlined.emission.dispatch.buffers["one"], kind="arg")
        assert verify_global_plan_emission(outlined.logical, plan, outlined.emission)
        return
    with pytest.raises(ValueError):
        outline_semantic_regions(source, entry="forward", plan=replace(plan, selected=tuple(rows)))


def test_source_slice_retains_policies_and_rejects_undeclared_physical_views():
    source = graph()
    region = extract_semantic_region(source, ("b",))
    assert region.node("a").effect == "input"
    assert region.node("a").type == source.node("a").type
    assert region.node("b").inputs == ("a", "a")
    with pytest.raises(ValueError, match="logical graph"):
        semantic_dispatch(replace(source, physical_extents=(("z", 4), ("x", 4), ("one", 4))), entry="forward")
    for ids in (("one",), ("a", "a"), ()):
        with pytest.raises(ValueError):
            extract_semantic_region(source, ids)
    with pytest.raises(ValueError, match="explicit transitions"):
        semantic_dispatch(
            replace(source, output_storages=("external", "external", "other", "external")), entry="forward"
        )


def test_parsed_contraction_keeps_its_maps_initializer_and_passthrough_output():
    from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
    from merlin.tests.ir.test_semantic_linalg_bridge import _generic_source

    source = translate_linalg_text(_generic_source(two_outputs=True), entry="work", target_identity="synthetic").request
    ids = tuple(node.id for node in source.nodes if node.effect == "pure")
    plan = plan_for(source, (tuple(range(len(ids))),))
    outlined = outline_semantic_regions(source, entry="work", plan=plan)
    region = outlined.regions[0][1]
    assert tuple(node for node in region.nodes if node.effect == "pure") == tuple(source.node(key) for key in ids)
    assert all(source.node(key).index_maps for key in ids)
    assert outlined.emission.dispatch.results == list(source.outputs)


def test_changed_arithmetic_or_constant_bytes_invalidate_the_source_binding():
    source = graph()
    plan = plan_for(source, ((0,), (1,), (2,), (3,)))
    outlined = outline_semantic_regions(source, entry="forward", plan=plan)
    changed_constant = replace(source, constants=(replace(source.constants[0], data_hex="02000000" * 4),))
    nodes = tuple(
        replace(node, attrs=(("rounding", "different"),)) if node.id == "a" else node for node in source.nodes
    )
    for changed in (changed_constant, replace(source, nodes=nodes)):
        with pytest.raises(ValueError, match="selected source plan"):
            outlined.emit_global_plan(semantic_dispatch(changed, entry="forward"), plan)


def test_a_region_cannot_destroy_an_application_input_needed_by_another_region():
    source = graph()
    source = replace(
        source,
        nodes=tuple(
            replace(node, attrs=(("input_retention", "reusable"),)) if node.id == "x" else node for node in source.nodes
        ),
    )
    region = extract_semantic_region(source, ("a",))
    assert dict(region.node("x").attrs)["input_retention"] == "preserve"
    assert dict(source.node("x").attrs)["input_retention"] == "reusable"
