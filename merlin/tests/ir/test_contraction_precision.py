"""Source graph precision binding preserves complete shapes and initialization."""

from dataclasses import replace

import pytest

from merlin.compile.semantic_regions import extract_semantic_region
from merlin.semantic_compiler.contraction_precision import lower_semantic_contraction
from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.precision_values import FloatBoundarySpec
from merlin.tests.ir.test_precision_intake import SOURCE, _tile_profile


def selected():
    profile = _tile_profile()
    profile = replace(
        profile,
        conversion_specs=(
            *(FloatBoundarySpec("f32-to-e4m3-rne", "fp8_e4m3", "encode") for _ in range(2)),
            FloatBoundarySpec("bf16-to-f32-exact", "bf16", "decode"),
        ),
    )
    graph = translate_linalg_text(
        SOURCE, entry="work", target_identity=profile.target_identity, lowering_policy="diagnostic"
    ).request
    first = next(node for node in graph.nodes if node.op == "source_f32_matmul_unqualified")
    return extract_semantic_region(graph, (first.id,)), profile


def test_parsed_contraction_and_serialized_graph_have_the_same_precision_plan():
    source, profile = selected()
    result = lower_semantic_contraction(source, profile=profile)
    assert [node.type.shape for node in result.request.nodes] == [(2, 3), (4, 3), (2, 4)]
    assert result.request.physical_extents == ()
    assert result.inputs[1].target_axes_from_source == (1, 0)
    assert result.inputs[0].source_value.kind == "semantic_node"
    assert source.nodes[result.inputs[0].source_value.index].id == source.nodes[-1].inputs[0]
    assert (
        lower_semantic_contraction(KernelRequest.from_record(source.record()), profile=profile).record()
        == result.record()
    )


def test_scaled_operands_require_the_inverse_result_scale():
    source, profile = selected()
    left, right, output = profile.conversion_specs
    profile = replace(
        profile,
        conversion_specs=(
            replace(left, pre_encode_scale_exponent=4),
            replace(right, pre_encode_scale_exponent=3),
            output,
        ),
    )
    with pytest.raises(ValueError, match="restore both operand scales"):
        lower_semantic_contraction(source, profile=profile)
    profile = replace(
        profile, conversion_specs=(*profile.conversion_specs[:2], replace(output, post_decode_scale_exponent=-7))
    )
    result = lower_semantic_contraction(source, profile=profile)
    assert result.record()["output"]["conversion_spec"]["post_decode_scale_exponent"] == -7


@pytest.mark.parametrize(
    "fault", ["negative_zero", "missing_zero", "maps", "rounding", "policy", "target", "extra_output"]
)
def test_changed_source_or_target_preconditions_are_rejected(fault):
    source, profile = selected()
    result = source.nodes[-1]
    if fault == "negative_zero":
        seed = source.constants[0]
        source = replace(source, constants=(replace(seed, data_hex="00000080" * (len(seed.data_hex) // 8)),))
    elif fault == "missing_zero":
        seed = source.constants[0]
        source = replace(source, constants=(replace(seed, data_hex="0000803f" * (len(seed.data_hex) // 8)),))
    elif fault in {"maps", "rounding", "policy"}:
        if fault == "maps":
            result = replace(result, index_maps=(result.index_maps[0], result.index_maps[0], *result.index_maps[2:]))
        elif fault == "rounding":
            result = replace(result, attrs=(("reduction_order", "axis-ascending"),))
        else:
            result = replace(result, type=replace(result.type, numerical_policy="different"))
        source = replace(source, nodes=(*source.nodes[:-1], result))
    elif fault == "target":
        profile = replace(profile, target_identity="other-target")
    else:
        source = replace(
            source, outputs=(*source.outputs, source.nodes[0].id), output_storages=("external", "external")
        )
    with pytest.raises(ValueError):
        lower_semantic_contraction(source, profile=profile)
