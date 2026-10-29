"""Extract current admitted SSA rather than supplying component entry graphs."""

from dataclasses import replace

import pytest
from test_connected_spatial import SOURCE, fixture
from xdsl.dialects import func
from xdsl.dialects.builtin import FunctionType

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.connected_spatial import extract_connected_spatial_region, group_connected_spatial
from merlin.xdsl_dialects._common import text


def test_actual_calls_destinations_and_all_live_results_are_cloned_without_source_mutation():
    module, rewrite = fixture()
    before = text(module, generic=True)
    original_calls = [op for op in module.walk() if isinstance(op, func.CallOp)]
    region = extract_connected_spatial_region(module, rewrite, ("producer", "pool"))
    assert text(module, generic=True) == before
    assert region.inputs == tuple(original_calls[0].arguments[:-1])
    assert region.outputs == tuple(original_calls[1].results)
    assert len(group_connected_spatial(region.module, region.rewrite).receipt["groups"]) == 1
    cloned_calls = [op for op in region.module.walk() if isinstance(op, func.CallOp)]
    assert [op.callee for op in cloned_calls] == [op.callee for op in original_calls]
    assert cloned_calls[1].arguments[0] is cloned_calls[0].results[0]
    assert list(region.module.body.block.first_op.body.block.last_op.arguments) == list(cloned_calls[1].results)
    assert region.rewrite.entries == rewrite.entries
    assert region.rewrite.programs == rewrite.programs
    assert region.receipt["target_execution"] is False


def test_full_owner_keeps_unrelated_public_outputs_and_all_pool_observers():
    module, rewrite = fixture()
    owner = module.body.block.first_op
    returned = owner.body.block.last_op
    pool_result = returned.arguments[0]
    retained_input = owner.body.block.args[0]
    owner.body.block.erase_op(returned, safe_erase=False)
    owner.body.block.add_op(func.ReturnOp(pool_result, pool_result, retained_input))
    owner.properties["function_type"] = FunctionType.from_lists(
        owner.function_type.inputs, [pool_result.type, pool_result.type, retained_input.type]
    )
    module.verify()
    before = text(module, generic=True)
    region = extract_connected_spatial_region(module, rewrite, ("producer", "pool"))
    assert region.outputs == (pool_result,)
    assert text(module, generic=True) == before
    assert len(owner.function_type.outputs) == 3
    assert len(tuple(pool_result.uses)) == 2


def test_region_sidecar_keeps_only_original_prepack_rows_for_its_actual_groups():
    module, rewrite = fixture()
    prepack = {
        "groups": [{"group": 5, "weight": {"array": "actual"}, "bias": {"elements": 17}},
                   {"group": 6}, {"group": 10, "weight": {"array": "unrelated"}}],
        "skipped": [{"group": 11}], "folded_bias_elements": 17, "device_weights": 2,
    }
    rewrite = replace(rewrite, prepack=prepack,
                      routed=tuple(replace(r, group=i) for r, i in zip(rewrite.routed, (5, 6), strict=True)))
    region = extract_connected_spatial_region(module, rewrite, ("producer", "pool"))
    assert [row["group"] for row in region.rewrite.prepack["groups"]] == [5, 6]
    assert region.rewrite.prepack["device_weights"] == 1
    assert region.rewrite.prepack["folded_bias_elements"] == 17
    assert region.rewrite.prepack["skipped"] == []
    assert [row["group"] for row in rewrite.prepack["groups"]] == [5, 6, 10]


@pytest.mark.parametrize("mutation", ["observer", "effect", "source_statement", "callee"])
def test_extraction_cannot_bypass_original_source_closure_or_numerical_statement(mutation):
    module, rewrite = fixture()
    if mutation == "observer":
        source = SOURCE.replace(
            "  %y=tensor.empty()", "  func.call @observe(%p):(tensor<50x17xi8>)->()\n  %y=tensor.empty()"
        )
        source = source.replace(
            " func.func private @pool", " func.func private @observe(tensor<50x17xi8>)\n func.func private @pool"
        )
        module = parse_mlir_text(source)
    elif mutation == "effect":
        source = SOURCE.replace("  %y=tensor.empty()", "  func.call @effect():()->()\n  %y=tensor.empty()")
        source = source.replace(" func.func private @pool", " func.func private @effect()\n func.func private @pool")
        module = parse_mlir_text(source)
    elif mutation == "source_statement":
        rewrite.entries["pool"]["acc_scale"] = 0.25
    pair = ("producer", "missing") if mutation == "callee" else ("producer", "pool")
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="unique closed"):
        extract_connected_spatial_region(module, rewrite, pair)
    assert text(module, generic=True) == before
