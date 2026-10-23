"""Unused literals cannot obscure a complete scalar conversion or copy proof."""

from mlir_fixtures import assert_unchanged, forward, generic, pointwise

from merlin.llvmlower.structural_folds import prune_unused_tensor_inputs


def test_unused_input_is_removed_with_its_map_and_argument():
    ins = [("%x", "tensor<4xi1>"), ("%c", "tensor<4xf32>")]
    body = generic(ins, "tensor<4xf32>", "%f=arith.uitofp %a : i1 to f32", yields="%f")
    module = forward("%x:tensor<4xi1>,%c:tensor<4xf32>", "tensor<4xf32>", body)
    assert prune_unused_tensor_inputs(module) == 1
    op = next(op for op in module.walk() if op.name == "linalg.generic")
    assert len(op.inputs) == 1 and len(op.get_indexing_maps()) == 2
    assert len(op.body.block.args) == 2
    assert prune_unused_tensor_inputs(module) == 0


def test_live_tensor_dependencies_are_not_pruned():
    assert_unchanged(pointwise(["tensor<4xi32>"], "tensor<4xi32>", "", yields="%a"), prune_unused_tensor_inputs)


def test_unused_scalar_operand_still_defines_a_reduction_iteration_domain():
    seed = "%zero=arith.constant 0.0 : f32\n%seed=tensor.splat %zero : tensor<f32>\n"
    maps = ["affine_map<(d0)->(d0)>", "affine_map<(d0)->()>"]
    total = generic(
        [("%x", "tensor<4xf32>")],
        "tensor<f32>",
        "",
        yields="%acc",
        maps=maps,
        iterators=["reduction"],
        out="%seed",
        acc="%acc",
    )
    module = forward("%x:tensor<4xf32>", "tensor<f32>", seed + total)
    assert_unchanged(module, prune_unused_tensor_inputs)
