"""Repeated tensor coordinates authorize exact reuse, never equal shape alone."""

import hashlib
import json
import struct

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.repeated_pointwise import hoist_repeated_pointwise
from merlin.xdsl_dialects._common import text


def fixture(*, strict=False):
    return """builtin.module {
  func.func @forward(%a: tensor<2x1x3xf32>, %scale: tensor<8x1xf32>) -> tensor<8x3xi8> ATTR {
    %e = tensor.empty() : tensor<2x4x3xf32>
    %repeat = linalg.generic {
      indexing_maps = [affine_map<(i,j,k)->(i,0,k)>, affine_map<(i,j,k)->(i,j,k)>],
      iterator_types = ["parallel","parallel","parallel"]
    } ins(%a : tensor<2x1x3xf32>) outs(%e : tensor<2x4x3xf32>) {
      ^bb0(%x: f32, %unused: f32):
        linalg.yield %x : f32
    } -> tensor<2x4x3xf32>
    %flat = tensor.collapse_shape %repeat [[0,1,2]] : tensor<2x4x3xf32> into tensor<24xf32>
    %view = tensor.expand_shape %flat [[0,1]] output_shape [8,3] : tensor<24xf32> into tensor<8x3xf32>
    %out = tensor.empty() : tensor<8x3xi8>
    %r = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->(i,0)>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel","parallel"]
    } ins(%view, %scale : tensor<8x3xf32>, tensor<8x1xf32>) outs(%out : tensor<8x3xi8>) {
      ^bb0(%x: f32, %s: f32, %unused: i8):
        %d = arith.divf %x, %s : f32
        %round = math.roundeven %d : f32
        %y = arith.fptosi %round : f32 to i8
        linalg.yield %y : i8
    } -> tensor<8x3xi8>
    return %r : tensor<8x3xi8>
  }
}""".replace("ATTR", "attributes {strictfp}" if strict else "")


def binding(words):
    raw = struct.pack("<8f", *words)
    return lambda argument: raw if argument.index == 1 else None


def transpose_fixture():
    source = fixture()
    before, tail = source.split("    %out =", 1)
    before = before.replace("-> tensor<8x3xi8>", "-> tensor<3x8xi8>")
    before += """    %tinit = tensor.empty() : tensor<3x8xf32>
    %transposed = linalg.transpose ins(%view : tensor<8x3xf32>) outs(%tinit : tensor<3x8xf32>)
      permutation = [1,0]
"""
    tail = tail.replace("8x3", "3x8").replace("ins(%view,", "ins(%transposed,")
    tail = tail.replace("affine_map<(i,j)->(i,0)>", "affine_map<(i,j)->(j,0)>")
    return before + "    %out =" + tail


def computed_fixture():
    source = fixture().replace("%scale: tensor<8x1xf32>)", "%scale: tensor<8x1xf32>, %runtime: tensor<f32>)")
    producer = """    %pinit = tensor.empty() : tensor<8x1xf32>
    %p = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->()>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel","parallel"]
    } ins(%scale, %runtime : tensor<8x1xf32>, tensor<f32>) outs(%pinit : tensor<8x1xf32>) {
      ^bb0(%s: f32, %r: f32, %unused: f32):
        %m = arith.mulf %s, %r : f32
        linalg.yield %m : f32
    } -> tensor<8x1xf32>
    %qinit = tensor.empty() : tensor<8x1xf32>
    %q = linalg.generic {
      indexing_maps = [affine_map<(i,j)->(i,j)>, affine_map<(i,j)->()>, affine_map<(i,j)->(i,j)>],
      iterator_types = ["parallel","parallel"]
    } ins(%p, %runtime : tensor<8x1xf32>, tensor<f32>) outs(%qinit : tensor<8x1xf32>) {
      ^bb0(%m: f32, %r: f32, %unused: f32):
        %sum = arith.addf %m, %r : f32
        linalg.yield %sum : f32
    } -> tensor<8x1xf32>
"""
    return source.replace("    %out =", producer + "    %out =").replace("ins(%view, %scale", "ins(%view, %q")


def test_exact_repeat_reduces_all_scalar_work_and_retains_body_order():
    source = parse_mlir_text(fixture())
    result, report = hoist_repeated_pointwise(source, immutable_argument_loader=binding([2] * 4 + [3] * 4))
    assert report["rewrites"] == 1
    assert report["removed_scalar_evaluations"] == {"arith.divf": 18, "math.roundeven": 18, "arith.fptosi": 18}
    small = next(
        op for op in result.walk() if op.name == "linalg.generic" and any(x.name == "arith.divf" for x in op.walk())
    )
    assert tuple(small.results[0].type.get_shape()) == (2, 3)
    assert [op.name for op in small.body.block.ops] == ["arith.divf", "math.roundeven", "arith.fptosi", "linalg.yield"]
    source.verify()
    result.verify()


@pytest.mark.parametrize(
    "loader", [None, binding([2, 2, 2, 3, 3, 3, 3, 3]), binding([0.0, -0.0, 0.0, 0.0, 3, 3, 3, 3])]
)
def test_unknown_or_differing_scale_words_leave_original_ir(loader):
    source = parse_mlir_text(fixture())
    result, report = hoist_repeated_pointwise(source, immutable_argument_loader=loader)
    assert report["rewrites"] == 0
    assert text(result, generic=True) == text(source, generic=True)


@pytest.mark.parametrize("attribute", ["strictfp", 'passthrough = ["strictfp"]'])
def test_strict_function_is_not_moved(attribute):
    source = parse_mlir_text(fixture(strict=True).replace("attributes {strictfp}", "attributes {" + attribute + "}"))
    result, report = hoist_repeated_pointwise(source, immutable_argument_loader=binding([2] * 4 + [3] * 4))
    assert report["rewrites"] == 0
    assert text(result, generic=True) == text(source, generic=True)


def test_output_initializer_read_prevents_domain_reduction():
    source = parse_mlir_text(fixture().replace("linalg.yield %y : i8", "linalg.yield %unused : i8"))
    result, report = hoist_repeated_pointwise(source, immutable_argument_loader=binding([2] * 4 + [3] * 4))
    assert report["rewrites"] == 0
    assert text(result, generic=True) == text(source, generic=True)


def test_unknown_effect_and_fast_math_refuse():
    for modified in (
        fixture()
        .replace("%d = arith.divf", "%opaque = func.call @opaque() : () -> f32\n        %d = arith.divf")
        .replace("builtin.module {", "builtin.module {\n func.func private @opaque() -> f32"),
        fixture().replace("arith.divf %x, %s : f32", "arith.divf %x, %s fastmath<fast> : f32"),
    ):
        source = parse_mlir_text(modified)
        result, report = hoist_repeated_pointwise(source, immutable_argument_loader=binding([2] * 4 + [3] * 4))
        assert report["rewrites"] == 0
        assert text(result, generic=True) == text(source, generic=True)


def test_computed_scale_dag_preserves_runtime_scalar_and_each_original_body():
    source = parse_mlir_text(computed_fixture())
    result, report = hoist_repeated_pointwise(source, immutable_argument_loader=binding([2] * 4 + [3] * 4))
    assert report["rewrites"] == 1 and report["computed_side_producers"] == 2
    assert report["removed_scalar_evaluations"]["arith.divf"] == 18
    assert report["cloned_side_scalar_evaluations"] == {"arith.mulf": 2, "arith.addf": 2}
    small = [
        op for op in result.walk() if op.name == "linalg.generic" and tuple(op.results[0].type.get_shape()) == (2, 1)
    ]
    assert [[x.name for x in op.body.block.ops] for op in small] == [
        ["arith.mulf", "linalg.yield"],
        ["arith.addf", "linalg.yield"],
    ]
    assert all(op.inputs[1] is result.body.block.first_op.body.block.args[2] for op in small)
    result.verify()


@pytest.mark.parametrize(
    "loader", [None, binding([2, 2, 2, 3, 3, 3, 3, 3]), binding([0.0, -0.0, 0.0, 0.0, 3, 3, 3, 3])]
)
def test_computed_scale_cannot_hide_runtime_or_differing_repeated_input(loader):
    source = parse_mlir_text(computed_fixture())
    result, report = hoist_repeated_pointwise(source, immutable_argument_loader=loader)
    assert report["rewrites"] == 0 and text(result, generic=True) == text(source, generic=True)


def test_computed_scale_strict_or_effectful_producer_is_refused():
    for source in (
        computed_fixture().replace("%m = arith.mulf %s, %r : f32", "%m = arith.mulf %s, %r fastmath<fast> : f32"),
        computed_fixture().replace("linalg.yield %m : f32", "linalg.yield %unused : f32"),
    ):
        original = parse_mlir_text(source)
        result, report = hoist_repeated_pointwise(original, immutable_argument_loader=binding([2] * 4 + [3] * 4))
        assert report["rewrites"] == 0 and text(result, generic=True) == text(original, generic=True)


def test_computed_scale_other_live_use_keeps_its_full_original_domain():
    source = computed_fixture().replace("-> tensor<8x3xi8>  {", "-> (tensor<8x3xi8>, tensor<8x1xf32>)  {")
    source = source.replace("return %r : tensor<8x3xi8>", "return %r, %q : tensor<8x3xi8>, tensor<8x1xf32>")
    module = parse_mlir_text(source)
    result, report = hoist_repeated_pointwise(module, immutable_argument_loader=binding([2] * 4 + [3] * 4))
    assert report["rewrites"] == 1
    returned = result.body.block.first_op.body.block.last_op.operands[1]
    assert tuple(returned.type.get_shape()) == (8, 1)
    assert [op.name for op in returned.owner.body.block.ops] == ["arith.addf", "linalg.yield"]
    result.verify()


def test_unselected_preparation_never_opens_a_bundle(tmp_path, monkeypatch):
    from merlin.llvmlower.repeated_pointwise import prepare_repeated_pointwise

    monkeypatch.delenv("MERLIN_PREPARE_REPEATED_POINTWISE", raising=False)
    missing = tmp_path / "not-a-source.mlir"
    assert prepare_repeated_pointwise(missing, tmp_path / "out", bundle_dir=tmp_path / "missing") == missing
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind, updates, rewrites", [
    ("buffer", None, 1), ("input", None, 0), ("buffer", "states", 0), ("buffer", "streams", 0),
])
def test_normal_stored_constant_binding_requires_explicit_immutable_manifest(
    tmp_path, monkeypatch, kind, updates, rewrites
):
    from merlin.llvmlower.repeated_pointwise import prepare_repeated_pointwise

    monkeypatch.setenv("MERLIN_PREPARE_REPEATED_POINTWISE", "1")
    source = tmp_path / "model.mlir"
    source.write_text(fixture())
    raw = struct.pack("<8f", *([2] * 4 + [3] * 4))
    header = json.dumps({"scale": {"dtype": "F32", "shape": [8, 1], "data_offsets": [0, len(raw)]}}).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + raw)
    manifest = {"1": {"kind": kind, "weight": "scale", "dtype": "float32", "shape": [8, 1]}}
    (tmp_path / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    if updates is not None:
        (tmp_path / "session_contract.yaml").write_text(f"version: 1\n{updates}:\n- input_arg: 1\n")
    selected = prepare_repeated_pointwise(source, tmp_path / "prepared", bundle_dir=tmp_path)
    report = json.loads((tmp_path / "prepared/report.json").read_text())
    assert report["rewrites"] == rewrites
    if rewrites:
        assert selected != source
        assert report["immutable_argument_sources"][0]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert report["immutable_argument_sources"][0]["argument_index"] == 1
    else:
        assert selected == source
        assert report["immutable_argument_sources"] == []


def test_stored_preparation_rechecks_generated_caller_updates_before_publication(tmp_path, monkeypatch):
    from merlin.llvmlower import repeated_pointwise

    monkeypatch.setenv("MERLIN_PREPARE_REPEATED_POINTWISE", "1")
    source = tmp_path / "model.mlir"
    source.write_text(fixture())
    raw = struct.pack("<8f", *([2] * 4 + [3] * 4))
    header = json.dumps({"scale": {"dtype": "F32", "shape": [8, 1], "data_offsets": [0, len(raw)]}}).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + raw)
    manifest = {"1": {"kind": "buffer", "weight": "scale", "dtype": "float32", "shape": [8, 1]}}
    (tmp_path / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    original = repeated_pointwise.hoist_repeated_pointwise

    def change(*args, **kwargs):
        selected = original(*args, **kwargs)
        (tmp_path / "session_contract.yaml").write_text("version: 1\nstates:\n- input_arg: 1\n")
        return selected

    monkeypatch.setattr(repeated_pointwise, "hoist_repeated_pointwise", change)
    with pytest.raises(ValueError, match="generated-caller binding"):
        repeated_pointwise.prepare_repeated_pointwise(source, tmp_path / "prepared", bundle_dir=tmp_path)
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("transpose", [False, True])
def test_fresh_native_outputs_include_ties_negative_values_and_all_repeated_rows(tmp_path, transpose):
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    source = parse_mlir_text(transpose_fixture() if transpose else fixture())
    transformed, _ = hoist_repeated_pointwise(source, immutable_argument_loader=binding([2] * 4 + [3] * 4))
    scale = np.array([2] * 4 + [3] * 4, dtype=np.float32).reshape(8, 1)
    functions = []
    for name, module in [("original", source), ("prepared", transformed)]:
        result = lower_model(text(module, generic=True), tmp_path / name)
        functions.append(
            HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.host_so.parent.resolve()))
        )
    for values in ([-5, -3, -1, 1.5, 4.5, 7.5], [0, -0.0, 2, 3, 6, 9]):
        value = np.array(values, dtype=np.float32).reshape(2, 1, 3)
        expected = np.rint(np.repeat(value, 4, axis=1).reshape(8, 3) / scale).astype(np.int8)
        if transpose:
            expected = expected.T.copy()
        for invoke in functions:
            output = np.full(expected.shape, 77, dtype=np.int8)
            invoke(
                [(value.ctypes.data, value.shape), (scale.ctypes.data, scale.shape), (output.ctypes.data, output.shape)]
            )
            np.testing.assert_array_equal(output, expected)


def test_computed_scale_dag_fresh_native_outputs_for_distinct_runtime_scalars(tmp_path):
    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    source = parse_mlir_text(computed_fixture())
    transformed, _ = hoist_repeated_pointwise(source, immutable_argument_loader=binding([2] * 4 + [3] * 4))
    functions = []
    for name, module in [("original", source), ("prepared", transformed)]:
        result = lower_model(text(module, generic=True), tmp_path / name)
        functions.append(
            HostModel.load(str(result.host_so), image_policy=PrivateHostImagePolicy(result.host_so.parent.resolve()))
        )
    value = np.array([-5, -3, -1, 1.5, 4.5, 7.5], dtype=np.float32).reshape(2, 1, 3)
    scale = np.array([2] * 4 + [3] * 4, dtype=np.float32).reshape(8, 1)
    for scalar in (0.5, 1.25, -0.75):
        runtime = np.array(scalar, dtype=np.float32)
        denominator = (scale * runtime).astype(np.float32) + runtime
        expected = np.rint(np.repeat(value, 4, axis=1).reshape(8, 3) / denominator).astype(np.int8)
        for invoke in functions:
            output = np.full(expected.shape, 77, dtype=np.int8)
            invoke([(array.ctypes.data, array.shape) for array in (value, scale, runtime, output)])
            np.testing.assert_array_equal(output, expected)
