"""Bundle layout, approximation declarations and content-addressed storage behave as the contract states."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.model_check import bundle as B
from merlin.model_check import catalog as CAT
from merlin.model_check.approximations import ApproximationError, declared
from merlin.model_check.models import load


def _models():
    return load(repo_root() / "examples" / "gemmini" / "numerics" / "models.yaml")


def test_stage_directories_follow_the_sealed_capture_layout(tmp_path):
    models = _models()
    assert B.stage_dir(tmp_path, models.model("resnet50"), "model") == tmp_path / "resnet50/run/capture"
    assert B.stage_dir(tmp_path, models.model("smolvla"), "flow_denoise") == (
        tmp_path / "smolvla/run/capture/stages/flow_denoise"
    )


def test_blobs_are_stored_once_by_content_and_typed_by_header(tmp_path):
    store = CAT.BlobStore(tmp_path)
    a = np.arange(12, dtype=np.int8).reshape(3, 4)
    k1, k2 = store.put(a), store.put(a.copy())
    assert k1 == k2 and len(list(tmp_path.glob("*.npy"))) == 1
    assert store.put(a.astype(np.int32)) != k1  # same values, different dtype: a different blob
    assert np.array_equal(store.get(k1), a)


def _manifest(tmp_path: Path, text: str) -> Path:
    (tmp_path / "manifest.yaml").write_text(text)
    return tmp_path


def test_no_numerics_block_means_every_kernel_is_exact(tmp_path):
    assert declared(_manifest(tmp_path, "artifact_type: x\n")) == []


def test_a_well_formed_approximation_is_returned(tmp_path):
    rows = declared(
        _manifest(
            tmp_path,
            "numerics:\n  approximations:\n    - id: attn_split\n      opcodes: [ATTENTION_QK]\n"
            "      description: d\n      reason: r\n",
        )
    )
    assert rows[0]["id"] == "attn_split"


@pytest.mark.parametrize(
    "text",
    [
        "numerics:\n  exactness: off\n",
        "numerics:\n  approximations:\n    - id: A\n      opcodes: [X]\n      description: d\n      reason: r\n",
        "numerics:\n  approximations:\n    - id: a\n      opcodes: []\n      description: d\n      reason: r\n",
        "numerics:\n  approximations:\n    - id: a\n      opcodes: [X]\n      description: d\n",
    ],
)
def test_a_malformed_declaration_is_refused(tmp_path, text):
    with pytest.raises(ApproximationError):
        declared(_manifest(tmp_path, text))


def test_census_counts_a_generic_contraction_by_its_iteration_domain():
    from merlin.common import mlir_query as mq

    text = """
func.func @forward(%a: tensor<2x8x5xf32>, %b: tensor<2x5x6xf32>, %c: tensor<2x8x6xf32>) -> tensor<2x8x6xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>, affine_map<(b, m, n, k) -> (b, k, n)>, affine_map<(b, m, n, k) -> (b, m, n)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<2x8x5xf32>, tensor<2x5x6xf32>) outs(%c : tensor<2x8x6xf32>) {
  ^bb0(%x: f32, %y: f32, %z: f32):
    %p = arith.mulf %x, %y : f32
    %s = arith.addf %z, %p : f32
    linalg.yield %s : f32
  } -> tensor<2x8x6xf32>
  return %r : tensor<2x8x6xf32>
}
"""
    op = next(o for o in mq.parse(text).walk() if o.name == "linalg.generic")
    assert CAT._is_contraction(op)
    assert CAT._macs(op) == 2 * 8 * 6 * 5


_MATMUL_GENERIC = """
  %{name} = linalg.generic {{indexing_maps = [affine_map<(m, n, k) -> (m, k)>, affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>], iterator_types = ["parallel", "parallel", "reduction"]}}
      ins(%{a}, %{b} : tensor<4x5x{t}>, tensor<5x6x{t}>) outs(%{c} : tensor<4x6x{acc}>) {{
  ^bb0(%x: {t}, %y: {t}, %z: {acc}):
    {body}
    linalg.yield %s : {acc}
  }} -> tensor<4x6x{acc}>"""


def test_census_files_an_unrouted_integer_contraction_as_policy_not_recipe(tmp_path):
    """An integerized contraction (int x int operands, no dequantize feeding it) WAS quantized by the
    recipe; left on the host it is a policy refusal. A float one with no captured quantization is not."""
    widen = "%xi = arith.extsi %x : i8 to i32\n    %yi = arith.extsi %y : i8 to i32\n    "
    integer = _MATMUL_GENERIC.format(
        name="r0",
        a="a",
        b="b",
        c="c",
        t="i8",
        acc="i32",
        body=widen + "%p = arith.muli %xi, %yi : i32\n    %s = arith.addi %z, %p : i32",
    )
    floating = _MATMUL_GENERIC.format(
        name="r1",
        a="f",
        b="g",
        c="h",
        t="f32",
        acc="f32",
        body="%p = arith.mulf %x, %y : f32\n    %s = arith.addf %z, %p : f32",
    )
    (tmp_path / "model.mlir").write_text(
        "func.func @forward(%a: tensor<4x5xi8>, %b: tensor<5x6xi8>, %c: tensor<4x6xi32>, %f: tensor<4x5xf32>, "
        "%g: tensor<5x6xf32>, %h: tensor<4x6xf32>) -> (tensor<4x6xi32>, tensor<4x6xf32>) {"
        + integer
        + floating
        + "\n  return %r0, %r1 : tensor<4x6xi32>, tensor<4x6xf32>\n}\n"
    )
    catalog = CAT.ProgramCatalog("t", tmp_path, tmp_path, tmp_path, {}, {"routed": []}, {})
    doc = CAT.census(catalog)
    assert [row["placement"] for row in doc["contractions"]] == ["host:policy_refused", "host:not_quantized_by_recipe"]
    assert doc["contractions"][0]["dtypes"] == ["i8", "i8"]


def _activation_catalog(batch):
    """One routed symbol: a contraction of two run-time int8 operands, its interface stating one slice."""
    from merlin.model_check import kernel_reference as KR

    entry = {
        "name": "scores",
        "kind": "op",
        "operand_dtype": "int8",
        "scale_granularity": None,
        "epilogue": [],
        "op": "matmul",
        "M": 4,
        "K": 20,
        "N": 6,
        "stationary_operand": "activation",
        "cat": "layers",
        "label": "dev",
        "source_role": "model_derived",
        "source_reference": "test",
    }
    target = "gemmini"  # a target whose interface emitter is registered; the shape is what is tested
    interface = KR.interface_for(entry, target=target, operand_dtype="int8", accum_dtype="int32")
    sidecar = {
        "call_buffers": {"sym": [{"role": "input_0"}, {"role": "weight_0"}, {"role": "out_0"}]},
        "programs": {"sym": {"batch_shape": list(batch)} if batch else {}},
    }
    return CAT.ProgramCatalog(target, Path("."), Path("."), Path("."), {}, sidecar, {"sym": interface})


def test_a_batched_host_call_is_one_device_call_per_slice(tmp_path):
    """The interface states one slice and the host shim loops over the batch (device_shim's logical ABI),
    so the reference answers each slice with the interface, records each as its own device call, and
    hands the host the slices stacked back under the batch."""
    rng = np.random.default_rng(0)
    lhs = rng.integers(-128, 128, size=(3, 4, 20)).astype(np.int8)
    rhs = rng.integers(-128, 128, size=(3, 20, 6)).astype(np.int8)
    calls = []
    store = CAT.BlobStore(tmp_path)
    answer = CAT._answers(_activation_catalog((3,)), calls, blobs=store, keep_call=None)["sym"]
    (out,) = answer([lhs, rhs, np.zeros((3, 4, 6), np.int32)])
    assert out.shape == (3, 4, 6)
    expected = np.einsum("bmk,bkn->bmn", lhs.astype(np.int64), rhs.astype(np.int64))
    assert np.array_equal(out.astype(np.int64), expected)
    assert [c["ordinal"] for c in calls] == [0, 1, 2]
    assert [store.get(k).shape for k in calls[1]["operands"]] == [(4, 20), (20, 6), (4, 6)]
    assert np.array_equal(store.get(calls[2]["operands"][1]), rhs[2])
    with pytest.raises(CAT.CatalogError):
        answer([lhs[:2], rhs[:2], np.zeros((2, 4, 6), np.int32)])


def test_an_unbatched_host_call_is_its_own_device_call(tmp_path):
    rng = np.random.default_rng(1)
    lhs = rng.integers(-128, 128, size=(4, 20)).astype(np.int8)
    rhs = rng.integers(-128, 128, size=(20, 6)).astype(np.int8)
    calls = []
    answer = CAT._answers(_activation_catalog(()), calls, blobs=CAT.BlobStore(tmp_path), keep_call=None)["sym"]
    (out,) = answer([lhs, rhs, np.zeros((4, 6), np.int32)])
    assert np.array_equal(out.astype(np.int64), lhs.astype(np.int64) @ rhs.astype(np.int64))
    assert len(calls) == 1
