"""AOT quantization of constant contraction weights used by the W8A8 lowering."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower import weight_prequant as wq


def _real_bundle(name: str) -> Path:
    """Find an optional local recapture without baking in a developer's path."""
    root = Path(repo_root())
    candidates = (
        root / "out" / "artifacts" / "recaptures" / name,
        root.parent / "oscar-merlin" / "out" / "artifacts" / "recaptures" / name,
    )
    return next((path for path in candidates if (path / "model.mlir").is_file()),
                candidates[0])


REAL_BUNDLE = _real_bundle(
    "lstmnetvit_int8_command_grouped_direct_consistent_pretransposed")
REAL_RESNET_BUNDLE = _real_bundle("resnet50_v1_5_int8_w8a8_consistent")


MLIR = '''builtin.module attributes {prov.weights_file = "SRC/weights.safetensors", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<2x3xf32>, %1: tensor<1x2xf32>) -> tensor<1x3xf32> {
    %2 = tensor.empty() : tensor<1x3xf32>
    %3 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%3 : f32) outs(%2 : tensor<1x3xf32>) -> tensor<1x3xf32>
    %5 = linalg.matmul ins(%1, %0 : tensor<1x2xf32>, tensor<2x3xf32>) outs(%4 : tensor<1x3xf32>) -> tensor<1x3xf32>
    func.return %5 : tensor<1x3xf32>
  }
}
'''


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    spell = {np.dtype(np.float32): "F32", np.dtype(np.int8): "I8"}
    header, payload, off = {}, [], 0
    for name, value in tensors.items():
        value = np.ascontiguousarray(value)
        data = value.tobytes()
        header[name] = {"dtype": spell[value.dtype], "shape": list(value.shape),
                        "data_offsets": [off, off + len(data)]}
        payload.append(data)
        off += len(data)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(payload))


def _bundle(root: Path, *, weight_name: str = "net.lstm.weight_ih_l0") -> tuple[Path, np.ndarray]:
    root.mkdir()
    weight = np.array([[-4.0, -1.0, 0.0], [2.0, 3.0, 8.0]], dtype=np.float32)
    other = np.array([9.0, 10.0], dtype=np.float32)
    (root / "model.mlir").write_text(MLIR.replace("SRC", str(root)))
    (root / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "param", "weight": weight_name,
              "dtype": "float32", "shape": [2, 3]},
        "1": {"kind": "param", "weight": "other", "dtype": "float32", "shape": [2]},
    }))
    _write_safetensors(root / "weights.safetensors", {
        weight_name: weight, "other": other})
    (root / "session_contract.yaml").write_text("version: 1\n")
    return root, weight


def test_prequantized_bundle_selects_by_storage_and_direct_matmul_use_not_weight_name(tmp_path):
    weight_name = "net.encoder.projection.weight"
    src, weight = _bundle(tmp_path / "src", weight_name=weight_name)

    dst, effect = wq.prequantized_bundle(src, cache_root=tmp_path / "cache")

    tensors = wq.read_tensors(dst / "weights.safetensors")
    assert tensors[weight_name].dtype == np.int8
    assert np.array_equal(
        tensors[f"{weight_name}.__merlin_int8_scale"],
        np.max(np.abs(weight), axis=0) / np.float32(127.0),
    )
    assert effect["weight_names"] == [weight_name]


def test_prequantized_bundle_follows_a_single_use_layout_chain_to_a_matmul(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    weight_name = "net.encoder.conv.weight"
    weight = np.arange(-12, 12, dtype=np.float32).reshape(2, 3, 2, 2)
    (src / "model.mlir").write_text(
        '''builtin.module attributes {prov.weights_file = "SRC/weights.safetensors", '''
        '''prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<2x3x2x2xf32>, %1: tensor<12x4xf32>) -> tensor<2x4xf32> {
    %2 = tensor.collapse_shape %0 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<2x3x2x2xf32> into tensor<24xf32>
    %3 = tensor.expand_shape %2 [[0 : i64, 1 : i64]] output_shape [2, 12] : tensor<24xf32> into tensor<2x12xf32>
    %4 = tensor.empty() : tensor<2x4xf32>
    %5 = linalg.matmul ins(%3, %1 : tensor<2x12xf32>, tensor<12x4xf32>) outs(%4 : tensor<2x4xf32>) -> tensor<2x4xf32>
    func.return %5 : tensor<2x4xf32>
  }
}
'''.replace("SRC", str(src)))
    (src / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "param", "weight": weight_name,
              "dtype": "float32", "shape": [2, 3, 2, 2]},
        "1": {"kind": "input", "shape": [12, 4]},
    }))
    _write_safetensors(src / "weights.safetensors", {weight_name: weight})

    dst, effect = wq.prequantized_bundle(src, cache_root=tmp_path / "cache")

    tensors = wq.read_tensors(dst / "weights.safetensors")
    scale = np.max(np.abs(weight), axis=(1, 2, 3)) / np.float32(127.0)
    assert np.array_equal(tensors[f"{weight_name}.__merlin_int8_scale"], scale)
    assert effect["weights_prequantized"] == 1
    text = (dst / "model.mlir").read_text()
    assert "tensor<2x3x2x2xi8> into tensor<24xi8>" in text
    assert "tensor<24xi8> into tensor<2x12xi8>" in text
    assert "axis = 0 : i64" in text
    assert "linalg.matmul ins(%merlin_wq_deq0, %1" in text


def test_prequantization_refuses_a_reshape_that_merges_the_weight_scale_axis(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    weight_name = "net.encoder.conv.weight"
    weight = np.arange(-12, 12, dtype=np.float32).reshape(2, 3, 2, 2)
    (src / "model.mlir").write_text('''builtin.module attributes {prov.weights_file = "SRC/weights.safetensors"} {
  func.func @forward(%0: tensor<2x3x2x2xf32>, %1: tensor<4x5xf32>) -> tensor<6x5xf32> {
    %2 = tensor.collapse_shape %0 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] : tensor<2x3x2x2xf32> into tensor<24xf32>
    %3 = tensor.expand_shape %2 [[0 : i64, 1 : i64]] output_shape [6, 4] : tensor<24xf32> into tensor<6x4xf32>
    %4 = tensor.empty() : tensor<6x5xf32>
    %5 = linalg.matmul ins(%3, %1 : tensor<6x4xf32>, tensor<4x5xf32>) outs(%4 : tensor<6x5xf32>) -> tensor<6x5xf32>
    func.return %5 : tensor<6x5xf32>
  }
}
'''.replace("SRC", str(src)))
    (src / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "param", "weight": weight_name,
              "dtype": "float32", "shape": [2, 3, 2, 2]},
        "1": {"kind": "input", "shape": [4, 5]},
    }))
    _write_safetensors(src / "weights.safetensors", {weight_name: weight})

    with pytest.raises(wq.PrequantizeRefused, match="scale axis"):
        wq.prequantized_bundle(src, cache_root=tmp_path / "cache")


def test_prequantized_bundle_stores_the_current_w8a8_formula_and_static_dequant(tmp_path):
    src, weight = _bundle(tmp_path / "src")
    dst, effect = wq.prequantized_bundle(src, cache_root=tmp_path / "cache")

    tensors = wq.read_tensors(dst / "weights.safetensors")
    scale = np.max(np.abs(weight), axis=0) / np.float32(127.0)
    quant = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
    assert np.array_equal(tensors["net.lstm.weight_ih_l0"], quant)
    assert np.array_equal(tensors["net.lstm.weight_ih_l0.__merlin_int8_scale"], scale)
    assert np.array_equal(tensors["other"], np.array([9.0, 10.0], dtype=np.float32))

    manifest = json.loads((dst / "weights.safetensors.manifest.json").read_text())
    assert manifest["0"]["dtype"] == "int8"
    assert manifest["2"]["shape"] == [3]
    assert manifest["2"]["weight"].endswith(".__merlin_int8_scale")
    text = (dst / "model.mlir").read_text()
    assert "%0: tensor<2x3xi8>" in text
    assert "%merlin_wq_scale0: tensor<3xf32>" in text
    assert '"quant_ext.dequantize_per_channel"' in text
    assert "linalg.matmul ins(%1, %merlin_wq_deq0" in text
    assert effect["weights_prequantized"] == 1
    assert (dst / "session_contract.yaml").read_text() == "version: 1\n"


def test_prequantized_bundle_is_cached_and_records_its_own_payload(tmp_path):
    src, _ = _bundle(tmp_path / "src")
    (src / "model.prepared.mlir").write_text("// stale\n")
    cache = tmp_path / "cache"
    first, first_effect = wq.prequantized_bundle(src, cache_root=cache)
    second, second_effect = wq.prequantized_bundle(src, cache_root=cache)

    assert first == second
    assert first_effect["cached"] is False and second_effect["cached"] is True
    assert wq.cache_key(src) in first.name
    assert not (first / "model.prepared.mlir").exists()
    text = (first / "model.mlir").read_text()
    assert f'prov.weights_file = "{first / "weights.safetensors"}"' in text
    records = json.loads((first / "bundle.rewrites.json").read_text())["rewrites"]
    assert records[-1]["name"] == wq.FEATURE
    assert records[-1]["effect"]["weight_args"] == [0]


def test_prequantization_refuses_a_weight_with_an_additional_reader(tmp_path):
    src, _ = _bundle(tmp_path / "src")
    text = (src / "model.mlir").read_text()
    text = text.replace(
        "    func.return %5 : tensor<1x3xf32>",
        "    %6 = tensor.empty() : tensor<2x3xf32>\n"
        "    %7 = linalg.copy ins(%0 : tensor<2x3xf32>) outs(%6 : tensor<2x3xf32>) "
        "-> tensor<2x3xf32>\n"
        "    func.return %5 : tensor<1x3xf32>")
    (src / "model.mlir").write_text(text)

    assert wq.plan(src).problems
    with pytest.raises(wq.PrequantizeRefused, match="2 readers"):
        wq.prequantized_bundle(src, cache_root=tmp_path / "cache")


def test_prequantization_refuses_a_zero_scale_channel(tmp_path):
    src, _ = _bundle(tmp_path / "src")
    tensors = wq.read_tensors(src / "weights.safetensors")
    tensors["net.lstm.weight_ih_l0"][:, 2] = 0
    _write_safetensors(src / "weights.safetensors", tensors)

    with pytest.raises(wq.PrequantizeRefused, match="zero output-channel scale"):
        wq.prequantized_bundle(src, cache_root=tmp_path / "cache")


@pytest.mark.skipif(not (REAL_BUNDLE / "model.mlir").is_file(), reason="LSTMNetVIT bundle absent")
def test_real_lstmnetvit_plan_selects_all_seventeen_dynamic_f32_weight_chains():
    candidate = wq.plan(REAL_BUNDLE)
    assert candidate.problems == ()
    assert [item.arg for item in candidate.weights] == [
        0, 4, 20, 40, 50, 60, 64, 80, 100, 110, 124, 125, 128, 129, 132, 133, 140,
    ]


@pytest.mark.skipif(not (REAL_RESNET_BUNDLE / "model.mlir").is_file(),
                    reason="ResNet-50 W8A8 bundle absent")
def test_real_resnet_plan_selects_all_fifty_three_convolution_weights():
    """The full-flow ResNet path must not quantize its entire convolution weight set per run."""
    candidate = wq.plan(REAL_RESNET_BUNDLE)
    assert candidate.problems == ()
    assert len(candidate.weights) == 53
    assert candidate.weights[0].weight == "model.conv1.weight"
    assert candidate.weights[-1].weight == "model.layer4.2.conv3.weight"


def test_feature_registration_has_no_compiler_hook():
    from merlin.llvmlower.impr_features import get, normalize

    assert wq.ensure_registered() == wq.FEATURE
    assert normalize([wq.FEATURE]) == frozenset({wq.FEATURE})
    feature = get(wq.FEATURE)
    assert (feature.edit_pipeline, feature.edit_schedule, feature.edit_cflags) == (None, None, None)
    assert feature.schedule_replace is False


def test_integer_preparation_reaches_the_existing_static_weight_fast_path(tmp_path):
    from merlin.runtime.backends.zephyr_model import prepare_for_lowering

    src, _ = _bundle(tmp_path / "src")
    dst, _ = wq.prequantized_bundle(src, cache_root=tmp_path / "cache")
    work = tmp_path / "prepare"
    work.mkdir()
    prepared, _ = prepare_for_lowering(dst / "model.mlir", work, int8_compute=True,
                                       features=frozenset(), blocking=False)
    text = prepared.read_text()
    # Only the runtime activation is dynamically quantized. The weight's stored i8 and scale feed
    # the integer matmul directly; no second abs-max/round-even chain survives.
    assert text.count("math.roundeven") == 1
    assert "tensor<2x3xi8>" in text
    assert '"quant_ext.dequantize_per_channel"' not in text
    assert "arith.muli" in text
    assert "tensor<1x2xi8>, tensor<2x3xi8>" in text
