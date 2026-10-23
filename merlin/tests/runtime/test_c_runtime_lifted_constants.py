"""Bind lifted tensors by their captured names before generating runtime data."""

import json
import struct

import numpy as np
import pytest

from merlin.llvmlower import c_runtime


def capture(tmp_path, *, constants, values, inputs=None, input_order=True):
    root = tmp_path / "capture"
    root.mkdir()
    scalar_args = ", ".join(f"%c{i}: tensor<f32>" for i in range(len(constants)))
    signature = (scalar_args + ", " if scalar_args else "") + "%x: tensor<2xf32>"
    source = f"""builtin.module {{
      func.func @forward({signature}) -> tensor<2xf32> {{
        func.return %x : tensor<2xf32>
      }}
    }}"""
    (root / "model.mlir").write_text(source)
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", 2) + b"{}")
    manifest = {str(i): {"kind": "input", "name": name} for i, name in enumerate(constants)}
    manifest[str(len(constants))] = {"kind": "input", "name": "x"}
    (root / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    np.savez(root / "inputs.npz", in0=np.array([7, 9], np.float32) if inputs is None else inputs)
    np.savez(root / "extra.npz", **values)
    if input_order:
        (root / "input_order.json").write_text(json.dumps({"x": 0}))
    return root


@pytest.mark.parametrize("name", ["c_lifted_tensor_0", "c_lm_model_lifted_tensor_0"])
def test_lifted_constant_uses_its_exact_name(tmp_path, name):
    root = capture(tmp_path, constants=[name], values={name: np.float32(2)})
    out = tmp_path / "generated"
    c_runtime.generate(root, out, root / "inputs.npz")
    assert "static float merlin_in_0[] = {2.0};" in (out / "model_io.h").read_text()


def test_constant_binding_does_not_depend_on_lexical_order(tmp_path):
    names = ["c_lifted_tensor_2", "c_lifted_tensor_10"]
    root = capture(tmp_path, constants=names, values={names[0]: np.float32(2), names[1]: np.float32(10)})
    out = tmp_path / "generated"
    c_runtime.generate(root, out, root / "inputs.npz")
    text = (out / "model_io.h").read_text()
    assert "static float merlin_in_0[] = {2.0};" in text
    assert "static float merlin_in_1[] = {10.0};" in text


def test_missing_module_constant_cannot_consume_the_token_input(tmp_path):
    root = capture(tmp_path, constants=["c_lm_model_lifted_tensor_28"], values={})
    with pytest.raises(ValueError, match="lifted|unbound|materialized"):
        c_runtime.generate(root, tmp_path / "generated", root / "inputs.npz")


def test_runtime_array_cardinality_must_match_the_signature(tmp_path):
    root = capture(tmp_path, constants=[], values={}, inputs=np.array([1, 2, 3], np.float32))
    with pytest.raises(ValueError, match="elements|shape|extent"):
        c_runtime.generate(root, tmp_path / "generated", root / "inputs.npz")


def test_authoritative_input_order_cannot_silently_fall_back(tmp_path):
    root = capture(tmp_path, constants=[], values={})
    (root / "input_order.json").write_text(json.dumps({"different_name": 0}))
    with pytest.raises(ValueError, match="input|unbound"):
        c_runtime.generate(root, tmp_path / "generated", root / "inputs.npz")
