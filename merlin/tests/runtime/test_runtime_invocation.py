"""Native scalar transport and runtime shapes preserve the actual program ABI."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower.c_runtime import generate
from merlin.llvmlower.runtime_invocation import resolve
from merlin.llvmlower.abi import ScalarArg
from merlin.runtime.input_observation import revalidate_observation, verify_observed_inputs
from merlin.runtime.out_bin import parse_binary_console_details
from test_whole_model_readback import _frame


@pytest.mark.parametrize("dtype,bits", [("f32", [0x80000000, 0x7f800001, 0xffc12345]),
    ("f64", [0x8000000000000000, 0x7ff0000000000001, 0xfff8123456789abc])])
def test_native_scalar_words_do_not_round_through_python_floats(dtype, bits):
    import ctypes
    for word in bits:
        raw = word.to_bytes(4 if dtype == "f32" else 8, "little")
        argument = ScalarArg.from_storage(raw, dtype).to_ctype()
        assert ctypes.string_at(ctypes.byref(argument), ctypes.sizeof(argument)) == raw


@pytest.mark.parametrize("storage,dtype", [(b"\x02", "i1"), (b"\x00", "i64"), (b"\x00\x00", "f16"), (4, "i32")])
def test_raw_scalar_form_refuses_incomplete_or_unqualified_ABI(storage, dtype):
    with pytest.raises(ValueError): ScalarArg.from_storage(storage, dtype)


def example(root, n=3):
    root.mkdir()
    (root / "model.mlir").write_text('''module {func.func @forward(%x:tensor<?xi64>, %n:i32,
      %alpha:f32, %choose:i1)->tensor<?xi64> {return %x:tensor<?xi64>}}''')
    contract = {"schema": "merlin.compile-signature.v2", "entry": "forward", "inputs": [
        {"dtype": "i64", "shape": ["n"]}, {"dtype": "i32", "shape": [], "kind": "scalar", "scalar": "n", "bounds": [0, 32]},
        {"dtype": "f32", "shape": [], "kind": "scalar", "scalar": "alpha"},
        {"dtype": "i1", "shape": [], "kind": "scalar", "scalar": "choose"}], "outputs": [{"dtype": "i64", "shape": ["n"]}]}
    (root / "signature.json").write_text(json.dumps(contract))
    (root / "weights.safetensors.manifest.json").write_text(json.dumps({str(i): {"kind": "input", "name": f"in{i}"} for i in range(4)}))
    np.savez(root / "inputs.npz", in0=np.arange(n, dtype=np.int64), in1=np.asarray(n, np.int32),
        in2=np.asarray(-0.0, np.float32), in3=np.asarray(True))
    return root


@pytest.mark.parametrize("n", (0, 3, 32))
def test_runtime_plan_checks_original_types_and_keeps_scalar_words(tmp_path, n):
    root = example(tmp_path / "source", n)
    original = (root / "model.mlir").read_bytes()
    plan = resolve(root / "model.mlir", root / "signature.json", root / "inputs.npz")
    assert plan["inputs"][0]["shape"] == [n] and plan["outputs"][0]["shape"] == [n]
    assert [entry["kind"] for entry in plan["inputs"]] == ["tensor", "scalar", "scalar", "scalar"]
    assert plan["scalars"][1]["word"] == np.asarray(-0.0, np.float32).tobytes().hex()
    assert plan["mlir_argument_specialization"] == "not-performed" and (root / "model.mlir").read_bytes() == original


def test_static_signature_v1_keeps_ordinal_native_scalar_fields(tmp_path):
    root = example(tmp_path / "source")
    model = (root / "model.mlir").read_text().replace("tensor<?xi64>", "tensor<3xi64>")
    (root / "model.mlir").write_text(model)
    contract = json.loads((root / "signature.json").read_text())
    contract["schema"] = "merlin.compile-signature.v1"
    contract["inputs"][0]["shape"] = [3]
    contract["outputs"][0]["shape"] = [3]
    for scalar in contract["inputs"][1:]: scalar.pop("scalar")
    (root / "signature.json").write_text(json.dumps(contract))
    plan = resolve(root / "model.mlir", root / "signature.json", root / "inputs.npz")
    assert all(word["name"] is None for word in plan["scalars"])


@pytest.mark.parametrize("tensor", ('tensor<?xi64, "opaque">', 'tensor<1x1x1x1x1x1x1x1x1xi64>'))
def test_runtime_transport_refuses_unimplemented_layouts_and_rank(tmp_path, tensor):
    root = example(tmp_path / "source")
    (root / "model.mlir").write_text((root / "model.mlir").read_text().replace("tensor<?xi64>", tensor))
    with pytest.raises(ValueError, match="unencoded tensors of rank at most eight"):
        resolve(root / "model.mlir", root / "signature.json", root / "inputs.npz")


@pytest.mark.parametrize("damage", ("count", "dtype", "scalar-kind", "scalar-shape", "bounds", "name", "shape-binding", "output", "archive", "array-type", "array-shape"))
def test_runtime_claims_cannot_replace_source_types_or_invocation_dimensions(tmp_path, damage):
    root = example(tmp_path / "source")
    contract = json.loads((root / "signature.json").read_text())
    if damage == "count": contract["inputs"].pop()
    elif damage == "dtype": contract["inputs"][1]["dtype"] = "i64"
    elif damage == "scalar-kind": contract["inputs"][1].pop("kind")
    elif damage == "scalar-shape": contract["inputs"][1]["shape"] = [1]
    elif damage == "bounds": contract["inputs"][1]["bounds"] = [0, 2]
    elif damage == "name": contract["inputs"][2]["scalar"] = "n"
    elif damage == "shape-binding": contract["inputs"][0]["shape"] = [3]
    elif damage == "output": contract["outputs"][0]["shape"] = ["alpha"]
    else:
        with np.load(root / "inputs.npz", allow_pickle=False) as data: arrays = {k: data[k].copy() for k in data.files}
        if damage == "archive": arrays["golden"] = np.arange(3)
        elif damage == "array-type": arrays["in1"] = np.asarray(3, np.int64)
        else: arrays["in0"] = np.arange(4, dtype=np.int64)
        np.savez(root / "inputs.npz", **arrays)
    (root / "signature.json").write_text(json.dumps(contract))
    with pytest.raises(ValueError): resolve(root / "model.mlir", root / "signature.json", root / "inputs.npz")


@pytest.mark.parametrize("n", (0, 3))
def test_c_codegen_retains_native_scalars_and_complete_empty_observations(tmp_path, n):
    root, out = example(tmp_path / "source", n), tmp_path / "generated"
    info = generate(root, out, root / "inputs.npz", runtime_signature=root / "signature.json", readonly_input_indices=(0, 1, 2, 3))
    assert info["outputs"] == [([n], "i64")]
    call = (out / "model_call.c").read_text()
    assert "void*,int32_t,float,_Bool,void*" in call and "memcpy(&scalar_2" in call
    assert (out / "runtime-abi.json").is_file()
    manifest = revalidate_observation(info["readonly_input_observation"])
    raw = Path(manifest["expected"]["path"]).read_bytes()
    console = b""
    for arg in manifest["arguments"]:
        payload = raw[arg["expected_offset"]:arg["expected_offset"] + arg["bytes"]]
        console += _frame(arg["frame"], np.frombuffer(payload, np.uint8).reshape(1, -1), 1, False)
    outputs, _, frames = parse_binary_console_details(console + b"DONE\n")
    remaining, _, observed = verify_observed_inputs(info["readonly_input_observation"], outputs, frames)
    assert not remaining and len(observed["arguments"]) == 4 and observed["arguments"][0]["bytes"] == n * 8
    if n == 0:
        lost = b"".join(_frame(arg["frame"], np.frombuffer(raw[arg["expected_offset"]:arg["expected_offset"] + arg["bytes"]], np.uint8).reshape(1, -1), 1, False)
            for arg in manifest["arguments"][1:]) + b"DONE\n"
        values, _, missing = parse_binary_console_details(lost)
        with pytest.raises(ValueError, match="exact complete byte frame"):
            verify_observed_inputs(info["readonly_input_observation"], values, missing)


def test_generated_native_scalar_bridge_uses_actual_C_argument_types(tmp_path):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not cc: pytest.skip("existing C compiler required")
    root, out = example(tmp_path / "source"), tmp_path / "generated"
    generate(root, out, root / "inputs.npz", runtime_signature=root / "signature.json")
    source = tmp_path / "main.c"
    source.write_text('''#include "merlin_model.h"
      #include <stdint.h>
      #include <string.h>
      static int calls;
      void _mlir_ciface_forward(void *x,int32_t n,float a,_Bool choose,void *out) {
        uint32_t bits;memcpy(&bits,&a,4);
        calls = n == -123 && bits == 0x80000000u && choose && x == out;
      }
      int main(void) {int32_t n=-123;float a=-0.0f;_Bool choose=1;
        merlin_descriptor_t nd={.aligned=&n},ad={.aligned=&a},bd={.aligned=&choose};
        void *args[]={0,&nd,&ad,&bd,0};merlin_invoke(args);return !calls;}''')
    built = subprocess.run([cc, "-std=c11", "-O2", "-I", str(runtime_dir() / "c"), str(out / "model_call.c"), str(source),
        "-o", str(tmp_path / "bridge")], capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    assert subprocess.run([str(tmp_path / "bridge")], timeout=10).returncode == 0
