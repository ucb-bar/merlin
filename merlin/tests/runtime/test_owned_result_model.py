"""The ordinary model builder executes production returned-result transport."""

import io
import json
import subprocess
import zipfile

import numpy as np
import pytest
from owned_result_fixture import COMPACT

from merlin.common.digest import sha256_file
from merlin.frontends.execution_result_signature import MODE, select_signature
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import toolchain
from merlin.llvmlower.owned_result_driver import load_arguments
from merlin.runtime.backends import spike, spike_model
from merlin.runtime.backends.owned_result_model import build_if_selected
from merlin.runtime.owned_result_readback import PROFILE, decode


def signature(source=COMPACT):
    return select_signature(parse_mlir_text(source), {
        "entry": "forward", "inputs": [{"dtype": "i32", "shape": ["n"]}],
        "dimension_bindings": {"n": {"argument": 0, "axis": 0}},
    }, MODE)


def setup(root, array):
    model = root / "model"
    model.mkdir()
    (model / "model.mlir").write_text(COMPACT)
    contract = signature()
    (root / "signature.json").write_text(json.dumps(contract))
    np.savez(root / "inputs.npz", in0=array)
    return model, contract


@pytest.mark.parametrize("words", [[], [0, -1, 4, 0, 9, -3], [-(1 << 31), (1 << 31) - 1, 10, -10]])
def test_production_model_builder_emits_observed_result_shapes_on_actual_image(tmp_path, words):
    if not toolchain.available() or not spike.available():
        pytest.skip("existing native/cross/Spike tools required")
    x = np.asarray(words, np.int32)
    model, contract = setup(tmp_path, x)
    report = spike_model.build(model, tmp_path / "image", inputs_npz=tmp_path / "inputs.npz",
        runtime_signature=tmp_path / "signature.json", backend="scalar", arena_mb=4,
        dram_bytes=64 << 20, code_reserve=2 << 20, full_readback=True, readonly_input_indices=(0,))
    assert report["readback_profile"] == PROFILE and report["allocation_context"] == "serial-nonconcurrent"
    assert spike_model.declared_harts(report["elf"]) == 1
    process = spike_model.run_process(report["elf"], isa=spike_model.declared_isa(report["elf"]), timeout=30)
    (tmp_path / "stdout.bin").write_bytes(process.stdout)
    (tmp_path / "stderr.bin").write_bytes(process.stderr)
    assert process.returncode == 0, (process.returncode, process.stdout, process.stderr)
    outputs, observed = decode(COMPACT, contract, [x], process.stdout)
    for actual, wanted in zip(outputs, (x[x > 0], x[x > 0], x), strict=True):
        assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
    recipe = json.loads((tmp_path / "image/compilation_recipe.json").read_text())
    assert recipe["status"] == "completed" and recipe["executable"]["sha256"] == sha256_file(report["elf"])
    assert observed["functional_inputs_preserved"]
    (tmp_path / "observation.json").write_text(json.dumps({
        "returncode": process.returncode, "readback": observed, "image": sha256_file(report["elf"]),
        "stdout": sha256_file(tmp_path / "stdout.bin"), "source": sha256_file(model / "model.mlir"),
        "compiler": report["index_lowering"], "isa": spike_model.declared_isa(report["elf"]),
        "memory": spike_model.declared_memory(report["elf"]),
    }, sort_keys=True, indent=2) + "\n")


@pytest.mark.parametrize("option,value", [
    ("full_readback", False), ("readonly_input_indices", ()), ("op_profile", True),
    ("host_math_policy", "expf_via_double"), ("device", object()),
])
def test_selected_unimplemented_options_refuse_before_building(tmp_path, option, value):
    model, _contract = setup(tmp_path, np.arange(3, dtype=np.int32))
    options = {"full_readback": True, "readonly_input_indices": (0,), option: value}
    with pytest.raises(ValueError, match="execution-owned target build"):
        spike_model.build(model, tmp_path / "image", inputs_npz=tmp_path / "inputs.npz",
                          runtime_signature=tmp_path / "signature.json", **options)
    assert not (tmp_path / "image").exists()


def test_legacy_runtime_signatures_keep_their_ordinary_builder(tmp_path):
    path = tmp_path / "signature.json"
    path.write_text(json.dumps({"schema": "merlin.compile-signature.v5"}))
    assert build_if_selected({"runtime_signature": path}) is None


@pytest.mark.parametrize("damage", ["roster", "duplicate", "huge-header", "truncated"])
def test_runtime_archives_are_bounded_before_loading_payloads(tmp_path, damage):
    _model, contract = setup(tmp_path, np.arange(3, dtype=np.int32))
    path = tmp_path / "bad.npz"
    payload = io.BytesIO()
    if damage == "huge-header":
        np.lib.format.write_array_header_1_0(payload, {"descr": "<i4", "fortran_order": False, "shape": (1 << 62,)})
    else:
        np.save(payload, np.arange(3, dtype=np.int32))
    raw = payload.getvalue()
    if damage == "truncated":
        raw = raw[:-1]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("in1.npy" if damage == "roster" else "in0.npy", raw)
        if damage == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("in0.npy", raw)
    with pytest.raises(ValueError, match="owned driver input archive"):
        load_arguments(COMPACT, contract, path)


def test_baremetal_main_requires_builder_supplied_input_placement(tmp_path):
    from merlin.common.paths import runtime_dir
    runtime, abi = runtime_dir() / "c", runtime_dir() / "abi"
    harness = spike.harness_dir()
    done = subprocess.run([str(toolchain.clang()), "-fsyntax-only", "-I", str(runtime), "-I", str(abi),
                           "-I", str(harness), str(harness / "owned_result_main.c")],
                          capture_output=True, text=True, check=False, timeout=10)
    assert done.returncode != 0 and "explicit input blob placement" in done.stderr


@pytest.mark.parametrize("exhausted", [False, True])
def test_zero_input_target_program_and_actual_arena_exhaustion(tmp_path, exhausted):
    if not toolchain.available() or not spike.available():
        pytest.skip("existing native/cross/Spike tools required")
    size = 262144 if exhausted else 2
    source = f'''module {{func.func @forward()->tensor<{size}xi64> {{
      %empty=tensor.empty():tensor<{size}xi64>
      %nine=arith.constant 9:i64
      %result=linalg.fill ins(%nine:i64) outs(%empty:tensor<{size}xi64>)->tensor<{size}xi64>
      return %result:tensor<{size}xi64>
    }}}}'''
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.mlir").write_text(source)
    contract = select_signature(parse_mlir_text(source), {"entry": "forward", "inputs": []}, MODE)
    (tmp_path / "signature.json").write_text(json.dumps(contract))
    np.savez(tmp_path / "inputs.npz")
    report = spike_model.build(model, tmp_path / "image", inputs_npz=tmp_path / "inputs.npz",
        runtime_signature=tmp_path / "signature.json", backend="scalar", arena_mb=1,
        dram_bytes=64 << 20, code_reserve=2 << 20, full_readback=True)
    process = spike_model.run_process(report["elf"], isa=spike_model.declared_isa(report["elf"]), timeout=30)
    (tmp_path / "stdout.bin").write_bytes(process.stdout)
    (tmp_path / "stderr.bin").write_bytes(process.stderr)
    (tmp_path / "outcome.json").write_text(json.dumps({"returncode": process.returncode,
        "exhausted": exhausted, "elf": sha256_file(report["elf"]),
        "stdout": sha256_file(tmp_path / "stdout.bin")}, sort_keys=True) + "\n")
    if exhausted:
        # Spike's encoded guest status can truncate to zero at the host exit
        # boundary. Complete DONE/frames remain mandatory for qualification.
        assert b"FATAL: model arena" in process.stdout and b"*** FAILED ***" in process.stderr
        assert b"OUT_BIN_BEGIN" not in process.stdout and b"DONE" not in process.stdout
        with pytest.raises(ValueError, match="DONE"):
            decode(source, contract, [], process.stdout)
    else:
        assert process.returncode == 0, process.stderr
        (actual,), observation = decode(source, contract, [], process.stdout)
        assert actual.tobytes() == np.full(2, 9, np.int64).tobytes()
        assert observation["functional_inputs_preserved"]
