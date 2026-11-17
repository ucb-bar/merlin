"""Changing loop generations execute through saved, C and target interfaces."""

import json
import subprocess

import numpy as np
import pytest
from owned_result_fixture import MULTI_RESIZE, RESIZE

from merlin.common.digest import sha256_file
from merlin.common.paths import runtime_dir
from merlin.frontends.execution_result_signature import MODE, select_signature
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import toolchain
from merlin.llvmlower.abi import ScalarArg
from merlin.llvmlower.owned_result_driver import generate
from merlin.llvmlower.owned_result_program import load, save
from merlin.llvmlower.owned_results import compile_owned_results
from merlin.runtime.backends import spike, spike_model
from merlin.runtime.owned_result_readback import decode

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")


def signature():
    return select_signature(parse_mlir_text(RESIZE), {"entry": "forward", "inputs": [
        {"dtype": "i32", "shape": ["n"]}, {"kind": "scalar", "dtype": "i32", "shape": []}],
        "dimension_bindings": {"n": {"argument": 0, "axis": 0}}}, MODE)


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    root = tmp_path_factory.mktemp("resizing-owned-results")
    value = compile_owned_results(RESIZE, workdir=root / "native", timeout=30,
                                 features=frozenset({"reuse_loop_tensor_destinations"}))
    saved = save(value)
    return root, value, saved


def test_saved_resizing_program_replays_independent_shapes_without_compilation(program, monkeypatch):
    root, original, saved = program
    import merlin.llvmlower.pipeline
    monkeypatch.setattr(merlin.llvmlower.pipeline, "lower_to_llvm_ir",
                        lambda *a, **k: pytest.fail("compiled during replay"))
    restored = load(saved)
    record = json.loads(saved.read_text())
    assert {"dealloc_lifetime.py", "dealloc_promotions.py"} <= set(record["implementations"])
    observations = []
    for capacity in (0, 1, 7):
        x = np.arange(capacity, dtype=np.int32) * -13
        before = x.tobytes()
        for steps in (0, 1, 7, 19):
            (actual,), observation = restored.execute((x, ScalarArg(steps, "i32")))
            wanted = x[:max(0, capacity-steps)]
            assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
            assert x.tobytes() == before and observation["source_calls"] == 1
            observations.append({"capacity": capacity, "steps": steps, **observation})
    assert sha256_file(original.library) == sha256_file(restored.library)
    (root / "replays.json").write_text(json.dumps(observations, sort_keys=True, indent=2) + "\n")


def test_production_host_driver_preserves_changed_trip_counts(program):
    root, value, _saved = program
    x = np.array([7, -1, -(1 << 31), (1 << 31)-1, 9, 0, -7], np.int32)
    contract = signature()
    record = generate(RESIZE, contract, [x, np.asarray(0, np.int32)], root / "driver")
    runtime, abi = runtime_dir() / "c", runtime_dir() / "abi"
    executable = root / "model"
    done = subprocess.run([str(toolchain.clang()), "-std=c11", "-Wall", "-Wextra", "-O2",
        "-I", str(runtime), "-I", str(abi), str(root / "driver/model_owned.c"), str(value.library.parent / "model.ll"),
        *(str(runtime / name) for name in ("owned_result_driver.c", "owned_result_readback.c",
                                         "owned_result_host_main.c")),
        *(str(abi / name) for name in ("owned_results.c", "owned_result_views.c", "mlir_runtime.c")),
        "-lm", "-o", str(executable)], capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 0, done.stderr
    for steps in (0, 1, 2, 19):
        blob = bytearray((root / "driver/invocation.bin").read_bytes())
        offset = record["inputs"][1]["offset"]
        blob[offset:offset+4] = np.asarray(steps, np.int32).tobytes()
        path = root / f"arguments-{steps}.bin"
        path.write_bytes(blob)
        done = subprocess.run([str(executable), str(path)], capture_output=True, timeout=10, check=False)
        assert done.returncode == 0, (done.stdout, done.stderr)
        (actual,), observed = decode(RESIZE, contract, [x, np.asarray(steps, np.int32)], done.stdout)
        wanted = x[:max(0, x.size-steps)]
        assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
        assert observed["functional_inputs_preserved"]
        (root / f"stdout-{steps}.bin").write_bytes(done.stdout)


def test_independent_generations_preserve_types_sizes_and_nonfinite_words(tmp_path):
    program = compile_owned_results(MULTI_RESIZE, workdir=tmp_path / "native", timeout=30)
    for nx, ny in ((0, 0), (7, 5), (1, 9)):
        x = np.arange(nx, dtype=np.int32) * -13
        y = np.resize(np.asarray([0x7FF0000000000123, 0xFFF8000000000456, 0x8000000000000000,
                                  0x3FF0000000000000, 1], np.uint64), ny).view(np.float64)
        before = x.tobytes(), y.tobytes()
        for steps in (0, 1, 2, 9):
            outputs, report = program.execute((x, y, ScalarArg(steps, "i32")))
            expected = (x[:max(0, nx-steps)], y[:max(0, ny-2*steps)])
            for actual, wanted in zip(outputs, expected, strict=True):
                assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
            assert (x.tobytes(), y.tobytes()) == before and report["source_calls"] == 1


@pytest.mark.parametrize("capacity,steps", [(0, 0), (7, 0), (7, 2), (7, 19), (7, -1)])
def test_runtime_resizing_reaches_actual_target_images_with_complete_readback(tmp_path, capacity, steps):
    if not spike.available():
        pytest.skip("existing cross/Spike tools required")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.mlir").write_text(RESIZE)
    contract = signature()
    (tmp_path / "signature.json").write_text(json.dumps(contract))
    x = np.arange(capacity, dtype=np.int32) * -13
    arrays = [x, np.asarray(steps, np.int32)]
    np.savez(tmp_path / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)})
    report = spike_model.build(model, tmp_path / "image", inputs_npz=tmp_path / "inputs.npz",
        runtime_signature=tmp_path / "signature.json", backend="scalar", arena_mb=4,
        features=frozenset({"reuse_loop_tensor_destinations"}), dram_bytes=64 << 20,
        code_reserve=2 << 20, full_readback=True, readonly_input_indices=(0, 1))
    process = spike_model.run_process(report["elf"], isa=spike_model.declared_isa(report["elf"]), timeout=30)
    (tmp_path / "stdout.bin").write_bytes(process.stdout)
    (tmp_path / "stderr.bin").write_bytes(process.stderr)
    observations = {"returncode": process.returncode, "capacity": capacity, "steps": steps,
        "elf": sha256_file(report["elf"]), "stdout": sha256_file(tmp_path / "stdout.bin"),
        "source": sha256_file(model / "model.mlir"), "compiler": report["index_lowering"],
        "isa": spike_model.declared_isa(report["elf"]), "memory": spike_model.declared_memory(report["elf"])}
    if steps < 0:
        assert process.returncode != 0 and b"source guard" in process.stdout
        assert b"OUT_BIN_BEGIN" not in process.stdout and b"DONE" not in process.stdout
    else:
        assert process.returncode == 0, (process.stdout, process.stderr)
        (actual,), readback = decode(RESIZE, contract, arrays, process.stdout)
        wanted = x[:max(0, capacity-steps)]
        assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
        observations["readback"] = readback
    (tmp_path / "observation.json").write_text(json.dumps(observations, sort_keys=True, indent=2) + "\n")


@pytest.mark.parametrize("steps", [0, 2])
def test_multiple_resizing_target_results_keep_f64_storage_words(tmp_path, steps):
    if not spike.available():
        pytest.skip("existing cross/Spike tools required")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.mlir").write_text(MULTI_RESIZE)
    contract = select_signature(parse_mlir_text(MULTI_RESIZE), {"entry": "forward", "inputs": [
        {"dtype": "i32", "shape": ["nx"]}, {"dtype": "f64", "shape": ["ny"]},
        {"kind": "scalar", "dtype": "i32", "shape": []}],
        "dimension_bindings": {"nx": {"argument": 0, "axis": 0}, "ny": {"argument": 1, "axis": 0}}}, MODE)
    (tmp_path / "signature.json").write_text(json.dumps(contract))
    x = np.asarray([7, -13, -(1 << 31), (1 << 31)-1], np.int32)
    y = np.asarray([0x7FF0000000000123, 0xFFF8000000000456, 0x8000000000000000,
                    0x3FF0000000000000, 1], np.uint64).view(np.float64)
    arrays = [x, y, np.asarray(steps, np.int32)]
    np.savez(tmp_path / "inputs.npz", **{f"in{i}": value for i, value in enumerate(arrays)})
    report = spike_model.build(model, tmp_path / "image", inputs_npz=tmp_path / "inputs.npz",
        runtime_signature=tmp_path / "signature.json", backend="scalar", arena_mb=4,
        dram_bytes=64 << 20, code_reserve=2 << 20, full_readback=True, readonly_input_indices=(0, 1, 2))
    process = spike_model.run_process(report["elf"], isa=spike_model.declared_isa(report["elf"]), timeout=30)
    (tmp_path / "stdout.bin").write_bytes(process.stdout)
    (tmp_path / "stderr.bin").write_bytes(process.stderr)
    assert process.returncode == 0, (process.stdout, process.stderr)
    outputs, observation = decode(MULTI_RESIZE, contract, arrays, process.stdout)
    for actual, wanted in zip(outputs, (x[:max(0, x.size-steps)], y[:max(0, y.size-2*steps)]), strict=True):
        assert (actual.shape, actual.dtype, actual.tobytes()) == (wanted.shape, wanted.dtype, wanted.tobytes())
    (tmp_path / "observation.json").write_text(json.dumps({"returncode": process.returncode,
        "steps": steps, "readback": observation, "elf": sha256_file(report["elf"]),
        "stdout": sha256_file(tmp_path / "stdout.bin"), "source": sha256_file(model / "model.mlir"),
        "compiler": report["index_lowering"], "isa": spike_model.declared_isa(report["elf"]),
        "memory": spike_model.declared_memory(report["elf"])}, sort_keys=True, indent=2) + "\n")
