"""Complete readonly argument bytes are observed, never inferred from outputs."""

import hashlib
import json
import shutil
import struct
import subprocess

import numpy as np
import pytest
from test_whole_model_readback import _frame

from merlin.common.paths import runtime_dir
from merlin.llvmlower import c_runtime
from merlin.runtime.input_observation import revalidate_observation, verify_observed_inputs
from merlin.runtime.out_bin import parse_binary_console_details
from merlin.runtime.whole_model_readback import ReadbackError, decode_outputs


def bundle(root, *, state=True):
    root.mkdir()
    (root / "model.mlir").write_text("""module {
      func.func @forward(%x:tensor<2xi8>,%w:tensor<3xi16>,%s:tensor<2xi32>)->tensor<2xi32> {
        return %s:tensor<2xi32>
      }}""")
    manifest = {
        "0": {"kind": "input", "name": "x"},
        "1": {"kind": "param", "weight": "w"},
        "2": {"kind": "input", "name": "s"},
    }
    (root / "weights.safetensors.manifest.json").write_text(json.dumps(manifest))
    weight = np.array([17, -300, 8191], dtype="<i2").tobytes()
    header = json.dumps({"w": {"dtype": "I16", "shape": [3], "data_offsets": [0, len(weight)]}}).encode()
    (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + weight)
    (root / "input_order.json").write_text(json.dumps({"x": 0, "s": 1}))
    np.savez(root / "inputs.npz", in0=np.array([-128, 127], np.int8), in1=np.array([-1, 23], np.int32))
    if state:
        (root / "session_contract.yaml").write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "recurrent_frames",
                    "paper_ready": False,
                    "stages": ["step"],
                    "steps": 1,
                    "states": [{"name": "s", "input_arg": 2, "output_index": 0}],
                    "streams": [],
                    "quality": {},
                }
            )
        )
    return root


def generated(tmp_path, **kwargs):
    capture = bundle(tmp_path / "capture")
    output = tmp_path / "generated"
    info = c_runtime.generate(capture, output, capture / "inputs.npz", readonly_input_indices=(0, 1, 2), **kwargs)
    return capture, output, info


def console(record):
    manifest = revalidate_observation(record)
    payload = open(manifest["expected"]["path"], "rb").read()
    data = b""
    for arg in manifest["arguments"]:
        start, count = arg["expected_offset"], arg["bytes"]
        values = np.frombuffer(payload[start : start + count], np.uint8).reshape(1, -1)
        data += _frame(arg["frame"], values, 1, False)
    return data + b"DONE\n"


def test_codegen_records_original_input_and_weight_storage_without_inventing_outputs(tmp_path):
    capture, output, info = generated(tmp_path)
    record = info["readonly_input_observation"]
    manifest = revalidate_observation(record)
    assert manifest["stage"] == "after_forward_before_state_commit"
    assert [a["original_index"] for a in manifest["arguments"]] == [0, 1, 2]
    assert [a["kind"] for a in manifest["arguments"]] == ["MERLIN_INPUT", "MERLIN_WEIGHT", "MERLIN_INPUT"]
    assert [a["bytes"] for a in manifest["arguments"]] == [2, 6, 8]
    assert info["n_outputs"] == 1 and info["n_state_pairs"] == 1
    header = (output / "readonly_inputs.h").read_text()
    assert "const volatile unsigned char" in header
    assert "MERLIN_INPUT_PTR[0]" in header and "MERLIN_INPUT_PTR[2]" in header
    assert "weights + 0ULL" in header
    raw, _metrics, frames = parse_binary_console_details(console(record))
    remaining, remaining_frames, observed = verify_observed_inputs(record, raw, frames)
    assert remaining == remaining_frames == {}
    assert [a["passed"] for a in observed["arguments"]] == [True] * 3
    assert (capture / "inputs.npz").is_file()


@pytest.mark.parametrize("indices", [True, [0], (True,), (0, 0), (-1,), (3,)])
def test_invalid_or_nonoriginal_argument_selection_refuses(tmp_path, indices):
    capture = bundle(tmp_path / "capture")
    with pytest.raises(ValueError):
        c_runtime.generate(capture, tmp_path / "out", capture / "inputs.npz", readonly_input_indices=indices)


@pytest.mark.parametrize("budget", [True, 0, -1, 15, (64 << 20) + 1])
def test_complete_observation_must_fit_original_byte_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="budget"):
        generated(tmp_path, max_readonly_input_bytes=budget)


def test_declared_mutability_is_not_overridden_by_a_readonly_request(tmp_path):
    capture = bundle(tmp_path / "capture")
    path = capture / "weights.safetensors.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["0"]["access"] = "readwrite"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="mutability"):
        c_runtime.generate(capture, tmp_path / "out", capture / "inputs.npz", readonly_input_indices=(0,))


def test_mutable_aliased_weight_is_a_conservative_barrier(tmp_path):
    capture = bundle(tmp_path / "capture", state=False)
    (capture / "model.mlir").write_text("""module {
      func.func @forward(%a:tensor<3xi16>,%b:tensor<3xi16>)->tensor<3xi16>{return %a:tensor<3xi16>}}""")
    (capture / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {"0": {"kind": "param", "weight": "w"}, "1": {"kind": "param", "weight": "w", "access": "readwrite"}}
        )
    )
    with pytest.raises(ValueError, match="aliases"):
        c_runtime.generate(capture, tmp_path / "out", capture / "inputs.npz", readonly_input_indices=(0,))


def test_observation_requires_original_manifest_and_complete_unmodified_frames(tmp_path):
    _capture, _output, info = generated(tmp_path)
    record = info["readonly_input_observation"]
    raw = console(record)
    outputs, _metrics, frames = parse_binary_console_details(raw)
    with pytest.raises(ReadbackError, match="results the forward does not have"):
        decode_outputs(outputs, frames, [])
    missing = dict(outputs)
    missing.pop("readonly_arg_1")
    with pytest.raises(ValueError, match="incomplete"):
        verify_observed_inputs(record, missing, frames)
    changed = {k: [list(row) for row in v] for k, v in outputs.items()}
    changed["readonly_arg_1"][0][-1] ^= 1
    with pytest.raises(ValueError, match="changed during forward"):
        verify_observed_inputs(record, changed, frames)
    expected = revalidate_observation(record)["expected"]["path"]
    with open(expected, "r+b") as stream:
        stream.write(b"\x00")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_observed_inputs(record, outputs, frames)


def test_default_codegen_remains_uninstrumented(tmp_path):
    capture = bundle(tmp_path / "capture")
    output = tmp_path / "out"
    info = c_runtime.generate(capture, output, capture / "inputs.npz")
    assert "readonly_input_observation" not in info
    assert not tuple(output.glob("readonly*"))


def test_argument_order_is_part_of_the_original_expected_byte_binding(tmp_path):
    capture, _output, info = generated(tmp_path)
    record = info["readonly_input_observation"]
    (capture / "input_order.json").write_text(json.dumps({"x": 1, "s": 0}))
    with pytest.raises(ValueError, match="artifact changed"):
        revalidate_observation(record)


def test_derivation_cannot_adopt_changed_source_metadata_after_reading_inputs(tmp_path, monkeypatch):
    capture = bundle(tmp_path / "capture")
    original = c_runtime._storage_bytes

    def change_order(array, dtype):
        data = original(array, dtype)
        (capture / "input_order.json").write_text(json.dumps({"x": 1, "s": 0}))
        return data

    monkeypatch.setattr(c_runtime, "_storage_bytes", change_order)
    with pytest.raises(ValueError, match="artifact changed"):
        c_runtime.generate(capture, tmp_path / "out", capture / "inputs.npz", readonly_input_indices=(0,))


def test_custom_stream_path_and_actual_step_zero_bytes_are_bound(tmp_path):
    capture = bundle(tmp_path / "capture")
    session = json.loads((capture / "session_contract.yaml").read_text())
    session.update(inputs="selected-stream.npz", streams=[{"name": "x", "input_arg": 0, "key": "x"}])
    (capture / "session_contract.yaml").write_text(json.dumps(session))
    selected = capture / "selected-stream.npz"
    np.savez(selected, x=np.array([[3, -4], [9, 11]], np.int8))
    info = c_runtime.generate(capture, tmp_path / "out", capture / "inputs.npz", readonly_input_indices=(0,))
    manifest = revalidate_observation(info["readonly_input_observation"])
    assert open(manifest["expected"]["path"], "rb").read() == bytes([3, 252])
    np.savez(selected, x=np.array([[3, -5], [9, 11]], np.int8))
    with pytest.raises(ValueError, match="artifact changed"):
        revalidate_observation(info["readonly_input_observation"])


def test_whole_model_decoder_accepts_only_the_exact_selected_extra_frame_roster(tmp_path):
    from merlin.compile.baremetal_model import read_full_outputs

    capture, _output, info = generated(tmp_path)
    record = info["readonly_input_observation"]
    state = np.array([[-1, 23]], np.int32)
    data = console(record).removesuffix(b"DONE\n") + _frame("out0", state, 4, True)
    data += b"METRIC build_hash selected\nMETRIC memref_rank_mismatch 0\nDONE\n"
    values, _metrics = read_full_outputs(data, capture, "selected", readonly_input_observation=record)
    assert values[0].tobytes() == state.tobytes()
    with pytest.raises(ReadbackError):
        read_full_outputs(data, capture, "selected")
    extra = _frame("readonly_arg_999", np.array([[0]], np.uint8), 1, False)
    with pytest.raises(ReadbackError):
        read_full_outputs(extra + data, capture, "selected", readonly_input_observation=record)


def test_genuine_but_different_observation_cannot_replace_original_request(tmp_path):
    from merlin.runtime.input_observation import require_observation_selection

    capture = bundle(tmp_path / "capture")
    info = c_runtime.generate(capture, tmp_path / "out", capture / "inputs.npz", readonly_input_indices=(0,))
    record = info["readonly_input_observation"]
    require_observation_selection(record, indices=(0,), max_bytes=1 << 20)
    for indices, budget in (((0, 1), 1 << 20), ((0,), 20)):
        with pytest.raises(ValueError, match="original selection"):
            require_observation_selection(record, indices=indices, max_bytes=budget)


@pytest.mark.parametrize("changed_argument", [None, 0, 1, 2])
def test_actual_native_observer_reads_original_buffers_before_state_commit(tmp_path, changed_argument):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable; no observer execution qualification")
    _capture, output, info = generated(tmp_path)
    record = info["readonly_input_observation"]
    weights = (output / "weights.bin").read_bytes()
    mutation = ["merlin_in_0[0] ^= 1;", "weights[5] ^= 1;", "merlin_in_2[1] ^= 1;"]
    change = "" if changed_argument is None else mutation[changed_argument]
    source = output / "observer.c"
    source.write_text(
        """#include <stdint.h>
#include <stdio.h>
#include "model_gen.h"
#include "model_io.h"
void htif_puts(const char *p) { fputs(p, stdout); }
void htif_putc(char c) { fputc((unsigned char)c, stdout); }
void htif_line_flush(int enabled) { (void)enabled; }
#include "readonly_inputs.h"
"""
        + "static unsigned char weights[] = {"
        + ",".join(map(str, weights))
        + "};\n"
        + f"""
int main(void) {{
 merlin_reset_session(); merlin_prepare_step(0);
 merlin_snapshot_readonly_inputs(weights);
 {change}
 merlin_capture_readonly_inputs();
 /* A legal subsequent state commit must not change the captured call observation. */
 merlin_in_2[0] = 123456;
 merlin_emit_readonly_inputs(); htif_puts("DONE\\n");
 return 0;
}}
"""
    )
    executable = output / "observer"
    argv = [compiler, "-std=c11", "-O2", f"-I{runtime_dir() / 'c'}", f"-I{output}", str(source), "-o", str(executable)]
    compiled = subprocess.run(argv, capture_output=True)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(executable)], capture_output=True, check=True)
    (output / "console.bin").write_bytes(executed.stdout)
    observations, _metrics, frames = parse_binary_console_details(executed.stdout)
    if changed_argument is None:
        _, _, checked = verify_observed_inputs(record, observations, frames)
        assert len(checked["arguments"]) == 3
    else:
        with pytest.raises(ValueError, match=f"argument {changed_argument} changed"):
            verify_observed_inputs(record, observations, frames)
    (output / "native-observer.json").write_text(
        json.dumps(
            {
                "scope": "native C argument observer only; no generated model or target execution",
                "argv": argv,
                "compiler": compiler,
                "compiler_sha256": hashlib.sha256(open(compiler, "rb").read()).hexdigest(),
                "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "console_sha256": hashlib.sha256(executed.stdout).hexdigest(),
                "mutated_argument": changed_argument,
            },
            indent=2,
        )
        + "\n"
    )


def test_ordinary_readback_setup_binds_manifest_to_actual_compilation_recipe(tmp_path):
    from pathlib import Path

    from merlin.llvmlower.compilation_recipe import CompilationRecipe
    from merlin.runtime.input_observation import setup_readback

    capture, output, info = generated(tmp_path)
    recipe = CompilationRecipe(output, producer=Path(c_runtime.__file__))
    setup_readback(output, capture, info, recipe, True, (0, 1, 2), 1 << 20)
    preparation = json.loads(recipe.path.read_text())["preparation"]["readonly_input_observation"]
    assert preparation == info["readonly_input_observation"]["manifest"]
    assert (output / "model_outputs.h").is_file()
    assert "MERLIN_MODEL_OUT" in (output / "model_outputs.h").read_text()


@pytest.mark.parametrize("bitcode", [False, True])
def test_actual_builder_registers_driver_observation_before_first_compile(tmp_path, monkeypatch, bitcode):
    from pathlib import Path

    from merlin.llvmlower import toolchain
    from merlin.llvmlower.compilation_recipe import CompilationRecipe
    from merlin.runtime.backends import spike, spike_model

    tools = (toolchain.clang(), toolchain.m2m_python(), spike.gcc_path())
    if not all(path.is_file() for path in tools):
        pytest.skip("ordinary upstream and cross compiler required for preparation-order regression")
    capture, work = tmp_path / "capture", tmp_path / "build"
    capture.mkdir()
    (capture / "model.mlir").write_text("""module {
      func.func @forward(%x:tensor<3xi8>)->tensor<3xi8> {return %x:tensor<3xi8>}
    }""")
    (capture / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"x"}}')
    np.savez(capture / "inputs.npz", in0=np.array([-128, 0, 127], np.int8))
    observed = []

    class FirstInvocationReached(Exception):
        pass

    def first(recipe, argv, *, runner, inputs, output, cwd=None):
        # Execute the actual builder down to its first CompilationRecipe command,
        # rather than constructing an artificial still-prepared recipe by hand.
        assert recipe.record["status"] == "prepared"
        assert recipe.record["commands"] == []
        preparation = recipe.record.get("preparation", {})
        policy = preparation["host_compiler_policy"]
        assert Path(policy["path"]).name == "host_compiler_policy.py"
        assert hashlib.sha256(Path(policy["path"]).read_bytes()).hexdigest() == policy["sha256"]
        assert "readonly_input_observation" in preparation
        pinned = preparation["readonly_input_observation"]
        manifest = json.loads(Path(pinned["path"]).read_text())
        assert [row["original_index"] for row in manifest["arguments"]] == [0]
        assert hashlib.sha256(Path(pinned["path"]).read_bytes()).hexdigest() == pinned["sha256"]
        assert (work / "cgen/readonly_inputs.h").is_file()
        assert (work / "cgen/model_outputs.h").is_file()
        assert not (work / "model.o").exists()
        observed.append(Path(output).name)
        raise FirstInvocationReached

    monkeypatch.setattr(CompilationRecipe, "_run", first)
    with pytest.raises(FirstInvocationReached):
        spike_model.build(
            capture,
            work,
            backend="scalar",
            arena_mb=1,
            full_readback=True,
            readonly_input_indices=(0,),
            host_llvm_bitcode=bitcode,
            cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding"],
        )
    assert observed == ["model.bc" if bitcode else "model.o"]
    assert not list(work.glob("*.elf"))
