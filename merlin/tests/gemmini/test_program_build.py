"""Bounded complete-source build seam; no target execution or candidate code."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.runtime.backends.base import get_backend


@pytest.fixture
def edge():
    backend = get_backend("gemmini")
    return importlib.import_module(backend.__package__ + ".gemmini_program_build")


def request():
    source = '''builtin.module {
      func.func @tiny(%a: tensor<1xi32>) -> tensor<1xi32> {
        func.return %a : tensor<1xi32>
      }
    }'''
    lowered = '''builtin.module {
      llvm.func @gemmini_kernel(%a: !llvm.ptr, %out: !llvm.ptr) {
        %x = llvm.load %a : !llvm.ptr -> i32
        llvm.store %x, %out : i32, !llvm.ptr
        llvm.return
      }
    }'''
    encoding = GroupedAxesStorage((1,), "i32", ((0,),), (1,), (1,), 1).to_dict()
    cb = {"abi_version": "0.1", "target": "gemmini", "commands": [],
        "tensors": {"a": {"shape": [1], "dtype": "i32", "role": "input"},
                    "out": {"shape": [1], "dtype": "i32", "role": "output"}},
        "kernel_abi": {"kind": "whole_program", "args": [
            {"tensor": "a", "access": "read"}, {"tensor": "out", "access": "write"}],
            "outputs": ["out"]},
        "params": {"storage_encodings": {name: copy.deepcopy(encoding) for name in ("a", "out")}}}
    return dict(source_text=source, lowered_text=lowered, command_buffer=cb,
        logical_payloads={"a": (7).to_bytes(4, "little")},
        source_evidence={"probe_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "lowered_sha256": hashlib.sha256(lowered.encode()).hexdigest(),
                         "origin": "trusted tiny test source; no model qualification"},
        python_executable=Path(sys.executable).absolute())


@pytest.mark.parametrize("kind", ["missing", "readwrite", "stale", "oversize", "renamed_model", "dynamic"])
def test_refuses_before_worker(edge, tmp_path, kind):
    kwargs = request()
    if kind == "missing":
        kwargs["logical_payloads"] = {}
    elif kind == "readwrite":
        kwargs["command_buffer"]["kernel_abi"]["args"][0]["access"] = "readwrite"
    elif kind == "stale":
        kwargs["lowered_text"] += "\n"
    elif kind == "oversize":
        kwargs["lowered_text"] = " " * (edge.MAX_LLVM_BYTES + 1)
    else:
        kwargs["source_text"] = kwargs["source_text"].replace("1xi32", "999999999xi32" if kind == "renamed_model" else "?xi32")
        kwargs["source_evidence"]["probe_source_sha256"] = hashlib.sha256(kwargs["source_text"].encode()).hexdigest()
    with pytest.raises(ValueError):
        edge.prepare_short_program_build(**kwargs, workdir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_stale_inputs_fail_before_callback(edge, tmp_path):
    prepared = edge.prepare_short_program_build(**request(), workdir=tmp_path)
    (tmp_path / "input_0.bin").write_bytes(b"bad!")
    def forbidden(*args, **kwargs):
        pytest.fail("stale input must not invoke any subprocess")
    with pytest.raises(ValueError, match="changed"):
        prepared.run(forbidden, timeout_s=60)


@pytest.mark.parametrize("legacy",[False,True])
def test_actual_tiny_elf_via_deadline_callback(edge, tmp_path, legacy):
    from merlin.llvmlower.toolchain import clang
    recipe = get_backend("gemmini").harness_build_recipe()
    if not Path(clang()).is_file() or not Path(recipe.compiler).is_file():
        pytest.skip("configured target compilers unavailable")
    kwargs=request()
    if legacy:
        kwargs["command_buffer"]["params"].pop("storage_encodings")
    original=copy.deepcopy(kwargs["command_buffer"])
    prepared = edge.prepare_short_program_build(**kwargs, workdir=tmp_path)
    assert kwargs["command_buffer"]==original
    if legacy:
        spec=json.loads(Path(prepared.request_path).read_text())
        assert spec["legacy_abi"]["dim"]>0
        assert "storage_encodings" not in json.loads((tmp_path/"command_buffer.json").read_text())["params"]
    calls = []
    def bounded(argv, *, timeout_s):
        calls.append((argv, timeout_s))
        env = dict(os.environ, PYTHONPATH=str(merlin_dir() / "python"), MERLIN_CACHE_STATE="cold")
        # Trusted fixture worker only. Production supplies the existing masked
        # run_native_probe closure, not this test's process launcher.
        proc = subprocess.Popen(argv, cwd=tmp_path, env=env, start_new_session=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
            return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
    receipt = prepared.run(bounded, timeout_s=60)
    assert len(calls) == 1 and calls[0][1] == 60
    assert receipt["status"] == "built_not_executed" and not receipt["target_executed"]
    assert receipt["warm_invocations_emitted"] == receipt["measured_invocations_emitted"] == 1
    harness = (tmp_path / "build/harness.c").read_text()
    assert harness.count("gemmini_kernel((void*)T_a, (void*)T_out);") == 2
    assert harness.index("gemmini_kernel((void*)T_a") < harness.index("uint64_t c0")
    assert harness.rindex("gemmini_kernel((void*)T_a") < harness.index("uint64_t c1")
    assert Path(receipt["elf_path"]).read_bytes()[:4] == b"\x7fELF"
    assert json.loads((tmp_path / "build_receipt.json").read_text())["request_sha256"]


def test_legacy_five_argument_boundary_keeps_initializer_and_accumulator_width(edge):
    import struct
    from merlin.targetgen.contract.build_service import load_build_package
    pure=load_build_package(merlin_dir()/"targets/gemmini/build_support/__init__.py")
    specs={"seed":([2,3],"i8","read"),"left":([2,5],"i8","read"),
        "right":([5,3],"i8","read"),"output":([2,3],"i8","write"),
        "accumulator":([2,3],"i32","write")}
    cb={"target":"gemmini","params":{},"commands":[],
        "tensors":{name:{"shape":shape,"dtype":dtype} for name,(shape,dtype,_) in specs.items()},
        "kernel_abi":{"kind":"whole_program","args":[{"tensor":name,"access":access} for name,(_,_,access) in specs.items()],
            "outputs":["output"]}}
    payloads={"seed":struct.pack("<6b",-128,-7,-1,0,1,127),
              "left":struct.pack("<10b",*range(-5,5)),"right":struct.pack("<15b",*range(-7,8))}
    original=copy.deepcopy(cb)
    # A non-default geometry verifies that the pure packer consumes host facts.
    inputs,storage=edge._inputs(cb,payloads,legacy_dim=7,legacy_format=pure.format)
    assert cb==original and set(inputs)==set(payloads)
    assert storage["left"]["row_stride_bytes"]==7
    assert storage["accumulator"]["row_stride_bytes"]==28
    assert storage["accumulator"]["storage_bytes"]==196
    for name,payload in payloads.items():
        assert struct.pack("<"+str(len(inputs[name].data))+"b",*inputs[name].data)==payload
    text=pure.render_whole_program(cb,inputs=inputs,legacy_dim=7)
    assert text.count("gemmini_kernel((void*)T_seed, (void*)T_left, (void*)T_right, (void*)T_output, (void*)T_accumulator);")==2
    assert "int32_t T_accumulator[49]" in text and "T_output[i * 7 + j]" in text
    cb["kernel_abi"]["args"][0]["access"]="readwrite"
    with pytest.raises(ValueError,match="readwrite"):
        edge._inputs(cb,payloads,legacy_dim=7,legacy_format=pure.format)


@pytest.mark.parametrize("problem",["absent","stale","expression","conflict"])
def test_legacy_header_geometry_requires_exact_selected_bytes(edge,tmp_path,problem):
    header=tmp_path/"gemmini_params.h"
    text="#define DIM 7\n" if problem!="expression" else "#define DIM (7 + 1)\n"
    header.write_text(text)
    row={"kind":"header","source":str(header),"destination":str(header),"sha256":hashlib.sha256(text.encode()).hexdigest()}
    spec={"dependencies":[row]}
    if problem=="absent":spec["dependencies"]=[]
    elif problem=="stale":row["sha256"]="0"*64
    elif problem=="conflict":
        other=tmp_path/"other";other.mkdir();other=other/"gemmini_params.h";other.write_text("#define DIM 9\n")
        spec["dependencies"].append({**row,"source":str(other),"destination":str(other),"sha256":hashlib.sha256(other.read_bytes()).hexdigest()})
    with pytest.raises(ValueError):edge._legacy_dimension(spec)


def test_evaluator_still_invokes_existing_reference(monkeypatch):
    backend = get_backend("gemmini")
    module = importlib.import_module(backend.__package__ + ".gemmini")
    from merlin.runtime import reference
    seen = []
    monkeypatch.setattr(module, "available", lambda _: True)
    monkeypatch.setattr(module, "compile_command_buffer", lambda *a, **k: Path("fixture.elf"))
    monkeypatch.setattr(module, "run_elf", lambda *a, **k: "fixture")
    monkeypatch.setattr(module, "parse_output", lambda _: ({"out": [7]}, {}))
    monkeypatch.setattr(reference, "reference_outputs", lambda cb: seen.append(cb) or {"out": [7]})
    monkeypatch.setattr(reference, "outputs_match", lambda actual, expected: actual == expected)
    cb = {"trusted": "test"}
    result = module.run_command_buffer(cb, workdir="fixture", simulator="spike")
    assert result["correct"] and seen == [cb]


def test_target_renderer_import_does_not_require_masked_answers():
    program = '''
import importlib.abc, sys
class BlockAnswers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"merlin.runtime.reference", "merlin.runtime.simulator"}:
            raise ImportError("masked answer module")
sys.meta_path.insert(0, BlockAnswers())
from merlin.runtime.backends.base import get_backend
backend = get_backend("gemmini")
assert callable(backend.render_harness)
assert "merlin.runtime.reference" not in sys.modules
assert "merlin.runtime.simulator" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(merlin_dir()/"python")), timeout=15)
    assert result.returncode == 0, result.stderr


def test_format_data_is_host_selected_typed_and_schema_bound(edge, tmp_path):
    prepared = edge.prepare_short_program_build(**request(), workdir=tmp_path)
    specification = json.loads(Path(prepared.request_path).read_text())['build_service']
    items = [row for row in prepared.dependencies if row['kind'] == 'format_data']
    assert {row['role'] for row in items} == {'numeric_format_registry', 'numeric_format_schema'}
    assert {Path(row['source']).name for row in items} == {
        'quant_formats.registry.yaml', 'quant_format.schema.yaml'}
    relation = specification['format_data_relation']
    assert relation['validation']['status'] == 'validated'
    assert relation['validation']['entry_count'] > 0
    assert relation['overlays_allowed'] is False
    expected = hashlib.sha256(edge._json(relation).encode()).hexdigest()
    assert all(row['format_data_relation_sha256'] == expected for row in items)


def test_numeric_overlay_is_not_silently_added_to_build_authority(edge, tmp_path, monkeypatch):
    monkeypatch.setenv('MERLIN_QUANT_FORMATS', str(tmp_path/'ungranted.yaml'))
    with pytest.raises(ValueError, match='overlays'):
        edge._prepare_build_service(None)
    assert not list(tmp_path.iterdir())


def test_fresh_numeric_schema_required_fields_cannot_use_cached_schema(edge, tmp_path):
    registry = tmp_path/'registry.yaml'
    schema = tmp_path/'schema.yaml'
    registry.write_text('version: 1\nformats:\n  tiny:\n    kind: int_affine\n    element_bits: 8\n')
    schema.write_text('required_top_level_fields: [name, kind, element_bits, fresh_required]\n')
    with pytest.raises(ValueError, match='freshly required'):
        edge._validate_format_data(registry, schema)
    schema.write_text('required_top_level_fields: [name, kind, element_bits]\n')
    assert edge._validate_format_data(registry, schema)['entry_count'] == 1
    registry.write_text('version: 1\nformats:\n  tiny:\n    kind: float_ieee\n    element_bits: 8\n    exp_bits: 5\n    mant_bits: 7\n')
    with pytest.raises(ValueError, match='element_bits'):
        edge._validate_format_data(registry, schema)
