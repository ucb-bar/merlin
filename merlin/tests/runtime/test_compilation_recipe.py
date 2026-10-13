"""Actual bare-metal build observations and failed invocation isolation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower import compilation_recipe, toolchain
from merlin.llvmlower.compilation_recipe import FILENAME, CompilationRecipe
from merlin.runtime.backends import spike, spike_model


def _read(work):
    return json.loads((work / FILENAME).read_text())


def _fake_completed_recipe(tmp_path, *, preparation=False):
    tool, producer = tmp_path / "tool", tmp_path / "producer.py"
    source, obj, elf = (tmp_path / name for name in ("source.c", "source.o", "model.elf"))
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    producer.write_text("# selected source\n")
    source.write_text("int value(void) { return 7; }\n")
    recipe = CompilationRecipe(tmp_path, producer=producer)
    if preparation:
        selected = tmp_path / "selected-policy.json"
        selected.write_text('{"mode":"selected"}\n')
        recipe.bind_preparation("policy", selected)

    def emit(output, data):
        def run(argv):
            output.write_bytes(data)
            return subprocess.CompletedProcess(argv, 0)

        return run

    recipe.run([tool, "-c", source, "-o", obj], runner=emit(obj, b"object"), inputs=[source], output=obj)
    recipe.run([tool, obj, "-o", elf], runner=emit(elf, b"executable"), inputs=[obj], output=elf)
    recipe.completed(elf)
    return tmp_path / FILENAME, elf, tool, producer, source, obj


def _verify(recipe, elf):
    return compilation_recipe.verify_completed_recipe(
        recipe, executable=elf, expected_recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest()
    )


def test_completed_recipe_refuses_postbuild_input_mutation(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    source, obj, elf = (tmp_path / name for name in ("source.c", "source.o", "model.elf"))
    source.write_text("int main(void) { return 7; }\n")
    recipe = CompilationRecipe(tmp_path, producer=Path(spike_model.__file__))
    recipe.run(
        [compiler, "-c", source, "-o", obj],
        runner=lambda argv: subprocess.run(argv, capture_output=True),
        inputs=[source],
        output=obj,
    )
    recipe.run(
        [compiler, obj, "-o", elf],
        runner=lambda argv: subprocess.run(argv, capture_output=True),
        inputs=[obj],
        output=elf,
    )
    recipe.completed(elf)
    receipt = tmp_path / FILENAME
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert (
        compilation_recipe.verify_completed_recipe(receipt, executable=elf, expected_recipe_sha256=receipt_sha)[
            "executable"
        ]["sha256"]
        == hashlib.sha256(elf.read_bytes()).hexdigest()
    )
    source.write_text("int main(void) { return 8; }\n")
    with pytest.raises(ValueError, match="input"):
        compilation_recipe.verify_completed_recipe(receipt, executable=elf, expected_recipe_sha256=receipt_sha)


@pytest.mark.parametrize("changed", ["tool", "producer", "source", "object", "executable"])
def test_completed_recipe_refuses_changed_bytes(tmp_path, changed):
    recipe, elf, tool, producer, source, obj = _fake_completed_recipe(tmp_path)
    assert _verify(recipe, elf)["status"] == "completed"
    target = {"tool": tool, "producer": producer, "source": source, "object": obj, "executable": elf}[changed]
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        _verify(recipe, elf)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("input_not_argv", "unrecorded direct file input"),
        ("output_not_argv", "output differs"),
        ("duplicate_input", "repeats an input identity"),
        ("duplicate_output", "output differs"),
        ("incomplete_command", "incomplete command"),
        ("environment", "environment digest"),
        ("final_executable", "final executable differs"),
        ("extra_direct_file", "absent direct file"),
        ("phantom_link_input", "link input has no direct argv reference"),
    ],
)
def test_completed_recipe_refuses_rehashed_document_mutation(tmp_path, mutation, reason):
    recipe, elf, tool, producer, source, obj = _fake_completed_recipe(tmp_path)
    document = _read(tmp_path)
    if mutation == "input_not_argv":
        alternate = tmp_path / "alternate.c"
        alternate.write_bytes(source.read_bytes())
        document["commands"][0]["inputs"][0]["path"] = str(alternate)
    elif mutation == "output_not_argv":
        document["commands"][-1]["argv"][-1] = str(obj)
    elif mutation == "duplicate_input":
        document["commands"][-1]["inputs"].append(document["commands"][-1]["inputs"][0])
    elif mutation == "duplicate_output":
        document["commands"][-1]["requested_output"] = str(obj)
        document["commands"][-1]["output"] = document["commands"][0]["output"]
    elif mutation == "incomplete_command":
        document["commands"][-1]["status"] = "invoked"
    elif mutation == "environment":
        document["commands"][-1]["environment_digest"] = "0" * 16
    elif mutation == "final_executable":
        document["executable"] = document["commands"][0]["output"]
    elif mutation == "extra_direct_file":
        document["commands"][-1]["argv"].append(str(tmp_path / "missing.o"))
    elif mutation == "phantom_link_input":
        archive = tmp_path / "libm.a"
        archive.write_bytes(b"!<arch>\n")
        document["commands"][-1]["inputs"].append(
            {
                "path": str(archive),
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "bytes": archive.stat().st_size,
            }
        )
    recipe.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=reason):
        _verify(recipe, elf)


def test_completed_recipe_refuses_indirect_identity_and_duplicate_json_key(tmp_path):
    recipe, elf, tool, producer, source, obj = _fake_completed_recipe(tmp_path)
    alias = tmp_path / "source-alias.c"
    alias.symlink_to(source)
    document = _read(tmp_path)
    document["commands"][0]["inputs"][0]["path"] = str(alias)
    recipe.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="indirect"):
        _verify(recipe, elf)
    recipe.write_text(
        recipe.read_text().replace('"status": "completed"', '"status": "completed", "status": "completed"', 1)
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        _verify(recipe, elf)


def test_completed_recipe_requires_selected_document_digest(tmp_path):
    recipe, elf, tool, producer, source, obj = _fake_completed_recipe(tmp_path)
    with pytest.raises(ValueError, match="selected.*digest"):
        compilation_recipe.verify_completed_recipe(recipe, executable=elf, expected_recipe_sha256="bad")
    previous = hashlib.sha256(recipe.read_bytes()).hexdigest()
    recipe.write_text(recipe.read_text() + " ")
    with pytest.raises(ValueError, match="selected digest"):
        compilation_recipe.verify_completed_recipe(recipe, executable=elf, expected_recipe_sha256=previous)


def test_completed_recipe_checks_preparation_and_selected_executable(tmp_path):
    recipe, elf, tool, producer, source, obj = _fake_completed_recipe(tmp_path, preparation=True)
    assert _verify(recipe, elf)["status"] == "completed"
    another = tmp_path / "another.elf"
    another.write_bytes(elf.read_bytes())
    with pytest.raises(ValueError, match="final executable differs"):
        _verify(recipe, another)
    (tmp_path / "selected-policy.json").write_text('{"mode":"changed"}\n')
    with pytest.raises(ValueError, match="preparation bytes changed"):
        _verify(recipe, elf)


def test_completed_recipe_refuses_duplicate_preparation_identity(tmp_path):
    recipe, elf, tool, producer, source, obj = _fake_completed_recipe(tmp_path, preparation=True)
    document = _read(tmp_path)
    document["preparation"]["second_role"] = document["preparation"]["policy"]
    recipe.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="repeats a preparation identity"):
        _verify(recipe, elf)


def test_failed_compiler_keeps_old_object_unqualified(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    source, obj = tmp_path / "source.c", tmp_path / "source.o"
    source.write_text("int value(void) { return 7; }\n")
    recipe = CompilationRecipe(tmp_path, producer=Path(spike_model.__file__))
    cmd = [compiler, "-c", source, "-o", obj]

    def run(argv):
        return subprocess.run(argv, capture_output=True)

    recipe.run(cmd, runner=run, inputs=[source], output=obj)
    old_bytes = obj.read_bytes()
    source.write_text("not valid C\n")
    with pytest.raises(RuntimeError, match="compilation command returned"):
        recipe.run(cmd, runner=run, inputs=[source], output=obj)
    record = _read(tmp_path)
    assert record["status"] == "failed"
    assert record["commands"][-1]["status"] == "failed"
    assert "output" not in record["commands"][-1]
    assert obj.read_bytes() == old_bytes
    assert "executable" not in record
    with pytest.raises(ValueError, match="complete successful command sequence"):
        recipe.completed(obj)


@pytest.mark.parametrize("kind", ["executable", "object", "llvm_ir"])
def test_completed_product_records_actual_kind_and_rejects_unknown_kind(tmp_path, kind):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    source, obj = tmp_path / "source.c", tmp_path / "source.o"
    source.write_text("int value(void) { return 7; }\n")
    recipe = CompilationRecipe(tmp_path, producer=Path(spike_model.__file__))
    recipe.run(
        [compiler, "-c", source, "-o", obj],
        runner=lambda argv: subprocess.run(argv, capture_output=True),
        inputs=[source],
        output=obj,
    )
    with pytest.raises(ValueError, match="supported compilation product kind"):
        recipe.completed_product(obj, kind="unknown")
    assert _read(tmp_path)["status"] == "invoked"
    if kind == "executable":
        recipe.completed(obj)
    else:
        recipe.completed_product(obj, kind=kind)
    record = _read(tmp_path)
    assert record["status"] == "completed"
    assert record[kind]["sha256"] == hashlib.sha256(obj.read_bytes()).hexdigest()
    assert not {"executable", "object", "llvm_ir"}.difference({kind}).intersection(record)


def test_normal_model_compile_runtime_and_link_are_observed(tmp_path, monkeypatch):
    if not spike.available() or not toolchain.m2m_python().is_file() or not toolchain.clang().is_file():
        pytest.skip("bare-metal and upstream toolchain unavailable")
    monkeypatch.setenv("MERLIN_TEST_API_TOKEN", "private-compiler-canary")
    bundle, work = tmp_path / "bundle", tmp_path / "build"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("""module {
      func.func @forward(%arg0: tensor<5xf32>) -> tensor<5xf32> {
        %init = tensor.empty() : tensor<5xf32>
        %out = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]
        } ins(%arg0 : tensor<5xf32>) outs(%init : tensor<5xf32>) {
          ^bb0(%value: f32, %unused: f32):
            %two = arith.constant 2.0 : f32
            %sum = arith.addf %value, %two : f32
            linalg.yield %sum : f32
        } -> tensor<5xf32>
        return %out : tensor<5xf32>
      }
    }""")
    (bundle / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    values = np.array([-5, -0.75, 0, 1.25, 7], dtype=np.float32)
    np.savez(bundle / "inputs.npz", in0=values)
    built = spike_model.build(
        bundle,
        work,
        arena_mb=1,
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
        host_math_policy="expf_via_double",
    )
    record = _read(work)
    assert record["status"] == "completed"
    assert record["executable"]["sha256"] == hashlib.sha256(built["elf"].read_bytes()).hexdigest()
    assert "private-compiler-canary" not in json.dumps(record)
    commands = record["commands"]
    assert all(command["status"] == "returned" for command in commands)
    by_output = {Path(command["output"]["path"]).name: command for command in commands}
    assert {"model.o", "mlir_rt.o", "host_math.o", "model_main.o", "weights_blob.o", "model.elf"} <= by_output.keys()
    assert by_output["mlir_rt.o"]["executable"] == by_output["model.o"]["executable"]
    assert by_output["mlir_rt.o"]["executable"] != by_output["model_main.o"]["executable"]
    assert "-march=rv64gc" in by_output["mlir_rt.o"]["argv"]
    assert by_output["weights_blob.o"]["cwd"] == str((work / "cgen").resolve())
    assert "-Wl,--wrap=expf" in by_output["model.elf"]["argv"]
    link_inputs = by_output["model.elf"]["inputs"]
    assert [Path(item["path"]).name for item in link_inputs] == [
        "model_link.ld",
        "model_call.o",
        "merlin_model.o",
        "model_main.o",
        "mlir_rt.o",
        "crt.o",
        "console.o",
        "libc_min.o",
        "malloc.o",
        "model.o",
        "weights_blob.o",
        "host_math.o",
        "libm.a",
    ]
    for command in commands:
        for identity in [command["executable"], *command["inputs"], command["output"]]:
            assert identity["sha256"] == hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest()
    # The recorded ordinary link is itself sufficient to reproduce these exact bytes.
    subprocess.run(commands[-1]["argv"], cwd=commands[-1]["cwd"], check=True, capture_output=True)
    assert hashlib.sha256(built["elf"].read_bytes()).hexdigest() == record["executable"]["sha256"]
    ran = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"], isa="rv64gc", timeout=60)
    np.testing.assert_array_equal(ran["outputs"], values + np.float32(2))

    with pytest.raises(ValueError, match="output_dump_cap"):
        spike_model.build(bundle, work, output_dump_cap=0)
    assert not (work / FILENAME).exists()
