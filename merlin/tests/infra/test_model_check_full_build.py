"""Whole-model reuse binds content and recurring inputs retain actual link evidence."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower import c_runtime
from merlin.llvmlower.compilation_recipe import CompilationRecipe, verify_completed_recipe
from merlin.model_check import full_build as FB
from merlin.model_check import full_build_identity as ID


@pytest.fixture
def built(tmp_path, monkeypatch):
    compiler = Path(shutil.which("cc") or "")
    if not compiler.is_file():
        pytest.skip("a host C compiler is required")
    compiler = compiler.resolve()
    stage = tmp_path / "capture"
    stage.mkdir()
    (stage / "model.mlir").write_text("""module {
      func.func @forward(%x: tensor<2xf32>, %w: tensor<2xf32>) -> tensor<2xf32> { return %x : tensor<2xf32> }
    }""")
    (stage / "weights.safetensors.manifest.json").write_text(json.dumps({
        "0": {"kind": "input", "name": "x"}, "1": {"kind": "param", "weight": "w"}
    }))
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    (stage / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header
                                               + np.array([5, 6], np.float32).tobytes())
    (stage / "input_order.json").write_text(json.dumps({"x": 0}))
    np.savez(stage / "inputs.npz", in0=np.array([1, 2], np.float32))
    source = tmp_path / "builder.py"
    source.write_text("original compiler source\n")
    calls = []

    def material(stage, target, label, submission, submission_sha, settings, offload):
        from merlin.compile.model_execution_inputs import strict_tree_sha256

        return {"stage": str(stage.resolve()), "capture": strict_tree_sha256(stage),
                "source": FB.sha256(source), "target": target, "offload": offload}

    monkeypatch.setattr(ID, "build_material", material)

    def compile_model(**kwargs):
        calls.append(kwargs)
        directory = kwargs["output"] / "build"
        directory.mkdir(parents=True)
        cgen = directory / "cgen"
        c_runtime.generate(stage, cgen, stage / "inputs.npz", prepared_dir=directory, max_session_steps=1)
        model = directory / "model.c"
        model.write_text("int shared_value(void) { return 5; }\n")
        driver = directory / "model_main.c"
        driver.write_text('#include <stdio.h>\n#include "model_gen.h"\n#include "model_io.h"\n'
                          'extern int shared_value(void);\nint main(void) { printf("%.0f %.0f %d\\n", '
                          'merlin_in_0[0], merlin_in_0[1], shared_value()); return 0; }\n')
        recipe = CompilationRecipe(directory, producer=source)

        def run(argv, **kwargs):
            return subprocess.run(argv, capture_output=True, text=True, check=True, **kwargs)

        for name, path in (("model", model), ("model_main", driver)):
            obj = directory / f"{name}.o"
            recipe.run([compiler, "-I", cgen, "-I", runtime_dir() / "c", "-c", path, "-o", obj],
                       runner=run, inputs=[path], output=obj)
        elf = directory / "model.elf"
        objects = [directory / "model.o", directory / "model_main.o"]
        recipe.run([compiler, *objects, "-lm", "-o", elf], runner=run, inputs=objects, output=elf)
        recipe.completed(elf)
        return {"output": {"elf": str(elf), "elf_sha256": FB.sha256(elf), "build_hash": "fixture-build",
                           "compilation_recipe": {"path": str(recipe.path), "sha256": FB.sha256(recipe.path)}}}

    from merlin.compile import baremetal_model

    monkeypatch.setattr(baremetal_model, "compile_saved_model", compile_model)
    directory = tmp_path / "models"
    settings = FB.BuildSettings("board", tmp_path / "boards", tmp_path / "host.dts", "scalar", 1, tmp_path / "host")

    def build():
        return FB.build(stage, target="t", label="model", submission=tmp_path / "submission",
                        submission_sha256="a" * 64, settings=settings, directory=directory,
                        admission=object(), offload=False)

    return build, stage, directory, source, calls


def test_build_reuse_checks_recipe_elf_bytes_and_current_capture_and_source(built):
    build, stage, directory, source, calls = built
    first = build()
    assert not first["reused"] and build()["reused"] and len(calls) == 1
    elf = Path(first["elf"])
    data = bytearray(elf.read_bytes())
    data[-1] ^= 1  # Same length; a size-only cache would accept this image.
    elf.write_bytes(data)
    assert not build()["reused"] and len(calls) == 2
    (Path(first["build_dir"]) / "model.c").write_text("changed producer input\n")
    assert not build()["reused"] and len(calls) == 3
    np.savez(stage / "inputs.npz", in0=np.array([3, 4], np.float32))
    assert not build()["reused"] and len(calls) == 4
    source.write_text("changed compiler source\n")
    assert not build()["reused"] and len(calls) == 5
    record_path = directory / "build.json"
    legacy = json.loads(record_path.read_text())
    legacy["schema"] = "merlin.model_check.full_build.v1"
    record_path.write_text(json.dumps(legacy))
    assert not build()["reused"] and len(calls) == 6


def test_recurrent_relinks_generate_weights_once_and_execute_current_inputs(built, tmp_path, monkeypatch):
    build, stage, _directory, _source, _calls = built
    record = build()
    calls = []
    generate = c_runtime.generate

    def counted(*args, **kwargs):
        calls.append(args)
        return generate(*args, **kwargs)

    monkeypatch.setattr(c_runtime, "generate", counted)
    for index, expected in enumerate(([3, 4], [8, 9], [-7, 6])):
        directory = tmp_path / f"step{index}"
        elf = FB.relink(record, stage, {"in0": np.array(expected, np.float32)}, directory)
        assert subprocess.check_output([elf], text=True).strip() == f"{expected[0]} {expected[1]} 5"
        recipe = directory / "compilation_recipe.json"
        observed = verify_completed_recipe(recipe, executable=elf, expected_recipe_sha256=FB.sha256(recipe))
        assert any(row["path"].endswith("model_io.h") for row in observed["commands"][0]["inputs"])
    assert len(calls) == 1, "unchanged weights and static call artifacts are generated once"


def test_relink_rejects_changed_static_objects_before_launching_the_link(built, tmp_path):
    build, stage, _directory, _source, _calls = built
    record = build()
    static = Path(record["build_dir"]) / "model.o"
    static.write_bytes(static.read_bytes() + b"changed")
    with pytest.raises(FB.FullBuildError, match="immutable original build input"):
        FB.relink(record, stage, {"in0": np.array([3, 4], np.float32)}, tmp_path / "step")
    assert not (tmp_path / "step/model.elf").exists()


def test_relink_rejects_changed_input_abi_and_another_capture(built, tmp_path):
    build, stage, _directory, _source, _calls = built
    record = build()
    with pytest.raises(FB.FullBuildError, match="another model capture"):
        FB.relink(record, tmp_path / "another", {}, tmp_path / "step")
    (stage / "input_order.json").write_text(json.dumps({"x": 1}))
    with pytest.raises(FB.FullBuildError, match="input ABI changed"):
        FB.relink(record, stage, {"in0": np.array([3, 4], np.float32)}, tmp_path / "step")


def test_relink_refuses_modified_cached_template_and_input_ownership(built, tmp_path):
    build, stage, _directory, _source, _calls = built
    record = build()
    inputs = {"in0": np.array([3, 4], np.float32)}
    FB.relink(record, stage, inputs, tmp_path / "step0")
    template = Path(record["build_dir"]) / "model_check_single_forward"
    manifest_path = template / "template.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bindings"][0]["argument"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FB.FullBuildError, match="input ownership"):
        FB.relink(record, stage, inputs, tmp_path / "step1")
    manifest["bindings"][0]["argument"] = 0
    manifest_path.write_text(json.dumps(manifest))
    (template / "model_gen.h").write_text("changed static ABI\n")
    with pytest.raises(FB.FullBuildError, match="template bytes changed"):
        FB.relink(record, stage, inputs, tmp_path / "step2")


def test_build_material_binds_direct_tools_sources_settings_and_input_bytes(tmp_path, monkeypatch):
    from merlin.common import paths
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike

    source = tmp_path / "src/merlin"
    source.mkdir(parents=True)
    compiler_source = source / "compiler.py"
    compiler_source.write_text("compiler one\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "driver.c").write_text("runtime one\n")
    host = tmp_path / "host"
    host.mkdir()
    (host / "manifest.yaml").write_text("host one\n")
    stage = tmp_path / "capture"
    stage.mkdir()
    weights = stage / "weights"
    weights.write_bytes(b"weights one")
    tool = tmp_path / "tool"
    tool.write_bytes(b"tool one")
    mlir_install = tmp_path / "llvm"
    mlir_optimizer = mlir_install / "bin/mlir-opt"
    mlir_optimizer.parent.mkdir(parents=True)
    mlir_optimizer.write_bytes(b"MLIR one")
    boards, dts = tmp_path / "boards.yaml", tmp_path / "cpu.dts"
    boards.write_text("boards one\n")
    dts.write_text("cpu one\n")
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "runtime_dir", lambda: runtime)
    for name in ("clang", "mlir_translate", "llvm_opt", "compiler_python", "objcopy", "nm", "readelf"):
        monkeypatch.setattr(toolchain, name, lambda: tool)
    monkeypatch.setattr(toolchain, "llvm_install", lambda: mlir_install)
    monkeypatch.setattr(spike, "gcc_path", lambda: tool)
    monkeypatch.setattr(ID, "target_inputs", lambda: {"facts": "pinned"})
    settings = FB.BuildSettings("board", boards, dts, "scalar", 1, host)

    def material():
        return ID.build_material(stage, "t", "program", tmp_path / "absent-submission", "a" * 64,
                                 settings, False)

    first = material()
    # A timestamp-preserving same-size tool replacement still invalidates reuse.
    import os

    previous = tool.stat()
    tool.write_bytes(b"tool two")
    os.utime(tool, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert material() != first
    first = material()
    previous = mlir_optimizer.stat()
    mlir_optimizer.write_bytes(b"MLIR two")
    os.utime(mlir_optimizer, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    changed = material()
    assert changed["tools"]["mlir_optimizer"]["sha256"] != first["tools"]["mlir_optimizer"]["sha256"]
    assert changed != first
    first = material()
    (tmp_path / ".env").write_text("MERLIN_COMPILER_DRIVER=stock-cli\n")
    assert material() != first, "dotenv compiler choices are actual build inputs, even without process-env changes"
    first = material()
    weights.write_bytes(b"weights two")
    assert material() != first
    first = material()
    compiler_source.write_text("compiler two\n")
    assert material() != first
    first = material()
    boards.write_text("boards two\n")
    assert material() != first
    first = material()
    (host / "manifest.yaml").write_text("host two\n")
    assert material() != first
    first = material()
    monkeypatch.setenv("MERLIN_SYNTHETIC_CODEGEN_FLAG", "changed")
    assert material() != first
