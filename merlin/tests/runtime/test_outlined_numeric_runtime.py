"""Exact existing runtime C bytes use the model compiler and environment headers."""

from pathlib import Path

import pytest

from merlin.llvmlower.native_runtime import NativeRuntimeError, dependency_paths


def test_dependency_parser_preserves_escaped_filename_bytes(tmp_path):
    names = ["source.c", "header with space.h", "hash#header.h", "dollar$header.h", "quote'header.h", "back\\header.h"]
    for name in names:
        (tmp_path / name).touch()
    text = (
        "merlin_runtime: source.c \\\n"
        " header\\ with\\ space.h hash\\#header.h dollar$$header.h "
        "quote'header.h back\\\\header.h\n"
    )
    assert dependency_paths(text, tmp_path) == tuple(sorted((tmp_path / n).resolve() for n in names))


@pytest.mark.parametrize(
    "text",
    [
        "wrong: source.c",
        "merlin_runtime:",
        "merlin_runtime: source.c $VAR",
        "merlin_runtime: source.c # comment",
        "merlin_runtime: source.c \\",
    ],
)
def test_unproved_dependency_syntax_refuses_before_path_resolution(tmp_path, text):
    (tmp_path / "source.c").touch()
    with pytest.raises(NativeRuntimeError):
        dependency_paths(text, tmp_path)


def test_actual_existing_numeric_runtime_compiler_and_headers(tmp_path):
    import os

    from merlin.common.paths import runtime_dir
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.native_runtime import build_numeric_runtime

    if not os.getenv("MERLIN_MLIR_INSTALL"):
        pytest.skip("explicit selected model compiler required")
    driver = Path(os.getenv("MERLIN_RUNTIME_TEST_HEADER_DRIVER", ""))
    if not driver.is_file():
        pytest.skip("explicit environment header compiler unavailable")
    built = build_numeric_runtime(
        tmp_path / "runtime",
        model_compiler=toolchain.clang(),
        header_driver=driver,
        model_flags=("--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d", "-O2", "-ffreestanding"),
    )
    assert built["status"] == "built" and Path(built["object"]).is_file()
    assert str((runtime_dir() / "abi/portable_math.h").resolve()) in built["input_sha256"]
    assert str((runtime_dir() / "abi/portable_math_tables.h").resolve()) in built["input_sha256"]
    assert any(path.endswith("/math.h") for path in built["dependencies"])
    assert built["model_compiler"] == str(toolchain.clang().resolve())
    assert built["source_math_changed"] is False and built["target_link_complete"] is False


@pytest.mark.parametrize("flags", [(), ("-o",), ("-MFelsewhere",), ("@unbound.rsp",), ("bad\0flag",)])
def test_unbound_output_or_response_options_refuse_before_tool_resolution(tmp_path, flags):
    from merlin.llvmlower.native_runtime import build_numeric_runtime

    with pytest.raises(NativeRuntimeError):
        build_numeric_runtime(
            tmp_path / "work",
            model_compiler=tmp_path / "missing",
            header_driver=tmp_path / "missing",
            model_flags=flags,
        )


@pytest.mark.parametrize("change", ["header", "dependency_set"])
def test_actual_compile_refuses_changed_dependency_or_source_bytes(tmp_path, monkeypatch, change):
    import os
    import shutil

    from merlin.common import paths, proc
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.native_runtime import build_numeric_runtime

    driver = Path(os.getenv("MERLIN_RUNTIME_TEST_HEADER_DRIVER", ""))
    if not driver.is_file() or not os.getenv("MERLIN_MLIR_INSTALL"):
        pytest.skip("explicit selected model compiler/header driver required")
    owned = tmp_path / "runtime"
    (owned / "abi").mkdir(parents=True)
    for name in ("mlir_runtime.c", "portable_math.h", "portable_math_tables.h"):
        shutil.copyfile(paths.runtime_dir() / "abi" / name, owned / "abi" / name)
    extra = tmp_path / "extra.h"
    extra.write_text("/* not an actual dependency */\n")
    monkeypatch.setattr(paths, "runtime_dir", lambda: owned)
    original = proc.run_checked

    def run(argv, **kwargs):
        result = original(argv, **kwargs)
        if "-c" in argv:
            if change == "header":
                header = owned / "abi/portable_math.h"
                header.write_bytes(header.read_bytes() + b"\n/* changed after compiler return */\n")
            else:
                depfile = Path(argv[argv.index("-MF") + 1])
                depfile.write_text(depfile.read_text().rstrip() + " " + str(extra) + "\n")
        return result

    monkeypatch.setattr(proc, "run_checked", run)
    with pytest.raises(NativeRuntimeError, match="changed"):
        build_numeric_runtime(
            tmp_path / "work",
            model_compiler=toolchain.clang(),
            header_driver=driver,
            model_flags=("--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d", "-O2", "-ffreestanding"),
        )


def test_public_builder_and_json_cli_link_actual_runtime_without_changing_kernels(tmp_path):
    import json
    import os
    from dataclasses import asdict

    from merlin.compile.outlined_program import build_outlined_program, main
    from merlin.runtime.staged_dispatch import StagedRoles

    driver = Path(os.getenv("MERLIN_RUNTIME_TEST_HEADER_DRIVER", ""))
    if not driver.is_file() or not os.getenv("MERLIN_COMPILER_PYTHON"):
        pytest.skip("explicit selected model compiler/header driver required")
    source = tmp_path / "source.mlir"
    source.write_text("""module {func.func @forward(%x:tensor<8xf32>)->tensor<8xf32>{
 %a=tensor.empty():tensor<8xf32>
 %r=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}
 ins(%x:tensor<8xf32>)outs(%a:tensor<8xf32>){
 ^bb0(%v:f32,%old:f32):
 %y=math.rsqrt %v:f32
 linalg.yield %y:f32
 }->tensor<8xf32>
 func.return %r:tensor<8xf32>
}}""")
    roles = StagedRoles((0, 1, 2), 2, 2, 2, 2, (0,), (0,))
    kwargs = dict(
        roles=roles,
        worker_slots=(0,),
        workspace_budget_bytes=4096,
        alignment_bytes=64,
        compute_cflags=("-march=rv64gcv",),
    )
    ordinary = build_outlined_program(source, tmp_path / "ordinary", **kwargs)
    assert "numeric_runtime" not in ordinary
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema": "merlin.outlined_program.v1",
                **kwargs,
                "roles": asdict(roles),
                "runtime_header_driver": str(driver),
            }
        )
    )
    assert main(["--source", str(source), "--config", str(config), "--output", str(tmp_path / "selected")]) == 0
    selected = json.loads((tmp_path / "selected/build.json").read_text())
    runtime = selected["numeric_runtime"]
    assert selected["kernel_builds"][0]["object_sha256"] == ordinary["kernel_builds"][0]["object_sha256"]
    assert selected["kernel_builds"][0]["llvm_sha256"] == ordinary["kernel_builds"][0]["llvm_sha256"]
    assert runtime["object"] in selected["link_objects"]
    assert selected["input_sha256"][runtime["object"]] == runtime["object_sha256"]
    assert all(selected["input_sha256"][p] == h for p, h in runtime["input_sha256"].items())
