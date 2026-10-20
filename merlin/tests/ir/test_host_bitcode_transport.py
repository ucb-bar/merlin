"""Actual whole-module assembly transport retains constants and callable results."""

import ctypes
import shutil
from pathlib import Path

import pytest

from merlin.common.proc import run_checked
from merlin.llvmlower import toolchain
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.llvmlower.host_bitcode_transport import assemble_host_bitcode


def tools():
    selected = shutil.which(str(toolchain.clang()))
    if selected is None:
        pytest.skip("selected LLVM toolchain unavailable")
    compiler = Path(selected).resolve()
    if not compiler.with_name("llvm-as").is_file() or not compiler.with_name("llvm-dis").is_file():
        pytest.skip("paired LLVM assembly tools unavailable")
    return compiler


def test_real_bitcode_transport_preserves_constant_and_function(tmp_path):
    compiler = tools()
    source = tmp_path / "module.ll"
    text = ('@payload = constant [4 x i8] c"ABC\\00"\n'
            'define i32 @value() {\nentry:\n %p = getelementptr [4 x i8], ptr @payload, i64 0, i64 1\n'
            ' %v = load i8, ptr %p\n %r = zext i8 %v to i32\n ret i32 %r\n}\n')
    source.write_text(text)
    recipe = CompilationRecipe(tmp_path, producer=Path(toolchain.__file__))
    bitcode, observation = assemble_host_bitcode(
        source, tmp_path / "transport", compiler=compiler, compilation=recipe, runner=run_checked,
    )
    assert source.read_text() == text
    assert observation["source"]["path"] == str(source)
    assert observation["bitcode"]["path"] == str(bitcode)
    restored = tmp_path / "restored.ll"
    run_checked([compiler.with_name("llvm-dis"), bitcode, "-o", restored])
    assert '@payload = constant [4 x i8] c"ABC\\00"' in restored.read_text()
    library = tmp_path / "module.so"
    recipe.run([compiler, "-shared", "-fPIC", bitcode, "-o", library],
               runner=run_checked, inputs=[bitcode], output=library)
    recipe.completed(library)
    loaded = ctypes.CDLL(str(library))
    loaded.value.restype = ctypes.c_int
    assert loaded.value() == 66


def test_changed_source_after_actual_assembly_refuses(tmp_path):
    compiler = tools()
    source = tmp_path / "module.ll"
    source.write_text('define i32 @value() { ret i32 42 }\n')
    recipe = CompilationRecipe(tmp_path, producer=Path(toolchain.__file__))

    def changing_runner(command):
        result = run_checked(command)
        source.write_text('define i32 @value() { ret i32 43 }\n')
        return result

    with pytest.raises(ValueError, match="source changed"):
        assemble_host_bitcode(source, tmp_path / "transport", compiler=compiler,
                              compilation=recipe, runner=changing_runner)
