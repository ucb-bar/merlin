"""Serial executor context is explicit and needs no hosted TLS or stdio ABI."""

import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower import toolchain


@pytest.mark.parametrize("macro", ["MERLIN_OWNED_RESULTS_SERIAL_CONTEXT", "MERLIN_OWNED_RESULTS_EXTERNAL_FAILURE"])
@pytest.mark.parametrize("value", [0, 2, -1])
def test_context_and_failure_selections_cannot_be_implicitly_disabled(tmp_path, macro, value):
    done = subprocess.run(
        [str(toolchain.clang()), "-std=c11", f"-D{macro}={value}", "-c",
         str(runtime_dir() / "abi/owned_results.c"), "-o", str(tmp_path / "invalid.o")],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode != 0 and f"{macro} must explicitly equal 1" in done.stderr
    assert not (tmp_path / "invalid.o").exists()


def test_actual_cross_runtime_uses_serial_context_and_explicit_failure_hook(tmp_path):
    from merlin.runtime.backends import spike
    from merlin.runtime.backends.host_compiler_policy import selected_model_compiler_plan

    try:
        gcc = spike.gcc_path()
    except (FileNotFoundError, RuntimeError):
        pytest.skip("existing cross compiler unavailable")
    if not gcc.is_file():
        pytest.skip("existing cross compiler unavailable")
    plan = selected_model_compiler_plan(backend="scalar", cflags_override=None, vlen=None, features=frozenset())
    output = tmp_path / "owned_runtime.o"
    done = subprocess.run(
        [str(gcc), *plan["gcc_cflags"], "-std=c11", "-ffreestanding",
         "-DMERLIN_OWNED_RESULTS_SERIAL_CONTEXT=1", "-DMERLIN_OWNED_RESULTS_EXTERNAL_FAILURE=1",
         "-c", str(runtime_dir() / "abi/owned_results.c"), "-o", str(output)],
        capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0, done.stderr
    symbols = subprocess.run(
        [str(toolchain.readelf()), "-sW", str(output)], capture_output=True, text=True, timeout=10,
    )
    assert symbols.returncode == 0, symbols.stderr
    assert not any(len(fields) >= 8 and fields[3] == "TLS" for fields in map(str.split, symbols.stdout.splitlines()))
    undefined = subprocess.run(
        [str(toolchain.nm()), "-u", str(output)], capture_output=True, text=True, timeout=10,
    )
    assert undefined.returncode == 0, undefined.stderr
    names = {line.split()[-1] for line in undefined.stdout.splitlines() if line.split()}
    assert "merlin_owned_results_failure" in names
    assert names <= {"malloc", "free", "memcpy", "memset", "merlin_owned_results_failure"}
    # Runtime objects alone do not establish final linking, resources or execution.
