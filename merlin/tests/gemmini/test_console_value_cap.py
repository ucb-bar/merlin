"""An output above the declared console cap is printed as a digest the host can recompute."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.runtime import commandbuffer as CB
from merlin.runtime.backends import base as bk
from merlin.targetgen.contract.build_service import load_build_package
from merlin.targetgen.target_registry import explicit_targets


def _package():
    return load_build_package(explicit_targets()["gemmini"] / "build_support/__init__.py")


def _cb(cap=None):
    tensors = {
        "A": {"shape": [2, 3], "dtype": "i8", "role": "input"},
        "SMALL": {"shape": [1, 2], "dtype": "i8", "role": "output"},
        "LARGE": {"shape": [4, 8], "dtype": "i8", "role": "output"},
    }
    cb = {
        "abi_version": "0.1",
        "target": "gemmini",
        "commands": [],
        "tensors": tensors,
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": name, "access": "read" if name == "A" else "write"} for name in tensors],
            "outputs": ["SMALL", "LARGE"],
        },
    }
    if cap is not None:
        cb["params"] = {CB.CONSOLE_VALUE_CAP_PARAM: cap}
    return cb


def _render(cb):
    from merlin.runtime.tensor import Tensor

    backend = bk.get_backend("gemmini")
    inputs = {"A": Tensor((2, 3), [1, 2, 3, 4, 5, 6], "i8")}
    return _package().render_whole_program(cb, inputs=inputs, legacy_dim=backend.gemmini_codegen.DIM)


def test_only_an_output_above_the_cap_is_digested() -> None:
    source = _render(_cb(cap=8))
    assert 'printf("OUT SMALL 1 2");' in source and "OUT LARGE" not in source
    assert f'printf("{CB.OUTPUT_DIGEST_LINE} LARGE 4 8 %016llx\\n", merlin_outsum);' in source
    assert "MERLIN_OUTSUM_ADD(" in source and "sprintf(b_, __VA_ARGS__)" in source
    # No cap declared: every output prints its values, and no digest machinery is emitted at all.
    plain = _render(_cb())
    assert "OUT LARGE 4 8" in plain and "merlin_outsum" not in plain


def test_a_cap_that_is_not_a_positive_integer_is_refused() -> None:
    for bad in (0, -1, True, "8"):
        with pytest.raises(_package().CodegenError, match=CB.CONSOLE_VALUE_CAP_PARAM):
            _render(_cb(cap=bad))


def test_the_host_recomputes_the_line_and_one_changed_value_changes_it() -> None:
    rows = [[-128, -1, 0, 1], [127, 5, -7, 100000]]
    line = bk.output_digest_line("Y1", rows)
    assert bk.parse_console_digests(f"noise\n{line}\nDONE\n") == {
        "Y1": {"rows": 2, "cols": 4, "digest": line.split()[-1]}
    }
    changed = [list(rows[0]), [127, 5, -7, 100001]]
    assert bk.output_digest_line("Y1", changed) != line
    # Order is part of the value: a transposed readback is a different output.
    assert bk.output_text_digest([1, 2]) != bk.output_text_digest([2, 1])


@pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")
def test_the_c_macro_and_the_host_digest_are_one_function(tmp_path) -> None:
    # The harness digests the text its own printf would have produced. Compile that exact macro
    # with the host compiler and compare with the Python definition, negatives included.
    values = [-128, -1, 0, 1, 127, 4096, -70000, 2147483647]
    source = _render(_cb(cap=8))
    macro = next(line for line in source.splitlines() if line.startswith("#define MERLIN_OUTSUM_ADD"))
    program = tmp_path / "digest.c"
    program.write_text(
        "#include <stdio.h>\nstatic unsigned long long merlin_outsum;\n" + macro + "\n"
        "int main(void) { int v[] = {" + ", ".join(str(v) for v in values) + "};\n"
        "  merlin_outsum = 1469598103934665603ULL;\n"
        f'  for (int i = 0; i < {len(values)}; i++) MERLIN_OUTSUM_ADD(" %d", (int)v[i]);\n'
        '  printf("%016llx\\n", merlin_outsum); return 0; }\n',
        encoding="utf-8",
    )
    binary = tmp_path / "digest"
    subprocess.run(["cc", "-O1", "-o", str(binary), str(program)], check=True, capture_output=True)
    printed = subprocess.run([str(binary)], check=True, capture_output=True, text=True).stdout.strip()
    assert printed == bk.output_text_digest(values)
