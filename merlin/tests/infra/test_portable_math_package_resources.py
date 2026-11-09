"""The public resource closure supports portable math without a checkout."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def test_packaged_portable_math_headers_compile_without_source_search_paths(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required")
    root = repo_root()
    files = json.loads((root / "build_tools/package_resources.json").read_text())["files"]
    bundled = tmp_path / "installed_data"
    for name in files:
        if name.startswith("merlin/runtime/abi/"):
            destination = bundled / Path(name).relative_to("merlin")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
    program = tmp_path / "portable.c"
    program.write_text(
        '#define MERLIN_PORTABLE_MATH_IMPLEMENTATION\n#include "portable_math.h"\n#include <assert.h>\n'
        "int main(void){volatile float zero=0.0f,one=1.0f;"
        "assert(merlin_expf(zero)==1.0f);assert(merlin_erff(zero)==0.0f);"
        "assert(merlin_tanhf(zero)==0.0f);assert(merlin_logf(one)==0.0f);return 0;}\n"
    )
    executable = tmp_path / "portable"
    subprocess.run(
        [compiler, "-std=c11", "-I", str(bundled / "runtime/abi"), str(program), "-lm", "-o", str(executable)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, timeout=10)
