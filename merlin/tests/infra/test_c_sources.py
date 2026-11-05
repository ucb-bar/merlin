"""The exported C/H pair compiles without the source checkout's include paths."""

import shutil
import subprocess

import pytest

from merlin.compile.c_sources import c_source_pair


def test_standalone_pair_compiles_and_runs_without_original_files(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    files = c_source_pair(
        "model",
        public_headers={"api.h": "int compute(int x);"},
        private_headers={"detail.h": '#include "api.h"\nint helper(int x);'},
        sources={
            "helper.c": '#include "detail.h"\nint helper(int x) { return x * 3; }',
            "entry.c": '#include "detail.h"\nint compute(int x) { return helper(x) + 1; }',
        },
    )
    assert set(files) == {"model.c", "model.h"}
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    (tmp_path / "main.c").write_text('#include "model.h"\nint main(void) { return compute(7) != 22; }')
    subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "model.c", "main.c", "-o", "check"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True)


@pytest.mark.parametrize("include", ['"missing.h"', '"../secret.h"', "HEADER_MACRO"])
def test_incomplete_includes_are_refused(include):
    with pytest.raises(ValueError, match="include"):
        c_source_pair("model", public_headers={}, private_headers={}, sources={"body.c": f"#include {include}"})


def test_header_order_and_duplicate_members_are_checked():
    with pytest.raises(ValueError, match="out-of-order"):
        c_source_pair(
            "model",
            public_headers={"a.h": '#include "b.h"', "b.h": "int x;"},
            private_headers={},
            sources={"body.c": "int y;"},
        )
    with pytest.raises(ValueError, match="distinct"):
        c_source_pair(
            "model", public_headers={"x.h": "int x;"}, private_headers={"x.h": "int x;"}, sources={"body.c": "int y;"}
        )
