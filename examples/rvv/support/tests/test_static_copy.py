"""Admission/refusal and native verification for explicit static copy loops."""

from __future__ import annotations

import json
import os
import subprocess

import pytest
from test_backend import BACKEND

API = BACKEND.copy_loops_api()
COMPILER = os.environ.get("RVV_TEST_COMPILER_PYTHON")


def rewrite(tmp_path, source, *, max_rank=4):
    if not COMPILER:
        pytest.skip(
            "set RVV_TEST_COMPILER_PYTHON to the selected torch_mlir interpreter"
        )
    inp, out, report, script = (
        tmp_path / name for name in ("in.mlir", "out.mlir", "report.json", "run.py")
    )
    inp.write_text(source)
    script.write_text(API.RUNNER)
    result = subprocess.run(
        [COMPILER, str(script), str(inp), str(out), str(report), str(max_rank)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    return out.read_text(), json.loads(report.read_text())


def copy_module(source_type, destination_type=None):
    destination_type = destination_type or source_type
    return f"""module {{ func.func @copy(%s: {source_type}, %d: {destination_type}) {{
        memref.copy %s, %d : {source_type} to {destination_type}
        return
    }} }}"""


@pytest.mark.parametrize("width", [8, 16, 32, 64])
def test_integer_strided_copy_rewrites_without_new_allocation_or_alias_facts(
    tmp_path, width
):
    typ = f"memref<3x7xi{width}, strided<[?, ?], offset: ?>>"
    text, report = rewrite(tmp_path, copy_module(typ))
    assert report["rewritten"] == 1
    assert report["copies"][0]["iteration_order"] == "lexicographic_sequential"
    assert report["copies"][0]["allocations_added"] == 0
    assert not report["copies"][0]["alias_assumptions_added"]
    assert "memref.copy" not in text and text.count("scf.for") == 2
    assert "memref.alloc" not in text
    assert report["compiler_binding_sha256"]


@pytest.mark.parametrize(
    "typ,reason",
    [
        ("memref<?x7xi32>", "dynamic_shape"),
        ("memref<*xi32>", "unranked_memref"),
        ("memref<3x7xf32, strided<[9, 1]>>", "unsupported_element_type"),
        ("memref<3x7xi32, strided<[9, 1]>, 3>", "non_cpu_memory_space"),
        ("memref<2x2x2x2x2xi8, strided<[48, 20, 8, 3, 1]>>", "rank_budget"),
        ("memref<3x7xi32>", "contiguous_upstream_fastpath"),
    ],
)
def test_unsupported_copy_keeps_normal_transport(tmp_path, typ, reason):
    text, report = rewrite(tmp_path, copy_module(typ))
    assert report["rewritten"] == 0
    assert report["refusals"] == {reason: 1}
    assert "memref.copy" in text


def test_dynamic_offsets_negative_stride_and_zero_extent_keep_descriptors(tmp_path):
    typ = "memref<0x7xi32, strided<[-9, 1], offset: ?>>"
    text, report = rewrite(tmp_path, copy_module(typ))
    assert report["rewritten"] == 1
    assert report["copies"][0]["source_strides"] == [-9, 1]
    assert "strided<[-9, 1], offset: ?>" in text


def test_pipeline_split_preserves_every_other_stage():
    pipeline = "canonicalize,func.func(convert-linalg-to-loops),convert-scf-to-cf,expand-strided-metadata,canonicalize"
    before, after = API.split_pipeline(pipeline)
    assert before + "," + after == pipeline


@pytest.mark.parametrize(
    "pipeline", ["canonicalize", "convert-scf-to-cf,x,convert-scf-to-cf,x"]
)
def test_missing_or_ambiguous_boundary_refuses(pipeline):
    with pytest.raises(ValueError):
        API.split_pipeline(pipeline)


def test_existing_output_is_unchanged(tmp_path):
    work = tmp_path / "old"
    work.mkdir()
    (work / "sentinel").write_bytes(b"immutable")
    with pytest.raises(FileExistsError):
        API.lower_file(
            __file__,
            pipeline="unused",
            mlir_opt=__file__,
            compiler_python=__file__,
            work=work,
        )
    assert list(work.iterdir()) == [work / "sentinel"]
    assert (work / "sentinel").read_bytes() == b"immutable"


@pytest.mark.parametrize("max_rank", [0, 9, True, 1.5])
def test_invalid_budget_refuses_before_creating_output(tmp_path, max_rank):
    work = tmp_path / "new"
    with pytest.raises(ValueError):
        API.lower_file(
            __file__,
            pipeline="unused",
            mlir_opt=__file__,
            compiler_python=__file__,
            work=work,
            max_rank=max_rank,
        )
    assert not work.exists()
