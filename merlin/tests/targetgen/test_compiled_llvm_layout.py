"""Actual selected object-driver layout observations, without ABI authority."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import invocation_record as I
from merlin.llvmlower.compiled_layout_query import bounded_mir_module
from merlin.llvmlower.layout_observation import observe_compiled_layout


@pytest.fixture(scope="module")
def tools():
    names = (
        "MERLIN_TEST_LLVM_LLC",
        "MERLIN_TEST_NATIVE_CXX",
        "MERLIN_TEST_LLVM_CONFIG",
        "MERLIN_TEST_LAYOUT_SOURCE_ROOT",
    )
    if any(not os.environ.get(name) for name in names):
        pytest.skip("selected LLVM object, native API tools and original public source products are required")
    result = {name: Path(os.environ[name]).absolute() for name in names}
    assert all(path.is_file() for name, path in result.items() if name != "MERLIN_TEST_LAYOUT_SOURCE_ROOT")
    return result


def object_product(tmp_path, tools, case="round_tail", extra=()):
    original = tools["MERLIN_TEST_LAYOUT_SOURCE_ROOT"] / case / "ordinary/model.ll"
    source, obj = tmp_path / "original.ll", tmp_path / "ordinary.o"
    source.write_bytes(original.read_bytes())
    environment = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
    (tmp_path / "environment.json").write_text(json.dumps(environment, sort_keys=True) + "\n")
    result = I.run(
        [
            str(tools["MERLIN_TEST_LLVM_LLC"]),
            "-O2",
            "-filetype=obj",
            "-relocation-model=pic",
            *extra,
            str(source),
            "-o",
            str(obj),
        ],
        directory=tmp_path,
        stage="object",
        cwd=tmp_path,
        env=environment,
        inputs=(source,),
        dependencies=(original,),
        outputs=(obj,),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
    record = next(tmp_path.rglob("invocation.json"))
    I.require_environment(record, environment=environment)
    assert obj.read_bytes().startswith(b"\x7fELF")
    return source, obj, record, environment


def observe(tmp_path, tools, record, environment, *, limit=100000):
    return observe_compiled_layout(
        object_record=record,
        integer_bits=64,
        native_compiler=tools["MERLIN_TEST_NATIVE_CXX"],
        llvm_config=tools["MERLIN_TEST_LLVM_CONFIG"],
        output_root=tmp_path / "layout",
        timeout_s=120,
        environment=environment,
        max_observation_bytes=limit,
    )


@pytest.mark.parametrize(
    ("case", "defect"),
    (("round_tail", "mir"), ("scalar_integer", "stale_ir"), ("round_tail", "missing_ir")),
)
def test_actual_ordinary_object_layout_is_queried_not_guessed(tmp_path, tools, case, defect):
    source, obj, record, environment = object_product(tmp_path, tools, case)
    before = source.read_bytes(), obj.read_bytes()
    result = observe(tmp_path, tools, record, environment)
    assert result.verify()["object_record"][0] == str(record)
    text = (result.root / "selected.ll").read_text()
    assert 'target datalayout = "' in text and 'target triple = "' in text
    assert "target datalayout" not in source.read_text()
    assert "_mlir_ciface_forward" in text and "load {" in text
    # Width/endian/stride values come solely from the compiled public accessor.
    query = result.root / "native/result.txt"
    assert query.read_text() == (
        f"{result.pointer_bits} {result.index_bits} {result.allocation_stride} "
        f"{result.abi_alignment} {result.byte_order}\n"
    )
    assert (source.read_bytes(), obj.read_bytes()) == before
    for selected, _ in result.records:
        actual = I.require_environment(Path(selected), environment=environment)
        if actual["stage"] == "layout_selected_object_compiler":
            assert actual["cwd"] == I.verify(record)["cwd"]
    pins = {path for path, _ in result.source_pins}
    assert str(result.root / "selected.mir") in pins
    with pytest.raises(ValueError, match="fresh actual"):
        replace(result).verify()
    selected = result.root / ("selected.mir" if defect == "mir" else "selected.ll")
    if defect == "missing_ir":
        selected.unlink()
    else:
        selected.write_bytes(b"changed actual driver observation\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        result.verify()


@pytest.mark.parametrize("defect", ("source", "object", "record", "environment", "missing_environment", "byte_budget"))
def test_actual_object_and_environment_drift_or_incomplete_query_refuses(tmp_path, tools, defect):
    source, obj, record, environment = object_product(tmp_path, tools)
    if defect in {"source", "object"}:
        (source if defect == "source" else obj).write_bytes(b"genuine changed original product\n")
    elif defect == "record":
        record.unlink()
    elif defect == "environment":
        environment = {**environment, "LANG": "C"}
    elif defect == "missing_environment":
        environment = None
    with pytest.raises((ValueError, OSError)):
        observe(tmp_path, tools, record, environment, limit=1 if defect == "byte_budget" else 100000)
    if defect != "byte_budget":
        assert not (tmp_path / "layout").exists()
    else:
        assert not (tmp_path / "layout/native").exists()


def test_real_supported_llvm_option_is_not_silently_stripped_from_unhandled_profile(tmp_path, tools):
    _, _, record, environment = object_product(tmp_path, tools, extra=("-verify-machineinstrs",))
    with pytest.raises(ValueError, match="unsupported ordinary object producer options"):
        observe(tmp_path, tools, record, environment)
    assert not (tmp_path / "layout").exists()


_MIR = (
    '--- |\n  target datalayout = "e-i64:64"\n'
    '  target triple = "independent-test-triple"\n'
    "  define void @entry() { ret void }\n...\n---\nname: entry\n"
)


@pytest.mark.parametrize(
    "defect",
    ("prefix", "indent", "truncated", "duplicate_layout", "missing_layout", "missing_triple", "empty_layout", "utf8"),
)
def test_bounded_first_embedded_module_refuses_missing_ambiguous_or_malformed_fields(tmp_path, defect):
    mutations = {
        "prefix": _MIR.replace("--- |", "---", 1),
        "indent": _MIR.replace("  target triple", " target triple"),
        "truncated": _MIR.split("...")[0],
        "duplicate_layout": _MIR.replace("  target triple", '  target datalayout = "e"\n  target triple'),
        "missing_layout": _MIR.replace('  target datalayout = "e-i64:64"\n', ""),
        "missing_triple": _MIR.replace('  target triple = "independent-test-triple"\n', ""),
        "empty_layout": _MIR.replace('"e-i64:64"', '""'),
        "utf8": b"\xff",
    }
    path = tmp_path / "observed.mir"
    data = mutations[defect]
    path.write_bytes(data if type(data) is bytes else data.encode())
    with pytest.raises(ValueError):
        bounded_mir_module(path, max_bytes=1000)


def test_embedded_module_reader_uses_explicit_byte_bound_before_text_expansion(tmp_path):
    path = tmp_path / "observed.mir"
    path.write_text(_MIR)
    expected = (
        'target datalayout = "e-i64:64"\ntarget triple = "independent-test-triple"\ndefine void @entry() { ret void }\n'
    )
    assert bounded_mir_module(path, max_bytes=len(path.read_bytes())) == expected
    for bound in (len(path.read_bytes()) - 1, 0, True, 1.0):
        with pytest.raises(ValueError):
            bounded_mir_module(path, max_bytes=bound)
