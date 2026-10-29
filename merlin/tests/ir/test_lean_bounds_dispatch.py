"""Destination-based bounds dispatch checks deep success and failure paths once per scope."""

import hashlib
import json
import subprocess
import time

import pytest
from test_lean_nested_definedness import _nested_source
from test_lean_reshape_compose import _verify_mlir

from merlin.verify.lean.export import export_file


def _source(depth, case, dirty):
    source = _nested_source(
        depth,
        dirty=dirty,
        read_destination=case in {"initialized_read", "dirty_read"},
        poison_input=case == "poison",
    )
    leaf = f"%sum{depth} = arith.addi %element{depth}, %bias : i32"
    if case == "division":
        source = source.replace(
            leaf, f"%zero = arith.constant 0 : i32\n%sum{depth} = arith.divsi %element{depth}, %zero : i32"
        )
    if case == "captured_bounds":
        source = source.replace("%bias: i32)", "%bias: i32, %payload: tensor<2xi32>)")
        source = source.replace("-> tensor<i32> {", "-> tensor<i32> {\n%position = arith.constant 1 : index", 1)
        source = source.replace(leaf, f"%sum{depth} = tensor.extract %payload[%position] : tensor<2xi32>")
    return source


def _direct_check(tmp_path, lean_tools, source, index_bits):
    program, exported = tmp_path / "source.mlir", tmp_path / "source.json"
    program.write_text(source)
    _verify_mlir(program, lean_tools.m2m_python)
    export_file(program, exported, m2m_python=lean_tools.m2m_python)
    started = time.monotonic()
    # This generous limit catches recursive path multiplication, not small timing changes.
    checked = subprocess.run(
        [str(lean_tools.lean_exe), "check-definedness-extended", str(exported), "forward", str(index_bits)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    result = json.loads(checked.stdout)
    (tmp_path / "execution.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": time.monotonic() - started,
                "timeout_seconds": 30,
                "checker_sha256": hashlib.sha256(lean_tools.lean_exe.read_bytes()).hexdigest(),
                "export_sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
                "exit": checked.returncode,
                "result": result,
                "stderr": checked.stderr,
            },
            indent=2,
        )
    )
    assert result["checker"] == "definedness-extended"
    assert result["theorem"] == "MerlinVerify.Check.DefinedExtended.check_sound"
    assert result["index_bits"] == index_bits
    assert checked.returncode == (0 if result["verdict"] == "verified" else 1), checked.stderr
    return result


@pytest.mark.parametrize("index_bits", [32, 64])
@pytest.mark.parametrize("depth", [3, 24])
@pytest.mark.parametrize(
    "case,dirty,expected",
    [
        ("initialized_read", False, "verified"),
        ("overwrite", True, "verified"),
        ("captured_bounds", False, "verified"),
        ("captured_bounds", True, "verified"),
        ("poison", False, "unsupported"),
        ("poison", True, "unsupported"),
        ("division", False, "unsupported"),
        ("dirty_read", True, "unsupported"),
    ],
)
def test_deep_bounds_dispatch(tmp_path, lean_tools, index_bits, depth, case, dirty, expected):
    result = _direct_check(tmp_path, lean_tools, _source(depth, case, dirty), index_bits)
    assert result["verdict"] == expected, result.get("first_refusal")
