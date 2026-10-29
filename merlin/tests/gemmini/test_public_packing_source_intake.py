"""Opt-in public V26 source intake; no linked image or execution qualification.

Select the exact public 16-file compiler with MERLIN_TEST_GEMMINI_V26_PACKAGE.
Only its declared parse command runs. Target-IR consumer permission, original
capture, live linked-ELF admission and runtime qualification remain separate.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from merlin.common.digest import sha256_file
from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.automatic_destinations import prepare_automatic_destinations
from merlin.targetgen import package_runtime

pytestmark = pytest.mark.target("gemmini")
_PACKAGE = "MERLIN_TEST_GEMMINI_V26_PACKAGE"
_ORIGINAL = "29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24"


def _public_tree(root):
    """Reopen the complete original submission, including its two author reports."""
    digest = hashlib.sha256()
    members = []
    for path in sorted(root.rglob("*")):
        assert not path.is_symlink(), "public compiler tree must not contain symlinks"
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode() + b"\0" + path.read_bytes() + b"\0")
            members.append(relative)
    assert len(members) == 16 and digest.hexdigest() == _ORIGINAL
    return digest.hexdigest()


def _abi(source):
    from xdsl.dialects import func

    module = parse_mlir_text(source)
    module.verify()
    entries = [op for op in module.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "forward"]
    assert len(entries) == 1
    return entries[0].function_type


def _invocation(directory, package, source, result):
    from merlin.common import invocation_record

    receipts = list(directory.rglob("invocation.json"))
    assert len(receipts) == 1
    path = receipts[0]
    if result.returncode == 0:
        record = invocation_record.verify(path)
    else:
        record = json.loads(path.read_text())
        with pytest.raises(ValueError, match="did not complete"):
            invocation_record.verify(path)
        for pin in (
            record["executable"],
            record["stdout"],
            record["stderr"],
            *record["inputs"],
            *record["dependencies"],
        ):
            assert pin["sha256"] == sha256_file(Path(pin["path"]))
    assert record["stage"] == "parse" and record["kind"] == "subprocess"
    assert record["cwd"] == str(package.directory)
    assert record["returncode"] == result.returncode
    assert record["inputs"] == [{"path": str(source), "sha256": sha256_file(source)}]
    assert all(record[name] is True for name in ("inputs_unchanged", "dependencies_unchanged", "executable_unchanged"))
    assert record["executable"]["path"] == str(Path(sys.executable).resolve())
    assert {"path": str(package.tool), "sha256": sha256_file(package.tool)} in record["dependencies"]
    expected = [
        token.replace("{tool}", str(package.tool)).replace("{input_mlir}", str(source))
        for token in package.manifest["commands"]["parse"]["argv"]
    ]
    assert record["argv"] == expected
    return {"path": str(path), "sha256": sha256_file(path), "executable": record["executable"]}


def test_original_public_parser_consumes_both_connected_packing_sources(tmp_path, monkeypatch, record_property):
    selected = os.environ.get(_PACKAGE)
    if not selected:
        pytest.skip(f"set {_PACKAGE}; absence establishes no compiler or target qualification")
    root = Path(selected).resolve(strict=True)
    identity = _public_tree(root)
    package = package_runtime.load_package(root)
    assert "target_ir_consumer" not in package.manifest
    assert "emit_target_bundle" not in package.manifest["commands"]
    # Select the test's existing interpreter for the declared python3 command;
    # this supplies no new package source, runtime or dependency qualification.
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    fixture = repo_root() / "merlin/tests/ir/producer_packing_connected.mlir"
    original = fixture.read_text()
    candidate, report = prepare_automatic_destinations(original, serial=True, permuted_outputs=True, index_bits=64)
    assert report["rewrites"] == 1 and original != candidate
    assert _abi(original) == _abi(candidate)
    observed = []
    for label, source in (("baseline", original), ("candidate", candidate)):
        path = tmp_path / (label + ".mlir")
        path.write_text(source)
        directory = tmp_path / (label + "-invocation")
        result = package_runtime.run_entrypoint(package, "parse", path, timeout=30, invocation_directory=directory)
        assert result.returncode == 0, result.stderr
        assert _public_tree(root) == identity
        observed.append(
            {
                "arm": label,
                "source_sha256": sha256_file(path),
                "returncode": result.returncode,
                "invocation": _invocation(directory, package, path, result),
            }
        )
    # An unconditional successful command is not evidence of source intake.
    invalid = tmp_path / "invalid.mlir"
    invalid.write_text("this is not MLIR\n")
    refusal = package_runtime.run_entrypoint(
        package, "parse", invalid, timeout=30, invocation_directory=tmp_path / "invalid-invocation"
    )
    assert refusal.returncode != 0 and _public_tree(root) == identity
    rejected = _invocation(tmp_path / "invalid-invocation", package, invalid, refusal)
    record_property(
        "public_source_intake",
        json.dumps(
            {
                "compiler_sha256": identity,
                "inputs": observed,
                "invalid_source_invocation": rejected,
                "rewrite_count": report["rewrites"],
                "scope": (
                    "original declared parser only; no capture, emission, linked-ELF admission, execution or timing"
                ),
            },
            sort_keys=True,
        ),
    )
