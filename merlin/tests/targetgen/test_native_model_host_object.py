"""A compiled host object must retain the selected device-call boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from merlin.targetgen import cli
from merlin.targetgen.contract.model_host_object import (
    _check_call_symbols,
    _check_object_receipt,
    _descriptor_positions,
    _llvm_ciface_pointer_parameters,
    _reduction_initializer_storage,
    _symbols,
)


def _bridge() -> dict:
    return {
        "entry_c_interface_symbol": "work",
        "external_symbols": ["work$kernel_0"],
        "c_interface_symbols": ["work$kernel_0"],
        "required_native_unmet": ["second"],
        "bindings": [
            {
                "region_id": "first", "symbol": "work$kernel_0",
                "native_status": "emitted", "static_inputs_status": "packed_or_runtime",
                "selected_route": "direct_native",
            },
            {
                "region_id": "second", "symbol": "work$kernel_1",
                "native_status": "unsupported_semantics", "static_inputs_status": "not_emitted",
                "selected_route": "retained_host",
            },
        ],
    }


def test_host_object_call_roster_rejects_native_to_host_substitution() -> None:
    rows = """_mlir_ciface_work T 0 20
_mlir_ciface_work$kernel_0 U
work$kernel_1 T 20 10
malloc U
"""
    symbols = _symbols(rows)
    assert _check_call_symbols(_bridge(), symbols) == (["second"], ["malloc"])
    assert _symbols(".L0 t 0 0\n.L0 t 1c 0\n" + rows) == symbols
    for changed, reason in (
        (rows.replace("_mlir_ciface_work$kernel_0 U", "_mlir_ciface_work$kernel_0 T 10 10"),
         "missing or unexpected device calls"),
        (rows.replace("work$kernel_1 T 20 10", "work$kernel_1 U"), "lacks a retained region body"),
        (rows.replace("_mlir_ciface_work T 0 20", "_mlir_ciface_work U"), "defined C-interface entry"),
        (rows + "_mlir_ciface_surprise U\n", "missing or unexpected device calls"),
    ):
        with pytest.raises(ValueError, match=reason):
            _check_call_symbols(_bridge(), _symbols(changed))
    with pytest.raises(ValueError, match="duplicate symbol"):
        _symbols(rows + "malloc U\n")
    stale = {**_bridge(), "required_native_unmet": []}
    with pytest.raises(ValueError, match="required-native failure"):
        _check_call_symbols(stale, symbols)


def test_object_receipt_rejects_stale_llvm_or_object(tmp_path: Path) -> None:
    ll = tmp_path / "model.ll"
    obj = tmp_path / "model.o"
    ll.write_bytes(b"LLVM IR")
    obj.write_bytes(b"object")
    receipt = tmp_path / "invocations" / "codegen" / "invocation.json"
    receipt.parent.mkdir(parents=True)

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "schema": "merlin.invocation_record.v1", "stage": "object", "status": "completed",
        "returncode": 0, "inputs_unchanged": True, "executable_unchanged": True,
        "argv": ["clang", "-c", str(ll), "-o", str(obj)],
        "inputs": [{"path": str(ll), "sha256": sha(ll)}],
        "outputs": [{"path": str(obj), "sha256": sha(obj)}],
    }
    receipt.write_text(json.dumps(record))
    assert _check_object_receipt(tmp_path, ll, obj) == sha(receipt)
    ll.write_bytes(b"changed LLVM IR")
    with pytest.raises(ValueError, match="differs from LLVM IR"):
        _check_object_receipt(tmp_path, ll, obj)
    ll.write_bytes(b"LLVM IR")
    obj.write_bytes(b"stale or changed object")
    with pytest.raises(ValueError, match="differs from LLVM IR"):
        _check_object_receipt(tmp_path, ll, obj)


def test_buffered_entry_appends_result_descriptors() -> None:
    assert _descriptor_positions(
        ("tensor<2xf32>",), ("tensor<3xf32>",),
        ("memref<2xf32>", "memref<3xf32>"),
    )["result_descriptor_positions"] == [1]
    with pytest.raises(ValueError, match="append"):
        _descriptor_positions(
            ("tensor<2xf32>",), ("tensor<3xf32>",),
            ("memref<3xf32>", "memref<2xf32>"),
        )


def test_llvm_c_interface_reader_requires_one_exact_entry() -> None:
    ir = b'define void @"_mlir_ciface_forward"(ptr %0, ptr %1) {\n}\n'
    assert _llvm_ciface_pointer_parameters(ir, "forward") == ["ptr %0", "ptr %1"]
    with pytest.raises(ValueError, match="no unique definition"):
        _llvm_ciface_pointer_parameters(ir + ir, "forward")
    with pytest.raises(ValueError, match="no unique definition"):
        _llvm_ciface_pointer_parameters(ir, "other")


def test_host_reduction_initializer_storage_tracks_source_alias_risk() -> None:
    from merlin.frontends.linalg_mlir import parse_mlir_text

    source = """
module {
  func.func @forward(%data: tensor<2x2xf32>, %external: tensor<2xf32>) -> tensor<2xf32> {
    %zero = arith.constant 0.0 : f32
    %fresh = tensor.splat %zero : tensor<2xf32>
    %result = linalg.reduce ins(%data : tensor<2x2xf32>) outs(%fresh : tensor<2xf32>) dimensions = [1]
      (%value: f32, %acc: f32) {
        %sum = arith.addf %value, %acc : f32
        linalg.yield %sum : f32
      }
    return %result : tensor<2xf32>
  }
}
"""

    def reduction(text: str):
        return next(op for op in parse_mlir_text(text).walk() if op.name == "linalg.reduce")

    assert _reduction_initializer_storage(reduction(source)) == ["fresh_single_use_splat"]
    shared = source.replace(
        "outs(%fresh : tensor<2xf32>)", "outs(%external : tensor<2xf32>)"
    )
    assert _reduction_initializer_storage(reduction(shared)) == ["requires_storage_check"]


def test_host_object_cli_requires_fresh_output_and_reports_failure(tmp_path: Path, monkeypatch) -> None:
    from merlin.targetgen.contract import model_host_object

    def checked(**kwargs):
        assert kwargs["probe_dir"] == tmp_path / "probe"
        return {"schema": "merlin.native_host_object_check.v1", "status": "object_symbols_checked"}

    monkeypatch.setattr(model_host_object, "check_compiled_host_object", checked)
    args = argparse.Namespace(
        probe=str(tmp_path / "probe"), compile_inputs=str(tmp_path / "inputs"),
        lowered=str(tmp_path / "lowered"), object=str(tmp_path / "lowered/model.o"),
        nm=str(tmp_path / "nm"), mlir_opt=str(tmp_path / "mlir-opt"),
        out=str(tmp_path / "checked"),
    )
    assert cli._cmd_native_check_host_object(args) == 0
    receipt = tmp_path / "checked/manifest.json"
    original = receipt.read_bytes()
    assert cli._cmd_native_check_host_object(args) == 2
    assert receipt.read_bytes() == original

    def refused(**kwargs):
        raise ValueError("stale source identity")

    monkeypatch.setattr(model_host_object, "check_compiled_host_object", refused)
    args.out = str(tmp_path / "refused")
    assert cli._cmd_native_check_host_object(args) == 2
    assert json.loads((tmp_path / "refused/manifest.json").read_text()) == {
        "schema": "merlin.native_host_object_check.v1",
        "status": "compile_error",
        "reason": "stale source identity",
    }
