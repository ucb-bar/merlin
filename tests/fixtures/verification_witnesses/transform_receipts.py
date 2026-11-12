from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from merlin.verification import HAS_XDSL, HAS_Z3

_TRANSLATOR = os.environ.get("MERLIN_VERIFY_TRANSLATOR")
pytestmark = pytest.mark.skipif(
    not (HAS_XDSL and HAS_Z3 and _TRANSLATOR and Path(_TRANSLATOR).is_file()),
    reason="requires xdsl, z3, and an explicit MERLIN_VERIFY_TRANSLATOR",
)


@pytest.fixture(autouse=True)
def explicit_test_target(tmp_path, monkeypatch):
    """Select self-contained reference metadata; installed Merlin has no target default."""
    root = tmp_path / "receipt_target"
    contracts = root / "contracts"
    contracts.mkdir(parents=True)
    (contracts / "target_contract.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "toy_npu",
                "features": ["resident_packed_tensor", "accumulator_commit", "command_buffer", "metrics"],
                "runtime": {"backends": ["simulator"], "default_backend": "simulator"},
                "capabilities": {"resident_storage_bytes": 131072},
            }
        )
    )
    (contracts / "dialect_plan.yaml").write_text(
        yaml.safe_dump(
            {
                "target": "toy_npu",
                "dialect_name": "toynpu",
                "types": [{"name": "resident_tensor"}, {"name": "accumulator"}],
                "construction": {"matmul_rhs_typed": True, "matmul_vl_policy": False},
                "runtime_opcodes": {
                    "res_pack": "RES_PACK",
                    "matmul": "MATMUL_RESIDENT",
                    "commit": "COMMIT",
                    "evict": "EVICT",
                    "vector_map": "VECTOR_MAP",
                    "vector_reduce": "VREDUCE",
                },
                "lowering": [
                    {"from": "interface.resident_pack", "to": "toynpu.res_pack"},
                    {"from": "interface.matmul", "to": "toynpu.matmul"},
                    {"from": "interface.commit", "to": "toynpu.commit"},
                    {"from": "interface.resident_evict", "to": "toynpu.evict"},
                    {"from": "interface.vector_map", "to": "toynpu.vector_map"},
                    {"from": "interface.vector_reduce", "to": "toynpu.vector_reduce"},
                ],
            }
        )
    )
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)


def _tool() -> tuple[str, str]:
    assert _TRANSLATOR is not None
    return (_TRANSLATOR, hashlib.sha256(Path(_TRANSLATOR).read_bytes()).hexdigest())


def _source_pair():
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i32
    from xdsl.dialects.linalg import ops as linalg_ops

    from merlin.xdsl_dialects.lowering.pipeline import build_input_module, lower_module

    source = build_input_module(reuse=2, m=2, k=2, n=2)
    func = next((op for op in source.walk() if op.name == "func.func"))
    block = func.body.block
    for matmul in [op for op in block.ops if op.name == "linalg.quantized_matmul"]:
        init = matmul.operands[-1]
        zero = arith.ConstantOp(IntegerAttr(0, i32))
        fill = linalg_ops.FillOp(inputs=(zero.result,), outputs=(init,), res=(matmul.results[0].type,))
        block.insert_op_before(zero, matmul)
        block.insert_op_before(fill, matmul)
        matmul.operands = (*matmul.operands[:-1], fill.results[0])
    result = lower_module(source, target="toy_npu")
    return (result.input_module, result.interface_module)


def _backend_pair():
    from merlin.targetgen.families import DEFAULT_EXAMPLE_TARGET
    from merlin.verification.evaluate import _finish_lowering, _lower_to_interface

    (interface, contract) = _lower_to_interface(2, 2, 2, 2, target=DEFAULT_EXAMPLE_TARGET)
    return (interface, _finish_lowering(interface, contract))


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "HAS_XDSL",
    "HAS_Z3",
    "Path",
    "_TRANSLATOR",
    "_backend_pair",
    "_source_pair",
    "_tool",
    "copy",
    "explicit_test_target",
    "hashlib",
    "json",
    "os",
    "pytest",
    "pytestmark",
    "replace",
    "yaml",
]
