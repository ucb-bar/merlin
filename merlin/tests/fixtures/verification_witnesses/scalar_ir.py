from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.perf.whole_model_chunks import chunk_forward
from merlin.verification.scalar_ir import ScalarTransformReceipt, qualify_scalar_receipt, verify_scalar_transform
from merlin.verify import HAS_Z3

SOURCE = "module {\n  func.func @forward(%x: i32) -> i32 {\n    %a = arith.addi %x, %x : i32\n    %b = arith.muli %a, %x : i32\n    %c = arith.subi %b, %a : i32\n    %d = arith.addi %c, %x : i32\n    func.return %d : i32\n  }\n}\n"

# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "HAS_Z3",
    "SOURCE",
    "ScalarTransformReceipt",
    "chunk_forward",
    "mq",
    "pytest",
    "qualify_scalar_receipt",
    "verify_scalar_transform",
]
