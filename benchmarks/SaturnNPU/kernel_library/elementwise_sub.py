from __future__ import annotations

from typing import Any

from npu_model.software import Instruction

try:
    from .manifest_loader import load_kernel
except ImportError:
    from manifest_loader import load_kernel


def test() -> list[Instruction[Any]]:
    return load_kernel("elementwise_sub")
