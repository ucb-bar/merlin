"""Assemble a stdlib-only scheduling artifact from the actual editable owners.

Generated packages may vendor this derived module without depending on Merlin.
The artifact contains the original definitions, with only internal imports and
owner docstrings removed. Source hashes identify every contributing owner;
the generated file is not a second editable implementation.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

OWNERS = ("model", "residency", "block", "convolution", "execute", "readout", "interface")


def block_schedule_source() -> bytes:
    """Return one closed, inspectable module with no Merlin runtime imports."""
    pieces = [
        '"""Generated shared block schedule; edit the recorded source owners."""\n',
        "from __future__ import annotations\n",
        "from dataclasses import dataclass, field, replace\n",
        "from typing import Any\n",
    ]
    for owner in OWNERS:
        source = Path(__file__).with_name(owner + ".py").read_bytes()
        text = source.decode("utf-8")
        lines = text.splitlines(keepends=True)
        pieces.append(f"\n# Owner: merlin.compiler.schedule.{owner}; sha256={hashlib.sha256(source).hexdigest()}\n")
        for node in ast.parse(text).body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            first = min([node.lineno, *(item.lineno for item in getattr(node, "decorator_list", ()))])
            pieces.append("\n" + "".join(lines[first - 1 : node.end_lineno]) + "\n")
    return "".join(pieces).encode("utf-8")
