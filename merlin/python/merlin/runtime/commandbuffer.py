"""Load/validate a Merlin command buffer and materialize its declared tensors.

The execution-oriented command buffer carries a ``tensors`` table (name -> shape/dtype/role)
in addition to the opcode list. Input/weight/bias tensors are materialized deterministically
(see :func:`Tensor.deterministic`) so a run is reproducible without external input files; an
explicit inputs mapping can override them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .tensor import Tensor


def load_command_buffer(path: str | Path) -> dict[str, Any]:
    """Load a command-buffer JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


REQUIRED_KEYS = ("abi_version", "target", "commands")


def validate_command_buffer(cb: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty == valid)."""
    problems: list[str] = []
    for k in REQUIRED_KEYS:
        if k not in cb:
            problems.append(f"missing key '{k}'")
    for i, cmd in enumerate(cb.get("commands", [])):
        if "opcode" not in cmd:
            problems.append(f"command {i} missing 'opcode'")
    return problems


def materialize_inputs(cb: dict[str, Any], inputs: dict[str, Any] | None = None) -> dict[str, Tensor]:
    """Create the leaf input tensors declared in the command buffer's ``tensors`` table.

    A tensor is a *leaf* (materialized here) when its role is input/weight/bias, i.e. it is
    not produced by a command. ``inputs`` may supply explicit nested-list data per name.
    """
    inputs = inputs or {}
    produced = set()
    for cmd in cb.get("commands", []):
        ops = cmd.get("operands", {})
        for key in ("dst",):
            if key in ops:
                produced.add(ops[key])
    env: dict[str, Tensor] = {}
    for name, spec in cb.get("tensors", {}).items():
        if name in produced:
            continue
        shape = tuple(spec["shape"])
        dtype = spec.get("dtype", "i8")
        if name in inputs:
            flat = _flatten(inputs[name])
            env[name] = Tensor(shape, flat, dtype)
        else:
            env[name] = Tensor.deterministic(name, shape, dtype)
    return env


def _flatten(nested) -> list[int]:
    out: list[int] = []
    if nested and isinstance(nested[0], list):
        for row in nested:
            out.extend(row)
    else:
        out.extend(nested)
    return out
