"""Output plumbing: build a SCHEMA-VALID command_buffer.json (the frozen ABI).

Target-agnostic: the command-buffer schema is fixed by the bench contract, identical for every target.
This builder removes the hand-rolled serializer (~80 LOC) AND the `command_buffer_schema` failure plane —
it validates against `bench_contract/schemas/command_buffer.schema.json` before writing. It does NOT pick
opcodes or operands for you (that's the agent's target lowering); it only guarantees a well-formed buffer.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[5]
_SCHEMA = _REPO / "bench_contract" / "schemas" / "command_buffer.schema.json"


class CommandBufferBuilder:
    """Accumulate tensors + commands, then .to_dict()/.write() a schema-valid command buffer.

    The OPCODES and OPERANDS you add are your target-lowering decisions — the builder just structures and
    validates them. (No target opcode set is baked in.)
    """

    def __init__(self, target: str, backend: str = "", abi_version: str = "0.1"):
        self._cb: dict[str, Any] = {"abi_version": abi_version, "target": target,
                                    "tensors": {}, "commands": []}
        if backend:
            self._cb["backend"] = backend

    def tensor(self, name: str, shape: list[int], dtype: str, role: str | None = None) -> "CommandBufferBuilder":
        spec: dict[str, Any] = {"shape": list(shape), "dtype": dtype}
        if role:
            spec["role"] = role
        self._cb["tensors"][name] = spec
        return self

    def command(self, opcode: str, operands: dict | None = None,
                attributes: dict | None = None) -> "CommandBufferBuilder":
        c: dict[str, Any] = {"opcode": opcode}
        if operands:
            c["operands"] = operands
        if attributes:
            c["attributes"] = attributes
        self._cb["commands"].append(c)
        return self

    def params(self, **kw) -> "CommandBufferBuilder":
        self._cb.setdefault("params", {}).update(kw)
        return self

    def validate(self) -> list[str]:
        """Return a list of schema problems (empty = valid). Uses jsonschema if available, else a
        minimal required-field check so it works in any environment."""
        problems: list[str] = []
        if not self._cb.get("commands"):
            problems.append("commands: must be non-empty")
        for k in ("abi_version", "target", "commands"):
            if k not in self._cb:
                problems.append(f"missing required top-level field: {k}")
        for i, c in enumerate(self._cb["commands"]):
            if "opcode" not in c:
                problems.append(f"commands[{i}]: missing 'opcode'")
        try:
            import jsonschema  # optional, stronger
            schema = json.loads(_SCHEMA.read_text())
            for e in jsonschema.Draft7Validator(schema).iter_errors(self._cb):
                problems.append(f"schema: {e.message} at {'/'.join(map(str, e.path))}")
        except ImportError:
            pass
        return problems

    def to_dict(self) -> dict[str, Any]:
        probs = self.validate()
        if probs:
            raise ValueError("command buffer is not schema-valid: " + "; ".join(probs[:5]))
        return self._cb

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
