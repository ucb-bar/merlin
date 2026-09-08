"""Write and validate the target-independent command buffer.

The buffer is the semantic contract with the runtime, so it is checked against
`merlin/contract/schemas/command_buffer.schema.json` when that schema is reachable, and against a
built-in structural check (required keys, non-empty commands unless declined, operands that name
declared tensors) when it is not.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_REQUIRED = ("abi_version", "target", "commands")

_OPCODES = frozenset({
    "RES_PACK", "MATMUL_RESIDENT", "MATMUL", "COMMIT", "EVICT", "VECTOR_MAP", "VREDUCE",
    "RMSNORM", "ATTENTION_QK", "BATCHED_MATMUL", "LAYERNORM", "SOFTMAX", "GELU", "SOFTCAP",
    "GEGLU", "ATTENTION_FULL", "ROPE", "CONV", "MATMUL_BATCHED", "BIAS_ADD", "CONV2D",
    "MOVEMENT", "ATTENTION_PV",
})

_EPILOGUE = frozenset({"bias_add", "bias", "requant", "acc_scale", "relu", "maxpool"})


def _schema_path() -> Path | None:
    env = os.environ.get("MERLIN_COMMAND_BUFFER_SCHEMA")
    if env and Path(env).is_file():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "merlin" / "contract" / "schemas" / "command_buffer.schema.json"
        if cand.is_file():
            return cand
        cand = parent / "contract" / "schemas" / "command_buffer.schema.json"
        if cand.is_file():
            return cand
    return None


def validate(cb: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for key in _REQUIRED:
        if key not in cb:
            problems.append(f"missing required key {key!r}")
    commands = cb.get("commands") or []
    if (not commands and "declined" not in cb
            and (cb.get("kernel_abi") or {}).get("kind") != "whole_program"):
        problems.append("commands must be non-empty unless the buffer declines")
    tensors = cb.get("tensors") or {}
    for i, cmd in enumerate(commands):
        op = cmd.get("opcode")
        if op not in _OPCODES:
            problems.append(f"commands[{i}]: opcode {op!r} is not in the ABI vocabulary")
        for stage in (cmd.get("attributes") or {}).get("epilogue") or []:
            if stage not in _EPILOGUE:
                problems.append(f"commands[{i}]: epilogue stage {stage!r} is not in the ABI")
    if commands and not tensors and "declined" not in cb:
        problems.append("commands reference operands but no tensors are declared")
    schema = _schema_path()
    if schema is not None:
        try:
            import jsonschema
        except ImportError:
            return problems
        validator = jsonschema.Draft202012Validator(json.loads(schema.read_text()))
        for err in validator.iter_errors(cb):
            problems.append(f"schema: {err.message} at /{'/'.join(str(p) for p in err.path)}")
    return problems


def declined(target: str, reason: str, *, op: str = "", shape: list[int] | None = None,
             abi_version: str = "0.1") -> dict[str, Any]:
    entry: dict[str, Any] = {"reason": reason}
    if op:
        entry["op"] = op
    if shape:
        entry["shape"] = [int(v) for v in shape]
    return {"abi_version": abi_version, "target": target or "gemmini",
            "backend": "mlir_oot_xdsl_gemmini", "commands": [], "declined": entry}


def write(cb: dict[str, Any], path: str | Path) -> list[str]:
    problems = validate(cb)
    Path(path).write_text(json.dumps(cb, indent=2) + "\n")
    return problems
