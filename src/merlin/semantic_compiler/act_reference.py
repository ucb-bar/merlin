"""Isolated ACT reference runner and non-executing reader for its Python output.

This is a diagnostic adapter. Only a pinned ACT-generated executable supplied
by the caller can produce a qualified ACT result; plumbing tests may use mocks.
The native selector never imports or calls this module.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ActInstruction:
    name: str
    operands: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class ActAssembly:
    kernel_name: str
    metadata: dict[str, Any]
    instructions: tuple[ActInstruction, ...]


@dataclass(frozen=True)
class ActRun:
    status: str
    input_sha256: str
    backend_sha256: str
    artifact_root: Path | None
    assembly: ActAssembly | None
    reason: str = ""


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) in {int, str, bool, type(None)}:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal(node.operand)
        if type(value) is int:
            return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        pairs = [(_literal(key), _literal(value)) for key, value in zip(node.keys, node.values)]
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("ACT metadata has duplicate keys")
        return dict(pairs)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "jnp":
        return f"jnp.{node.attr}"
    raise ValueError("ACT output contains unsupported executable metadata")


def parse_act_assembly(source: str) -> ActAssembly:
    """Accept only the emitted function/decorator/API-call shape; never eval it."""
    module = ast.parse(source)
    statements = module.body
    if statements and isinstance(statements[0], ast.Import):
        imports = statements[0].names
        if len(imports) != 1 or imports[0].name != "jax.numpy" or imports[0].asname != "jnp":
            raise ValueError("ACT output has an unexpected import")
        statements = statements[1:]
    if len(statements) != 1 or not isinstance(statements[0], ast.FunctionDef):
        raise ValueError("ACT output needs one kernel function")
    outer = statements[0]
    if [arg.arg for arg in outer.args.args] != ["kernel", "api"] or len(outer.body) != 2:
        raise ValueError("ACT kernel wrapper has unexpected structure")
    inner, returned = outer.body
    if not isinstance(inner, ast.FunctionDef) or not isinstance(returned, ast.Return):
        raise ValueError("ACT kernel body or return is missing")
    if not isinstance(returned.value, ast.Name) or returned.value.id != inner.name:
        raise ValueError("ACT kernel returns an unexpected object")
    if inner.args.args or len(inner.decorator_list) != 1:
        raise ValueError("ACT inner kernel has unexpected arguments or decorators")
    decorator = inner.decorator_list[0]
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Name) or decorator.func.id != "kernel":
        raise ValueError("ACT kernel decorator is unsupported")
    if decorator.args or any(keyword.arg is None for keyword in decorator.keywords):
        raise ValueError("ACT kernel decorator needs named literal fields")
    metadata = {keyword.arg: _literal(keyword.value) for keyword in decorator.keywords}
    if set(metadata) != {"hbm", "input", "constant", "output"}:
        raise ValueError("ACT kernel metadata is incomplete")
    if not isinstance(metadata["input"], list) or not isinstance(metadata["output"], list):
        raise ValueError("ACT input/output metadata is invalid")
    instructions: list[ActInstruction] = []
    for statement in inner.body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            raise ValueError("ACT kernel contains a non-instruction statement")
        call = statement.value
        if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name):
            raise ValueError("ACT kernel contains a non-API call")
        if call.func.value.id != "api" or call.args or any(keyword.arg is None for keyword in call.keywords):
            raise ValueError("ACT instruction uses unsupported API arguments")
        if not call.func.attr.isidentifier() or call.func.attr.startswith("_"):
            raise ValueError("ACT instruction has an invalid name")
        operands: list[tuple[str, int]] = []
        for keyword in call.keywords:
            value = _literal(keyword.value)
            if type(value) is not int:
                raise ValueError("ACT instruction operands must be integer literals")
            operands.append((keyword.arg, value))
        if len({name for name, _ in operands}) != len(operands):
            raise ValueError("ACT instruction has duplicate operands")
        instructions.append(ActInstruction(call.func.attr, tuple(operands)))
    if not instructions:
        raise ValueError("ACT output has no selected instructions")
    return ActAssembly(outer.name, metadata, tuple(instructions))


def run_act_reference(
    *, backend: Path, expected_backend_sha256: str, hlo: bytes, artifact_root: Path, timeout_s: int = 60,
) -> ActRun:
    """Run a supplied ACT binary in a fresh directory with distinct logs and input identity."""
    input_sha = hashlib.sha256(hlo).hexdigest()
    if not backend.is_file() or not os.access(backend, os.X_OK):
        return ActRun("tool_unavailable", input_sha, "", None, None, "ACT backend executable is unavailable")
    backend_sha = hashlib.sha256(backend.read_bytes()).hexdigest()
    if backend_sha != expected_backend_sha256:
        return ActRun("tool_unavailable", input_sha, backend_sha, None, None, "ACT backend identity differs")
    if timeout_s <= 0:
        raise ValueError("ACT wall timeout must be positive")
    artifact_root.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="act-reference-", dir=artifact_root))
    input_path, output_path, log_path = job / "input.hlo", job / "program.py", job / "logs"
    input_path.write_bytes(hlo)
    manifest = {"schema": "merlin.act_reference_run.v1", "input_sha256": input_sha, "backend_sha256": backend_sha}
    (job / "identity.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    try:
        result = subprocess.run(
            [str(backend), "--input", str(input_path), "--output", str(output_path), "--log", str(log_path)],
            capture_output=True, timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (job / "timeout.txt").write_text(str(exc) + "\n")
        return ActRun("search_timeout", input_sha, backend_sha, job, None, "ACT backend exceeded wall timeout")
    except OSError as exc:
        return ActRun("tool_unavailable", input_sha, backend_sha, job, None, str(exc))
    (job / "stdout.log").write_bytes(result.stdout)
    (job / "stderr.log").write_bytes(result.stderr)
    if result.returncode:
        return ActRun("compile_error", input_sha, backend_sha, job, None, f"ACT backend exited {result.returncode}")
    if not output_path.is_file() or not output_path.stat().st_size:
        return ActRun("compile_error", input_sha, backend_sha, job, None, "ACT backend produced no fresh output")
    try:
        assembly = parse_act_assembly(output_path.read_text())
    except (UnicodeError, SyntaxError, ValueError) as exc:
        return ActRun("unsupported_semantics", input_sha, backend_sha, job, None, f"ACT output import failed: {exc}")
    return ActRun("candidate_imported", input_sha, backend_sha, job, assembly)
