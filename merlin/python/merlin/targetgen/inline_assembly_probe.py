"""Generic inline-assembly behavioral probe adapter.

The descriptor selects a source plus assembler and runner hooks.  This module owns only the protocol:
parse target-neutral byte-memory annotations, require the assembled words to contain the declared
instruction operations, execute them, and compare captured memory.  Concrete syntax, toolchains, and
simulators remain in target-owned hooks; there are no target or mnemonic branches here.
"""
from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from merlin.common.paths import merlin_dir, repo_root


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    for root in (merlin_dir(), repo_root()):
        candidate = root / path
        if candidate.is_file():
            return candidate
    return repo_root() / path


def load_hook(reference: str) -> Callable:
    """Resolve ``module:callable`` or descriptor-owned ``path/to/file.py:callable``."""
    owner, sep, attr = str(reference).partition(":")
    if not sep or not owner or not attr:
        raise ValueError(f"invalid inline-assembly hook reference {reference!r}")
    if owner.endswith(".py") or "/" in owner:
        path = _resolve_path(owner)
        if not path.is_file():
            raise ValueError(f"inline-assembly hook file is absent: {path}")
        name = f"_merlin_preflight_hook_{abs(hash(path.resolve())):x}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not import inline-assembly hook file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(owner)
    hook = getattr(module, attr, None)
    if not callable(hook):
        raise ValueError(f"inline-assembly hook {reference!r} is not callable")
    return hook


def _memory_annotations(source: str) -> tuple[list[tuple[int, bytes]], list[tuple[int, bytes]]]:
    preload: list[tuple[int, bytes]] = []
    expected: list[tuple[int, bytes]] = []
    for lineno, line in enumerate(source.splitlines(), 1):
        text = line.strip()
        if text.startswith("#"):
            text = text[1:].strip()
        elif text.startswith("//"):
            text = text[2:].strip()
        else:
            continue
        fields = text.split()
        if not fields or fields[0] not in {"@PRELOAD_MEMORY", "@EXPECT_MEMORY"}:
            continue
        if len(fields) != 3:
            raise ValueError(f"line {lineno}: inline memory directive requires address and hex bytes")
        kind, raw_address, raw_bytes = fields
        kind = kind.removeprefix("@")
        raw_bytes = raw_bytes.removeprefix("0x")
        if not raw_bytes or any(char not in "0123456789abcdefABCDEF" for char in raw_bytes):
            raise ValueError(f"line {lineno}: invalid inline memory hex")
        if len(raw_bytes) % 2:
            raise ValueError(f"line {lineno}: inline memory hex must contain whole bytes")
        try:
            item = (int(raw_address, 0), bytes.fromhex(raw_bytes))
        except ValueError as exc:
            raise ValueError(f"line {lineno}: invalid inline memory directive") from exc
        (preload if kind == "PRELOAD_MEMORY" else expected).append(item)
    if not expected:
        raise ValueError("inline assembly probe requires at least one @EXPECT_MEMORY directive")
    return preload, expected


def operation_coverage(target: str, words: list[int], required: list[str]) -> dict[str, Any]:
    """Decode exact required operations through the target's discovered ISA model."""
    from . import isa_disasm
    from .isa_model import isa_model_for_target

    model = isa_model_for_target(target)
    records = isa_disasm.disassemble(model, words)
    return isa_disasm.coverage(model, records, required=required)


def _captured(memory: Mapping[Any, Any], address: int) -> bytes | None:
    value = memory.get(address)
    if value is None:
        value = memory.get(str(address))
    if value is None:
        value = memory.get(hex(address))
    if isinstance(value, str):
        try:
            return bytes.fromhex(value.removeprefix("0x"))
        except ValueError:
            return None
    return bytes(value) if isinstance(value, (bytes, bytearray, memoryview)) else None


def run_capability_probe(*, te, probe, workdir, timeout: int = 600) -> dict[str, Any]:
    """Assemble and run one descriptor-owned source, returning generic operation observations."""
    fixture = probe.fixture
    if fixture.get("kind") != "inline_assembly_memory":
        raise ValueError("inline assembly adapter requires fixture.kind=inline_assembly_memory")
    source_path = _resolve_path(str(fixture.get("source") or ""))
    if not source_path.is_file():
        raise ValueError(f"inline assembly source is absent: {source_path}")
    assembler = load_hook(str(fixture.get("assembler") or ""))
    runner = load_hook(str(fixture.get("runner") or ""))
    source = source_path.read_text(encoding="utf-8")
    preload, expected = _memory_annotations(source)
    assembled = assembler(source=source, source_path=source_path, fixture=fixture, te=te)
    if isinstance(assembled, Mapping):
        words = list(assembled.get("words") or ())
    else:
        words = list(assembled or ())
    if not words:
        raise ValueError("inline assembly hook returned no instruction words")
    operations = list(probe.requirements.get("operations") or ())
    if any(operation.get("domain") != "instruction" for operation in operations):
        raise ValueError("inline assembly adapter accepts instruction-domain operations only")
    coverage = operation_coverage(
        te.target, words, [str(operation["operation"]) for operation in operations])
    if coverage["missing"] or coverage["n_illegal"]:
        detail = (f"fixture instruction coverage failed: missing={coverage['missing']}, "
                  f"undecodable={coverage['n_illegal']}")
        return {
            "reason": detail,
            "observations": [{**operation, "status": "unsupported",
                              "evidence": {"kind": "preflight_static", "detail": detail}}
                             for operation in operations],
            "instruction_coverage": coverage,
        }
    Path(workdir).mkdir(parents=True, exist_ok=True)
    result = runner(
        te=te,
        words=words,
        preload=preload,
        readback=[(address, len(value)) for address, value in expected],
        max_cycles=int(fixture.get("max_cycles") or 20000),
        timeout=timeout,
        workdir=Path(workdir),
        fixture=fixture,
    )
    if not isinstance(result, Mapping):
        raise ValueError("inline assembly runner result must be a mapping")
    memory = result.get("memory")
    if not isinstance(memory, Mapping):
        raise ValueError("inline assembly runner result requires a memory mapping")
    problems = []
    if result.get("halted") is not True:
        problems.append("program did not halt")
    for address, wanted in expected:
        got = _captured(memory, address)
        if got != wanted:
            problems.append(
                f"memory mismatch at {address:#x}: got={None if got is None else got.hex()} "
                f"expected={wanted.hex()}")
    status = "unsupported" if problems else "supported"
    checked = ", ".join(f"{address:#x} ({len(value)} bytes)" for address, value in expected)
    detail = "; ".join(problems) if problems else (
        f"halted in {int(result.get('cycles') or 0)} cycles; matched memory at {checked}")
    return {
        "reason": detail,
        "observations": [{**operation, "status": status,
                          "evidence": {"kind": "rtl_preflight", "detail": detail}}
                         for operation in operations],
        "instruction_coverage": coverage,
        "run": {key: result.get(key) for key in ("cycles", "reads", "writes", "halted")},
    }
