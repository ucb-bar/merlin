"""Input plumbing: parse the fixed `merlin_iface` interface grammar into a plain model.

Wraps the existing, tested `contract.interface_emit.parse_interface_mlir` (the canonical parser for the
frozen grammar) so the agent does not rebuild it. Target-agnostic: the grammar is fixed by the contract,
identical for every accelerator. No target semantics here.
"""
from __future__ import annotations
from typing import Any

from ..contract import interface_emit as _IE


def parse_interface(mlir_text: str) -> dict[str, Any]:
    """Parse `merlin_iface` MLIR text -> {abi_version, target, tensors, commands, params}.

    Thin pass-through to the contract parser. Use this instead of writing your own regex parser.
    """
    return _IE.parse_interface_mlir(mlir_text)


def emit_interface(cb: dict[str, Any]) -> str:
    """Inverse (round-trips with parse_interface) — handy for tests/debugging."""
    return _IE.emit_interface_mlir(cb)
