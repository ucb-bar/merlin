"""Input plumbing: parse the fixed `merlin_iface` interface grammar into a plain model.

Wraps the existing, tested `contract.interface_emit.parse_interface_mlir` (the canonical parser for the
frozen grammar) so the agent does not rebuild it. Target-agnostic: the grammar is fixed by the contract,
identical for every accelerator. No target semantics here.
"""
from __future__ import annotations
from typing import Any

try:
    from ..contract import interface_emit as _IE
except ImportError:  # sandbox: staged flat on sys.path (no parent package) — use the flat interface_emit shim
    import interface_emit as _IE  # type: ignore[no-redef]

try:
    from ..contract import linalg_iface as _LI
except ImportError:  # sandbox: staged flat on sys.path
    import linalg_iface as _LI  # type: ignore[no-redef]


def parse_interface(mlir_text: str) -> dict[str, Any]:
    """Parse `merlin_iface` MLIR text -> {abi_version, target, tensors, commands, params}.

    Thin pass-through to the contract parser. Use this instead of writing your own regex parser.
    """
    return _IE.parse_interface_mlir(mlir_text)


def is_linalg_on_tensors(mlir_text: str) -> bool:
    """True when the interface MLIR is the `linalg-on-tensors` grammar (a `func.func @forward` of
    linalg/tensor/math ops) rather than `merlin_iface` v0.1. Use it to route your `parse` entrypoint
    between the two grammars."""
    return _LI.is_linalg_on_tensors(mlir_text)


def parse_linalg(mlir_text: str) -> dict[str, Any]:
    """Parse `linalg-on-tensors` MLIR text -> a structural workload inventory
    {level, entry, args, results, ops:[{kind, op, family, prov, ins, outs, results, extents,
    body_ops, reduction_dims}]}.

    Thin pass-through to the contract reader (:func:`merlin.targetgen.contract.linalg_iface`).
    Parses the real IR with xDSL — use it instead of hand-rolling a linalg text parser. It is a
    READER, not a lowering: you author the lowering from the inventory (matmul-family records reuse
    the residency command path; elementwise/reduction ops name their semantics in ``body_ops``)."""
    return _LI.parse_linalg_mlir(mlir_text)


def emit_interface(cb: dict[str, Any]) -> str:
    """Inverse (round-trips with parse_interface) — handy for tests/debugging."""
    return _IE.emit_interface_mlir(cb)
