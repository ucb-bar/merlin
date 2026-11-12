"""Explicit, invocation-local transforms of fully prepared model MLIR.

The caller owns source semantics, effects and any provider implementation. This
seam preserves the public entry types and records the selected bytes; it grants
no numerical policy, device capability or profitability permission.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from merlin.common.digest import sha256_file
from merlin.common.jsonio import write_pretty_json

if TYPE_CHECKING:
    from xdsl.dialects.builtin import FunctionType

PreparedModelTransform = Callable[[Path, Path], Path]
RECEIPT = "prepared_model_transform.json"


def _public_interfaces(text: str) -> dict[str, FunctionType]:
    from xdsl.dialects.func import FuncOp
    from xdsl.parser import Parser

    from merlin.compiler.frontend.linalg_mlir import make_context, strip_paren_results

    module = Parser(make_context(), strip_paren_results(text)).parse_module()
    module.verify()
    result: dict[str, FunctionType] = {}
    for op in module.body.block.ops:
        if (
            not isinstance(op, FuncOp)
            or not op.body.blocks
            or (op.sym_visibility is not None and op.sym_visibility.data == "private")
        ):
            continue
        if op.sym_name.data in result:
            raise ValueError("prepared model has duplicate public function definitions")
        result[op.sym_name.data] = op.function_type
    return result


def apply_prepared_model_transform(
    prepared: Path,
    workdir: Path,
    callback: PreparedModelTransform | None,
) -> Path:
    """Apply a selected transform after ordinary preparation, before lowering.

    Empty selection returns the original path without reads, directories or IR
    parsing. A selected callback receives a private byte-identical input copy
    and its private output directory. It must preserve that copy, original input
    and public definition types, and return verified regular MLIR in that
    directory. New private helpers are permitted; their ABI/effects/source
    obligations remain caller-owned. A receipt is published only on success.
    """
    if callback is None:
        return prepared
    prepared, workdir = Path(prepared), Path(workdir)
    source_bytes = prepared.read_bytes()
    interfaces = _public_interfaces(source_bytes.decode("utf-8"))
    workdir.mkdir(parents=True, exist_ok=False)
    snapshot = workdir / "input.mlir"
    snapshot.write_bytes(source_bytes)
    selected = Path(callback(snapshot, workdir))
    if prepared.read_bytes() != source_bytes or snapshot.read_bytes() != source_bytes:
        raise ValueError("prepared model transform changed its immutable source")
    if selected.is_symlink() or not selected.is_file() or not selected.resolve().is_relative_to(workdir.resolve()):
        raise ValueError("prepared model transform must return a regular private model file")
    selected_bytes = selected.read_bytes()
    if _public_interfaces(selected_bytes.decode("utf-8")) != interfaces:
        raise ValueError("prepared model transform changed public entry types")
    write_pretty_json(
        workdir / RECEIPT,
        {
            "schema": "merlin.prepared_model_transform.v1",
            "source_path": str(prepared.resolve()),
            "source_sha256": sha256_file(prepared),
            "input_snapshot_path": str(snapshot.resolve()),
            "input_snapshot_sha256": sha256_file(snapshot),
            "selected_path": str(selected.resolve()),
            "selected_sha256": sha256_file(selected),
            "public_definitions": sorted(interfaces),
            "scope": (
                "verified source/selection identity and public entry types; "
                "semantic, effect and provider proofs are caller-owned"
            ),
        },
    )
    return selected
