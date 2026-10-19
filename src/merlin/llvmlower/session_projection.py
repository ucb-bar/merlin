"""Rechecked prepared entry projections for generated multi-program callers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common import mlir_query
from merlin.common.digest import sha256_file

from .entry_weight_projection import (
    EntryWeightProjection,
    GeneratedDispatchABI,
    apply_entry_weight_projection,
    derive_entry_argument_table,
    module_text,
    plan_entry_weight_projection,
    rename_projected_entry,
)


@dataclass(frozen=True)
class ProgramEntryProjection:
    """Explicit original source, preparation directory, and rederived typed proof."""

    source: Path
    prepared_dir: Path
    projection: EntryWeightProjection


def read_program_entry_projection(*, model_dir: Path, prepared_dir: Path,
                                 source: Path, receipt: Path) -> ProgramEntryProjection:
    """Recompute every removal and reopen selected bytes; never trust a JSON remap."""
    data = json.loads(receipt.read_text())
    if data.get("schema") != "merlin.entry_weight_projection.v1":
        raise ValueError("unsupported entry projection receipt")
    if "source_path" in data and (Path(data["source_path"]).resolve() != source.resolve() or
                                  sha256_file(source) != data.get("source_file_sha256")):
        raise ValueError("entry projection source file differs from its receipt")
    try:
        contract = GeneratedDispatchABI(**data["contract"])
    except (KeyError, TypeError) as error:
        raise ValueError("invalid generated entry projection contract") from error
    table = derive_entry_argument_table(model_dir, prepared_dir)
    module = mlir_query.parse(source)
    plan = plan_entry_weight_projection(module, entry_contract=contract, argument_table=table)
    expected = json.loads(json.dumps(asdict(plan)))
    if (any(data.get(key) != value for key, value in expected.items()) or
            data.get("argument_table_sha256") != table.identity):
        raise ValueError("entry projection receipt differs from its current source proof")
    selected = Path(data["selected_path"])
    if (selected.is_symlink() or not selected.is_file() or
            sha256_file(selected) != data["selected_file_sha256"] or
            module_text(mlir_query.parse(selected)) != module_text(apply_entry_weight_projection(module, plan))):
        raise ValueError("selected entry projection differs from its rederived source")
    return ProgramEntryProjection(source.resolve(), prepared_dir.resolve(), plan)


def generation_options(selection: ProgramEntryProjection, *, entrypoint: str, out: Path):
    """Select a checked renamed proof for C generation and save its typed source."""
    if type(selection) is not ProgramEntryProjection:
        raise ValueError("session projection requires a typed source selection")
    original = mlir_query.parse(selection.source)
    renamed, plan = rename_projected_entry(original, selection.projection, entrypoint)
    out.mkdir(parents=True, exist_ok=True)
    source = out / "projection_source.mlir"
    source.write_text(module_text(renamed))
    selected = apply_entry_weight_projection(renamed, plan)
    options = {"prepared_dir": selection.prepared_dir,
               "entry_projection": plan, "entry_projection_source": source}
    return options, module_text(selected)
