"""Retain exact original observer bytes when a release stages an already selected observer."""

from __future__ import annotations

import json
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.targetgen.rocc.selected_operand_protocol import _json, _read_pin, selected_inputs

from ..spec import SpecError, read_yaml


def existing_selection(contract: tuple[Path, str] | None, facts: Path | None):
    """Return original byte sources only after checking the complete selected facts document."""
    if contract is None:
        return None
    source, digest = contract
    if source.is_symlink() or sha256_file(source) != digest:
        raise SpecError("selected capability contract changed before observer relocation")
    document = read_yaml(source)
    selected = document.get("rocc_operand_selection")
    if selected is None:
        return None
    if facts is None:
        raise SpecError("existing operand selection requires selected Phase 0 facts")
    selected_inputs(selected)
    original = _read_pin(selected["original_facts"], 16 * 1024 * 1024)
    exported = facts.read_bytes()
    options = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
    if json.dumps(_json(original, 16 * 1024 * 1024), **options) != json.dumps(
        _json(exported, 16 * 1024 * 1024), **options
    ):
        raise SpecError("original operand facts differ from the selected Phase 0 document")
    if sha256_file(source) != digest:
        raise SpecError("selected capability contract changed during observer relocation")
    return {
        "facts": Path(selected["original_facts"]["path"]),
        "sidecar": (Path(selected["sidecar"]["path"]), selected["sidecar"]["sha256"]),
        "phase0_export": {"path": str(facts), "sha256": sha256_file(facts)},
    }
