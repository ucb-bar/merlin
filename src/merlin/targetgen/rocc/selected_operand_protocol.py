"""Explicit source-pinned operand observations over unchanged original facts.

The caller must select the sidecar byte pin through its original-owned profile.
This loader rechecks that data selection; it neither grants an original owner nor
discovers a provider, substitutes original facts, or qualifies execution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.targetgen.rtl.firrtl_operand_slices import apply_selected_layouts, verify_selected_layout_source

from .normalized_operands import BoundOperandObserver
from .semantics import BoundSemantics

SCHEMA = "merlin.selected_operand_protocol.v1"


def _read_pin(pin, limit):
    if type(pin) is not dict or set(pin) != {"path", "sha256"} or type(pin["path"]) is not str:
        raise ValueError("operand selection file pin is malformed")
    path = Path(pin["path"])
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("operand selection requires an ordinary canonical selected file")
    if path.stat().st_size > limit:
        raise ValueError("operand selection exceeds its bounded byte document")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != _sha(pin["sha256"]):
        raise ValueError("operand selection file differs from its selected byte pin")
    return raw


def selected_inputs(selection):
    """Explicit checker-only files that a fresh original profile must freeze."""
    if type(selection) is not dict or set(selection) != {"original_facts", "sidecar"}:
        raise ValueError("operand selection requires only original facts and a pinned sidecar")
    _read_pin(selection["original_facts"], 16 * 1024 * 1024)
    selected = _json(_read_pin(selection["sidecar"], 2 * 1024 * 1024), 2 * 1024 * 1024)
    if type(selected) is not dict or selected.get("schema") != SCHEMA:
        raise ValueError("operand selection sidecar has an unsupported schema")
    _pins(selected.get("source_pins"))
    layouts = selected.get("layouts")
    if type(layouts) is not dict or type(layouts.get("source")) is not str:
        raise ValueError("operand selection omits its exact FIRRTL source pin")
    # The layout reader supplies the separate large-FIRRTL bounds and byte check.
    verify_selected_layout_source(layouts)
    source = {"path": layouts["source"], "sha256": _sha(layouts["source_sha256"])}
    rows = [selection["original_facts"], selection["sidecar"], *selected["source_pins"], source]
    return tuple(dict.fromkeys(row["path"] for row in rows))


def bind_selected_observer(original, selection):
    """Consume only explicit selected contract data, with final checks before return."""
    if type(selection) is not dict or set(selection) != {"original_facts", "sidecar"}:
        raise ValueError("operand selection requires only original facts and a pinned sidecar")
    facts = _read_pin(selection["original_facts"], 16 * 1024 * 1024)
    sidecar = _read_pin(selection["sidecar"], 2 * 1024 * 1024)
    observer = bind_original_operand_selection(
        original, original_facts_bytes=facts, selection_bytes=sidecar,
        expected_selection_sha256=selection["sidecar"]["sha256"],
    )
    if (_read_pin(selection["original_facts"], 16 * 1024 * 1024) != facts
            or _read_pin(selection["sidecar"], 2 * 1024 * 1024) != sidecar):
        raise ValueError("operand selection changed before observer publication")
    return observer


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("operand selection needs an exact SHA-256 byte pin")
    return value


def _json(data, limit):
    if type(data) is not bytes or not 0 < len(data) <= limit:
        raise ValueError("operand selection exceeds its bounded byte document")

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("operand selection has duplicate JSON ownership")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("operand selection is not finite JSON data")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _pins(rows):
    if type(rows) is not list or not 1 <= len(rows) <= 64:
        raise ValueError("operand selection needs a bounded original source roster")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256"} or type(row["path"]) is not str:
            raise ValueError("operand source pin is malformed")
        path = Path(row["path"])
        if (not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file()
                or path in seen or path.stat().st_size > 8 * 1024 * 1024):
            raise ValueError("operand source is not a unique bounded ordinary selected file")
        seen.add(path)
        expected = _sha(row["sha256"])
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("selected operand semantic source bytes changed")


def bind_original_operand_selection(original, *, original_facts_bytes, selection_bytes,
                                    expected_selection_sha256):
    """Return an optional observer; original facts/contract/raw decode stay exact.

    Both byte documents are explicitly resolved by the caller's sealed original
    selection. Controller/ISA pins and selected FIRRTL packing are re-opened
    before publication. Selector validity, guard truth, queue acceptance and
    geometry/schedule proofs remain separate obligations.
    """
    if type(original) is not BoundSemantics or original.operand_observer is not None:
        raise ValueError("operand sidecar requires the exact original unextended semantics selection")
    expected_selection_sha256 = _sha(expected_selection_sha256)
    if type(selection_bytes) is not bytes or hashlib.sha256(selection_bytes).hexdigest() != expected_selection_sha256:
        raise ValueError("operand sidecar differs from its explicitly selected byte pin")
    selected = _json(selection_bytes, 2 * 1024 * 1024)
    if (type(selected) is not dict or set(selected) != {
            "schema", "target", "original_facts_sha256", "source_pins", "layouts", "protocol"}
            or selected["schema"] != SCHEMA or selected["target"] != original._target):
        raise ValueError("operand sidecar is not a closed selection for the original target")
    original_facts = _json(original_facts_bytes, 16 * 1024 * 1024)
    if (hashlib.sha256(original_facts_bytes).hexdigest() != _sha(selected["original_facts_sha256"])
            or original_facts != original._facts):
        raise ValueError("operand sidecar does not bind the unchanged original facts bytes and owner")
    inputs = original_facts.get("inputs") or {}
    isa = {"path": inputs.get("isa_path"), "sha256": inputs.get("isa_sha256")}
    _pins(selected["source_pins"])
    if isa not in selected["source_pins"]:
        raise ValueError("operand sidecar omits the original selected ISA source pin")
    raw_isa, width = original.isa_constants(original._target), original._width
    observed_facts = apply_selected_layouts(original_facts, selected["layouts"])
    observer = BoundOperandObserver(target=original._target, protocol=selected["protocol"], facts=observed_facts,
                                    raw_isa=raw_isa, word_bits=width)
    _pins(selected["source_pins"])
    verify_selected_layout_source(selected["layouts"])
    if (original._facts != original_facts or original._target != selected["target"] or original._width != width
            or original.isa_constants(original._target) != raw_isa or original.operand_observer is not None):
        raise ValueError("original semantics changed before operand sidecar publication")
    # Do not mutate original._facts or its selected observer while returning this
    # detached view. All selected sources have final checks before publication.
    return observer
