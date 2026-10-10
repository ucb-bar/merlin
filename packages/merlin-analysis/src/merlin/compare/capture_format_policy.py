"""Explicit study format requirements and selected-byte transport, not authority."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from merlin.capture import contraction_formats as F
from merlin.common.pinned_files import PinnedFile


def declared_format_cells(raw: dict) -> dict[tuple[str, str], str]:
    """Derive required cells only from the explicit versioned declaration."""
    if "contraction_format_requirements" not in raw:
        if raw.get("version") == 3:
            raise ValueError("Declared contraction format selection is unavailable.")
        return {}
    requirements = raw["contraction_format_requirements"]
    if (
        type(raw.get("version")) is not int
        or raw["version"] != 3
        or type(requirements) is not dict
        or not requirements
        or any(
            type(key) is not str or type(value) is not str or value != "complete_integer.v1"
            for key, value in requirements.items()
        )
    ):
        raise ValueError("Declared contraction format membership is incomplete.")
    models = raw.get("models")
    if type(models) is not list or not models:
        raise ValueError("Declared contraction format membership is incomplete.")
    cells = {}
    observed_precisions = set()
    for model in models:
        if type(model) is not dict or type(model.get("name")) is not str or type(model.get("precisions")) is not list:
            raise ValueError("Declared contraction format membership is incomplete.")
        for precision in model["precisions"]:
            if type(precision) is not str:
                raise ValueError("Declared contraction format membership is incomplete.")
            observed_precisions.add(precision)
            if precision in requirements:
                key = (model["name"], precision)
                if key in cells:
                    raise ValueError("Declared contraction format membership is incomplete.")
                cells[key] = requirements[precision]
    if not set(requirements) <= observed_precisions or not cells:
        raise ValueError("Declared contraction format membership is incomplete.")
    return cells


def validate_format_selection_membership(raw, *, originals=None, prepared=None):
    required = declared_format_cells(raw)
    originals = {} if originals is None else dict(originals)
    prepared = {} if prepared is None else dict(prepared)
    if (
        set(originals) & set(prepared)
        or set(originals) - set(required)
        or any(type(value) is not F.CaptureContractionOriginalSelection for value in originals.values())
        or any(type(value) is not F.CaptureContractionSelection for value in prepared.values())
    ):
        raise ValueError("Declared contraction format membership is incomplete.")
    if not set(required) <= set(originals) | set(prepared):
        raise ValueError("Declared contraction format selection is unavailable.")
    if any((originals.get(key) or prepared[key]).policy != policy for key, policy in required.items()):
        raise ValueError("Declared contraction format membership is incomplete.")
    return originals, prepared


def prepare_declared_formats(raw, capture_roots, *, originals=None, prepared=None):
    originals, selections = validate_format_selection_membership(raw, originals=originals, prepared=prepared)
    if not (set(originals) | set(selections)) <= set(capture_roots):
        raise ValueError("Declared contraction format membership is incomplete.")
    for key, original in originals.items():
        selections[key] = F.prepare_capture_contraction_selection(capture_roots[key], original)
    for key, selected in selections.items():
        F.require_capture_contraction_formats(capture_roots[key], selected)
    return selections


def _pin(raw):
    if type(raw) is not dict or set(raw) != {"path", "sha256"} or type(raw["path"]) is not str:
        raise ValueError("Declared contraction format membership is incomplete.")
    return PinnedFile(Path(raw["path"]), raw["sha256"])


def _limits(raw):
    if type(raw) is not dict or set(raw) != set(asdict(F.ContractionFormatLimits())):
        raise ValueError("Declared contraction format membership is incomplete.")
    return F.ContractionFormatLimits(**raw)


def load_original_format_selections(descriptor: PinnedFile) -> dict:
    """Load bounded explicit file/SHA data; no saved eligibility is consumed."""
    if type(descriptor) is not PinnedFile:
        raise ValueError("Declared contraction format selection is unavailable.")
    limits = F.ContractionFormatLimits()
    document = F._json(F._read(descriptor, limits), limits)
    if (
        set(document) != {"schema", "cells"}
        or document["schema"] != "merlin.capture_contraction_original_selections.v1"
        or type(document["cells"]) is not list
        or not document["cells"]
        or len(document["cells"]) > limits.programs
    ):
        raise ValueError("Declared contraction format membership is incomplete.")
    result = {}
    program_count = 0
    for cell in document["cells"]:
        if (
            type(cell) is not dict
            or set(cell) != {"model", "precision", "policy", "limits", "programs"}
            or any(type(cell[key]) is not str or not cell[key] for key in ("model", "precision"))
            or type(cell["programs"]) is not list
        ):
            raise ValueError("Declared contraction format membership is incomplete.")
        program_count += len(cell["programs"])
        if program_count > limits.programs:
            raise ValueError("Contraction input exceeds its selected budget.")
        rows = []
        for row in cell["programs"]:
            if type(row) is not dict or set(row) != {"program", "bundle", "original_graph"}:
                raise ValueError("Declared contraction format membership is incomplete.")
            rows.append(F.OriginalProgramContractionInput(row["program"], row["bundle"], _pin(row["original_graph"])))
        key = (cell["model"], cell["precision"])
        if key in result:
            raise ValueError("Declared contraction format membership is incomplete.")
        result[key] = F.CaptureContractionOriginalSelection(tuple(rows), cell["policy"], _limits(cell["limits"]))
    # The descriptor selects a whole roster. Admit its aggregate bytes before
    # parsing any original graph, rather than multiplying the per-cell budget.
    total = 0
    for selected in result.values():
        for row in selected.programs:
            total += len(F._read(row.original_graph, selected.limits))
            if total > limits.total_bytes:
                raise ValueError("Contraction input exceeds its selected budget.")
    for selected in result.values():
        F.verify_original_contraction_selection(selected)
    F._read(descriptor, limits)
    return result


def selection_records(selections: Mapping) -> list[dict]:
    """Retain exact selected bytes for replay; no census status is an input."""

    def pin(value):
        return {"path": str(value.path), "sha256": value.sha256}

    return [
        {
            "model": key[0],
            "precision": key[1],
            "policy": selected.policy,
            "limits": asdict(selected.limits),
            "session_contract": pin(selected.session_contract),
            "programs": [
                {
                    "program": row.program,
                    "original_graph": pin(row.original_graph),
                    "frontend_trace": pin(row.frontend_trace),
                    "final_mlir": pin(row.final_mlir),
                }
                for row in selected.programs
            ],
        }
        for key, selected in sorted(selections.items())
    ]


def require_frozen_formats(raw, capture_roots):
    """Reconstruct data selections and re-evaluate a declared frozen requirement."""
    required = declared_format_cells(raw)
    if not required:
        return
    records = (raw.get("freeze") or {}).get("contraction_format_selections")
    if type(records) is not list or len(records) != len(required):
        raise ValueError("Declared contraction format selection is unavailable.")
    selections = {}
    for record in records:
        if (
            type(record) is not dict
            or set(record) != {"model", "precision", "policy", "limits", "session_contract", "programs"}
            or type(record["programs"]) is not list
            or any(type(record[key]) is not str for key in ("model", "precision"))
        ):
            raise ValueError("Declared contraction format membership is incomplete.")
        rows = []
        for row in record["programs"]:
            if type(row) is not dict or set(row) != {"program", "original_graph", "frontend_trace", "final_mlir"}:
                raise ValueError("Declared contraction format membership is incomplete.")
            rows.append(
                F.ProgramContractionInput(
                    row["program"], _pin(row["original_graph"]), _pin(row["frontend_trace"]), _pin(row["final_mlir"])
                )
            )
        key = (record["model"], record["precision"])
        if key in selections:
            raise ValueError("Declared contraction format membership is incomplete.")
        selections[key] = F.CaptureContractionSelection(
            _pin(record["session_contract"]), tuple(rows), record["policy"], _limits(record["limits"])
        )
    if set(selections) != set(required):
        raise ValueError("Declared contraction format membership is incomplete.")
    prepare_declared_formats(raw, capture_roots, prepared=selections)
