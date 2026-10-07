"""PERF COVERAGE: does every form that holds a meaningful share of a model's cycles have a per-group perf
capsule at the model's own shape, with a reference-arm measurement -- and where is the package far from
that bar?

    forms = statement_forms(model_capsule, target=t)            # the capture's own forms, derived
    report = coverage_report(forms, ours=..., reference=..., capsules=..., share_threshold=0.02, ratio_threshold=1.5)

WHAT A FORM IS. The capsule corpus's own definition
(:func:`merlin_experiments.phase0.model_forms.form_key`, over the entry in the orientation the program
asks for it, :func:`merlin.xdsl_dialects.lowering.group_command.device_orientation`): the op, its
epilogue, its operand dtype, its scale class, its bias and the part of its geometry a lowering must
handle differently -- never its extents. Every group of any capture is put in exactly one form, from the
capture's whole-program statement (the same statement the builder asks the package from), so the forms
are a function of a capture and nothing here names a model, a group or a shape. A PREDICTED cost per
group is the same derived bound the phase-0 perf-cells product ranks by
(:mod:`merlin.perf.group_headroom`, the target's own RTL facts); a group it has no formula for predicts
UNKNOWN, never zero.

WHICH FORMS MUST BE COVERED. A form is required when its SHARE of the model's cycles reaches
``share_threshold`` on ANY basis available: its predicted share, its share of the reference arm's
measured cycles, or its share of the package's measured cycles. Taking the largest is deliberate: a form
the bound calls cheap but the package spends 6% of the model on is exactly the one a predicted share
alone would miss. A form whose every basis is UNKNOWN is required (fail closed). A required form is
covered when at least one of its groups has a perf capsule at model shape whose REFERENCE arm was
measured (``capsules``: ``{group: {"reference_cycles": int, ...}}``).

WHERE THE GAPS ARE. Every form with both arms measured reports ``ours / reference`` over its groups;
one above ``ratio_threshold`` is listed as a gap form. Measurements only: which forms, how far, never
how to close them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA",
    "capsule_rows",
    "contract_of_entry",
    "coverage_report",
    "form_of_entry",
    "load_counts",
    "matching_groups",
    "statement_forms",
]

SCHEMA = "merlin_perf_coverage_v1"
UNKNOWN = "UNKNOWN"


def form_of_entry(entry: Mapping[str, Any], *, window_mean: bool) -> tuple[dict[str, Any], str]:
    """The corpus's form key for one stated entry, in the orientation the program asks for it."""
    from merlin.xdsl_dialects.lowering.group_command import device_orientation
    from merlin_experiments.phase0 import model_forms as MF

    oriented = device_orientation(dict(entry), {"stored_operand": None}) if window_mean else dict(entry)
    key = MF.form_key(oriented)
    return key, MF.key_text(key)


def contract_of_entry(contract: Any):
    """``exactness(group, entry)`` for a one-group program build: the exactness contract (``to_dict``) its
    form is given, matched over the entry's own fields and its form key (an exactness contract names a
    form by either), with the op's own declared tolerance where the contract does not name the form."""

    def exactness(group: int, entry: Mapping[str, Any]) -> dict[str, Any]:
        form = dict(entry or {})
        try:
            key, _text = form_of_entry(form, window_mean=False)
            form.update(key)
        except Exception:  # noqa: BLE001 -- a form key that cannot be formed leaves the entry's own fields
            pass
        return contract.resolve(form, op_bound_lsb=(entry or {}).get("bound_lsb")).to_dict()

    return exactness


def statement_forms(model_capsule: str | Path, *, target: str) -> list[dict[str, Any]]:
    """Every compute group of the capture, with its form and its predicted (derived-bound) cycles."""
    from merlin.perf import group_headroom as GH
    from merlin.perf import whole_model_build as W

    capsule = W.load_model_capsule(model_capsule)
    buffer = W.state(capsule, target=target)
    machine = GH.machine_for(target)
    rows = []
    for row in (buffer.get("whole_program") or {}).get("per_group") or ():
        entry = row.get("entry")
        if not isinstance(entry, Mapping):
            continue
        operands = row.get("operands") or {}
        window_mean = str(operands.get("lhs") or "").startswith("ONES_")
        key, text = form_of_entry(entry, window_mean=window_mean)
        op = str(entry.get("op"))
        facts = W._shape_facts(op, entry)
        if facts is not None and "operand_dtype" not in facts:
            facts = {**facts, "operand_dtype": entry.get("operand_dtype"), "output_dtype": entry.get("output_dtype")}
        estimate = GH.macs_and_bytes(op, facts) if facts is not None else None
        bound = GH.group_bound(*estimate, machine) if estimate is not None else None
        rows.append(
            {
                "group": str(row["group"]),
                "op": op,
                "form": key,
                "form_text": text,
                "shape": {k: v for k, v in (facts or {}).items() if k not in ("operand_dtype", "output_dtype")} or None,
                "predicted_cycles": (bound or {}).get("bound_cycles"),
                "predicted_limiter": (bound or {}).get("limiter"),
            }
        )
    return rows


def matching_groups(form_text: str, forms: Sequence[Mapping[str, Any]]) -> list[int]:
    """The groups of a capture (its :func:`statement_forms`) that are the same form: a held-out model's
    instances of a cell's form, found by the form's own key, never by a shape or an id."""
    return [int(row["group"]) for row in forms if row.get("form_text") == form_text]


def _share(values: Mapping[str, Any], members: Sequence[str]) -> float | None:
    numbers = {g: v for g, v in values.items() if isinstance(v, (int, float))}
    total = sum(numbers.values())
    if not total or any(g not in numbers for g in members):
        return None
    return sum(numbers[g] for g in members) / total


def coverage_report(
    forms: Sequence[Mapping[str, Any]],
    *,
    ours: Mapping[str, Any],
    reference: Mapping[str, Any],
    capsules: Mapping[str, Mapping[str, Any]],
    share_threshold: float,
    ratio_threshold: float,
) -> dict[str, Any]:
    """Per form: its groups, its share on every basis, whether it must be covered and is, and its
    ours-over-reference ratio. ``ours``/``reference`` map group -> measured cycles (whole-model or
    standalone, the caller says which in ``basis``); ``capsules`` maps group -> its perf-capsule row."""
    predicted = {str(r["group"]): r.get("predicted_cycles") for r in forms}
    ours = {str(k): v for k, v in ours.items()}
    reference = {str(k): v for k, v in reference.items()}
    by_form: dict[str, dict[str, Any]] = {}
    for row in forms:
        slot = by_form.setdefault(row["form_text"], {"form": row["form"], "op": row["op"], "groups": []})
        slot["groups"].append(str(row["group"]))
    documents = []
    for text, slot in by_form.items():
        members = slot["groups"]
        shares = {
            "predicted": _share(predicted, members),
            "reference_measured": _share(reference, members),
            "ours_measured": _share(ours, members),
        }
        known = [s for s in shares.values() if s is not None]
        share = max(known) if known else None
        required = share is None or share >= share_threshold
        with_capsule = [g for g in members if isinstance((capsules.get(g) or {}).get("reference_cycles"), int)]
        a = [ours.get(g) for g in members]
        b = [reference.get(g) for g in members]
        ratio = sum(a) / sum(b) if all(isinstance(v, (int, float)) for v in a + b) and sum(b) else None
        documents.append(
            {
                "form": slot["form"],
                "form_text": text,
                "op": slot["op"],
                "groups": members,
                "shares": shares,
                "share": share if share is not None else UNKNOWN,
                "required": required,
                "capsule_groups": with_capsule,
                "covered": bool(with_capsule),
                "ours_cycles": sum(a) if ratio is not None else None,
                "reference_cycles": sum(b) if ratio is not None else None,
                "ratio": round(ratio, 3) if ratio is not None else None,
                "gap": ratio is not None and ratio > ratio_threshold,
            }
        )
    documents.sort(key=lambda d: -(d["share"] if isinstance(d["share"], float) else 2.0))
    uncovered = [d for d in documents if d["required"] and not d["covered"]]
    gaps = sorted((d for d in documents if d["gap"]), key=lambda d: -(d["ours_cycles"] - d["reference_cycles"]))
    return {
        "schema": SCHEMA,
        "share_threshold": share_threshold,
        "ratio_threshold": ratio_threshold,
        "share_basis": (
            "the largest of the predicted, reference-measured and package-measured shares; UNKNOWN is required"
        ),
        "forms": documents,
        "uncovered": [{"groups": d["groups"], "op": d["op"], "share": d["share"]} for d in uncovered],
        "gap_forms": [
            {
                "groups": d["groups"],
                "op": d["op"],
                "ratio": d["ratio"],
                "ours": d["ours_cycles"],
                "reference": d["reference_cycles"],
            }
            for d in gaps
        ],
        "passed": not uncovered,
    }


def load_counts(path: str | Path) -> dict[str, Any]:
    """``{group: cycles}`` from a whole-model result or a perf-capsule rows document."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if "verdict" in document:
        return {str(r["group"]): r.get("cycles") for r in (document["verdict"].get("groups") or ())}
    return {str(r["group"]): r.get("cycles") for r in document.get("rows") or () if not r.get("model")}


def capsule_rows(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """``{group: {"reference_cycles", "device", "source"}}`` from perf-capsule measurement documents
    (:func:`.group_capsules.measure_on_gsim` output): a group is
    capsuled when its REFERENCE arm was measured correct at model shape (the objective model only)."""
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in document.get("rows") or ():
            ok = row.get("admitted") if "admitted" in row else row.get("correct")
            if row.get("model") or row.get("arm") != "reference" or not ok or not isinstance(row.get("cycles"), int):
                continue
            found.setdefault(
                str(row["group"]), {"reference_cycles": row["cycles"], "device": row.get("device"), "source": str(path)}
            )
    return found
