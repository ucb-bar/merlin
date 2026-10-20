"""Interval-based public diagnostic priorities, separate from the objective."""

from . import _public_native_observation as OBSERVATION
from .contracts import StageGateError, canonical_json, document_sha256

SCHEMA = "merlin.public_native_exploration_feedback.v1"


def join(document, *, owner):
    selected = {tuple(key) for key in document["selected_members"]}
    rows = []
    for cell in document["cells"]:
        key = cell["family"], cell["capsule"]
        row = {
            "family": key[0],
            "capsule": key[1],
            "status": "UNKNOWN",
            "recommendation": None,
            "conservative_model_saved_cycles": None,
            "reason": "unselected original public member",
        }
        if key in selected:
            projections = [cell[arm]["native_exploration"] for arm in ("baseline", "candidate")]
            for projection in projections:
                OBSERVATION.validate_projection(projection, owner=owner)
            if any(projection["status"] != "predicted" for projection in projections):
                row["reason"] = "one or both original arms lack qualified native predictions"
            else:
                a, b = [projection["prediction"]["interval"] for projection in projections]
                row["status"] = "paired_predictions"
                row["reason"] = "residual intervals overlap; ordering UNKNOWN"
                if a[0] > b[1]:
                    row.update(
                        recommendation="candidate",
                        reason="separated qualified prediction intervals",
                        conservative_model_saved_cycles=a[0] - b[1],
                    )
                elif b[0] > a[1]:
                    row.update(
                        recommendation="baseline",
                        reason="separated qualified prediction intervals",
                        conservative_model_saved_cycles=0,
                    )
        rows.append(row)
    return {
        "schema": SCHEMA,
        "selection_sha256": document_sha256(owner.record()),
        "members": rows,
        "order": [
            [row["family"], row["capsule"]]
            for row in sorted(
                rows,
                key=lambda row: (
                    row["conservative_model_saved_cycles"] is None,
                    -(row["conservative_model_saved_cycles"] or 0),
                    row["family"],
                    row["capsule"],
                ),
            )
        ],
        "complete_performance_objective": False,
        "measured_cycles": None,
        "attainment": None,
        "promotion_authority": "NONE",
        "scope": "selected public diagnostic members; no aggregate objective, incumbent or plateau",
    }


def validate(document, *, owner):
    if canonical_json(document.get("native_exploration")) != canonical_json(join(document, owner=owner)):
        raise StageGateError("public native feedback changed original per-member interval priorities")
