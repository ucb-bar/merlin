"""Flag a candidate whose gains on the primary model do not carry to the held-out models.

A lowering that speeds up one op kind on one model's shapes can be a genuine improvement to that
kind, or it can be tuned to that model's own extents -- the overfitting signature.  This takes the
SAME per-op-kind comparison the loop computes for its primary model (:func:`.feedback.compare`'s
``by_kind``) and the identical comparison for every held-out model the launch configures, and flags
every op kind where the primary shows a real gain but SOME held-out model, on the very same kind,
does not -- never a verdict about the candidate as a whole.

Nothing here names a model: the primary and the held-out models are PARAMETERS of the launch's
objective config (its ``held_out`` section, one measurement service per model).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "merlin_whole_model_transfer_check_v1"

#: A kind counts as a claimed GAIN on the primary model only past this margin below parity (a ratio of
#: candidate cycles over reference cycles).
GAIN_MARGIN = 0.95
#: A held-out model's own ratio for the kind counts as NOT CARRYING the gain at or above this floor.
NO_GAIN_FLOOR = 1.0


def kind_ratios(by_kind: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    """``{op kind: candidate-cycles / reference-cycles}``, from a ``compare()``'s own ``by_kind``."""
    return {str(row["kind"]): row.get("ratio") for row in by_kind if row.get("kind")}


def transfer_report(
    primary: str,
    primary_by_kind: Sequence[Mapping[str, Any]],
    held_out_by_kind: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Every op kind where ``primary`` shows a real gain but some held-out model does not.  A kind no
    held-out model exercises is ``untested``, never silently passed."""
    checked: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    for kind, primary_ratio in sorted(kind_ratios(primary_by_kind).items()):
        if primary_ratio is None or primary_ratio >= GAIN_MARGIN:
            continue  # not a claimed gain: nothing to check the transfer of
        evidence = {
            model: ratios[kind]
            for model, by_kind in held_out_by_kind.items()
            for ratios in (kind_ratios(by_kind),)
            if kind in ratios and ratios[kind] is not None
        }
        row: dict[str, Any] = {
            "kind": kind,
            "primary": primary,
            "primary_ratio": primary_ratio,
            "held_out_ratios": evidence,
        }
        not_carried = sorted(model for model, ratio in evidence.items() if ratio >= NO_GAIN_FLOOR)
        if not evidence:
            row.update(status="untested", why="no held-out model exercises this op kind")
        elif not_carried:
            row.update(
                status="flagged",
                why=f"{primary} gains {1 - primary_ratio:.1%} on {kind!r}, but {not_carried} show no gain "
                f"(ratio >= {NO_GAIN_FLOOR}) on the same kind -- possible overfitting to {primary}'s own shapes",
            )
            flagged.append(row)
        else:
            row.update(status="carries", why=f"every held-out model exercising {kind!r} also gains on it")
        checked.append(row)
    return {
        "schema": SCHEMA,
        "primary": primary,
        "held_out_models": sorted(held_out_by_kind),
        "gain_margin": GAIN_MARGIN,
        "no_gain_floor": NO_GAIN_FLOOR,
        "checked": checked,
        "flagged": flagged,
        "overfit_suspected": bool(flagged),
        "means": "a flagged kind is a gain claimed on the primary model that the same op kind does not show on "
        "some held-out model; it names which kind and which model(s), never a verdict about the candidate",
    }


def build_transfer_report(
    *,
    primary_name: str,
    primary_result: Mapping[str, Any],
    primary_reference: Mapping[str, Any] | None,
    held_out_results: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any] | None]],
) -> dict[str, Any]:
    """:func:`transfer_report`, computing every ``by_kind`` itself from each model's own (candidate
    result, same-machine reference result) pair."""
    from . import feedback as F

    primary_by_kind = F.compare(primary_result, primary_reference).get("by_kind") or []
    held_out_by_kind = {
        name: F.compare(found, reference).get("by_kind") or [] for name, (found, reference) in held_out_results.items()
    }
    return transfer_report(primary_name, primary_by_kind, held_out_by_kind)


__all__ = ["GAIN_MARGIN", "NO_GAIN_FLOOR", "SCHEMA", "build_transfer_report", "kind_ratios", "transfer_report"]
