"""Explicit stock compiler optimization for the whole-model object only.

No arithmetic flags, ISA, device emission or runtime compilation changes are
introduced. Stock optimization remains subject to the selected source numeric
contract and complete execution checks; this choice grants no new permission.
"""

from __future__ import annotations

FEATURE = "stock_host_optimization_o3"
_LEVELS = frozenset({"-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "-Og"})


def edit_cflags(flags: list[str]) -> list[str]:
    """Replace one explicit stock level, preserving every other flag exactly."""
    result = list(flags)
    if "-Ofast" in result or "-O4" in result:
        raise ValueError("stock O3 selection requires an explicit ordinary optimization level")
    positions = [i for i, flag in enumerate(result) if flag in _LEVELS]
    if len(positions) > 1:
        raise ValueError("stock O3 selection refuses ambiguous optimization levels")
    if positions:
        result[positions[0]] = "-O3"
    else:
        result.append("-O3")
    return result


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE,
            action_class="HEURISTIC",
            description=(
                "Select stock O3 for the whole-model object, preserving all numeric/ISA flags; "
                "explicit and default off, with original output and target checks."
            ),
            edit_cflags=edit_cflags,
        ))
    return FEATURE
