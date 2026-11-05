"""Explicit stock post-allocation scheduling for the model object.

The compiler still owns dependence, register, instruction and target legality.
This changes no arithmetic permission, ISA, calling convention or runtime flag.
"""

from __future__ import annotations

FEATURE = "stock_host_post_ra_scheduling"
_OPTION = "enable-post-misched"


def edit_cflags(flags: list[str]) -> list[str]:
    """Enable the stock scheduler, refusing an explicit contradictory selection."""
    result = list(flags)
    selected = False
    for position, flag in enumerate(result):
        option = flag.removeprefix("-mllvm=")
        key, separator, value = option.lstrip("-").partition("=")
        if key != _OPTION:
            continue
        forwarded = flag.startswith("-mllvm=") or (
            position > 0 and result[position - 1] == "-mllvm"
        )
        if not forwarded or separator and value not in {"1", "true"}:
            raise ValueError("post-allocation scheduling refuses a conflicting backend option")
        selected = True
    if not selected:
        result.extend(["-mllvm", "-" + _OPTION])
    return result


def ensure_registered() -> str:
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(
            name=FEATURE,
            action_class="HEURISTIC",
            description=(
                "Select stock post-allocation machine scheduling for the model object; "
                "preserve all numeric/ISA flags and runtime objects, explicit and default off."
            ),
            edit_cflags=edit_cflags,
        ))
    return FEATURE
