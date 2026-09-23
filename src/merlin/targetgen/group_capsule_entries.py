"""Deterministic capsule entries for captured compute groups.

Restates compiler grouping decisions in the shared capsule vocabulary. No corpus
writes, golden generation, grading, promotion, or optional experiment imports.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Collection, Mapping
from typing import Any

SCHEMA = "group_capsules_v1"
#: The schema's own role for a capsule whose shape is a real model's and deliberately not tile-relative.
SOURCE_ROLE = "model_derived"
#: Entry keys that make two groups the same program. The name and the multiplier's VALUE do not:
#: a unit's command stream is the same for every positive multiplier.
_IDENTITY_DROPS = ("name", "acc_scale", "lhs_scale", "rhs_scale", "comment", "source_reference")


def _identity(entry: Mapping[str, Any]) -> str:
    return json.dumps({k: v for k, v in entry.items() if k not in _IDENTITY_DROPS}, sort_keys=True)


def _label(entry: Mapping[str, Any]) -> str:
    stages = "_".join(str(s) for s in entry.get("epilogue") or ()) or "raw"
    if entry["op"] == "conv2d":
        extent = (
            f"c{entry['ci']}x{entry['Himg']}x{entry['Wimg']}_"
            f"k{entry['kh']}x{entry['kw']}s{entry['stride'][0]}_n{entry['N']}"
        )
    elif "K" in entry:
        extent = f"m{entry['M']}k{entry['K']}n{entry['N']}"
    else:
        # An elementwise entry reduces over nothing. Its declared bound is part of what is asked:
        # the same extents under two bounds are two demands.
        extent = f"m{entry['M']}n{entry['N']}" + (f"_b{entry['bound_lsb']}" if "bound_lsb" in entry else "")
    return f"G_{entry['op']}_{extent}_{stages}"


def _element_range(dtype: str) -> tuple[int, int]:
    """The whole range of an integer element format, from the format registry."""
    from merlin.common import quant_formats as qf
    from merlin.runtime.commandbuffer import SIGNED_STIMULUS_RANGE

    try:
        fmt = qf.get(dtype)
    except (KeyError, ValueError):
        return tuple(SIGNED_STIMULUS_RANGE)
    if fmt.kind != "int_affine":
        return tuple(SIGNED_STIMULUS_RANGE)
    return -(1 << (fmt.element_bits - 1)), (1 << (fmt.element_bits - 1)) - 1


def entries(
    target: str,
    module,
    *,
    weight_args: Collection[int] | None = None,
    model: str = "",
    with_raw: bool = True,
    oracle=None,
) -> dict[str, Any]:
    """Generator entries for every distinct accelerator group of ``module`` on ``target``."""
    from merlin.runtime.commandbuffer import SIGNED_STIMULUS_RANGE
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC

    found: dict[str, dict[str, Any]] = {}
    unstated: Counter = Counter()
    groups = CG.form_groups(module, target, oracle=oracle)
    for group in groups:
        if group.placement == CG.HOST or group.root is None:
            continue
        try:
            stated = GC.program(group, weight_args=weight_args)
        except CG.NoCapsuleForm as error:
            unstated[str(error)] += 1
            continue
        entry = dict(stated.entry)
        row = found.setdefault(_identity(entry), {"entry": entry, "count": 0, "groups": [], "program": stated})
        row["count"] += 1
        row["groups"].append(group.index)
    out: list[dict[str, Any]] = []
    for row in sorted(found.values(), key=lambda r: (-r["count"], _label(r["entry"]))):
        entry = row["entry"]
        name = _label(entry)
        entry.update(
            {
                "name": name,
                "cat": "layers",
                "kind": "layer",
                "label": "dev",
                "source_role": SOURCE_ROLE,
                "source_reference": (
                    f"{row['count']} compute group(s) of {model or 'a captured model'} on {target}: "
                    f"groups {row['groups'][:8]}{'...' if len(row['groups']) > 8 else ''}"
                ),
            }
        )
        if "bound_lsb" in entry:
            # A declared bound is a claim about rounding and saturation at the edges of the type,
            # and an elementwise sum cannot overflow an accumulator the way a contraction's small
            # stimulus guards against: only the whole element range can fail it.
            entry["stimulus_range"] = list(_element_range(str(entry.get("operand_dtype") or "")))
        elif entry.get("epilogue"):
            # Every fused stage behaves differently below zero; a stimulus that cannot go there
            # cannot fail it.
            entry["stimulus_range"] = list(SIGNED_STIMULUS_RANGE)
        entry.pop("scale_granularity", None)
        out.append(
            {
                "name": name,
                "count": row["count"],
                "groups": row["groups"],
                "entry": entry,
                "program": row["program"].to_dict(),
            }
        )
        if with_raw and entry.get("epilogue"):
            raw = {
                k: v
                for k, v in entry.items()
                if k not in ("epilogue", "acc_scale", "stimulus_range") and not k.startswith("pool_")
            }
            raw["epilogue"] = []
            raw["name"] = _label(raw)
            raw["source_reference"] = f"the bare-accumulator sibling of {name}"
            if not any(other["name"] == raw["name"] for other in out):
                out.append(
                    {"name": raw["name"], "count": row["count"], "groups": row["groups"], "entry": raw, "raw_of": name}
                )
    return {
        "schema": SCHEMA,
        "target": target,
        "model": model,
        "accelerator_groups": sum(1 for g in groups if g.placement != CG.HOST and g.root is not None),
        "stated": sum(row["count"] for row in found.values()),
        "distinct": len(found),
        "unstated": dict(unstated),
        "entries": out,
    }
