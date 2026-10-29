"""Existing dense and convolution library routes suggested by public traffic evidence, never a speed verdict."""
from __future__ import annotations

import math

from . import mechanism_hints

# Generic shared library modules, not endpoint names, instructions or winning settings.
_ROUTES = {"dense_array_retention": "array_retention", "tile_load_coalescing": "wide_loads"}
# A convolution reaches resident weights (RHS phases) and rolling multi-row input windows only through the
# shared convolution consumer; dense array retention composes with it through checked output groups.
_CONV_ROUTES = {**_ROUTES, "convolution_exploration": "window_reuse"}
CONV_ORDER = {"window_reuse": 0, "array_retention": 0, "wide_loads": 1}
CONV_RECIPE = "heavy-pass-recipes/reference/convolution.md"
_REUSE_WORDS = ("resid", "rolling", "reuse", "retention", "wide")
LOW_FENCES_PER_TILE = 0.05  # Advisory ranking threshold; this is not a hardware or completion law.


def route_kind(row, convolution=False):
    return (_CONV_ROUTES if convolution else _ROUTES).get(str(row.get("api", "")).rsplit(".", 1)[-1])


def _capabilities(module):
    """Shared library capabilities about reuse or wide transfer whose symbols live in a selected module."""
    from merlin.compile.scheduling.api_catalog import scheduling_capabilities

    found = []
    for capability in scheduling_capabilities(23)["capabilities"]:
        words = str(capability.get("name", "")).split("_")
        if (any(str(symbol).rsplit(".", 1)[0] == module for symbol in capability.get("symbols", ()))
                and any(word.startswith(_REUSE_WORDS) for word in words)):
            found.append(capability)
    return found


def library_routes(selection, convolution=False):
    """Read actual selected catalogue IDs/APIs and unresolved prerequisites; invent no availability."""
    rows = []
    for row in (selection or {}).get("all_mechanisms", ()):
        if (not isinstance(row, dict) or not route_kind(row, convolution) or row.get("status") == "unsupported"
                or "unsupported" in (row.get("requirements") or {}).values()):
            continue
        observed = mechanism_hints.opportunity(row, selection)
        route = dict(id=row["id"], api=f"{row.get('api')}:{row.get('symbol')}",
                     kind=route_kind(row, convolution), requirements=row.get("requirements", {}), **observed)
        if convolution:
            opened = _capabilities(str(row.get("api")))
            route["capabilities"] = [c["name"] for c in opened]
            if route["kind"] != "wide_loads":
                route["recipe"] = CONV_RECIPE
        rows.append(route)
    if convolution:
        return sorted(rows, key=lambda row: CONV_ORDER[row["kind"]])
    return sorted(rows, key=lambda row: row["kind"] != "array_retention")


def region(states):
    """The declared region whose traffic routes exist: a dense contraction, else a convolution, else None."""
    if states.get("workload.dense_contraction") == "supported" and states.get("workload.convolution") == "unsupported":
        return "dense"
    return "convolution" if states.get("workload.convolution") == "supported" else None


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def signal(entry, states, selection=None):
    """Actual requested-byte amplification over a counted fence census, scoped to a declared region.

    Dense: low fence density, as before. Convolution: the counted fences are reported, not gated -- the
    convolution emitter drains around each transferred input window, so its fence count follows the very
    transfers the reuse routes remove. It applies only when the case's own selection offers the shared
    convolution consumer route (resident weights, rolling input rows); dense retention alone is not one.
    """
    kind = region(states)
    if kind is None:
        return None
    dense = kind == "dense"
    fence = entry.get("fence_overlap") or {}
    if (dense and fence.get("fence_per_tile")) or fence.get("count_kind") not in {
            "executed", "executed commands", "static sites"}:
        return None
    fences, tiles = fence.get("fences"), fence.get("output_tiles")
    if not (_number(fences) and _number(tiles) and tiles > 0 and (not dense or fences / tiles <= LOW_FENCES_PER_TILE)):
        return None
    if not dense and not any(r["kind"] == "window_reuse" for r in library_routes(selection, convolution=True)):
        return None
    reads = (entry.get("capacity_use") or {}).get("operand_reads") or {}
    actual, compulsory = reads.get("device_read_bytes"), reads.get("compulsory_bytes")
    if not (_number(actual) and _number(compulsory) and compulsory > 0 and actual > compulsory):
        return None
    found = dict(fences=fences, output_tiles=tiles, fences_per_output_tile=round(fences / tiles, 4),
                 count_basis=fence.get("count_kind", "unknown"), requested_read_bytes=actual,
                 compulsory_operand_bytes=compulsory, read_amplification=round(actual / compulsory, 2))
    census = fence.get("dynamic") or {}
    if _number(census.get("loads")) and _number(census.get("output_stores")) and census["output_stores"] > 0:
        found["loads_per_output_tile"] = round(census["loads"] / census["output_stores"], 2)
        found["load_count_basis"] = census.get("basis", "executed command census")
    density = "low fence density (" if dense else "convolution region ("
    reading = (f"{density}{found['fences_per_output_tile']} fences/output tile, {found['count_basis']}); "
               f"requested operand reads {found['read_amplification']}x compulsory bytes ({actual}/{compulsory} B)")
    if "loads_per_output_tile" in found:
        reading += f"; {found['loads_per_output_tile']} loads/output tile ({found['load_count_basis']})"
    if not dense:
        found["region"] = kind
    return dict(found, reading=reading, interpretation="compare reuse and transfer routes; no timing bottleneck proved")


def refusal(failure):
    """Bounded public text can suggest a route, never establish its capability or delivery to an old run."""
    if not isinstance(failure, dict):
        return None
    text = " ".join(str(failure.get(key) or "")[:2048] for key in ("category", "detail", "reason"))
    words = mechanism_hints._words(text.lower())
    if not any(word.startswith(("refus", "unavail", "unsupport", "unadmit")) for word in words):
        return None
    if "wide" in words and any(word.startswith(("address", "dma", "load")) for word in words):
        return dict(kind="wide_loads", basis="public refusal text; route hypothesis, capability unverified")
    if any(word.startswith(("retain", "retention")) for word in words) and {"array", "weight"} & set(words):
        return dict(kind="array_retention", basis="public refusal text; route hypothesis, capability unverified")
    return None
