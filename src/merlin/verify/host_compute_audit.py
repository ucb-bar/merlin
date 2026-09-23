"""Veto host code that is doing the accelerator's work.

A group placed on an accelerator is a claim: the unit computes it. The emitted program can break
that claim without any gate noticing, because every gate checks the numbers and the numbers are
right. A backend that issues the contraction to the device and then finishes the requantization in
a host loop passes numerics, passes the trace check (the device commands are all there), and
spends its time on the host.

What separates the two is how host work SCALES. Inside an accelerator group the host legitimately
issues commands, computes addresses and walks tiles: work proportional to the number of commands.
Host code doing the group's arithmetic is proportional to the number of ELEMENTS: it loads,
computes and stores each one. So the audit counts, per group, the dynamic host operations of the
function that implements it, and compares them with the elements the group produces.

The host is always a RISC-V core, scalar or vector, so nothing here depends on the accelerator:
the counts come from the emitted LLVM-dialect function (:mod:`merlin.perf.host_cfg_activity`), the
placement from the compute-group plan, and the join from the function symbol.

* An ACCELERATOR group is audited against the budget.
* A HOST group is doing what it was declared to do and is reported, not judged: covering an
  unsupported quantization, activation or operation on the host is a placement, not a defect.
* A group whose function's control flow cannot be counted is ``unknown``, never ``clean``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SCHEMA = "host_compute_audit_v1"
HOST = "host"

CLEAN = "clean"
HOST_COMPUTE = "host_compute_in_accelerator_group"
UNKNOWN = "unknown"
DECLARED_HOST = "declared_host"
NOT_EMITTED = "not_emitted"

#: Categories of :mod:`host_cfg_activity` that are arithmetic on values (as opposed to control,
#: comparison or memory traffic).
_ARITHMETIC = ("integer_arithmetic", "floating_arithmetic", "conversion")
#: Arithmetic that an integer index computation never needs. Any of it at element scale is the
#: payload being computed on, whatever else the function does.
_VALUE_ONLY = ("floating_arithmetic", "conversion")


@dataclass(frozen=True)
class Budget:
    """How much host work an accelerator group may carry, per element it produces.

    The defaults separate the two populations by an order of magnitude on each side. Command issue
    for a tiled unit costs tens of host operations per COMMAND and a command covers a tile of
    elements, so it lands well under one operation per element; per-element host arithmetic costs
    a load, the operation and a store at the very least, so it lands at three or more. A target
    with a smaller tile declares a looser budget in its contract rather than editing this.
    """

    arithmetic_per_element: float = 1.0
    value_arithmetic_per_element: float = 0.25
    payload_bytes_per_element_byte: float = 0.5


@dataclass(frozen=True)
class GroupSite:
    """One compute group as emitted: where it was placed and which function implements it."""

    group: int
    placement: str
    symbol: str
    elements: int
    element_bytes: int = 1
    stages: tuple[str, ...] = ()


def sites_from_outline(dispatches: Iterable[Any], *, element_bytes: Mapping[str, int] | None = None) -> list[GroupSite]:
    """Group sites from an outliner's dispatch table (``DispatchInfo`` rows with a group)."""
    from merlin.common import mlir_query as mq

    sites: list[GroupSite] = []
    for row in dispatches:
        if getattr(row, "group", None) is None:
            continue
        elements, width = 0, 1
        for spelled in row.result_types:
            shape, dtype = _shape_of_type_text(spelled)
            total = 1
            for extent in shape:
                total *= max(int(extent), 0)
            elements += total if shape else 0
            width = max(width, (element_bytes or mq._DTYPE_BYTES).get(dtype, 1))
        sites.append(
            GroupSite(
                group=int(row.group),
                placement=str(row.placement),
                symbol=str(row.symbol),
                elements=elements,
                element_bytes=width,
                stages=tuple(row.stages),
            )
        )
    return sites


def _shape_of_type_text(spelled: str) -> tuple[list[int], str]:
    """``tensor<4x16xi8>`` -> ``([4, 16], "i8")``, structurally; ``([], spelled)`` otherwise."""
    text = spelled.strip()
    if not text.startswith("tensor<") or not text.endswith(">"):
        return [], text
    parts = text[len("tensor<") : -1].split("x")
    if not all(part.isdigit() for part in parts[:-1]):
        return [], parts[-1]
    return [int(part) for part in parts[:-1]], parts[-1]


def _functions(module) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for op in module.walk():
        if op.name not in ("llvm.func", "func.func") or not op.regions or not op.regions[0].blocks:
            continue
        name = getattr(op.properties.get("sym_name") or op.attributes.get("sym_name"), "data", None)
        if isinstance(name, str):
            found[name] = op
    return found


def audit_group(site: GroupSite, function: Any | None, budget: Budget = Budget()) -> dict[str, Any]:
    """One group's verdict, with the numbers it rests on."""
    row: dict[str, Any] = {
        "group": site.group,
        "placement": site.placement,
        "symbol": site.symbol,
        "stages": list(site.stages),
        "elements": site.elements,
    }
    if function is None:
        row.update(verdict=NOT_EMITTED, why="no function with this symbol is in the emitted module")
        return row
    from merlin.perf.host_cfg_activity import analyze_host_cfg_activity

    activity = analyze_host_cfg_activity(function)
    dynamic = activity.get("dynamic_operations")
    row["activity_status"] = activity.get("status")
    if dynamic is None:
        row.update(
            verdict=UNKNOWN,
            why="dynamic host work could not be counted: "
            + "; ".join(activity.get("problems") or ["no reason recorded"])[:300],
        )
        return row
    arithmetic = sum(int(dynamic.get(name, 0)) for name in _ARITHMETIC)
    value_arithmetic = sum(int(dynamic.get(name, 0)) for name in _VALUE_ONLY)
    payload = (activity.get("load_payload_bytes") or 0) + (activity.get("store_payload_bytes") or 0)
    top = (activity.get("block_signatures") or [{}])[0]
    row.update(
        host_arithmetic=arithmetic,
        host_value_arithmetic=value_arithmetic,
        host_payload_bytes=payload,
        dominant_block={
            "signature": top.get("signature"),
            "trips": top.get("trips"),
            "dynamic_total": top.get("dynamic_total"),
        },
    )
    if site.placement == HOST:
        row.update(verdict=DECLARED_HOST, why="placed on the host; reported, not judged")
        return row
    if site.elements <= 0:
        row.update(verdict=UNKNOWN, why="the group's element count is unknown, so host work cannot be compared with it")
        return row
    per_element = arithmetic / site.elements
    value_per_element = value_arithmetic / site.elements
    payload_ratio = payload / (site.elements * max(site.element_bytes, 1))
    row.update(
        arithmetic_per_element=round(per_element, 4),
        value_arithmetic_per_element=round(value_per_element, 4),
        payload_ratio=round(payload_ratio, 4),
    )
    breaches = []
    if per_element > budget.arithmetic_per_element:
        breaches.append(
            f"{per_element:.2f} host arithmetic operations per element (budget {budget.arithmetic_per_element})"
        )
    if value_per_element > budget.value_arithmetic_per_element:
        breaches.append(
            f"{value_per_element:.2f} float or conversion operations per element "
            f"(budget {budget.value_arithmetic_per_element})"
        )
    if payload_ratio > budget.payload_bytes_per_element_byte:
        breaches.append(
            f"scalar loads and stores move {payload_ratio:.2f}x the group's output "
            f"through the host (budget {budget.payload_bytes_per_element_byte})"
        )
    if breaches:
        row.update(
            verdict=HOST_COMPUTE,
            why=f"placed on {site.placement} and the host still works at element scale: "
            + "; ".join(breaches)
            + f". Dominant block: {compact_signature(top.get('signature'))!r} "
            f"x{top.get('trips')}",
        )
    else:
        row.update(verdict=CLEAN, why="host work stays at command scale")
    return row


def compact_signature(signature: Any, *, limit: int = 160) -> str:
    """A block signature a person can read: runs of one operation, or of one pair, written once.

    A host loop a compiler unrolled is thousands of ``getelementptr store`` in one block, and a
    finding that quotes it whole pushes everything after it out of whatever carries the finding. The
    full signature stays on the row; this is what a sentence quotes.
    """
    tokens = str(signature or "").split()
    out: list[str] = []
    index = 0
    while index < len(tokens):
        for period in (1, 2):
            unit = tokens[index : index + period]
            repeats = 1
            while len(unit) == period and tokens[index + repeats * period : index + (repeats + 1) * period] == unit:
                repeats += 1
            if repeats > 1:
                out.append(f"({' '.join(unit)}) x{repeats}")
                index += repeats * period
                break
        else:
            out.append(tokens[index])
            index += 1
    text = " ".join(out)
    return text if len(text) <= limit else text[: limit - 4].rstrip() + " ..."


def audit(module: Any, sites: Sequence[GroupSite], budget: Budget = Budget()) -> dict[str, Any]:
    """Audit every group site against the functions of an emitted LLVM-dialect module."""
    functions = _functions(module)
    rows = [audit_group(site, functions.get(site.symbol), budget) for site in sites]
    judged = [row for row in rows if row["placement"] != HOST]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    findings = [row for row in judged if row["verdict"] == HOST_COMPUTE]
    return {
        "schema": SCHEMA,
        "groups": rows,
        "verdicts": dict(sorted(counts.items())),
        "accelerator_groups": len(judged),
        "accelerator_groups_clean": sum(1 for row in judged if row["verdict"] == CLEAN),
        "accelerator_groups_unknown": sum(1 for row in judged if row["verdict"] in (UNKNOWN, NOT_EMITTED)),
        "findings": [{"group": row["group"], "symbol": row["symbol"], "why": row["why"]} for row in findings],
        # The veto. Unknown is not a pass: a group nobody could count has not been shown clean.
        "vetoed": bool(findings),
        "proven_clean": bool(judged) and all(row["verdict"] == CLEAN for row in judged),
        "budget": {
            "arithmetic_per_element": budget.arithmetic_per_element,
            "value_arithmetic_per_element": budget.value_arithmetic_per_element,
            "payload_bytes_per_element_byte": budget.payload_bytes_per_element_byte,
        },
    }


class HostComputeVeto(RuntimeError):
    """An accelerator-placed group whose host code works at element scale."""


def require_clean(report: Mapping[str, Any]) -> None:
    if report["vetoed"]:
        first = report["findings"][0]
        raise HostComputeVeto(
            f"{len(report['findings'])} accelerator group(s) compute on the host; "
            f"group {first['group']} ({first['symbol']}): {first['why']}"
        )
