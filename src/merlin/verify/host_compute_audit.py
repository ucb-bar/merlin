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
Only independently proved pointer-plus-constant setup is excluded from the compute
budget. The raw arithmetic and excluded address counts remain visible: a narrow
output may require many commands without any host arithmetic on tensor values.

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


def _command_only_integer_orchestration(function: Any, target: str | None) -> bool:
    """Prove a high instruction count cannot be host tensor computation.

    A pointer-only kernel with no scalar memory access or value arithmetic can
    spend many integer instructions calculating transfer addresses and command
    operands. Its only effects must be target-owned RoCC commands and fences;
    unrecognized inline assembly, helpers, scalar memory, or a scalar return
    keeps the ordinary element-scale veto in force.
    """
    if target is None:
        return False
    from xdsl.dialects.builtin import UnitAttr
    from xdsl.dialects.llvm import LLVMPointerType, LLVMVoidType

    from merlin.perf.host_cfg_activity import _INTEGER, _real_name
    from merlin.targetgen.rocc.decode import _asm_template, _parse_insn, isa_constants

    try:
        isa = isa_constants(target)
        opcode, classes = isa["CUSTOM_OPCODE"], isa["FUNCT_CLASS"]
    except (KeyError, TypeError, ValueError):
        return False
    blocks = tuple(function.body.blocks)
    if not blocks or not blocks[0].args or not all(isinstance(arg.type, LLVMPointerType) for arg in blocks[0].args):
        return False
    commands = 0
    scalar_ops = _INTEGER | {
        "llvm.func", "llvm.mlir.constant", "llvm.mlir.addressof", "llvm.getelementptr",
        "llvm.ptrtoint", "llvm.inttoptr", "llvm.bitcast", "llvm.icmp", "llvm.select",
        "llvm.trunc", "llvm.zext", "llvm.sext", "llvm.br", "llvm.cond_br", "llvm.return",
    }
    for op in function.walk():
        name = _real_name(op)
        if name not in scalar_ops and name != "llvm.inline_asm":
            return False
        if name == "llvm.return" and op.operands:
            return False
        if name != "llvm.inline_asm":
            continue
        void_result = (len(op.results) == 1 and isinstance(op.results[0].type, LLVMVoidType)
                       and not op.results[0].uses)
        if (
            (op.results and not void_result) or not isinstance(op.has_side_effects, UnitAttr)
            or "~{memory}" not in op.constraints.data.split(",")
        ):
            return False
        template = _asm_template(op)
        if template == "fence":
            if op.operands:
                return False
            continue
        parsed = _parse_insn(template, opcode)
        if parsed is None or not parsed[1] or parsed[0] not in classes or len(op.operands) != 2:
            return False
        commands += 1
    return commands > 0


def audit_group(
    site: GroupSite, function: Any | None, budget: Budget = Budget(), *, input_preparation=None, target=None
) -> dict[str, Any]:
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
    if dynamic is None or activity.get("status") != "derived":
        row.update(
            verdict=UNKNOWN,
            activity_problems=list(activity.get("problems") or [])[:32],
            why="dynamic host work or scalar memory payload could not be counted: "
            + "; ".join(activity.get("problems") or ["unknown scalar memory payload"])[:300],
        )
        return row
    # Pointer-plus-constant setup is permitted command work, not tensor compute.
    # Keep the raw instruction totals visible; exclude only the independently
    # proved SSA subset from the compute budget, never all integer arithmetic.
    address_arithmetic = activity.get("dynamic_pointer_setup_arithmetic")
    if not isinstance(address_arithmetic, int) or isinstance(address_arithmetic, bool) or address_arithmetic < 0:
        row.update(verdict=UNKNOWN, why="pointer-setup arithmetic has no derived execution count")
        return row
    address_loop_arithmetic = activity.get("dynamic_address_loop_control_arithmetic", 0)
    if type(address_loop_arithmetic) is not int or address_loop_arithmetic < 0:
        row.update(verdict=UNKNOWN, why="address-loop control has no derived execution count")
        return row
    pointer_arithmetic = address_arithmetic
    address_arithmetic += address_loop_arithmetic
    total_arithmetic = sum(int(dynamic.get(name, 0)) for name in _ARITHMETIC)
    arithmetic = total_arithmetic - address_arithmetic
    if arithmetic < 0:
        row.update(verdict=UNKNOWN, why="proved address/control counts exceed raw arithmetic")
        return row
    value_arithmetic = sum(int(dynamic.get(name, 0)) for name in _VALUE_ONLY)
    payload = (activity.get("load_payload_bytes") or 0) + (activity.get("store_payload_bytes") or 0)
    preparation = None
    if input_preparation is not None and site.placement != HOST:
        from merlin.verify.conv_input_preparation import prove_input_preparation

        preparation = prove_input_preparation(function, activity, input_preparation, target=target)
        row["input_preparation"] = preparation
        if preparation.get("status") == "proven":
            # All exclusions are actual closed address/copy execution counts;
            # ordinary budgets still judge every operation outside that proof.
            arithmetic -= preparation["arithmetic"] - address_arithmetic
            arithmetic -= preparation.get("conversions", 0)
            value_arithmetic -= preparation.get("conversions", 0)
            payload -= preparation["payload_bytes"]
            if arithmetic < 0 or value_arithmetic < 0 or payload < 0:
                row.update(verdict=UNKNOWN, why="input preparation counts exceed original raw accounting")
                return row
    command_only = (
        site.placement != HOST
        and (arithmetic > site.elements * budget.arithmetic_per_element
             or value_arithmetic > site.elements * budget.value_arithmetic_per_element)
        and payload == 0
        and _command_only_integer_orchestration(function, target)
    )
    if command_only:
        # The raw dynamic count remains visible. Only the proven pointer-only
        # command orchestration is removed from the tensor-compute budget.
        row["host_command_orchestration_arithmetic"] = arithmetic
        row["host_command_orchestration_conversions"] = value_arithmetic
        arithmetic = 0
        value_arithmetic = 0
    top = (activity.get("block_signatures") or [{}])[0]
    row.update(
        arithmetic_basis=("pointer_only_target_commands_v1" if command_only else "pointer_offset_provenance_v1"),
        address_loop_basis="closed_induction_address_uses_v1",
        host_arithmetic=arithmetic,
        host_total_arithmetic=total_arithmetic,
        host_address_arithmetic=address_arithmetic,
        host_pointer_setup_arithmetic=pointer_arithmetic,
        host_address_loop_control_arithmetic=address_loop_arithmetic,
        host_value_arithmetic=value_arithmetic,
        host_payload_bytes=payload,
        host_raw_payload_bytes=(activity.get("load_payload_bytes") or 0) + (activity.get("store_payload_bytes") or 0),
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


def audit(
    module: Any, sites: Sequence[GroupSite], budget: Budget = Budget(), *, input_preparation=None, target=None
) -> dict[str, Any]:
    """Audit every group site against the functions of an emitted LLVM-dialect module."""
    functions = _functions(module)
    rows = [
        audit_group(
            site,
            functions.get(site.symbol),
            budget,
            input_preparation=(input_preparation or {}).get(site.symbol),
            target=target,
        )
        for site in sites
    ]
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
