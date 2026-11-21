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
the counts come from the emitted LLVM-dialect function (:mod:`merlin.analysis.programs.cfg_activity`), the
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

# ---- host_compute_audit ------------------------------------------------------------------------

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
    #: Scalar parameter loads (a scale or stride read from memory only to configure the unit) a
    #: command-only kernel may execute per call. A kernel configures a handful of registers, once
    #: or per tile pass; a load repeated per output element exceeds any small bound at real sizes.
    command_parameter_loads: int = 64


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
    from merlin.compiler.ir import mlir_query as mq

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
    from merlin.analysis.programs.cfg_activity import analyze_host_cfg_activity

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
    address_loop_arithmetic = activity.get("dynamic_address_loop_control_arithmetic")
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
        from merlin.verification.host.convolution_inputs import prove_input_preparation

        preparation = prove_input_preparation(function, activity, input_preparation, target=target)
        row["input_preparation"] = preparation
        if preparation.get("status") == "proven":
            # All exclusions are actual closed address/copy execution counts;
            # ordinary budgets still judge every operation outside that proof.
            arithmetic -= preparation["arithmetic"] - address_arithmetic
            conversions = preparation.get("conversions")
            if type(conversions) is not int or conversions < 0:
                row.update(verdict=UNKNOWN, why="input preparation lacks a derived conversion count")
                return row
            arithmetic -= conversions
            value_arithmetic -= conversions
            payload -= preparation["payload_bytes"]
            if arithmetic < 0 or value_arithmetic < 0 or payload < 0:
                row.update(verdict=UNKNOWN, why="input preparation counts exceed original raw accounting")
                return row
    if payload > 0 and site.placement != HOST and target is not None:
        from merlin.verification.host.command_parameters import prove_command_parameters

        parameters = prove_command_parameters(function, activity, target=target)
        row["command_parameters"] = parameters
        if parameters.get("status") == "proven":
            arithmetic -= parameters["arithmetic"] + parameters["conversions"]
            value_arithmetic -= parameters["conversions"]
            payload -= parameters["payload_bytes"]
            if min(arithmetic, value_arithmetic, payload) < 0:
                row.update(verdict=UNKNOWN, why="command parameter proof exceeds original raw accounting")
                return row
    orchestration = None
    if site.placement != HOST and (
        arithmetic > site.elements * budget.arithmetic_per_element
        or value_arithmetic > site.elements * budget.value_arithmetic_per_element
        or payload > site.elements * max(site.element_bytes, 1) * budget.payload_bytes_per_element_byte
    ):
        orchestration = prove_command_orchestration(
            function, activity, target=target, parameter_loads=budget.command_parameter_loads
        )
        if orchestration["status"] == "proven" and payload != orchestration["parameter_payload_bytes"]:
            # Earlier proofs already excluded part of this traffic; the two accountings must agree.
            orchestration = {
                "status": "refused",
                "reason": f"{payload} bytes of host memory traffic are not the "
                f"{orchestration['parameter_payload_bytes']} bytes of proven configuration loads",
            }
        if orchestration["status"] != "proven":
            row["command_only_refusal"] = orchestration["reason"]
    command_only = orchestration is not None and orchestration["status"] == "proven"
    if command_only:
        # The raw dynamic count remains visible. Only the proven pointer-only
        # command orchestration is removed from the tensor-compute budget.
        row["host_command_orchestration_arithmetic"] = arithmetic
        row["host_command_orchestration_conversions"] = value_arithmetic
        row["host_command_parameter_loads"] = orchestration["parameter_loads"]
        row["host_command_parameter_payload_bytes"] = orchestration["parameter_payload_bytes"]
        arithmetic = 0
        value_arithmetic = 0
        payload = 0
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
    refusal = row.get("command_only_refusal")
    dominant = f"Dominant block: {compact_signature(top.get('signature'))!r} x{top.get('trips')}"
    if breaches and refusal and value_arithmetic == 0 and payload == 0:
        # No tensor value is computed or moved: the integer work is unproven, not shown to be compute.
        row.update(
            verdict=HOST_COMPUTE,
            why=f"placed on {site.placement}: {'; '.join(breaches)}, with no float arithmetic or scalar "
            f"memory traffic; the command-only exemption fails on {refusal[:160]}. {dominant}",
        )
    elif breaches:
        row.update(
            verdict=HOST_COMPUTE,
            why=f"placed on {site.placement} and the host still works at element scale: "
            + "; ".join(breaches)
            # Float or conversion work at element scale already shows the compute; otherwise
            # say which construct kept the integer work from counting as orchestration.
            + (
                f"; the command-only exemption fails on {refusal[:120]}"
                if refusal and value_per_element <= budget.value_arithmetic_per_element
                else ""
            )
            + f". {dominant}",
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

        # Preserve historical object lookup; implementation and completion state live here.


# ---- command_orchestration ---------------------------------------------------------------------
# Prove that a pointer-only kernel's host work is only target-command orchestration.
#
# The host-compute audit charges host arithmetic against the elements a group produces. A kernel
# that only walks tiles can still spend many integer instructions per element on transfer addresses
# and command operands without touching a tensor value. This proof decides, structurally, that a
# function cannot be computing on tensor values, so its integer work may leave the element budget:
#
# * every argument is a pointer, nothing is returned and nothing is stored;
# * every effect is a recognised target command (an ``.insn r`` of the target's custom opcode whose
#   funct the target's ISA facts name) or a plain fence. Each register field is parsed: the
#   destination is ``x0`` (no value comes back), each source is ``x0`` or one of the asm's own
#   operands, every operand is consumed, and every operand constraint is a plain register input.
#   Any other literal register is refused: the kernel never set it, so it could carry a host value;
# * a scalar memory load is admitted only as a configuration PARAMETER: its address derives from a
#   kernel argument, its value flows only through casts and bit packing into operands of the
#   target's configuration commands (the class the ISA facts name), never across a block boundary,
#   and the loads execute a bounded number of times per call, independent of the output size.
#
# Anything else refuses with the construct that defeated it, so a veto can say why. A refusal never
# makes a group clean or vetoed by itself: the caller falls back to the ordinary element budget.

#: Value-preserving or narrowing casts a loaded parameter may pass through on its way to a
#: command register (a float scale reinterpreted as its bits, or converted to an integer field).
_PARAMETER_CASTS = frozenset({"llvm.zext", "llvm.sext", "llvm.trunc", "llvm.bitcast", "llvm.fptosi", "llvm.fptoui"})
#: Bit packing that places a parameter into its field of a command operand.
_PARAMETER_PACKING = frozenset({"llvm.and", "llvm.or", "llvm.xor", "llvm.shl", "llvm.lshr", "llvm.ashr"})
#: The RISC-V host's spellings of the hard-wired zero register.
_ZERO_REGISTER = frozenset({"x0", "zero"})


class _Refused(Exception):
    """The construct that defeats the proof, in words a finding can quote."""


def prove_command_orchestration(function: Any, activity: dict, *, target: str | None, parameter_loads: int) -> dict:
    """``{"status": "proven", ...}`` with the admitted parameter loads, or ``{"status": "refused", "reason"}``."""
    try:
        return _prove(function, activity, target=target, parameter_loads=parameter_loads)
    except _Refused as exc:
        return {"status": "refused", "reason": str(exc)[:300]}


def _orchestration_ops() -> frozenset[str]:
    from merlin.analysis.programs.cfg_activity import _INTEGER

    return _INTEGER | {
        "llvm.func",
        "llvm.mlir.constant",
        "llvm.mlir.addressof",
        "llvm.getelementptr",
        "llvm.ptrtoint",
        "llvm.inttoptr",
        "llvm.bitcast",
        "llvm.icmp",
        "llvm.select",
        "llvm.trunc",
        "llvm.zext",
        "llvm.sext",
        "llvm.br",
        "llvm.cond_br",
        "llvm.return",
        # Stack scratch carries no host value: stores are refused and loads must come from a
        # kernel argument, so only the target's own commands can touch it.
        "llvm.alloca",
    }


def _isa(target: str | None) -> dict:
    if target is None:
        raise _Refused("no target is selected, so its commands cannot be recognised")
    from merlin.target.isa.rocc.decode import isa_constants

    try:
        isa = isa_constants(target)
        if not isinstance(isa["CUSTOM_OPCODE"], int) or not isinstance(isa["FUNCT_CLASS"], dict):
            raise TypeError("ISA facts lack an integer custom opcode or funct class table")
    except Exception as exc:  # noqa: BLE001 -- facts that cannot be read prove nothing: refuse
        raise _Refused(f"the target's ISA facts are unavailable ({type(exc).__name__})") from None
    return isa


def _int_field(text: str) -> int | None:
    try:
        return int(text, 0)
    except ValueError:
        return None


def _is_fence(template: str) -> bool:
    """A host memory fence, bare or with predecessor/successor sets (``fence iorw, iorw``).

    A fence orders memory and moves no value, so it is orchestration in any spelling the host
    ISA defines; anything else spelled ``fence`` is not recognised.
    """
    parts = template.split(None, 1)
    if not parts or parts[0] != "fence":
        return False
    if len(parts) == 1:
        return True
    sets = parts[1]
    fields = [field.strip() for field in sets.split(",")]
    return len(fields) == 2 and all(field and set(field) <= set("iorw") for field in fields)


def _command(op: Any, isa: dict) -> str | None:
    """The command class of one ``llvm.inline_asm``; ``None`` for a fence. Refuses anything else."""
    from xdsl.dialects.builtin import IntegerType, UnitAttr
    from xdsl.dialects.llvm import LLVMPointerType, LLVMVoidType

    from merlin.target.isa.rocc.decode import _asm_template

    template = _asm_template(op)
    void_result = len(op.results) == 1 and isinstance(op.results[0].type, LLVMVoidType) and not op.results[0].uses
    if op.results and not void_result:
        raise _Refused(f"inline asm {template[:60]!r} returns a value to the host")
    if not isinstance(op.has_side_effects, UnitAttr):
        raise _Refused(f"inline asm {template[:60]!r} is not marked as having side effects")
    constraints = [entry.strip() for entry in op.constraints.data.split(",")] if op.constraints.data else []
    inputs, clobbers = constraints[: len(op.operands)], constraints[len(op.operands) :]
    if inputs != ["r"] * len(op.operands) or not all(c.startswith("~{") and c.endswith("}") for c in clobbers):
        raise _Refused(f"inline asm constraints {op.constraints.data!r} are not plain register inputs and clobbers")
    if "~{memory}" not in clobbers:
        raise _Refused(f"inline asm {template[:60]!r} does not clobber memory, so its ordering is unproven")
    if any(not isinstance(v.type, (IntegerType, LLVMPointerType)) for v in op.operands):
        raise _Refused(f"inline asm {template[:60]!r} takes a non-integer register operand")
    if _is_fence(template):
        if op.operands:
            raise _Refused("a fence takes operands")
        return None
    fields = [field.strip() for field in template.split(",")]
    head = fields[0].split()
    if len(fields) != 6 or len(head) != 3 or head[:2] != [".insn", "r"]:
        raise _Refused(f"inline asm {template[:60]!r} is not one six-field '.insn r' command")
    opcode, funct3, funct = _int_field(head[2]), _int_field(fields[1]), _int_field(fields[2])
    if opcode is None or funct3 is None or funct is None:
        raise _Refused(f"command {template[:60]!r} has a non-integer opcode or funct field")
    if opcode != isa["CUSTOM_OPCODE"]:
        raise _Refused(f"inline asm {template[:60]!r} is not the target's custom opcode")
    label = isa["FUNCT_CLASS"].get(funct)
    if label is None:
        raise _Refused(f"funct {funct} in {template[:60]!r} is not a command the target defines")
    rd, sources = fields[3], fields[4:]
    if rd not in _ZERO_REGISTER:
        raise _Refused(f"command {template[:60]!r} writes register {rd!r} back to the host")
    used = set()
    for source in sources:
        if source in _ZERO_REGISTER:
            continue
        index = _int_field(source[1:]) if source.startswith("$") and source[1:].isdigit() else None
        if index is None:
            raise _Refused(
                f"command {template[:60]!r} reads literal register {source!r}, which the kernel never set "
                "and which could carry a host value"
            )
        if index >= len(op.operands):
            raise _Refused(f"command {template[:60]!r} names operand {source!r} the asm does not have")
        used.add(index)
    if used != set(range(len(op.operands))):
        raise _Refused(f"command {template[:60]!r} leaves asm operands unconsumed")
    return str(label)


def _argument_root(pointer: Any, entry: Any) -> bool:
    from xdsl.ir import BlockArgument

    while not isinstance(pointer, BlockArgument):
        owner = pointer.owner
        if getattr(owner, "name", None) != "llvm.getelementptr":
            return False
        pointer = owner.operands[0]
    return pointer.block is entry


def _parameter_slice(load: Any, commands: dict, config_class: Any, entry: Any) -> set:
    """The casts and packing between a parameter load and the configuration commands it feeds."""
    from xdsl.dialects.builtin import IntegerType, VectorType
    from xdsl.dialects.llvm import LLVMPointerType

    from merlin.analysis.programs.cfg_activity import _payload_bytes, _real_name

    value = load.results[0]
    if (
        isinstance(value.type, (LLVMPointerType, VectorType))
        or _payload_bytes(value.type) is None
        or set(load.properties) - {"ordering", "alignment"}
        or load.attributes
        or getattr(getattr(load.properties.get("ordering"), "value", None), "data", 0) != 0
    ):
        raise _Refused("a memory load that is not one plain scalar read")
    if not _argument_root(load.operands[0], entry):
        raise _Refused("a memory load whose address does not derive from a kernel argument")
    if not value.uses:
        raise _Refused("a memory load whose value reaches no command")
    sliced, queue = set(), [value]
    while queue:
        current = queue.pop()
        for use in current.uses:
            op, name = use.operation, _real_name(use.operation)
            if name in _PARAMETER_CASTS | _PARAMETER_PACKING:
                if not all(isinstance(r.type, IntegerType) or _payload_bytes(r.type) for r in op.results):
                    raise _Refused(f"memory load feeding {name} with a non-scalar result")
                if op not in sliced:
                    sliced.add(op)
                    queue.extend(op.results)
            elif name == "llvm.inline_asm" and op in commands:
                if config_class is None or commands[op] != config_class:
                    raise _Refused(
                        f"memory load feeding a {commands[op] or 'fence'} operand; only configuration "
                        "commands take loaded parameters"
                    )
            else:
                raise _Refused(f"unproven memory load feeding {name}")
    return sliced


def _prove(function: Any, activity: dict, *, target: str | None, parameter_loads: int) -> dict:
    from xdsl.dialects.llvm import LLVMPointerType

    from merlin.analysis.programs.cfg_activity import _payload_bytes, _real_name

    isa = _isa(target)
    blocks = tuple(function.body.blocks)
    if not blocks or not blocks[0].args:
        raise _Refused("the kernel has no pointer arguments")
    for index, arg in enumerate(blocks[0].args):
        if not isinstance(arg.type, LLVMPointerType):
            raise _Refused(f"argument {index} is a {arg.type} value, not a pointer")
    allowed = _orchestration_ops()
    commands: dict[Any, str | None] = {}
    loads = []
    for op in function.walk():
        name = _real_name(op)
        if name == "llvm.inline_asm":
            commands[op] = _command(op, isa)
        elif name == "llvm.load":
            loads.append(op)
        elif name == "llvm.store":
            raise _Refused("a host memory store (llvm.store)")
        elif name == "llvm.return" and op.operands:
            raise _Refused("the kernel returns a host value")
    if not any(label is not None for label in commands.values()):
        raise _Refused("no target command is issued")
    sliced: set = set()
    for load in loads:
        sliced |= _parameter_slice(load, commands, isa.get("CONFIG_CLASS"), blocks[0])
    for op in function.walk():
        name = _real_name(op)
        if name not in allowed and name not in ("llvm.inline_asm", "llvm.load") and op not in sliced:
            raise _Refused(f"host operation {name} is not command orchestration")
    counts = {}
    for row in activity.get("blocks") or ():
        if type(row.get("block")) is int and 0 <= row["block"] < len(blocks):
            counts[blocks[row["block"]]] = row.get("execution_count")
    executions = 0
    payload = 0
    for load in loads:
        count = counts.get(load.parent_block())
        if type(count) is not int or count < 0:
            raise _Refused("a parameter load has no derived execution count")
        executions += count
        payload += count * _payload_bytes(load.results[0].type)
    if executions > parameter_loads:
        raise _Refused(
            f"{executions} parameter loads per call exceed the bound of {parameter_loads}: a load "
            "repeated with the output is payload, not configuration"
        )
    return {
        "status": "proven",
        "commands": sum(1 for label in commands.values() if label is not None),
        "parameter_loads": executions,
        "parameter_payload_bytes": payload,
    }

    # Preserve historical object lookup; implementation and completion state live here.
