"""Target-owned declarations for source-specific RTL fact readers.

Shared extractors can parse several hardware classes. The vocabulary locating a source's
instruction span and accumulator ports belongs to the selected target contract, not to Merlin.
Missing declarations disable only that optional fallback; malformed declarations fail closed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

# ---- extraction_contract -----------------------------------------------------------------------


def _rtl_extraction(target: str) -> dict[str, Any]:
    import yaml

    from merlin.target.rtl.facts import target_contract_path

    path = target_contract_path(target)
    if not path.is_file():
        return {}
    contract = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(contract, dict):
        raise ValueError(f"{path}: target contract must be a mapping")
    extraction = contract.get("rtl_extraction") or {}
    if not isinstance(extraction, dict):
        raise ValueError(f"{path}: rtl_extraction must be a mapping")
    return extraction


def scala_funct_block(target: str) -> tuple[str, str] | None:
    """The target's Scala funct-span anchors, or no header fallback when undeclared."""
    block = _rtl_extraction(target).get("scala_funct_block")
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("scala_funct_block must be a mapping")
    start, stop = block.get("start_comment"), block.get("stop_declaration")
    if (
        not isinstance(start, str)
        or not start.strip().startswith("//")
        or not isinstance(stop, str)
        or not stop.isidentifier()
    ):
        raise ValueError("scala_funct_block needs a Scala comment and identifier stop_declaration")
    return start, stop


def accumulator_layout(target: str) -> dict[str, str] | None:
    """The target's optional accumulator HW-port vocabulary, never a guessed module name."""
    layout = _rtl_extraction(target).get("accumulator_memory")
    if layout is None:
        return None
    required = ("module", "memory_name", "address_port", "data_prefix", "data_suffix", "mask_prefix")
    if not isinstance(layout, dict) or any(not isinstance(layout.get(key), str) or not layout[key] for key in required):
        raise ValueError(f"accumulator_memory needs nonempty {', '.join(required)}")
    result = {key: layout[key] for key in required}
    field = layout.get("firrtl_data_field")
    if field is not None:
        if not isinstance(field, str) or not all(part.isidentifier() for part in field.split(".")):
            raise ValueError("accumulator_memory.firrtl_data_field must be a dotted FIRRTL port path")
        result["firrtl_data_field"] = field
    return result


def firrtl_role_probe(target: str) -> dict[str, Any] | None:
    """Optional source-specific role anchors declared by the selected support provider.

    The reader implements FIRRTL parsing; module names, source markers, and published role
    names belong to the target. Missing declarations leave the scoped census in charge.
    """
    probe = _rtl_extraction(target).get("firrtl_role_probe")
    if probe is None:
        return None
    if not isinstance(probe, dict):
        raise ValueError("firrtl_role_probe must be a mapping")
    shapes = {
        "array": ("name", "parent_module_contains", "child_module_prefix"),
        "memory": ("name", "declaration_contains", "source_contains", "datapath_name"),
    }
    for kind, fields in shapes.items():
        block = probe.get(kind)
        if block is not None and (
            not isinstance(block, dict)
            or any(not isinstance(block.get(field), str) or not block[field] for field in fields)
        ):
            raise ValueError(f"firrtl_role_probe.{kind} needs nonempty {', '.join(fields)}")
    memory = probe.get("memory")
    if memory is not None and not memory["declaration_contains"].endswith("UInt<"):
        raise ValueError("firrtl_role_probe.memory.declaration_contains must end in UInt<")
    interfaces = probe.get("interfaces", [])
    if not isinstance(interfaces, list) or any(
        not isinstance(item, dict)
        or any(not isinstance(item.get(field), str) or not item[field] for field in ("module", "name", "evidence"))
        for item in interfaces
    ):
        raise ValueError("firrtl_role_probe.interfaces needs module, name, and evidence strings")
    if not probe.get("array") and not memory and not interfaces:
        raise ValueError("firrtl_role_probe must declare at least one role anchor")
    return probe


def boolean_feature_gate(target: str) -> dict[str, str] | None:
    """One optional FIRRTL build gate whose feature name and source names are target-owned."""
    gate = _rtl_extraction(target).get("boolean_feature_gate")
    if gate is None:
        return None
    required = ("feature", "module", "node", "dynamic_operand_contains", "config_field")
    if not isinstance(gate, dict) or any(not isinstance(gate.get(key), str) or not gate[key] for key in required):
        raise ValueError(f"boolean_feature_gate needs nonempty {', '.join(required)}")
    return {key: gate[key] for key in required}


# ---- feature_gates -----------------------------------------------------------------------------
# Exact structural FIRRTL boolean feature-gate parsing; missing stays unknown.


def _firrtl_bool_literal(expr: str) -> bool | None:
    """A FIRRTL one-bit UInt literal, parsed exactly (not substring/regex matched)."""
    expr = expr.strip()
    prefix = "UInt<1>("
    if not expr.startswith(prefix) or not expr.endswith(")"):
        return None
    token = expr[len(prefix) : -1].strip().lower()
    try:
        value = int(token[2:], 16) if token.startswith("0h") else int(token, 10)
    except ValueError:
        return None
    return bool(value) if value in (0, 1) else None


def _firrtl_call_args(expr: str, callee: str) -> list[str] | None:
    """Top-level operands of one FIRRTL primitive call, preserving nested expressions."""
    expr = expr.strip()
    head = f"{callee}("
    if not expr.startswith(head) or not expr.endswith(")"):
        return None
    body = expr[len(head) : -1]
    args: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return None
        elif char == "," and depth == 0:
            args.append(body[start:i].strip())
            start = i + 1
    if depth != 0:
        return None
    args.append(body[start:].strip())
    return args


def extract_boolean_feature_gate(fir_text: str, *, gate: dict[str, str]) -> dict[str, Any] | None:
    """Read a declared FIRRTL ``and`` build gate with one literal and one dynamic operand.

    Missing, duplicate or differently-shaped nodes are unknown, not inferred from adjacent ISA fields.
    """
    found: list[dict[str, Any]] = []
    prefix = f"node {gate['node']} = "
    module: str | None = None
    for line_no, raw in enumerate(fir_text.splitlines(), 1):
        code = raw.split("@[", 1)[0].strip()
        if code.startswith("module ") and " :" in code:
            module = code[len("module ") : code.index(" :")].strip()
            continue
        if module != gate["module"]:
            continue
        if not code.startswith(prefix):
            continue
        expr = code[len(prefix) :].strip()
        args = _firrtl_call_args(expr, "and")
        if args is None or len(args) != 2:
            continue
        literals = [(idx, _firrtl_bool_literal(arg)) for idx, arg in enumerate(args)]
        literals = [(idx, val) for idx, val in literals if isinstance(val, bool)]
        if len(literals) != 1:
            continue
        literal_index, value = literals[0]
        dynamic = args[1 - literal_index]
        # The other operand must be the target-declared dynamic enable, not merely a second literal.
        if gate["dynamic_operand_contains"].lower() not in dynamic.lower():
            continue
        found.append({"value": value, "line": line_no, "expression": expr})
    if len(found) != 1:
        return None
    return found[0]


# ---- memory_roles ------------------------------------------------------------------------------
# Name scoped-census memories by the target's declared roles, keeping the physical inventory.
#
# The census names a store after the Chisel file and declaration that emitted it (``scratchpad.mem``);
# consumers address stores by the role the target declares for them (``scratchpad``,
# ``accumulator``). Both names describe one piece of storage, so a correctly scoped census must not
# rename a role or report one store twice:
#
# * a census row whose geometry equals a declared role-probe store (same banks, depth, row
#   elements, element bits and bytes, declared at the probe's source site) takes the role's name and
#   the address-space view tag, keeping its census name and evidence;
# * a census row that physically backs a declared address-space view (equal depth, row bits and
#   total bytes, one copy per view bank) moves to ``census.physical_memories`` with ``backs`` naming
#   that view, so the role view is not double-counted;
# * every other census row stays a memory under its census name.
#
# Nothing is dropped: the complete physical inventory is kept under ``census.physical_memories``.
# When the census is empty (an unscoped elaboration), the declared role-probe rows remain the
# memories, exactly as before.

_GEOMETRY = ("banks", "depth", "row_elems", "elem_bits", "bytes")


def _site(row: dict) -> str | None:
    evidence = row.get("evidence")
    if not isinstance(evidence, str) or " @ " not in evidence:
        return None
    return evidence.rpartition(" @ ")[2].split()[0] if evidence.rpartition(" @ ")[2].split() else None


def label_roles(census: list[dict], probes: list[dict]) -> list[dict]:
    """Census memories with declared role-probe names applied (or the probes when no census)."""
    declared = [row for row in probes or () if row.get("bytes") is not None]
    if not census:
        return [{**row, "source": "selected_firrtl_role_probe", "view_kind": "address_space"} for row in declared]
    out = []
    for row in census:
        matches = [
            probe
            for probe in declared
            if all(row.get(key) == probe.get(key) for key in _GEOMETRY)
            and _site(probe) is not None
            and _site(probe) in (row.get("evidence") or "")
        ]
        if len(matches) == 1:
            out.append(
                {
                    **row,
                    "name": matches[0]["name"],
                    "census_name": row["name"],
                    "view_kind": "address_space",
                    "role_basis": "census geometry equals the declared role-probe store at its source site",
                }
            )
        else:
            out.append(row)
    return out


def separate_backings(memories: list[dict], views: list[dict], facts: dict[str, Any]) -> list[dict]:
    """Move census rows that physically back a declared view into ``census.physical_memories``."""
    kept, physical = [], []
    for row in memories:
        if row.get("source") != "firrtl_census":
            kept.append(row)
            continue
        physical.append(row)
        copies = row.get("copies", 1) * row.get("banks", 1)
        backs = [
            view
            for view in views
            if view.get("view_kind") == "address_space"
            and view.get("source") != "firrtl_census"
            and view.get("depth") == row.get("depth")
            and view.get("row_bytes") is not None
            and view["row_bytes"] * 8 == row.get("row_bits_rtl")
            and view.get("banks") == copies
            and view.get("bytes") == copies * (row.get("bytes") or 0)
        ]
        if len(backs) == 1:
            physical[-1] = {**row, "backs": backs[0]["name"], "backing_basis": "equal depth, row bits and total bytes"}
        else:
            kept.append(row)
    census = facts.setdefault("census", {})
    if census.get("unit_root") is not None:  # an unscoped census has no inventory to state
        census["physical_memories"] = physical
    return kept


# ---- firrtl_memory_lines -----------------------------------------------------------------------
# Read explicit lowered FIRRTL memory blocks without changing hardware sources.
#
# The census consumes one declaration line per memory. Lowered FIRRTL instead
# stores type and depth on indented property lines; this streaming observer joins
# those exact fields into the same structural memory vocabulary. It never guesses
# element decomposition, numeric formats, latency behavior or memory roles.


def _declaration(header: str, properties: list[str]) -> str:
    stripped = header.strip()
    lhs, separator, annotation = stripped.partition(":")
    if not separator or len(lhs.split()) != 2 or lhs.split()[0] != "mem":
        raise ValueError("malformed lowered FIRRTL memory declaration")
    selected = {}
    for line in properties:
        key, arrow, value = line.strip().partition("=>")
        if not arrow:
            raise ValueError("malformed lowered FIRRTL memory property")
        key, value = key.strip(), value.strip().split("@[", 1)[0].strip()
        if key in {"data-type", "depth"}:
            if key in selected:
                raise ValueError("duplicate lowered FIRRTL memory type or depth")
            selected[key] = value
    depth, typ = selected.get("depth"), selected.get("data-type")
    if depth is None or not depth.isdecimal() or int(depth) <= 0 or not typ:
        raise ValueError("lowered FIRRTL memory needs explicit positive depth and data type")
    # This is a structural observation string, never replacement FIRRTL.
    # Existing type parsing preserves an opaque/noninteger row as unknown.
    return header[: len(header) - len(header.lstrip())] + lhs + " : " + typ + " [" + depth + "] " + annotation + "\n"


def observed_memory_lines(lines: Iterable[str]) -> Iterator[str]:
    """Yield original lines, joining only exact lowered ``mem`` property blocks."""
    header = None
    properties = []
    indentation = 0
    for raw in lines:
        stripped = raw.strip()
        level = len(raw) - len(raw.lstrip())
        if header is not None:
            if stripped and level > indentation:
                properties.append(raw)
                continue
            if not stripped:
                continue
            yield _declaration(header, properties)
            header, properties = None, []
        if stripped.startswith("mem "):
            header, indentation = raw, level
        else:
            yield raw
    if header is not None:
        yield _declaration(header, properties)
