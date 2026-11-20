"""Independently check retained source program phases, without semantic grants.

This reader reconstructs original source scope, live results and phase payload
from typed generic IR. It imports no frontend producer or reference evaluator.
The executable memory body and ordinary native ownership gates remain required.
"""

from xdsl.dialects import builtin
from xdsl.parser import Parser

_SPACES = ((1, "local"), (2, "global_read"), (4, "global_write"), (8, "tensor_read"), (16, "tensor_write"))
_PROFILE = "t2m.synchronous-program-barrier.v1"


def _name(op):
    return op.op_name.data if isinstance(op, builtin.UnregisteredOp) else op.name


def _ids(location):
    if isinstance(location, builtin.NameLoc):
        return [location.desc.data, *_ids(location.location)]
    if isinstance(location, builtin.FusedLoc):
        return list(dict.fromkeys(s for item in location.locations for s in _ids(item)))
    if isinstance(location, builtin.CallSiteLoc):
        return list(dict.fromkeys([*_ids(location.callee), *_ids(location.caller)]))
    return []


def checked_program_barriers(graph, reports, generic, protected):
    """Reconstruct the explicit optional phase extension from actual source."""
    try:
        _checked_program_barriers(graph, reports, generic, protected)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("malformed allocation source barrier protection") from error


def _checked_program_barriers(graph, reports, generic, protected):
    records = graph["operations"]
    required = [row for row in records if row.get("opcode") == "ttg.barrier"]
    from .linalg_mlir import make_context

    original = Parser(make_context(), generic.decode()).parse_module()
    present = any(_name(op) == "ttg.barrier" for op in original.walk())
    if not present and not required and reports is None:
        return
    if not present or not required or not isinstance(reports, list) or not reports:
        raise ValueError("allocation source barriers require complete phase declarations")
    retained = Parser(make_context(), protected.decode()).parse_module()
    ops = [op for op in original.walk() if op is not original]
    if (
        len(ops) != len(records)
        or any(_name(op) != row.get("opcode") for op, row in zip(ops, records, strict=True))
        or len({row.get("id") for row in records}) != len(records)
    ):
        raise ValueError("allocation barrier inventory differs from actual original operations")
    identities = {op: row["id"] for op, row in zip(ops, records, strict=True)}
    effects = {"tt.load", "tt.store", "tt.call", "tt.assert", "cf.assert", "tt.print"}
    effect_ids = {identities[op] for op in ops if _name(op) in effects}
    source = {identities[op]: op for op in ops if _name(op) == "ttg.barrier"}
    if len(reports) != len(source):
        raise ValueError("allocation barrier phase roster differs from its source")
    for op in original.walk():
        ctas = op.attributes.get("ttg.num-ctas", op.properties.get("ttg.num-ctas"))
        if ctas is not None and (not isinstance(ctas, builtin.IntegerAttr) or ctas.value.data != 1):
            raise ValueError("allocation source barriers need single-CTA participation")
        if (
            _name(op).startswith(("ttg.", "ttng.", "gpu.", "nvvm.", "rocdl."))
            and _name(op) != "ttg.barrier"
            or _name(op) in {"tt.atomic_rmw", "tt.atomic_cas", "tt.elementwise_inline_asm", "tt.inline_asm"}
        ):
            raise ValueError("allocation source barrier has an unqualified physical/effect composition")
        arguments = [arg for region in op.regions for block in region.blocks for arg in block.args]
        if any(
            isinstance(v.type, builtin.TensorType) and not isinstance(v.type.encoding, builtin.NoneAttr)
            for v in (*op.operands, *op.results, *arguments)
        ):
            raise ValueError("allocation source barrier requires pre-layout values")
    phases = {}
    for op in retained.walk():
        if _name(op).startswith(("ttg.", "ttng.", "gpu.", "nvvm.", "rocdl.")):
            raise ValueError("protected program barrier retains an unresolved physical operation")
        selected = set(_ids(op.location)) & set(source)
        if _name(op) == "scf.execute_region" and selected:
            if len(selected) != 1 or next(iter(selected)) in phases:
                raise ValueError("allocation barrier has duplicate or ambiguous phases")
            phases[next(iter(selected))] = op
    if set(phases) != set(source):
        raise ValueError("allocation source barrier lost an executable phase")

    original_order = [(_name(op), identities[op]) for op in ops if _name(op) in effects]
    retained_order = []
    for child in retained.walk():
        if _name(child) not in effects:
            continue
        selected = set(_ids(child.location)) & effect_ids
        if not selected and _name(child) != "cf.assert" or len(selected) > 1:
            raise ValueError("allocation barrier contains an unaccounted source memory effect")
        if selected:
            retained_order.append((_name(child), next(iter(selected))))
    if retained_order != original_order:
        raise ValueError("allocation barrier protection changed original source memory order")

    def position(op, *, retained=False):
        path = []
        while op.parent_op() is not None:
            block, region, parent = op.parent_block(), op.parent_region(), op.parent_op()
            parent_ids = set(_ids(parent.location)) & set(identities.values()) if retained else {identities.get(parent)}
            if parent is not original and _name(parent) != "builtin.module" and not parent_ids & set(source):
                if len(parent_ids) != 1 or None in parent_ids:
                    raise ValueError("allocation phase has an ambiguous source control ancestor")
                path.append(
                    (next(iter(parent_ids)), list(parent.regions).index(region), list(region.blocks).index(block))
                )
            op = parent
        return path

    seen = set()
    for report in reports:
        fields = {
            "kind",
            "source_ids",
            "contract",
            "phase_source_ids",
            "live_value_types",
            "representation",
            "ownership",
        }
        if not isinstance(report, dict) or set(report) != fields or not isinstance(report.get("contract"), dict):
            raise ValueError("unknown allocation source barrier phase declaration")
        for key in ("source_ids", "phase_source_ids", "live_value_types"):
            if not isinstance(report[key], list) or any(not isinstance(value, str) for value in report[key]):
                raise ValueError("allocation source barrier fields require typed string lists")
        if len(set(report["source_ids"])) != len(report["source_ids"]) or len(set(report["phase_source_ids"])) != len(
            report["phase_source_ids"]
        ):
            raise ValueError("allocation source barrier identities must be unique")
        source_id = report["contract"].get("source_id")
        if source_id not in source or source_id in seen:
            raise ValueError("allocation barrier phase has an unknown or duplicate original identity")
        seen.add(source_id)
        op, phase = source[source_id], phases[source_id]
        space = op.properties.get("addrSpace")
        if (
            op.operands
            or op.results
            or op.regions
            or op.successors
            or set(op.properties) != {"addrSpace"}
            or not isinstance(space, builtin.IntegerAttr)
            or space.type != builtin.i32
            or not 0 <= space.value.data <= 31
            or any(not key.startswith(("prov.", "op_name")) for key in op.attributes)
        ):
            raise ValueError("allocation source barrier differs from its pinned typed fields")
        parent = op.parent_op()
        while parent is not None and parent is not original:
            if _name(parent) not in {"tt.func", "scf.for", "scf.if", "scf.while"} or any(
                len(region.blocks) != 1 for region in parent.regions
            ):
                raise ValueError("allocation barrier participation needs whole-program structured control")
            parent = parent.parent_op()
        expected_contract = {
            "schema": _PROFILE,
            "source_id": source_id,
            "address_space_mask": space.value.data,
            "address_spaces": [name for bit, name in _SPACES if space.value.data & bit],
            "participants": "complete-pre-layout-program; single-CTA",
            "ordering": "complete-synchronous-tile-operations-at-source-control-position",
            "external_participants": "not-admitted",
            "physical_mapping": "undecided",
        }
        row = records[ops.index(op)]
        if (
            report["contract"] != expected_contract
            or row.get("program_barrier") != expected_contract
            or row.get("properties") != {"addrSpace": str(space)}
        ):
            raise ValueError("allocation barrier contract differs from actual original scope")
        prefix = []
        for sibling in op.parent_block().ops:
            if sibling is op:
                break
            if _name(sibling) == "ttg.barrier":
                prefix.clear()
            else:
                prefix.append(sibling)
        inside = {nested for sibling in prefix for nested in sibling.walk()}
        live = [
            value
            for sibling in prefix
            for value in sibling.results
            if any(u.operation not in inside for u in value.uses)
        ]
        prefix_ids = {identities[item] for item in inside}
        if (
            report["kind"] != "synchronous-program-memory-phase"
            or not isinstance(report["source_ids"], list)
            or source_id not in report["source_ids"]
            or set(report["phase_source_ids"]) != prefix_ids
            or report["live_value_types"] != [str(v.type) for v in live]
            or report["representation"] != "scf.execute_region; full tensor versions or synchronous standard memory"
            or report["ownership"] != "unchanged source bounds/initialization/program-conflict gates"
        ):
            raise ValueError("allocation barrier phase differs from its actual original prefix or live results")
        marker = phase.properties.get("no_inline", phase.attributes.get("no_inline"))
        if (
            len(phase.regions) != 1
            or len(phase.regions[0].blocks) != 1
            or phase.regions[0].block.args
            or not isinstance(marker, builtin.UnitAttr)
            or position(op) != position(phase, retained=True)
            or [str(v.type) for v in phase.results] != report["live_value_types"]
        ):
            raise ValueError("allocation barrier changed its executable phase scope or results")
        terminator = phase.regions[0].block.last_op
        if (
            terminator is None
            or _name(terminator) != "scf.yield"
            or [v.type for v in terminator.operands] != [v.type for v in phase.results]
        ):
            raise ValueError("allocation barrier phase lacks its complete typed yield")
        observed = {sid for child in phase.walk() for sid in _ids(child.location)}
        if not prefix_ids <= observed:
            raise ValueError("allocation barrier phase lost original source payload")
        before = [
            (_name(child), identities[child])
            for sibling in prefix
            for child in sibling.walk()
            if _name(child) in effects
        ]
        after = []
        for child in phase.walk():
            if _name(child) not in effects:
                continue
            selected = set(_ids(child.location)) & effect_ids
            if not selected and _name(child) != "cf.assert" or len(selected) > 1:
                raise ValueError("allocation barrier contains an unaccounted source memory effect")
            if selected:
                after.append((_name(child), next(iter(selected))))
        if before != after:
            raise ValueError("allocation barrier phase changed original source memory effects or order")
