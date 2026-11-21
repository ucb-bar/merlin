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


def _checked_cfg_participation(function):
    """Reprove scalar whole-program source edges without producer imports."""
    from xdsl.utils.exceptions import VerifyException

    if len(function.regions) != 1 or not function.regions[0].blocks:
        raise ValueError("allocation barrier CFG requires one nonempty source function region")
    blocks = set(function.regions[0].blocks)
    for block in blocks:
        terminal = block.last_op
        if terminal is None or any(op.successors for op in list(block.ops)[:-1]):
            raise ValueError("allocation barrier CFG requires complete terminal control edges")
        if _name(terminal) == "tt.return":
            if terminal.successors or terminal.results or terminal.regions:
                raise ValueError("allocation barrier CFG return has unexpected control fields")
            continue
        if _name(terminal) not in ("cf.br", "cf.cond_br", "cf.switch"):
            raise ValueError("allocation barrier CFG has an unsupported terminator")
        if not terminal.successors or any(target not in blocks for target in terminal.successors):
            raise ValueError("allocation barrier CFG edges must stay in the original program region")
        if _name(terminal) != "cf.br" and (
            not terminal.operands
            or not isinstance(terminal.operands[0].type, (builtin.IntegerType, builtin.IndexType))
            or _name(terminal) == "cf.cond_br"
            and terminal.operands[0].type != builtin.i1
        ):
            raise ValueError("allocation barrier CFG conditions require actual scalar integer control")
        try:
            terminal.verify()
            if _name(terminal) == "cf.br":
                edges = [(terminal.successor, terminal.arguments)]
            elif _name(terminal) == "cf.cond_br":
                edges = [(terminal.then_block, terminal.then_arguments), (terminal.else_block, terminal.else_arguments)]
            else:
                edges = [
                    (terminal.default_block, terminal.default_operands),
                    *zip(terminal.case_blocks, terminal.case_operand, strict=True),
                ]
            if any([value.type for value in values] != [arg.type for arg in target.args] for target, values in edges):
                raise ValueError("allocation barrier CFG successor arguments differ from their original block types")
        except VerifyException as error:
            raise ValueError("allocation barrier CFG has invalid typed successor arguments") from error
    return "whole-program-scalar-cfg-v1"


def checked_program_barriers(graph, reports, generic, protected):
    """Reconstruct the explicit optional phase extension from actual source."""
    try:
        _checked_program_barriers(graph, reports, generic, protected)
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("malformed allocation source barrier protection") from error


def _checked_program_barriers(graph, reports, generic, protected):
    records = graph["operations"]
    required = [row for row in records if row.get("opcode") == "ttg.barrier"]
    from merlin.compiler.frontend.linalg_mlir import make_context

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
    source_records = dict(zip(ops, records, strict=True))
    identities = {op: row["id"] for op, row in source_records.items()}
    effects = {"tt.load", "tt.store", "tt.call", "tt.assert", "cf.assert", "tt.print", "tt.atomic_rmw", "tt.atomic_cas"}
    effect_ids = {identities[op] for op in ops if _name(op) in effects}
    source = {identities[op]: op for op in ops if _name(op) == "ttg.barrier"}
    atomic_source = [op for op in ops if _name(op) in ("tt.atomic_rmw", "tt.atomic_cas")]
    atomic_declarations = {identities[op]: source_records[op].get("atomic") for op in atomic_source}
    atomic_effects = reports[0].get("contract", {}).get("atomic_effects") if atomic_source else None
    if atomic_source:
        from merlin.compiler.frontend.intake.kernel_atomic_evidence import checked_source_atomic

        for op in atomic_source:
            checked_source_atomic(op, source_records[op].get("atomic"), atomic_effects)
    from merlin.compiler.frontend.intake.kernel_volatile_evidence import PROFILE as volatile_profile
    from merlin.compiler.frontend.intake.kernel_volatile_evidence import checked_source_volatile

    volatile_source = {
        identities[op]: op
        for op in ops
        if _name(op) == "tt.load"
        and isinstance(op.properties.get("isVolatile"), builtin.IntegerAttr)
        and op.properties["isVolatile"].value.data
    }
    volatile_effects = {"profile": volatile_profile} if volatile_source else None
    if volatile_source and atomic_source and atomic_effects.get("profile") != "private-invocation-typed-atomics-v2":
        raise ValueError("combined program phases require the selected typed atomic component")
    for sid, op in volatile_source.items():
        checked_source_volatile(op, source_records[op].get("volatile_read"), sid)
    if len(reports) != len(source):
        raise ValueError("allocation barrier phase roster differs from its source")
    for op in original.walk():
        ctas = op.attributes.get("ttg.num-ctas", op.properties.get("ttg.num-ctas"))
        if ctas is not None and (not isinstance(ctas, builtin.IntegerAttr) or ctas.value.data != 1):
            raise ValueError("allocation source barriers need single-CTA participation")
        if (
            _name(op).startswith(("ttg.", "ttng.", "gpu.", "nvvm.", "rocdl."))
            and _name(op) != "ttg.barrier"
            or _name(op) in {"tt.elementwise_inline_asm", "tt.inline_asm"}
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
            sid = next(iter(selected))
            if _name(child) in ("tt.atomic_rmw", "tt.atomic_cas"):
                if sid not in atomic_declarations:
                    raise ValueError("allocation barrier contains an unaccounted atomic effect")
                checked_source_atomic(child, atomic_declarations[sid], atomic_effects)
            if _name(child) == "tt.load":
                if sid in volatile_source:
                    checked_source_volatile(child, source_records[volatile_source[sid]]["volatile_read"], sid)
                else:
                    flag = child.properties.get("isVolatile")
                    if isinstance(flag, builtin.IntegerAttr) and flag.value.data:
                        raise ValueError("allocation barrier protection invented a volatile read")
            retained_order.append((_name(child), sid))
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
        source_control = None
        while parent is not None and parent is not original:
            if _name(parent) not in {"tt.func", "scf.for", "scf.if", "scf.while"}:
                raise ValueError("allocation barrier participation needs whole-program structured control")
            if any(len(region.blocks) != 1 for region in parent.regions):
                if _name(parent) != "tt.func":
                    raise ValueError("allocation barrier participation needs single-block structured regions")
                source_control = _checked_cfg_participation(parent)
            parent = parent.parent_op()
        expected_contract = {
            "schema": (
                "t2m.synchronous-program-barrier.v4"
                if volatile_source and atomic_source
                else "t2m.synchronous-program-barrier.v3"
                if volatile_source
                else "t2m.synchronous-program-barrier.v2"
                if atomic_source
                else _PROFILE
            ),
            **({"source_control": source_control} if source_control is not None else {}),
            **({"atomic_effects": atomic_effects} if atomic_source else {}),
            **({"volatile_effects": volatile_effects} if volatile_source else {}),
            "source_id": source_id,
            "address_space_mask": space.value.data,
            "address_spaces": [name for bit, name in _SPACES if space.value.data & bit],
            "participants": "complete-pre-layout-program; single-CTA",
            "ordering": "complete-synchronous-tile-operations-at-source-control-position",
            "external_participants": "not-admitted",
            "physical_mapping": "undecided",
        }
        row = source_records[op]
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
