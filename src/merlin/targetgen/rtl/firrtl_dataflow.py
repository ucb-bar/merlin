"""Classify selected FIRRTL cell dataflow codes by product destinations.

This is the FIRRTL counterpart of timing_structure.dataflow_modes. The target
supplies the cell/select anchor; codes come from comparisons in the pinned RTL.
A product reaching only registers is OS; reaching only output ports is WS.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from . import firrtl_struct as F
from .firrtl_operand_slices import _source
from .timing_structure import _classify


def _ports(module):
    def leaves(typ, path, incoming):
        if typ[0] == "bundle":
            for name, flip, child in typ[1]:
                yield from leaves(child, (*path, name), incoming != flip)
        elif typ[0] in {"uint", "sint"}:
            yield F.Ref(path[0], path[1:]), incoming, typ[1]

    for name, (kind, text, _) in module.decls.items():
        if kind in {"input", "output"}:
            yield from leaves(F.parse_type(text), (name,), kind == "input")


def _evaluate(module, expr, select, code, active=()):
    if expr == select:
        return code
    if isinstance(expr, F.Lit):
        return expr.value
    if isinstance(expr, F.Ref) and expr.root in module.nodes and not expr.segments:
        if expr.root in active or len(active) >= 128:
            raise ValueError("dataflow guard is cyclic or oversized")
        return _evaluate(module, module.nodes[expr.root][0], select, code, (*active, expr.root))
    if not isinstance(expr, F.Prim):
        return None
    args = [_evaluate(module, arg, select, code, active) for arg in expr.args]
    if expr.op == "and" and 0 in args:
        return 0
    if None in args:
        return None
    if expr.op == "eq" and len(args) == 2:
        return int(args[0] == args[1])
    if expr.op == "neq" and len(args) == 2:
        return int(args[0] != args[1])
    if expr.op in {"and", "or"} and len(args) == 2:
        return args[0] & args[1] if expr.op == "and" else args[0] | args[1]
    if expr.op == "not" and len(args) == 1 and args[0] in {0, 1}:
        return 1 - args[0]
    return None


def _products(module, children):
    products = set()
    for instance, child in children.items():
        for port, incoming, _ in _ports(child):
            if incoming:
                continue
            leaves, prims, _ = F.Cone(child).walk(port)
            state = any(child.decls.get(F.parse_expr(leaf).root, (None,))[0] in {"reg", "regreset"} for leaf in leaves)
            if any(p.op == "mul" for p in prims) and not state:
                products.add(F.Ref(instance, (port.root, *port.segments)).text())
    # A cell may contain its multiplier directly instead of a child MAC.
    products.update(
        name for name, (expr, _, _) in module.nodes.items() if isinstance(expr, F.Prim) and expr.op == "mul"
    )
    if not products:
        raise ValueError("cell has no combinational product source")
    return products


def _select_arms(module, expr, select, code):
    """Keep only the mux arms this select code can reach, like pruned ``when`` guards."""
    if not isinstance(expr, F.Prim):
        return expr
    if expr.op == "mux" and len(expr.args) == 3:
        value = _evaluate(module, expr.args[0], select, code)
        if value is not None:
            return _select_arms(module, expr.args[1] if value else expr.args[2], select, code)
    return F.Prim(expr.op, tuple(_select_arms(module, arg, select, code) for arg in expr.args))


def _destinations(cell, select, code, products):
    def possible(connect):
        return all(
            (value := _evaluate(cell, guard.cond, select, code)) is None or bool(value) == guard.positive
            for guard in connect.guards
        )

    pruned = replace(
        cell,
        connects=[
            replace(row, rhs=_select_arms(cell, row.rhs, select, code)) for row in cell.connects if possible(row)
        ],
        nodes={name: (_select_arms(cell, expr, select, code), *rest) for name, (expr, *rest) in cell.nodes.items()},
    )
    sinks = {"registers": [], "combinational_outputs": []}
    outputs = {port for port, incoming, _ in _ports(cell) if not incoming}
    for row in pruned.connects:
        kind = cell.decls.get(row.lhs.root, (None,))[0]
        category = (
            "registers" if kind in {"reg", "regreset"} else "combinational_outputs" if row.lhs in outputs else None
        )
        if category is None:
            continue
        leaves, prims, lines = F.Cone(pruned).walk(row.rhs)
        if not ((leaves | {r.text() for r in F._refs_in(row.rhs)}) & products or any(p.op == "mul" for p in prims)):
            continue
        guards = sorted({line for guard in row.guards for line in F.Cone(cell).walk(guard.cond)[2]})
        sinks[category].append(
            {
                "firrtl_line": row.line,
                "source": F.locator_citation(row.locator),
                "cone_lines": sorted(set(lines)),
                "guard_lines": guards,
            }
        )
    return sinks


# FIRRTL spec field names of the data a ``mem`` write (``data``) or read-write (``wdata``) port stores.
_WRITE_DATA_FIELDS = frozenset({"data", "wdata"})


def _memory_ports(path, module):
    """``{port: memory}`` for every CHIRRTL ``mport`` the module declares, read off its statements."""
    ports = {}
    for _number, raw in F.read_module_bodies(path, [module]).get(module, []):
        head = F.split_locator(raw.strip())[0].split(None, 2)
        if len(head) == 3 and head[0] in {"read", "write", "infer", "rdwr"} and head[1] == "mport":
            port, separator, rhs = head[2].partition("=")
            if not separator or "[" not in rhs:
                raise ValueError("runtime select cone has an unexplained memory port")
            ports[port.strip()] = rhs.split("[", 1)[0].strip()
    return ports


def _project(expr, suffix):
    """The expressions an aggregate ``expr`` contributes to its field ``suffix``.

    A reference takes the suffix; a mux keeps its select and projects both arms; any other
    primitive is kept whole (an over-approximation, never a dropped dependency).
    """
    if not suffix:
        return [expr]
    if isinstance(expr, F.Ref):
        return [F.Ref(expr.root, (*expr.segments, *suffix), expr.dynamic)]
    if isinstance(expr, F.Prim) and expr.op == "mux" and len(expr.args) == 3:
        return [expr.args[0], *_project(expr.args[1], suffix), *_project(expr.args[2], suffix)]
    return [expr]


def _update(module, value, register, active=()):
    """The parts of a register's assigned ``value`` that can change it, its hold path removed.

    A mux arm that is the register's own value (directly or through nodes) is the hold path, so
    that mux is a write enable: its select and held arm drop out and only the written arm stays.
    """
    if isinstance(value, F.Ref) and value.root in module.nodes and not value.segments:
        if value.root in active or len(active) >= 128:
            raise ValueError("register update cone is cyclic or oversized")
        return _update(module, module.nodes[value.root][0], register, (*active, value.root))
    if isinstance(value, F.Prim) and value.op == "mux" and len(value.args) == 3:
        arms = [arm for arm in value.args[1:] if not _holds(module, arm, register)]
        if len(arms) < 2:
            return [part for arm in arms for part in _update(module, arm, register, active)]
    if _holds(module, value, register):
        return []
    return [value]


def _holds(module, value, register):
    """Whether ``value`` reads ``register`` itself (its hold path), seen through plain nodes."""
    for _ in range(128):
        if not (isinstance(value, F.Ref) and value.root in module.nodes and not value.segments):
            break
        value = module.nodes[value.root][0]
    return isinstance(value, F.Ref) and value.root == register.root and F._compatible(value, register)


def _drivers(rows, expr, *, guarded=True):
    """Right-hand sides that can be assigned to ``expr``, plus the conditions choosing them.

    An assignment aliases ``expr`` when one path prefixes the other, a dynamic index matching any
    static one; a field suffix is carried onto the assigned reference. A ``when`` guard or a dynamic
    destination index selects between the assigned value and the held one, like a mux select, so it
    belongs to the driving cone; memory write data passes ``guarded=False`` (data only).
    """
    for row in rows:
        if row.lhs.root != expr.root or not F._compatible(row.lhs, expr):
            continue
        yield from _project(row.rhs, expr.segments[len(row.lhs.segments) :])
        if guarded:
            yield from (guard.cond for guard in row.guards)
            yield from row.lhs.dynamic


# A clock or reset input times or initialises state; it carries no data a select could take.
_CLOCK_AND_RESET = frozenset({("other", "Clock"), ("other", "Reset"), ("other", "AsyncReset")})


def _incoming(module, ref):
    """Whether ``ref`` names one ground data leaf flowing into ``module`` (not a clock or reset)."""
    declaration = module.decls.get(ref.root)
    if not declaration or declaration[0] not in {"input", "output"}:
        return False
    typ, incoming = F.parse_type(declaration[1]), declaration[0] == "input"
    for segment in ref.segments:
        if typ[0] == "vec" and isinstance(segment, int):
            typ = typ[1]
        elif typ[0] == "bundle" and isinstance(segment, str):
            field = next((row for row in typ[1] if row[0] == segment), None)
            if field is None:
                return False
            incoming, typ = incoming != field[1], field[2]
        else:
            return False
    return incoming and typ[0] not in {"bundle", "vec"} and typ not in _CLOCK_AND_RESET


def _runtime_select(path, cell_name, select, root=None):
    """Whether every occurrence's select is runtime-driven, and how many occurrences there are.

    The select's driving cone is traced backward through the pinned hierarchy. It is runtime-driven
    only when the cone reaches a data input of the root (not a clock or reset). Combinational values
    are followed through their operands, mux selects, ``when`` guards and dynamic indices. A
    register is followed through the values written into it, never its reset value nor its hold
    path (a self-assignment, or the held arm and enable of a mux); a memory read is followed through
    the data of its memory's writes. So a register or memory written only from constants, or holding
    its reset value, is not runtime-driven. Cyclic, oversized or unavailable cones raise.
    """
    hierarchy, current, circuit = {}, None, None
    with path.open() as source:
        for line in source:
            words = F.split_locator(line)[0].strip().split()
            if words[:1] == ["circuit"]:
                circuit = words[1]
            if words[:1] == ["module"]:
                current = words[1]
                hierarchy[current] = {}
            elif words[:1] == ["extmodule"]:
                current = None
            elif current and words[:1] == ["inst"] and len(words) == 4 and words[2] == "of":
                hierarchy[current][words[1]] = words[3]
    root = root or circuit
    paths = []

    def visit(name, instances, kinds):
        if name in kinds or len(kinds) > 128:
            raise ValueError("dataflow hierarchy is cyclic or oversized")
        if name == cell_name:
            paths.append((instances, (*kinds, name)))
        for instance, child in hierarchy.get(name, {}).items():
            visit(child, (*instances, instance), (*kinds, name))

    visit(root, (), ())
    if not paths:
        return False, 0
    modules = F.load_modules(path, {kind for _, kinds in paths for kind in kinds})
    memory_ports, connects = {}, {}

    def assigned(module, name):
        """Connects assigning the declaration ``name`` (indexed once per module)."""
        if module.name not in connects:
            connects[module.name] = {}
            for row in module.connects:
                connects[module.name].setdefault(row.lhs.root, []).append(row)
        return connects[module.name].get(name, ())

    def module_at(instances):
        name = root
        for instance in instances:
            name = hierarchy[name][instance]
        if name not in modules:
            modules.update(F.load_modules(path, [name]))
        if name not in modules:
            raise ValueError("runtime select cone has an unavailable module")
        return modules[name]

    def driven(instances):
        work, seen = [(instances, select)], set()
        while work:
            instance_path, expr = work.pop()
            if (instance_path, expr) in seen:
                continue
            seen.add((instance_path, expr))
            if len(seen) > 100000:
                raise ValueError("runtime select cone is oversized")
            module = module_at(instance_path)
            if isinstance(expr, F.Prim):
                work.extend((instance_path, arg) for arg in expr.args)
            elif isinstance(expr, F.Ref):
                work.extend((instance_path, index) for index in expr.dynamic)
                if expr.root in module.nodes:
                    work.extend((instance_path, part) for part in _project(module.nodes[expr.root][0], expr.segments))
                    continue
                if _incoming(module, expr):
                    if not instance_path:
                        return True
                    work.append((instance_path[:-1], F.Ref(instance_path[-1], (expr.root, *expr.segments))))
                    continue
                kind = module.decls.get(expr.root, (None,))[0]
                if kind == "inst" and expr.segments:
                    child_path = (*instance_path, expr.root)
                    child = module_at(child_path)
                    port = F.Ref(expr.segments[0], expr.segments[1:])
                    if not _incoming(child, port):
                        work.append((child_path, port))
                elif kind == "mport":
                    if module.name not in memory_ports:
                        memory_ports[module.name] = _memory_ports(path, module.name)
                    ports = memory_ports[module.name]
                    if expr.root not in ports:
                        raise ValueError("runtime select cone has an unexplained memory port")
                    for writer in (port for port, memory in ports.items() if memory == ports[expr.root]):
                        written = F.Ref(writer, expr.segments)
                        work.extend(
                            (instance_path, rhs)
                            for rhs in _drivers(assigned(module, written.root), written, guarded=False)
                        )
                    continue
                elif kind in {"reg", "regreset"}:
                    for rhs in _drivers(assigned(module, expr.root), expr, guarded=False):
                        work.extend((instance_path, part) for part in _update(module, rhs, expr))
                    continue
                elif kind == "mem" and len(expr.segments) >= 2:
                    for row in assigned(module, expr.root):
                        lhs = row.lhs
                        if len(lhs.segments) >= 2 and lhs.segments[1] in _WRITE_DATA_FIELDS:
                            written = F.Ref(lhs.root, (*lhs.segments[:2], *expr.segments[2:]))
                            work.extend(
                                (instance_path, rhs)
                                for rhs in _drivers(assigned(module, written.root), written, guarded=False)
                            )
                    continue
                work.extend((instance_path, driver) for driver in _drivers(assigned(module, expr.root), expr))
        return False

    return all(driven(instances) for instances, _ in paths), len(paths)


def derive(facts, declaration):
    """Return modes and line citations from the facts' exact byte-pinned source."""
    inputs = facts.get("inputs", {})
    path = _source(Path(inputs.get("fir_path", "")).resolve(), inputs.get("fir_sha256"))
    name, anchor = declaration.get("cell_module"), declaration.get("select_port")
    if not name or not anchor:
        raise ValueError("dataflow cell/select anchors are absent")
    modules = F.load_modules(path, [name])
    if name not in modules:
        raise ValueError("selected dataflow cell is absent")
    cell = modules[name]
    selected = [
        (port, bits)
        for port, incoming, bits in _ports(cell)
        if incoming and "_".join((port.root, *port.segments)) == anchor
    ]
    if len(selected) != 1 or selected[0][1] != 1:
        raise ValueError("dataflow anchor is not one scalar Boolean input")
    select = selected[0][0]
    children = {instance: kind for instance, (decl, kind, _) in cell.decls.items() if decl == "inst"}
    loaded = F.load_modules(path, children.values())
    if set(children.values()) - set(loaded):
        raise ValueError("dataflow cell has an unavailable child")
    products = _products(cell, {instance: loaded[kind] for instance, kind in children.items()})
    observations = {str(code): _destinations(cell, select, code, products) for code in range(1 << selected[0][1])}
    modes = {code: _classify(sinks) for code, sinks in observations.items()}
    runtime, occurrences = _runtime_select(path, name, select, declaration.get("unit_root"))
    _source(path, inputs["fir_sha256"])
    if set(modes.values()) != {"output_stationary", "weight_stationary"}:
        raise ValueError("cell does not distinguish both stationary modes")
    return {
        "status": "derived" if runtime else "unknown",
        "modes": modes,
        "support": "both" if runtime else "unknown",
        "select_runtime_driven": runtime,
        "cell_occurrences": occurrences,
        "source": str(path),
        "source_sha256": inputs["fir_sha256"],
        "evidence": observations,
    }


def _unit_root(facts, contract):
    """The unit whose inputs a runtime select must reach: declared, else the facts' or a fresh census.

    The census scopes the unit by the contract's declared generator tree over the facts' byte-pinned
    elaboration. Without one, the trace is rooted at the circuit (whose inputs rarely reach a select,
    so the select stays underived rather than being assumed runtime).
    """
    declared = contract.get("rtl_extraction", {}).get("timing_structure", {}).get("unit_root")
    if declared:
        return declared
    census = facts.get("facts", facts).get("census", {}).get("unit_root")
    if isinstance(census, str):
        return census
    inputs, generator = facts.get("inputs", {}), contract.get("rtl_source", {}).get("generator")
    if not generator or not inputs.get("hierarchy_path"):
        return None
    from .introspect import census_facts

    fir = _source(Path(inputs.get("fir_path", "")).resolve(), inputs.get("fir_sha256"))
    hierarchy = _source(Path(inputs["hierarchy_path"]).resolve(), inputs.get("hierarchy_sha256"))
    return census_facts(fir, hierarchy, generator=generator)["census"]["unit_root"]


def select_code(facts, contract, mode):
    body = facts.get("facts", facts)
    dataflow = body.get("timing_structure", {}).get("dataflow")
    if dataflow is None or dataflow.get("status") != "derived":
        try:
            timing = contract.get("rtl_extraction", {}).get("timing_structure", {})
            declaration = {**timing.get("dataflow", {}), "unit_root": _unit_root(facts, contract)}
            dataflow = derive(facts, declaration)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"UNKNOWN: {mode.replace('_', '-')} select code is underived: {error}") from error
    codes = [int(code) for code, label in dataflow.get("modes", {}).items() if label == mode]
    if dataflow.get("status") != "derived" or dataflow.get("support") not in {"both", mode} or len(codes) != 1:
        raise ValueError(f"UNKNOWN: {mode.replace('_', '-')} select code is underived")
    return codes[0]
