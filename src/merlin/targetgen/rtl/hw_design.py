"""Hierarchical structural walks over one elaborated CIRCT HW design.

The readers answer *structure* questions about state between ports, never semantics:

* :meth:`Design.read_storage` -- the storage words that can present their value at a point
  (registers, memory rows, a bypass from an input). Its size is the number of locations a reader
  of that point can select: a FIFO's entries at its dequeue port, a reservation table's slots at an
  issue port.
* :meth:`Design.buffers` -- a module's outputs that read local storage written from one of its
  inputs, with that storage's location count. This is a capacity, never a latency.
* :meth:`Design.forward_buffers` -- the buffers a payload crosses after leaving a port, and whether
  it arrives intact (a whole-value copy) or as derived bits.
* :meth:`Design.port_stages` -- register stages over every data path between two ports, with the
  conditions that keep a count from being a fixed latency (an enable/hold, storage retention,
  feedback, an opaque element).
* :meth:`Design.value_set` -- the finite set of values a bit field can carry, or None.
* :meth:`Design.index_slots` -- state slots selected by a value used as an index or decoded.

Data paths follow multiplexer data operands, slices, packing, bitwise selection and arithmetic;
multiplexer selects and array indices are control. Every reader is target-agnostic: modules,
ports and instances arrive as parameters from the target's own declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects.builtin import IntegerType, StringAttr
from xdsl.ir import BlockArgument, OpResult

from .hw_instance_inputs import _instance, _signature
from .hw_memory_ports import _memory_type
from .hw_observations import _attribute, _integer, _module_name, _name

REGISTERS = frozenset({"seq.firreg", "seq.compreg"})
READ_PORTS = frozenset({"seq.firmem.read_port", "seq.firmem.read_write_port"})
WRITE_PORTS = frozenset({"seq.firmem.write_port", "seq.firmem.read_write_port"})
#: Operations whose result carries (a selection, slice or packing of) an operand's value.
CARRIERS = frozenset(
    {"comb.mux", "comb.extract", "comb.concat", "hw.array_get", "hw.array_create", "hw.array_concat", "hw.wire"}
)
#: Carriers that keep a whole operand value intact in their result.
INTACT = frozenset({"comb.mux", "hw.array_get", "hw.array_create", "hw.array_concat", "hw.wire"})
BITWISE = frozenset({"comb.and", "comb.or", "comb.xor"})
CONSTANTS = frozenset({"hw.constant", "hw.aggregate_constant"})


def width(value) -> int | None:
    typ = value.type
    return typ.width.data if isinstance(typ, IntegerType) else None


@dataclass(frozen=True)
class InstanceView:
    name: str
    callee: str
    op: Any
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def _extern_ports(op) -> list[Any]:
    """Ports of an external (possibly parameterized) module; its body is opaque to every reader."""
    from xdsl.dialects.builtin import UnregisteredAttr

    from .hw_instance_inputs import OriginalPort
    from .ports import _hw_port_entries

    typ = _attribute(op, "module_type")
    entries = _hw_port_entries("(" + typ.value.data + ")") if isinstance(typ, UnregisteredAttr) else None
    if entries is None:
        raise ValueError(f"external module {_module_name(op)} has unreadable ports")
    ports = []
    for entry in entries:
        head, _, type_text = entry.partition(":")
        direction, name = head.split()
        ports.append(OriginalPort(direction, name, type_text.strip()))
    return ports


def _instance_view(op, roster: dict[str, Any]) -> InstanceView:
    from xdsl.dialects.builtin import SymbolRefAttr

    module = _attribute(op, "moduleName")
    callee = module.root_reference.data if isinstance(module, SymbolRefAttr) else None
    if callee in roster and _name(roster[callee]) == "hw.module.extern":
        names, results = _attribute(op, "argNames"), _attribute(op, "resultNames")
        return InstanceView(
            _attribute(op, "instanceName").data,
            callee,
            op,
            tuple(n.data for n in names.data),
            tuple(n.data for n in results.data),
        )
    inst, callee, ports = _instance(op, roster)
    return InstanceView(
        inst,
        callee,
        op,
        tuple(p.name for p in ports if p.direction == "input"),
        tuple(p.name for p in ports if p.direction == "output"),
    )


class ModuleView:
    """One module's ports, instances and use-def access."""

    def __init__(self, op, roster: dict[str, Any]):
        self.op = op
        self.name = _module_name(op)
        self.extern = _name(op) == "hw.module.extern"
        self.ports = _extern_ports(op) if self.extern else _signature(op)
        self.inputs = [port.name for port in self.ports if port.direction == "input"]
        self.outputs = [port.name for port in self.ports if port.direction == "output"]
        self.input_value: dict[str, Any] = {}
        self.output_value: dict[str, Any] = {}
        self.input_name: dict[Any, str] = {}
        self.instances: dict[str, InstanceView] = {}
        self.instance_by_op: dict[Any, InstanceView] = {}
        self.ordinal: dict[Any, int] = {}
        if self.extern:
            return
        block = op.regions[0].block
        self.input_value = dict(zip(self.inputs, block.args, strict=True))
        self.input_name = {arg: name for name, arg in self.input_value.items()}
        self.output_value = dict(zip(self.outputs, block.last_op.operands, strict=True))
        for ordinal, child in enumerate(block.ops):
            self.ordinal[child] = ordinal
            if _name(child) == "hw.instance":
                view = _instance_view(child, roster)
                if view.name in self.instances:
                    raise ValueError(f"module {self.name} declares instance {view.name!r} twice")
                self.instances[view.name] = view
                self.instance_by_op[child] = view

    def output_names(self, value) -> list[str]:
        return [name for name, out in self.output_value.items() if out is value]

    def state_name(self, op) -> str:
        name = _attribute(op, "name")
        label = name.data if isinstance(name, StringAttr) and name.data else _name(op)
        return f"{label}@{self.ordinal.get(op, -1)}"


@dataclass
class Storage:
    """Storage words that can present their value at one point, plus what else can."""

    registers: set[str] = field(default_factory=set)
    memories: dict[str, int] = field(default_factory=dict)
    inputs: set[str] = field(default_factory=set)
    opaque: set[str] = field(default_factory=set)
    computed: bool = False

    @property
    def locations(self) -> int:
        return len(self.registers) + sum(self.memories.values())


@dataclass(frozen=True)
class Stages:
    """Register stages from one input to one output over every data path between them."""

    minimum: int
    maximum: int | None  # None: a path re-enters state (feedback), so no finite bound
    held: bool = False  # a register on a path can hold its value (enable), or storage retains it
    opaque: bool = False  # a path crosses an element this reader cannot see into

    def then(self, delay: int, *, held: bool = False, opaque: bool = False) -> Stages:
        top = None if self.maximum is None else self.maximum + delay
        return Stages(self.minimum + delay, top, self.held or held, self.opaque or opaque)

    def join(self, other: Stages) -> Stages:
        top = None if self.maximum is None or other.maximum is None else max(self.maximum, other.maximum)
        return Stages(min(self.minimum, other.minimum), top, self.held or other.held, self.opaque or other.opaque)

    def chain(self, after: Stages) -> Stages:
        top = None if self.maximum is None or after.maximum is None else self.maximum + after.maximum
        return Stages(self.minimum + after.minimum, top, self.held or after.held, self.opaque or after.opaque)

    @property
    def status(self) -> str:
        if self.opaque:
            return "unknown"
        if self.maximum is None or self.maximum != self.minimum:
            return "variable"
        return "fixed_when_enabled" if self.held else "fixed"

    def record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "min_stages": self.minimum,
            "max_stages": self.maximum,
            "enable_or_retention_on_path": self.held,
            "opaque_on_path": self.opaque,
        }


class Design:
    """A parsed generic HW module tree."""

    def __init__(self, parsed, *, pins: dict[str, dict[str, int]] | None = None):
        roster: dict[str, Any] = {}
        for op in parsed.body.block.ops:
            if _name(op) in {"hw.module", "hw.module.extern"}:
                symbol = _module_name(op)
                if symbol in roster:
                    raise ValueError(f"HW design defines module {symbol!r} twice")
                roster[symbol] = op
        self.modules = {symbol: ModuleView(op, roster) for symbol, op in roster.items()}
        self.pins = pins or {}
        self._read: dict[tuple[str, Any], Storage] = {}
        self._buffers: dict[str, dict[str, dict[str, Any]]] = {}
        self._stages: dict[tuple[str, str], dict[str, Stages]] = {}
        self._paths: dict[str, dict[str, str]] = {}

    def module(self, name: str) -> ModuleView:
        if name not in self.modules:
            raise ValueError(f"HW design has no module {name!r}")
        return self.modules[name]

    def pinned(self, pins: dict[str, dict[str, int]]) -> Design:
        """The same parsed design under other select pins (fresh pin-dependent caches)."""
        other = object.__new__(Design)
        other.modules, other.pins = self.modules, pins
        other._read, other._buffers, other._paths = self._read, self._buffers, self._paths
        other._stages = {}
        return other

    # ------------------------------------------------------------------ hierarchy
    def instance_paths(self, root: str) -> dict[str, str]:
        """``{instance path: module}`` for every occurrence under ``root`` (root path = root)."""
        if root not in self._paths:
            out, stack = {root: root}, [(root, root)]
            while stack:
                path, name = stack.pop()
                for inst in self.module(name).instances.values():
                    child = f"{path}/{inst.name}"
                    out[child] = inst.callee
                    if not self.module(inst.callee).extern:
                        stack.append((child, inst.callee))
            self._paths[root] = out
        return self._paths[root]

    def occurrences(self, root: str, module: str) -> list[str]:
        return sorted(path for path, name in self.instance_paths(root).items() if name == module)

    def _instance_at(self, root: str, path: str) -> tuple[ModuleView, InstanceView]:
        parent_path, _, name = path.rpartition("/")
        parent = self.module(self.instance_paths(root)[parent_path])
        return parent, parent.instances[name]

    # ------------------------------------------------------------------ storage at a point
    def read_storage(self, module: str, value) -> Storage:
        """Local storage that can present its value at ``value`` (instances stay opaque)."""
        view = self.module(module)
        key = (module, value)
        if key in self._read:
            return self._read[key]
        result, stack, seen = Storage(), [value], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if isinstance(current, BlockArgument):
                if current in view.input_name:
                    result.inputs.add(view.input_name[current])
                continue
            if not isinstance(current, OpResult):
                result.computed = True
                continue
            op = current.owner
            kind = _name(op)
            if kind in REGISTERS:
                result.registers.add(view.state_name(op))
            elif kind in READ_PORTS:
                memory = op.operands[0].owner
                result.memories[view.state_name(memory)] = _memory_type(memory.results[0].type)[0]
            elif kind == "hw.instance":
                inst = view.instance_by_op[op]
                result.opaque.add(f"{inst.name}.{inst.outputs[current.index]}")
            elif kind in CONSTANTS:
                continue
            elif kind == "comb.mux":
                stack.extend(op.operands[1:])
            elif kind == "hw.array_get":
                stack.append(op.operands[0])
            elif kind in CARRIERS:
                stack.extend(op.operands)
            elif kind in BITWISE:
                stack.extend(
                    operand
                    for operand in op.operands
                    if width(operand) == width(current)
                    and not (isinstance(operand, OpResult) and _name(operand.owner) in CONSTANTS | {"comb.replicate"})
                )
            else:
                result.computed = True
        self._read[key] = result
        return result

    def _write_data(self, memory) -> list[Any]:
        """The data operand of every write port of one memory."""
        data = []
        for use in memory.results[0].uses:
            port = use.operation
            if _name(port) in WRITE_PORTS:
                segments = list(_attribute(port, "operandSegmentSizes").get_values())
                if segments[4]:  # memory, address, clock, enable precede the data operand
                    data.append(port.operands[sum(segments[:4])])
        return data

    def _writers(self, module: str, op) -> set[str]:
        """Module inputs whose value reaches the data written into a register or memory."""
        if _name(op) in REGISTERS:
            return set(self.read_storage(module, op.operands[0]).inputs)
        found: set[str] = set()
        for data in self._write_data(op):
            found |= self.read_storage(module, data).inputs
        return found

    def _read_slices(self, view: ModuleView, value) -> set[tuple[str, int, int]]:
        """``(storage, low bit, width)`` storage-word slices presented whole at ``value``."""
        size = width(value) or 0
        found, stack, seen = set(), [(value, 0)], set()
        while stack:
            current, low = stack.pop()
            if (current, low) in seen or not isinstance(current, OpResult):
                continue
            seen.add((current, low))
            op, kind = current.owner, _name(current.owner)
            if kind in REGISTERS:
                found.add((view.state_name(op), low, size))
            elif kind in READ_PORTS:
                found.add((view.state_name(op.operands[0].owner), low, size))
            elif kind == "comb.extract":
                stack.append((op.operands[0], low + _integer(op, "lowBit")))
            elif kind == "comb.mux":
                stack.extend((operand, low) for operand in op.operands[1:])
            elif kind == "hw.array_get":
                elements = op.operands[0].owner if isinstance(op.operands[0], OpResult) else None
                if elements is not None and _name(elements) == "hw.array_create":
                    stack.extend((operand, low) for operand in elements.operands)
            elif kind == "hw.wire":
                stack.append((op.operands[0], low))
        return found

    def _write_slices(self, view: ModuleView, name: str, data) -> dict[str, set[tuple[str, int, int]]]:
        """``{input port: {(storage, low bit, width)}}`` for inputs written whole into a word slice."""
        found: dict[str, set[tuple[str, int, int]]] = {}
        stack, seen = [(data, 0)], set()
        while stack:
            current, low = stack.pop()
            if (current, low) in seen:
                continue
            seen.add((current, low))
            if isinstance(current, BlockArgument):
                if current in view.input_name:
                    found.setdefault(view.input_name[current], set()).add((name, low, width(current) or 0))
                continue
            if not isinstance(current, OpResult):
                continue
            op, kind = current.owner, _name(current.owner)
            if kind == "comb.concat":
                offset = low
                for operand in reversed(op.operands):  # the last operand holds the low bits
                    stack.append((operand, offset))
                    offset += width(operand) or 0
            elif kind == "comb.mux":
                stack.extend((operand, low) for operand in op.operands[1:])
            elif kind == "hw.wire":
                stack.append((op.operands[0], low))
        return found

    def buffers(self, module: str) -> dict[str, dict[str, Any]]:
        """``{output port: buffer record}`` for outputs reading local storage written from inputs."""
        if module in self._buffers:
            return self._buffers[module]
        view = self.module(module)
        out: dict[str, dict[str, Any]] = {}
        if not view.extern:
            states = {
                view.state_name(op): op
                for op in view.op.regions[0].block.ops
                if _name(op) in REGISTERS or _name(op) == "seq.firmem"
            }
            for port, value in view.output_value.items():
                if (width(value) or 0) <= 1:
                    continue
                storage = self.read_storage(module, value)
                if not storage.registers and not storage.memories:
                    continue
                written = [self._writers(module, states[name]) for name in (*storage.registers, *storage.memories)]
                common = set.intersection(*written)
                if not common:
                    continue
                stored: dict[str, set] = {}
                for name in (*storage.registers, *storage.memories):
                    op = states[name]
                    sources = [op.operands[0]] if _name(op) in REGISTERS else self._write_data(op)
                    for data in sources:
                        for entered, slices in self._write_slices(view, name, data).items():
                            stored.setdefault(entered, set()).update(slices)
                presented = self._read_slices(view, value)
                out[port] = {
                    "locations": storage.locations,
                    "kind": "memory" if storage.memories else "registers",
                    "written_from": sorted(common),
                    # Inputs whose stored slice this output presents whole: a copy, not derived bits.
                    "copies_of": sorted(entered for entered, slices in stored.items() if slices & presented),
                    "bypass_from": sorted(storage.inputs),
                    "storage_count": len(storage.registers) + len(storage.memories),
                    "storage_sample": sorted((*storage.registers, *storage.memories))[:4],
                }
        self._buffers[module] = out
        return out

    # ------------------------------------------------------------------ forward payload walk
    def forward_buffers(self, root: str, path: str, ports: list[str], *, max_steps: int = 400_000) -> dict:
        """Buffers crossed by the payload leaving output ``ports`` of the instance at ``path``.

        Walks forward through carrier and bitwise data uses, into non-buffer instances and back out
        through their outputs. Entering a buffer records it with the locations of the buffered
        outputs its entry feeds and whether each origin port's value arrived intact (a whole-value
        copy rather than derived bits), then continues from those outputs. Other state ends a path.
        """
        occurrences = self.instance_paths(root)
        parent, inst = self._instance_at(root, path)
        parent_path = path.rpartition("/")[0]
        work = [(parent_path, inst.op.results[inst.outputs.index(p)], path, p, True) for p in ports]
        hops: dict[str, dict[str, Any]] = {}
        terminal: dict[str, int] = {}
        seen: set[tuple[str, int, str, bool]] = set()
        steps = 0
        while work:
            where, value, came_from, origin, intact = work.pop()
            key = (where, id(value), origin, intact)
            if key in seen:
                continue
            seen.add(key)
            steps += 1
            if steps > max_steps:
                raise ValueError("forward payload walk exceeds its explicit step budget")
            view = self.module(occurrences[where])
            for use in list(value.uses):
                op, index, kind = use.operation, use.index, _name(use.operation)
                if kind == "hw.instance":
                    callee = view.instance_by_op[op]
                    child, entered = f"{where}/{callee.name}", callee.inputs[index]
                    exits = [out for out, row in self.buffers(callee.callee).items() if entered in row["written_from"]]
                    if exits:
                        hop = hops.setdefault(
                            child,
                            {"instance": child, "module": callee.callee, "locations": set(), "reached_from": set()},
                        )
                        hop["locations"] |= {self.buffers(callee.callee)[out]["locations"] for out in exits}
                        hop["reached_from"].add(came_from)
                        hop.setdefault("arrivals", {}).setdefault(origin, set()).add(intact)
                        for out in exits:
                            copied = intact and entered in self.buffers(callee.callee)[out]["copies_of"]
                            work.append((where, op.results[callee.outputs.index(out)], child, origin, copied))
                    elif not self.module(callee.callee).extern:
                        target = self.module(callee.callee).input_value[entered]
                        work.append((child, target, came_from, origin, intact))
                elif kind == "hw.output":
                    if where == root:
                        continue
                    up, _, name = where.rpartition("/")
                    holder = self.module(occurrences[up]).instances[name]
                    for port in view.output_names(value):
                        work.append((up, holder.op.results[holder.outputs.index(port)], came_from, origin, intact))
                elif kind in REGISTERS or kind in WRITE_PORTS:
                    terminal[where] = terminal.get(where, 0) + 1
                elif kind == "comb.mux" and index == 0:
                    continue
                elif kind == "hw.array_get" and index != 0:
                    continue
                elif kind in CARRIERS or (kind in BITWISE and width(op.results[0]) == width(value)):
                    work.append((where, op.results[0], came_from, origin, intact and kind in INTACT | BITWISE))
        for hop in hops.values():
            hop["locations"] = sorted(hop["locations"])
            hop["reached_from"] = sorted(hop["reached_from"])
            hop["intact_from"] = sorted(port for port, flags in hop.pop("arrivals").items() if True in flags)
        return {"buffers": hops, "terminal_state": dict(sorted(terminal.items()))}

    # ------------------------------------------------------------------ register stages
    def _known(self, module: str, value, depth: int = 0) -> int | None:
        """Three-valued evaluation of a 1-bit select under the explicit pins."""
        view = self.module(module)
        if depth > 32:
            return None
        if isinstance(value, BlockArgument):
            return self.pins.get(module, {}).get(view.input_name.get(value))
        if not isinstance(value, OpResult):
            return None
        op, kind = value.owner, _name(value.owner)
        if kind == "hw.constant" and width(value) == 1:
            return _integer(op, "value") & 1
        if kind == "comb.mux":
            sel = self._known(module, op.operands[0], depth + 1)
            if sel is not None:
                return self._known(module, op.operands[2 - sel], depth + 1)
            a, b = (self._known(module, x, depth + 1) for x in op.operands[1:])
            return a if a is not None and a == b else None
        if kind in BITWISE and width(value) == 1:
            values = [self._known(module, x, depth + 1) for x in op.operands]
            if kind == "comb.and":
                return 0 if 0 in values else (1 if all(v == 1 for v in values) else None)
            if kind == "comb.or":
                return 1 if 1 in values else (0 if all(v == 0 for v in values) else None)
            return None if None in values else sum(values) & 1
        return None

    def port_stages(self, module: str, output: str) -> dict[str, Stages]:
        """``{input port: Stages}`` for every input reaching ``output`` over data paths.

        Two passes. Combinational cones are acyclic, so each value's sources (module inputs and
        registers, with the stage summaries of crossed instances) are computed once. Registers then
        form a graph: a register reached from its own next state only through carriers holds (an
        enable); any other cycle is feedback, and every path through it has no finite upper bound.
        """
        key = (module, output)
        if key not in self._stages:
            self._stages[key] = {}  # re-entry through an instance cycle reads as no path
            self._stages[key] = self._output_stages(self.module(module), output)
        return self._stages[key]

    def _cone(self, view: ModuleView, root, memo: dict) -> dict[Any, Stages]:
        """``{source: Stages}`` with sources ``("port", name)`` / ``("reg", op)``; no register crossed."""
        stack = [(root, False)]
        while stack:
            value, expanded = stack.pop()
            if value in memo and not expanded:
                continue
            if isinstance(value, BlockArgument):
                name = view.input_name.get(value)
                memo[value] = {("port", name): Stages(0, 0)} if name else {}
                continue
            if not isinstance(value, OpResult):
                memo[value] = {}
                continue
            if _name(value.owner) in REGISTERS:
                memo[value] = {("reg", value.owner): Stages(0, 0)}
                continue
            edges = self._cone_edges(view, value)
            if not expanded:
                memo[value] = None  # in progress: a combinational loop reads as no path
                stack.append((value, True))
                stack.extend((operand, False) for operand, _ in edges if operand not in memo)
                continue
            out: dict[Any, Stages] = {}
            for operand, transform in edges:
                for source, stages in (memo.get(operand) or {}).items():
                    shifted = _apply(transform, stages)
                    out[source] = shifted if source not in out else out[source].join(shifted)
            memo[value] = out
        return memo[root] or {}

    def _cone_edges(self, view: ModuleView, value) -> list[tuple[Any, tuple]]:
        op, kind = value.owner, _name(value.owner)
        if kind in READ_PORTS:
            memory = op.operands[0].owner
            read, write = _integer(memory, "readLatency") or 0, _integer(memory, "writeLatency") or 0
            edges = [(op.operands[1], ("delay", read, False))]
            # Stored data is retained until overwritten: a write-to-read path, never a fixed latency.
            return edges + [(data, ("delay", read + write, True)) for data in self._write_data(memory)]
        if kind == "hw.instance":
            inst = view.instance_by_op[op]
            if self.module(inst.callee).extern:
                return [(operand, ("opaque",)) for operand in op.operands]
            inner = self.port_stages(inst.callee, inst.outputs[value.index])
            return [(op.operands[inst.inputs.index(port)], ("chain", stages)) for port, stages in inner.items()]
        if kind in CONSTANTS:
            return []
        if kind == "comb.mux":
            sel = self._known(view.name, op.operands[0])
            branches = op.operands[1:] if sel is None else [op.operands[2 - sel]]
            return [(operand, ("delay", 0, False)) for operand in branches]
        if kind == "hw.array_get":
            return [(op.operands[0], ("delay", 0, False))]
        return [(operand, ("delay", 0, False)) for operand in op.operands if str(operand.type) != "!seq.clock"]

    def _self_loop(self, view: ModuleView, register) -> tuple[bool, bool]:
        """``(holds, feeds_back)``: the register's own value reaches its next state through carriers
        only (a hold) / through any computation (feedback)."""
        holds = feeds = False
        stack, seen = [(register.operands[0], True)], set()
        while stack:
            value, carried = stack.pop()
            if (value, carried) in seen or not isinstance(value, OpResult):
                continue
            seen.add((value, carried))
            op, kind = value.owner, _name(value.owner)
            if op is register:
                holds, feeds = holds or carried, feeds or not carried
                continue
            if kind in REGISTERS or kind == "hw.instance" or kind in READ_PORTS or kind in CONSTANTS:
                continue
            if kind == "comb.mux":
                sel = self._known(view.name, op.operands[0])
                branches = op.operands[1:] if sel is None else [op.operands[2 - sel]]
                stack.extend((operand, carried) for operand in branches)
            else:
                carry = carried and kind in INTACT
                stack.extend((operand, carry) for operand in op.operands if str(operand.type) != "!seq.clock")
        return holds, feeds

    def _output_stages(self, view: ModuleView, output: str) -> dict[str, Stages]:
        memo: dict[Any, Any] = {}
        start = self._cone(view, view.output_value[output], memo)
        preds: dict[Any, dict[Any, Stages]] = {}
        loops: dict[Any, tuple[bool, bool]] = {}
        frontier = [source[1] for source in start if source[0] == "reg"]
        while frontier:
            register = frontier.pop()
            if register in preds:
                continue
            cone = dict(self._cone(view, register.operands[0], memo))
            cone.pop(("reg", register), None)
            preds[register], loops[register] = cone, self._self_loop(view, register)
            frontier.extend(source[1] for source in cone if source[0] == "reg" and source[1] not in preds)
        cyclic = _cyclic_registers(preds, loops)
        consumers: dict[Any, list[tuple[Any, Stages]]] = {register: [] for register in preds}
        for register, cone in preds.items():
            for source, stages in cone.items():
                if source[0] == "reg":
                    consumers[source[1]].append((register, stages))
        # down[r]: stages from register r's output to the module output.
        down: dict[Any, Stages] = {}
        for _ in range(len(preds) + 2):  # Bellman-Ford relaxation; every stage weight is positive
            changed = False
            for register in preds:
                best = start.get(("reg", register))
                for consumer, stages in consumers[register]:
                    if consumer in down:
                        through = stages.chain(_register_stage(consumer, loops)).chain(down[consumer])
                        best = through if best is None else best.join(through)
                if best is None:
                    continue
                if register in cyclic:
                    best = Stages(best.minimum, None, best.held, best.opaque)
                if down.get(register) != best:
                    down[register], changed = best, True
            if not changed:
                break
        result: dict[str, Stages] = {}

        def add(port: str, stages: Stages) -> None:
            result[port] = stages if port not in result else result[port].join(stages)

        for source, stages in start.items():
            if source[0] == "port":
                add(source[1], stages)
        for register, cone in preds.items():
            if register in down:
                tail = _register_stage(register, loops).chain(down[register])
                for source, stages in cone.items():
                    if source[0] == "port":
                        add(source[1], stages.chain(tail))
        return result

    # ------------------------------------------------------------------ value sets
    def value_set(self, module: str, value, *, budget: int = 20_000, limit: int = 256) -> set[int] | None:
        """The finite set of values ``value`` can carry, or None when any contributor is unknown.

        Bit ranges are followed through slices and packing, multiplexer data operands, registers
        (whose hold re-reads their own value), memories (their written data) and instance
        boundaries in both directions. Comparisons carry {0, 1}. A design input above ``module``
        or any other operation makes the set unknown.
        """
        spent = [0]
        memo: dict[tuple, set[int] | None] = {}
        active: set[tuple] = set()

        def bits(frames: tuple, value, low: int, size: int) -> set[int] | None:
            spent[0] += 1
            if spent[0] > budget or size <= 0:
                return None
            module_name = frames[-1][0]
            view = self.module(module_name)
            key = (frames, id(value), low, size)
            if key in memo:
                return memo[key]
            if key in active:
                return set()  # a hold re-reads this value: it adds no new value
            active.add(key)
            result = _bits(frames, view, value, low, size)
            active.discard(key)
            if result is not None and len(result) > limit:
                result = None
            memo[key] = result
            return result

        def _bits(frames, view, value, low, size):
            mask = (1 << size) - 1
            if isinstance(value, BlockArgument):
                if len(frames) == 1:
                    return None
                caller_frames, (_, call) = frames[:-1], frames[-1]
                return bits(caller_frames, call.operands[view.inputs.index(view.input_name[value])], low, size)
            if not isinstance(value, OpResult):
                return None
            op, kind = value.owner, _name(value.owner)
            if kind == "hw.constant":
                return {(_integer(op, "value") >> low) & mask}
            if kind == "comb.icmp":
                return {0, 1} if low == 0 and size == 1 else {0}
            if kind == "comb.extract":
                return bits(frames, op.operands[0], low + _integer(op, "lowBit"), size)
            if kind == "comb.concat":
                result, offset, shift = {0}, 0, 0
                for operand in reversed(op.operands):  # least significant operand last in the list
                    part_width = width(operand) or 0
                    lo, hi = max(low, offset), min(low + size, offset + part_width)
                    if lo < hi:
                        part = bits(frames, operand, lo - offset, hi - lo)
                        if part is None or len(result) * len(part) > limit:
                            return None
                        result = {r | (p << shift) for r in result for p in part}
                        shift += hi - lo
                    offset += part_width
                return result
            if kind == "comb.mux":
                parts = [bits(frames, x, low, size) for x in op.operands[1:]]
                return None if None in parts else parts[0] | parts[1]
            if kind in REGISTERS:
                return bits(frames, op.operands[0], low, size)
            if kind == "comb.replicate":
                part = bits(frames, op.operands[0], 0, width(op.operands[0]) or 0)
                step = width(op.operands[0]) or 0
                if part is None or not step:
                    return None
                return {_replicate(v, step, low, size) for v in part}
            if kind in BITWISE:
                result = None
                for operand in op.operands:
                    part = bits(frames, operand, low, size)
                    if part is None or (result is not None and len(result) * len(part) > limit):
                        return None
                    result = part if result is None else {_bitwise(kind, a, b) for a in result for b in part}
                return result
            if kind in READ_PORTS:
                parts = [bits(frames, data, low, size) for data in self._write_data(op.operands[0].owner)]
                return None if not parts or None in parts else set().union(*parts)
            if kind == "hw.instance":
                inst = view.instance_by_op[op]
                callee = self.module(inst.callee)
                if callee.extern:
                    return None
                out = callee.output_value[inst.outputs[value.index]]
                return bits((*frames, (inst.callee, op)), out, low, size)
            return None

        return bits(((module, None),), value, 0, width(value) or 0)

    # ------------------------------------------------------------------ indexed slots
    def index_slots(self, root: str, path: str, port: str, *, max_steps: int = 100_000) -> list[dict[str, Any]]:
        """State slots selected by the value entering input ``port`` of the instance at ``path``.

        Follows the value (and its slices/zero-extensions) forward through instances in both
        directions. An array read indexed by it yields its element count; a one-hot decode (a shift
        of constant one by it) yields the decoded width; equality comparisons against constants yield
        the number of distinct compared values.
        """
        occurrences = self.instance_paths(root)
        work = [(path, self.module(occurrences[path]).input_value[port])]
        found: list[dict[str, Any]] = []
        compared: dict[str, set[int]] = {}
        seen: set[tuple[str, int]] = set()
        while work and len(seen) <= max_steps:
            where, value = work.pop()
            if (where, id(value)) in seen:
                continue
            seen.add((where, id(value)))
            view = self.module(occurrences[where])
            for use in list(value.uses):
                op, index, kind = use.operation, use.index, _name(use.operation)
                if kind == "hw.instance":
                    inst = view.instance_by_op[op]
                    if not self.module(inst.callee).extern:
                        work.append((f"{where}/{inst.name}", self.module(inst.callee).input_value[inst.inputs[index]]))
                elif kind == "hw.output" and where != root:
                    up, _, name = where.rpartition("/")
                    holder = self.module(occurrences[up]).instances[name]
                    for out in view.output_names(value):
                        work.append((up, holder.op.results[holder.outputs.index(out)]))
                elif kind == "hw.array_get" and index == 1:
                    elements = op.operands[0].owner if isinstance(op.operands[0], OpResult) else None
                    count = (
                        len(elements.operands)
                        if elements is not None and _name(elements) == "hw.array_create"
                        else None
                    )
                    found.append({"kind": "indexed_array", "instance": where, "slots": count})
                elif kind == "comb.shl" and index == 1:
                    base = op.operands[0]
                    if isinstance(base, OpResult) and _name(base.owner) == "hw.constant":
                        if _integer(base.owner, "value") == 1:
                            found.append({"kind": "one_hot_decode", "instance": where, "slots": width(op.results[0])})
                elif kind == "comb.icmp":
                    other = op.operands[1 - index]
                    if isinstance(other, OpResult) and _name(other.owner) == "hw.constant":
                        compared.setdefault(where, set()).add(_integer(other.owner, "value"))
                elif kind in {"comb.concat", "comb.extract", "hw.wire"} or (kind == "comb.mux" and index > 0):
                    work.append((where, op.results[0]))
        for where, values in sorted(compared.items()):
            found.append({"kind": "compared_constants", "instance": where, "slots": len(values)})
        unique = {(row["kind"], row["instance"], row["slots"]) for row in found}
        return [{"kind": k, "instance": i, "slots": s} for k, i, s in sorted(unique, key=str)]

    # ------------------------------------------------------------------ products and drivers
    def product_reach(self, cell: str) -> dict[str, Any]:
        """Where each multiplication result inside ``cell`` goes under the explicit pins.

        Starts at every ``comb.mul`` in the cell's closure and walks forward through data uses
        (pruning multiplexer branches a pinned select rules out), into instances and back out up to
        the cell boundary. Returns the cell outputs reached with no register in between and the cell
        registers the product is written into.
        """
        root = cell
        occurrences = self.instance_paths(root)
        work = []
        for path, name in occurrences.items():
            view = self.module(name)
            if view.extern:
                continue
            for op in view.op.walk():
                if _name(op) == "comb.mul":
                    work.append((path, op.results[0]))
        outputs, registers, seen = set(), set(), set()
        while work:
            where, value = work.pop()
            if (where, id(value)) in seen:
                continue
            seen.add((where, id(value)))
            view = self.module(occurrences[where])
            for use in list(value.uses):
                op, index, kind = use.operation, use.index, _name(use.operation)
                if kind == "hw.instance":
                    inst = view.instance_by_op[op]
                    if not self.module(inst.callee).extern:
                        work.append((f"{where}/{inst.name}", self.module(inst.callee).input_value[inst.inputs[index]]))
                elif kind == "hw.output":
                    if where == root:
                        outputs.update(view.output_names(value))
                        continue
                    up, _, name = where.rpartition("/")
                    holder = self.module(occurrences[up]).instances[name]
                    for out in view.output_names(value):
                        work.append((up, holder.op.results[holder.outputs.index(out)]))
                elif kind in REGISTERS:
                    if index == 0:
                        registers.add(f"{where}:{view.state_name(op)}")
                elif kind == "comb.mux":
                    sel = self._known(view.name, op.operands[0])
                    if index > 0 and (sel is None or index == 2 - sel):
                        work.append((where, op.results[0]))
                elif op.results and kind not in CONSTANTS:
                    work.append((where, op.results[0]))
        return {"combinational_outputs": sorted(outputs), "registers": sorted(registers)}

    def root_input_reach(self, root: str, *, max_steps: int = 5_000_000) -> set[tuple[str, int]]:
        """``(instance path, value id)`` of every value a root input can influence.

        One forward pass from all root inputs through every operation, register, memory and instance
        boundary (both directions). A value outside the set is driven only by constants and state no
        root input ever writes.
        """
        occurrences = self.instance_paths(root)
        view = self.module(root)
        work = [(root, value) for value in view.input_value.values() if str(value.type) != "!seq.clock"]
        seen: set[tuple[str, int]] = set()
        while work:
            where, value = work.pop()
            if (where, id(value)) in seen:
                continue
            seen.add((where, id(value)))
            if len(seen) > max_steps:
                raise ValueError("root input reach exceeds its explicit step budget")
            here = self.module(occurrences[where])
            for use in list(value.uses):
                op, index, kind = use.operation, use.index, _name(use.operation)
                if kind == "hw.instance":
                    inst = here.instance_by_op[op]
                    if not self.module(inst.callee).extern:
                        callee = self.module(inst.callee)
                        work.append((f"{where}/{inst.name}", callee.input_value[inst.inputs[index]]))
                    else:
                        work.extend((where, result) for result in op.results)
                elif kind == "hw.output":
                    if where != root:
                        up, _, name = where.rpartition("/")
                        holder = self.module(occurrences[up]).instances[name]
                        for out in here.output_names(value):
                            work.append((up, holder.op.results[holder.outputs.index(out)]))
                elif kind in WRITE_PORTS:
                    memory = op.operands[0].owner
                    for read in memory.results[0].uses:
                        if _name(read.operation) in READ_PORTS:
                            work.extend((where, result) for result in read.operation.results)
                else:
                    work.extend((where, result) for result in op.results)
        return seen

    def driven_by_root_inputs(self, root: str, path: str, port: str, reach: set[tuple[str, int]]) -> bool:
        """Whether the operand bound to input ``port`` of the instance at ``path`` is in ``reach``."""
        _, inst = self._instance_at(root, path)
        return (path.rpartition("/")[0], id(inst.op.operands[inst.inputs.index(port)])) in reach


def _apply(transform: tuple, stages: Stages) -> Stages:
    if transform[0] == "delay":
        return stages.then(transform[1], held=transform[2])
    if transform[0] == "opaque":
        return stages.then(0, opaque=True)
    return stages.chain(transform[1])


def _register_stage(register, loops) -> Stages:
    holds, feeds = loops[register]
    return Stages(1, None if feeds else 1, holds)


def _cyclic_registers(preds: dict[Any, dict[Any, Stages]], loops) -> set[Any]:
    """Registers with feedback: a computed self-loop or a cycle through other registers (Tarjan)."""
    cyclic = {register for register, (_, feeds) in loops.items() if feeds}
    index: dict[Any, int] = {}
    low: dict[Any, int] = {}
    on_stack: set[Any] = set()
    order: list[Any] = []

    def children(node):
        return iter([source[1] for source in preds[node] if source[0] == "reg" and source[1] in preds])

    for root in preds:
        if root in index:
            continue
        index[root] = low[root] = len(index)
        order.append(root)
        on_stack.add(root)
        work = [(root, children(root))]
        while work:
            node, pending = work[-1]
            descended = False
            for child in pending:
                if child not in index:
                    index[child] = low[child] = len(index)
                    order.append(child)
                    on_stack.add(child)
                    work.append((child, children(child)))
                    descended = True
                    break
                if child in on_stack:
                    low[node] = min(low[node], index[child])
            if descended:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = order.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member is node:
                        break
                if len(component) > 1:
                    cyclic.update(component)
    return cyclic


def _replicate(value: int, step: int, low: int, size: int) -> int:
    whole = 0
    for index in range((low + size) // step + 1):
        whole |= value << (index * step)
    return (whole >> low) & ((1 << size) - 1)


def _bitwise(kind: str, a: int, b: int) -> int:
    return a & b if kind == "comb.and" else (a | b if kind == "comb.or" else a ^ b)


def load_design(text: str, *, pins: dict[str, dict[str, int]] | None = None) -> Design:
    from .hw_graph import parse_generic_hw

    return Design(parse_generic_hw(text), pins=pins)
