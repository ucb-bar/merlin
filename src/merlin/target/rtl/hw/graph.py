"""Lossless generic CIRCT HW graph ingest, including typed parameter attributes."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr, SymbolRefAttr, UnregisteredAttr
from xdsl.ir import BlockArgument, OpResult

from merlin.target.rtl.firrtl.ports import _hw_port_entries
from merlin.target.rtl.hw.instance_ports import _instance, _memory_type, _signature

# ---- hw_graph ----------------------------------------------------------------------------------


def parse_generic_hw(text: str, *, reject_dense_literals: bool = False):
    """Parse a lossless generic HW module without requiring discovery tooling.

    xDSL's unregistered-attribute parser stops at ``>``; CIRCT's HW parameter
    declarations may also carry a trailing type. Preserve that type in the opaque
    attribute representation rather than deleting external modules or parameters.
    The graph is for analysis only; never serialize it as replacement hardware.
    The explicit scalar-observation mode rejects builtin dense literals before
    xDSL expands a shaped splat. The historical parser remains unchanged by
    default; the mode is not a general parser or process resource sandbox.
    """
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin, UnregisteredAttr
    from xdsl.parser import Parser

    class HardwareParser(Parser):
        def _parse_builtin_dense_attr(self):
            if reject_dense_literals:
                self.raise_error("scalar HW observation refuses dense literals before materialization")
            return super()._parse_builtin_dense_attr()

        def _parse_dialect_type_or_attribute_body(self, attr_name, is_type, is_opaque, starting_opaque_pos):
            attribute = super()._parse_dialect_type_or_attribute_body(
                attr_name, is_type, is_opaque, starting_opaque_pos
            )
            if attr_name == "hw.param.decl" and self.parse_optional_punctuation(":") is not None:
                parameter_type = self.parse_type()
                if not isinstance(attribute, UnregisteredAttr):
                    self.raise_error("unexpected registered HW parameter declaration parser")
                return type(attribute)(
                    attr_name, is_type, is_opaque, attribute.value.data + " : " + str(parameter_type)
                )
            return attribute

    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    return HardwareParser(context, text).parse_module()


def load_hw_graph(path: str | Path, *, circt_opt):
    """Use upstream discovery graphs without changing selected hardware bytes."""
    from mlc.discover.irgraph import HwGraph, to_generic

    from merlin.target.rtl.extraction.source_selection import active_selection, digest

    selected = active_selection()
    if selected is None:
        generic = to_generic(path, circt_opt=circt_opt)
    else:
        generic = Path(selected["_generic_hw_output"])
        if generic.is_symlink():
            raise ValueError("selected CIRCT genericization output may not be a symlink")
        receipt = selected.get("_genericization")
        if receipt is None:
            # Bind the executed command to the same canonical paths as its
            # receipt. Relative argv otherwise makes later validation depend
            # on the launcher's cwd, even when every selected byte is intact.
            generic = generic.resolve()
            source, tool = Path(path).resolve(), Path(circt_opt).resolve()
            generic.parent.mkdir(parents=True, exist_ok=True)
            command = [str(tool), "--mlir-print-op-generic", str(source), "-o", str(generic)]
            subprocess.run(command, check=True, capture_output=True)
            selected["_genericization"] = {
                "kind": "circt_generic_serialization",
                "command": command,
                "returncode": 0,
                "input": {"path": str(source), "sha256": digest(source)},
                "output": {"path": str(generic), "sha256": digest(generic)},
                "tool": {"path": str(tool), "sha256": digest(tool)},
            }
        elif digest(path) != receipt["input"]["sha256"] or digest(generic) != receipt["output"]["sha256"]:
            raise ValueError("selected CIRCT genericization bytes changed during observation")
    original = parse_generic_hw(generic.read_text())
    projected, declared = _discovery_projection(original)
    graph = HwGraph(projected)
    actual = {
        name: (member.name, tuple((p.direction, p.name, p.type_str) for p in member.ports))
        for name, member in graph.modules.items()
    }
    if actual != declared:
        raise ValueError("HW discovery graph differs from original module/port identity")
    return graph


def _discovery_projection(module):
    """Adapt unique original fields to discovery's property-only accessors.

    Legacy CIRCT stores these fields in attributes; current CIRCT uses
    properties. Only this private analysis clone is projected. Selected generic
    source bytes and the lossless parser's original representation are unchanged.
    Duplicate field ownership or hardware symbols refuse before discovery can
    silently overwrite a module. This is representation compatibility, not
    hardware legality or execution evidence.
    """
    from xdsl.dialects.builtin import StringAttr, UnregisteredAttr

    from merlin.target.rtl.firrtl.ports import _hw_port_entries

    projected = module.clone()
    symbols, declared = set(), {}
    for op in projected.walk():
        kind = _name(op)
        for field in op.attributes.keys() & op.properties.keys():
            _attribute(op, field)
        if kind in {"hw.module", "hw.module.extern"}:
            symbol = _attribute(op, "sym_name")
            if not isinstance(symbol, StringAttr) or not symbol.data:
                raise ValueError("HW discovery module has no complete explicit symbol name")
            if symbol.data in symbols:
                raise ValueError("HW discovery source has duplicate module symbol names")
            symbols.add(symbol.data)
            if kind == "hw.module":
                _inputs(op)
                typ = _attribute(op, "module_type")
                if not isinstance(typ, UnregisteredAttr) or typ.attr_name.data != "hw.modty":
                    raise ValueError("HW discovery module has no complete original port type")
                entries = _hw_port_entries("(" + typ.value.data + ")")
                if entries is None:
                    raise ValueError("HW discovery module has unreadable original ports")
                ports = []
                for entry in entries:
                    head, separator, type_text = entry.partition(":")
                    words = head.split()
                    if not separator or len(words) != 2 or words[0] not in {"input", "output"}:
                        raise ValueError("HW discovery graph cannot retain original port form")
                    ports.append((words[0], words[1], type_text.strip()))
                if len({name for _, name, _ in ports}) != len(ports):
                    raise ValueError("HW discovery module has duplicate original port names")
                declared[symbol.data] = (symbol.data, tuple(ports))
        if op.name == "builtin.unregistered":
            for field in tuple(op.attributes):
                if field != "op_name__":
                    op.properties[field] = op.attributes.pop(field)
    return projected, declared


# ---- hw_observations ---------------------------------------------------------------------------
# Observe exact HW input slices and equality constants without assigning ISA roles.
#
# The observer follows only block arguments, extracts and contiguous concatenations.
# Registers, memories, instances and every other operation terminate tracing. Local
# observations never become a complete instruction set or an effect certificate.


def _name(op):
    return op.attributes["op_name__"].data if op.name == "builtin.unregistered" else op.name


def _width(value):
    return value.type.width.data if isinstance(value.type, IntegerType) else None


def _attribute(op, name):
    if name in op.attributes and name in op.properties:
        raise ValueError("HW source has ambiguous attribute/property ownership")
    return op.attributes.get(name, op.properties.get(name))


def _module_name(op):
    name = _attribute(op, "sym_name")
    if not isinstance(name, StringAttr):
        raise ValueError("HW module has no complete explicit symbol name")
    return name.data


def _integer(op, name):
    value = _attribute(op, name)
    return value.value.data if isinstance(value, IntegerAttr) else None


def _inputs(op):
    typ = _attribute(op, "module_type")
    if not isinstance(typ, UnregisteredAttr) or typ.attr_name.data != "hw.modty":
        raise ValueError("HW module has no lossless explicit module type")
    entries = _hw_port_entries("(" + typ.value.data + ")")
    if entries is None:
        raise ValueError("HW module port entries could not be read completely")
    names, types = [], []
    for entry in entries:
        head, separator, type_text = entry.partition(":")
        words = head.split()
        if not separator or len(words) != 2 or words[0] not in {"input", "output", "inout"}:
            raise ValueError("HW module contains an unreadable port")
        if words[0] in {"input", "inout"}:
            names.append(words[1])
            types.append(type_text.strip())
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("HW module has no single analysis block")
    block = op.regions[0].block
    if len(block.args) != len(names) or len(set(names)) != len(names):
        raise ValueError("HW module input names do not exactly match arguments")
    if any(declared != str(value.type) for declared, value in zip(types, block.args, strict=True)):
        raise ValueError("HW module declared input types differ from original SSA")
    return dict(zip(block.args, names, strict=True))


def _instance_output(value):
    """Describe a direct instance result; this never follows into the instance."""
    if not isinstance(value, OpResult) or _name(value.owner) != "hw.instance" or _width(value) is None:
        return None
    op = value.owner
    names, module, instance = (_attribute(op, key) for key in ("resultNames", "moduleName", "instanceName"))
    if (
        not isinstance(names, ArrayAttr)
        or len(names.data) != len(op.results)
        or not all(isinstance(name, StringAttr) for name in names.data)
        or not isinstance(module, SymbolRefAttr)
        or module.nested_references.data
        or not isinstance(instance, StringAttr)
    ):
        raise ValueError("HW instance result lacks complete explicit module, instance and result names")
    return instance.data, module.root_reference.data, names.data[value.index].data, _width(value)


def input_observations(module):
    """Return exact local slices/comparisons; opaque operations remain unknown."""
    from xdsl.dialects.comb import ICMP_COMPARISON_OPERATIONS

    # Use the installed generic dialect's equality predicate vocabulary.
    equality = ICMP_COMPARISON_OPERATIONS.index("eq")
    out = []
    for op in module.walk():
        if _name(op) != "hw.module":
            continue
        inputs = _inputs(op)
        records = {}
        instance_records = {}

        def trace(value, seen=frozenset()):
            if value in seen or _width(value) is None:
                return None
            if isinstance(value, BlockArgument):
                return (inputs[value], 0, _width(value)) if value in inputs else None
            if not isinstance(value, OpResult):
                return None
            parent, seen = value.owner, seen | {value}
            if _name(parent) == "comb.extract" and len(parent.operands) == 1 and len(parent.results) == 1:
                base = trace(parent.operands[0], seen)
                low = _integer(parent, "lowBit")
                width = _width(value)
                if base is not None and low is not None and low >= 0 and low + width <= base[2]:
                    return (base[0], base[1] + low, width)
            if _name(parent) == "comb.concat" and len(parent.results) == 1:
                parts = [trace(part, seen) for part in reversed(parent.operands)]
                if not parts or any(part is None for part in parts):
                    return None
                name, low, total = parts[0]
                for part in parts[1:]:
                    if part[0] != name or part[1] != low + total:
                        return None
                    total += part[2]
                if total == _width(value):
                    return name, low, total
            return None

        def record(value):
            observed = trace(value)
            if observed is None:
                return None
            return records.setdefault(observed, set())

        for child in op.regions[0].block.ops:
            if _name(child) in {"comb.extract", "comb.concat"}:
                for value in child.results:
                    record(value)
            if _name(child) != "comb.icmp" or _integer(child, "predicate") != equality:
                continue
            for signal, constant in (tuple(child.operands), tuple(reversed(child.operands))):
                if not isinstance(constant, OpResult) or _name(constant.owner) != "hw.constant":
                    continue
                value = _integer(constant.owner, "value")
                bucket = record(signal)
                if value is not None and bucket is not None and _width(signal) == _width(constant):
                    bucket.add(value % (1 << _width(signal)))
                boundary = _instance_output(signal)
                if value is not None and boundary is not None and _width(signal) == _width(constant):
                    instance_records.setdefault(boundary, set()).add(value % (1 << _width(signal)))
        out.append(
            {
                "module": _module_name(op),
                "inputs": [{"name": name, "width": _width(arg)} for arg, name in inputs.items()],
                "observations": [
                    {"input": name, "offset": low, "width": width, "equality_constants": sorted(values)}
                    for (name, low, width), values in sorted(records.items())
                ],
                "instance_output_observations": [
                    {
                        "scope": "opaque_instance_result",
                        "instance": instance,
                        "module": module,
                        "output": name,
                        "width": width,
                        "equality_constants": sorted(values),
                        "input_correspondence": "unknown",
                        "instance_effects": "unknown",
                    }
                    for (instance, module, name, width), values in sorted(instance_records.items())
                ],
            }
        )
    return {"scope": "local_module_input", "complete_isa": False, "modules": out}


# ---- hw_design ---------------------------------------------------------------------------------
# Hierarchical structural walks over one elaborated CIRCT HW design.
#
# The readers answer *structure* questions about state between ports, never semantics:
#
# * :meth:`Design.read_storage` -- the storage words that can present their value at a point
#   (registers, memory rows, a bypass from an input). Its size is the number of locations a reader
#   of that point can select: a FIFO's entries at its dequeue port, a reservation table's slots at an
#   issue port.
# * :meth:`Design.buffers` -- a module's outputs that read local storage written from one of its
#   inputs, with that storage's location count. This is a capacity, never a latency.
# * :meth:`Design.forward_buffers` -- the buffers a payload crosses after leaving a port, and whether
#   it arrives intact (a whole-value copy) or as derived bits.
# * :meth:`Design.port_stages` -- register stages over every data path between two ports, with the
#   conditions that keep a count from being a fixed latency (an enable/hold, storage retention,
#   feedback, an opaque element).
# * :meth:`Design.value_set` -- the finite set of values a bit field can carry, or None.
# * :meth:`Design.index_slots` -- state slots selected by a value used as an index or decoded.
#
# Data paths follow multiplexer data operands, slices, packing, bitwise selection and arithmetic;
# multiplexer selects and array indices are control. Every reader is target-agnostic: modules,
# ports and instances arrive as parameters from the target's own declarations.

REGISTERS = frozenset({"seq.firreg", "seq.compreg"})


READ_PORTS = frozenset({"seq.firmem.read_port", "seq.firmem.read_write_port"})


WRITE_PORTS = frozenset({"seq.firmem.write_port", "seq.firmem.read_write_port"})


CARRIERS = frozenset(
    {"comb.mux", "comb.extract", "comb.concat", "hw.array_get", "hw.array_create", "hw.array_concat", "hw.wire"}
)


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

    from merlin.target.rtl.firrtl.ports import _hw_port_entries
    from merlin.target.rtl.hw.instance_ports import OriginalPort

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
