"""Observe exact HW input slices and equality constants without assigning ISA roles.

The observer follows only block arguments, extracts and contiguous concatenations.
Registers, memories, instances and every other operation terminate tracing. Local
observations never become a complete instruction set or an effect certificate.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, StringAttr, SymbolRefAttr, UnregisteredAttr
from xdsl.ir import BlockArgument, OpResult

from .ports import _hw_port_entries


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
