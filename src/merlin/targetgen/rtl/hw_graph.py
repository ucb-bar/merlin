"""Lossless generic CIRCT HW graph ingest, including typed parameter attributes."""

import subprocess
from pathlib import Path


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

    from .source_selection import active_selection, digest

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

    from .hw_observations import _attribute, _inputs, _name
    from .ports import _hw_port_entries

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
