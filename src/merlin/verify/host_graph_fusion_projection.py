"""Independent actual host call/body projection for logical dispatch fusion.

The admitted IR consists of total pure integer pointwise stages with unread
destinations. Function expansion, attribute interpretation and source-to-model
correspondence remain trusted. No planner report or producer expansion is used.
"""

from __future__ import annotations

import hashlib

from xdsl.dialects import builtin, func
from xdsl.ir import Block

from .gate_errors import GateUnsupported
from .lean_graph_regions_projection import _integer_tensor, _logical_operands, ir_text, parse_source

MAX_BYTES = 1 << 20
MAX_NODES = 512
MAX_EXPANDED_OPERATIONS = 4096


def _host_attributes(operation):
    if any(
        name != "merlin.placement" or value != builtin.StringAttr("host")
        for name, value in operation.attributes.items()
    ):
        raise GateUnsupported("host fusion has unknown metadata or non-host placement")


def _primitive_key(operation):
    # Outer operands become independent formal ports. Actual aliasing is in
    # Node.operands, never inferred from a spelling or a producer-selected ID.
    formal = Block(arg_types=[value.type for value in operation.operands])
    cloned = operation.clone()
    cloned.operands = tuple(formal.args)
    for op in cloned.walk():
        for value in op.results:
            value.name_hint = None
        for region in op.regions:
            for block in region.blocks:
                for value in block.args:
                    value.name_hint = None
    return hashlib.sha256(ir_text(cloned).encode()).hexdigest()


def _functions(module):
    if module.attributes or module.properties:
        raise GateUnsupported("host fusion requires an unannotated logical module")
    functions, public = {}, []
    for op in module.body.block.ops:
        if not isinstance(op, func.FuncOp) or len(op.body.blocks) != 1:
            raise GateUnsupported("host fusion requires defined single-block functions")
        _host_attributes(op)
        if set(op.properties) - {"sym_name", "function_type", "sym_visibility"}:
            raise GateUnsupported("host fusion has unsupported function properties")
        name = op.sym_name.data
        if name in functions:
            raise GateUnsupported("host fusion has duplicate function definitions")
        functions[name] = op
        visibility = op.properties.get("sym_visibility")
        if visibility is None or visibility == builtin.StringAttr("public"):
            public.append(name)
        elif visibility != builtin.StringAttr("private"):
            raise GateUnsupported("host fusion has unsupported function visibility")
        for value in op.body.block.args:
            _integer_tensor(value)
        # Validate unreachable private definitions as well. They do not become
        # public entry points, but unknown effects are outside this profile.
        for child in op.body.block.ops:
            if child.name == "linalg.generic":
                _logical_operands(child)
            elif child.name == "tensor.empty":
                if child.operands or len(child.results) != 1 or child.attributes or child.properties:
                    raise GateUnsupported("host fusion requires plain static tensor constructors")
                _integer_tensor(child.results[0])
            elif child.name == "func.call":
                _host_attributes(child)
                if set(child.properties) != {"callee"} or child.regions or child.successors:
                    raise GateUnsupported("host fusion has unsupported call effects")
                if child.callee.nested_references.data:
                    raise GateUnsupported("host fusion requires flat direct calls")
            elif child.name == "func.return":
                if child.attributes or child.properties or child.regions or child.successors:
                    raise GateUnsupported("host fusion has unsupported return metadata")
            else:
                raise GateUnsupported("host fusion does not model " + child.name)
    if len(public) != 1:
        raise GateUnsupported("host fusion currently requires one public entry")
    return functions, public[0]


def project(source: str) -> dict:
    """Project all public results and actual ordered callee bodies without hints."""
    if type(source) is not str or not source.isascii() or len(source.encode()) > MAX_BYTES:
        raise GateUnsupported("host fusion input exceeds the IR byte budget")
    try:
        module = parse_source(source)
        functions, entry = _functions(module)
    except GateUnsupported:
        raise
    except Exception as error:
        raise GateUnsupported("host fusion input is outside the supported typed IR") from error
    nodes, groups = [], []
    arguments = functions[entry].body.block.args
    next_value, operations = len(arguments), 0

    def execute(name, values, stack):
        nonlocal next_value, operations
        if name not in functions or name in stack or len(stack) >= 32:
            raise GateUnsupported("host fusion refuses external, recursive or overly deep calls")
        block = functions[name].body.block
        if len(values) != len(block.args):
            raise GateUnsupported("host fusion call arity differs")
        environment = dict(zip(block.args, values, strict=True))
        result = None
        for op in block.ops:
            operations += 1
            if operations > MAX_EXPANDED_OPERATIONS or len(nodes) > MAX_NODES:
                raise GateUnsupported("host fusion exceeds its syntax expansion budget")
            if result is not None or any(value not in environment for value in op.operands):
                raise GateUnsupported("host fusion has an early return or captured SSA value")
            if op.name == "tensor.empty":
                environment[op.results[0]] = None
            elif op.name == "linalg.generic":
                if len(nodes) >= MAX_NODES:
                    raise GateUnsupported("host fusion exceeds its projected node budget")
                inputs, _, _ = _logical_operands(op)
                ids = [environment[value] for value in inputs]
                if any(value is None for value in ids):
                    raise GateUnsupported("host fusion cannot read an uninitialized tensor")
                node = {
                    "ordinal": len(nodes),
                    "semanticKey": _primitive_key(op),
                    "operands": ids,
                    "results": [next_value],
                    "operandTypes": [str(value.type) for value in inputs],
                    "resultTypes": [str(op.results[0].type)],
                }
                nodes.append(node)
                environment[op.results[0]] = next_value
                next_value += 1
            elif op.name == "func.call":
                start = len(nodes)
                inputs = [environment[value] for value in op.operands]
                outputs = execute(op.callee.root_reference.data, inputs, (*stack, name))
                if len(outputs) != len(op.results):
                    raise GateUnsupported("host fusion call result arity differs")
                environment.update(zip(op.results, outputs, strict=True))
                if not stack:
                    groups.append(
                        {
                            "start": start,
                            "count": len(nodes) - start,
                            "body": nodes[start:].copy(),
                            "inputs": list(dict.fromkeys(v for v in inputs if v is not None)),
                            "outputs": outputs,
                        }
                    )
            else:
                result = [environment[value] for value in op.operands]
                if any(value is None for value in result):
                    raise GateUnsupported("host fusion cannot publish an uninitialized tensor")
        if result is None:
            raise GateUnsupported("host fusion function has no return")
        return result

    outputs = execute(entry, list(range(len(arguments))), ())
    graph = {
        "nodes": nodes,
        "inputs": list(range(len(arguments))),
        "observations": outputs,
        "context": [entry, str(functions[entry].function_type)],
    }
    return {"entry": entry, "graph": graph, "calls": groups}


def project_pair(before: str, after: str) -> dict:
    source, target = project(before), project(after)
    if source["entry"] != target["entry"]:
        raise GateUnsupported("host fusion changed the public entry")
    if any(call["count"] > 1 for call in source["calls"]):
        raise GateUnsupported("host fusion source already contains a multi-stage dispatch")
    groups = [call for call in target["calls"] if call["count"] > 1]
    if len(groups) != 1:
        raise GateUnsupported("host fusion currently checks one actual fused region per entry")
    return {"before": source, "after": target, "fusion": groups[0]}
