"""Bounded exact source value connectivity, without endpoint semantic roles.

Selections identify original SSA slots in an exact source and occurrence. A
graph may follow defined combinational module bindings. State, clocks, memory,
parameterized/external bodies and unsupported values stay explicit cuts. The
complete original state/effect membership of visited definitions is retained.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import BlockArgument, OpResult

from merlin.targetgen.contract.mlir_source_admission import admit_mlir_source

from .hw_combinational import _expression
from .hw_graph import parse_generic_hw
from .hw_hierarchy_bindings import _binding, _parameters, _static_signature
from .hw_memory_ports import _COMBINATIONAL
from .hw_observations import _module_name, _name

SCHEMA = "merlin.hw_value_bindings.v1"
_STATE = {"seq.firreg", "seq.compreg", "seq.compreg.ce", "seq.shiftreg"}
_CLOCK = {"seq.to_clock", "seq.from_clock"}
_PURE = _COMBINATIONAL | {"comb.replicate", "comb.shru"}
_KINDS = {"module_input", "module_output", "operation_result", "operation_operand"}


@dataclass(frozen=True)
class ValueBindingLimits:
    source_bytes: int
    modules: int
    operations: int
    occurrences: int
    port_bindings: int
    selections: int
    nodes: int
    scalar_bits: int
    bit_work: int
    hierarchy_depth: int
    expression_depth: int
    metadata_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError("Value bindings require explicit positive limits.")


@dataclass(frozen=True)
class OriginalValueSelection:
    source_sha256: str
    occurrence: tuple[str, ...]
    module: str
    kind: str
    ordinal: int
    slot: int
    type: str

    def __post_init__(self):
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_sha256)
            or type(self.occurrence) is not tuple
            or not self.occurrence
            or any(type(name) is not str or not name for name in self.occurrence)
            or type(self.module) is not str
            or not self.module
            or type(self.kind) is not str
            or self.kind not in _KINDS
            or type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.slot) is not int
            or self.slot < 0
            or type(self.type) is not str
            or not self.type
            or (self.kind in {"module_input", "module_output"} and self.slot != 0)
        ):
            raise ValueError("Value binding selection has unsupported identity fields.")


def prepare_value_bindings(text, *, root, selections, limits):
    """Parse exact selected bytes; selection descriptors grant no source roles."""
    if type(limits) is not ValueBindingLimits or type(text) is not str or len(text) > limits.source_bytes:
        raise ValueError("Value binding source exceeds its explicit parse budget.")
    if (
        type(root) is not str
        or not root
        or type(selections) is not tuple
        or not selections
        or len(selections) > limits.selections
        or any(type(row) is not OriginalValueSelection for row in selections)
        or len(set(selections)) != len(selections)
    ):
        raise ValueError("Value bindings require an exact nonempty selection roster.")
    admit_mlir_source(
        text,
        max_source_bytes=limits.source_bytes,
        max_nesting=64,
        max_integer_bits=max(64, limits.scalar_bits),
        allow_dense=False,
        allow_dense_resource=False,
    )
    identity = hashlib.sha256(text.encode()).hexdigest()
    if any(row.source_sha256 != identity for row in selections):
        raise ValueError("Value binding source identity differs from its selections.")
    result = hierarchical_value_bindings(
        parse_generic_hw(text, reject_dense_literals=True), root=root, selections=selections, limits=limits
    )
    result["source_sha256"] = identity
    result["source_byte_binding"] = "exact supplied source bytes"
    return result


def hierarchical_value_bindings(parsed, *, root, selections, limits):
    """Export structural dependencies only, preserving all original source cuts."""
    if (
        type(limits) is not ValueBindingLimits
        or type(root) is not str
        or not root
        or type(selections) is not tuple
        or not selections
        or len(selections) > limits.selections
        or any(type(row) is not OriginalValueSelection for row in selections)
        or len(set(selections)) != len(selections)
    ):
        raise ValueError("Value bindings require explicit limits and original selections.")
    modules, operation_count = {}, 0
    for op in parsed.walk():
        operation_count += 1
        if operation_count > limits.operations:
            raise ValueError("Value binding source exceeds its complete operation budget.")
        if _name(op) in {"hw.module", "hw.module.extern"}:
            name = _module_name(op)
            if name in modules or len(modules) >= limits.modules:
                raise ValueError("Value binding module membership differs or exceeds its budget.")
            modules[name] = op
    if root not in modules or _name(modules[root]) != "hw.module" or _parameters(modules[root]):
        raise ValueError("Value binding root body is unavailable or parameterized.")

    definitions, operation_paths, block_paths = {}, {}, {}
    for name, module in modules.items():
        ports = _static_signature(module)
        block = module.regions[0].block if _name(module) == "hw.module" else None
        ops = tuple(block.ops) if block is not None else ()
        children, names = {}, set()
        for ordinal, op in enumerate(ops):
            if _name(op) == "hw.instance":
                info = _binding(op, modules)
                if info[0] in names or op.regions:
                    raise ValueError("Value binding instance membership is ambiguous.")
                names.add(info[0])
                children[op] = ordinal, info
        definitions[name] = ports, block, ops, {op: i for i, op in enumerate(ops)}, children
        operation_paths[name], block_paths[name] = {}, {}

        def locate(original_block, path):
            block_paths[name][original_block] = path
            for ordinal, child in enumerate(original_block.ops):
                child_path = path + (ordinal,)
                operation_paths[name][child] = child_path
                for region_ordinal, region in enumerate(child.regions):
                    for block_ordinal, nested in enumerate(region.blocks):
                        locate(nested, child_path + (region_ordinal, block_ordinal))

        if block is not None:
            locate(block, ())

    sizes = {}

    def size(module, ancestors=()):
        if module in ancestors or len(ancestors) >= limits.hierarchy_depth:
            raise ValueError("Value binding hierarchy is recursive or exceeds its depth budget.")
        if module in sizes:
            count, ports, height = sizes[module]
        else:
            count, ports, height = 1, 0, 1
            for _, (_, (_, callee, signature, _)) in definitions[module][4].items():
                child_count, child_ports, child_height = size(callee, ancestors + (module,))
                count += child_count
                ports += len(signature) + child_ports
                height = max(height, child_height + 1)
                if count > limits.occurrences or ports > limits.port_bindings:
                    raise ValueError("Value binding hierarchy exceeds its pre-expansion budget.")
            sizes[module] = count, ports, height
        if len(ancestors) + height > limits.hierarchy_depth:
            raise ValueError("Value binding hierarchy exceeds its complete depth budget.")
        return count, ports, height

    # Validate unused definition edges too; no malformed body is silently dropped.
    for module in modules:
        size(module)
    expected, expected_ports, height = size(root)
    if expected > limits.occurrences or expected_ports > limits.port_bindings:
        raise ValueError("Value binding hierarchy exceeds its pre-expansion budget.")
    frames, by_path, child_frames = [], {}, {}

    def expand(module, path, parent=None, incoming=None, boundary=None):
        index = len(frames)
        if path in by_path:
            raise ValueError("Value binding occurrence identity is ambiguous.")
        by_path[path] = index
        ports, block, _, _, children = definitions[module]
        stop = boundary or (
            "external_body" if block is None else "parameterized_body" if _parameters(modules[module]) else None
        )
        frames.append(
            {
                "id": index,
                "module": module,
                "path": list(path),
                "parent": parent,
                "stop": stop,
                "original_ports": [asdict(port) for port in ports],
                "incoming": incoming,
            }
        )
        for op, (_, (name, callee, _, child_stop)) in children.items():
            child_frames[index, op] = expand(callee, path + (name,), index, op, child_stop)
        return index

    expand(root, (root,))
    if len(frames) != expected:
        raise ValueError("Value binding complete occurrence membership differs.")
    totals = {
        "modules": len(modules),
        "operations": operation_count,
        "occurrences": expected,
        "port_bindings": expected_ports,
        "hierarchy_depth": height,
        "nodes": 0,
        "bit_work": 0,
        "metadata_bytes": 0,
    }

    def metadata(op):
        attrs = {key: str(value) for key, value in op.attributes.items()}
        props = {key: str(value) for key, value in op.properties.items()}
        totals["metadata_bytes"] += sum(
            len(key.encode()) + len(value.encode()) for key, value in (*attrs.items(), *props.items())
        )
        if totals["metadata_bytes"] > limits.metadata_bytes:
            raise ValueError("Value binding metadata exceeds its aggregate budget.")
        return {"original_attributes": attrs, "original_properties": props, "location": str(op.location)}

    def reference(module, value):
        _, block, _, indices, _ = definitions[module]
        if isinstance(value, BlockArgument) and value.block is block:
            return {"kind": "module_input", "ordinal": value.index, "type": str(value.type)}
        if isinstance(value, OpResult) and value.owner.parent is block:
            return {
                "kind": "operation_result",
                "ordinal": indices[value.owner],
                "slot": value.index,
                "type": str(value.type),
            }
        if isinstance(value, OpResult) and value.owner in operation_paths[module]:
            return {
                "kind": "nested_operation_result",
                "path": list(operation_paths[module][value.owner]),
                "slot": value.index,
                "type": str(value.type),
            }
        if isinstance(value, BlockArgument) and value.block in block_paths[module]:
            return {
                "kind": "nested_block_argument",
                "path": list(block_paths[module][value.block]),
                "ordinal": value.index,
                "type": str(value.type),
            }
        raise ValueError("Value binding original SSA ownership is unavailable.")

    def width(value):
        if str(value.type) == "!seq.clock":
            return None
        if not isinstance(value.type, IntegerType) or value.type != IntegerType(value.type.width.data):
            return None
        bits = value.type.width.data
        if not 0 < bits <= limits.scalar_bits:
            raise ValueError("Value binding scalar exceeds its explicit width budget.")
        return bits

    nodes, visiting, visited_modules = {}, set(), set()

    def trace(frame_id, value, depth=0):
        key = frame_id, value
        if key in nodes:
            return nodes[key]["id"]
        if key in visiting or depth >= limits.expression_depth:
            raise ValueError("Value binding expression is cyclic or exceeds its depth budget.")
        if totals["nodes"] >= limits.nodes:
            raise ValueError("Value binding expression exceeds its node budget.")
        bits = width(value)
        totals["bit_work"] += bits or 0
        if totals["bit_work"] > limits.bit_work:
            raise ValueError("Value binding expression exceeds its bit-work budget.")
        row = {"id": totals["nodes"], "frame": frame_id, "type": str(value.type), "width": bits}
        totals["nodes"] += 1
        frame = frames[frame_id]
        module = frame["module"]
        visited_modules.add(module)
        _, block, _, indices, _ = definitions[module]
        row["original_value"] = reference(module, value)
        visiting.add(key)
        if frame["stop"] is not None:
            row.update(kind="opaque_instance_result", stop=frame["stop"])
        elif str(value.type) == "!seq.clock":
            row.update(kind="clock_value")
        elif isinstance(value, BlockArgument) and value.block is block:
            parent = frame["parent"]
            if parent is None:
                row.update(kind="root_input")
            else:
                row.update(
                    kind="instance_input_binding",
                    operands=[trace(parent, frame["incoming"].operands[value.index], depth + 1)],
                )
        elif isinstance(value, OpResult) and value.owner.parent is block:
            op, kind = value.owner, _name(value.owner)
            row.update(
                operation=kind,
                **metadata(op),
                original_operand_values=[reference(module, operand) for operand in op.operands],
                original_result_types=[str(result.type) for result in op.results],
                original_regions=len(op.regions),
            )
            if kind == "hw.instance":
                child = child_frames[frame_id, op]
                callee = frames[child]["module"]
                stop = frames[child]["stop"]
                if stop is not None:
                    row.update(kind="opaque_instance_result", child_frame=child, stop=stop)
                else:
                    output = definitions[callee][1].last_op.operands[value.index]
                    row.update(
                        kind="instance_output_binding", child_frame=child, operands=[trace(child, output, depth + 1)]
                    )
            elif kind in _STATE:
                row.update(kind="state_result")
            elif kind in _CLOCK:
                row.update(kind="clock_value")
            elif kind in {"seq.firmem.read_port", "seq.firmem.read_write_port"}:
                row.update(kind="memory_read_result")
            elif kind in _PURE and bits is not None and not op.regions and len(op.results) == 1:
                widths = [width(operand) for operand in op.operands]
                if any(bit is None for bit in widths):
                    row.update(kind="unsupported_result")
                else:
                    try:
                        expression, parameter = _expression(op, widths, bits, conditional_logic=True)
                    except ValueError as error:
                        row.update(kind="unsupported_result", reason=str(error))
                    else:
                        totals["bit_work"] += sum(widths)
                        if totals["bit_work"] > limits.bit_work:
                            raise ValueError("Value binding expression exceeds its bit-work budget.")
                        row.update(
                            kind="combinational",
                            expression=expression,
                            parameter=parameter,
                            operands=[trace(frame_id, operand, depth + 1) for operand in op.operands],
                        )
            else:
                row.update(kind="unsupported_result")
        else:
            row.update(kind="unsupported_result")
        visiting.remove(key)
        nodes[key] = row
        return row["id"]

    selected = []
    for selection in selections:
        frame_id = by_path.get(selection.occurrence)
        if frame_id is None or frames[frame_id]["module"] != selection.module:
            raise ValueError("Value binding selected occurrence differs from original source.")
        _, block, ops, _, _ = definitions[selection.module]
        if block is None or frames[frame_id]["stop"] is not None:
            raise ValueError("Value binding selected body is opaque.")
        try:
            if selection.kind == "module_input":
                value = block.args[selection.ordinal]
            elif selection.kind == "module_output":
                value = block.last_op.operands[selection.ordinal]
            elif selection.kind == "operation_result":
                value = ops[selection.ordinal].results[selection.slot]
            else:
                value = ops[selection.ordinal].operands[selection.slot]
        except IndexError:
            raise ValueError("Value binding selected original SSA slot is unavailable.") from None
        if str(value.type) != selection.type:
            raise ValueError("Value binding selected original type differs.")
        selected.append({"selection": asdict(selection), "frame": frame_id, "value": trace(frame_id, value)})

    members = []
    for module in modules:
        if module not in visited_modules:
            continue
        ports, block, ops, _, _ = definitions[module]
        retained = []
        for ordinal, op in enumerate(ops):
            retained.append(
                {
                    "ordinal": ordinal,
                    "operation": _name(op),
                    **metadata(op),
                    "operands": [reference(module, value) for value in op.operands],
                    "result_types": [str(value.type) for value in op.results],
                    "regions": [
                        [[str(v.type) for v in block.args] for block in region.blocks] for region in op.regions
                    ],
                    "nested_operations": [
                        {
                            "operation": _name(child),
                            **metadata(child),
                            "path": list(operation_paths[module][child]),
                            "operands": [reference(module, v) for v in child.operands],
                            "result_types": [str(v.type) for v in child.results],
                            "regions": [
                                [[str(v.type) for v in block.args] for block in region.blocks]
                                for region in child.regions
                            ],
                        }
                        for child in op.walk()
                        if child is not op
                    ],
                }
            )
        members.append(
            {
                "module": module,
                "original_ports": [asdict(port) for port in ports],
                "operation_count": len(ops),
                **metadata(modules[module]),
                "complete_operation_members": retained,
                "member_semantics": "UNKNOWN",
            }
        )
    public_frames = []
    for frame in frames:
        row = {key: value for key, value in frame.items() if key != "incoming"}
        op = frame["incoming"]
        if op is not None:
            parent_module = frames[frame["parent"]]["module"]
            row["incoming_binding"] = {
                "ordinal": definitions[parent_module][3][op],
                **metadata(op),
                "operands": [reference(parent_module, value) for value in op.operands],
                "result_types": [str(value.type) for value in op.results],
            }
        public_frames.append(row)
    return {
        "schema": SCHEMA,
        "root": root,
        "limits": asdict(limits),
        "cost": totals,
        "frames": public_frames,
        "selections": selected,
        "expressions": sorted(nodes.values(), key=lambda row: row["id"]),
        "visited_definition_members": members,
        "unknowns": [
            "source_member_semantic_roles",
            "unvisited_expression_semantics",
            "state_clock_memory_and_opaque_cut_semantics",
            "source_and_runtime_correspondence",
            "getter_return_sample_event_custody",
            "physical_units_and_complete_costs",
        ],
        "scope": "conditional typed SSA connectivity only",
        "admission_authority": False,
        "source_byte_binding": "not reopened by parsed-graph API",
    }
