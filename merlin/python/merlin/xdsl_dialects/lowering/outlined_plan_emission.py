"""Emit a whole-model fusion plan as verified linalg functions and a new driver.

This adapter owns no accelerator facts. It preserves each selected operation's IR and brings
producer/consumer operations inside one function, exposing their tensors to subsequent bufferization
and target fusion. It proves exact computation by recursively inlining the before/after drivers and
checking structural equivalence. It does not claim that a target eliminated those tensors in memory,
or that fewer calls alone prove a cycle improvement.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
import hashlib

from .dispatch_program import DispatchProgram, Node, build_dispatch_program
from .global_plan import (
    BufferRepresentation, CycleInterval, GlobalPlan, RegionAlternative, ValueRepresentation,
)
from .global_plan_emission import (
    BoundaryMapping, EmittedComponent, GlobalPlanEmission, _component_boundary, dispatch_digest,
    verify_global_plan_emission,
)
from .outline import OutlineResult


def plan_dispatch_fusion(
        program: DispatchProgram, groups: Sequence[Sequence[int]], *, placement: str,
        representation: Callable[[str], ValueRepresentation]) -> GlobalPlan:
    """Make an exact whole-graph cover from proposed contiguous fusion groups.

    Unselected nodes remain explicit singleton regions. The caller supplies legal representations
    and placement; no hardware geometry, timing, or permission to change encodings is inferred.
    Costs remain UNKNOWN until mechanism-equivalent probes establish them.
    """
    grouped: dict[int, tuple[int, ...]] = {}
    covered: set[int] = set()
    for raw in groups:
        group = tuple(raw)
        if (len(group) < 2 or any(isinstance(i, bool) or not isinstance(i, int) for i in group)
                or group != tuple(range(group[0], group[-1] + 1))
                or group[0] < 0 or group[-1] >= len(program.nodes)):
            raise ValueError("fusion groups must contain at least two consecutive graph nodes")
        if covered.intersection(group):
            raise ValueError("fusion groups overlap")
        covered.update(group)
        grouped[group[0]] = group
    alternatives = []
    unknown = CycleInterval.unknown("structural fusion has no calibrated timing claim")
    index = 0
    order = {name: i for i, name in enumerate(program.buffers)}
    while index < len(program.nodes):
        group = grouped.get(index, (index,))
        inputs, outputs = _component_boundary(program, group)
        symbol = (f"{program.entry}$kernel_fused_{index}" if len(group) > 1 else
                  program.nodes[index].op)
        alternatives.append(RegionAlternative(
            id=f"region_{index}", nodes=group, implementation=symbol, placement=placement,
            cycles=unknown,
            inputs=tuple(BufferRepresentation(b, representation(b))
                         for b in sorted(inputs, key=order.__getitem__)),
            outputs=tuple(BufferRepresentation(b, representation(b))
                          for b in sorted(outputs, key=order.__getitem__))))
        index = group[-1] + 1
    return GlobalPlan(tuple(alternatives), (), unknown,
                      notes=("structural fusion; timing and physical movement remain unmeasured",))


def _inline_operations(operations, mapping, destination, functions, stack=()):
    """Clone an SSA region, replacing only defined single-block function calls by their bodies."""
    for op in operations:
        if op.name == "func.return":
            return [mapping[value] for value in op.operands]
        if op.name == "func.call":
            symbol = op.callee.string_value()
            if symbol in stack or symbol not in functions:
                raise ValueError(f"cannot prove expansion of recursive or undefined call {symbol!r}")
            function = functions[symbol]
            if len(function.body.blocks) != 1:
                raise ValueError("structural fusion requires a single-block outlined function")
            block = function.body.blocks[0]
            inner = dict(zip(block.args, (mapping[value] for value in op.operands), strict=True))
            results = _inline_operations(block.ops, inner, destination, functions, (*stack, symbol))
            if results is None:
                raise ValueError("outlined function has no return")
            mapping.update(zip(op.results, results, strict=True))
        else:
            if any(sub.name == "func.call" for region in op.regions for sub in region.walk()):
                raise ValueError("nested calls require a region-aware fusion adapter")
            cloned = op.clone(value_mapper=mapping)
            destination.add_op(cloned)
            mapping.update(zip(op.results, cloned.results, strict=True))
    return None


def _expanded_driver(module, entry):
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region

    functions = {op.sym_name.data: op for op in module.body.block.ops if op.name == "func.func"}
    driver = functions[entry]
    if len(driver.body.blocks) != 1:
        raise ValueError("structural fusion requires a single-block model driver")
    source = driver.body.blocks[0]
    block = Block(arg_types=[arg.type for arg in source.args])
    mapping = dict(zip(source.args, block.args, strict=True))
    results = _inline_operations(source.ops, mapping, block, functions, (entry,))
    if results is None:
        raise ValueError("model driver has no return")
    block.add_op(ReturnOp(*results))
    return FuncOp(entry, driver.function_type, Region([block]))


class OutlinedGlobalPlanEmitter:
    """Concrete shared emitter for contiguous regions of a whole outlined model.

    ``module`` is populated only after SSA, component wiring, and inlined IR equivalence pass.
    Target codegen consumes it with its usual linalg pipeline. This provides a compiler emission
    seam for global fusion while keeping instruction selection in the target compiler.
    """

    def __init__(self, outlined: OutlineResult):
        self.outlined = outlined
        self.module = None
        self.proof = None

    def emit_global_plan(self, program: DispatchProgram, plan: GlobalPlan) -> GlobalPlanEmission:
        from xdsl.dialects.builtin import ModuleOp, StringAttr
        from xdsl.dialects.func import CallOp, FuncOp, ReturnOp
        from xdsl.ir import Block, Region

        self.module = self.proof = None
        actual = build_dispatch_program(self.outlined, entry=program.entry)
        if dispatch_digest(actual) != dispatch_digest(program):
            raise ValueError("fusion plan must bind the exact unpruned outlined model graph")
        if plan.transitions:
            raise ValueError("representation transitions require a target conversion emitter")
        for item in plan.selected:
            for binding in (*item.inputs, *item.outputs):
                spec = program.buffers.get(binding.buffer)
                expected = (ValueRepresentation("tensor_ssa", "logical", spec.dtype)
                            if spec is not None else None)
                if binding.representation != expected:
                    raise ValueError(
                        "outlined fusion preserves logical tensor SSA; physical encodings require "
                        "a target representation emitter")
        functions = {op.sym_name.data: op for op in self.outlined.module.body.block.ops
                     if op.name == "func.func"}
        original = functions[program.entry]
        source = original.body.blocks[0]
        source_ops = [op for op in source.ops if op.name != "func.return"]
        if len(source_ops) != len(program.nodes):
            raise ValueError("model graph does not account for every driver operation")
        values = {buffer.id: source.args[buffer.arg_index]
                  for buffer in program.buffers.values() if buffer.kind == "arg"}
        for node, op in zip(program.nodes, source_ops, strict=True):
            values.update(zip(node.outputs, op.results, strict=True))
        block = Block(arg_types=[arg.type for arg in source.args])
        driver_map = dict(zip(source.args, block.args, strict=True))
        nodes, receipts, fused_functions = [], [], []
        selected = sorted(plan.selected, key=lambda item: item.nodes)
        flat = [index for item in selected for index in item.nodes]
        if flat != list(range(len(program.nodes))):
            raise ValueError("outlined fusion requires an ordered contiguous exact cover")
        for item in selected:
            indices = item.nodes
            inputs = [value.buffer for value in item.inputs]
            outputs = [value.buffer for value in item.outputs]
            if len(indices) == 1:
                node, op = program.nodes[indices[0]], source_ops[indices[0]]
                if item.implementation != node.op:
                    raise ValueError("singleton implementation differs from its outlined operation")
                cloned = op.clone(value_mapper=driver_map)
                block.add_op(cloned)
                driver_map.update(zip(op.results, cloned.results, strict=True))
                nodes.append(replace(node, inputs=list(node.inputs), outputs=list(node.outputs)))
            else:
                if item.implementation in functions:
                    raise ValueError("fused symbol collides with an existing function")
                body = Block(arg_types=[values[name].type for name in inputs])
                mapping = dict(zip((values[name] for name in inputs), body.args, strict=True))
                _inline_operations([source_ops[index] for index in indices], mapping, body, functions)
                body.add_op(ReturnOp(*(mapping[values[name]] for name in outputs)))
                fused = FuncOp(item.implementation,
                               ([values[name].type for name in inputs],
                                [values[name].type for name in outputs]), Region([body]))
                fused.sym_visibility = StringAttr("private")
                fused_functions.append(fused)
                functions[item.implementation] = fused
                call = CallOp(item.implementation,
                              [driver_map[values[name]] for name in inputs],
                              [values[name].type for name in outputs])
                block.add_op(call)
                driver_map.update(zip((values[name] for name in outputs), call.results, strict=True))
                nodes.append(Node("dispatch", item.implementation, inputs, outputs, captures=[]))
            receipts.append(EmittedComponent(
                item.id, (len(nodes) - 1,), tuple((name, name) for name in inputs),
                tuple((name, name) for name in outputs)))
        block.add_op(ReturnOp(*(driver_map[values[name]] for name in program.results)))
        driver = FuncOp(program.entry, original.function_type, Region([block]))
        driver.attributes.update(original.attributes)
        module = ModuleOp([driver, *[op.clone() for op in self.outlined.module.body.block.ops
                                    if op is not original], *fused_functions])
        module.verify()
        if not _expanded_driver(self.outlined.module, program.entry).is_structurally_equivalent(
                _expanded_driver(module, program.entry)):
            raise ValueError("global fusion changed the expanded model computation")
        live = set(program.results)
        live.update(name for node in nodes for name in (*node.inputs, *node.outputs))
        live.update(name for name, spec in program.buffers.items() if spec.kind == "arg")
        dispatch = DispatchProgram(program.entry, list(program.args),
                                   {name: replace(spec, shape=list(spec.shape))
                                    for name, spec in program.buffers.items() if name in live},
                                   nodes, list(program.results))
        emission = GlobalPlanEmission(
            dispatch, plan.digest, dispatch_digest(program), tuple(receipts), (),
            tuple(BoundaryMapping("input", name, name) for name, spec in program.buffers.items()
                  if spec.kind == "arg") +
            tuple(BoundaryMapping("output", name, name) for name in program.results),
            ("exact single-block call inlining and SSA cloning",))
        errors = verify_global_plan_emission(program, plan, emission)
        if errors:
            raise ValueError("invalid outlined global fusion: " + "; ".join(errors))
        self.module = module
        self.proof = {
            "schema": "outlined_global_fusion_proof_v1",
            "logical_dispatch_digest": dispatch_digest(program),
            "emitted_dispatch_digest": dispatch_digest(dispatch),
            "plan_digest": plan.digest,
            "original_module_sha256": hashlib.sha256(str(self.outlined.module).encode()).hexdigest(),
            "emitted_module_sha256": hashlib.sha256(str(module).encode()).hexdigest(),
            "computation": "expanded_driver_structurally_equivalent",
            "logical_nodes": len(program.nodes), "emitted_nodes": len(nodes),
            "logical_dispatches": program.n_dispatches,
            "emitted_dispatches": dispatch.n_dispatches,
            "timing": "UNKNOWN", "physical_movement": "UNKNOWN",
            "full_model_simulated": False,
        }
        return emission
