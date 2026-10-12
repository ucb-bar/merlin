"""Explicit logical graph optimization on the ordinary host compilation route.

Contracts select semantics and structural cost proxies, never hardware schedules.
Typed ABI, scalar bodies and the fusion emitter's expanded driver correspondence
are checked independently of pass reports. This supplies no Lean or timing proof.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from math import prod

from xdsl.dialects import builtin, func
from xdsl.ir import Operation
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.traits import get_effects

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.xdsl_dialects._common import text


@dataclass(frozen=True)
class GraphOptimizationContract:
    source_sha256: str
    entry: str
    propagate_layout: bool = False
    independent_element_order_unobserved: bool = False
    floating_environment_unobserved_nontrapping: bool = False
    minimum_transpose_bytes_saved: int = 1
    fusion_groups: tuple[tuple[int, ...], ...] = ()
    minimum_dispatches_saved: int = 1
    maximum_added_operations: int = 8192

    def validate(self, source: str):
        if self.source_sha256 != hashlib.sha256(source.encode()).hexdigest():
            raise ValueError("graph optimization contract does not bind the actual input source")
        if not isinstance(self.entry, str) or not self.entry:
            raise ValueError("graph optimization requires its explicit logical entry")
        for field in (
            "propagate_layout",
            "independent_element_order_unobserved",
            "floating_environment_unobserved_nontrapping",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError("graph permissions must be explicit booleans")
        for field in ("minimum_transpose_bytes_saved", "minimum_dispatches_saved", "maximum_added_operations"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError("graph cost constraints must be explicit positive integers")
        if type(self.fusion_groups) is not tuple or any(type(group) is not tuple for group in self.fusion_groups):
            raise ValueError("fusion groups require immutable node selections")


def _functions(module):
    rows = [op for op in module.body.block.ops if type(op) is func.FuncOp]
    result = {op.sym_name.data: op for op in rows}
    if len(result) != len(rows):
        raise ValueError("graph optimization refuses duplicate function symbols")
    return result


def _signature(module, entry):
    functions = _functions(module)
    if entry not in functions or len(functions[entry].body.blocks) != 1:
        raise ValueError("graph optimization requires its defined single-block entry")
    return functions[entry].function_type


def _layout_bodies(module):
    """Independent registry of actual scalar regions before transformation."""
    return tuple(
        region.clone()
        for op in module.walk()
        if op.name in {"linalg.generic", "linalg.reduce"}
        for region in op.regions
    )


def _transpose_bytes(module):
    from .layout_propagation import _permutation

    total = 0
    for op in module.walk():
        permutation = _permutation(op)
        if permutation is None or permutation == tuple(range(len(permutation))) or not op.results[0].uses:
            continue
        dtype = op.results[0].type
        if not isinstance(dtype, builtin.TensorType) or any(d <= 0 for d in dtype.get_shape()):
            raise ValueError("layout materialization cost requires positive static tensor extents")
        element = dtype.get_element_type()
        bits = element.width.data if isinstance(element, builtin.IntegerType) else getattr(element, "bitwidth", None)
        if type(bits) is not int or bits <= 0 or bits % 8:
            raise ValueError("layout materialization cost has unknown byte-addressable storage")
        total += prod(dtype.get_shape()) * (bits // 8)
    return total


def _check_layout_output(original, modified, bodies):
    before = _functions(original)
    after = _functions(modified)
    if set(before) != set(after) or any(before[name].function_type != after[name].function_type for name in before):
        raise ValueError("layout propagation changed complete function interfaces")
    if any(
        before[name].attributes != after[name].attributes or before[name].properties != after[name].properties
        for name in before
    ):
        raise ValueError("layout propagation changed function declarations or contracts")
    for op in modified.walk():
        if op.name not in {"linalg.generic", "linalg.reduce"}:
            continue
        for region in op.regions:
            if not any(region.is_structurally_equivalent(source) for source in bodies):
                raise ValueError("layout propagation introduced unaccounted scalar arithmetic or evaluation order")
    modified.verify()
    if _layout_expression(original) != _layout_expression(modified):
        raise ValueError("layout propagation changed complete logical output coordinates or scalar expressions")


def _layout_expression(module):
    """Bounded independent coordinate interpretation, without enumerating elements.

    The first selected layout domain is static all-parallel pointwise arithmetic
    and permutations. Other layout rewrite forms retain their standalone APIs
    but require a separate semantics checker before this production route uses them.
    """
    outputs = []
    for name, function in _functions(module).items():
        if len(function.body.blocks) != 1:
            raise ValueError("layout coordinate checking requires defined single-block functions")
        block = function.body.block
        arguments = {value: index for index, value in enumerate(block.args)}
        cache = {}
        visiting = set()
        for operation in block.ops:
            if operation.name != "linalg.generic":
                continue
            if len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1:
                raise ValueError("layout coordinate checker refuses unmodeled scalar regions")
            for scalar_op in operation.regions[0].block.ops:
                if scalar_op.name == "linalg.yield":
                    continue
                if (
                    scalar_op.regions
                    or scalar_op.successors
                    or get_effects(scalar_op) != set()
                    or not scalar_op.name.startswith(("arith.", "math."))
                ):
                    raise ValueError("layout coordinate checker refuses unmodeled scalar effects")

        def element(value, coordinates):
            key = value, coordinates
            if key in cache:
                return cache[key]
            if len(cache) >= 8192:
                raise ValueError("layout coordinate checking exceeded its bounded expression budget")
            if value in visiting or len(visiting) >= 256:
                raise ValueError("layout expression has excessive nesting or a cyclic value")
            visiting.add(value)
            if value in arguments:
                result = ("argument", arguments[value], str(value.type), coordinates)
            else:
                op = value.owner
                if not isinstance(op, Operation):
                    raise ValueError("layout expression has an unknown value owner")
                if op.name == "arith.constant":
                    result = (
                        "constant",
                        str(op.value),
                        coordinates if isinstance(value.type, builtin.TensorType) else (),
                    )
                elif op.name == "linalg.transpose":
                    permutation = tuple(op.permutation.get_values())
                    if sorted(permutation) != list(range(len(coordinates))):
                        raise ValueError("layout expression has an invalid permutation")
                    result = element(
                        op.operands[0], tuple(coordinates[permutation.index(i)] for i in range(len(coordinates)))
                    )
                elif op.name == "linalg.generic":
                    rank = len(coordinates)
                    maps = tuple(item.data for item in op.indexing_maps)
                    if (
                        len(op.results) != 1
                        or len(op.outputs) != 1
                        or len(op.regions) != 1
                        or len(op.body.blocks) != 1
                        or any(item.data.value != "parallel" for item in op.iterator_types)
                        or len(op.iterator_types) != rank
                        or maps[-1] != AffineMap.identity(rank)
                        or len(maps) != len(op.operands)
                        or any(index_map.num_symbols or index_map.num_dims != rank for index_map in maps)
                    ):
                        raise ValueError("layout coordinate checker does not cover this generic/reduction form")
                    body = op.body.block
                    bindings = {
                        arg: (
                            operand,
                            tuple(index_map.replace_dims_and_symbols(list(coordinates), [], rank, 0).results),
                        )
                        for arg, operand, index_map in zip(body.args, op.operands, maps, strict=True)
                    }
                    scalar_cache = {}
                    scalar_visiting = set()

                    def scalar(item):
                        if item in scalar_cache:
                            return scalar_cache[item]
                        if item in scalar_visiting or len(scalar_visiting) >= 256:
                            raise ValueError("layout scalar expression exceeded its bounded nesting budget")
                        scalar_visiting.add(item)
                        if item in bindings:
                            operand, indices = bindings[item]
                            scalar_result = element(operand, indices)
                        else:
                            operation = item.owner
                            if (
                                not isinstance(operation, Operation)
                                or operation.parent is not body
                                or operation.regions
                                or operation.successors
                                or get_effects(operation) != set()
                                or not operation.name.startswith(("arith.", "math."))
                            ):
                                raise ValueError("layout coordinate checker refuses unmodeled scalar effects")
                            scalar_result = (
                                operation.name,
                                tuple(operation.results).index(item),
                                str(item.type),
                                tuple(sorted((key, str(value)) for key, value in operation.attributes.items())),
                                tuple(sorted((key, str(value)) for key, value in operation.properties.items())),
                                tuple(scalar(operand) for operand in operation.operands),
                            )
                        scalar_cache[item] = scalar_result
                        scalar_visiting.remove(item)
                        return scalar_result

                    terminator = body.last_op
                    if terminator is None or terminator.name != "linalg.yield" or len(terminator.operands) != 1:
                        raise ValueError("layout coordinate checker requires a complete scalar yield")
                    result = (
                        "pointwise",
                        tuple(sorted((key, str(value)) for key, value in op.attributes.items())),
                        scalar(terminator.operands[0]),
                    )
                else:
                    raise ValueError("layout coordinate checker has no semantics for this tensor producer")
            cache[key] = result
            visiting.remove(value)
            return result

        returned = block.last_op
        if returned is None or type(returned) is not func.ReturnOp:
            raise ValueError("layout coordinate checker requires complete ordered returns")
        values = []
        for value in returned.operands:
            if not isinstance(value.type, builtin.TensorType) or any(size <= 0 for size in value.type.get_shape()):
                raise ValueError("layout coordinate checker requires positive static logical output tensors")
            values.append(element(value, tuple(AffineExpr.dimension(i) for i in range(len(value.type.get_shape())))))
        effects = []
        for operation in block.ops:
            if operation.name in {
                "arith.constant",
                "tensor.empty",
                "linalg.transpose",
                "linalg.generic",
                "func.return",
            }:
                continue
            if operation.regions or operation.successors:
                raise ValueError("layout coordinate checker refuses opaque region/control semantics")
            operands = []
            for value in operation.operands:
                rank = len(value.type.get_shape()) if isinstance(value.type, builtin.TensorType) else 0
                operands.append(element(value, tuple(AffineExpr.dimension(i) for i in range(rank))))
            effects.append(
                (
                    operation.name,
                    tuple(sorted((key, str(value)) for key, value in operation.attributes.items())),
                    tuple(sorted((key, str(value)) for key, value in operation.properties.items())),
                    tuple(str(result.type) for result in operation.results),
                    tuple(operands),
                )
            )
        outputs.append((name, function.function_type, tuple(values), tuple(effects)))
    return tuple(outputs)


def _check_fusion_growth(outlined, plan, entry, original_count, maximum_added):
    """Count bounded call expansion before the emitter clones fused bodies."""
    functions = _functions(outlined.module)
    driver = functions[entry]
    # The selected plan is built from this entry; its implementation symbols
    # cannot add new functions before emission. Do not infer callees from names.
    budget = original_count + maximum_added
    total = sum(1 for _ in outlined.module.walk()) - sum(1 for _ in driver.walk()) + 2
    source_ops = [op for op in driver.body.block.ops if op.name != "func.return"]

    def expansion(operation, stack=()):
        if len(stack) >= 256:
            raise ValueError("fusion exceeds its bounded call nesting budget")
        if operation.name == "func.call":
            symbol = operation.callee.string_value()
            if symbol in stack or symbol not in functions:
                raise ValueError("fusion refuses recursive or undefined call expansion")
            function = functions[symbol]
            if not function.body.blocks and symbol in outlined.external_symbols:
                return 1
            if len(function.body.blocks) != 1:
                raise ValueError("fusion growth checking requires single-block functions")
            count = 0
            for child in function.body.block.ops:
                if child.name == "func.return":
                    break
                count += expansion(child, (*stack, symbol))
                if count > budget:
                    raise ValueError("fusion exceeds its explicit compilation-space cost contract")
            return count
        if any(child.name == "func.call" for region in operation.regions for child in region.walk()):
            raise ValueError("fusion refuses opaque nested-call expansion")
        return sum(1 for _ in operation.walk())

    expanded_count = 0
    for operation in source_ops:
        expanded_count += expansion(operation)
        if expanded_count > budget:
            raise ValueError("fusion exceeds its explicit compilation-space cost contract")
    for item in plan.selected:
        if len(item.nodes) == 1:
            total += sum(1 for _ in source_ops[item.nodes[0]].walk())
        else:
            total += 3  # Fused definition, return, and one driver call.
            for index in item.nodes:
                total += expansion(source_ops[index])
                if total > budget:
                    raise ValueError("fusion exceeds its explicit compilation-space cost contract")
        if total > budget:
            raise ValueError("fusion exceeds its explicit compilation-space cost contract")


def optimize_host_graph(source: str, contract: GraphOptimizationContract) -> tuple[str, dict]:
    """Select the existing layout/fusion mechanisms before host preprocessing.

    The admitted source-check domain is static pointwise tensors/permutations.
    Fusion concerns logical host tensor SSA only. Physical representations,
    device residency and target conversion emitters require separate OOT APIs.
    Lowering/LLVM conformance and independent numerical execution remain required.
    """
    if type(contract) is not GraphOptimizationContract:
        raise ValueError("graph optimization needs the explicit supported contract")
    contract.validate(source)
    module = parse_mlir_text(source)
    module.verify()
    signature = _signature(module, contract.entry)
    report = {
        "schema": "merlin.host_graph_optimization.v1",
        "contract": asdict(contract),
        "source_sha256": contract.source_sha256,
        "semantics": "conditional_structural_checks",
        "timing": "UNKNOWN",
        "hardware_conformance": "not_established",
    }
    if contract.propagate_layout:
        from .layout_propagation import rewrite_module

        if not contract.independent_element_order_unobserved:
            raise ValueError("layout propagation requires explicit independent-element order permission")
        floating = any(
            isinstance(value.type, builtin.TensorType)
            and not isinstance(value.type.get_element_type(), builtin.IntegerType)
            for op in module.walk()
            for value in (*op.operands, *op.results)
        )
        if floating and not contract.floating_environment_unobserved_nontrapping:
            raise ValueError("floating layout propagation requires its nontrapping/unobserved environment contract")
        original = module.clone()
        _layout_expression(original)
        bodies = _layout_bodies(original)
        before_bytes = _transpose_bytes(original)
        layout = rewrite_module(module, reduction_channel_block=0)
        _check_layout_output(original, module, bodies)
        after_bytes = _transpose_bytes(module)
        if before_bytes - after_bytes < contract.minimum_transpose_bytes_saved:
            raise ValueError("layout rewrite does not satisfy its materialization-byte cost contract")
        report["layout"] = {
            "rewrite": layout.to_dict(),
            "before_transpose_bytes": before_bytes,
            "after_transpose_bytes": after_bytes,
            "cost_kind": "structural_byte_proxy",
        }
    if contract.fusion_groups:
        from merlin.xdsl_dialects.lowering.dispatch_program import build_dispatch_program
        from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation
        from merlin.xdsl_dialects.lowering.global_plan_emission import emit_global_plan
        from merlin.xdsl_dialects.lowering.outline import outline_dispatches
        from merlin.xdsl_dialects.lowering.outlined_plan_emission import (
            OutlinedGlobalPlanEmitter,
            _expanded_driver,
            plan_dispatch_fusion,
        )

        original_count = sum(1 for _ in module.walk())
        outlined = outline_dispatches(module, forward=contract.entry)
        program = build_dispatch_program(outlined, entry=contract.entry)
        plan = plan_dispatch_fusion(
            program,
            contract.fusion_groups,
            placement="host",
            representation=lambda name: ValueRepresentation("tensor_ssa", "logical", program.buffers[name].dtype),
        )
        _check_fusion_growth(outlined, plan, contract.entry, original_count, contract.maximum_added_operations)
        actual_source = _expanded_driver(module, contract.entry)
        _layout_expression(actual_source)
        actual_bodies = _layout_bodies(actual_source)
        emitter = OutlinedGlobalPlanEmitter(outlined)
        emitted = emit_global_plan(program, plan, emitter)
        if emitter.module is None or emitter.proof is None or emitter.emitted_outline is None:
            raise ValueError("fusion emitter produced no checked executable graph")
        _check_layout_output(actual_source, _expanded_driver(emitter.module, contract.entry), actual_bodies)
        module = emitter.module
        if program.n_dispatches - emitted.dispatch.n_dispatches < contract.minimum_dispatches_saved:
            raise ValueError("fusion does not satisfy its dispatch-count cost contract")
        if sum(1 for _ in module.walk()) - original_count > contract.maximum_added_operations:
            raise ValueError("fusion exceeds its explicit compilation-space cost contract")
        report["fusion"] = {
            "proof": emitter.proof,
            "plan_digest": plan.digest,
            "before_dispatches": program.n_dispatches,
            "after_dispatches": emitted.dispatch.n_dispatches,
            "cost_kind": "structural_dispatch_proxy",
            "source_correspondence": "static_pointwise_coordinates_and_scalar_regions",
        }
    if _signature(module, contract.entry) != signature:
        raise ValueError("graph optimization changed the complete public entry ABI")
    output = text(module, generic=True)
    report["transformed_sha256"] = hashlib.sha256(output.encode()).hexdigest()
    return output, report
