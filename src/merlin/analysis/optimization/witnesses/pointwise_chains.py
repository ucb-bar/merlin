"""Host-owned extraction and independent evaluation of a small actual source chain.

This is a numerical mechanism witness, not a timing capsule or whole-layer surrogate.
Scalar regions and indexing maps are cloned from source; only statically related tensor
extents are reduced. Unsupported operations abstain instead of borrowing candidate semantics.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from itertools import product
from typing import Any, Sequence

# ---- host_source_witness -----------------------------------------------------------------------


def extract_pointwise_chain(
    source_text: str,
    source_indices: Sequence[int],
    *,
    max_extent: int = 3,
    max_operations: int = 8,
    mechanism: str = "chain",
) -> tuple[str, dict[str, Any]]:
    """Extract the largest single-use equal-domain chain in a host-selected changed task.

    Reuses the existing section extractors' SSA cloning/boundary convention. Input constants
    remain actual source constants; boundary tensors become explicit function arguments.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter

    from merlin.analysis.optimization.witnesses.gathers import _info
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if not 1 <= max_extent <= 8 or not 2 <= max_operations <= 16:
        raise ValueError("source witness must use bounded small extents and chain length")
    if mechanism not in {"chain", "fanout", "quantized_epilogue"}:
        raise ValueError("unsupported source witness mechanism")

    def info(op):
        return _info(op, allow_quantized_epilogue=mechanism == "quantized_epilogue")

    module = parse_mlir_text(source_text)
    functions = [o for o in module.body.block.ops if o.name == "func.func" and len(o.body.blocks)]
    if len(functions) != 1:
        raise ValueError("source witness requires one unoutlined source function")
    ops = [o for o in functions[0].body.block.ops if o.name != "func.return"]
    positions = {op: i for i, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(not isinstance(i, int) or not 0 <= i < len(ops) for i in permitted):
        raise ValueError("source operation indices are outside the source function")
    pairs = {}
    for index in permitted:
        op = ops[index]
        current_info = info(op)
        uses = list(op.results[0].uses) if current_info else []
        if not current_info or len(uses) != 1:
            continue
        use = uses[0]
        consumer = use.operation
        cindex = positions.get(consumer)
        following = info(consumer)
        if cindex not in permitted or not following or following[0] != current_info[0] or use.index >= following[2]:
            continue
        mapping = following[1][use.index]
        if (
            len(mapping.results) == len(current_info[0])
            and all(isinstance(e, AffineDimExpr) for e in mapping.results)
            and sorted(e.position for e in mapping.results) == list(range(len(current_info[0])))
        ):
            pairs[index] = cindex
    chains = []
    for start in pairs:
        if start in pairs.values():
            continue
        chain = [start]
        while chain[-1] in pairs and len(chain) < max_operations:
            chain.append(pairs[chain[-1]])
        chains.append(chain)
    fanout = None
    if mechanism == "fanout":
        stars = []
        for index in permitted:
            current_info = info(ops[index])
            if current_info is None:
                continue
            uses = list(ops[index].results[0].uses)
            consumers = {positions.get(use.operation) for use in uses}
            if len(consumers) < 2 or not consumers <= permitted or len(consumers) + 1 > max_operations:
                continue
            if any(
                info(use.operation) is None
                or info(use.operation)[0] != current_info[0]
                or use.index >= info(use.operation)[2]
                for use in uses
            ):
                continue
            # Preserve every actual root use, including repeated operands of one consumer.
            # Boundary results keep both branches live; no synthetic consumer is invented.
            stars.append((math.prod(current_info[0]), index, sorted(consumers), uses))
        if not stars:
            raise ValueError("changed host task has no supported complete-use pointwise fanout")
        _, root_index, leaves, uses = max(stars, key=lambda item: item[0])
        chain = sorted([root_index, *leaves])
        result_indices = leaves
        fanout = {
            "root_source_index": root_index,
            "uses": sorted([[positions[use.operation], use.index] for use in uses]),
            "all_source_root_uses_preserved": True,
            "root_dtype": str(ops[root_index].results[0].type.get_element_type()),
        }
    else:
        if not chains:
            raise ValueError("changed host task has no supported single-consumer pointwise chain")
        chain = max(chains, key=lambda c: sum(math.prod(info(ops[i])[0]) for i in c[:-1]))
        result_indices = [chain[-1]]
    selected = {ops[i] for i in chain}
    constant_like = {"arith.constant", "tensor.empty", "tensor.splat", "linalg.fill"}
    keep = set(selected)
    pending = list(selected)
    while pending:
        for nested in pending.pop().walk():
            for value in nested.operands:
                owner = value.owner
                scalar_tensor = (
                    mechanism == "quantized_epilogue"
                    and owner in positions
                    and info(owner) is not None
                    and info(owner)[0] == ()
                )
                if owner in positions and owner not in keep and (owner.name in constant_like or scalar_tensor):
                    keep.add(owner)
                    pending.append(owner)
    kept = [op for op in ops if op in keep]
    produced = {v for op in kept for v in op.results}
    boundary = []
    for op in kept:
        for nested in op.walk():
            for value in nested.operands:
                if value.owner is functions[0].body.block or value.owner in positions:
                    if value not in produced and value not in boundary:
                        boundary.append(value)
    if any(not isinstance(v.type, TensorType) for v in boundary):
        raise ValueError("non-tensor source capture is not a supported native witness ABI")
    shapes = {}
    for index in chain:
        op = ops[index]
        shape, maps, _ = info(op)
        reduced = tuple(min(n, max_extent) for n in shape)
        permutation = [e.position for e in maps[-1].results]
        bounds = [reduced[permutation.index(d)] for d in range(len(shape))]
        shapes[op.results[0]] = reduced
        for value, amap in zip(op.operands, maps):
            needed = tuple(bounds[e.position] if isinstance(e, AffineDimExpr) else e.value + 1 for e in amap.results)
            if any(n <= 0 or n > original for n, original in zip(needed, value.type.get_shape())):
                raise ValueError("source indexing does not admit bounded shape reduction")
            if value in shapes:
                needed = tuple(max(a, b) for a, b in zip(shapes[value], needed))
            shapes[value] = needed
    for op in reversed(kept):
        if op.name == "linalg.fill" and op.results[0] in shapes:
            shapes[op.operands[-1]] = shapes[op.results[0]]
        if mechanism == "quantized_epilogue":
            for value in (*op.operands, *op.results):
                if isinstance(value.type, TensorType) and value.type.get_shape() == ():
                    shapes[value] = ()

    def value_type(value):
        if isinstance(value.type, TensorType):
            if value not in shapes:
                raise ValueError("a source tensor extent has no witness constraint")
            return TensorType(value.type.get_element_type(), shapes[value])
        return value.type

    block = Block(arg_types=[value_type(v) for v in boundary])
    mapping = dict(zip(boundary, block.args))
    for op in kept:
        clone = op.clone(value_mapper=mapping)
        block.add_op(clone)
        for old, new in zip(op.results, list(clone.results)):
            replacement = Rewriter.replace_value_with_new_type(new, value_type(old))
            mapping[old] = replacement
    results = [mapping[ops[index].results[0]] for index in result_indices]
    block.add_op(ReturnOp(*results))
    witness = ModuleOp(
        [
            FuncOp(
                "forward",
                FunctionType.from_lists([v.type for v in block.args], [v.type for v in results]),
                Region([block]),
            )
        ]
    )
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    scalar_stream = io.StringIO()
    printer = Printer(stream=scalar_stream, print_generic_format=True)
    for index in chain:
        for scalar in ops[index].regions[0].blocks[0].ops:
            printer.print_op(scalar)
    outputs = [
        {"shape": list(result.type.get_shape()), "dtype": str(result.type.get_element_type())} for result in results
    ]
    if fanout is not None:
        fanout["probe_shape"] = list(shapes[ops[fanout["root_source_index"]].results[0]])
    return text, {
        "schema": "actual_source_pointwise_witness_v1",
        "source_indices": chain,
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "source_shapes": [list(info(ops[i])[0]) for i in chain],
        "probe_shapes": [list(shapes[ops[i].results[0]]) for i in chain],
        "inputs": [{"shape": list(shapes[v]), "dtype": str(v.type.get_element_type())} for v in boundary],
        "output": outputs[0],
        "outputs": outputs,
        "mechanism": mechanism,
        "fanout": fanout,
        **(
            {
                "input_source_values": [
                    {
                        "source_op_index": positions.get(v.owner),
                        "entry_argument_index": v.index if v.owner is functions[0].body.block else None,
                        "result_index": v.index,
                    }
                    for v in boundary
                ],
                "source_input_shapes": [list(v.type.get_shape()) for v in boundary],
                "auxiliary_source_indices": [positions[op] for op in kept if op not in selected],
                "typed_intermediate_rounding_preserved": True,
            }
            if mechanism == "quantized_epilogue"
            else {}
        ),
        "scope": "same scalar regions and maps at reduced extents; no timing or whole-model correctness claim",
    }


def pointwise_scalar_expression(source_text: str) -> dict[str, Any]:
    """Recover a typed scalar DAG from a reduced, source-cloned pointwise witness.

    No algebraic normalization: every f32 rounding boundary and operand order
    survives. The receipt is not a claim that reduced channel extents imply a
    uniform full-layer scale; callers must retain the extraction's source maps.
    """
    import struct

    from merlin.analysis.optimization.witnesses.gathers import _info, _props
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("scalar DAG requires one source witness function")
    function = functions[0]
    env = {
        arg: {"kind": "input", "index": index, "dtype": str(arg.type.get_element_type())}
        for index, arg in enumerate(function.body.block.args)
    }

    def scalar(op, scope):
        dtype = str(op.results[0].type)
        if op.name == "arith.constant":
            value = _props(op, "value").value.data
            if dtype == "f32":
                return {
                    "kind": "constant",
                    "dtype": dtype,
                    "bits": struct.pack("<f", value).hex(),
                    "value": float(value),
                }
            return {"kind": "constant", "dtype": dtype, "value": int(value)}
        return {
            "kind": "operation",
            "opcode": op.name,
            "dtype": dtype,
            "operands": [scope.get(value, env.get(value)) for value in op.operands],
        }

    for op in function.body.block.ops:
        if op.name == "func.return":
            if len(op.operands) != 1:
                raise ValueError("scalar DAG requires one output")
            expression = env[op.operands[0]]

            def complete(node):
                return node is not None and all(complete(child) for child in node.get("operands", []))

            if not complete(expression):
                raise ValueError("scalar DAG has an unresolved source operand")
            return expression
        if op.name == "arith.constant":
            env[op.results[0]] = scalar(op, {})
        elif op.name in {"tensor.splat", "linalg.fill"}:
            env[op.results[0]] = env[op.operands[0]]
        elif op.name == "tensor.empty":
            env[op.results[0]] = None
        elif op.name == "linalg.generic":
            info = _info(op, allow_quantized_epilogue=True)
            if info is None:
                raise ValueError("scalar DAG requires supported all-parallel source regions")
            shape, maps, count = info
            # Scalar-DAG comparison intentionally requires a single reduced point.
            if any(extent != 1 for extent in shape):
                raise ValueError("scalar DAG comparison requires one reduced output point")
            local = {arg: env[value] for arg, value in zip(op.regions[0].blocks[0].args[:count], op.operands[:count])}
            for operation in op.regions[0].blocks[0].ops:
                if operation.name == "linalg.yield":
                    env[op.results[0]] = local.get(operation.operands[0], env.get(operation.operands[0]))
                else:
                    local[operation.results[0]] = scalar(operation, local)
        else:
            raise ValueError(f"unsupported scalar DAG source operation {op.name}")
    raise ValueError("scalar DAG source has no return")


def assess_narrow_readout_equivalence(source_text: str, extraction: dict, *, capability: dict) -> dict:
    """Compare source scalar DAG against a target-edge supplied narrow-readout contract.

    Admission is exact structure, not fitted numeric samples or guessed bias/scale
    conversion. A refusal still enables exact host fusion of the source stages.
    This is a semantic selection aid, not runtime/cycle promotion authority.
    """
    import struct

    if hashlib.sha256(source_text.encode()).hexdigest() != extraction.get("probe_source_sha256"):
        raise ValueError("stale epilogue source binding")
    required = {"schema", "accumulator_dtype", "output_dtype", "scale_dtype", "clamp_min", "clamp_max", "provenance"}
    if (
        not required <= capability.keys()
        or capability["schema"] != "scalar_narrow_readout_contract_v1"
        or not capability["provenance"]
    ):
        raise ValueError("missing target-owned narrow readout contract")
    expression = pointwise_scalar_expression(source_text)

    def op(name, dtype, *operands):
        return {"kind": "operation", "opcode": name, "dtype": dtype, "operands": list(operands)}

    def const(value):
        return {
            "kind": "constant",
            "dtype": capability["scale_dtype"],
            "bits": struct.pack("<f", value).hex(),
            "value": float(value),
        }

    nodes = []

    def walk(node):
        nodes.append(node)
        for child in node.get("operands", []):
            walk(child)

    walk(expression)
    accumulators = [
        node for node in nodes if node.get("kind") == "input" and node.get("dtype") == capability["accumulator_dtype"]
    ]
    multiplications = [node for node in nodes if node.get("opcode") == "arith.mulf"]
    reasons = []
    expected = None
    if len(accumulators) == 1 and len(multiplications) == 1:
        multiplication = multiplications[0]
        cast = op("arith.sitofp", capability["scale_dtype"], accumulators[0])
        if multiplication["operands"][0] == cast:
            scale = multiplication["operands"][1]
            if scale.get("kind") not in {"constant", "input"} or scale.get("dtype") != capability["scale_dtype"]:
                reasons.append("scale_is_not_one_typed_scalar_operand")
            else:
                expected = op(
                    "arith.fptosi",
                    capability["output_dtype"],
                    op(
                        "arith.minimumf",
                        capability["scale_dtype"],
                        op(
                            "arith.maximumf",
                            capability["scale_dtype"],
                            op("math.roundeven", capability["scale_dtype"], multiplication),
                            const(capability["clamp_min"]),
                        ),
                        const(capability["clamp_max"]),
                    ),
                )
                if scale.get("kind") == "input":
                    original_shape = extraction.get("source_input_shapes", [])[scale["index"]]
                    if math.prod(original_shape) != 1:
                        reasons.append("source_scale_is_not_uniform_per_command_requires_channel_partition")
        else:
            reasons.append("accumulator_conversion_and_scale_dataflow_do_not_match")
    else:
        if len(accumulators) != 1:
            reasons.append("not_one_integer_accumulator_input")
        if len(multiplications) != 1:
            reasons.append("multiple_rounded_float_scalings_cannot_be_reassociated")
    if any(node.get("opcode") == "arith.addf" for node in nodes):
        reasons.append("floating_add_bias_or_zero_point_has_no_integer_bias_equivalence_proof")
    if expected != expression:
        reasons.append("typed_scalar_DAG_differs_from_native_scale_round_clamp")
    return {
        "status": "eligible_exact_scalar_DAG" if not reasons else "refused_exact_native_readout",
        "reasons": reasons,
        "source_sha256": extraction["source_sha256"],
        "probe_source_sha256": extraction["probe_source_sha256"],
        "source_indices": extraction["source_indices"],
        "source_expression": expression,
        "target_contract": capability,
        "exact_host_fusion_preserving_stages": "permitted_in_principle_requires_emission_and_numerical_witness",
        "full_model_numerics": "UNPROVEN",
        "performance_promotion": False,
    }


def extract_generic_reduction(
    source_text: str, source_indices: Sequence[int], *, max_extent: int = 3
) -> tuple[str, dict[str, Any]]:
    """Clone one direct ``linalg.generic`` reduction at bounded actual-source extents.

    The affine maps, iterator order, initializer, and typed scalar body are retained. This is
    the source shape emitted for depthwise contractions, where there is no separate pointwise
    producer for :func:`extract_pointwise_reduction` to select.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter

    from merlin.analysis.optimization.witnesses.gathers import _generic_reduction_info
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if not 1 <= max_extent <= 8:
        raise ValueError("generic reduction witness extent must lie in [1,8]")
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("generic reduction witness requires one unoutlined source function")
    function = functions[0]
    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    positions = {op: index for index, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(type(index) is not int or not 0 <= index < len(ops) for index in permitted):
        raise ValueError("generic reduction source indices outside source function")
    options = []
    for index in sorted(permitted):
        info = _generic_reduction_info(ops[index])
        if info is not None:
            output_elements = math.prod(ops[index].results[0].type.get_shape())
            reduction_steps = math.prod(info["iteration_shape"][dim] for dim in info["reduction_dimensions"])
            options.append((output_elements * reduction_steps, index, info))
    if not options:
        raise ValueError("no supported direct generic reduction in changed source task")
    _, index, info = max(options)
    reduction = ops[index]
    reduced_bounds = tuple(min(extent, max_extent) for extent in info["iteration_shape"])
    return _clone_bounded_reduction(source_text, function, reduction, info, reduced_bounds, entry="forward")


def _clone_bounded_reduction(source_text, function, reduction, info, reduced_bounds, *, entry):
    """Shared source cloning for a prevalidated, explicitly bounded reduction.

    Callers own domain/semantic admission. This helper preserves the operation,
    scalar body, initializer producers, and boundary identity without evaluating it.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter

    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    positions = {op: index for index, op in enumerate(ops)}
    index = positions[reduction]

    shapes = {}
    for value, amap in zip(reduction.operands, info["maps"]):
        needed = tuple(
            reduced_bounds[expr.position] if isinstance(expr, AffineDimExpr) else expr.value + 1
            for expr in amap.results
        )
        original = tuple(value.type.get_shape())
        if len(needed) != len(original) or any(size <= 0 or size > old for size, old in zip(needed, original)):
            raise ValueError("generic reduction map does not admit reduced extents")
        if value in shapes and shapes[value] != needed:
            raise ValueError("aliased generic reduction operands need inconsistent witness shapes")
        shapes[value] = needed
    shapes[reduction.results[0]] = shapes[reduction.operands[-1]]

    # Retain actual constant initializers. A function argument remains a boundary input. An
    # undefined tensor.empty accumulator is deliberately unsupported because it has no stable
    # independent numerical meaning.
    keep = {reduction}
    pending = [reduction]
    constant_like = {"arith.constant", "tensor.splat", "linalg.fill"}
    while pending:
        for nested in pending.pop().walk():
            for value in nested.operands:
                owner = value.owner
                if owner in positions and owner not in keep and owner.name in constant_like:
                    keep.add(owner)
                    pending.append(owner)
    if reduction.operands[-1].owner in positions and reduction.operands[-1].owner not in keep:
        raise ValueError("generic reduction initializer is not a defined boundary or constant fill")
    kept = [op for op in ops if op in keep]
    for op in reversed(kept):
        if op.name in {"tensor.splat", "linalg.fill"} and op.results[0] in shapes:
            shapes[op.operands[-1]] = shapes[op.results[0]]
    produced = {value for op in kept for value in op.results}
    boundary = []
    for op in kept:
        for nested in op.walk():
            for value in nested.operands:
                if (
                    (value.owner is function.body.block or value.owner in positions)
                    and value not in produced
                    and value not in boundary
                ):
                    boundary.append(value)
    if any(not isinstance(value.type, TensorType) for value in boundary):
        raise ValueError("non-tensor generic reduction boundary is unsupported")

    def value_type(value):
        if not isinstance(value.type, TensorType):
            return value.type
        if value not in shapes:
            raise ValueError("generic reduction source tensor has unconstrained witness shape")
        return TensorType(value.type.get_element_type(), shapes[value])

    block = Block(arg_types=[value_type(value) for value in boundary])
    mapping = dict(zip(boundary, block.args))
    for op in kept:
        cloned = op.clone(value_mapper=mapping)
        block.add_op(cloned)
        for old, new in zip(op.results, list(cloned.results)):
            mapping[old] = Rewriter.replace_value_with_new_type(new, value_type(old))
    result = mapping[reduction.results[0]]
    block.add_op(ReturnOp(result))
    witness = ModuleOp(
        [FuncOp(entry, FunctionType.from_lists([value.type for value in block.args], [result.type]), Region([block]))]
    )
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    scalar_stream = io.StringIO()
    printer = Printer(stream=scalar_stream, print_generic_format=True)
    for scalar_op in reduction.regions[0].blocks[0].ops:
        printer.print_op(scalar_op)
    dtype = str(result.type.get_element_type())
    width = int(dtype[1:])
    output = {"shape": list(result.type.get_shape()), "dtype": dtype}
    return text, {
        "schema": "actual_source_generic_reduction_witness_v1",
        "mechanism": "generic_reduction",
        "source_indices": [index],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "source_iteration_shape": list(info["iteration_shape"]),
        "probe_iteration_shape": list(reduced_bounds),
        "parallel_dimensions": info["parallel_dimensions"],
        "reduction_dimensions": info["reduction_dimensions"],
        "inputs": [
            {"shape": list(value.type.get_shape()), "dtype": str(value.type.get_element_type())} for value in block.args
        ],
        "outputs": [output],
        "output": output,
        "input_source_values": [
            {"kind": "entry_argument", "index": list(function.body.block.args).index(value)}
            if value.owner is function.body.block
            else {
                "kind": "operation_result",
                "source_op_index": positions[value.owner],
                "result_index": list(value.owner.results).index(value),
            }
            for value in boundary
        ],
        "probe_output_payload_bytes": math.prod(result.type.get_shape()) * ((width + 7) // 8),
        "probe_reduction_steps_per_output": math.prod(reduced_bounds[dim] for dim in info["reduction_dimensions"]),
        "scope": "one cloned generic reduction; actual affine maps/initializer/scalar body and "
        "lexicographic iterator order at reduced extents",
    }


def extract_named_reduction(
    source_text: str, source_indices: Sequence[int], *, max_extent: int = 3
) -> tuple[str, dict[str, Any]]:
    """Clone one direct ``linalg.reduce`` at bounded actual-source extents.

    The source reduction dimensions, initializer, typed scalar body, and lexicographic input
    traversal are retained. Unsupported bodies and dimension orders abstain so this witness
    exercises exactly the register-resident named-reduction lowering, rather than a synthetic
    approximation of it.
    """
    from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.ir import Block, Region
    from xdsl.printer import Printer
    from xdsl.rewriter import Rewriter

    from merlin.analysis.optimization.witnesses.gathers import _props
    from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

    if not 1 <= max_extent <= 8:
        raise ValueError("named reduction witness extent must lie in [1,8]")
    module = parse_mlir_text(source_text)
    functions = [op for op in module.body.block.ops if op.name == "func.func"]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("named reduction witness requires one unoutlined source function")
    function = functions[0]
    ops = [op for op in function.body.block.ops if op.name != "func.return"]
    positions = {op: index for index, op in enumerate(ops)}
    permitted = set(source_indices)
    if any(type(index) is not int or not 0 <= index < len(ops) for index in permitted):
        raise ValueError("named reduction source indices outside source function")

    supported = {
        "arith.constant",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.maxsi",
        "arith.minsi",
        "arith.maxui",
        "arith.minui",
        "linalg.yield",
    }
    options = []
    for index in sorted(permitted):
        reduction = ops[index]
        if reduction.name != "linalg.reduce" or len(reduction.operands) != 2 or len(reduction.results) != 1:
            continue
        input_value, initial_value = reduction.operands
        values = [input_value, initial_value, reduction.results[0]]
        if any(not isinstance(value.type, TensorType) for value in values):
            continue
        input_shape = tuple(input_value.type.get_shape())
        dimensions = tuple(_props(reduction, "dimensions").get_values())
        output_shape = tuple(extent for axis, extent in enumerate(input_shape) if axis not in dimensions)
        body = reduction.regions[0].blocks[0]
        dtype = str(input_value.type.get_element_type())
        if (
            not input_shape
            or any(extent <= 0 for extent in input_shape)
            or not dimensions
            or dimensions != tuple(sorted(set(dimensions)))
            or any(not 0 <= dim < len(input_shape) for dim in dimensions)
            or tuple(initial_value.type.get_shape()) != output_shape
            or tuple(reduction.results[0].type.get_shape()) != output_shape
            or any(str(value.type.get_element_type()) != dtype for value in values)
            or dtype not in {"f32", "i8", "i16", "i32", "i64"}
            or len(body.args) != 2
            or not list(body.args[-1].uses)
            or any(nested.name not in supported or nested.regions for nested in body.ops)
        ):
            continue
        work = math.prod(input_shape)
        options.append((work, index, dimensions, input_shape))
    if not options:
        raise ValueError("no supported direct named reduction in changed source task")
    _, index, dimensions, input_shape = max(options)
    reduction = ops[index]
    input_value, initial_value = reduction.operands
    reduced_input_shape = tuple(min(extent, max_extent) for extent in input_shape)
    reduced_output_shape = tuple(extent for axis, extent in enumerate(reduced_input_shape) if axis not in dimensions)
    shapes = {
        input_value: reduced_input_shape,
        initial_value: reduced_output_shape,
        reduction.results[0]: reduced_output_shape,
    }

    keep = {reduction}
    pending = [reduction]
    constant_like = {"arith.constant", "tensor.splat", "linalg.fill"}
    while pending:
        for nested in pending.pop().walk():
            for value in nested.operands:
                owner = value.owner
                if owner in positions and owner not in keep and owner.name in constant_like:
                    keep.add(owner)
                    pending.append(owner)
    if initial_value.owner in positions and initial_value.owner not in keep:
        raise ValueError("named reduction initializer is not a defined boundary or constant fill")
    kept = [op for op in ops if op in keep]
    for op in reversed(kept):
        if op.name in {"tensor.splat", "linalg.fill"} and op.results[0] in shapes:
            shapes[op.operands[-1]] = shapes[op.results[0]]
    produced = {value for op in kept for value in op.results}
    boundary = []
    for op in kept:
        for nested in op.walk():
            for value in nested.operands:
                if (
                    (value.owner is function.body.block or value.owner in positions)
                    and value not in produced
                    and value not in boundary
                ):
                    boundary.append(value)
    if any(not isinstance(value.type, TensorType) for value in boundary):
        raise ValueError("non-tensor named reduction boundary is unsupported")

    def value_type(value):
        if not isinstance(value.type, TensorType):
            return value.type
        if value not in shapes:
            raise ValueError("named reduction source tensor has unconstrained witness shape")
        return TensorType(value.type.get_element_type(), shapes[value])

    block = Block(arg_types=[value_type(value) for value in boundary])
    mapping = dict(zip(boundary, block.args))
    for op in kept:
        cloned = op.clone(value_mapper=mapping)
        block.add_op(cloned)
        for old, new in zip(op.results, list(cloned.results)):
            mapping[old] = Rewriter.replace_value_with_new_type(new, value_type(old))
    result = mapping[reduction.results[0]]
    block.add_op(ReturnOp(result))
    witness = ModuleOp(
        [
            FuncOp(
                "forward", FunctionType.from_lists([value.type for value in block.args], [result.type]), Region([block])
            )
        ]
    )
    witness.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(witness)
    text = stream.getvalue()
    scalar_stream = io.StringIO()
    printer = Printer(stream=scalar_stream, print_generic_format=True)
    for scalar_op in reduction.regions[0].blocks[0].ops:
        printer.print_op(scalar_op)
    dtype = str(result.type.get_element_type())
    width = int(dtype[1:])
    output = {"shape": list(result.type.get_shape()), "dtype": dtype}
    return text, {
        "schema": "actual_source_named_reduction_witness_v1",
        "mechanism": "named_reduction",
        "source_indices": [index],
        "source_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "scalar_region_sha256": hashlib.sha256(scalar_stream.getvalue().encode()).hexdigest(),
        "source_input_shape": list(input_shape),
        "probe_input_shape": list(reduced_input_shape),
        "reduction_dimensions": list(dimensions),
        "inputs": [
            {"shape": list(value.type.get_shape()), "dtype": str(value.type.get_element_type())} for value in block.args
        ],
        "outputs": [output],
        "output": output,
        "probe_output_payload_bytes": math.prod(result.type.get_shape()) * ((width + 7) // 8),
        "probe_reduction_steps_per_output": math.prod(reduced_input_shape[dim] for dim in dimensions),
        "scope": "one cloned named reduction; actual dimensions/initializer/scalar body and "
        "lexicographic input traversal at reduced extents",
    }
