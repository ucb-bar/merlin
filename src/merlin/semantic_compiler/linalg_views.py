"""Translate the views source family, preserving numerical boundaries."""

from __future__ import annotations

from math import prod

from xdsl.dialects.builtin import IntegerAttr

from .linalg_recognition import LinalgBridgeError, _transpose_maps, _type
from .linalg_state import TranslationState
from .model import IndexMap, SemanticNode


def translate(state: TranslationState, op, ordinal):
    nodes = state.nodes
    env = state.env
    partial_tensors = state.partial_tensors
    scalars = state.scalars
    i64_scalars = state.i64_scalars
    f32_scalars = state.f32_scalars
    bf16_zeros = state.bf16_zeros
    zero_fills = state.zero_fills
    source = state.source
    append_splat = state.append_splat
    if op.name in {"tensor.expand_shape", "tensor.collapse_shape"}:
        expected_properties = (
            {"reassociation", "static_output_shape"} if op.name == "tensor.expand_shape" else {"reassociation"}
        )
        if (
            len(op.operands) != 1
            or len(op.results) != 1
            or set(op.properties) != expected_properties
            or any(not key.startswith("prov.") for key in op.attributes)
        ):
            raise LinalgBridgeError("tensor reshape has an unsupported signature or attributes")
        source_value = op.operands[0]
        if source_value not in env or env[source_value] is None:
            raise LinalgBridgeError("tensor reshape reads an undefined or uninitialized source")
        source_type, result_type = _type(source_value.type), _type(op.results[0].type)
        if (source_type.dtype, source_type.numerical_policy) != (result_type.dtype, result_type.numerical_policy):
            raise LinalgBridgeError("tensor reshape changes element type or numerical policy")
        groups = tuple(tuple(int(axis.value.data) for axis in group.data) for group in op.reassociation.data)
        expanded = op.name == "tensor.expand_shape"
        partitioned_shape = result_type.shape if expanded else source_type.shape
        grouped_shape = source_type.shape if expanded else result_type.shape
        if (
            len(groups) != len(grouped_shape)
            or any(not group for group in groups)
            or tuple(axis for group in groups for axis in group) != tuple(range(len(partitioned_shape)))
            or any(
                prod(partitioned_shape[axis] for axis in group) != extent
                for group, extent in zip(groups, grouped_shape)
            )
            or prod(source_type.shape) != prod(result_type.shape)
            or (expanded and tuple(op.static_output_shape.get_values()) != result_type.shape)
        ):
            raise LinalgBridgeError("tensor reshape has invalid static reassociation or shape")
        node_id = f"op{ordinal}r0"
        nodes.append(SemanticNode(node_id, "reshape", (env[source_value],), result_type))
        env[op.results[0]] = node_id
        return True

    if op.name == "tensor.concat":
        if (
            not op.operands
            or len(op.results) != 1
            or set(op.properties) != {"dim"}
            or op.regions
            or op.successors
            or any(not key.startswith("prov.") for key in op.attributes)
        ):
            raise LinalgBridgeError("tensor.concat needs one or more tensors and one static axis")
        dim = op.properties["dim"]
        if not isinstance(dim, IntegerAttr) or str(dim.type) != "i64":
            raise LinalgBridgeError("tensor.concat axis must be a typed i64 integer")
        axis = dim.value.data
        result_type = _type(op.results[0].type)
        source_types = tuple(_type(operand.type) for operand in op.operands)
        if (
            not 0 <= axis < len(result_type.shape)
            or any(operand not in env or env[operand] is None for operand in op.operands)
            or any(
                len(source.shape) != len(result_type.shape)
                or (source.dtype, source.numerical_policy) != (result_type.dtype, result_type.numerical_policy)
                or any(
                    source.shape[dim] != result_type.shape[dim] for dim in range(len(result_type.shape)) if dim != axis
                )
                for source in source_types
            )
            or sum(source.shape[axis] for source in source_types) != result_type.shape[axis]
        ):
            raise LinalgBridgeError("tensor.concat has incompatible source shape, type, or axis")
        node_id = f"op{ordinal}r0"
        if len(op.operands) == 1:
            nodes.append(SemanticNode(node_id, "identity", (env[op.operands[0]],), result_type))
            env[op.results[0]] = node_id
            return True
        nodes.append(
            SemanticNode(
                node_id,
                "concat",
                tuple(env[operand] for operand in op.operands),
                result_type,
                attrs=(("axis", axis),),
            )
        )
        env[op.results[0]] = node_id
        return True

    if op.name == "tensor.extract_slice":
        if (
            len(op.operands) != 1
            or len(op.results) != 1
            or set(op.properties)
            != {
                "static_offsets",
                "static_sizes",
                "static_strides",
                "operandSegmentSizes",
            }
            or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 0, 0, 0)
            or op.offsets
            or op.sizes
            or op.strides
            or any(not key.startswith("prov.") for key in op.attributes)
        ):
            raise LinalgBridgeError("tensor.extract_slice needs one source and fully static indices")
        source_value = op.operands[0]
        if source_value not in env:
            raise LinalgBridgeError("tensor.extract_slice reads an undefined or uninitialized source")
        source_type, result_type = _type(source_value.type), _type(op.results[0].type)
        rank = len(source_type.shape)
        offsets = tuple(op.properties["static_offsets"].get_values())
        sizes = tuple(op.properties["static_sizes"].get_values())
        strides = tuple(op.properties["static_strides"].get_values())
        if (
            len(result_type.shape) != rank
            or sizes != result_type.shape
            or (source_type.dtype, source_type.numerical_policy) != (result_type.dtype, result_type.numerical_policy)
            or any(len(values) != rank for values in (offsets, sizes, strides))
        ):
            raise LinalgBridgeError("tensor.extract_slice has a rank-reduced or mismatched result")
        if any(offset < 0 or stride <= 0 for offset, stride in zip(offsets, strides)):
            raise LinalgBridgeError("tensor.extract_slice has an invalid offset or stride")
        if any(
            offset + (size - 1) * stride >= extent
            for offset, size, stride, extent in zip(offsets, sizes, strides, source_type.shape)
        ):
            raise LinalgBridgeError("tensor.extract_slice exceeds source bounds")
        if env[source_value] is None:
            if not tuple(op.results[0].uses):
                # A dead, verified view does not read a tensor element.
                # Keep unknown initialization; never supply a zero value.
                env[op.results[0]] = None
                return True
            raise LinalgBridgeError("tensor.extract_slice reads an undefined or uninitialized source")
        source_map = IndexMap(
            rank,
            tuple(tuple(strides[dim] if axis == dim else 0 for axis in range(rank)) for dim in range(rank)),
            offsets,
        )
        identity = IndexMap(
            rank,
            tuple(tuple(int(axis == dim) for axis in range(rank)) for dim in range(rank)),
            (0,) * rank,
        )
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id,
                "gather",
                (env[source_value],),
                result_type,
                index_maps=(source_map, identity),
            )
        )
        env[op.results[0]] = node_id
        return True

    if op.name == "tensor.insert_slice":
        if (
            len(op.operands) != 2
            or len(op.results) != 1
            or set(op.properties)
            != {
                "static_offsets",
                "static_sizes",
                "static_strides",
                "operandSegmentSizes",
            }
            or tuple(op.properties["operandSegmentSizes"].get_values()) != (1, 1, 0, 0, 0)
            or op.offsets
            or op.sizes
            or op.strides
            or any(not key.startswith("prov.") for key in op.attributes)
        ):
            raise LinalgBridgeError("tensor.insert_slice needs a fully static source and destination")
        source_value, destination = op.source, op.dest
        if source_value not in env or env[source_value] is None:
            raise LinalgBridgeError("tensor.insert_slice reads an undefined source")
        if destination not in env:
            raise LinalgBridgeError("tensor.insert_slice reads an uninitialized destination")
        source_type, destination_type, result_type = (
            _type(value.type) for value in (source_value, destination, op.results[0])
        )
        rank = len(source_type.shape)
        offsets = tuple(op.properties["static_offsets"].get_values())
        sizes = tuple(op.properties["static_sizes"].get_values())
        strides = tuple(op.properties["static_strides"].get_values())
        if (
            len(destination_type.shape) != rank
            or destination_type != result_type
            or (source_type.dtype, source_type.numerical_policy)
            != (destination_type.dtype, destination_type.numerical_policy)
            or sizes != source_type.shape
            or any(len(values) != rank for values in (offsets, sizes, strides))
        ):
            raise LinalgBridgeError("tensor.insert_slice sizes differ from source or result type")
        if any(offset < 0 or stride <= 0 for offset, stride in zip(offsets, strides)):
            raise LinalgBridgeError("tensor.insert_slice has an invalid offset or stride")
        if any(
            offset + (size - 1) * stride >= extent
            for offset, size, stride, extent in zip(
                offsets,
                sizes,
                strides,
                destination_type.shape,
            )
        ):
            raise LinalgBridgeError("tensor.insert_slice exceeds destination bounds")
        if env[destination] is None:
            from .tensor_assembly import insert_partial

            try:
                partial, completed = insert_partial(
                    destination,
                    result_type,
                    env[source_value],
                    source_type,
                    offsets,
                    strides,
                    partial_tensors.get(destination),
                    node_id=f"op{ordinal}r0",
                )
            except LinalgBridgeError as error:
                raise LinalgBridgeError(f"tensor.insert_slice has an uninitialized destination: {error}") from error
            if completed is not None:
                nodes.append(completed)
                env[op.results[0]] = completed.id
            else:
                partial_tensors[op.results[0]] = partial
                env[op.results[0]] = None
            return True
        identity = IndexMap(
            rank,
            tuple(tuple(int(axis == dim) for axis in range(rank)) for dim in range(rank)),
            (0,) * rank,
        )
        placement = IndexMap(
            rank,
            tuple(tuple(strides[dim] if axis == dim else 0 for axis in range(rank)) for dim in range(rank)),
            offsets,
        )
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id,
                "insert_slice",
                (env[source_value], env[destination]),
                result_type,
                index_maps=(identity, placement, placement),
            )
        )
        env[op.results[0]] = node_id
        return True

    if op.name == "linalg.fill":
        if len(op.operands) != 2 or len(op.results) != 1:
            raise LinalgBridgeError("linalg.fill requires one scalar and one destination")
        scalar, destination = op.operands
        if destination not in env:
            raise LinalgBridgeError("linalg.fill requires a declared destination")
        type_ = _type(op.results[0].type)
        if type_ != _type(destination.type):
            raise LinalgBridgeError("linalg.fill result/destination has unsupported type")
        if type_.dtype == "bf16" and scalar in bf16_zeros and env[destination] is None:
            if any(not key.startswith("prov.") for key in op.attributes):
                raise LinalgBridgeError("BF16 zero fill has unknown attributes")
            zero_fills.add(op.results[0])
            env[op.results[0]] = None
            return True
        if type_.dtype == "f32" and scalar in f32_scalars:
            value = f32_scalars[scalar]
            encoding, data = "f32-le-raw", value.to_bytes(4, "little")
        elif type_.dtype == "i64" and scalar in i64_scalars:
            value = i64_scalars[scalar]
            encoding, data = "i64-le", value.to_bytes(8, "little", signed=True)
        elif type_.dtype == "i32" and scalar in scalars:
            value = scalars[scalar]
            encoding, data = "i32-le", value.to_bytes(4, "little", signed=True)
        else:
            raise LinalgBridgeError("linalg.fill requires an admitted scalar and destination")
        env[op.results[0]] = append_splat(f"op{ordinal}r0", type_, value, encoding, data)
        return True

    if op.name == "linalg.copy":
        if len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1 or op.attributes:
            raise LinalgBridgeError("copy has an unsupported signature or attributes")
        source_value, destination = op.inputs[0], op.outputs[0]
        if source_value not in env or env[source_value] is None or destination not in env:
            raise LinalgBridgeError("copy reads an undefined or uninitialized input or destination")
        source_type = _type(source_value.type)
        if source_type != _type(destination.type) or source_type != _type(op.results[0].type):
            raise LinalgBridgeError("copy requires a type-preserving input and result")
        node_id = f"op{ordinal}r0"
        nodes.append(SemanticNode(node_id, "identity", (env[source_value],), source_type))
        env[op.results[0]] = node_id
        return True

    if op.name == "linalg.transpose":
        if len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
            raise LinalgBridgeError("transpose has an unsupported signature")
        source_value, destination = op.inputs[0], op.outputs[0]
        if source_value not in env or env[source_value] is None or destination not in env:
            raise LinalgBridgeError("transpose reads an undefined or uninitialized input or destination")
        # The named transpose writes every result element, so an empty
        # destination contributes storage but no semantic input value.
        source_type = _type(source_value.type)
        destination_type = _type(destination.type)
        result_type = _type(op.results[0].type)
        permutation = tuple(op.permutation.get_values())
        rank = len(source_type.shape)
        if (
            rank < 1
            or len(permutation) != rank
            or sorted(permutation) != list(range(rank))
            or set(op.properties) != {"permutation"}
            or any(not name.startswith("prov.") for name in op.attributes)
            or result_type != destination_type
            or result_type.dtype != source_type.dtype
            or result_type.numerical_policy != source_type.numerical_policy
            or result_type.shape != tuple(source_type.shape[axis] for axis in permutation)
        ):
            raise LinalgBridgeError("transpose permutation, shape, or numerical policy is unsupported")
        try:
            op.verify()
        except Exception as exc:  # noqa: BLE001 - xDSL verifier is a fail-closed boundary
            raise LinalgBridgeError(f"transpose failed xDSL verification: {exc}") from exc
        node_id = f"op{ordinal}r0"
        nodes.append(
            SemanticNode(
                node_id, "transpose", (env[source_value],), result_type, index_maps=_transpose_maps(permutation)
            )
        )
        env[op.results[0]] = node_id
        return True

    return False
