"""Conservative parsed-Linalg entry to Merlin's typed semantic kernel graph.

The admitted region uses static integer tensors for contractions, signed i64
indices for checked embedding lookup, opaque f32/BF16/FP8 tensors for
bit-preserving movement including static same-rank slices, signed-i64
casts and reciprocals with explicit f32 rounding, strict f32
projected pointwise arithmetic and explicitly ordered reduction, bounded BF16 ReLU/add forms, and
annotated ordered FP8/BF16 reset and continuation contractions. Every source
operation is accounted for; unsupported bodies and initializers raise. Ordinary
f32 matmul and reductions without a declared order remain source-only nodes
until explicit numerical decisions replace them. Their reduction order is not
inferred from the Linalg operation name.

Closed scalar regions also preserve binary64 arithmetic, explicit casts,
selection and Boolean projections. Math intrinsics keep unresolved source
policies. This source support does not grant target implementations or narrow
binary64. Runtime-sized compaction and general control flow remain unsupported.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import prod

from .linalg_recognition import (
    LinalgBridgeError,
    _embedding_lookup_operands,
    _f32_reduce_maps,
    _i64_to_f32_bits,
    _linear_affine_expression,
    _maps,
    _named_matmul_maps_are_default,
    _oriented_matmul_accumulate_maps,
    _oriented_matmul_maps,
    _recognize_f32_reduce,
    _scalar_f32_bits,
    _scalar_i64,
    _scalar_integer,
    _source_f32_batched_matmul,
    _static_f32_square_maps,
    _static_index_copy_map,
    _transpose_maps,
    _type,
)
from .model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from .reference import TensorValue


@dataclass(frozen=True)
class LinalgTranslation:
    request: KernelRequest
    constants: dict[str, TensorValue]
    source_operations: tuple[str, ...]


def translate_linalg_text(
    source: str,
    *,
    entry: str,
    target_identity: str,
    lowering_policy: str = "strict-native",
    boundary_storage: str = "external",
    argument_constants: Mapping[int, bytes] | None = None,
) -> LinalgTranslation:
    """Translate one admitted function, preserving its init and ordered returns.

    The source is parsed by Merlin's established xDSL frontend. The shared
    generic-matmul recognizer checks maps, iterators, scalar body,
    signed extension, widths, and operand connections before translation.
    """
    from xdsl.dialects.builtin import DictionaryAttr, IntegerAttr, StringAttr
    from xdsl.dialects.linalg.attrs import IteratorType

    from merlin.frontends.linalg_bf16_add import (
        recognize_bf16_f32_chop_add,
        recognize_bf16_f32_chop_sub,
    )
    from merlin.frontends.linalg_bf16_mul import recognize_bf16_product_rne
    from merlin.frontends.linalg_bf16_relu import recognize_bf16_sign_clamp_relu
    from merlin.frontends.linalg_composite_math import recognize_static_composite_math_body
    from merlin.frontends.linalg_fp8_cast import recognize_bf16_fp8_truncation
    from merlin.frontends.linalg_math_patterns import recognize_static_f32_exp_rsqrt_body
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.frontends.linalg_ordered_fp8 import (
        is_positive_bf16_zero,
        recognize_ordered_fp8_bf16_reset_matmul,
    )
    from merlin.frontends.linalg_patterns import (
        InvalidLinalgPattern,
        recognize_f32_relu_ordered_select,
        recognize_signed_i8_i32_matmul,
        recognize_static_f32_division,
        recognize_static_projected_pointwise,
    )
    from merlin.frontends.linalg_scalar_support import (
        StaticIotaResourceLimit,
        recognize_f32_nonzero,
        recognize_static_iota,
    )

    module = parse_mlir_text(source)
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1:
        raise LinalgBridgeError(f"expected exactly one entry function @{entry}")
    function = functions[0]
    if len(function.body.blocks) != 1:
        raise LinalgBridgeError("entry function must have one block")
    block = function.body.block
    argument_constants = {} if argument_constants is None else dict(argument_constants)
    if any(
        type(index) is not int or not 0 <= index < len(block.args) or not isinstance(data, bytes)
        for index, data in argument_constants.items()
    ):
        raise LinalgBridgeError("lifted argument constants have invalid indices or bytes")
    operations = tuple(block.ops)
    if not operations or operations[-1].name != "func.return":
        raise LinalgBridgeError("entry function must end in func.return")

    nodes: list[SemanticNode] = []
    env: dict[object, str | None] = {}
    partial_tensors = {}
    scalars: dict[object, int] = {}
    i64_scalars: dict[object, int] = {}
    f32_scalars: dict[object, int] = {}
    f64_scalars: dict[object, int] = {}
    boolean_scalars: dict[object, int] = {}
    bf16_zeros: set[object] = set()
    zero_fills: set[object] = set()
    constants: dict[str, TensorValue] = {}
    constant_bindings: list[ConstantBinding] = []
    boundaries: list[tuple[str, str]] = []
    source_operations: list[str] = []
    argument_attrs = function.arg_attrs.data if function.arg_attrs is not None else (None,) * len(block.args)
    if len(argument_attrs) != len(block.args):
        raise LinalgBridgeError("entry argument attributes do not match the function signature")
    for index, (arg, attributes) in enumerate(zip(block.args, argument_attrs)):
        type_ = _type(arg.type)
        node_id = f"arg{index}"
        domain_attrs: tuple[tuple[str, str], ...] = ()
        if attributes is not None:
            if not isinstance(attributes, DictionaryAttr) or any(
                key != "merlin.input_domain" and not key.startswith("prov.") for key in attributes.data
            ):
                raise LinalgBridgeError("entry argument has unknown attributes")
            domain = attributes.data.get("merlin.input_domain")
            if domain is not None:
                if not isinstance(domain, StringAttr) or not domain.data:
                    raise LinalgBridgeError("entry input domain must be a nonempty string")
                domain_attrs = (("input_domain", domain.data),)
        data = argument_constants.get(index)
        if data is None:
            nodes.append(SemanticNode(node_id, "input", (), type_, attrs=domain_attrs, effect="input"))
            boundaries.append((node_id, boundary_storage))
        else:
            from .linalg_constants import read_tensor_bytes

            encoding, value = read_tensor_bytes(type_, data)
            nodes.append(SemanticNode(node_id, "constant", (), type_, attrs=domain_attrs, effect="constant"))
            constants[node_id] = value
            constant_bindings.append(ConstantBinding(node_id, encoding, data.hex()))
        env[arg] = node_id

    splat_literals = {}

    def append_splat(node_id, type_, value, encoding, element):
        # Shape, policy and exact bits all participate in constant identity.
        # Share compiler-owned literal bytes; input ports retain their identity.
        key = type_, encoding, element
        if key not in splat_literals:
            attrs = (("value", value),) if type_.dtype in {"i32", "i64"} else ()
            nodes.append(SemanticNode(node_id, "constant", (), type_, attrs=attrs, effect="constant"))
            constants[node_id] = TensorValue(type_, (value,) * prod(type_.shape))
            constant_bindings.append(ConstantBinding(node_id, encoding, (element * prod(type_.shape)).hex()))
            splat_literals[key] = node_id
        return splat_literals[key]

    def append_scalar_region(op, ordinal):
        from .scalar_region import translate_scalar_region

        sources = tuple(env.get(value) for value in op.inputs)
        if any(key is None for key in sources) or env.get(op.outputs[0], "undefined") is not None:
            raise LinalgBridgeError("scalar region needs defined inputs and an empty output")
        translated = translate_scalar_region(op, sources, prefix=f"op{ordinal}r0")
        nodes.extend(translated.nodes)
        constant_bindings.extend(translated.constants)
        constants.update(translated.values)
        env[op.results[0]] = translated.output

    from .indexed_update_source import recognize_indexed_update

    indexed_updates, indexed_prelude = {}, set()
    ordered_scans, scan_prelude = {}, set()
    for operation in operations:
        if operation.name == "scf.for":
            from .ordered_scan_source import recognize_ordered_scan

            scan = recognize_ordered_scan(operation)
            ordered_scans[operation] = scan
            scan_prelude.update(scan.prelude)
        if operation.name == "tensor.insert_slice" and operation.offsets:
            try:
                indexed = recognize_indexed_update(operation)
            except ValueError as error:
                raise LinalgBridgeError(str(error)) from error
            indexed_updates[operation] = indexed
            indexed_prelude.update(indexed.prelude)
    for ordinal, op in enumerate(operations):
        source_operations.append(op.name)
        if op.name == "linalg.generic" and len(op.results) == len(op.outputs) == 2:
            from merlin.frontends.linalg_extremum_patterns import recognize_static_i64_argmin

            if len(op.inputs) != 1:
                raise LinalgBridgeError("two-result source reduction needs one admitted input")
            source_type = _type(op.inputs[0].type)
            index_bits = max(2, max(source_type.shape).bit_length() + 1)
            try:
                pattern = recognize_static_i64_argmin(op, index_bits=index_bits)
            except InvalidLinalgPattern as error:
                raise LinalgBridgeError(str(error)) from error
            source_id = env.get(op.inputs[0])
            if source_id is None:
                raise LinalgBridgeError("minimum/first-index reduction reads an uninitialized input")
            # These are pure projections of the same mathematical reduction.
            # A physical multi-result instruction uses the existing descriptor
            # bundle mechanism; separate realizations must count their work.
            for result_index, (value, operation) in enumerate(
                zip(op.results, ("i64_min_value", "i64_first_min_index"))
            ):
                node = SemanticNode(
                    f"op{ordinal}r{result_index}",
                    operation,
                    (source_id,),
                    _type(value.type),
                    attrs=(
                        ("axis", pattern.axis),
                        ("minimum_index_bits", index_bits),
                        ("numeric_semantics", "signed-i64-min-first-index-v1"),
                    ),
                )
                nodes.append(node)
                env[value] = node.id
            continue
        if op.name == "linalg.generic" and len(op.results) == 1 and len(op.outputs) == 1:
            from .constant_indexing import translate_constant_index_read

            indexed = translate_constant_index_read(op, env, nodes, constants, node_id=f"op{ordinal}r0")
            if indexed is not None:
                nodes.append(indexed)
                env[op.results[0]] = indexed.id
                continue
            element_types = tuple(str(getattr(value.type, "element_type", None)) for value in (*op.inputs, *op.results))
            if (
                len(op.inputs) == 1
                and _type(op.inputs[0].type).dtype in {"bf16", "fp8_e4m3", "i8", "i32"}
                and len(op.regions) == 1
                and len(op.regions[0].blocks) == 1
                and tuple(item.name for item in op.regions[0].block.ops) == ("linalg.yield",)
            ):
                value, destination = op.inputs[0], op.outputs[0]
                if env.get(value) is None or env.get(destination, "undefined") is not None:
                    raise LinalgBridgeError("indexed copy needs a defined source and an empty destination")
                maps = _static_index_copy_map(op, _type(value.type), _type(op.results[0].type))
                node = SemanticNode(
                    f"op{ordinal}r0", "gather", (env[value],), _type(op.results[0].type), index_maps=maps
                )
                nodes.append(node)
                env[op.results[0]] = node.id
                continue
            if (
                len(op.inputs) == 1
                and element_types in {("i1", "i64"), ("i64", "i64")}
                and any(item.data == IteratorType.REDUCTION for item in op.iterator_types)
            ):
                from merlin.frontends.linalg_integer_reductions import recognize_static_integer_reduction

                source_type = _type(op.inputs[0].type)
                index_bits = max(2, max(source_type.shape).bit_length() + 1)
                try:
                    pattern = recognize_static_integer_reduction(op, index_bits=index_bits)
                except InvalidLinalgPattern as error:
                    raise LinalgBridgeError(str(error)) from error
                source_id = env.get(op.inputs[0])
                if source_id is None:
                    raise LinalgBridgeError("integer prefix sum reads an uninitialized input")
                node = SemanticNode(
                    f"op{ordinal}r0",
                    "i64_cumsum",
                    (source_id,),
                    _type(op.results[0].type),
                    attrs=(
                        ("axis", pattern.axis[0]),
                        ("minimum_index_bits", index_bits),
                        ("numeric_semantics", "signed-i64-prefix-sum-wrap-v1"),
                    ),
                )
                nodes.append(node)
                env[op.results[0]] = node.id
                continue
            if element_types == ("f32", "f32", "i64") and any(
                item.data == IteratorType.REDUCTION for item in op.iterator_types
            ):
                from .scalar_region import translate_bucketize

                node = translate_bucketize(op, tuple(env.get(value) for value in op.inputs), node_id=f"op{ordinal}r0")
                nodes.append(node)
                env[op.results[0]] = node.id
                continue
            if (
                ("f64" in element_types and not any(item.name == "tensor.extract" for item in op.walk()))
                or (
                    element_types[-1] == "f32"
                    and len(op.regions) == 1
                    and len(op.regions[0].blocks) == 1
                    and tuple(item.name for item in op.regions[0].block.ops)
                    in {
                        ("math.sin", "linalg.yield"),
                        ("math.cos", "linalg.yield"),
                        ("math.powf", "linalg.yield"),
                    }
                )
                or (len(op.inputs) == 3 and all(item.data == IteratorType.PARALLEL for item in op.iterator_types))
                or (
                    op.inputs
                    and all(dtype == "i64" for dtype in element_types)
                    and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
                )
                or (
                    len(op.inputs) == 1
                    and (element_types[0], element_types[-1]) in {("bf16", "f32"), ("f32", "bf16"), ("f32", "i64")}
                )
                or (
                    element_types == ("i64", "f32")
                    and len(op.regions) == 1
                    and len(op.regions[0].blocks) == 1
                    and tuple(item.name for item in op.regions[0].block.ops) == ("arith.sitofp", "linalg.yield")
                )
                or (
                    "bf16" in element_types
                    and element_types[-1] == "f32"
                    and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
                )
                or (
                    element_types[-1] == "bf16"
                    and len(op.inputs) == 2
                    and all(dtype == "bf16" for dtype in element_types)
                    and all(key.startswith("prov.") for key in op.attributes)
                    and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
                )
                or (
                    element_types
                    and all(dtype == "i1" for dtype in element_types)
                    and all(item.data == IteratorType.PARALLEL for item in op.iterator_types)
                )
            ):
                append_scalar_region(op, ordinal)
                continue
        if (
            op.name == "linalg.generic"
            and len(op.results) == 1
            and (
                (str(getattr(op.results[0].type, "element_type", None)) == "i1" and len(op.inputs) == 2)
                or (
                    str(getattr(op.results[0].type, "element_type", None)) == "i64"
                    and len(op.inputs) == 1
                    and str(getattr(op.inputs[0].type, "element_type", None)) == "i1"
                )
            )
        ):
            from .masks import recognize_mask

            try:
                _, maps, attrs = recognize_mask(op)
            except InvalidLinalgPattern as error:
                raise LinalgBridgeError(str(error)) from error
            source_ids = tuple(env.get(value) for value in op.inputs)
            if any(value is None for value in source_ids):
                raise LinalgBridgeError("mask reads an uninitialized input")
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(node_id, "mask", source_ids, _type(op.results[0].type), attrs=attrs, index_maps=maps)
            )
            env[op.results[0]] = node_id
            continue
        if op.name == "linalg.generic" and len(op.results) == 1:
            # A signed reciprocal contains a cast as well as a division. Keep
            # both rounding boundaries instead of treating its i64 input as
            # an embedding index or assuming a fixed mask population.
            if len(op.inputs) == 1 and _type(op.inputs[0].type).dtype == "i64":
                try:
                    composite = recognize_static_composite_math_body(op)
                except InvalidLinalgPattern:
                    composite = None
                if composite is not None and composite.operation == "reciprocal_signed_i64_to_f32":
                    source_id = env.get(op.inputs[0])
                    if source_id is None or env.get(op.outputs[0], "undefined") is not None:
                        raise LinalgBridgeError("signed reciprocal needs a defined input and an empty output")
                    type_ = _type(op.results[0].type)
                    node_id = f"op{ordinal}r0"
                    cast_id, one_id = f"{node_id}_cast", f"{node_id}_one"
                    maps = pointwise_identity_maps(len(type_.shape), 1)
                    nodes.extend(
                        (
                            SemanticNode(
                                cast_id,
                                "signed_i64_to_f32",
                                (source_id,),
                                type_,
                                attrs=(("numeric_semantics", "signed-i64-to-f32-rne-v1"),),
                                index_maps=maps,
                            ),
                            SemanticNode(one_id, "constant", (), type_, effect="constant"),
                            SemanticNode(
                                node_id,
                                "f32_div",
                                (one_id, cast_id),
                                type_,
                                attrs=(("numeric_semantics", "ieee754-f32-div-rne-v1"),),
                                index_maps=pointwise_identity_maps(len(type_.shape), 2),
                            ),
                        )
                    )
                    count = prod(type_.shape)
                    constants[one_id] = TensorValue(type_, (0x3F800000,) * count)
                    constant_bindings.append(ConstantBinding(one_id, "f32-le-raw", "0000803f" * count))
                    env[op.results[0]] = node_id
                    continue
            try:
                generated = recognize_static_iota(op) if not op.inputs else None
                truth_shape = (
                    recognize_f32_nonzero(op)
                    if len(op.inputs) == 1 and str(getattr(op.results[0].type, "element_type", None)) == "i1"
                    else None
                )
            except InvalidLinalgPattern:
                generated, truth_shape = None, None
            except StaticIotaResourceLimit as error:
                from .target_binding import NativeCompilationError

                raise NativeCompilationError("resource_limit", str(error)) from error
            if generated is not None:
                node_id = f"op{ordinal}r0"
                type_ = _type(op.results[0].type)
                nodes.append(SemanticNode(node_id, "constant", (), type_, effect="constant"))
                constants[node_id] = TensorValue(type_, generated)
                data = b"".join(value.to_bytes(8, "little", signed=True) for value in generated)
                constant_bindings.append(ConstantBinding(node_id, "i64-le", data.hex()))
                env[op.results[0]] = node_id
                continue
            if truth_shape is not None:
                source_id = env.get(op.inputs[0])
                if source_id is None:
                    raise LinalgBridgeError("truth conversion reads an uninitialized tensor")
                node_id = f"op{ordinal}r0"
                nodes.append(
                    SemanticNode(
                        node_id,
                        "f32_nonzero",
                        (source_id,),
                        _type(op.results[0].type),
                        attrs=(("numeric_semantics", "f32-une-zero-v1"),),
                        index_maps=pointwise_identity_maps(len(truth_shape), 1),
                    )
                )
                env[op.results[0]] = node_id
                continue
        if op in scan_prelude:
            continue
        if op in ordered_scans:
            scan = ordered_scans[op]
            source = env.get(scan.source)
            if source is None:
                raise LinalgBridgeError("ordered floating scan reads an undefined or uninitialized input")
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    "ordered_f32_scan_f64",
                    (source,),
                    _type(op.results[0].type),
                    attrs=(
                        ("stride", scan.stride),
                        ("extent", scan.extent),
                        ("numeric_semantics", "ordered-f32-prefix-f64-rne-v1"),
                    ),
                )
            )
            env[op.results[0]] = node_id
            continue
        if op in indexed_prelude:
            continue
        if op in indexed_updates:
            indexed = indexed_updates[op]
            values = (indexed.source, indexed.destination, indexed.positions)
            if any(value not in env or env[value] is None for value in values):
                raise LinalgBridgeError("indexed update reads an undefined or uninitialized source")
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    "indexed_insert_slice",
                    tuple(env[value] for value in values),
                    _type(op.results[0].type),
                    (("axis", indexed.axis), ("index_domain", "nonnegative-below-destination-axis")),
                )
            )
            env[op.results[0]] = node_id
            continue
        if op.name == "arith.constant":
            if len(op.results) != 1:
                raise LinalgBridgeError("scalar constant has an unsupported result count")
            from xdsl.dialects.builtin import TensorType as XdslTensorType

            if isinstance(op.results[0].type, XdslTensorType):
                from .linalg_constants import read_dense_literal

                type_, encoding, data, value = read_dense_literal(op)
                node_id = f"op{ordinal}r0"
                nodes.append(SemanticNode(node_id, "constant", (), type_, effect="constant"))
                constants[node_id] = value
                constant_bindings.append(ConstantBinding(node_id, encoding, data.hex()))
                env[op.results[0]] = node_id
                continue
            if is_positive_bf16_zero(op):
                bf16_zeros.add(op.results[0])
            elif str(op.results[0].type) == "f32":
                f32_scalars[op.results[0]] = _scalar_f32_bits(op)
            elif str(op.results[0].type) == "f64":
                from math import isnan
                from struct import pack

                from xdsl.dialects.builtin import FloatAttr

                literal = op.properties.get("value")
                if (
                    not isinstance(literal, FloatAttr)
                    or str(literal.type) != "f64"
                    or set(op.properties) != {"value"}
                    or any(not key.startswith("prov.") for key in op.attributes)
                    or isnan(literal.value.data)
                ):
                    raise LinalgBridgeError("f64 scalar needs an exact non-NaN literal")
                f64_scalars[op.results[0]] = int.from_bytes(pack("<d", literal.value.data), "little")
            elif str(op.results[0].type) == "i64":
                i64_scalars[op.results[0]] = _scalar_i64(op)
            elif str(op.results[0].type) == "i1":
                literal = op.properties.get("value")
                if (
                    not isinstance(literal, IntegerAttr)
                    or str(literal.type) != "i1"
                    or literal.value.data not in (-1, 0, 1)
                    or set(op.properties) != {"value"}
                    or any(not key.startswith("prov.") for key in op.attributes)
                ):
                    raise LinalgBridgeError("Boolean scalar needs an exact i1 literal")
                boolean_scalars[op.results[0]] = literal.value.data & 1
            else:
                scalars[op.results[0]] = _scalar_integer(op)
        elif op.name == "arith.sitofp":
            if (
                len(op.operands) != 1
                or len(op.results) != 1
                or op.properties
                or any(not key.startswith("prov.") for key in op.attributes)
                or str(op.operands[0].type) != "i64"
                or str(op.results[0].type) != "f32"
                or op.operands[0] not in i64_scalars
            ):
                raise LinalgBridgeError("scalar signed-i64-to-f32 cast needs a declared constant")
            try:
                op.verify()
            except Exception as exc:  # noqa: BLE001 - xDSL owns source verification
                raise LinalgBridgeError(f"scalar signed-i64-to-f32 cast failed verification: {exc}") from exc
            f32_scalars[op.results[0]] = _i64_to_f32_bits(i64_scalars[op.operands[0]])
        elif op.name == "tensor.empty":
            if len(op.results) != 1:
                raise LinalgBridgeError("tensor.empty has an unsupported result count")
            _type(op.results[0].type)
            env[op.results[0]] = None
        elif op.name == "tensor.splat":
            if (
                len(op.operands) != 1
                or len(op.results) != 1
                or op.properties
                or any(not key.startswith("prov.") for key in op.attributes)
            ):
                raise LinalgBridgeError("tensor.splat has an unsupported signature or attributes")
            scalar = op.operands[0]
            type_ = _type(op.results[0].type)
            node_id = f"op{ordinal}r0"
            if type_.dtype == "f32" and scalar in f32_scalars:
                bits = f32_scalars[scalar]
                value, encoding, data = bits, "f32-le-raw", bits.to_bytes(4, "little")
            elif type_.dtype == "f64" and scalar in f64_scalars:
                bits = f64_scalars[scalar]
                value, encoding, data = bits, "f64-le-raw", bits.to_bytes(8, "little")
            elif type_.dtype == "i1" and scalar in boolean_scalars:
                value = boolean_scalars[scalar]
                encoding, data = "i1-raw", bytes((value,))
            elif type_.dtype == "i32" and scalar in scalars:
                value = scalars[scalar]
                encoding, data = "i32-le", value.to_bytes(4, "little", signed=True)
            elif type_.dtype == "i64" and scalar in i64_scalars:
                value = i64_scalars[scalar]
                encoding, data = "i64-le", value.to_bytes(8, "little", signed=True)
            elif type_.dtype == "bf16" and scalar in bf16_zeros:
                value, encoding, data = 0, "bf16-le-raw", b"\x00\x00"
            else:
                raise LinalgBridgeError("tensor.splat requires an admitted constant with matching element type")
            env[op.results[0]] = append_splat(node_id, type_, value, encoding, data)
        elif op.name in {"tensor.expand_shape", "tensor.collapse_shape"}:
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
        elif op.name == "tensor.concat":
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
                        source.shape[dim] != result_type.shape[dim]
                        for dim in range(len(result_type.shape))
                        if dim != axis
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
                continue
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
        elif op.name == "tensor.extract_slice":
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
                or (source_type.dtype, source_type.numerical_policy)
                != (result_type.dtype, result_type.numerical_policy)
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
                    continue
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
        elif op.name == "tensor.insert_slice":
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
                continue
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
        elif op.name == "linalg.fill":
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
                continue
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
        elif op.name == "linalg.copy":
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
        elif op.name == "linalg.transpose":
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
        elif op.name == "linalg.generic" and len(op.inputs) == 1 and (_type(op.inputs[0].type).dtype == "i64"):
            indices, table, destination = _embedding_lookup_operands(op)
            if (
                indices not in env
                or env[indices] is None
                or table not in env
                or env[table] is None
                or destination not in env
                or env[destination] is not None
            ):
                raise LinalgBridgeError("embedding lookup needs defined indices/table and an empty destination")
            table_type = _type(table.type)
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    "embedding_lookup",
                    (env[indices], env[table]),
                    _type(op.results[0].type),
                    attrs=(
                        ("index_domain", "nonnegative-below-table-rows"),
                        ("minimum_index_bits", max(2, (table_type.shape[0] - 1).bit_length() + 1)),
                    ),
                )
            )
            env[op.results[0]] = node_id
        elif (
            op.name == "linalg.generic"
            and len(op.inputs) == 2
            and ("merlin.numeric_semantics" in op.attributes and "merlin.reduction_order" not in op.attributes)
        ):
            try:
                policy = op.attributes["merlin.numeric_semantics"]
                if not isinstance(policy, StringAttr):
                    raise LinalgBridgeError("BF16 binary numerical policy must be a string")
                if policy.data == "selected-f32-rne-bf16-chop-v1":
                    body = (
                        tuple(op.regions[0].block.ops)
                        if len(op.regions) == 1 and len(op.regions[0].blocks) == 1
                        else ()
                    )
                    if len(body) > 2 and body[2].name == "arith.subf":
                        semantic_op = "sub_bf16"
                        lhs, rhs, destination, semantics, domain = recognize_bf16_f32_chop_sub(op)
                    else:
                        semantic_op = "add_bf16"
                        lhs, rhs, destination, semantics, domain = recognize_bf16_f32_chop_add(op)
                elif policy.data == "bf16-product-rne-v1":
                    semantic_op = "mul_bf16"
                    lhs, rhs, destination, semantics, domain = recognize_bf16_product_rne(op)
                else:
                    raise LinalgBridgeError(f"unadmitted BF16 binary numerical policy {policy.data!r}")
            except InvalidLinalgPattern as exc:
                raise LinalgBridgeError(str(exc)) from exc
            if any(value not in env or env[value] is None for value in (lhs, rhs)) or (
                destination not in env or env[destination] is not None
            ):
                raise LinalgBridgeError("BF16 binary operation needs two defined sources and an empty destination")
            source_ids = (env[lhs], env[rhs])
            source_nodes = {node.id: node for node in nodes}
            if any(
                source_nodes[source_id].op == "input"
                and dict(source_nodes[source_id].attrs).get("input_domain") != domain
                for source_id in source_ids
            ):
                raise LinalgBridgeError("BF16 binary premise differs from its source boundary")
            result_type = _type(op.results[0].type)
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    semantic_op,
                    source_ids,
                    result_type,
                    attrs=(("input_domain", domain), ("numeric_semantics", semantics)),
                    index_maps=pointwise_identity_maps(2, 2),
                )
            )
            env[op.results[0]] = node_id
        elif op.name == "linalg.generic" and len(op.inputs) == 2 and ("merlin.reduction_order" in op.attributes):
            try:
                lhs, rhs, init, semantics, domain = recognize_ordered_fp8_bf16_reset_matmul(op)
            except InvalidLinalgPattern as exc:
                raise LinalgBridgeError(str(exc)) from exc
            if lhs not in env or rhs not in env:
                raise LinalgBridgeError("ordered FP8 matmul needs declared inputs")
            source_ids = (env[lhs], env[rhs])
            if any(source_id is None for source_id in source_ids):
                raise LinalgBridgeError("ordered FP8 matmul reads uninitialized operands")
            source_nodes = {node.id: node for node in nodes}
            if any(
                (
                    source_nodes[source_id].op == "input"
                    and dict(source_nodes[source_id].attrs).get("input_domain") != domain
                )
                or (
                    source_nodes[source_id].op != "input"
                    and not (
                        source_nodes[source_id].op == "pack_fp8"
                        and source_nodes[source_id].type.dtype == "fp8_e4m3"
                        and source_nodes[source_id].type.numerical_policy != "raw-bit-pattern"
                    )
                )
                for source_id in source_ids
            ):
                raise LinalgBridgeError("ordered FP8 matmul needs declared boundary inputs or a computed FP8 pack")
            # Preserve a computed pack's policy and def-use edge. The selected
            # target must establish that its result satisfies this matmul's
            # declared input domain; parsing alone cannot prove that closure.
            node_id = f"op{ordinal}r0"
            result_type = _type(op.results[0].type)
            accumulator_id = None if init in zero_fills else env.get(init)
            if init not in zero_fills:
                producer = source_nodes.get(accumulator_id)
                if (
                    producer is None
                    or producer.op not in {"matmul", "matmul_accumulate"}
                    or producer.type != TensorType(result_type.shape, result_type.dtype, semantics)
                ):
                    raise LinalgBridgeError("ordered FP8 continuation needs a matching prior ordered accumulator")
            nodes.append(
                SemanticNode(
                    node_id,
                    "matmul" if accumulator_id is None else "matmul_accumulate",
                    source_ids if accumulator_id is None else (*source_ids, accumulator_id),
                    TensorType(result_type.shape, result_type.dtype, semantics),
                    index_maps=(
                        _oriented_matmul_maps() if accumulator_id is None else _oriented_matmul_accumulate_maps()
                    ),
                )
            )
            env[op.results[0]] = node_id
        elif op.name == "linalg.generic" and len(op.inputs) == 1 and (_type(op.results[0].type).dtype == "fp8_e4m3"):
            try:
                source_value, destination, semantics, domain, scale_code = recognize_bf16_fp8_truncation(op)
            except InvalidLinalgPattern as exc:
                raise LinalgBridgeError(str(exc)) from exc
            if (
                source_value not in env
                or env[source_value] is None
                or destination not in env
                or env[destination] is not None
            ):
                raise LinalgBridgeError("BF16-to-FP8 truncation needs a defined source and empty destination")
            source_id = env[source_value]
            source_node = next(node for node in nodes if node.id == source_id)
            if source_node.op == "input" and dict(source_node.attrs).get("input_domain") != domain:
                raise LinalgBridgeError("BF16-to-FP8 input domain differs from its source boundary")
            # A computed producer may discharge the declared input-domain
            # premise through a target rule. Keep its def-use edge intact;
            # the frontend does not infer numerical closure from an op name.
            result_type = _type(op.results[0].type)
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    "pack_fp8",
                    (source_id,),
                    TensorType(result_type.shape, result_type.dtype, semantics),
                    attrs=(("input_domain", domain), ("scale_code", scale_code)),
                    index_maps=pointwise_identity_maps(2, 1),
                )
            )
            env[op.results[0]] = node_id
        elif op.name == "linalg.generic" and len(op.inputs) == 1 and (_type(op.results[0].type).dtype == "bf16"):
            try:
                source_value, destination = recognize_bf16_sign_clamp_relu(op)
            except InvalidLinalgPattern as exc:
                raise LinalgBridgeError(str(exc)) from exc
            if source_value not in env or env[source_value] is None or destination not in env:
                raise LinalgBridgeError("BF16 ReLU reads an undefined or uninitialized tensor")
            source_type = _type(source_value.type)
            destination_type = _type(destination.type)
            result_type = _type(op.results[0].type)
            if (
                source_type.dtype != "bf16"
                or destination_type != result_type
                or result_type.dtype != "bf16"
                or source_type.shape != result_type.shape
            ):
                raise LinalgBridgeError("BF16 ReLU result or destination differs from its source shape and dtype")
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    "relu",
                    (env[source_value],),
                    result_type,
                    attrs=(("input_domain", "finite-normal-or-zero-bf16"), ("numeric_semantics", "bf16-sign-clamp-v1")),
                    index_maps=pointwise_identity_maps(2, 1),
                )
            )
            env[op.results[0]] = node_id
        elif (
            op.name == "linalg.generic"
            and len(op.inputs) == 2
            and len(op.results) == 1
            and _type(op.results[0].type).dtype == "f32"
            and any(iterator.data == IteratorType.REDUCTION for iterator in op.iterator_types.data)
        ):
            lhs, rhs, init, maps, add_order = _source_f32_batched_matmul(op)
            if any(value not in env or env[value] is None for value in (lhs, rhs, init)):
                raise LinalgBridgeError("source f32 batched matmul reads an undefined initializer or operand")
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    "source_f32_matmul_unqualified",
                    tuple(env[value] for value in (lhs, rhs, init)),
                    _type(op.results[0].type),
                    attrs=(
                        ("reduction_order", "unresolved"),
                        ("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
                        ("scalar_add_order", add_order),
                    ),
                    index_maps=maps,
                )
            )
            env[op.results[0]] = node_id
        elif op.name == "linalg.generic" and _type(op.results[0].type).dtype == "f32":
            if len(op.inputs) == 2 and len(op.outputs) == 1:
                try:
                    pointwise = recognize_static_projected_pointwise(op)
                except InvalidLinalgPattern as exc:
                    try:
                        pointwise = recognize_static_f32_division(op)
                    except InvalidLinalgPattern as division_exc:
                        raise LinalgBridgeError(
                            f"f32 linalg.generic needs an admitted semantic decomposition: {exc}; {division_exc}"
                        ) from division_exc
                operation = {
                    "arith.addf": "f32_add",
                    "arith.subf": "f32_sub",
                    "arith.mulf": "f32_mul",
                    "arith.divf": "f32_div",
                }.get(pointwise.operation)
                if operation is None:
                    raise LinalgBridgeError("f32 linalg.generic needs an admitted semantic decomposition")
                lhs, rhs = op.inputs
                destination = op.outputs[0]
                result_type = _type(op.results[0].type)
                if (
                    lhs not in env
                    or env[lhs] is None
                    or rhs not in env
                    or env[rhs] is None
                    or destination not in env
                    or env[destination] is not None
                    or any(
                        _type(value.type).dtype != "f32" or _type(value.type).numerical_policy != "raw-bit-pattern"
                        for value in (lhs, rhs)
                    )
                    or _type(destination.type) != result_type
                ):
                    raise LinalgBridgeError("f32 arithmetic needs defined inputs and an empty output")
                rank = len(result_type.shape)
                maps = tuple(
                    IndexMap(
                        rank,
                        tuple(row for row, _ in rows),
                        tuple(offset for _, offset in rows),
                    )
                    for rows in (
                        tuple(_linear_affine_expression(expression, rank) for expression in mapping.data.results)
                        for mapping in op.get_indexing_maps()
                    )
                )
                node_id = f"op{ordinal}r0"
                nodes.append(
                    SemanticNode(
                        node_id,
                        operation,
                        (env[lhs], env[rhs]),
                        result_type,
                        attrs=(("numeric_semantics", f"ieee754-f32-{operation[4:]}-rne-v1"),),
                        index_maps=maps,
                    )
                )
                env[op.results[0]] = node_id
            elif len(op.inputs) != 1 or len(op.outputs) != 1:
                raise LinalgBridgeError("f32 linalg.generic needs an admitted semantic decomposition")
            else:
                source_value, destination = op.inputs[0], op.outputs[0]
                if (
                    source_value not in env
                    or env[source_value] is None
                    or destination not in env
                    or env[destination] is not None
                ):
                    raise LinalgBridgeError("indexed copy needs a defined source and an empty destination")
                source_type, result_type = _type(source_value.type), _type(op.results[0].type)
                if (
                    source_type.dtype != "f32"
                    or result_type != _type(destination.type)
                    or source_type.numerical_policy != result_type.numerical_policy
                ):
                    raise LinalgBridgeError("indexed copy changes type or numerical policy")
                try:
                    maps = _static_index_copy_map(op, source_type, result_type)
                    semantic_op = "gather"
                    inputs = (env[source_value],)
                    attrs = ()
                except LinalgBridgeError as copy_error:
                    try:
                        maps = _static_f32_square_maps(op, source_type, result_type)
                    except LinalgBridgeError as square_error:
                        try:
                            checked_source, checked_destination = recognize_f32_relu_ordered_select(op)
                        except InvalidLinalgPattern as relu_error:
                            try:
                                math_source = recognize_static_f32_exp_rsqrt_body(op)
                            except InvalidLinalgPattern as math_error:
                                try:
                                    composite = recognize_static_composite_math_body(op)
                                except InvalidLinalgPattern as composite_error:
                                    try:
                                        # Preserve the existing closed f32 forms.
                                        # This additional family is the ordered
                                        # literal clamp, not arbitrary scalar SSA.
                                        if tuple(item.name for item in op.regions[0].block.ops) != (
                                            "arith.constant",
                                            "arith.cmpf",
                                            "arith.select",
                                            "linalg.yield",
                                        ):
                                            raise LinalgBridgeError("not an ordered literal comparison/select")
                                        append_scalar_region(op, ordinal)
                                    except LinalgBridgeError as scalar_error:
                                        raise LinalgBridgeError(
                                            "f32 linalg.generic needs an admitted semantic decomposition: "
                                            f"{copy_error}; {square_error}; {relu_error}; "
                                            f"{math_error}; {composite_error}; {scalar_error}"
                                        ) from scalar_error
                                    continue
                                semantic_op = "source_f32_composite_unqualified"
                                maps = pointwise_identity_maps(len(result_type.shape), 1)
                                inputs = (env[source_value],)
                                attrs = (
                                    ("numeric_semantics", "source-f32-composite-unqualified-v1"),
                                    ("operation", composite.operation),
                                    ("literal_bits", ",".join(composite.literal_bits)),
                                )
                            else:
                                kind = math_source.operation.removeprefix("math.")
                                semantic_op = f"source_f32_{kind}_unqualified"
                                maps = pointwise_identity_maps(len(result_type.shape), 1)
                                inputs = (env[source_value],)
                                attrs = (("numeric_semantics", f"source-f32-{kind}-unqualified-v1"),)
                        else:
                            if checked_source is not source_value or checked_destination is not destination:
                                raise LinalgBridgeError("f32 zero select operands differ from the parsed source")
                            semantic_op = "f32_relu_ordered_select"
                            maps = pointwise_identity_maps(len(result_type.shape), 1)
                            inputs = (env[source_value],)
                            attrs = (("numeric_semantics", "f32-ordered-lt-positive-zero-select-v1"),)
                    else:
                        semantic_op = "f32_mul"
                        inputs = (env[source_value], env[source_value])
                        attrs = (("numeric_semantics", "ieee754-f32-mul-rne-v1"),)
                node_id = f"op{ordinal}r0"
                nodes.append(
                    SemanticNode(
                        node_id,
                        semantic_op,
                        inputs,
                        result_type,
                        attrs=attrs,
                        index_maps=maps,
                    )
                )
                env[op.results[0]] = node_id
        elif op.name in {"linalg.generic", "linalg.matmul"}:
            if op.name == "linalg.generic":
                try:
                    lhs, rhs, init = recognize_signed_i8_i32_matmul(op)
                except InvalidLinalgPattern as exc:
                    raise LinalgBridgeError(str(exc)) from exc
            else:
                if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
                    raise LinalgBridgeError("named matmul has an unsupported signature")
                if not _named_matmul_maps_are_default(op):
                    raise LinalgBridgeError("named matmul has nonstandard indexing or iterators")
                lhs, rhs, init = (*op.inputs, op.outputs[0])
            if any(value not in env or env[value] is None for value in (lhs, rhs, init)):
                raise LinalgBridgeError("matmul reads an undefined or uninitialized operand")
            lhs_type, rhs_type, init_type, result_type = (
                _type(value.type) for value in (lhs, rhs, init, op.results[0])
            )
            if lhs_type.dtype != rhs_type.dtype or lhs_type.numerical_policy != rhs_type.numerical_policy:
                raise LinalgBridgeError("matmul input dtypes or numerical policies differ")
            if op.name == "linalg.matmul" and lhs_type.dtype not in {"i32", "f32"}:
                raise LinalgBridgeError("named matmul needs i32 operands or a source-only f32 graph node")
            if (
                len(lhs_type.shape) != 2
                or len(rhs_type.shape) != 2
                or (
                    lhs_type.shape[1] != rhs_type.shape[0]
                    or result_type.shape != (lhs_type.shape[0], rhs_type.shape[1])
                    or init_type != result_type
                    or result_type.dtype != ("f32" if lhs_type.dtype == "f32" else "i32")
                )
            ):
                raise LinalgBridgeError("matmul shape or initialized result type differs")
            node_id = f"op{ordinal}r0"
            if op.name == "linalg.matmul" and lhs_type.dtype == "f32":
                from .precision_intake import _check_matmul_body

                if set(op.properties) != {"indexing_maps", "operandSegmentSizes"} or any(
                    not key.startswith("prov.") for key in op.attributes
                ):
                    raise LinalgBridgeError("source f32 matmul has an unknown semantic attribute")
                _check_matmul_body(op)
                try:
                    op.verify()
                except Exception as exc:  # noqa: BLE001 - xDSL owns source verification
                    raise LinalgBridgeError(f"source f32 matmul failed verification: {exc}") from exc
                nodes.append(
                    SemanticNode(
                        node_id,
                        "source_f32_matmul_unqualified",
                        tuple(env[value] for value in (lhs, rhs, init)),
                        result_type,
                        attrs=(
                            ("reduction_order", "unresolved"),
                            ("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
                        ),
                        index_maps=_maps(),
                    )
                )
                env[op.results[0]] = node_id
                continue
            nodes.append(
                SemanticNode(
                    node_id,
                    "matmul_accumulate",
                    tuple(env[value] for value in (lhs, rhs, init)),
                    result_type,
                    index_maps=_maps(),
                )
            )
            env[op.results[0]] = node_id
        elif op.name == "func.return":
            if ordinal != len(operations) - 1 or not op.operands:
                raise LinalgBridgeError("func.return must be final and return tensors")
            if tuple(value.type for value in op.operands) != tuple(function.function_type.outputs.data):
                raise LinalgBridgeError("returned tensor types differ from the entry signature")
            outputs = tuple(env.get(value) for value in op.operands)
            if any(output is None for output in outputs):
                raise LinalgBridgeError("function returns an undefined or uninitialized tensor")
            request = KernelRequest(
                nodes=tuple(nodes),
                outputs=outputs,
                output_storages=(boundary_storage,) * len(outputs),
                input_storages=tuple(boundaries),
                target_identity=target_identity,
                lowering_policy=lowering_policy,
                source_identity=sha256(source.encode()).hexdigest(),
                constants=tuple(constant_bindings),
            )
            return LinalgTranslation(request, constants, tuple(source_operations))
        elif op.name == "linalg.reduce" and op.results and _type(op.results[0].type).dtype == "i64":
            from merlin.frontends.linalg_integer_reductions import recognize_static_integer_reduction

            source_type = _type(op.input.type)
            # This is a residual index-width requirement, not an observation of
            # the target's scalar registers or the host pointer width.
            index_bits = max(2, max(source_type.shape).bit_length() + 1)
            try:
                pattern = recognize_static_integer_reduction(op, index_bits=index_bits)
            except InvalidLinalgPattern as error:
                raise LinalgBridgeError(str(error)) from error
            if len(pattern.axis) != 1:
                raise LinalgBridgeError("integer sum currently needs one axis")
            sources = tuple(env.get(value) for value in (op.input, op.init))
            if any(value is None for value in sources):
                raise LinalgBridgeError("integer sum reads an uninitialized input or seed")
            node_id = f"op{ordinal}r0"
            nodes.append(
                SemanticNode(
                    node_id,
                    "i64_reduce_add",
                    sources,
                    _type(op.results[0].type),
                    attrs=(
                        ("axis", pattern.axis[0]),
                        ("minimum_index_bits", index_bits),
                        ("numeric_semantics", "i64-add-modulo-2pow64-v1"),
                    ),
                    index_maps=_f32_reduce_maps(len(source_type.shape), pattern.axis),
                )
            )
            env[op.results[0]] = node_id
        elif op.name == "linalg.reduce" and op.results and _type(op.results[0].type).dtype == "f32":
            source_value, init, axes, order, add_order, scalar_op = _recognize_f32_reduce(op)
            if any(value not in env or env[value] is None for value in (source_value, init)):
                raise LinalgBridgeError("ordered f32 reduction needs a defined input and initializer")
            source_type = _type(source_value.type)
            node_id = f"op{ordinal}r0"
            maximum = scalar_op == "arith.maximumf"
            nodes.append(
                SemanticNode(
                    node_id,
                    (
                        "source_f32_reduce_max_unqualified"
                        if maximum
                        else "f32_reduce_add"
                        if order == "axis-ascending"
                        else "source_f32_reduce_add_unqualified"
                    ),
                    (env[source_value], env[init]),
                    _type(op.results[0].type),
                    attrs=(
                        ("axis", axes[0]) if len(axes) == 1 else ("axes", ",".join(map(str, axes))),
                        ("reduction_order", order),
                        (
                            "numeric_semantics",
                            "source-f32-maximum-unqualified-v1"
                            if maximum
                            else "ieee754-f32-add-rne-v1"
                            if order == "axis-ascending"
                            else "source-f32-add-unqualified-v1",
                        ),
                        *(
                            (("scalar_operand_order" if maximum else "scalar_add_order", add_order),)
                            if order == "unresolved"
                            else ()
                        ),
                    ),
                    index_maps=_f32_reduce_maps(len(source_type.shape), axes),
                )
            )
            env[op.results[0]] = node_id
        else:
            raise LinalgBridgeError(f"no admitted translation for source operation {op.name}")
    raise LinalgBridgeError("entry function has no return")
