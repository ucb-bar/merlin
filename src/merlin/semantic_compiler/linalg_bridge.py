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
from .reference_values import TensorValue
from .linalg_state import LinalgTranslation, TranslationState
from . import linalg_scalar, linalg_preparation, linalg_views, linalg_arithmetic, linalg_contractions, linalg_returns




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
            recognize_signed_i8_i32_matmul,
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
    state = TranslationState(
        nodes=nodes,
        env=env,
        partial_tensors=partial_tensors,
        scalars=scalars,
        i64_scalars=i64_scalars,
        f32_scalars=f32_scalars,
        f64_scalars=f64_scalars,
        boolean_scalars=boolean_scalars,
        bf16_zeros=bf16_zeros,
        zero_fills=zero_fills,
        constants=constants,
        constant_bindings=constant_bindings,
        boundaries=boundaries,
        source_operations=source_operations,
        operations=operations,
        function=function,
        boundary_storage=boundary_storage,
        target_identity=target_identity,
        lowering_policy=lowering_policy,
        source=source,
        append_splat=append_splat,
        append_scalar_region=append_scalar_region,
        scan_prelude=scan_prelude,
        ordered_scans=ordered_scans,
        indexed_prelude=indexed_prelude,
        indexed_updates=indexed_updates,
    )
    handlers = (linalg_scalar.translate, linalg_preparation.translate, linalg_views.translate, linalg_arithmetic.translate, linalg_contractions.translate, linalg_returns.translate)
    for ordinal, op in enumerate(operations):
        source_operations.append(op.name)
        for handler in handlers:
            result = handler(state, op, ordinal)
            if isinstance(result, LinalgTranslation):
                return result
            if result:
                break
        else:
            raise LinalgBridgeError(f"no admitted translation for source operation {op.name}")
    raise LinalgBridgeError("entry function has no return")
