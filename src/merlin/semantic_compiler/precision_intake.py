"""Check source-bound precision decisions against parsed Linalg contractions.

This is an intake check, not a precision transformation. A target provider
must validate its policy fields and implement the selected arithmetic before
these regions can enter native instruction selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from struct import pack

from xdsl.dialects.builtin import FloatAttr, StringAttr, TensorType

from merlin.frontends.linalg_mlir import parse_mlir_text

from .linalg_bridge import (
    LinalgBridgeError,
    _named_matmul_maps_are_default,
    _source_f32_batched_matmul,
)


@dataclass(frozen=True)
class SourceValueRef:
    kind: str
    index: int
    result_index: int = 0

    def record(self) -> dict[str, str | int]:
        return {"kind": self.kind, "index": self.index, "result_index": self.result_index}


@dataclass(frozen=True)
class SourceMatmulRegion:
    region_id: str
    operation_ordinal: int
    lhs_shape: tuple[int, int]
    rhs_shape: tuple[int, int]
    result_shape: tuple[int, int]
    lhs_source: SourceValueRef
    rhs_source: SourceValueRef
    result_source: SourceValueRef


@dataclass(frozen=True)
class SourceBatchedMatmulRegion:
    region_id: str
    operation_ordinal: int
    batch_shape: tuple[int, ...]
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    result_shape: tuple[int, ...]
    lhs_source: SourceValueRef
    rhs_source: SourceValueRef
    result_source: SourceValueRef
    scalar_add_order: str


def _f32_shape(value, *, rank: int = 2) -> tuple[int, ...]:
    type_ = value.type
    if not isinstance(type_, TensorType) or not type_.has_static_shape() or str(type_.element_type) != "f32":
        raise LinalgBridgeError("precision preflight requires static f32 tensor operands")
    shape = tuple(int(size) for size in type_.get_shape())
    if len(shape) != rank or any(size <= 0 for size in shape):
        label = "rank-two" if rank == 2 else f"rank-{rank}"
        raise LinalgBridgeError(f"precision preflight requires positive {label} tensor shapes")
    return shape


def _check_positive_zero_init(init, *, rank: int = 2) -> None:
    fill = getattr(init, "owner", None)
    if getattr(fill, "name", None) == "tensor.splat":
        if (
            len(fill.operands) != 1 or len(fill.results) != 1 or fill.results[0] is not init
            or fill.properties or any(not key.startswith("prov.") for key in fill.attributes)
        ):
            raise LinalgBridgeError("precision preflight requires a closed tensor.splat reset")
        scalar = fill.operands[0]
        _f32_shape(init, rank=rank)
    elif getattr(fill, "name", None) == "linalg.fill" and len(fill.operands) == 2:
        scalar, destination = fill.operands
        if getattr(getattr(destination, "owner", None), "name", None) != "tensor.empty":
            raise LinalgBridgeError("precision preflight requires an empty tensor before the zero fill")
        if destination.type != init.type or _f32_shape(destination, rank=rank) != _f32_shape(init, rank=rank):
            raise LinalgBridgeError("precision preflight init and empty tensor types differ")
    else:
        raise LinalgBridgeError("precision preflight requires an explicit zero-filled matmul init")
    constant = getattr(scalar, "owner", None)
    attr = None if getattr(constant, "name", None) != "arith.constant" else constant.properties.get("value")
    if (
        str(scalar.type) != "f32"
        or not isinstance(attr, FloatAttr)
        or pack("<f", float(attr.value.data)) != b"\x00\x00\x00\x00"
    ):
        raise LinalgBridgeError("precision preflight requires a positive-zero f32 matmul init")


def _check_matmul_body(op) -> None:
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise LinalgBridgeError("precision preflight requires one parsed matmul scalar region")
    block = op.regions[0].blocks[0]
    body = tuple(block.ops)
    if (
        len(block.args) != 3
        or tuple(item.name for item in body) != ("arith.mulf", "arith.addf", "linalg.yield")
        or tuple(body[0].operands) != (block.args[0], block.args[1])
        or tuple(body[1].operands) != (body[0].results[0], block.args[2])
        or tuple(body[2].operands) != (body[1].results[0],)
    ):
        raise LinalgBridgeError("precision preflight matmul scalar body differs from multiply/add/yield")
    for arithmetic in body[:2]:
        flags = arithmetic.properties.get("fastmath")
        if flags is not None and flags.data:
            raise LinalgBridgeError("precision preflight rejects fast-math arithmetic")


def check_source_bound_f32_matmuls(
    program: str,
    *,
    entry: str,
    policy: dict,
    original_source_sha256: str,
) -> tuple[SourceMatmulRegion, ...]:
    """Match declared rank-two linalg.matmul decisions exactly once.

    The original capture digest can differ from the staged IR digest because
    staging rewrites the external weights path. Only source correspondence and
    the local f32 contraction shape/body are checked here. Policy arithmetic,
    quantization quality, and target execution remain separate obligations.
    Batched linalg.generic contractions have a separate source inventory.
    """
    if not isinstance(policy, dict) or policy.get("source_sha256") != original_source_sha256:
        raise LinalgBridgeError("precision policy differs from captured source identity")
    decisions = policy.get("matmul_regions")
    if not isinstance(decisions, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, dict)
        for key, value in decisions.items()
    ):
        raise LinalgBridgeError("precision policy needs one decision object per matmul region")
    module = parse_mlir_text(program)
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise LinalgBridgeError("precision preflight needs one parsed entry block")
    block = functions[0].body.block
    operations = tuple(block.ops)
    ordinals = {op: index for index, op in enumerate(operations)}

    def source(value) -> SourceValueRef:
        if value.owner is block:
            return SourceValueRef("argument", tuple(block.args).index(value))
        if value.owner in ordinals:
            return SourceValueRef("operation_result", ordinals[value.owner], value.index)
        raise LinalgBridgeError("precision preflight matmul operand has an unknown source")

    regions: list[SourceMatmulRegion] = []
    for ordinal, op in enumerate(operations):
        if op.name != "linalg.matmul":
            continue
        if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
            raise LinalgBridgeError("precision preflight matmul has an unsupported signature")
        if not _named_matmul_maps_are_default(op):
            raise LinalgBridgeError("precision preflight matmul maps or iterators differ")
        if any(not key.startswith("prov.") for key in op.attributes):
            raise LinalgBridgeError("precision preflight matmul has an unknown semantic attribute")
        lhs_shape, rhs_shape, init_shape, result_shape = (
            _f32_shape(value) for value in (*op.inputs, op.outputs[0], op.results[0])
        )
        if lhs_shape[1] != rhs_shape[0] or init_shape != result_shape or result_shape != (
            lhs_shape[0], rhs_shape[1]
        ):
            raise LinalgBridgeError("precision preflight matmul shapes differ")
        _check_positive_zero_init(op.outputs[0])
        _check_matmul_body(op)
        region_attr = op.attributes.get("prov.region_id")
        if region_attr is not None and (not isinstance(region_attr, StringAttr) or not region_attr.data):
            raise LinalgBridgeError("precision preflight has an invalid source region ID")
        region_id = region_attr.data if region_attr is not None else f"matmul#{len(regions)}"
        if any(region.region_id == region_id for region in regions):
            raise LinalgBridgeError(f"duplicate source matmul region {region_id!r}")
        regions.append(SourceMatmulRegion(
            region_id, ordinal, lhs_shape, rhs_shape, result_shape,
            source(op.inputs[0]), source(op.inputs[1]), source(op.results[0]),
        ))
    source_ids = {region.region_id for region in regions}
    if source_ids != decisions.keys():
        missing = sorted(source_ids - decisions.keys())
        extra = sorted(decisions.keys() - source_ids)
        raise LinalgBridgeError(f"precision policy matmul regions differ: missing={missing}, extra={extra}")
    return tuple(regions)


def inventory_source_f32_batched_matmuls(program: str, *, entry: str) -> tuple[SourceBatchedMatmulRegion, ...]:
    """Identify parsed batched f32 contractions that still need a target route.

    This inventory does not select a precision policy or flatten a batch into
    independent tiles. It rejects nonzero initializers and changed scalar
    bodies before a later target-specific tiler may consider the region.
    """
    return _batched_matmul_inventory(program, entry=entry, strict=True)[0]


def survey_source_f32_batched_matmuls(
    program: str, *, entry: str,
) -> tuple[tuple[SourceBatchedMatmulRegion, ...], tuple[tuple[int, str], ...]]:
    """Inventory batched contractions without refusing the noncanonical ones.

    A route listing that selects no batched precision decision keeps a
    rank-3+ f32 contraction that is not the canonical batched matmul (for
    example a transposed operand map or a nonzero initializer) as an unrouted
    source operation. It returns those operations' ordinals and refusal
    reasons beside the canonical regions instead of failing the program.
    """
    return _batched_matmul_inventory(program, entry=entry, strict=False)


def _batched_matmul_inventory(
    program: str, *, entry: str, strict: bool,
) -> tuple[tuple[SourceBatchedMatmulRegion, ...], tuple[tuple[int, str], ...]]:
    from xdsl.dialects.linalg.attrs import IteratorType

    module = parse_mlir_text(program)
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == entry]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise LinalgBridgeError("batched precision inventory needs one parsed entry block")
    block = functions[0].body.block
    operations = tuple(block.ops)
    ordinals = {op: index for index, op in enumerate(operations)}

    def source(value) -> SourceValueRef:
        if value.owner is block:
            return SourceValueRef("argument", tuple(block.args).index(value))
        if value.owner in ordinals:
            return SourceValueRef("operation_result", ordinals[value.owner], value.index)
        raise LinalgBridgeError("batched source operand has an unknown producer")

    regions = []
    refused: list[tuple[int, str]] = []
    for ordinal, op in enumerate(operations):
        if (
            op.name != "linalg.generic" or len(op.inputs) != 2 or len(op.results) != 1
            or not isinstance(op.results[0].type, TensorType)
            or len(op.results[0].type.get_shape()) < 3
            or str(op.results[0].type.element_type) != "f32"
            or not any(iterator.data == IteratorType.REDUCTION for iterator in op.iterator_types.data)
        ):
            continue
        rank = len(op.results[0].type.get_shape())
        try:
            lhs, rhs, init, _, scalar_add_order = _source_f32_batched_matmul(op)
            _check_positive_zero_init(init, rank=rank)
            lhs_shape, rhs_shape, result_shape = (
                _f32_shape(value, rank=rank) for value in (lhs, rhs, op.results[0])
            )
        except LinalgBridgeError as error:
            if strict:
                raise
            refused.append((ordinal, str(error)))
            continue
        region_attr = op.attributes.get("prov.region_id")
        if region_attr is not None and (not isinstance(region_attr, StringAttr) or not region_attr.data):
            raise LinalgBridgeError("batched precision inventory has an invalid source region ID")
        region_id = region_attr.data if region_attr is not None else f"batch_matmul#{len(regions)}"
        if any(region.region_id == region_id for region in regions):
            raise LinalgBridgeError(f"duplicate source batched matmul region {region_id!r}")
        regions.append(SourceBatchedMatmulRegion(
            region_id, ordinal, result_shape[:-2], lhs_shape, rhs_shape, result_shape,
            source(lhs), source(rhs), source(op.results[0]), scalar_add_order,
        ))
    return tuple(regions), tuple(refused)


def check_source_bound_f32_batched_matmuls(
    program: str,
    *,
    entry: str,
    policy: dict,
    original_source_sha256: str,
) -> tuple[SourceBatchedMatmulRegion, ...]:
    """Match every batched contraction to one explicit precision decision."""
    if not isinstance(policy, dict) or policy.get("source_sha256") != original_source_sha256:
        raise LinalgBridgeError("batched precision policy differs from captured source identity")
    decisions = policy.get("batched_matmul_regions")
    if not isinstance(decisions, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, dict)
        for key, value in decisions.items()
    ):
        raise LinalgBridgeError("batched precision policy needs one decision object per region")
    regions = inventory_source_f32_batched_matmuls(program, entry=entry)
    source_ids = {region.region_id for region in regions}
    if source_ids != decisions.keys():
        missing = sorted(source_ids - decisions.keys())
        extra = sorted(decisions.keys() - source_ids)
        raise LinalgBridgeError(f"batched precision policy regions differ: missing={missing}, extra={extra}")
    return regions
