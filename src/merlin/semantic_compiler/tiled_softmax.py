"""Decompose an explicitly selected row-softmax policy across column tiles.

The selected policy defines the rounding at every primitive, including the
left-to-right combination of tile sums. This is not a reassociation rule for
an arbitrary floating-point reduction. Target layout and instructions remain
in the selected provider; descriptors come from its existing catalog.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

from merlin.runtime.fp8_formats import float_format_params, storage_bits

from .model import ConstantBinding, KernelRequest, SemanticNode, TensorType
from .rules import InstructionDescriptor

_ARITIES = {
    "row_max": 1,
    "combine_max": 2,
    "subtract": 2,
    "exp": 1,
    "row_sum": 1,
    "combine_sum": 2,
    "reciprocal": 1,
    "multiply": 2,
}


@dataclass(frozen=True)
class RowSoftmaxRegion:
    row_start: int
    valid_rows: int
    request: KernelRequest


@dataclass(frozen=True)
class RowSoftmaxTiling:
    source: KernelRequest
    tile_type: TensorType
    regions: tuple[RowSoftmaxRegion, ...]
    pad_element: bytes
    policy: str
    descriptor_records: tuple[tuple[str, str], ...]
    empty_rows_zero: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        return self.source.nodes[0].type.shape

    @property
    def masked(self) -> bool:
        return self.source.nodes[-1].op == "masked_softmax"

    @property
    def column_tiles(self) -> int:
        return (self.shape[1] + self.tile_type.shape[1] - 1) // self.tile_type.shape[1]

    def record(self) -> dict:
        record = {
            "schema": "merlin.row_softmax_tiling.v1",
            "source_request_digest": self.source.digest(),
            "policy": self.policy,
            "tile_type": self.tile_type.record(),
            "column_tiles": self.column_tiles,
            "pad_element_hex": self.pad_element.hex(),
            "tile_max_order": "left_to_right",
            "tile_sum_order": "left_to_right",
            "normalization_scope": "all_valid_columns_in_each_source_row",
            "descriptors": dict(self.descriptor_records),
            "regions": [
                {
                    "row_start": region.row_start,
                    "valid_rows": region.valid_rows,
                    "request_digest": region.request.digest(),
                }
                for region in self.regions
            ],
        }
        if self.masked:
            record.update(
                schema="merlin.row_softmax_tiling.v2",
                mask_encoding="same-dtype-zero-or-one",
                mask_application="multiply-exponentials-before-global-row-sum",
            )
        if self.empty_rows_zero:
            record.update(
                schema="merlin.row_softmax_tiling.v3",
                all_masked_rows="positive-zero",
                denominator_floor="one-before-reciprocal",
            )
        return record


def decompose_row_softmax(
    source: KernelRequest,
    *,
    tile_type: TensorType,
    operations: Mapping[str, InstructionDescriptor],
    policy: str,
    pad_element: bytes,
    empty_rows_zero: bool = False,
) -> RowSoftmaxTiling:
    """Use one global maximum and denominator for every row, including N tails.

    The provider qualifies pad_element as no greater than any valid score.
    Masks are explicit constants after exponentiation, so padded lanes cannot
    contribute to the denominator. Invalid output lanes are trimmed separately.
    """
    masked = source.nodes[-1].op == "masked_softmax"
    if type(empty_rows_zero) is not bool or empty_rows_zero and not masked:
        raise ValueError("zero empty rows require an explicitly masked softmax")
    if (
        len(source.nodes) != (3 if masked else 2)
        or source.constants
        or source.physical_extents
        or source.boundary_layouts
        or source.output_layouts
        or len(source.outputs) != 1
        or len(tile_type.shape) != 2
        or tile_type.dtype not in {"f32", "bf16"}
        or not policy
        or set(operations) != set(_ARITIES)
    ):
        raise ValueError("row softmax needs one plain source and a complete primitive catalog")
    scores, result = source.nodes[0], source.nodes[-1]
    keep = source.nodes[1] if masked else None
    (storage,) = source.output_storages
    if (
        scores.op != "input"
        or scores.effect != "input"
        or scores.index_maps
        or result.op != ("masked_softmax" if masked else "softmax")
        or result.effect != "pure"
        or result.inputs != ((scores.id, keep.id) if masked else (scores.id,))
        or result.attrs != (("axis", 1),)
        or result.index_maps
        or result.type.shape != scores.type.shape
        or len(scores.type.shape) != 2
        or scores.type.dtype != tile_type.dtype
        or result.type.dtype != tile_type.dtype
        or scores.type.numerical_policy != tile_type.numerical_policy
        or result.type.numerical_policy != policy
        or source.outputs != (result.id,)
        or source.input_storages != tuple((n.id, storage) for n in source.nodes[:-1])
        or (masked and (keep.op != "input" or keep.effect != "input" or keep.index_maps or keep.type != scores.type))
    ):
        raise ValueError("row softmax source, axis, policy or boundary differs")
    width = storage_bits(tile_type.dtype) // 8
    if type(pad_element) is not bytes or len(pad_element) != width:
        raise ValueError("row softmax padding differs from the element width")
    _, mantissa_bits, bias, _ = float_format_params(tile_type.dtype)
    mask_zero = bytes(width)
    mask_one = (bias << mantissa_bits).to_bytes(width, "little")
    for role, descriptor in operations.items():
        if (
            not isinstance(descriptor, InstructionDescriptor)
            or len(descriptor.input_storages) != _ARITIES[role]
            or descriptor.output_dtype != tile_type.dtype
            or descriptor.numerical_policy != tile_type.numerical_policy
            or any(dtype != tile_type.dtype for dtype in descriptor.input_dtypes)
            or any(numerics != tile_type.numerical_policy for numerics in descriptor.input_numerical_policies)
            or not descriptor.index_maps
            or not descriptor.required_attrs
            or descriptor.parameter_attrs
            or descriptor.bundle_name
        ):
            raise ValueError("row softmax primitive has an incompatible signature")
    m, n = scores.type.shape
    tm, tn = tile_type.shape
    count = (n + tn - 1) // tn
    descriptor_records = tuple(
        (role, sha256(json.dumps(descriptor.record(), sort_keys=True).encode()).hexdigest())
        for role, descriptor in sorted(operations.items())
    )
    regions = []
    for row_start in range(0, m, tm):
        nodes, constants = [], []

        def operation(role, children):
            descriptor = operations[role]
            name = f"v{len(nodes)}"
            nodes.append(
                SemanticNode(
                    name,
                    descriptor.computation,
                    tuple(children),
                    tile_type,
                    descriptor.required_attrs,
                    index_maps=descriptor.index_maps,
                )
            )
            return name

        inputs = []
        for col in range(count):
            name = f"column_{col}"
            nodes.append(
                SemanticNode(name, "input", (), tile_type, (operations["subtract"].required_attrs[0],), "input")
            )
            inputs.append(name)
        masks = []
        if masked:
            for col in range(count):
                name = f"keep_{col}"
                nodes.append(
                    SemanticNode(name, "input", (), tile_type, (operations["multiply"].required_attrs[0],), "input")
                )
                masks.append(name)
        maxima = [operation("row_max", (value,)) for value in inputs]
        maximum = maxima[0]
        for partial in maxima[1:]:
            maximum = operation("combine_max", (maximum, partial))
        exponentials, sums = [], []
        for col, value in enumerate(inputs):
            shifted = operation("subtract", (value, maximum))
            exponential = operation("exp", (shifted,))
            if masked:
                exponential = operation("multiply", (exponential, masks[col]))
            elif col == count - 1 and n % tn:
                mask = b"".join(
                    (mask_one if column < n % tn else mask_zero) for _row in range(tm) for column in range(tn)
                )
                name = f"mask_{col}"
                nodes.append(
                    SemanticNode(
                        name, "constant", (), tile_type, (operations["multiply"].required_attrs[0],), "constant"
                    )
                )
                constants.append(ConstantBinding(name, tile_type.dtype + "-le-raw", mask.hex()))
                exponential = operation("multiply", (exponential, name))
            exponentials.append(exponential)
            sums.append(operation("row_sum", (exponential,)))
        denominator = sums[0]
        for partial in sums[1:]:
            denominator = operation("combine_sum", (denominator, partial))
        if empty_rows_zero:
            # Nonempty rows have an exp(0)=1 lane. Empty rows have zero sum.
            # Flooring by one therefore preserves the former and keeps the
            # latter inside the reciprocal instruction's positive domain.
            one = "denominator_one"
            nodes.append(
                SemanticNode(one, "constant", (), tile_type, operations["combine_max"].required_attrs[:1], "constant")
            )
            constants.append(ConstantBinding(one, tile_type.dtype + "-le-raw", (mask_one * (tm * tn)).hex()))
            denominator = operation("combine_max", (denominator, one))
        reciprocal = operation("reciprocal", (denominator,))
        outputs = tuple(operation("multiply", (exponential, reciprocal)) for exponential in exponentials)
        identity = sha256(
            json.dumps(
                {"source": source.digest(), "row": row_start, "descriptors": descriptor_records}, sort_keys=True
            ).encode()
        ).hexdigest()
        request = KernelRequest(
            tuple(nodes),
            outputs,
            (storage,) * count,
            tuple((name, storage) for name in (*inputs, *masks)),
            source.target_identity,
            source.lowering_policy,
            constants=tuple(constants),
            source_identity=identity,
        )
        regions.append(RowSoftmaxRegion(row_start, min(tm, m - row_start), request))
    return RowSoftmaxTiling(source, tile_type, tuple(regions), pad_element, policy, descriptor_records, empty_rows_zero)


def materialize_row_softmax_inputs(
    plan: RowSoftmaxTiling, source: bytes, keep: bytes | None = None
) -> tuple[dict[str, bytes], ...]:
    """Pack logical tiles; the provider applies its physical lane layout."""
    m, n = plan.shape
    tm, tn = plan.tile_type.shape
    width = len(plan.pad_element)
    if type(source) is not bytes or len(source) != m * n * width:
        raise ValueError("row softmax input bytes differ from the source shape")
    if plan.masked:
        if type(keep) is not bytes or len(keep) != len(source):
            raise ValueError("masked softmax needs a complete logical keep tensor")
        _, mantissa_bits, bias, _ = float_format_params(plan.tile_type.dtype)
        one = (bias << mantissa_bits).to_bytes(width, "little")
        zero = bytes(width)
        if any(keep[at : at + width] not in (zero, one) for at in range(0, len(keep), width)):
            raise ValueError("softmax keep values must be exact zero or one")
        if not plan.empty_rows_zero and any(
            not any(keep[(row * n + col) * width : (row * n + col + 1) * width] == one for col in range(n))
            for row in range(m)
        ):
            raise ValueError("softmax row has no kept element")
    elif keep is not None:
        raise ValueError("unmasked softmax does not accept a keep tensor")
    result = []
    for region in plan.regions:
        packed = {}
        for col in range(plan.column_tiles):
            tile = bytearray(plan.pad_element * (tm * tn))
            for row in range(region.valid_rows):
                count = min(tn, n - col * tn) * width
                start = ((region.row_start + row) * n + col * tn) * width
                tile[row * tn * width : row * tn * width + count] = source[start : start + count]
            packed[f"column_{col}"] = bytes(tile)
            if plan.masked:
                tile = bytearray(zero * (tm * tn))
                for row in range(tm):
                    for column in range(min(tn, n - col * tn)):
                        at = ((region.row_start + row) * n + col * tn + column) * width
                        value = keep[at : at + width] if row < region.valid_rows else one
                        start = (row * tn + column) * width
                        tile[start : start + width] = value
                packed[f"keep_{col}"] = bytes(tile)
        result.append(packed)
    return tuple(result)


def assemble_row_softmax_outputs(plan: RowSoftmaxTiling, outputs: Sequence[Sequence[bytes]]) -> bytes:
    """Join complete tile groups after the provider removes physical layout."""
    m, n = plan.shape
    tm, tn = plan.tile_type.shape
    width = len(plan.pad_element)
    if len(outputs) != len(plan.regions) or any(len(group) != plan.column_tiles for group in outputs):
        raise ValueError("row softmax output groups omit or add tiles")
    result = bytearray(m * n * width)
    for region, group in zip(plan.regions, outputs, strict=True):
        for col, tile in enumerate(group):
            if type(tile) is not bytes or len(tile) != tm * tn * width:
                raise ValueError("row softmax output tile has the wrong byte extent")
            for row in range(region.valid_rows):
                count = min(tn, n - col * tn) * width
                start = ((region.row_start + row) * n + col * tn) * width
                result[start : start + count] = tile[row * tn * width : row * tn * width + count]
    return bytes(result)
