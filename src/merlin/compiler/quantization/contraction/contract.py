"""Build an explicit diagnostic target tile from a source-bound contraction.

The source f32 program and the target program have different arithmetic.
Boundary conversion is recorded as work still required of a qualified host or
device route. It is never inserted into an equality-saturation graph as an
equivalence rule.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import Any

from merlin.compiler.semantic.linalg_bridge import LinalgBridgeError
from merlin.compiler.semantic.model import IndexMap, KernelRequest, SemanticNode, TensorType

from .intake import (
    SourceValueRef,
    check_source_bound_f32_batched_matmuls,
    check_source_bound_f32_matmuls,
    inventory_source_f32_batched_matmuls,
)
from .values import FloatBoundarySpec

_SOURCE_MAPS = (
    ((1, 0, 0), (0, 0, 1)),  # lhs[m, k]
    ((0, 0, 1), (0, 1, 0)),  # rhs[k, n]
    ((1, 0, 0), (0, 1, 0)),  # result[m, n]
)


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class MatmulTileProfile:
    """Target-supplied facts for one exact tile shape and arithmetic mode."""

    target_identity: str
    loop_shape: tuple[int, int, int]  # m, n, k
    maps: tuple[IndexMap, IndexMap, IndexMap]
    operand_dtypes: tuple[str, str]
    operand_policies: tuple[str, str]
    operand_domains: tuple[str, str]
    operand_conversion_policies: tuple[str, str]
    result_dtype: str
    result_policy: str
    result_conversion_policy: str
    storage: str
    input_extents: tuple[int, int]
    required_policy_fields: tuple[tuple[str, str | int], ...]
    selected_decision: tuple[tuple[str, str | int], ...]
    optional_policy_fields: tuple[str, ...] = ()
    conversion_specs: tuple[FloatBoundarySpec, FloatBoundarySpec, FloatBoundarySpec] | None = None

    def __post_init__(self) -> None:
        if (
            not self.target_identity
            or not self.storage
            or len(self.loop_shape) != 3
            or any(type(size) is not int or size <= 0 for size in self.loop_shape)
            or len(self.maps) != 3
            or any(not isinstance(index_map, IndexMap) for index_map in self.maps)
            or any(
                len(row) != 2
                for row in (
                    self.operand_dtypes,
                    self.operand_policies,
                    self.operand_domains,
                    self.operand_conversion_policies,
                )
            )
            or len(self.input_extents) != 2
            or any(type(extent) is not int or extent <= 0 for extent in self.input_extents)
            or not self.result_dtype
            or not self.result_policy
            or not self.result_conversion_policy
            or any(
                not isinstance(value, str) or not value
                for row in (
                    self.operand_dtypes,
                    self.operand_policies,
                    self.operand_domains,
                    self.operand_conversion_policies,
                )
                for value in row
            )
        ):
            raise ValueError("target matmul tile facts are incomplete")
        for facts in (self.required_policy_fields, self.selected_decision):
            if (
                not facts
                or len(dict(facts)) != len(facts)
                or any(not isinstance(key, str) or not key or type(value) not in (str, int) for key, value in facts)
            ):
                raise ValueError("target precision requirements need unique scalar fields")
        if (
            len(set(self.optional_policy_fields)) != len(self.optional_policy_fields)
            or any(not isinstance(key, str) or not key for key in self.optional_policy_fields)
            or set(self.optional_policy_fields) & {key for key, _ in self.required_policy_fields}
        ):
            raise ValueError("optional target precision fields must be distinct names")
        if self.conversion_specs is not None:
            if len(self.conversion_specs) != 3 or any(
                not isinstance(spec, FloatBoundarySpec)
                or spec.policy != policy
                or spec.direction != direction
                or spec.format != dtype
                for spec, policy, direction, dtype in zip(
                    self.conversion_specs,
                    (*self.operand_conversion_policies, self.result_conversion_policy),
                    ("encode", "encode", "decode"),
                    (*self.operand_dtypes, self.result_dtype),
                    strict=True,
                )
            ):
                raise ValueError("boundary conversion specs differ from the target tile")

    def record(self) -> dict[str, Any]:
        return {
            "target_identity": self.target_identity,
            "loop_shape": list(self.loop_shape),
            "maps": [index_map.record() for index_map in self.maps],
            "operand_dtypes": list(self.operand_dtypes),
            "operand_policies": list(self.operand_policies),
            "operand_domains": list(self.operand_domains),
            "operand_conversion_policies": list(self.operand_conversion_policies),
            "result_dtype": self.result_dtype,
            "result_policy": self.result_policy,
            "result_conversion_policy": self.result_conversion_policy,
            "storage": self.storage,
            "input_extents": list(self.input_extents),
            "required_policy_fields": dict(self.required_policy_fields),
            "selected_decision": dict(self.selected_decision),
            "optional_policy_fields": list(self.optional_policy_fields),
            "conversion_specs": (
                None if self.conversion_specs is None else [spec.record() for spec in self.conversion_specs]
            ),
        }


@dataclass(frozen=True)
class BoundaryConversion:
    source_role: str
    source_value: SourceValueRef
    source_type: TensorType
    target_type: TensorType
    target_axes_from_source: tuple[int, int]
    conversion_policy: str
    spec: FloatBoundarySpec | None = None
    target_offsets: tuple[int, int] = (0, 0)
    source_offsets: tuple[int, int] = (0, 0)
    source_window_shape: tuple[int, int] | None = None
    source_batch_index: tuple[int, ...] = ()

    @property
    def window_shape(self) -> tuple[int, int]:
        return self.source_type.shape[-2:] if self.source_window_shape is None else self.source_window_shape

    def __post_init__(self) -> None:
        window = self.window_shape
        if (
            self.target_axes_from_source not in ((0, 1), (1, 0))
            or len(self.source_type.shape) < 2
            or len(self.target_type.shape) != 2
            or len(self.source_batch_index) != len(self.source_type.shape) - 2
            or any(
                type(index) is not int or not 0 <= index < extent
                for index, extent in zip(
                    self.source_batch_index,
                    self.source_type.shape[:-2],
                    strict=True,
                )
            )
            or len(window) != 2
            or len(self.target_offsets) != 2
            or len(self.source_offsets) != 2
            or any(type(size) is not int or size <= 0 for size in window)
            or any(type(offset) is not int or offset < 0 for offset in self.target_offsets)
            or any(type(offset) is not int or offset < 0 for offset in self.source_offsets)
            or any(
                offset + size > extent
                for offset, size, extent in zip(
                    self.source_offsets,
                    window,
                    self.source_type.shape[-2:],
                    strict=True,
                )
            )
            or any(
                offset + window[source_axis] > extent
                for offset, source_axis, extent in zip(
                    self.target_offsets,
                    self.target_axes_from_source,
                    self.target_type.shape,
                    strict=True,
                )
            )
        ):
            raise ValueError("boundary window lies outside the source or target tensor")

    def convert_values(self, values):
        """Convert a concrete logical tensor; physical storage remains target-owned.

        An encode takes the complete source f32 tensor and selects its window.
        A decode takes target code elements and returns the source window.
        Neither operation implies original-model quality or runtime placement.
        """
        import numpy as np

        if self.spec is None or self.spec.policy != self.conversion_policy:
            raise ValueError("boundary conversion has no matching executable specification")
        if self.source_type.dtype != "f32" or self.target_type.dtype != self.spec.format:
            raise ValueError("boundary types differ from the executable specification")
        if self.spec.direction == "encode":
            if not isinstance(values, np.ndarray) or values.shape != self.source_type.shape:
                raise ValueError("source tensor shape differs from the selected boundary")
            if values.dtype != np.float32:
                raise ValueError("source tensor dtype differs from the selected boundary")
            source_window = self.source_batch_index + tuple(
                slice(offset, offset + size)
                for offset, size in zip(
                    self.source_offsets,
                    self.window_shape,
                    strict=True,
                )
            )
            ordered = np.transpose(values[source_window], self.target_axes_from_source)
            padded = np.zeros(self.target_type.shape, dtype=np.float32)
            window = tuple(
                slice(offset, offset + size)
                for offset, size in zip(
                    self.target_offsets,
                    ordered.shape,
                    strict=True,
                )
            )
            padded[window] = ordered
            return self.spec.encode(padded)
        if not isinstance(values, np.ndarray) or values.shape != self.target_type.shape:
            raise ValueError("target tensor shape differs from the selected boundary")
        decoded = self.spec.decode(values)
        window = tuple(
            slice(offset, offset + self.window_shape[source_axis])
            for offset, source_axis in zip(
                self.target_offsets,
                self.target_axes_from_source,
                strict=True,
            )
        )
        source_axes = tuple(self.target_axes_from_source.index(axis) for axis in range(2))
        ordered = np.transpose(decoded[window], source_axes)
        if ordered.shape != self.window_shape:
            raise ValueError("decoded tensor shape differs from the source window")
        return ordered

    def record(self) -> dict[str, Any]:
        return {
            "source_role": self.source_role,
            "source_value": self.source_value.record(),
            "source_type": self.source_type.record(),
            **({"source_batch_index": list(self.source_batch_index)} if self.source_batch_index else {}),
            "source_offsets": list(self.source_offsets),
            "source_window_shape": list(self.window_shape),
            "target_type": self.target_type.record(),
            "target_axes_from_source": list(self.target_axes_from_source),
            "target_offsets": list(self.target_offsets),
            "padding_value": (
                "positive_zero_f32"
                if self.source_type.dtype == "f32"
                and self.target_type.dtype.startswith("fp8_")
                and tuple(self.window_shape[axis] for axis in self.target_axes_from_source) != self.target_type.shape
                else None
            ),
            "conversion_policy": self.conversion_policy,
            "conversion_spec": None if self.spec is None else self.spec.record(),
            "status": "unmaterialized_boundary_obligation",
        }


@dataclass(frozen=True)
class DiagnosticMatmulTile:
    """Source correspondence and conversions for a tile or full contraction.

    A full contraction retains one semantic reduction. Its backend must still
    choose and check an implementation; the conversion plan grants no capacity.
    """

    request: KernelRequest
    region_id: str
    original_source_sha256: str
    staged_program_sha256: str
    precision_policy_sha256: str
    inputs: tuple[BoundaryConversion, ...]
    output: BoundaryConversion
    tile_origin: tuple[int, int] | None = None
    ordered_k_continuation: bool = False
    batch_index: tuple[int, ...] | None = None
    whole_contraction: bool = False

    def record(self) -> dict[str, Any]:
        return {
            "schema": (
                "merlin.diagnostic_precision_contraction.v1"
                if self.whole_contraction
                else "merlin.diagnostic_precision_tile.v4"
                if self.batch_index is not None
                else "merlin.diagnostic_precision_tile.v3"
                if self.ordered_k_continuation
                else "merlin.diagnostic_precision_tile.v2"
                if self.tile_origin is not None
                else "merlin.diagnostic_precision_tile.v1"
            ),
            "region_id": self.region_id,
            **({"batch_index": list(self.batch_index)} if self.batch_index is not None else {}),
            "tile_origin": None if self.tile_origin is None else list(self.tile_origin),
            "ordered_k_continuation": self.ordered_k_continuation,
            **({"whole_contraction": True} if self.whole_contraction else {}),
            "original_source_sha256": self.original_source_sha256,
            "staged_program_sha256": self.staged_program_sha256,
            "precision_policy_sha256": self.precision_policy_sha256,
            "request_digest": self.request.digest(),
            "inputs": [conversion.record() for conversion in self.inputs],
            "output": self.output.record(),
            "quality_status": "unqualified",
        }
