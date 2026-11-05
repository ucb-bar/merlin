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

from .linalg_bridge import LinalgBridgeError
from .model import IndexMap, KernelRequest, SemanticNode, TensorType
from .precision_intake import (
    SourceValueRef,
    check_source_bound_f32_batched_matmuls,
    check_source_bound_f32_matmuls,
    inventory_source_f32_batched_matmuls,
)
from .precision_values import FloatBoundarySpec

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


def _permutation(index_map: IndexMap, source_rows: tuple[tuple[int, ...], ...]) -> tuple[int, int]:
    if (
        index_map.loop_rank != 3
        or index_map.offsets != (0, 0)
        or len(index_map.coefficients) != 2
        or set(index_map.coefficients) != set(source_rows)
    ):
        raise LinalgBridgeError("target matmul map needs a zero-offset permutation of source axes")
    return tuple(source_rows.index(row) for row in index_map.coefficients)


def build_diagnostic_matmul_tile(
    program: str,
    *,
    entry: str,
    policy_document: bytes,
    original_source_sha256: str,
    region_id: str,
    profile: MatmulTileProfile,
    tile_origin: tuple[int, int] | None = None,
    ordered_k_continuation: bool = False,
    batch_index: tuple[int, ...] | None = None,
    whole_contraction: bool = False,
) -> DiagnosticMatmulTile:
    """Create a target request and checked, unmaterialized conversion plan.

    A caller must separately qualify conversions and original-model quality.
    Only diagnostic mode is emitted because those obligations are open.
    """
    if not isinstance(policy_document, bytes):
        raise LinalgBridgeError("precision policy document must be exact bytes")
    if (
        not isinstance(original_source_sha256, str)
        or len(original_source_sha256) != 64
        or any(char not in "0123456789abcdef" for char in original_source_sha256)
    ):
        raise LinalgBridgeError("original source identity must be a canonical SHA-256")
    if not isinstance(profile, MatmulTileProfile) or not isinstance(region_id, str) or not region_id:
        raise LinalgBridgeError("a selected target tile and source region are required")

    def unique_object(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise LinalgBridgeError("precision policy has duplicate JSON fields")
        return result

    try:
        policy = json.loads(policy_document, object_pairs_hook=unique_object)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LinalgBridgeError("precision policy document is invalid JSON") from exc
    if not isinstance(policy, dict):
        raise LinalgBridgeError("precision policy document must be an object")
    precision_policy_sha256 = sha256(policy_document).hexdigest()
    named_regions = check_source_bound_f32_matmuls(
        program,
        entry=entry,
        policy=policy,
        original_source_sha256=original_source_sha256,
    )
    batched_regions = (
        check_source_bound_f32_batched_matmuls(
            program,
            entry=entry,
            policy=policy,
            original_source_sha256=original_source_sha256,
        )
        if "batched_matmul_regions" in policy or batch_index is not None
        else ()
    )
    if {item.region_id for item in named_regions} & {item.region_id for item in batched_regions}:
        raise LinalgBridgeError("source contraction IDs overlap across named and batched regions")
    if batch_index is None:
        region = next((item for item in named_regions if item.region_id == region_id), None)
        decision_key = "matmul_regions"
    else:
        region = next((item for item in batched_regions if item.region_id == region_id), None)
        decision_key = "batched_matmul_regions"
    if region is None:
        raise LinalgBridgeError(f"unknown source matmul region {region_id!r}")
    required_fields = {"source_sha256", "matmul_regions", *(key for key, _ in profile.required_policy_fields)}
    if "batched_matmul_regions" in policy or batch_index is not None:
        required_fields.add("batched_matmul_regions")
    if not required_fields <= set(policy) <= required_fields | set(profile.optional_policy_fields):
        raise LinalgBridgeError("source precision policy has missing or unknown fields")
    for key, value in profile.required_policy_fields:
        if policy.get(key) != value:
            raise LinalgBridgeError(f"selected target precision field {key!r} differs")
    if policy[decision_key][region_id] != dict(profile.selected_decision):
        raise LinalgBridgeError("source matmul decision differs from selected target tile")
    if batch_index is not None and (
        not isinstance(batch_index, tuple)
        or len(batch_index) != len(region.batch_shape)
        or any(
            type(index) is not int or not 0 <= index < extent
            for index, extent in zip(
                batch_index,
                region.batch_shape,
                strict=True,
            )
        )
    ):
        raise LinalgBridgeError("source batch index is outside the contraction")
    source_m, source_k = region.lhs_shape[-2:]
    _, source_n = region.rhs_shape[-2:]
    if type(whole_contraction) is not bool or (whole_contraction and tile_origin is not None):
        raise LinalgBridgeError("whole-contraction lowering cannot select an output tile origin")
    # Preserve a single semantic reduction. The backend, rather than this
    # boundary transform, chooses compact loops or another legal implementation.
    m, n, k = (source_m, source_n, source_k) if whole_contraction else profile.loop_shape
    if (source_k > k and not ordered_k_continuation) or (tile_origin is None and (source_m > m or source_n > n)):
        raise LinalgBridgeError("source contraction shape exceeds selected target tile")
    if tile_origin is None:
        m_start, n_start = 0, 0
    else:
        if (
            not isinstance(tile_origin, tuple)
            or len(tile_origin) != 2
            or any(type(value) is not int or value < 0 for value in tile_origin)
            or tile_origin[0] % m
            or tile_origin[1] % n
            or tile_origin[0] >= source_m
            or tile_origin[1] >= source_n
        ):
            raise LinalgBridgeError("source tile origin is unaligned or outside the contraction")
        m_start, n_start = tile_origin
    valid_m, valid_n = min(m, source_m - m_start), min(n, source_n - n_start)
    source_shapes = (region.lhs_shape, region.rhs_shape, region.result_shape)
    continuation = source_k > k
    first_k_count = source_k % k or k
    k_segments = (
        ((0, first_k_count), *((start, k) for start in range(first_k_count, source_k, k)))
        if continuation
        else ((0, source_k),)
    )
    permutations = tuple(
        _permutation(index_map, source_rows) for index_map, source_rows in zip(profile.maps, _SOURCE_MAPS, strict=True)
    )
    full_source_shapes = ((m, k), (k, n), (m, n))
    target_shapes = tuple(
        tuple(shape[axis] for axis in permutation)
        for shape, permutation in zip(full_source_shapes, permutations, strict=True)
    )
    source_types = tuple(TensorType(shape, "f32", "raw-bit-pattern") for shape in source_shapes)
    target_types = (
        TensorType(target_shapes[0], profile.operand_dtypes[0], profile.operand_policies[0]),
        TensorType(target_shapes[1], profile.operand_dtypes[1], profile.operand_policies[1]),
        TensorType(target_shapes[2], profile.result_dtype, profile.result_policy),
    )
    specs = profile.conversion_specs or (None, None, None)
    inputs = tuple(
        BoundaryConversion(
            (f"{role}__k{k_index}" if continuation else role),
            (region.lhs_source, region.rhs_source)[operand],
            source_types[operand],
            target_types[operand],
            permutations[operand],
            profile.operand_conversion_policies[operand],
            specs[operand],
            # A short first K segment must be head padded. Its extra products
            # precede, rather than follow, the source reduction order.
            tuple(k - k_count if axis == (1 if operand == 0 else 0) else 0 for axis in permutations[operand]),
            (m_start, k_start) if operand == 0 else (k_start, n_start),
            (valid_m, k_count) if operand == 0 else (k_count, valid_n),
            () if batch_index is None else batch_index,
        )
        for k_index, (k_start, k_count) in enumerate(k_segments)
        for operand, role in enumerate(("lhs", "rhs"))
    )
    output = BoundaryConversion(
        "result",
        region.result_source,
        source_types[2],
        target_types[2],
        permutations[2],
        profile.result_conversion_policy,
        specs[2],
        source_offsets=(m_start, n_start),
        source_window_shape=(valid_m, valid_n),
        source_batch_index=() if batch_index is None else batch_index,
    )
    staged_program_sha256 = sha256(program.encode()).hexdigest()
    identity_fields = {
        "original_source_sha256": original_source_sha256,
        "staged_program_sha256": staged_program_sha256,
        "precision_policy_sha256": precision_policy_sha256,
        "region_id": region_id,
        "target_profile": profile.record(),
    }
    if tile_origin is not None:
        identity_fields["tile_origin"] = list(tile_origin)
    if batch_index is not None:
        identity_fields["batch_index"] = list(batch_index)
    if continuation:
        identity_fields["ordered_k_segments"] = [list(segment) for segment in k_segments]
    if whole_contraction:
        identity_fields["whole_contraction"] = True
    source_identity = _digest(identity_fields)
    input_nodes = tuple(
        SemanticNode(
            item.source_role,
            "input",
            (),
            item.target_type,
            attrs=(("input_domain", profile.operand_domains[index % 2]),),
            effect="input",
        )
        for index, item in enumerate(inputs)
    )
    reductions = []
    for index in range(len(k_segments)):
        children = (inputs[2 * index].source_role, inputs[2 * index + 1].source_role)
        if index:
            children += (reductions[-1].id,)
        reductions.append(
            SemanticNode(
                "result" if index == len(k_segments) - 1 else f"result__k{index}",
                "matmul" if index == 0 else "matmul_accumulate",
                children,
                target_types[2],
                index_maps=(profile.maps if index == 0 else (*profile.maps, profile.maps[2])),
            )
        )
    request = KernelRequest(
        (*input_nodes, *reductions),
        ("result",),
        (profile.storage,),
        tuple((item.source_role, profile.storage) for item in inputs),
        profile.target_identity,
        lowering_policy="diagnostic",
        source_identity=source_identity,
        physical_extents=()
        if whole_contraction
        else tuple((item.source_role, profile.input_extents[index % 2]) for index, item in enumerate(inputs)),
    )
    return DiagnosticMatmulTile(
        request,
        region_id,
        original_source_sha256,
        staged_program_sha256,
        precision_policy_sha256,
        inputs,
        output,
        tile_origin,
        continuation,
        batch_index,
        whole_contraction,
    )


def build_diagnostic_matmul_tiles(
    program: str,
    *,
    entry: str,
    policy_document: bytes,
    original_source_sha256: str,
    region_id: str,
    profile: MatmulTileProfile,
    ordered_k_continuation: bool = False,
    batch_index: tuple[int, ...] | None = None,
) -> tuple[DiagnosticMatmulTile, ...]:
    """Partition independent M/N output windows under one selected tile profile.

    K continuation and cross-tile model composition are separate obligations.
    The first tile validates the source, policy, and target before enumeration.
    """
    args = dict(
        entry=entry,
        policy_document=policy_document,
        original_source_sha256=original_source_sha256,
        region_id=region_id,
        profile=profile,
        ordered_k_continuation=ordered_k_continuation,
        batch_index=batch_index,
    )
    first = build_diagnostic_matmul_tile(program, **args, tile_origin=(0, 0))
    source_m = first.inputs[0].source_type.shape[-2]
    source_n = first.inputs[1].source_type.shape[-1]
    tile_m, tile_n, _ = profile.loop_shape
    if source_m <= tile_m and source_n <= tile_n:
        return (build_diagnostic_matmul_tile(program, **args),)
    return tuple(
        first
        if m_start == n_start == 0
        else build_diagnostic_matmul_tile(
            program,
            **args,
            tile_origin=(m_start, n_start),
        )
        for m_start in range(0, source_m, tile_m)
        for n_start in range(0, source_n, tile_n)
    )


def build_diagnostic_batched_matmul_tiles(
    program: str,
    *,
    entry: str,
    policy_document: bytes,
    original_source_sha256: str,
    region_id: str,
    profile: MatmulTileProfile,
    ordered_k_continuation: bool = False,
) -> tuple[DiagnosticMatmulTile, ...]:
    """Enumerate every batch and M/N window of one declared contraction."""
    regions = inventory_source_f32_batched_matmuls(program, entry=entry)
    region = next((item for item in regions if item.region_id == region_id), None)
    if region is None:
        raise LinalgBridgeError(f"unknown source batched matmul region {region_id!r}")
    return tuple(
        tile
        for index in product(*(range(extent) for extent in region.batch_shape))
        for tile in build_diagnostic_matmul_tiles(
            program,
            entry=entry,
            policy_document=policy_document,
            original_source_sha256=original_source_sha256,
            region_id=region_id,
            profile=profile,
            ordered_k_continuation=ordered_k_continuation,
            batch_index=index,
        )
    )


def assemble_diagnostic_matmul_outputs(
    plans: Sequence[DiagnosticMatmulTile],
    windows: Sequence[Any],
):
    """Place decoded M/N windows into one source-shaped diagnostic result.

    This only joins already decoded target-program results. It does not assert
    equality with the source f32 contraction or qualify its precision policy.
    """
    import numpy as np

    if not plans or len(plans) != len(windows):
        raise ValueError("diagnostic output windows need matching nonempty plans and values")
    first = plans[0]
    if not isinstance(first, DiagnosticMatmulTile):
        raise ValueError("diagnostic output window has no tile plan")
    identity = (
        first.region_id,
        first.original_source_sha256,
        first.staged_program_sha256,
        first.precision_policy_sha256,
        first.request.target_identity,
        first.output.source_value,
        first.output.source_type,
        first.output.target_type,
        first.output.target_axes_from_source,
        first.output.conversion_policy,
        first.output.spec,
    )
    shape = first.output.source_type.shape
    output = np.empty(shape, dtype=np.float32)
    covered = np.zeros(shape, dtype=np.bool_)
    for plan, values in zip(plans, windows, strict=True):
        if (
            not isinstance(plan, DiagnosticMatmulTile)
            or (
                plan.region_id,
                plan.original_source_sha256,
                plan.staged_program_sha256,
                plan.precision_policy_sha256,
                plan.request.target_identity,
                plan.output.source_value,
                plan.output.source_type,
                plan.output.target_type,
                plan.output.target_axes_from_source,
                plan.output.conversion_policy,
                plan.output.spec,
            )
            != identity
        ):
            raise ValueError("diagnostic output windows come from different sources or policies")
        origin = (0, 0) if plan.tile_origin is None else plan.tile_origin
        batch = () if plan.batch_index is None else plan.batch_index
        if (
            len(batch) != len(shape) - 2
            or batch != plan.output.source_batch_index
            or any(item.source_batch_index != batch for item in plan.inputs)
        ):
            raise ValueError("diagnostic output window has inconsistent batch coordinates")
        if origin != plan.output.source_offsets or (plan.tile_origin is None and len(plans) != 1 and not batch):
            raise ValueError("diagnostic output window origin differs from its tile plan")
        tile_shape = tuple(
            plan.output.target_type.shape[plan.output.target_axes_from_source.index(axis)] for axis in range(2)
        )
        if any(start % extent for start, extent in zip(origin, tile_shape, strict=True)) or (
            plan.output.window_shape
            != tuple(
                min(extent, total - start) for start, extent, total in zip(origin, tile_shape, shape[-2:], strict=True)
            )
        ):
            raise ValueError("diagnostic output window is not a canonical source tile")
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.float32
            or (values.shape != plan.output.window_shape)
        ):
            raise ValueError("diagnostic output window shape or dtype differs from its plan")
        selected = batch + tuple(
            slice(start, start + size)
            for start, size in zip(
                origin,
                plan.output.window_shape,
                strict=True,
            )
        )
        if covered[selected].any():
            raise ValueError("diagnostic output windows overlap")
        output[selected] = values
        covered[selected] = True
    if not covered.all():
        raise ValueError("diagnostic output windows leave source elements unwritten")
    return output
