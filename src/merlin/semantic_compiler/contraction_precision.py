"""Bind a parsed semantic contraction to an explicit target precision profile.

This is the graph counterpart of source-bound Linalg tile conversion. It keeps
the complete contraction for the existing outer tiler, and records a precision
change rather than adding an equality-saturation rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from .model import IndexMap, KernelRequest, SemanticNode, TensorType
from .precision_intake import SourceValueRef
from .precision_transform import _SOURCE_MAPS, BoundaryConversion, MatmulTileProfile, _permutation
from .precision_values import contraction_restore_exponent


@dataclass(frozen=True)
class SemanticContractionPrecision:
    source: KernelRequest
    profile: MatmulTileProfile
    region_id: str
    request: KernelRequest
    inputs: tuple[BoundaryConversion, BoundaryConversion]
    output: BoundaryConversion

    def record(self) -> dict:
        return {
            "schema": "merlin.semantic_contraction_precision.v1",
            "source_request_digest": self.source.digest(),
            "region_id": self.region_id,
            "request_digest": self.request.digest(),
            "target_profile": self.profile.record(),
            "inputs": [item.record() for item in self.inputs],
            "output": self.output.record(),
            "qualification": "diagnostic_precision_change; original_model_quality_unmeasured",
        }


def lower_semantic_contraction(
    source: KernelRequest,
    *,
    profile: MatmulTileProfile,
) -> SemanticContractionPrecision:
    """Check a closed rank-two f32 multiply/add contraction with a +0 seed.

    SourceValueRef(kind="semantic_node", index=i) names node i in the exact
    source request, whose digest accompanies every conversion. It is not an
    ordinal in a printed MLIR function. Constants remain compiler inputs; no
    runtime tensor or reference result is read here.
    """
    computations = [node for node in source.nodes if node.effect == "pure"]
    if len(computations) != 1 or source.lowering_policy != "diagnostic":
        raise ValueError("semantic contraction needs one diagnostic source computation")
    result = computations[0]
    arithmetic = dict(result.attrs)
    scalar_add_order = arithmetic.pop("scalar_add_order", None)
    maps = tuple(IndexMap(3, rows, (0, 0)) for rows in _SOURCE_MAPS)
    if (
        result.op != "source_f32_matmul_unqualified"
        or len(result.inputs) != 3
        or scalar_add_order not in {None, "accumulator-product", "product-accumulator"}
        or arithmetic != {"reduction_order": "unresolved", "numeric_semantics": "source-f32-mul-add-unqualified-v1"}
        or result.index_maps != (*maps, maps[-1])
        or source.outputs != (result.id,)
        or source.target_identity != profile.target_identity
        or source.physical_extents
        or source.boundary_layouts
        or any(spec is None for spec in profile.conversion_specs or (None,))
    ):
        raise ValueError("semantic contraction has unsupported arithmetic, maps or target bindings")
    lhs, rhs, seed = (source.node(key) for key in result.inputs)
    if (
        any(
            node.type.dtype != "f32" or node.type.numerical_policy != "raw-bit-pattern" or len(node.type.shape) != 2
            for node in (lhs, rhs, seed, result)
        )
        or lhs.type.shape[1] != rhs.type.shape[0]
        or result.type.shape != (lhs.type.shape[0], rhs.type.shape[1])
        or seed.type != result.type
        or set(node.id for node in source.nodes) != {lhs.id, rhs.id, seed.id, result.id}
    ):
        raise ValueError("semantic contraction shapes or complete source accounting differ")
    bindings = {item.node_id: item for item in source.constants}
    initialization = bindings.get(seed.id)
    if (
        seed.effect != "constant"
        or seed.op != "constant"
        or seed.attrs
        or seed.index_maps
        or initialization is None
        or initialization.encoding != "f32-le-raw"
        or any(bytes.fromhex(initialization.data_hex))
    ):
        raise ValueError("semantic contraction requires an exact positive-zero initializer")
    permutations = tuple(
        _permutation(index_map, rows) for index_map, rows in zip(profile.maps, _SOURCE_MAPS, strict=True)
    )
    # The existing looped result interface returns logical M-by-N order.
    if permutations[2] != (0, 1):
        raise ValueError("semantic contraction result needs a separate layout conversion")
    source_nodes = {node.id: index for index, node in enumerate(source.nodes)}
    types = tuple(
        TensorType(tuple(node.type.shape[axis] for axis in axes), dtype, policy)
        for node, axes, dtype, policy in zip(
            (lhs, rhs, result),
            permutations,
            (*profile.operand_dtypes, profile.result_dtype),
            (*profile.operand_policies, profile.result_policy),
            strict=True,
        )
    )
    assert profile.conversion_specs is not None
    if profile.conversion_specs[2].post_decode_scale_exponent != contraction_restore_exponent(
        *(spec.pre_encode_scale_exponent for spec in profile.conversion_specs[:2])
    ):
        raise ValueError("contraction result conversion must restore both operand scales")
    inputs = tuple(
        BoundaryConversion(
            role,
            SourceValueRef("semantic_node", source_nodes[node.id]),
            node.type,
            type_,
            axes,
            conversion,
            spec,
        )
        for role, node, type_, axes, conversion, spec in zip(
            ("lhs", "rhs"),
            (lhs, rhs),
            types[:2],
            permutations[:2],
            profile.operand_conversion_policies,
            profile.conversion_specs[:2],
            strict=True,
        )
    )
    output = BoundaryConversion(
        "result",
        SourceValueRef("semantic_node", source_nodes[result.id]),
        result.type,
        types[2],
        permutations[2],
        profile.result_conversion_policy,
        profile.conversion_specs[2],
    )
    identity = sha256(
        json.dumps([source.digest(), profile.record()], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    request = KernelRequest(
        (
            *tuple(
                SemanticNode(item.source_role, "input", (), item.target_type, (("input_domain", domain),), "input")
                for item, domain in zip(inputs, profile.operand_domains, strict=True)
            ),
            SemanticNode("result", "matmul", ("lhs", "rhs"), types[2], index_maps=profile.maps),
        ),
        ("result",),
        (profile.storage,),
        (("lhs", profile.storage), ("rhs", profile.storage)),
        profile.target_identity,
        "diagnostic",
        identity,
    )
    return SemanticContractionPrecision(source, profile, result.id, request, inputs, output)
