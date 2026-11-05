"""Separate independent dense batches without changing a contraction's K order."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from .contraction_precision import SemanticContractionPrecision
from .model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType
from .precision_intake import SourceValueRef
from .precision_transform import _SOURCE_MAPS, BoundaryConversion, MatmulTileProfile


@dataclass(frozen=True)
class BatchedContractionPlan:
    source: KernelRequest
    matrix_source: KernelRequest
    batch_shape: tuple[int, ...]
    operand_ids: tuple[str, str]
    result_id: str

    @property
    def input_ids(self):
        return tuple(dict.fromkeys(self.operand_ids))

    @property
    def batch_count(self):
        return prod(self.batch_shape)

    @property
    def matrix_elements(self):
        return tuple(prod(self.source.node(key).type.shape[-2:]) for key in (*self.operand_ids, self.result_id))

    def record(self):
        return {
            "schema": "merlin.dense_batched_contraction.v1",
            "source_request_digest": self.source.digest(),
            "matrix_request_digest": self.matrix_source.digest(),
            "batch_shape": list(self.batch_shape),
            "batch_count": self.batch_count,
            "operand_ids": list(self.operand_ids),
            "result_id": self.result_id,
            "matrix_elements": list(self.matrix_elements),
            "element_order": "row-major-unchanged",
            "batch_order": "lexicographic",
            "reduction_partition": "independent-complete-K-per-batch",
        }


def plan_batched_contraction(source: KernelRequest) -> BatchedContractionPlan:
    """Prove canonical [...,M,K] × [...,K,N] batches with explicit +0 seeds.

    Both operands must carry the same leading dimensions. Broadcasted batches
    need a separate index-map proof. Constants stay in the outer application;
    the reusable matrix body takes two boundary operands and initializes each
    batch independently. No runtime data is used to construct this body.
    """
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1 or len(pure[0].inputs) != 3:
        raise ValueError("batched contraction needs one complete source contraction")
    result = pure[0]
    lhs, rhs, seed = (source.node(key) for key in result.inputs)
    rank = len(result.type.shape)
    if (
        rank < 3
        or source.outputs != (result.id,)
        or source.lowering_policy != "diagnostic"
        or result.op != "source_f32_matmul_unqualified"
        or source.physical_extents
        or source.boundary_layouts
        or source.output_layouts
        or any(
            node.type.dtype != "f32" or node.type.numerical_policy != "raw-bit-pattern" or len(node.type.shape) != rank
            for node in (lhs, rhs, seed, result)
        )
        or any(node.effect not in {"input", "constant"} or node.index_maps or node.attrs for node in (lhs, rhs))
        or {node.id for node in source.nodes} != {*result.inputs, result.id}
    ):
        raise ValueError("batched contraction needs dense diagnostic f32 tensors and complete source accounting")
    batch = result.type.shape[:-2]
    m, k = lhs.type.shape[-2:]
    n = rhs.type.shape[-1]
    if (
        lhs.type.shape != (*batch, m, k)
        or rhs.type.shape != (*batch, k, n)
        or result.type.shape != (*batch, m, n)
        or seed.type != result.type
    ):
        raise ValueError("batched contraction dimensions or batch prefixes differ")
    loops = rank + 1
    axes = tuple(tuple(int(i == j) for j in range(loops)) for i in range(loops))
    prefix = axes[: rank - 2]
    maps = tuple(
        IndexMap(loops, (*prefix, *rows), (0,) * rank)
        for rows in (
            (axes[-3], axes[-1]),
            (axes[-1], axes[-2]),
            (axes[-3], axes[-2]),
        )
    )
    if result.index_maps != (*maps, maps[-1]):
        raise ValueError("batched contraction maps mix batches or change matrix indexing")
    initialization = next((item for item in source.constants if item.node_id == seed.id), None)
    if (
        seed.op != "constant"
        or seed.effect != "constant"
        or seed.attrs
        or seed.index_maps
        or initialization is None
        or initialization.encoding != "f32-le-raw"
        or any(bytes.fromhex(initialization.data_hex))
    ):
        raise ValueError("batched contraction requires an exact positive-zero initializer")
    matrix_maps = tuple(IndexMap(3, rows, (0, 0)) for rows in _SOURCE_MAPS)
    matrix = KernelRequest(
        (
            SemanticNode("lhs", "input", (), replace(lhs.type, shape=(m, k)), effect="input"),
            SemanticNode("rhs", "input", (), replace(rhs.type, shape=(k, n)), effect="input"),
            SemanticNode("seed", "constant", (), replace(seed.type, shape=(m, n)), effect="constant"),
            SemanticNode(
                "product",
                result.op,
                ("lhs", "rhs", "seed"),
                replace(result.type, shape=(m, n)),
                result.attrs,
                index_maps=(*matrix_maps, matrix_maps[-1]),
            ),
        ),
        ("product",),
        ("external",),
        (("lhs", "external"), ("rhs", "external")),
        source.target_identity,
        "diagnostic",
        source.digest() + ":dense-batch-body",
        constants=(ConstantBinding("seed", "f32-le-raw", "00000000" * (m * n)),),
    )
    return BatchedContractionPlan(source, matrix, batch, (lhs.id, rhs.id), result.id)


@dataclass(frozen=True)
class BatchedOperandConversion:
    source_value: SourceValueRef
    source_type: TensorType
    matrix_conversion: BoundaryConversion

    def convert_values(self, values):
        import numpy as np

        if not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.shape != self.source_type.shape:
            raise ValueError("batched operand differs from its declared f32 source")
        batch = self.source_type.shape[:-2]
        matrices = values.reshape(prod(batch), *self.source_type.shape[-2:])
        converted = [self.matrix_conversion.convert_values(matrix) for matrix in matrices]
        return np.stack(converted).reshape(*batch, *self.matrix_conversion.target_type.shape)

    def record(self):
        return {
            "source_value": self.source_value.record(),
            "source_type": self.source_type.record(),
            "batch_order": "lexicographic",
            "matrix_conversion": self.matrix_conversion.record(),
        }


@dataclass(frozen=True)
class BatchedContractionPrecision:
    plan: BatchedContractionPlan
    matrix: SemanticContractionPrecision

    @property
    def request(self):
        return self.matrix.request

    @property
    def inputs(self):
        return tuple(
            BatchedOperandConversion(
                SourceValueRef("semantic_node", self.plan.source.nodes.index(self.plan.source.node(key))),
                self.plan.source.node(key).type,
                conversion,
            )
            for key, conversion in zip(self.plan.operand_ids, self.matrix.inputs, strict=True)
        )

    def record(self):
        return {
            "schema": "merlin.batched_contraction_precision.v1",
            "source_request_digest": self.plan.source.digest(),
            "batching": self.plan.record(),
            "matrix_precision": self.matrix.record(),
            "inputs": [item.record() for item in self.inputs],
            "qualification": "diagnostic_precision_change; original_model_quality_unmeasured",
        }


def lower_batched_contraction(source: KernelRequest, *, profile: MatmulTileProfile) -> BatchedContractionPrecision:
    from .contraction_precision import lower_semantic_contraction

    plan = plan_batched_contraction(source)
    return BatchedContractionPrecision(plan, lower_semantic_contraction(plan.matrix_source, profile=profile))
