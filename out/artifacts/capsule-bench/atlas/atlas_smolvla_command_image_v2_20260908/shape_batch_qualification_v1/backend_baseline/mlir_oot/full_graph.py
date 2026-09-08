"""Conservative full-graph inventory and proposed Atlas partition boundaries.

This module does not lower a model.  It turns the verified capture into an
auditable inventory whose boundary policy is intentionally fail-closed: only
semantics with an existing isolated emitter are device candidates, simple
shape/view regions may bridge adjacent candidates, and everything else cuts a
partition.
"""
import re
from collections import Counter, OrderedDict

from xdsl.dialects.builtin import TensorType

from .frontend import _str_attr


MESH_CANDIDATES = frozenset({"matmul", "addmm", "batch_matmul"})
VECTOR_CANDIDATES = frozenset(
    {"add", "gelu", "softmax", "reduce_sum", "layer_norm"}
)
LAYOUT_BRIDGES = frozenset({"view", "unsqueeze", "expand", "copy"})


_TENSOR_ELEMENT = re.compile(r"tensor<.*x([^x<>]+)>")


def _tensor_dtype(value) -> str | None:
    match = _TENSOR_ELEMENT.fullmatch(str(value.type))
    return match.group(1) if match else None


def _tensor_shape(value) -> tuple[int, ...] | None:
    if not isinstance(value.type, TensorType):
        return None
    return tuple(int(v) for v in value.type.get_shape())


def _partition_class(
    semantic: str,
    tensor_dtypes: set[str],
    tensor_shapes: set[tuple[int, ...]],
) -> tuple[str, str]:
    if semantic in MESH_CANDIDATES:
        candidate = "mesh_candidate"
        reason = "an isolated Atlas mesh emitter exists"
    elif semantic in VECTOR_CANDIDATES:
        candidate = "vector_candidate"
        reason = "an isolated Atlas BF16 vector emitter exists"
    elif semantic in LAYOUT_BRIDGES:
        return (
            "layout_bridge_candidate",
            "shape-equivalent bridge; extraction must still prove view-only semantics",
        )
    else:
        return "host_required", "no isolated Atlas emitter for this semantic"
    unsupported = sorted(tensor_dtypes - {"f32", "bf16"})
    if unsupported:
        return (
            "host_required",
            "isolated emitter does not support tensor dtype(s): " + ", ".join(unsupported),
        )
    ranks = {len(shape) for shape in tensor_shapes}
    rank2 = [shape for shape in tensor_shapes if len(shape) == 2]
    if candidate == "vector_candidate":
        if semantic == "add":
            shape_supported = bool(rank2) and ranks <= {1, 2}
        elif semantic in {"gelu", "softmax", "reduce_sum"}:
            shape_supported = bool(rank2) and ranks <= {1, 2}
            if semantic in {"softmax", "reduce_sum"}:
                shape_supported &= any(s[-1] <= 32 or s[-1] == 64 for s in rank2)
        else:  # layer_norm
            shape_supported = ranks <= {1, 2} and any(s[-1] == 64 for s in rank2)
        if not shape_supported:
            return (
                "host_required",
                "isolated vector emitter does not support this tensor rank/extent",
            )
    return candidate, reason


def _semantic_partition_class(semantic: str) -> str:
    """Classify only by whether an isolated semantic emitter exists."""
    if semantic in MESH_CANDIDATES:
        return "mesh_candidate"
    if semantic in VECTOR_CANDIDATES:
        return "vector_candidate"
    if semantic in LAYOUT_BRIDGES:
        return "layout_bridge_candidate"
    return "host_required"


def inventory_full_graph(workload) -> dict:
    """Inventory logical regions and implement a conservative linear cut plan."""
    funcs = [op for op in workload.module.walk() if op.name == "func.func"]
    if len(funcs) != 1:
        raise ValueError(f"expected one func.func, found {len(funcs)}")
    block = funcs[0].body.blocks[0]

    regions: OrderedDict[str, dict] = OrderedDict()
    unattributed = Counter()
    for op in block.ops:
        region_id = _str_attr(op, "prov.region_id")
        if not region_id:
            unattributed[op.name] += 1
            continue
        semantic = _str_attr(op, "prov.op")
        family = _str_attr(op, "prov.family")
        row = regions.setdefault(
            region_id,
            {
                "region_id": region_id,
                "semantic": semantic,
                "family": family,
                "mlir_ops": Counter(),
                "tensor_dtypes": set(),
                "tensor_shapes": set(),
            },
        )
        if row["semantic"] != semantic or row["family"] != family:
            raise ValueError(f"inconsistent provenance for {region_id}")
        row["mlir_ops"][op.name] += 1
        for value in (*op.operands, *op.results):
            if (dtype := _tensor_dtype(value)) is not None:
                row["tensor_dtypes"].add(dtype)
            if (shape := _tensor_shape(value)) is not None:
                row["tensor_shapes"].add(shape)

    ordered = []
    for row in regions.values():
        partition_class, reason = _partition_class(
            row["semantic"], row["tensor_dtypes"], row["tensor_shapes"]
        )
        ordered.append(
            {
                **row,
                "mlir_ops": dict(sorted(row["mlir_ops"].items())),
                "tensor_dtypes": sorted(row["tensor_dtypes"]),
                "tensor_shapes": [list(s) for s in sorted(row["tensor_shapes"])],
                "partition_class": partition_class,
                "partition_reason": reason,
            }
        )

    # This is an executable boundary policy, not a whole-model lowering:
    # contiguous non-host regions form a proposed extraction island.  A host
    # region cuts on both sides.  Layout bridges remain inside an island so a
    # later extractor can prove them as views/index maps instead of eagerly
    # materialising them.
    islands = []
    start = None
    for index in range(len(ordered) + 1):
        eligible = (
            index < len(ordered)
            and ordered[index]["partition_class"] != "host_required"
        )
        if eligible and start is None:
            start = index
        if not eligible and start is not None:
            rows = ordered[start:index]
            islands.append(
                {
                    "index": len(islands),
                    "first_region": rows[0]["region_id"],
                    "last_region": rows[-1]["region_id"],
                    "region_count": len(rows),
                    "by_partition_class": dict(
                        sorted(Counter(r["partition_class"] for r in rows).items())
                    ),
                    "by_semantic": dict(
                        sorted(Counter(r["semantic"] for r in rows).items())
                    ),
                }
            )
            start = None

    by_semantic = Counter(row["semantic"] for row in ordered)
    by_class = Counter(row["partition_class"] for row in ordered)
    by_semantic_class = Counter(
        _semantic_partition_class(row["semantic"]) for row in ordered
    )
    host_semantics = Counter(
        row["semantic"]
        for row in ordered
        if row["partition_class"] == "host_required"
    )
    attributed_ops = sum(sum(row["mlir_ops"].values()) for row in regions.values())
    physical = Counter()
    for row in ordered:
        physical["rank2_matmul"] += row["mlir_ops"].get("linalg.matmul", 0)
        if row["semantic"] == "batch_matmul":
            physical["batched_matmul"] += row["mlir_ops"].get("linalg.generic", 0)

    return {
        "schema": "atlas_full_graph_partition_inventory_v1",
        "claim": "partition inventory only; no extracted image or whole-model execution",
        "logical_regions": len(ordered),
        "attributed_top_level_ops": attributed_ops,
        "unattributed_top_level_ops": sum(unattributed.values()),
        "unattributed_by_mlir_op": dict(sorted(unattributed.items())),
        "regions_by_semantic": dict(sorted(by_semantic.items())),
        "regions_by_partition_class": dict(sorted(by_class.items())),
        "regions_by_semantic_capability": dict(sorted(by_semantic_class.items())),
        "host_required_breakdown": {
            "no_semantic_emitter": by_semantic_class["host_required"],
            "emitter_shape_or_dtype_refused": (
                by_class["host_required"] - by_semantic_class["host_required"]
            ),
            "effective_total": by_class["host_required"],
        },
        "host_required_by_semantic": dict(sorted(host_semantics.items())),
        "physical_contractions": {
            **dict(sorted(physical.items())),
            "total": sum(physical.values()),
        },
        "boundary_policy": {
            "cut_before_and_after": "host_required",
            "retain_inside_candidate": "layout_bridge_candidate",
            "post_extraction_requirements": [
                "prove every bridge is an equivalent view/index map or emit movement",
                "split each kernel image to at most 32768 instruction words",
                "allocate live intermediates across host/device boundaries",
                "numerically grade every partition before whole-model composition",
            ],
        },
        "candidate_island_count": len(islands),
        "candidate_islands": islands,
    }
