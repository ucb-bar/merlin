"""Verified MLIR inventory and conservative Atlas family candidates.

Run with the pinned compiler Python containing torch_mlir. This accepts any
MLIR artifact; parsing is not evidence of capture completion or execution.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


MAPPING_REVISION = "0333da0"
FAMILIES = {
    "projection_gemm": "FP8 E4M3 projection GEMM",
    "epilogue": "fused SiLU and gate multiply with FP8 output",
    "rne_add": "exact BF16 residual add in projection epilogues",
    "rmsnorm": "RMSNorm with FP8 output",
    "rope": "rotary embedding in q/k epilogues",
    "attention": "attention core with GQA and head dimension 64",
    "smol_fusion": "vision fc1 bias and GELU(tanh) with FP8 output",
    "resnet_pool": "ResNet50 3x3 stride-2 max pool",
    "resnet_patch_gather": "ResNet50 3x3 im2col patch gather",
}


def _walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _tensor_type(typ, ir):
    if isinstance(typ, ir.RankedTensorType):
        tensor = ir.RankedTensorType(typ)
        return {
            "shape": [None if size < 0 else size for size in tensor.shape],
            "dtype": str(tensor.element_type),
        }
    if isinstance(typ, ir.UnrankedTensorType):
        return {"shape": None, "dtype": str(ir.UnrankedTensorType(typ).element_type)}
    return None


def _signature(operation):
    return {
        "op": operation.name,
        "operands": [str(value.type) for value in operation.operands],
        "results": [str(value.type) for value in operation.results],
    }


def _candidates(name, body_ops):
    if name.startswith("linalg.") and any(
        word in name for word in ("matmul", "matvec", "vecmat", "dot", "contract", "conv")
    ):
        return ["projection_gemm"], "contraction geometry only; panel/layout and precision admission required"
    if name == "linalg.generic" and {"arith.mulf", "arith.addf"} <= set(body_ops):
        return ["projection_gemm", "rmsnorm", "epilogue"], "scalar body resemblance only; indexing and full dataflow unqualified"
    if name == "arith.addf":
        return ["rne_add", "smol_fusion"], "BF16 residual or fused bias candidate only"
    if name == "arith.mulf":
        return ["epilogue", "rmsnorm", "rope", "attention"], "multiply alone does not establish a complete kernel pattern"
    if name in {"math.exp", "math.tanh", "math.rsqrt", "arith.divf", "arith.maximumf", "arith.maxnumf"}:
        return ["attention", "rmsnorm", "epilogue", "smol_fusion"], "possible subexpression only; no standalone general kernel established"
    if name.startswith("linalg.pooling") and "max" in name:
        return ["resnet_pool"], "only qualified 3x3 stride-2 geometry is documented"
    return [], "no direct family established by this operation kind"


def inventory(source: bytes) -> dict:
    from torch_mlir import ir
    from torch_mlir.dialects import torch as torch_dialect

    kinds, tensor_types, dtypes, f32_kinds = Counter(), Counter(), Counter(), Counter()
    generics, contractions, classifications = [], [], {}
    with ir.Context() as context:
        torch_dialect.register_dialect(context)
        module = ir.Module.parse(source.decode("utf-8"))
        if not module.operation.verify():
            raise ValueError("MLIR verification failed")
        for index, operation in enumerate(_walk(module.operation)):
            name = operation.name
            kinds[name] += 1
            typed_values = [*operation.operands, *operation.results]
            tensors = [_tensor_type(value.type, ir) for value in typed_values]
            tensors = [tensor for tensor in tensors if tensor is not None]
            has_f32 = any(str(value.type) == "f32" for value in typed_values)
            for tensor in tensors:
                tensor_types[json.dumps(tensor, sort_keys=True)] += 1
                dtypes[tensor["dtype"]] += 1
                has_f32 |= tensor["dtype"] == "f32"
            if has_f32:
                f32_kinds[name] += 1
            body = list(_walk(operation))[1:] if name == "linalg.generic" else []
            body_ops = [child.name for child in body]
            families, reason = _candidates(name, body_ops)
            entry = classifications.setdefault(name, {
                "geometric_family_candidates": [], "direct_support": "NOT_ESTABLISHED",
                "reasons": [],
            })
            entry["geometric_family_candidates"] = sorted(set(entry["geometric_family_candidates"]) | set(families))
            entry["family_status"] = "geometric_candidate" if entry["geometric_family_candidates"] else "gap_or_host_control"
            if reason not in entry["reasons"]:
                entry["reasons"].append(reason)
            if has_f32:
                entry["contains_f32"] = True
                entry["f32_direct_support_no_quantization"] = "NOT_SUPPORTED"
            record = {"ordinal": index, **_signature(operation)}
            if name == "linalg.generic":
                record["body_signatures"] = [_signature(child) for child in body]
                record["indexing_maps"] = str(operation.attributes["indexing_maps"])
                record["iterator_types"] = str(operation.attributes["iterator_types"])
                generics.append(record)
            named_contraction = name.startswith("linalg.") and any(
                word in name for word in ("matmul", "matvec", "vecmat", "dot", "contract", "conv")
            )
            generic_contraction = (
                name == "linalg.generic" and "reduction" in record["iterator_types"]
                and any(word in body_ops for word in ("arith.mulf", "arith.muli"))
                and any(word in body_ops for word in ("arith.addf", "arith.addi"))
            )
            if named_contraction or generic_contraction:
                contractions.append({**record, "tensor_occurrences": tensors,
                                     "recognition": "named" if named_contraction else "reduction_multiply_add_candidate"})
    return {
        "schema": "merlin.multi_task_dit.mlir_inventory.v1",
        "source_sha256": hashlib.sha256(source).hexdigest(), "verified": True,
        "scope": "supplied MLIR only; capture completion and execution are unqualified",
        "mapping_reference": {"repository": "atlas-kernels", "revision": MAPPING_REVISION,
                              "path": "coverage/mapping.yaml", "families": FAMILIES},
        "precision_policy": {
            "quantization": "none", "f32_direct_support": "NOT_SUPPORTED",
            "warning": "All family matches are geometric candidates, not supported f32 routes. Existing projection kernels use FP8 E4M3 and residual add uses BF16. Fused kernels have FP8/BF16 boundary restrictions; attention is qualified for d=64 only. No casts or quantization are authorized by this inventory.",
        },
        "counting": "recursive operation occurrences; tensor types count each operand/result occurrence, including repeated SSA uses",
        "op_counts": dict(sorted(kinds.items())),
        "f32_op_counts": dict(sorted(f32_kinds.items())),
        "tensor_dtype_occurrences": dict(sorted(dtypes.items())),
        "tensor_shape_dtype_occurrences": [
            {**json.loads(key), "count": count} for key, count in sorted(tensor_types.items())
        ],
        "contractions": contractions, "generic_bodies": generics,
        "classification_by_op_kind": dict(sorted(classifications.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.mlir.read_bytes())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
