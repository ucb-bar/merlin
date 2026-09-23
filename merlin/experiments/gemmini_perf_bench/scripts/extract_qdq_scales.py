#!/usr/bin/env python3
"""Extract the per-tensor quantization table of a QDQ ONNX model into one JSON file.

The table is the single source of scales that both a library-style C build of the model and a
Merlin capture of the same model read, so the two are compiled under ONE numerics contract and can
share one golden. For every compute node it records the dequantize scale of each operand and the
quantize scale of its output, plus the derived store-path requant scale:

    Conv / Gemm : s_x, s_w, s_bias, s_out, requant = s_x * s_w / s_out, relu (fused after it?)
    Add         : s_a, s_b, s_out                   (residual add; each input rescaled to s_out)
    MaxPool / GlobalAveragePool : s_in, s_out

It refuses (exit 2) on anything a per-tensor contract cannot represent: a non-scalar scale, a nonzero
zero point, or a compute node whose operand is not produced by a DequantizeLinear.

Runs in any Python with ``onnx`` + ``numpy`` (the merlin venv has neither; use the model2MLIR venv):

    python \\
        merlin/experiments/gemmini_perf_bench/scripts/extract_qdq_scales.py \\
        --model /path/to/model.q.int8.onnx \\
        --out out/artifacts/perf-bench/gemmini/spec_model_scales/spec_scales.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


class NotPerTensor(ValueError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(model_path: Path) -> dict:
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.shape_inference.infer_shapes(onnx.load(str(model_path)))
    g = model.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    shapes: dict[str, list[int]] = {}
    for vi in [*g.input, *g.value_info, *g.output]:
        dims = vi.type.tensor_type.shape.dim
        if dims and all(d.HasField("dim_value") for d in dims):
            shapes[vi.name] = [int(d.dim_value) for d in dims]
    producer = {o: n for n in g.node for o in n.output}
    consumers: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def scalar(name: str) -> float:
        if name not in init:
            raise NotPerTensor(f"scale {name} is not a constant")
        v = init[name]
        if v.size != 1:
            raise NotPerTensor(f"scale {name} has shape {v.shape}; not per-tensor")
        return float(v.reshape(()).item())

    def zero_point_is_zero(node) -> None:
        if len(node.input) > 2 and node.input[2]:
            zp = init.get(node.input[2])
            if zp is None or np.any(zp != 0):
                raise NotPerTensor(f"{node.name}: nonzero or non-constant zero point")

    def dequant_scale(tensor: str) -> float:
        dq = producer.get(tensor)
        if dq is None or dq.op_type != "DequantizeLinear":
            raise NotPerTensor(f"operand {tensor} is not produced by DequantizeLinear")
        zero_point_is_zero(dq)
        return scalar(dq.input[1])

    def only_consumer(tensor: str):
        nxt = consumers.get(tensor, [])
        return nxt[0] if len(nxt) == 1 else None

    def output_quant(tensor: str) -> tuple[float, bool, float | None]:
        """The output's quantize scale, and whether a Relu follows it with its own quantize.

        ORT's static QDQ places every Relu between a DequantizeLinear and its own QuantizeLinear:
        ``op -> Q(s_out) -> DQ -> Relu -> Q(s_relu)``. So a fused op+relu is TWO roundings in the
        reference graph. Both scales are recorded; a store path that applies relu at one requant
        would use ``s_relu`` (see ``fused_requant_scale``), which is a declared contract choice, not
        the reference graph's arithmetic.
        """
        n = only_consumer(tensor)
        if n is None:
            raise NotPerTensor(f"{tensor} has {len(consumers.get(tensor, []))} consumers before quantization")
        if n.op_type == "Relu":  # Relu on the float output, then one quantize
            q = only_consumer(n.output[0])
            if q is None or q.op_type != "QuantizeLinear":
                raise NotPerTensor(f"{tensor}: Relu not followed by a single QuantizeLinear")
            zero_point_is_zero(q)
            s = scalar(q.input[1])
            return s, True, s
        if n.op_type != "QuantizeLinear":
            raise NotPerTensor(f"{tensor} reaches {n.op_type} before any QuantizeLinear")
        zero_point_is_zero(n)
        s_out = scalar(n.input[1])
        dq = only_consumer(n.output[0])
        if dq is None or dq.op_type != "DequantizeLinear":
            return s_out, False, None
        relu = only_consumer(dq.output[0])
        if relu is None or relu.op_type != "Relu":
            return s_out, False, None
        q2 = only_consumer(relu.output[0])
        if q2 is None or q2.op_type != "QuantizeLinear":
            raise NotPerTensor(f"{tensor}: Relu after dequantize is not requantized")
        zero_point_is_zero(q2)
        return s_out, True, scalar(q2.input[1])

    nodes = []
    for n in g.node:
        if n.op_type in ("Conv", "Gemm"):
            s_x = dequant_scale(n.input[0])
            s_w = dequant_scale(n.input[1])
            s_b = dequant_scale(n.input[2]) if len(n.input) > 2 and n.input[2] else None
            s_out, relu, s_relu = output_quant(n.output[0])
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            w = init[producer[n.input[1]].input[0]]
            nodes.append(
                {
                    "name": n.name,
                    "op": n.op_type,
                    "s_x": s_x,
                    "s_w": s_w,
                    "s_bias": s_b,
                    "s_out": s_out,
                    "relu": relu,
                    "s_relu": s_relu,
                    "requant_scale": s_x * s_w / s_out,
                    "fused_requant_scale": s_x * s_w / (s_relu if relu else s_out),
                    "bias_scale_is_product": None
                    if s_b is None
                    else bool(np.isclose(s_b, s_x * s_w, rtol=1e-6, atol=0.0)),
                    "weight_shape": list(w.shape),
                    "in_shape": shapes.get(n.input[0]),
                    "out_shape": shapes.get(n.output[0]),
                    "attributes": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in attrs.items()},
                }
            )
        elif n.op_type == "Add":
            s_a, s_b = dequant_scale(n.input[0]), dequant_scale(n.input[1])
            s_out, relu, s_relu = output_quant(n.output[0])
            nodes.append(
                {"name": n.name, "op": "Add", "s_a": s_a, "s_b": s_b, "s_out": s_out, "relu": relu, "s_relu": s_relu}
            )
        elif n.op_type in ("MaxPool", "GlobalAveragePool"):
            s_in = dequant_scale(n.input[0])
            s_out, _, _ = output_quant(n.output[0])
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            nodes.append(
                {
                    "name": n.name,
                    "op": n.op_type,
                    "s_in": s_in,
                    "s_out": s_out,
                    "attributes": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in attrs.items()},
                }
            )

    in_q = [n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] in {i.name for i in g.input}]
    return {
        "schema": "qdq_per_tensor_scale_table_v1",
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "input_scale": scalar(in_q[0].input[1]) if len(in_q) == 1 else None,
        "counts": {
            op: sum(1 for x in nodes if x["op"] == op) for op in ("Conv", "Gemm", "Add", "MaxPool", "GlobalAveragePool")
        },
        "nodes": nodes,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        table = extract(args.model)
    except NotPerTensor as why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({table['counts']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
