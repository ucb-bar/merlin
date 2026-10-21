"""Explicit integer spatial means using the ordinary contraction interface.

The selected numerical recipe is an exact integer sum followed by ONE f32
readout, half-even rounding and the selected I8 clamp. It is not a statement
that an older f32 reduction has the same intermediate rounding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

if __package__:
    from ._integer_residual import ResidualError, _f32, _shape, _storage_shape
else:
    from _integer_residual import ResidualError, _f32, _shape, _storage_shape

SCHEMA = "integer_mean_contraction_v1"


def check(policy):
    fields = {"schema", "max_window_elements", "max_weight_elements", "max_macs"}
    if (
        not isinstance(policy, Mapping)
        or set(policy) != fields
        or policy["schema"] != SCHEMA
        or any(type(policy[k]) is not int or policy[k] < 1 for k in fields - {"schema"})
    ):
        raise ResidualError("integer mean requires its closed schema and positive finite work limits")
    return dict(policy)


def rewrite(gm, policy, *, epilogue):
    """Rewrite selected complete spatial means; reject a malformed selected site."""
    import torch

    selected = check(policy)
    aten, qd = torch.ops.aten, torch.ops.quantized_decomposed
    rows = []
    for node in list(gm.graph.nodes):
        adaptive = node.target == aten.adaptive_avg_pool2d.default
        if node.target != aten.mean.dim and not adaptive:
            continue
        if (len(node.users) != 1 or (adaptive and (len(node.args) != 2 or node.kwargs))
                or (not adaptive and (len(node.args) not in (2, 3) or set(node.kwargs) - {"dtype"}
                                      or node.kwargs.get("dtype") not in (None, torch.float32)))):
            raise ResidualError("selected integer mean lacks a closed single-readout ABI")
        dq = node.args[0]
        if (
            not isinstance(dq, torch.fx.Node)
            or dq.target != qd.dequantize_per_tensor.default
            or len(dq.args) != 6
            or type(dq.args[2]) is not int
            or dq.args[2] != 0
            or list(dq.args[3:5]) != [-128, 127]
            or dq.args[5] != torch.int8
            or set(dq.kwargs) - {"out_dtype"}
            or dq.kwargs.get("out_dtype") not in (None, torch.float32)
        ):
            raise ResidualError("integer mean requires a symmetric signed-I8 per-tensor source")
        shape = _storage_shape(dq.args[0])
        if adaptive and (len(shape) != 4 or not isinstance(node.args[1], (list, tuple))
                         or list(node.args[1]) != [1, 1]
                         or any(type(d) is not int for d in node.args[1])):
            raise ResidualError("integer adaptive mean requires a complete literal1x1 spatial output")
        axes = [2, 3] if adaptive else node.args[1]
        if (
            not isinstance(axes, (list, tuple))
            or any(type(a) is not int for a in axes)
            or any(not -len(shape) <= a < len(shape) for a in axes)
        ):
            raise ResidualError("integer mean axes must be static literal integers")
        axes = [a % len(shape) for a in axes]
        # This is a spatial/global mean, not an arbitrary reassociation or channel reduction.
        if len(shape) < 3 or sorted(axes) != list(range(2, len(shape))):
            raise ResidualError("integer mean supports only complete unique trailing spatial axes")
        keep = True if adaptive else node.args[2] if len(node.args) == 3 else False
        if type(keep) is not bool:
            raise ResidualError("integer mean keepdim must be a literal boolean")
        out_shape = shape[:2] + (1,) * (len(shape) - 2) if keep else shape[:2]
        if _shape(node) != out_shape:
            raise ResidualError("integer mean observed output geometry changed")
        last = node
        views = (aten.flatten.using_ints, aten.reshape.default, aten.view.default, aten._unsafe_view.default)
        quantize = next(iter(last.users))
        while quantize.target in views and quantize.op == "call_function":
            if (not quantize.args or quantize.args[0] is not last or len(quantize.users) != 1
                    or math.prod(_shape(quantize)) != math.prod(out_shape)):
                raise ResidualError("integer mean needs one shape-preserving path to its readout")
            last = quantize
            quantize = next(iter(last.users))
        clamp = (epilogue.get("requant") or {}).get("clamp")
        if (
            quantize.target != qd.quantize_per_tensor.default
            or len(quantize.args) != 6
            or quantize.kwargs
            or type(quantize.args[2]) is not int
            or quantize.args[2] != 0
            or list(quantize.args[3:5]) != clamp
            or quantize.args[5] != torch.int8
        ):
            raise ResidualError("integer mean needs the selected single I8 readout")
        source_scale, output_scale = _f32(dq.args[1]), _f32(quantize.args[1])
        window, count = math.prod(shape[2:]), math.prod(shape[:2])
        if (
            window > selected["max_window_elements"]
            or window > selected["max_weight_elements"]
            or count * window > selected["max_macs"]
            or window * 128 > 1 << 24
        ):
            raise ResidualError("integer mean exceeds the selected work or exact-carrier budget")
        gain = _f32(source_scale / (window * output_scale))
        if gain * window * 128 > 3.4028234663852886e38:
            raise ResidualError("integer mean selected readout overflows f32")
        name = f"_merlin_integer_mean_{len(rows)}_ones"
        if hasattr(gm, name):
            raise ResidualError("integer mean constant owner already exists")
        gm.register_buffer(name, torch.ones((1, window), dtype=torch.int8))
        graph = gm.graph

        def call(target, args):
            created = graph.call_function(target, args)
            custom = dict(node.meta.get("custom") or {})
            if custom.get("m2m_lineage"):
                created.meta["custom"] = custom | {"m2m_transform": SCHEMA}
            return created

        with graph.inserting_before(node):
            raw = call(aten.reshape.default, (dq.args[0], [count, window]))
            activation = call(qd.dequantize_per_tensor.default, (raw, 1.0, 0, -128, 127, torch.int8))
            kernel = graph.get_attr(name)
            weight = call(qd.dequantize_per_tensor.default, (kernel, 1.0, 0, -128, 127, torch.int8))
            total = call(aten.linear.default, (activation, weight))
            scaled = call(aten.mul.Tensor, (total, gain))
            value = call(aten.reshape.default, (scaled, list(out_shape)))
        value.meta["val"] = node.meta["val"]
        node.replace_all_uses_with(value)
        quantize.args = (quantize.args[0], 1.0, 0, *quantize.args[3:])
        rows.append(
            {
                "node": node.name,
                "schema": SCHEMA,
                "status": "rewritten",
                "window_elements": window,
                "macs": count * window,
                "weight_bytes": window,
                "carrier_abs_bound": window * 128,
                "gain_f32": gain,
                "arithmetic": "Q_i8(f32(integer_sum) * gain_f32), restoring original output grid",
            }
        )
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return {"schema": SCHEMA, "rewritten": len(rows), "new_contractions": len(rows), "sites": rows}
