"""Opt-in shared-input linear cores with one native requantized output.

Pack frozen signed-I8 weight columns, accumulator bias and readout gains; split
the resulting I8 codes back into the original independent output grids. This
changes neither the dot-product reduction nor any projection's arithmetic.
It supplies no placement, physical input-reuse or execution qualification.
"""

from __future__ import annotations

import copy
import dis
import hashlib
import math
from collections import defaultdict
from typing import Any

SCHEMA = "shared_input_projection_v1"
KEY = "projection_input_reuse"


class ProjectionReuseError(ValueError):
    """The explicitly selected projection policy is not representable."""


def validate(policy: Any, columns_per_scale: Any) -> None:
    if policy != {"schema": SCHEMA} or type(policy) is not dict:
        raise ProjectionReuseError("projection input reuse requires its closed explicit schema")
    if type(columns_per_scale) is not int or columns_per_scale < 1:
        raise ProjectionReuseError("projection input reuse requires a selected positive physical scale width")


class _Unsupported(ValueError):
    pass


def _attribute(gm, node, torch):
    if not isinstance(node, torch.fx.Node) or node.op != "get_attr":
        raise _Unsupported("constant is not a frozen tensor attribute")
    value = gm
    for part in str(node.target).split("."):
        value = getattr(value, part)
    if (
        type(value) not in (torch.Tensor, torch.nn.Parameter)
        or value.device.type != "cpu"
        or value.layout != torch.strided
        or value.requires_grad
    ):
        raise _Unsupported("constant must be a frozen ordinary CPU tensor")
    return value.detach()


def _unit(node, torch):
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target != torch.ops.quantized_decomposed.dequantize_per_tensor.default
        or len(node.args) != 6
        or type(node.args[1]) not in (int, float)
        or node.args[1] != 1
        or type(node.args[2]) is not int
        or node.args[2] != 0
        or any(type(x) is not int for x in node.args[3:5])
        or not -128 <= node.args[3] < node.args[4] <= 127
        or node.args[5] != torch.int8
        or set(node.kwargs) - {"out_dtype"}
        or node.kwargs.get("out_dtype") not in (None, torch.float32)
    ):
        raise _Unsupported("linear input is not a literal unit-grid signed-I8/f32 dequantization")
    return node.args[0]


def _input_key(node, torch):
    """Only identical SSA or exact duplicate pure Q/DQ/view expressions share input."""
    allowed = {
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
    }

    def key(value):
        if isinstance(value, torch.fx.Node):
            if value.op == "call_function" and value.target in allowed:
                return (value.target, key(value.args), key(dict(value.kwargs)))
            return value
        if isinstance(value, dict):
            return tuple((k, key(v)) for k, v in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(key(v) for v in value)
        return (type(value), value)

    return key(node)


def _input_evaluations(node, torch):
    """Include duplicate quantizer evaluation times in the no-effect span."""
    allowed = {
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
    }
    found = set()

    def visit(value):
        if isinstance(value, torch.fx.Node) and value.op == "call_function" and value.target in allowed:
            if value not in found:
                found.add(value)
                visit(value.args)
                visit(dict(value.kwargs))
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(node)
    return found


def _empty_guard(gm, node):
    try:
        return _guard_details(gm, node)
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError, RuntimeError):
        return False


def _guard_details(gm, node):
    """Recognize only the framework carrier's closed read-only input assertions.

    Inspect bytecode without calling a guard. No arbitrary module-name grant,
    state/global write, closure, hook, tensor mutator or backward jump is legal.
    The original guard node remains in its original order in the graph.
    """
    if node.op != "call_module":
        return False
    from torch.export._unlift import GuardsFn

    guard = gm.get_submodule(node.target)
    if not (
        type(guard) is GuardsFn
        and not guard._parameters
        and not guard._buffers
        and not guard._modules
        and not guard._forward_hooks
        and not guard._forward_pre_hooks
        and not guard._backward_hooks
        and not guard._backward_pre_hooks
    ):
        return False
    function = guard.__dict__.get("forward", GuardsFn.forward)
    import types

    import torch

    if type(function) is types.FunctionType and function.__closure__:
        from torch._dynamo.decorators import DynamoConfigPatchProxy
        from torch._dynamo.external_utils import wrap_dunder_call_ctx_manager

        wrapper = next(
            code
            for code in wrap_dunder_call_ctx_manager.__code__.co_consts
            if type(code) is types.CodeType and code.co_name == "inner"
        )
        cells = dict(zip(function.__code__.co_freevars, function.__closure__))
        if function.__code__ is not wrapper or set(cells) != {"func", "self"}:
            return False
        context = cells["self"].cell_contents
        ordinary = torch._dynamo.dont_skip_tracing()
        if type(context) is not DynamoConfigPatchProxy:
            return False
        patch, expected = vars(context).get("config_patch"), ordinary.config_patch
        if (
            type(type(patch)) is not type
            or set(type(patch).__dict__) != set(type(expected).__dict__)
            or type(patch).__mro__[1:] != type(expected).__mro__[1:]
            or vars(patch).get("changes") != {"dont_skip_tracing": True}
        ):
            return False
        closures = {}
        for method in ("__init__", "__enter__", "__exit__"):
            actual = type(patch).__dict__[method]
            if (
                type(actual) is not types.FunctionType
                or actual.__code__ is not type(expected).__dict__[method].__code__
            ):
                return False
            closures[method] = dict(zip(actual.__code__.co_freevars, actual.__closure__ or ()))
        enter, exit_ = closures["__enter__"], closures["__exit__"]
        if (
            enter["config"].cell_contents is not torch._dynamo.config
            or exit_["config"].cell_contents is not torch._dynamo.config
            or type(enter["prior"].cell_contents) is not dict
            or enter["prior"].cell_contents
            or exit_["prior"].cell_contents is not enter["prior"].cell_contents
        ):
            return False
        function = cells["func"].cell_contents
    if (
        type(function) is not types.FunctionType
        or function.__closure__
        or function.__globals__.get("torch") is not torch
    ):
        return False
    allowed = {
        "RESUME",
        "NOP",
        "LOAD_GLOBAL",
        "LOAD_ATTR",
        "LOAD_FAST",
        "LOAD_CONST",
        "BINARY_SUBSCR",
        "CALL",
        "PRECALL",
        "PUSH_NULL",
        "RETURN_CONST",
        "RETURN_VALUE",
        "POP_TOP",
        "COMPARE_OP",
        "BINARY_OP",
        "UNARY_NEGATIVE",
        "UNARY_POSITIVE",
        "UNARY_NOT",
        "JUMP_FORWARD",
        "POP_JUMP_IF_FALSE",
        "POP_JUMP_IF_TRUE",
        "JUMP_IF_FALSE_OR_POP",
        "JUMP_IF_TRUE_OR_POP",
        "COPY",
        "SWAP",
    }
    attributes = {
        "_assert",
        "size",
        "stride",
        "storage_offset",
        "dim",
        "numel",
        "element_size",
        "is_contiguous",
        "dtype",
        "device",
        "ndim",
        "requires_grad",
    }
    for instruction in dis.get_instructions(function):
        if instruction.opname not in allowed:
            return False
        if instruction.opname == "LOAD_GLOBAL" and instruction.argval != "torch":
            return False
        if instruction.opname == "LOAD_ATTR" and instruction.argval not in attributes:
            return False
        if instruction.opname == "LOAD_CONST" and type(instruction.argval) not in (int, float, str, bool, type(None)):
            return False
        if instruction.opname == "BINARY_OP" and instruction.argrepr.endswith("="):
            return False
    return True


def _frozen_constants(gm, nodes, selected, torch):
    """A frozen attribute cannot be written through another graph-visible alias."""
    for node in nodes:
        if _empty_guard(gm, node):
            continue
        if node.op in ("call_module", "call_method") or (
            node.op == "call_function"
            and (
                not isinstance(node.target, torch._ops.OpOverload)
                or node.target.namespace not in ("aten", "quantized_decomposed")
            )
        ):
            return False
    pointers = {value.untyped_storage().data_ptr() for value in selected}
    pending = []
    for node in nodes:
        if node.op == "get_attr":
            value = gm
            for part in str(node.target).split("."):
                value = getattr(value, part)
            if (
                type(value) in (torch.Tensor, torch.nn.Parameter)
                and value.device.type == "cpu"
                and value.layout == torch.strided
                and value.untyped_storage().data_ptr() in pointers
            ):
                pending.append(node)
    seen = set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        for user in node.users:
            if user.op != "call_function" or user.target._schema.is_mutable:
                return False
            if any(result.alias_info is not None for result in user.target._schema.returns):
                pending.append(user)
    return True


def _activation_shape(gm, node, torch):
    raw = _unit(node, torch)
    if not isinstance(raw, torch.fx.Node):
        raise _Unsupported("activation storage is not a typed graph value")
    if raw.op == "get_attr":
        value = _attribute(gm, raw, torch)
        if value.dtype != torch.int8:
            raise _Unsupported("activation storage is not signed I8")
    else:
        value = raw.meta.get("val")
        if value is None:
            value = raw.meta.get("tensor_meta")
        if raw.op == "call_function" and raw.target == torch.ops.quantized_decomposed.quantize_per_tensor.default:
            if (
                len(raw.args) != 6
                or raw.kwargs
                or raw.args[-1] != torch.int8
                or type(raw.args[1]) not in (int, float)
                or not math.isfinite(raw.args[1])
                or raw.args[1] <= 0
                or type(raw.args[2]) is not int
                or any(type(n) is not int for n in raw.args[3:5])
                or not -128 <= raw.args[3] < raw.args[4] <= 127
                or not isinstance(raw.args[0], torch.fx.Node)
            ):
                raise _Unsupported("activation quantizer lacks the closed signed-I8 ABI")
            value = raw.args[0].meta.get("val")
            if value is None:
                value = raw.args[0].meta.get("tensor_meta")
            if getattr(value, "dtype", None) != torch.float32:
                raise _Unsupported("activation quantizer lacks observed f32 source geometry")
        elif getattr(value, "dtype", None) != torch.int8:
            raise _Unsupported("activation storage lacks observed signed-I8 geometry")
    shape = tuple(getattr(value, "shape", ()))
    if len(shape) < 2 or any(type(n) is not int or n < 1 for n in shape):
        raise _Unsupported("activation has no static positive tensor geometry")
    return shape


def _site(gm, node, torch):
    if len(node.args) not in (2, 3) or node.kwargs:
        raise _Unsupported("linear does not have the closed positional input/weight/bias ABI")
    activation_shape = _activation_shape(gm, node.args[0], torch)
    weight = _attribute(gm, _unit(node.args[1], torch), torch)
    if weight.dtype != torch.int8 or weight.ndim != 2 or any(n < 1 for n in weight.shape):
        raise _Unsupported("weight is not a positive rank-two frozen signed-I8 matrix")
    channels, reduction = weight.shape
    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("tensor_meta")
    # Do not infer unobserved tensor geometry from the weight or a node name.
    shape = tuple(getattr(value, "shape", ()))
    if (
        getattr(value, "dtype", None) != torch.float32
        or len(shape) < 2
        or any(type(d) is not int or d < 1 for d in shape)
        or shape[-1] != channels
        or activation_shape[-1] != reduction
        or shape[:-1] != activation_shape[:-1]
    ):
        raise _Unsupported("linear lacks observed static positive f32 output geometry")
    bias = _attribute(gm, node.args[2], torch) if len(node.args) == 3 and node.args[2] is not None else None
    if bias is not None and (
        bias.dtype != torch.float32
        or tuple(bias.shape) != (channels,)
        or not bool(torch.isfinite(bias).all())
        or not bool((bias == bias.round()).all())
        or bool((bias.abs() > 2**24).any())
    ):
        raise _Unsupported("bias is not a frozen exact integer-code f32 vector")
    magnitude = max(abs(node.args[0].args[3]), abs(node.args[0].args[4]))
    bound = weight.to(torch.int64).abs().sum(1) * magnitude
    if bias is not None:
        bound += bias.to(torch.int64).abs()
    if bool((bound > 2**24).any()):
        raise _Unsupported("every partial dot and biased sum must fit the exact f32 integer carrier")
    users = list(node.users)
    if len(users) != 1:
        raise _Unsupported("linear has another live-out before the selected readout")
    mul = users[0]
    if mul.target != torch.ops.aten.mul.Tensor or mul.op != "call_function" or mul.kwargs:
        raise _Unsupported("linear does not feed exactly one native scale")
    if len(mul.args) != 2 or mul.args[0] is not node:
        raise _Unsupported("readout scale has a different arithmetic order")
    gain = _attribute(gm, mul.args[1], torch)
    if (
        gain.dtype != torch.float32
        or gain.numel() != channels
        or tuple(gain.shape) not in ((channels,), (1, channels))
        or not bool(torch.isfinite(gain).all())
        or not bool((gain > 0).all())
    ):
        raise _Unsupported("readout gain is not a positive finite per-column f32 vector")
    users = list(mul.users)
    if len(users) != 1:
        raise _Unsupported("scale has another live-out before native requantization")
    quantize = users[0]
    if (
        quantize.op != "call_function"
        or quantize.target != torch.ops.quantized_decomposed.quantize_per_tensor.default
        or len(quantize.args) != 6
        or quantize.kwargs
        or quantize.args[0] is not mul
        or type(quantize.args[1]) not in (int, float)
        or quantize.args[1] != 1
        or type(quantize.args[2]) is not int
        or quantize.args[2] != 0
        or any(type(x) is not int for x in quantize.args[3:5])
        or not -128 <= quantize.args[3] < quantize.args[4] <= 127
        or quantize.args[5] != torch.int8
    ):
        raise _Unsupported("readout is not a closed unit-grid native-I8 quantize")
    return dict(
        node=node,
        mul=mul,
        quantize=quantize,
        weight=weight,
        bias=bias,
        gain=gain.reshape(-1),
        channels=channels,
        reduction=reduction,
        shape=shape,
    )


def _pure_span(gm, nodes, start, end, torch):
    for node in nodes[start : end + 1]:
        if node.op == "get_attr" or _empty_guard(gm, node):
            continue
        if node.op != "call_function" or not isinstance(node.target, torch._ops.OpOverload):
            return False
        schema = node.target._schema
        if (
            node.target.namespace not in ("aten", "quantized_decomposed")
            or schema.is_mutable
            or torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return False
    return True


def _metadata(nodes):
    origins = sorted({s for node in nodes for s in (node.meta.get("custom") or {}).get("m2m_lineage", ())})
    return {"custom": {"m2m_lineage": origins, "m2m_transform": SCHEMA}} if origins else {}


def rewrite(gm: Any, policy: Any, *, columns_per_scale: int) -> dict:
    """Fuse legal native requantized siblings; all other bodies stay unchanged.

    The caller supplies the selected physical gain-vector width. No padding or
    arbitrary scale repartition is invented. Copies of each I8 output slice
    preserve independent allocation/contiguous layout and downstream effects.
    """
    validate(policy, columns_per_scale)
    import torch

    if not isinstance(gm, torch.fx.GraphModule):
        raise ProjectionReuseError("projection input reuse requires an owned FX GraphModule")
    nodes = list(gm.graph.nodes)
    positions = {node: i for i, node in enumerate(nodes)}
    candidates, refused = defaultdict(list), []
    for node in nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue
        try:
            site = _site(gm, node, torch)
            q = site["quantize"]
            key = (_input_key(node.args[0], torch), site["reduction"], site["shape"][:-1], q.args[3:])
            candidates[key].append(site)
        except _Unsupported as error:
            refused.append({"node": node.name, "reason": str(error)})
    plans = []
    for sites in candidates.values():
        if len(sites) < 2:
            continue
        evaluations = {node for s in sites for node in _input_evaluations(s["node"].args[0], torch)}
        start = min(positions[node] for node in (*evaluations, sites[0]["node"]))
        if not _pure_span(gm, nodes, start, positions[sites[-1]["node"]], torch):
            refused.append({"nodes": [s["node"].name for s in sites], "reason": "intervening effect or unknown call"})
            continue
        constants = [s[k] for s in sites for k in ("weight", "bias", "gain") if s[k] is not None]
        if not _frozen_constants(gm, nodes, constants, torch):
            refused.append(
                {
                    "nodes": [s["node"].name for s in sites],
                    "reason": "frozen constant may be written or escape through an alias",
                }
            )
            continue
        gains = torch.cat([s["gain"] for s in sites])
        if any(
            torch.unique(gains[i : i + columns_per_scale]).numel() != 1
            for i in range(0, gains.numel(), columns_per_scale)
        ):
            refused.append(
                {
                    "nodes": [s["node"].name for s in sites],
                    "reason": "packed output gain crosses an incompatible physical scale group",
                }
            )
            continue
        plans.append((sites, gains))
    groups = []
    for sites, gains in plans:
        graph = gm.graph
        first = sites[0]["node"]
        meta = _metadata([s["node"] for s in sites])
        base = "_merlin_shared_projection"
        index = 0
        while any(hasattr(gm, f"{base}_{index}_{suffix}") for suffix in ("weight", "bias", "gain")):
            index += 1
        prefix = f"{base}_{index}"
        weight = torch.cat([s["weight"] for s in sites]).contiguous()
        bias = torch.cat(
            [s["bias"] if s["bias"] is not None else torch.zeros(s["channels"], dtype=torch.float32) for s in sites]
        )
        for suffix, value in (("weight", weight), ("bias", bias), ("gain", gains)):
            gm.register_buffer(f"{prefix}_{suffix}", value.clone())
        with graph.inserting_before(first):

            def create(target, args, kwargs=None):
                result = graph.call_function(target, args, kwargs or {})
                result.meta.update(copy.deepcopy(meta))
                return result

            w = graph.get_attr(f"{prefix}_weight")
            b = graph.get_attr(f"{prefix}_bias")
            gain = graph.get_attr(f"{prefix}_gain")
            unit = create(
                torch.ops.quantized_decomposed.dequantize_per_tensor.default, (w, 1.0, 0, -128, 127, torch.int8)
            )
            combined = create(torch.ops.aten.linear.default, (first.args[0], unit, b))
            scaled = create(torch.ops.aten.mul.Tensor, (combined, gain))
            codes = create(
                torch.ops.quantized_decomposed.quantize_per_tensor.default, (scaled, *sites[0]["quantize"].args[1:])
            )
            offset = 0
            for site in sites:
                part = create(
                    torch.ops.aten.slice.Tensor, (codes, len(site["shape"]) - 1, offset, offset + site["channels"])
                )
                independent = create(torch.ops.aten.clone.default, (part,), {"memory_format": torch.contiguous_format})
                independent.meta.update(copy.deepcopy(site["quantize"].meta))
                if independent.meta.get("custom"):
                    independent.meta["custom"]["m2m_transform"] = SCHEMA
                site["quantize"].replace_all_uses_with(independent)
                offset += site["channels"]
        groups.append(
            {
                "members": [s["node"].name for s in sites],
                "output_columns": [s["channels"] for s in sites],
                "reduction": sites[0]["reduction"],
                "columns_per_scale": columns_per_scale,
                "weight_sha256": hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
                "output_cuts": "independent_contiguous_I8_before_original_output_grid_DQ",
            }
        )
    if groups:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
    return {
        "schema": SCHEMA,
        "groups": groups,
        "refused": refused,
        "logical_contractions_fused": sum(len(g["members"]) for g in groups),
        "physical_contractions": len(groups),
        "contractions_removed": sum(len(g["members"]) - 1 for g in groups),
        "native_i8_exits_preserved": True,
    }
