"""Recognize closed read-only grad wrappers without moving or deleting them.

This supplies only a global frozen-constant boundary. It grants no permission
to move a contraction across the wrapper, nor to pass a selected constant or
one of its aliases into an opaque call.
"""

from __future__ import annotations

import dis
import operator
import types


def _same_code(actual, expected):
    keys = (
        "co_code",
        "co_consts",
        "co_names",
        "co_varnames",
        "co_argcount",
        "co_posonlyargcount",
        "co_kwonlyargcount",
        "co_flags",
        "co_freevars",
        "co_cellvars",
    )
    return all(getattr(actual, key) == getattr(expected, key) for key in keys)


def _forward_matches_graph(body, torch):
    actual = getattr(body.forward, "__func__", None)
    if type(actual) is not types.FunctionType or actual.__closure__:
        return False
    # Compare actual executable code with the selected FX generator's fresh
    # rendering of this exact attribute-free graph, not a mutable .code label.
    graph = torch.fx.Graph()
    values = {}
    from torch.fx.node import map_arg

    # Rebuild only already-closed semantic nodes. Do not invoke a foreign
    # graph code generator or deepcopy arbitrary source metadata.
    for node in body.graph.nodes:
        if not any(node.type is value for value in (None, torch.Tensor, int, float, bool)):
            return False
        values[node] = graph.create_node(
            node.op,
            node.target,
            map_arg(node.args, values.__getitem__),
            map_arg(node.kwargs, values.__getitem__),
            node.name,
            node.type,
        )
    generated = torch.fx.GraphModule(torch.nn.Module(), graph)
    expected = generated.forward.__func__
    from torch.fx.graph_module import _WrappedCall

    wrapper = body._wrapped_call
    if (
        type(body).__bases__ != (torch.fx.GraphModule,)
        or body._compiled_call_impl is not None
        or "_call_impl" in vars(body)
        or "_wrapped_call" in vars(body)
        or type(body)._call_impl is not torch.nn.Module._call_impl
        or type(wrapper) is not _WrappedCall
        or wrapper.cls is not type(body)
        or wrapper.cls_call is not None
        or not _same_code(type(body).__call__.__code__, type(generated).__call__.__code__)
    ):
        return False
    if not _same_code(actual.__code__, expected.__code__):
        return False
    return all(
        actual.__globals__.get(i.argval) is expected.__globals__.get(i.argval)
        for i in dis.get_instructions(expected)
        if i.opname == "LOAD_GLOBAL"
    )


def _body(gm, node, torch):
    from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled
    from torch.fx.immutable_collections import immutable_dict, immutable_list
    from torch.nn.modules import module

    # Module.__call__ executes global hooks too; their closures are opaque and
    # can write constants even when the wrapper receives none of those values.
    if any(
        getattr(module, name, True)
        for name in (
            "_global_forward_pre_hooks",
            "_global_forward_hooks",
            "_global_backward_pre_hooks",
            "_global_backward_hooks",
        )
    ):
        return None

    if (
        node.op != "call_function"
        or node.target is not wrap_with_set_grad_enabled
        or node.kwargs
        or len(node.args) < 2
        or type(node.args[0]) is not bool
    ):
        return None
    reference = node.args[1]
    if (
        not isinstance(reference, torch.fx.Node)
        or reference.op != "get_attr"
        or reference.args
        or reference.kwargs
        or set(reference.users) != {node}
    ):
        return None
    body = gm.get_submodule(reference.target)
    if (
        not isinstance(body, torch.fx.GraphModule)
        or body._parameters
        or body._buffers
        or body._modules
        or body._forward_hooks
        or body._forward_pre_hooks
        or body._backward_hooks
        or body._backward_pre_hooks
    ):
        return None
    nodes = list(body.graph.nodes)
    if not 1 <= len(nodes) <= 256 or len(nodes[-1].args) != 1 or nodes[-1].op != "output":
        return None
    placeholders = [n for n in nodes if n.op == "placeholder"]
    if len(placeholders) != len(node.args) - 2 or any(n.args or n.kwargs for n in placeholders):
        return None
    owned = set(nodes)

    def closed(value):
        if isinstance(value, torch.fx.Node):
            return value in owned
        if type(value) in (tuple, list, immutable_list):
            return all(closed(v) for v in value)
        if type(value) in (dict, immutable_dict):
            return all(type(k) is str and closed(v) for k, v in value.items())
        return type(value) in (
            int,
            float,
            bool,
            str,
            type(None),
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
        )

    for child in nodes:
        if not closed(child.args) or not closed(child.kwargs):
            return None
        if child.op in ("placeholder", "output"):
            continue
        if (
            child.op != "call_function"
            or not isinstance(child.target, torch._ops.OpOverload)
            or child.target.namespace != "aten"
            or child.target._schema.is_mutable
            or torch.Tag.nondeterministic_seeded in child.target.tags
        ):
            return None
    outputs = nodes[-1].args[0]
    if type(outputs) is not tuple or not outputs or not all(isinstance(v, torch.fx.Node) for v in outputs):
        return None
    if not _forward_matches_graph(body, torch):
        return None
    return len(outputs)


def closed_grad_nodes(gm, nodes, torch):
    """Only exact wrappers and bounded tuple projections; unknown forms stay opaque."""
    result = set()
    for node in nodes:
        try:
            count = _body(gm, node, torch)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            count = None
        if count is None:
            continue
        projections = []
        for user in node.users:
            if (
                user.op != "call_function"
                or user.target is not operator.getitem
                or user.kwargs
                or len(user.args) != 2
                or user.args[0] is not node
                or type(user.args[1]) is not int
                or not 0 <= user.args[1] < count
            ):
                break
            projections.append(user)
        else:
            result.update((node, *projections))
    return result
