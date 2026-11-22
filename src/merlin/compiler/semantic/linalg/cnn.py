"""Translate closed CNN scalar bodies without selecting an implementation target.

Recognizers are ported from J's 77a9bb13 source families. Numerical declarations
remain source premises; these nodes alone do not admit a hardware lowering.
"""

from __future__ import annotations

from merlin.compiler.frontend.patterns.linalg_batch_norm import (
    INPUT_DOMAIN as BATCH_NORM_DOMAIN,
)
from merlin.compiler.frontend.patterns.linalg_batch_norm import (
    NUMERIC_SEMANTICS as BATCH_NORM_SEMANTICS,
)
from merlin.compiler.frontend.patterns.linalg_batch_norm import (
    recognize_bf16_batch_norm,
)
from merlin.compiler.frontend.patterns.linalg_bf16_sigmoid import (
    INPUT_DOMAIN as SIGMOID_DOMAIN,
)
from merlin.compiler.frontend.patterns.linalg_bf16_sigmoid import (
    NUMERIC_SEMANTICS as SIGMOID_SEMANTICS,
)
from merlin.compiler.frontend.patterns.linalg_bf16_sigmoid import (
    recognize_bf16_sigmoid,
)
from merlin.compiler.frontend.patterns.linalg_patterns import InvalidLinalgPattern
from merlin.compiler.frontend.patterns.linalg_window_max import (
    INPUT_DOMAIN as WINDOW_MAX_DOMAIN,
)
from merlin.compiler.frontend.patterns.linalg_window_max import (
    NUMERIC_SEMANTICS as WINDOW_MAX_SEMANTICS,
)
from merlin.compiler.frontend.patterns.linalg_window_max import (
    recognize_bf16_window_max,
)
from merlin.compiler.semantic.linalg.linalg_types import LinalgBridgeError, _type
from merlin.compiler.semantic.model import IndexMap, SemanticNode, pointwise_identity_maps


def prepare(operations):
    """Prove window reductions before consuming their exclusive producer chain."""
    windows, prelude = {}, set()
    for op in operations:
        if (
            op.name != "linalg.generic"
            or len(op.inputs) != 2
            or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1
            or tuple(inner.name for inner in op.regions[0].block.ops)
            != ("arith.cmpf", "arith.cmpf", "arith.ori", "arith.select", "linalg.yield")
        ):
            continue
        try:
            windows[op] = recognize_bf16_window_max(op)
        except InvalidLinalgPattern as exc:
            raise LinalgBridgeError(str(exc)) from exc
        prelude.update(windows[op].prelude)
    return windows, prelude


def translate(state, op, ordinal, window_maxes=None):
    """Return true only for a proved CNN family with its defined source values."""
    from xdsl.dialects.builtin import StringAttr

    if op.name != "linalg.generic":
        return False
    policy = op.attributes.get("merlin.numeric_semantics")
    semantics = policy.data if isinstance(policy, StringAttr) else None
    pool = (window_maxes or {}).get(op)
    env, nodes = state.env, state.nodes
    try:
        if pool is not None:
            (kh, kw), (sh, sw), (dh, dw), (ph, pw) = pool.window, pool.stride, pool.dilation, pool.padding
            operands = (pool.source,)
            operation = "bf16_window_max"
            attrs = (
                ("input_domain", WINDOW_MAX_DOMAIN),
                ("numeric_semantics", WINDOW_MAX_SEMANTICS),
                ("window_h", kh),
                ("window_w", kw),
            )
            maps = (
                IndexMap(
                    6,
                    ((1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, sh, 0, dh, 0), (0, 0, 0, sw, 0, dw)),
                    (0, 0, -ph, -pw),
                ),
                IndexMap(6, tuple(tuple(int(axis == loop) for loop in range(6)) for axis in range(4)), (0,) * 4),
            )
        elif semantics == BATCH_NORM_SEMANTICS:
            norm = recognize_bf16_batch_norm(op)
            roles = tuple(role for role in ("mean", "var", "weight", "bias") if role in norm.channel)
            operands = (norm.source, *(norm.channel[role] for role in roles))
            operation = "bf16_batch_norm_inference"
            attrs = (
                ("channel_operands", ",".join(roles)),
                ("epsilon_f32_bits", norm.epsilon_bits),
                ("input_domain", BATCH_NORM_DOMAIN),
                ("numeric_semantics", semantics),
            )
            rank = len(_type(op.results[0].type).shape)
            identity = pointwise_identity_maps(rank, 1)[0]
            channel = IndexMap(rank, (tuple(int(axis == 1) for axis in range(rank)),), (0,))
            maps = (identity, *(channel,) * len(roles), identity)
        elif semantics == SIGMOID_SEMANTICS:
            operands = (recognize_bf16_sigmoid(op),)
            operation = "bf16_sigmoid"
            attrs = (("input_domain", SIGMOID_DOMAIN), ("numeric_semantics", semantics))
            maps = pointwise_identity_maps(len(_type(op.results[0].type).shape), 1)
        else:
            return False
    except InvalidLinalgPattern as exc:
        raise LinalgBridgeError(str(exc)) from exc
    if any(env.get(value) is None for value in operands):
        raise LinalgBridgeError("CNN operation reads an undefined or uninitialized tensor")
    node_id = f"op{ordinal}r0"
    nodes.append(
        SemanticNode(
            node_id,
            operation,
            tuple(env[value] for value in operands),
            _type(op.results[0].type),
            attrs=attrs,
            index_maps=maps,
        )
    )
    env[op.results[0]] = node_id
    return True
