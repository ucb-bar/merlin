"""Each compute group of a captured model as OUR schedule, beside the vendor call it replaces.

``group_model_program`` renders one vendor library call per compute group, and that program is the
measurement this whole effort is aimed at: 28,084,005 cycles per image, 1.17x the same-design vendor
library, against 1,709,045,472 for the compiled program. The number is only worth what the route that
produced it is, so this does NOT replace the vendor call -- it adds a second renderer for the same
program, so the two run the same groups over the same buffers on the same harness and the difference
between them is the schedule and nothing else.

A step whose recipe refuses it keeps its vendor call and says why. That is the point rather than a
shortfall: a program that silently fell back would report our schedule's cycles for the library's
work, and the mix is what tells us which shapes we can actually express.
"""

from __future__ import annotations

import importlib
from typing import Any

from merlin.runtime.backends import base
from merlin.sched.codegen import emit_c_function
from merlin.sched.ir import TensorArg
from merlin.sched.isa import IsaError

#: The C types the generated program declares its buffers with, by the dtype our TensorArgs carry.
_CAST = {"i8": "elem_t", "i32": "acc_t"}


class _Params:
    """Parameter names for the emitted function, and the buffer each one is passed at the call site.

    The two must not be the same string. A TensorArg's name becomes a C PARAMETER name, and the
    program's buffers include ``IMAGE``, which is a macro (``((const elem_t *)IMAGE_DATA)``) -- so a
    parameter named after it is macro-expanded inside the parameter list and the file does not
    compile. Naming parameters by ROLE also stops every emitted function from shadowing the globals
    it sits beside.
    """

    def __init__(self) -> None:
        self.binding: dict[str, str] = {}

    def arg(self, role: str, buffer: str, shape: tuple[int, ...], dtype: str, access: str) -> TensorArg:
        self.binding[role] = buffer
        return TensorArg(role, shape, dtype, access)


def _conv(step: dict[str, Any], sched, facts) -> tuple[Any, _Params]:
    pool = step.get("pool") or {"size": 0, "stride": 0, "padding": 0}
    if step.get("scale") is None:
        raise IsaError("the group leaves as an accumulator; the convolution recipe stores a narrow readout")
    ci, n, k = step["ci"], step["n"], step["kernel"]
    out_dim = step["out_dim"]
    # The buffer the group writes is the POOLED extent when the readout pools -- the same extent the
    # extractor sized it at, computed the library's own way rather than carried separately.
    final_dim = out_dim
    if pool["stride"]:
        final_dim = (out_dim + 2 * pool["padding"] - pool["size"]) // pool["stride"] + 1
    p = _Params()
    ops = {
        "input": p.arg("p_in", step["in"], (1, step["in_dim"], step["in_dim"], ci), "i8", "read"),
        "weights": p.arg("p_w", step["weight"], (k, k, ci, n), "i8", "read"),
        "output": p.arg("p_out", step["out"], (1, final_dim, final_dim, n), "i8", "write"),
    }
    if step.get("bias"):
        ops["bias"] = p.arg("p_bias", step["bias"], (n,), "i32", "read")
    kernel = sched.conv_reference(
        name=f"g{step['group']}",
        batch=1,
        in_dim=step["in_dim"],
        in_channels=ci,
        out_channels=n,
        kernel=k,
        stride=step["stride"],
        padding=step["padding"],
        operands=ops,
        relu=bool(step["relu"]),
        scale=float(step["scale"]),
        facts=facts,
        pool_size=int(pool["size"]),
        pool_stride=int(pool["stride"]),
        pool_padding=int(pool["padding"]),
    )
    return kernel, p


def _matmul(step: dict[str, Any], sched, facts) -> tuple[Any, _Params]:
    m, k, n = step["m"], step["k"], step["n"]
    # A group with no readout multiplier leaves as the ACCUMULATOR (a model's final classifier), so its
    # buffer is acc_t and the recipe is asked for the full-width readout by passing no scale.
    scale = None if step.get("scale") is None else float(step["scale"])
    p = _Params()
    ops = {
        "a": p.arg("p_in", step["in"], (m, k), "i8", "read"),
        "b": p.arg("p_w", step["weight"], (k, n), "i8", "read"),
        "c": p.arg("p_out", step["out"], (m, n), "i8" if scale is not None else "i32", "write"),
    }
    if step.get("bias"):
        ops["d"] = p.arg("p_bias", step["bias"], (n,), "i32", "read")
    kernel = sched.matmul_reference(
        name=f"g{step['group']}",
        m=m,
        n=n,
        k=k,
        operands=ops,
        relu=bool(step["relu"]),
        scale=scale,
        facts=facts,
    )
    return kernel, p


def _sum(step: dict[str, Any], sched, facts) -> tuple[Any, _Params]:
    rows, cols = step["rows"], step["cols"]
    p = _Params()
    ops = {
        "a": p.arg("p_lhs", step["lhs"], (rows, cols), "i8", "read"),
        "b": p.arg("p_rhs", step["rhs"], (rows, cols), "i8", "read"),
        "c": p.arg("p_out", step["out"], (rows, cols), "i8", "write"),
    }
    kernel = sched.resadd_reference(
        name=f"g{step['group']}",
        rows=rows,
        cols=cols,
        operands=ops,
        relu=bool(step["relu"]),
        a_scale=float(step["lhs_load"]),
        b_scale=float(step["rhs_load"]),
        c_scale=float(step["readout"]),
        facts=facts,
    )
    return kernel, p


def _mean(step: dict[str, Any], sched, facts) -> tuple[Any, _Params]:
    # The capture's buffer is [window positions] by [channels], which is exactly the plane this
    # contraction reads -- so the ones vector goes on the LEFT and no operand is transposed. The
    # vendor call spells the same reduction the other way round, with a transposed A.
    window, channels = step["window"], step["rows"]
    p = _Params()
    ops = {
        "a": p.arg("p_ones", "ONES", (1, window), "i8", "read"),
        "b": p.arg("p_in", step["in"], (window, channels), "i8", "read"),
        "c": p.arg("p_out", step["out"], (1, channels), "i8", "write"),
    }
    kernel = sched.window_mean_reference(
        name=f"g{step['group']}",
        rows=window,
        channels=channels,
        operands=ops,
        facts=facts,
        # The capture's multiplier already folds the division together with the requantization;
        # recomputing 1/window underneath it would divide twice.
        scale=float(step["multiplier"]),
    )
    return kernel, p


_RECIPES = {"conv2d": _conv, "matmul": _matmul, "sum": _sum, "mean": _mean}


def render_kernels(model: dict[str, Any], *, target: str) -> dict[str, Any]:
    """Emit one C function per group our recipes can express, and say what happened to every group.

    Returns ``{"definitions": <C>, "calls": {group: <C statement>}, "census": [...]}``. A group absent
    from ``calls`` keeps its vendor call; its census row carries the refusal, spelled the way the
    recipe spelled it.
    """
    backend = base.get_backend(target)
    iset = backend.sched_instruction_set()
    sched = importlib.import_module(f"merlin._oot_backends.{target}.{target}_sched")
    facts = iset.facts

    definitions: list[str] = []
    calls: dict[int, str] = {}
    census: list[dict[str, Any]] = []
    for step in model["steps"]:
        group, kind = step["group"], step["kind"]
        build = _RECIPES.get(kind)
        if build is None:
            census.append({"group": group, "kind": kind, "on": "vendor", "why": f"no recipe for {kind!r}"})
            continue
        try:
            kernel, params = build(step, sched, facts)
            symbol = f"gm_g{group}_{kind}"
            definitions.append(emit_c_function(kernel, iset, symbol=symbol))
        except IsaError as exc:
            census.append({"group": group, "kind": kind, "on": "vendor", "why": str(exc)})
            continue
        # The blobs are `const` and the emitted parameters are not; the cast is at the call site so
        # the refusal surface stays the recipe's rather than the C compiler's.
        actuals = ", ".join(f"({_CAST[a.dtype]} *){params.binding[a.name]}" for a in kernel.args)
        calls[group] = f"{symbol}({actuals});"
        census.append({"group": group, "kind": kind, "on": "sched", "recipe": dict(kernel.attrs)["recipe"]})
    return {"definitions": "\n".join(definitions), "calls": calls, "census": census}


def summarize(census: list[dict[str, Any]]) -> str:
    """One line per outcome, so a reader sees the MIX rather than a single headline."""
    ours = [c for c in census if c["on"] == "sched"]
    theirs = [c for c in census if c["on"] != "sched"]
    lines = [f"{len(ours)} of {len(census)} groups on our schedules, {len(theirs)} still on the vendor call"]
    by_reason: dict[str, list[int]] = {}
    for c in theirs:
        by_reason.setdefault(c["why"], []).append(c["group"])
    for why, groups in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {len(groups):3d} group(s): {why}")
    return "\n".join(lines)
