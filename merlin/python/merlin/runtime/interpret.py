"""Host interpreter for a (possibly sliced) dispatch program — the runtime side of "run a section".

Walks a :class:`~merlin.xdsl_dialects.lowering.dispatch_program.DispatchProgram` in DAG order,
invoking each dispatch kernel and evaluating each view op through INJECTED callables, and returns the
program's result buffers. Injection keeps this decoupled and testable: the real path wires
``dispatch_runtime``'s compiled kernels + numpy view evaluation; a test wires plain numpy.

This is what makes a sliced section runnable in isolation: feed the section's boundary inputs (the
upstream regions' outputs, from ``region_goldens.npz``) as the slice's ``arg`` buffers, walk the
sub-DAG, and read the section's outputs (validated against the region's output golden). Whole-model
and single-section runs are the same call — the slice is just a smaller program.
"""
from __future__ import annotations

from typing import Any, Callable


def run_dispatch_program(
    prog,
    inputs: dict[str, Any],
    *,
    invoke_kernel: Callable[[str, list], list],
    eval_view: Callable[[str, list, Any], list] | None = None,
    consts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute ``prog`` and return ``{result_buffer_id: value}``.

    ``inputs`` binds every ``arg`` buffer by id (a slice's boundary inputs come from the region
    goldens). ``invoke_kernel(symbol, [in_values]) -> [out_values]`` runs one compiled kernel;
    ``eval_view(op_name, [in_values], node) -> [out_values]`` evaluates a driver-side glue op
    (required only if the program has view nodes). ``consts`` optionally pre-binds const buffers.
    Fail-closed: a missing arg input or an unhandled view raises rather than fabricating a value.
    """
    env: dict[str, Any] = dict(consts or {})
    env.update(inputs)

    missing = [b.id for b in prog.buffers.values()
               if b.kind == "arg" and b.id not in env]
    if missing:
        raise KeyError(f"run_dispatch_program: no input bound for arg buffer(s) {missing}")

    for node in prog.nodes:
        in_vals = [env[b] for b in node.inputs]
        if node.kind == "dispatch":
            outs = invoke_kernel(node.op, in_vals)
        elif eval_view is not None:
            outs = eval_view(node.op, in_vals, node)
        else:
            raise NotImplementedError(
                f"run_dispatch_program: view op {node.op!r} but no eval_view was provided")
        outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
        if len(outs) != len(node.outputs):
            raise ValueError(
                f"{node.op}: produced {len(outs)} value(s) for {len(node.outputs)} output buffer(s)")
        for bid, val in zip(node.outputs, outs):
            env[bid] = val

    return {bid: env[bid] for bid in prog.results}
