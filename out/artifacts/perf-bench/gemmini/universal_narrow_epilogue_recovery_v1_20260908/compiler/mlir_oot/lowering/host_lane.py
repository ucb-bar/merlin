"""Lower a `linalg-on-tensors` module by PLACING its regions, then generating the host lane.

A region this hardware has no datapath for belongs on the HOST lane, and saying so is a routing
decision rather than a refusal: the placement is recorded in `params.lane_placement`, and the
emitted target module carries NO accelerator instruction for a host-placed region (the capsules
in this family forbid the `on_mesh` lane, and that gate is honoured against the decoded stream).

The computation itself is still COMPILER-GENERATED: `codegen/host_linalg.py` lowers the module's
own linalg IR into straight-line f32 code on the scalar lane.  A pure host program uses the ABI's
explicit `whole_program` pointer boundary and an empty accelerator command stream.  Its result is
stored in the output tensor's declared container; the runner transports f32/bf16/f16 bit patterns
as integer words and decodes them from that declaration.
"""
from __future__ import annotations

from typing import Any

from ..frontend.linalg_reader import HOST_LANE, LinalgWorkload
from .plan import Buffer, LoweringDeclined, Plan

#: Element types `codegen/host_linalg.py` can materialise on the scalar lane: the float formats
#: it computes in, and the integer widths it carries exactly in the integer domain.
HOST_LANE_ELEMENT_DTYPES = frozenset({"f32", "bf16", "f16",
                                      "i1", "i8", "i16", "i32", "i64"})


def build(wl: LinalgWorkload, target: str = "gemmini", abi_version: str = "0.1") -> Plan:
    buffers: dict[str, Buffer] = {}
    order: list[str] = []
    in_names: list[str] = []
    out_names: list[str] = []
    for i, (shape, dtype) in enumerate(wl.args):
        name = f"arg{i}"
        buffers[name] = Buffer(name, list(shape) or [1], dtype, "input")
        order.append(name)
        in_names.append(name)
    for i, (shape, dtype) in enumerate(wl.results):
        name = f"Y{i}"
        buffers[name] = Buffer(name, list(shape) or [1], dtype, "output")
        order.append(name)
        out_names.append(name)

    cb: dict[str, Any] = {
        "abi_version": abi_version,
        "target": target,
        "backend": "mlir_oot_xdsl_gemmini",
        "tensors": {n: {"shape": buffers[n].shape, "dtype": buffers[n].dtype,
                        "role": buffers[n].role} for n in order},
        "commands": [],
        "params": {
            "lane_placement": [
                {"region": r.region_id, "family": r.family, "op": r.op, "dtype": r.dtype,
                 "lane": r.lane, "reason": r.reason} for r in wl.regions],
            "host_lane_regions": [r.region_id for r in wl.host_regions],
            "mesh_regions": [r.region_id for r in wl.mesh_regions],
        },
    }

    cb["params"]["lanes"] = {
        "reported": sorted({r.lane for r in wl.regions}),
        "on_mesh": [r.region_id for r in wl.mesh_regions],
        "scalar_rvv_lane": [r.region_id for r in wl.host_regions],
    }

    if wl.whole_model:
        # One generated host task owns the complete top-level source program. Nested linalg
        # regions are owned by their containing operation, as in the mixed plan.
        source_op_count = wl.source_op_count
        cb["params"]["global_program_plan"] = {
            "schema": "host_program_plan_v1",
            "source_op_count": source_op_count,
            "tasks": [{"task_index": 0, "kind": "host",
                       "source_op_indices": list(range(source_op_count)),
                       "reads": list(in_names), "writes": list(out_names)}],
            "entry_bindings": list(in_names), "output_bindings": list(out_names),
            "ownership_granularity": (
                "entry-block operations; nested regions are owned by their parent operation"
            ),
        }

    if not out_names:
        cb["declined"] = {"reason": f"@{wl.entry} returns no tensor to write",
                          "op": wl.regions[0].op or "linalg_on_tensors"}
        return Plan(target, buffers, [], cb, list(order))

    unsupported = sorted({buffers[n].dtype for n in order
                          if buffers[n].dtype not in HOST_LANE_ELEMENT_DTYPES})
    if wl.host_regions and unsupported:
        cb["declined"] = {
            "reason": (
                f"@{wl.entry} places {len(wl.host_regions)} region(s) on the {HOST_LANE} lane, "
                f"and the generated CPU-lane program has no scalar format for the element "
                f"type(s) {unsupported} its operands are declared in; the emitted kernel is "
                f"straight-line single-block code, so there is no lowering to fall back to"),
            "op": wl.regions[0].op or wl.regions[0].family or "linalg_on_tensors",
            "shape": list(buffers[out_names[0]].shape),
        }
        return Plan(target, buffers, [], cb, list(order))

    if not wl.mesh_regions:
        # The accelerator command stream is intentionally empty: every region is on the scalar
        # lane.  This applies equally to an operation slice and a whole model: neither may invent
        # accelerator work merely to make its command stream nonempty.  The explicit boundary is
        # what makes this a complete submitted program rather than a decline, and the declared
        # output dtype tells the runner how to decode its stored bit-pattern words.
        cb["kernel_abi"] = {
            "kind": "whole_program",
            "args": [{"tensor": name, "access": "write" if name in out_names else "read"}
                     for name in order],
            "outputs": list(out_names),
        }
        cb["params"]["host_lane_program_emitted"] = True
        cb["params"]["host_lane_transport"] = "declared_dtype_bit_pattern"
        return Plan(target, buffers, [], cb, list(order))

    # Reaching the host builder with mesh-placed regions means the mixed lowering failed.  Leave
    # the accelerator stream empty so the caller can turn that failure into an explicit decline;
    # fabricating an unrelated carrier matmul would contradict both placement and source ownership.
    return Plan(target, buffers, [], cb, list(order))


def host_instrs(plan: Plan, module, wl: LinalgWorkload, *, input_prologue=None) -> list:
    """The instruction stream for a host-placed module: one compiler-generated CPU-lane program.

    No accelerator instruction is scheduled -- the placement said the mesh takes none of this
    module's regions, and the capsules in this family enforce that against the decoded stream.
    """
    from .schedule import Instr

    params = plan.command_buffer.get("params") or {}
    if not plan.kernel_args:
        return []
    if plan.command_buffer.get("declined") and not params.get("host_lane_program_emitted"):
        return []
    func_op = None
    for op in module.walk():
        if op.name == "func.func":
            func_op = op
            break
    if func_op is None:
        return []
    from ..codegen.host_linalg import HOST_LINALG_ELEMENT_BUDGET, estimate_cost

    body = list(func_op.regions[0].blocks[0].ops)
    cost = estimate_cost(body)
    if cost > HOST_LINALG_ELEMENT_BUDGET and not wl.whole_model:
        # Refuse from the extents rather than after emitting up to the budget: an entrypoint that
        # takes minutes to say "no" reads as a timeout, not as a decline.
        raise LoweringDeclined(
            f"the CPU-lane program for @{wl.entry} needs about {cost} straight-line element "
            f"evaluations, past this backend's {HOST_LINALG_ELEMENT_BUDGET} budget; the emitted "
            f"kernel is single-block so there is no loop to roll them into",
            op="host_lane",
            shape=list(plan.buffers[outs[0]].shape) if (outs := [n for n in plan.buffers
                       if plan.buffers[n].role == "output"]) else [])
    args = [n for n in plan.buffers if plan.buffers[n].role == "input"
            and n.startswith("arg")]
    outs = [n for n in plan.buffers if plan.buffers[n].role == "output"]
    return [Instr("host_linalg",
                  {"func": func_op, "arg_buffers": args, "out_buffers": outs,
                   **({"global_task_index": 0} if wl.whole_model else {}),
                   **({"input_prologue": input_prologue} if input_prologue else {}),
                   "regions_placed": [r.region_id for r in wl.host_regions]},
                  bufs=args + outs)]
