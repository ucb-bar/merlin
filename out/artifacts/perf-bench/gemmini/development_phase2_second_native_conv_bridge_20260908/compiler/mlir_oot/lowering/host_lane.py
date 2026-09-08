"""Compiler-generated host programs using the canonical whole-program pointer ABI.

The source owns every operation; the command list is empty when no accelerator is used.
Input/output pointer order is explicit and float readback needs no integer carrier command.
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

    # Pure host programs use the same explicit whole-program pointer ABI as mixed
    # programs. Empty accelerator commands are honest: the emitted host program owns
    # all computation, including float results, without carrier matrices or RES_PACK.
    cb["kernel_abi"] = {
        "kind": "whole_program",
        "args": [{"tensor": name, "access": "write" if name in out_names else "read"}
                 for name in order],
        "outputs": out_names,
    }
    cb["params"]["host_lane_program_emitted"] = True
    return Plan(target, buffers, [], cb, list(order))


def host_instrs(plan: Plan, module, wl: LinalgWorkload) -> list:
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
    if cost > HOST_LINALG_ELEMENT_BUDGET:
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
                   "regions_placed": [r.region_id for r in wl.host_regions]},
                  bufs=args + outs)]
