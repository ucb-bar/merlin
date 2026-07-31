"""Lift a matmul inventory into contract-level facts and drive the pipeline.

The reuse structure of a VLA policy like smolVLA: the capture unit is one
flow-matching denoise step, and `select_action` runs ``num_steps`` of them per
control tick — every weight is therefore reused across invocations even when it
appears in a single matmul per step. That is exactly the `repeated_rhs_matmul`
pattern, with real layer shapes.

:func:`drive_pipeline` runs the existing core-dialect lowering with a record's real
(M, K, N); :func:`record_dse` measures residency variants and emits `dse` IR + a
``dse_result.yaml``-shaped dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .linalg_mlir import MatmulRecord

DTYPE_BYTES = {"f32": 4, "bf16": 2, "f16": 2, "i8": 1, "i32": 4}


@dataclass
class WeightReuseFact:
    """Contract-level facts about one model weight used by matmuls."""

    weight_name: str
    weight_arg_index: int
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    uses_per_invocation: int
    reused_across_invocations: bool   # denoise loop: weights live across steps
    gemm_shapes: list[tuple[int, int, int]]


def lift_weight_reuse(inventory: list[MatmulRecord],
                      invocations: int = 10) -> list[WeightReuseFact]:
    """Group matmuls by resolved weight and lift immutability/reuse facts."""
    by_weight: dict[int, list[MatmulRecord]] = {}
    for rec in inventory:
        if rec.weight_arg_index is not None:
            by_weight.setdefault(rec.weight_arg_index, []).append(rec)
    facts = []
    for idx, recs in sorted(by_weight.items()):
        r0 = recs[0]
        elems = 1
        for d in r0.rhs_shape:
            elems *= d
        facts.append(WeightReuseFact(
            weight_name=r0.weight_name or f"arg{idx}",
            weight_arg_index=idx,
            shape=r0.rhs_shape,
            dtype=r0.dtype,
            nbytes=elems * DTYPE_BYTES.get(r0.dtype, 4),
            uses_per_invocation=len(recs),
            reused_across_invocations=invocations > 1,
            gemm_shapes=[(r.m, r.k, r.n) for r in recs
                         if r.m is not None],
        ))
    return facts


def select_gemm(inventory: list[MatmulRecord],
                max_macs: int | None = None) -> MatmulRecord:
    """A representative 2-D weight GEMM (smallest that fits the MAC budget)."""
    candidates = [r for r in inventory
                  if r.m is not None and r.weight_arg_index is not None]
    if not candidates:
        raise ValueError("no 2-D weight matmuls in the inventory")
    candidates.sort(key=lambda r: r.m * r.k * r.n)
    if max_macs is not None:
        fitting = [r for r in candidates if r.m * r.k * r.n <= max_macs]
        if fitting:
            return fitting[-1]   # largest that fits the budget
    return candidates[0]


def drive_pipeline(record: MatmulRecord, *, target: str, reuse: int = 2):
    """Lower the record's real GEMM shape through the core-dialect pipeline.

    The integer pipeline executes the layer's i8 deployment GEMM (the shapes are
    identical across the fp32/int8 variants of the model); dtype provenance from the
    capture is preserved on the returned result via ``record``. ``target`` is required —
    the caller names the backend to lower onto (no default; a frontend does not pick a
    target for you).
    """
    from ..xdsl_dialects.lowering import lower_repeated_rhs_matmul

    return lower_repeated_rhs_matmul(reuse=reuse, m=record.m, k=record.k, n=record.n,
                                     target=target)


def baseline_variant(cb: dict[str, Any]) -> dict[str, Any]:
    """The same command buffer without residency: no pack/evict, RHS used directly."""
    import copy

    out = copy.deepcopy(cb)
    packed_src = {}
    for cmd in out["commands"]:
        if cmd["opcode"] == "RES_PACK":
            packed_src[cmd["operands"]["dst"]] = cmd["operands"]["src"]
    cmds = []
    for cmd in out["commands"]:
        if cmd["opcode"] in ("RES_PACK", "EVICT"):
            continue
        ops = cmd.get("operands", {})
        if cmd["opcode"] in ("MATMUL_RESIDENT", "MATMUL") and ops.get("rhs") in packed_src:
            cmd = dict(cmd)
            cmd["opcode"] = "MATMUL"
            cmd["operands"] = dict(ops, rhs=packed_src[ops["rhs"]])
        cmds.append(cmd)
    out["commands"] = cmds
    return out


def record_dse(record: MatmulRecord, resident_cb: dict[str, Any],
               spike_metrics: dict[str, Any] | None = None,
               workload: str = "smolvla_gemm") -> dict[str, Any]:
    """Measure residency variants on the engine (+ optional spike numbers) and emit
    `dse` IR plus a ``dse_result``-shaped dict. Returns
    {module, results, regime, candidate}."""
    from merlin.runtime import simulate

    from ..xdsl_dialects import _common, dse

    base_cb = baseline_variant(resident_cb)
    runs = {
        "software_visible": simulate(resident_cb)["metrics"],
        "baseline": simulate(base_cb)["metrics"],
    }
    if spike_metrics is not None:
        runs["software_visible_spike"] = spike_metrics

    sv, base = runs["software_visible"], runs["baseline"]
    # Exploitability on this evidence: cycle/byte advantage of the visible variant.
    if sv["cycles"] < base["cycles"] or sv["bytes_read"] < base["bytes_read"]:
        regime = "exploitable"
    elif sv["cycles"] == base["cycles"]:
        regime = "marginal"
    else:
        regime = "irrelevant"

    module = None
    if _common.HAS_XDSL:
        from xdsl.ir import Block, Region
        from xdsl.dialects.builtin import (ArrayAttr, DictionaryAttr, FunctionType,
                                           IntegerAttr, ModuleOp, StringAttr)
        from xdsl.dialects.func import FuncOp, ReturnOp

        blk = Block()
        cand = dse.CandidateOp(
            result_types=[dse.InterfaceCandidateType(StringAttr("resident_packed_tensor"))],
            properties={
                "candidate_name": StringAttr("resident_packed_tensor"),
                "interface_ops": ArrayAttr([StringAttr("interface.resident_pack"),
                                            StringAttr("interface.matmul"),
                                            StringAttr("interface.resident_evict")]),
                "justified_by": ArrayAttr([StringAttr(workload)]),
            })
        ops = [cand]
        for variant_name in ("baseline", "software_visible"):
            metrics = runs[variant_name]
            ops.append(dse.ResultOp(operands=[cand.candidate], properties={
                "variant": dse.VariantAttr(_common.Visibility(variant_name)),
                "workload": StringAttr(workload),
                "backend": StringAttr("simulator"),
                "metrics": DictionaryAttr({
                    name: IntegerAttr(int(metrics[name]), 64)
                    for name in ("cycles", "bytes_moved", "bytes_read",
                                 "resident_hits", "resident_misses")}),
            }))
        ops.append(dse.RegimeTagOp(operands=[cand.candidate], properties={
            "regime": dse.RegimeAttr(dse.Regime(regime)),
            "reason": StringAttr(
                f"measured on {workload} "
                f"({record.m}x{record.k}x{record.n}, weight={record.weight_name})")}))
        ops.append(ReturnOp())
        blk.add_ops(ops)
        module = ModuleOp([FuncOp("dse_records", FunctionType.from_lists([], []),
                                  Region([blk]))])

    results = {
        "candidate": "resident_packed_tensor",
        "workload": workload,
        "gemm": {"m": record.m, "k": record.k, "n": record.n,
                 "weight": record.weight_name, "capture_dtype": record.dtype},
        "variants": {name: {k: v for k, v in metrics.items()
                            if not isinstance(v, dict)}
                     for name, metrics in runs.items()},
        "regime": regime,
    }
    return {"module": module, "results": results, "regime": regime,
            "candidate": "resident_packed_tensor"}
