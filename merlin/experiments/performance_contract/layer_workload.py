#!/usr/bin/env python3
"""Drive :mod:`merlin.perf.workload_gen` end to end: generate a layer-scale workload, prove its DRAM
footprint does not wrap, project its wall clock, run it on an RTL oracle, and gate the result BIT-EXACT
against a host-computed golden.

This is the experiment edge, so it is the one place that is legitimately about a specific target: the
selection of WHICH of a target's instructions plays each role in the emitted kernel lives here, in
:data:`OP_SELECTIONS`, because a machine with two matrix units and both an overwriting and an
accumulating multiply has four instructions carrying the single derived role ``matmul``, and choosing
between them is a scheduling decision rather than a fact about the hardware. Onboarding a second target
means adding a selection here; the generator itself is unchanged.

Two facts are MEASURED on the tier that will grade the run and then cached, because they are not
portable and getting either wrong returns a plausible number rather than an error:

* the branch contract (immediate scale + delay slots). The two runners for the same machine measured
  here DISAGREE -- the RTL halves the immediate and treats the PC as a word index, its functional model
  reads the immediate as a byte offset -- so a contract probed on one tier is wrong on the other.
* the settle counts. There is no interlock between the scalar stream and the tensor pipelines; the
  probe finds the smallest wait that makes a single tile bit-exact and carries a margin.

Usage::

    layer_workload.py probe   --target T --tier vsim
    layer_workload.py run     --target T --tier vsim --shape 32x1024x512
    layer_workload.py project --target T --tier vsim --shape 416x832x416
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from merlin.common import artifacts as A                     # noqa: E402
from merlin.common import provenance as PV                    # noqa: E402
from merlin.common.paths import ext_path                      # noqa: E402
from merlin.perf import workload_gen as WG                    # noqa: E402
from merlin.targetgen import program_oracle as PO             # noqa: E402

# Which of each target's instructions plays each role. Derived candidates for any target are printed by
# ``probe --show-candidates``; the choice among same-role instructions (which matrix unit, overwriting or
# accumulating multiply, which accumulate format's readout) is this experiment's, not the hardware's.
OP_SELECTIONS = {
    "atlas": dict(
        add="ADD", add_imm="ADDI", load_upper="LUI", branch_ne="BNE", stall="DELAY", halt="EBREAK",
        dma_load="DMA_LOAD_CH0", dma_store="DMA_STORE_CH0", dma_wait="DMA_WAIT_CH0",
        tile_load="VLOAD", tile_store="VSTORE", transpose="VTRPOSE_XLU",
        weight_push="VMATPUSH_WEIGHT_MXU0", contract="VMATMUL_MXU0",
        contract_accumulate="VMATMUL_ACC_MXU0", acc_read="VMATPOP_BF16_ACC_MXU0"),
}
# The accumulate format each selection's readout instruction reads. It must agree with the datapath the
# workload's golden models, so it is pinned alongside the selection rather than inferred.
ACCUM_SELECTIONS = {"atlas": "bf16"}
# The hardware revisions an atlas cycle/correctness verdict is about (merlin/contract/hardware_pins.yaml).
PINS = {"atlas": ("atlas_npu", "atlas_fp_units", "atlas_npu_model")}
# The model project that owns the operand byte layout, per target contract.
TIERS = ("func", "arc", "vsim")


def kernel_ops(target: str) -> WG.KernelOps:
    sel = OP_SELECTIONS.get(target)
    if sel is None:
        raise SystemExit(
            f"no instruction-role selection for target {target!r}; add one to OP_SELECTIONS "
            f"(run `probe --show-candidates` for the derived menu)")
    return WG.KernelOps(**sel)


def facts_for(target: str) -> WG.MachineFacts:
    return WG.machine_facts(target, accum_dtype=ACCUM_SELECTIONS.get(target))


def model_ext_for(target: str) -> str:
    from merlin.targetgen.capsule_runner import _endpoint_of
    _kind, ext = _endpoint_of(target)
    if not ext:
        raise SystemExit(f"{target!r}: its contract declares no runner.model_ext; the program oracle "
                         f"needs the model project to lay out operands")
    return ext


def make_runner(target: str, tier: str, workdir: Path):
    """``(cb, kernel_s, max_cycles, timeout) -> oracle result``. The tier is the caller's, and it travels
    with every number this driver reports: the three do not agree about control flow."""
    model_ext = model_ext_for(target)

    def run(cb, kernel_text, max_cycles, timeout=7200):
        wd = workdir / f"{tier}_{int(time.time() * 1000)}"
        wd.mkdir(parents=True, exist_ok=True)
        ks = wd / "kernel.S"
        ks.write_text(kernel_text)
        if tier == "func":
            return PO.run_program_functional_oracle(target, model_ext=model_ext, cb=cb, kernel_s=ks,
                                                    max_cycles=max_cycles, workdir=wd, timeout=timeout)
        if tier == "arc":
            return PO.run_program_oracle(target, model_ext=model_ext, cb=cb, kernel_s=ks,
                                         max_cycles=max_cycles, workdir=wd, timeout=timeout)
        if tier == "vsim":
            return PO.run_program_verilator_oracle(target, model_ext=model_ext,
                                                   vsim_dir=ext_path(f"{target}_vsim"), cb=cb,
                                                   kernel_s=ks, max_cycles=max_cycles, workdir=wd,
                                                   timeout=timeout)
        raise SystemExit(f"unknown tier {tier!r}; one of {TIERS}")
    return run


# ---------------------------------------------------------------------------------------------------
# the measured machine contract, cached per (target, tier)
# ---------------------------------------------------------------------------------------------------
def contract_path(target: str, tier: str) -> Path:
    return A.cache_dir(f"perf_workload/{target}") / f"machine_contract_{tier}.json"


def probe(target: str, tier: str, workdir: Path, *, force: bool = False) -> dict:
    """Measure the branch contract and the settle counts on ``tier`` and cache them."""
    p = contract_path(target, tier)
    if p.is_file() and not force:
        return json.loads(p.read_text())
    facts, ops = facts_for(target), kernel_ops(target)
    runner = make_runner(target, tier, workdir)
    # a minimal command buffer: the probe kernel computes nothing, but the oracle needs an output region
    probe_cb = {"tensors": {"probe": {"role": "output", "shape": [1, facts.tile.cols],
                                      "dtype": facts.accum_dtype,
                                      "base": facts.dram_base + 0x40}}}

    def run_kernel(src, max_cycles):
        try:
            return int(runner(probe_cb, src, max_cycles)["cycles"])
        except PO.ProgramDidNotHalt:
            return None

    cf = WG.probe_control_flow(facts, ops, run_kernel)

    def run_matmul(plan, max_cycles):
        return _run_plan(runner, plan, max_cycles)[1]

    from merlin.targetgen import corpus_operands as CO
    import numpy as np
    e = facts.tile.rows
    A0 = np.asarray(CO.operand_values((e, e), facts.operand_dtype, salt=0xA7),
                    dtype=np.float32).reshape(e, e)
    W0 = np.asarray(CO.operand_values((e, e), facts.operand_dtype, salt=0x5E),
                    dtype=np.float32).reshape(e, e)
    settle = WG.probe_settle(facts, ops, cf, run_matmul, operands=(A0, W0))
    out = {"target": target, "tier": tier,
           "control_flow": {"branch_imm_scale": cf.branch_imm_scale, "delay_slots": cf.delay_slots,
                            "provenance": cf.provenance},
           "settle": {"tensor": settle.tensor, "mxu": settle.mxu, "vpu": settle.vpu,
                      "provenance": settle.provenance}}
    p.write_text(json.dumps(out, indent=2) + "\n")
    return out


def load_contract(target: str, tier: str, workdir: Path) -> tuple[WG.ControlFlow, WG.Settle]:
    d = probe(target, tier, workdir)
    c, s = d["control_flow"], d["settle"]
    return (WG.ControlFlow(c["branch_imm_scale"], c["delay_slots"], c["provenance"]),
            WG.Settle(s["tensor"], s["mxu"], s["vpu"], s["provenance"]))


# ---------------------------------------------------------------------------------------------------
# running a plan
# ---------------------------------------------------------------------------------------------------
def _run_plan(runner, plan: WG.MatmulPlan, max_cycles: int, timeout: int = 7200):
    cb = plan.command_buffer()
    for p in plan.placements:
        raw = plan.operands.get(p.name + "_bytes")
        if raw is not None:
            cb["tensors"][p.name]["preload_b64"] = base64.b64encode(raw).decode()
    t0 = time.time()
    res = runner(cb, plan.kernel_s, max_cycles, timeout)
    res["wall_seconds"] = time.time() - t0
    return res, next(iter(res["outputs"].values()))


def build_plan(target: str, tier: str, shape: str, workdir: Path, *,
               with_operands: bool = True) -> WG.MatmulPlan:
    """The plan for ``MxKxN``. ``with_operands`` False builds the PROGRAM only -- the shape, the
    addresses and the kernel -- which is all a footprint check or a wall-clock projection needs, and
    skips synthesising a layer's worth of operands and its reference."""
    import numpy as np

    from merlin.targetgen import corpus_operands as CO
    m, k, n = (int(x) for x in shape.split("x"))
    facts, ops = facts_for(target), kernel_ops(target)
    cf, settle = load_contract(target, tier, workdir)
    A = W = None
    if with_operands:
        A = np.asarray(CO.operand_values((m, k), facts.operand_dtype, salt=0xA7),
                       dtype=np.float32).reshape(m, k)
        W = np.asarray(CO.operand_values((k, n), facts.operand_dtype, salt=0x5E),
                       dtype=np.float32).reshape(k, n)
    return WG.plan_matmul(facts, ops, m=m, k=k, n=n, control_flow=cf, settle=settle, A=A, W=W)


def describe(plan: WG.MatmulPlan, rep: WG.AliasReport, *, rate: float | None) -> dict:
    mt, kt, nt = plan.tiles
    passes = mt * kt * nt
    d = {
        "shape": {"m": plan.m, "k": plan.k, "n": plan.n},
        "tiles": {"m": mt, "k": kt, "n": nt, "passes": passes},
        "macs": plan.total_macs(),
        "kernel_words": len(plan.words),
        "unrolled_words": plan.unrolled_word_estimate(),
        "section_words": dict(plan.section_words),
        "instruction_memory": plan.instruction_memory_fit(),
        "footprint_bytes": rep.footprint_bytes,
        "moved_bytes": plan.moved_bytes(),
        "transfer_amplification": round(plan.transfer_amplification(), 3),
        "alias": {"ok": rep.ok, "window": rep.window, "reason": rep.reason,
                  "wrapped": list(rep.wrapped),
                  "collisions": [list(c) for c in rep.collisions],
                  "spans": [{"name": s.name, "base": s.base, "nbytes": s.nbytes,
                             "reduced_lo": s.reduced_lo, "reduced_hi": s.reduced_hi,
                             "wraps": s.wraps} for s in rep.spans]},
        "control_flow": {"branch_imm_scale": plan.control_flow.branch_imm_scale,
                         "delay_slots": plan.control_flow.delay_slots,
                         "provenance": plan.control_flow.provenance},
        "settle": {"tensor": plan.settle.tensor, "mxu": plan.settle.mxu, "vpu": plan.settle.vpu,
                   "provenance": plan.settle.provenance},
    }
    if rate:
        d["projection"] = {"cycles_per_second": rate,
                           "projected_seconds": None}
    return d


def cmd_probe(args) -> int:
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    if args.show_candidates:
        facts = facts_for(args.target)
        print(json.dumps(WG.candidate_ops(facts.isa), indent=2))
        return 0
    print(json.dumps(probe(args.target, args.tier, wd, force=args.force), indent=2))
    return 0


def cmd_project(args) -> int:
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.target, args.tier, args.shape, wd, with_operands=False)
    rep = WG.alias_report(plan.placements, plan.facts.dram_window)
    d = describe(plan, rep, rate=args.rate)
    mt, kt, nt = plan.tiles
    if args.cycles_per_pass:
        cycles = args.cycles_per_pass * mt * kt * nt
        d["projection"] = {"cycles_per_pass": args.cycles_per_pass, "projected_cycles": cycles,
                           "cycles_per_second": args.rate,
                           "projected_seconds": (cycles / args.rate) if args.rate else None}
    print(json.dumps(d, indent=2))
    return 0 if rep.ok else 1


def cmd_run(args) -> int:
    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    plan = build_plan(args.target, args.tier, args.shape, wd)
    rep = WG.alias_report(plan.placements, plan.facts.dram_window)
    rec = describe(plan, rep, rate=args.rate)
    rec["tier"] = args.tier
    rec["target"] = args.target
    if not rep.ok:
        # A run whose footprint aliases would still return a cycle count. It would be a number for a
        # computation nobody asked for, so it is refused BEFORE the oracle is asked.
        rec["result"] = {"ran": False, "refused_because": rep.reason}
        _emit(args, plan, rec)
        print(json.dumps(rec["result"], indent=2))
        return 1
    runner = make_runner(args.target, args.tier, wd)
    budget = args.max_cycles or plan.suggested_max_cycles()
    try:
        res, out = _run_plan(runner, plan, budget, timeout=args.timeout)
    except PO.ProgramDidNotHalt as e:
        rec["result"] = {"ran": True, "halted": False, "detail": str(e), "budget": budget}
        _emit(args, plan, rec)
        print(json.dumps(rec["result"], indent=2))
        return 1
    ok = plan.matches(out)
    rec["result"] = {"ran": True, "halted": True, "bit_exact": ok, "cycles": int(res["cycles"]),
                     "wall_seconds": round(res["wall_seconds"], 2),
                     "cycles_per_second": round(int(res["cycles"]) / max(res["wall_seconds"], 1e-9), 1),
                     "cycles_per_tile_pass": round(int(res["cycles"]) / max(1, rec["tiles"]["passes"]), 1),
                     "oracle": res.get("oracle"),
                     "gate": f"bit-exact vs {plan.facts.accum_dtype} accumulation"}
    if not ok:
        rec["result"]["divergence"] = plan.divergence(out)
    _emit(args, plan, rec)
    print(json.dumps(rec["result"], indent=2))
    return 0 if ok else 1


def _emit(args, plan: WG.MatmulPlan, rec: dict) -> None:
    """Write the record + the emitted kernel as a versioned product under the generated-output root."""
    pins = {}
    for name in PINS.get(args.target, ()):  # noqa: B007
        try:
            pins[name] = PV.verify(name)
        except Exception as e:  # noqa: BLE001 — an unverifiable pin is recorded as such, never dropped
            rec.setdefault("pin_errors", {})[name] = f"{type(e).__name__}: {e}"
    rec["provenance"] = PV.record(pins=pins, extra={
        "generator": "merlin.perf.workload_gen.plan_matmul",
        "oracle_tier": args.tier,
        "machine_facts": plan.facts.provenance,
    })
    if args.no_product:
        return
    prod = A.new_product("perf-workload", version=1, target=args.target,
                         notes=f"layer-scale generated matmul {plan.m}x{plan.k}x{plan.n} on {args.tier}")
    prod.add_artifact("workload_record.json").write_text(json.dumps(rec, indent=2) + "\n")
    prod.add_artifact("kernel.S").write_text(plan.kernel_s)
    prod.add_artifact("command_buffer.json").write_text(
        json.dumps(plan.command_buffer(), indent=2) + "\n")
    prod.write_manifest()
    print(f"# product: {prod.path}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workdir", default="/scratch/agustin/tmp/perf_workload")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "project", "run"):
        s = sub.add_parser(name)
        s.add_argument("--target", required=True)
        s.add_argument("--tier", default="vsim", choices=TIERS)
        s.add_argument("--workdir", default="/scratch/agustin/tmp/perf_workload")
        if name == "probe":
            s.add_argument("--force", action="store_true")
            s.add_argument("--show-candidates", action="store_true")
            s.set_defaults(func=cmd_probe)
        else:
            s.add_argument("--shape", required=True, help="MxKxN, e.g. 32x1024x512")
            s.add_argument("--rate", type=float, default=None,
                           help="measured cycles/second on this tier, for the wall-clock projection")
            if name == "project":
                s.add_argument("--cycles-per-pass", type=float, default=None)
                s.set_defaults(func=cmd_project)
            else:
                s.add_argument("--max-cycles", type=int, default=None)
                s.add_argument("--timeout", type=int, default=7200)
                s.add_argument("--no-product", action="store_true")
                s.set_defaults(func=cmd_run)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
