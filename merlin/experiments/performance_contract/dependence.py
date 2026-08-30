#!/usr/bin/env python3
"""Drive the dependence layer end to end: derive operand direction on the device, build the
dependence DAG for a generated kernel, and RANK the schedules the emitter can already produce.

This is the experiment edge, so it is the one place that is legitimately about a specific target. Two
selections live here and nowhere else:

* :data:`PROBE_SELECTIONS` -- the handful of instructions the direction probe needs in order to WRITE
  architectural state before it reads it back (materialise a scalar constant, fill a tensor register,
  hold issue, terminate). Which of a target's several ways to do each is a choice, not a fact.
* the shape and the tier a kernel is analysed at, which travel with every number reported.

Everything else -- what an operand means, which register file it names, which instructions depend on
which, how long the critical path is -- is derived by :mod:`merlin.targetgen.isa_direction` and
:mod:`merlin.perf.depgraph` from the target's own ISA model and its own oracle.

Usage::

    dependence.py direction --target T [--jobs 6] [--force]
    dependence.py analyse   --target T [--shape 32x32x32] [--tier vsim]
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from merlin.common import artifacts as A                      # noqa: E402
from merlin.common import provenance as PV                    # noqa: E402
from merlin.perf import depgraph as DG                        # noqa: E402
from merlin.perf import workload_gen as WG                    # noqa: E402
from merlin.targetgen import isa_direction as ID              # noqa: E402
from merlin.targetgen import program_oracle as PO             # noqa: E402

import layer_workload as LW                                   # noqa: E402

# Which instructions the direction probe uses to establish known state, and under how many initial
# states. The seeders put NON-ZERO content into each observable state file before the instruction under
# test runs, because an operation on zeros changes nothing and is indistinguishable from one that does
# nothing. Two preambles, not one, because a file the oracle publishes as PRESENCE ONLY with only a
# couple of slots cannot evidence a write and a read under the same initial state: fill the accumulator
# and every definition into it is hidden (the bank already held something), leave it empty and every
# read of it is hidden (there was nothing to read). The second preamble differs from the first only in
# seeding the accumulator, and the two results are merged keeping whichever resolved each operand.
PROBE_SELECTIONS = {
    "atlas": {
        "accumulator_empty": dict(
            scalar_imm="ADDI", scalar_upper="LUI", stall="DELAY", halt="EBREAK",
            seed_slot_field="vd", attribution_value=2,
            seeders=(("VLI_ALL", {"vd": 1, "imm": 0x3F80}),
                     ("VLI_ALL", {"vd": 2, "imm": 0x4040}),
                     ("VMATPUSH_WEIGHT_MXU0", {"vd": 0, "vs1": 1, "vs2": 0}),
                     ("VSTORE", {"vd": 1, "rs1": 0, "imm": 0}))),
        "accumulator_seeded": dict(
            scalar_imm="ADDI", scalar_upper="LUI", stall="DELAY", halt="EBREAK",
            seed_slot_field="vd", attribution_value=2,
            seeders=(("VLI_ALL", {"vd": 1, "imm": 0x3F80}),
                     ("VLI_ALL", {"vd": 2, "imm": 0x4040}),
                     ("VMATPUSH_WEIGHT_MXU0", {"vd": 0, "vs1": 1, "vs2": 0}),
                     ("VSTORE", {"vd": 1, "rs1": 0, "imm": 0}),
                     ("VMATMUL_MXU0", {"vd": 0, "vs1": 1, "vs2": 0}))),
    },
}


def probe_ops(target: str, settle: int) -> dict[str, ID.ProbeOps]:
    sel = PROBE_SELECTIONS.get(target)
    if not sel:
        raise SystemExit(
            f"no direction-probe selection for target {target!r}; add one to PROBE_SELECTIONS "
            f"(it needs an immediate load, an upper immediate, a stall, a terminator, and seeders "
            f"that make each observable state file non-empty)")
    return {name: ID.ProbeOps(settle=int(settle), **spec) for name, spec in sel.items()}


def settle_for(target: str, tier: str) -> tuple[int, str]:
    """The measured settle for this target/tier, from the workload driver's cached machine contract.

    Raises rather than defaulting: on a machine with no interlock a settle that is too short makes a
    seeded state write vanish, and the probe then reads every instruction as writing nothing."""
    p = LW.contract_path(target, tier)
    if not p.is_file():
        raise SystemExit(
            f"no machine contract cached for {target!r} on tier {tier!r} ({p}); run "
            f"`layer_workload.py probe --target {target} --tier {tier}` first -- the probe needs the "
            f"MEASURED settle, and picking one is the failure this whole layer exists to prevent")
    blob = json.loads(p.read_text())
    s = blob["settle"]
    return int(max(s["tensor"], s["mxu"], s["vpu"])), str(s.get("provenance") or "")


def probe_budget(preambles: dict) -> int:
    """A cycle budget sized to the probe programs themselves: every settle the preamble emits, the two
    after the instruction under test, and room for the instructions in between."""
    return int(max((len(o.seeders) + 2) * int(o.settle) + 8 * int(o.scalar_seeds) + 256
                   for o in preambles.values()))


def direction_path(target: str, tier: str) -> Path:
    return A.cache_dir(f"perf_depgraph/{target}") / f"operand_direction_{tier}.json"


# ---------------------------------------------------------------------------------------------------
# 3.1 -- derive operand direction on the device
# ---------------------------------------------------------------------------------------------------
def derive_direction(target: str, tier: str, workdir: Path, *, jobs: int = 6,
                     force: bool = False) -> ID.DirectionModel:
    path = direction_path(target, tier)
    if path.is_file() and not force:
        return ID.DirectionModel.from_json(json.loads(path.read_text()))
    parts = sorted(path.parent.glob(f"{path.stem}__*{path.suffix}"))
    if parts and not force:
        model = None
        for part in parts:
            m = ID.DirectionModel.from_json(json.loads(part.read_text()))
            model = m if model is None else model.merge(m)
        if model is not None:
            path.write_text(json.dumps(model.to_json(), indent=1))
            return model

    facts = LW.facts_for(target)
    settle, settle_prov = settle_for(target, tier)
    preambles = probe_ops(target, settle)
    model_ext = LW.model_ext_for(target)
    cb = {"tensors": {"probe": {"role": "output", "shape": [1, facts.tile.cols],
                                "dtype": facts.accum_dtype, "base": facts.dram_base + 0x40}}}
    counter = {"n": 0}
    # A probe program is a preamble, one instruction under test, and a couple of settles -- a few
    # thousand cycles at most. The budget is sized to that rather than left at a generic ceiling,
    # because the programs that do NOT halt (a branch under test that jumps back over the preamble)
    # are the expensive ones, and every cycle past the point where a healthy probe would already have
    # finished is spent proving something the first few hundred already established.
    budget = probe_budget(preambles)

    def run_probe(kernel_s: str) -> dict:
        counter["n"] += 1
        wd = workdir / f"probe_{counter['n']:06d}_{time.time_ns() & 0xFFFFFF:06x}"
        wd.mkdir(parents=True, exist_ok=True)
        ks = wd / "kernel.S"
        ks.write_text(kernel_s)
        try:
            return PO.run_program_debug(target, model_ext=model_ext, cb=cb, kernel_s=ks,
                                        dump_regions=[], state_summary=True, max_cycles=budget,
                                        workdir=wd, timeout=600)
        except PO.ProgramDidNotHalt:
            return {"halted": False, "halt_reason": "did not halt"}

    names = sorted(facts.isa.by_mnemonic)
    shards = [s for s in (names[i::max(1, jobs)] for i in range(max(1, jobs))) if s]
    t0 = time.time()
    model: ID.DirectionModel | None = None
    for preamble_name, ops in preambles.items():
        merged: dict[str, dict] = {}
        refused: dict[str, str] = {}
        provenance = ""
        with _cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = [pool.submit(ID.derive_directions, facts.isa, ops, run_probe, mnemonics=s)
                       for s in shards]
            for fut in futures:
                part = fut.result()
                merged.update(part.by_mnemonic)
                refused.update(part.refused)
                provenance = part.provenance
        part_model = ID.DirectionModel(target=target, by_mnemonic=merged, refused=refused,
                                       provenance=f"[{preamble_name}] {provenance}")
        # Each preamble's OWN result is kept beside the merged one. The merge rule is a judgement --
        # which evidence outranks which -- and keeping the parts means revisiting that judgement costs
        # a re-merge rather than another few thousand probe programs on the device.
        part_path = path.with_name(f"{path.stem}__{preamble_name}{path.suffix}")
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_text(json.dumps(part_model.to_json(), indent=1))
        model = part_model if model is None else model.merge(part_model)
    assert model is not None
    model = ID.DirectionModel(
        target=target, by_mnemonic=model.by_mnemonic, refused=model.refused,
        provenance=(f"{model.provenance}; tier={tier}, settle={settle} ({settle_prov}); "
                    f"{counter['n']} probe programs in {time.time() - t0:.0f}s"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_json(), indent=1))
    return model


# ---------------------------------------------------------------------------------------------------
# 3.2-3.4 -- the dependence graph over a generated kernel, and the ranking
# ---------------------------------------------------------------------------------------------------
def issue_model(target: str, tier: str, workdir: Path) -> DG.IssueModel:
    """Measure what one instruction and one stall unit cost on ``tier``, and cache it.

    Two points per parameter: a single measurement cannot separate a per-instruction rate from the
    program's fixed overhead, and the whole schedule comparison is a difference of these two numbers.
    """
    path = A.cache_dir(f"perf_depgraph/{target}") / f"issue_model_{tier}.json"
    if path.is_file():
        blob = json.loads(path.read_text())
        return DG.IssueModel(**blob)
    facts = LW.facts_for(target)
    ops = LW.kernel_ops(target)
    settle, _ = settle_for(target, tier)
    probe = probe_ops(target, settle)["accumulator_empty"]
    runner = LW.make_runner(target, tier, workdir)
    cb = {"tensors": {"probe": {"role": "output", "shape": [1, facts.tile.cols],
                                "dtype": facts.accum_dtype, "base": facts.dram_base + 0x40}}}

    # The tail after the stall is REQUIRED, not padding: a stall in the terminator's shadow was
    # measured to cost one cycle whatever its immediate, so a probe without a tail measures every
    # wait in the program as free.
    tail = 4

    def build_padding(n_instructions: int, stall_cycles: int) -> str:
        e = ID._Emitter(facts.isa, probe)
        for _ in range(int(n_instructions)):
            e.emit(ops.add_imm, rd=0, rs1=0, imm=0)
        if stall_cycles:
            e.settle(int(stall_cycles))
        for _ in range(tail):
            e.emit(ops.add_imm, rd=0, rs1=0, imm=0)
        e.emit(ops.halt)
        return e.kernel_s()

    def run_kernel(src: str):
        try:
            return int(runner(cb, src, 200000)["cycles"])
        except PO.ProgramDidNotHalt:
            return None

    model = DG.probe_issue_model(run_kernel, build_padding, tier=tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.__dict__, indent=1))
    return model


def _regions_with_trips(program: DG.Program, plan) -> tuple[DG.Region, ...]:
    """How many times each straight-line region of the emitted plan actually runs.

    The trip counts come from the PLAN's own tile counts, not from anything read off the program: the
    generator emits one backward branch per axis whose tile count exceeds one, innermost axis first,
    and the peeled first K-step means the K body runs one time fewer than there are K tiles. A region
    inside two loops runs the product of both, which is the whole reason a saving inside the innermost
    body is worth more than the same saving in the prologue.
    """
    mt, kt, nt = plan.tiles
    # Loops as the generator nests them, innermost first, each with the number of times its body runs.
    declared = [(kt - 1) if kt > 1 else None, nt if nt > 1 else None, mt if mt > 1 else None]
    loops = sorted(((int(i.branch_target), i.index) for i in program.instructions
                    if i.branches_backward), key=lambda pair: -pair[0])
    counts = [c for c in declared if c is not None]
    if len(counts) != len(loops):
        return program.regions          # the structure does not match the plan; claim nothing
    spans = [(start, branch, count) for (start, branch), count in zip(loops, counts)]
    out = []
    for region in program.regions:
        trips = 1
        for start, branch, count in spans:
            if start <= region.start and region.end <= branch + 1:
                trips *= int(count)
        out.append(DG.Region(name=region.name, start=region.start, end=region.end, trips=trips))
    return tuple(out)


def analyse(target: str, tier: str, shape: tuple[int, int, int], workdir: Path,
            *, jobs: int = 6, measured_cycles: float | None = None) -> dict:
    facts = LW.facts_for(target)
    ops = LW.kernel_ops(target)
    directions = derive_direction(target, tier, workdir, jobs=jobs)
    issue = issue_model(target, tier, workdir)
    contract = json.loads(LW.contract_path(target, tier).read_text())
    cf = WG.ControlFlow(int(contract["control_flow"]["branch_imm_scale"]),
                        int(contract["control_flow"]["delay_slots"]),
                        str(contract["control_flow"]["provenance"]))
    settle = WG.Settle(**{k: v for k, v in contract["settle"].items()})
    m, k, n = shape
    plan = WG.plan_matmul(facts, ops, m=m, k=k, n=n, control_flow=cf, settle=settle)

    program = DG.program_from_plan(plan, directions)
    program = DG.Program(instructions=program.instructions, effects=program.effects,
                         regions=_regions_with_trips(program, plan), roles=program.roles,
                         capacities=program.capacities)
    # The one register-file capacity this target publishes: a tensor register is the compute array's
    # own byte geometry, and the file's slot count is what the ISA's own destination field can name.
    vd_bits = len(facts.isa.fields_of(ops.tile_load).get("vd") or ())
    capacities = {}
    for p_ in DG.pressure(DG.liveness(program.instructions, program.effects)[0]):
        if p_.file.startswith("mrf") and vd_bits:
            capacities[p_.file] = 1 << vd_bits
    report = DG.analyse_program(program, directions, issue=issue, stall_mnemonic=ops.stall,
                                hoist_role=facts.isa.by_mnemonic.get(ops.dma_load, {}).get("role"),
                                capacities=capacities, measured_cycles=measured_cycles)
    report["target"] = target
    report["tier"] = tier
    report["shape"] = {"m": m, "k": k, "n": n}
    report["tiles"] = list(plan.tiles)
    report["kernel_words"] = len(plan.words)
    report["settle"] = {"cycles": settle.tensor, "provenance": settle.provenance}
    report["direction_summary"] = directions.summary()
    # A cycle verdict has to say which hardware revision it is about. The measured number, the settle,
    # the issue model and the emitted encodings all come from the pinned atlas revisions; recording
    # the verification here is what stops the number being cited against the wrong device.
    pins = {}
    for name in LW.PINS.get(target, ()):  # noqa: B007 - a missing pin is recorded, not swallowed
        try:
            pins[name] = PV.verify(name)
        except Exception as exc:  # noqa: BLE001
            report.setdefault("provenance_warnings", []).append(
                f"{name}: {type(exc).__name__}: {exc}")
    report["provenance"] = PV.record(pins=pins)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("direction", "analyse"):
        s = sub.add_parser(name)
        s.add_argument("--target", required=True)
        s.add_argument("--tier", default="vsim")
        s.add_argument("--jobs", type=int, default=6)
        s.add_argument("--workdir", default=None)
        if name == "direction":
            s.add_argument("--force", action="store_true")
        else:
            s.add_argument("--shape", default="32x32x32")
            s.add_argument("--measured", default=None,
                           help="a measured cycle count for this shape, to check the bound against")
            s.add_argument("--product", action="store_true",
                           help="write the report as a versioned artifact product")
    args = ap.parse_args(argv)

    wd = Path(args.workdir) if args.workdir else A.cache_dir(f"perf_depgraph/{args.target}/work")
    wd.mkdir(parents=True, exist_ok=True)

    if args.cmd == "direction":
        model = derive_direction(args.target, args.tier, wd, jobs=args.jobs, force=args.force)
        print(json.dumps(model.summary(), indent=1))
        for line in model.unknown_operands()[:40]:
            print("  UNKNOWN", line)
        for mn, why in sorted(model.refused.items())[:40]:
            print("  REFUSED", mn, "--", why)
        print(f"\ncached at {direction_path(args.target, args.tier)}")
        return 0

    shape = tuple(int(v) for v in str(args.shape).lower().split("x"))
    if len(shape) != 3:
        raise SystemExit("--shape must be MxKxN")
    report = analyse(args.target, args.tier, shape, wd, jobs=args.jobs,
                     measured_cycles=(float(args.measured) if args.measured else None))
    print(json.dumps(report, indent=1, default=str))
    if args.product:
        out = A.new_product("dependence", target=args.target, version=0)
        (out / "dependence_report.json").write_text(json.dumps(report, indent=1, default=str))
        print(f"\nproduct: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
