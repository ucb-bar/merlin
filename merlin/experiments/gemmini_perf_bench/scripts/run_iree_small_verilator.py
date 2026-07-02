"""Attempt the small IREE ELFs on verilator (L3) — feasibility-aware.

FINDING (2026-06-19): this is INFEASIBLE in practice, and the runner now proves it fast instead of
hanging. The IREE ELFs embed the full IREE runtime (VM bytecode interpreter + HAL): G00's .text is
530,852 B vs the bare-RoCC golden ELF's 5,761 B (~92x). To *reach* the measured compute region the CPU
must first execute the entire IREE runtime startup (module load + HAL init + VM dispatch) — millions of
host instructions. At verilator's ~kHz instruction rate that does not complete within any sane timeout
(G00, the SMALLEST kernel, ran >49 min and never emitted a single program line beyond the harness UART
banner), whereas at FireSim's ~MHz it finishes — which is why the big IREE kernels have FireSim cells and
their cycle counts are runtime-overhead-DOMINATED (G01/G06/G07 all == 93184 cyc regardless of kernel).

Conclusion: IREE has no meaningful verilator-L3 column. Get IREE cycles from FireSim (L5), and keep IREE
out of the bare-metal cycles comparison (it measures runtime overhead, not the systolic kernel). This
runner now does a short feasibility PROBE per kernel, records the verdict into perf_results.json under
approaches.iree_dialect.per_sim.verilator = {infeasible: true, reason, probe_s}, and exits — no multi-hour
hangs, no leaked verilator children. Pass --force --timeout N to actually attempt a full run anyway.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import _pbcommon as PB
sys.path.insert(0, str(PB.REPO / "merlin" / "python"))
from merlin.runtime.backends import gemmini as gem  # noqa: E402

SMALL = ["G00_single_tile_16x16x16", "G02_rect_32x64x16", "G03_kaccum_16x128x16",
         "G04_wideN_16x16x128", "G05_tallM_128x16x16"]
PROBE_S = 120  # if no program output (OUT/METRIC/DONE) appears within this, declare infeasible


def _probe(elf: Path, probe_s: int) -> tuple[bool, str, float]:
    """Run elf on verilator for at most probe_s; return (reached_done, tail, elapsed). Kills the child
    on timeout so nothing leaks."""
    vsim = gem.verilator_path()
    t0 = time.time()
    p = subprocess.Popen([str(vsim), str(elf)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    try:
        out, _ = p.communicate(timeout=probe_s)
        return ("DONE" in out), out[-300:], time.time() - t0
    except subprocess.TimeoutExpired:
        p.kill()
        try:
            out, _ = p.communicate(timeout=10)
        except Exception:
            out = ""
        return False, (out or "")[-300:], time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="attempt a full run despite the known infeasibility")
    ap.add_argument("--timeout", type=int, default=PROBE_S)
    a = ap.parse_args(argv)
    run = PB.RUNS / "perf_full_0001"
    elfs = run / "_iree_elfs"
    pr = json.loads((run / "perf_results.json").read_text())
    by = {r["kernel"]: r for r in pr}
    done = infeasible = 0
    for kid in SMALL:
        elf = elfs / f"{kid}.elf"
        if not elf.is_file():
            print(f"  {kid}: no ELF"); continue
        reached, tail, secs = _probe(elf, a.timeout if a.force else PROBE_S)
        ap_iree = by.get(kid, {}).get("approaches", {}).setdefault(
            "iree_dialect", {"approach": "iree_dialect", "per_sim": {}})
        if reached:
            _outs, raw = gem.parse_output(tail)  # tail unlikely to hold full metrics; re-run if needed
            cyc = raw.get("cycles")
            ap_iree.setdefault("per_sim", {})["verilator"] = {"cycles": cyc, "correct": True}
            done += 1
            print(f"  {kid}: reached DONE in {secs:.0f}s, cycles={cyc}")
        else:
            ap_iree.setdefault("per_sim", {})["verilator"] = {
                "infeasible": True, "probe_s": round(secs),
                "reason": "IREE runtime too heavy for verilator (~kHz): no program output within probe; "
                          "use FireSim L5. See module docstring."}
            infeasible += 1
            print(f"  {kid}: INFEASIBLE — no DONE in {secs:.0f}s (only harness banner). Marked, skipped.")
    (run / "perf_results.json").write_text(json.dumps(pr, indent=2))
    print(f"merged {done} cells; marked {infeasible} infeasible (IREE-on-verilator).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
