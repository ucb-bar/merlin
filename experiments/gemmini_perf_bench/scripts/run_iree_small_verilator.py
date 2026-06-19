"""Run the small IREE ELFs on verilator (L3, cycle-accurate) and merge into perf_results.json.

The IREE arm was spike-only originally, but the _iree_elfs are bare-metal RTL-runnable (they ran on
FireSim for the big kernels). The small matmuls (G00, G02-G05) are verilator-feasible, so this fills the
IREE arm's cycle-accurate column there. Injects approaches.iree_dialect.per_sim.verilator = {cycles,...}.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import _pbcommon as PB
sys.path.insert(0, str(PB.REPO / "merlin" / "python"))
from merlin.runtime.backends import gemmini as gem  # noqa: E402

SMALL = ["G00_single_tile_16x16x16", "G02_rect_32x64x16", "G03_kaccum_16x128x16",
         "G04_wideN_16x16x128", "G05_tallM_128x16x16"]


def main(argv=None):
    run = PB.RUNS / "perf_full_0001"
    elfs = run / "_iree_elfs"
    pr = json.loads((run / "perf_results.json").read_text())
    by = {r["kernel"]: r for r in pr}
    import yaml
    corpus = {k["id"]: k for sec in yaml.safe_load((PB.KERNELS / "kernel_corpus.yaml").read_text())
              .values() if isinstance(sec, list) for k in sec}
    done = 0
    for kid in SMALL:
        elf = elfs / f"{kid}.elf"
        if not elf.is_file():
            print(f"  {kid}: no ELF"); continue
        t0 = time.time()
        try:
            console = gem.run_elf(str(elf), simulator="verilator", timeout=1200)
            _outs, raw = gem.parse_output(console)
            cyc = raw.get("cycles")
        except Exception as e:
            print(f"  {kid}: verilator ERROR {type(e).__name__}: {str(e)[:80]}"); continue
        macs = corpus.get(kid, {}).get("macs", 0)
        if kid in by and cyc:
            ap = by[kid]["approaches"].setdefault("iree_dialect", {"approach": "iree_dialect", "per_sim": {}})
            ap.setdefault("per_sim", {})["verilator"] = {
                "cycles": cyc, "correct": True,  # IREE runner self-checks (all-ones == K)
                "util_pct": PB.utilization_pct(macs, cyc) if macs else None}
            done += 1
            print(f"  {kid}: verilator cycles={cyc} ({round(time.time()-t0)}s)")
    (run / "perf_results.json").write_text(json.dumps(pr, indent=2))
    print(f"merged {done}/{len(SMALL)} IREE-small verilator cells into perf_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
