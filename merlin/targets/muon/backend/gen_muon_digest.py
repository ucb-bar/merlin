"""Render muon_facts.json -> MUON_DIGEST.md (the Muon analog of gen_rtl_digest.py).

The "+CIRCT" token-saving advantage: the agent reads ONE RTL-grounded digest (geometry, register
budget, shared-memory capacity, FP peak, legal SIMT instruction classes + intrinsics) instead of
crawling Radiance Scala. Every line traces to an ``evidence`` string in muon_facts.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import muon_introspect as MI


def render(facts: dict) -> str:
    f = facts["facts"]
    simt, regs, smem, fp, isa = (f["simt"], f["registers"], f["shared_memory"],
                                 f["fp_datapath"], f["isa"])
    lines = [
        "# MUON_DIGEST — RadianceMuonConfig hardware facts (RTL-grounded)",
        "",
        f"*Generated from `muon_facts.json` by `{facts['generator']['name']}`. "
        "Method: " + facts["generator"]["method"] + ".*",
        "",
        "## SIMT geometry",
        f"- **{simt['cores']} cores** × **{simt['warps_per_core']} warps/core** × "
        f"**{simt['lanes_per_warp']} threads/warp (VLEN)** = {simt['threads_per_core']} threads/core.",
        f"  - _{simt['evidence']}_",
        "",
        "## Register budget (occupancy)",
        f"- ISA max **{regs['arch_max']}** regs/thread; compiler currently limits to "
        f"**{regs['compiler_limit']}**. Stay within this to avoid spills.",
        f"  - _{regs['evidence']}_",
        "",
        "## Shared memory",
        f"- **{smem['bytes_per_cluster'] // 1024} KiB / cluster.** Stage tiles here "
        f"(`store_shared`/`load32_shared`) and `mu_barrier` before reuse.",
        f"  - RTL bank evidence: `{smem['rtl_bank_evidence']}`",
        "",
        "## FP datapath & peak (the utilization denominator)",
        f"- fp32 FMA, **{fp['peak_flops_per_cycle']} flop/cycle** "
        f"({simt['cores']}×{simt['lanes_per_warp']}×2) @ {fp['clock_hz']/1e6:g} MHz = "
        f"**{fp['peak_gflops']:g} GFLOP/s** peak.",
        f"  - Report achieved GFLOP/s = flops/(cycles/{int(fp['clock_hz'])}) as a % of this.",
        "",
        "## Legal SIMT instruction classes (decode/trace target)",
        f"- {', '.join('`'+c+'`' for c in isa['instruction_classes'])}",
        f"- {isa['encoding_bits']}-bit encoding, up to {isa['max_src_operands']} src / "
        f"{isa['max_dst_operands']} dst, predicated execution.",
        "",
        "## Intrinsics surface (what your kernel emits)",
        f"- shared mem: {', '.join('`'+i+'`' for i in isa['intrinsics']['smem'])}",
        f"- sync: {', '.join('`'+i+'`' for i in isa['intrinsics']['sync'])}",
        f"- SIMT: {', '.join('`'+i+'`' for i in isa['intrinsics']['simt'])}",
        "",
        "## Sequencing rules the advisory checker enforces (derived from these facts)",
        f"- shared-memory footprint per cluster ≤ {smem['bytes_per_cluster'] // 1024} KiB",
        f"- live registers/thread ≤ {regs['compiler_limit']} (occupancy)",
        f"- threadblock ≤ {simt['threads_per_core']} threads; barrier-before-smem-reuse; "
        "split/join nesting balanced.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="render MUON_DIGEST.md from muon_facts.json")
    ap.add_argument("--facts", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    facts_path = Path(a.facts) if a.facts else MI.default_facts_path()
    if not facts_path.is_file():
        facts = MI.build_facts()
    else:
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
    out = Path(a.out) if a.out else facts_path.parent / "MUON_DIGEST.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(facts), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
