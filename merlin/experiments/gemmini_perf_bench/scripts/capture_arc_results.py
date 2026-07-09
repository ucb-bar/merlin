"""Capture arc middle-tier results into arc_results.json for plotting: per-capsule bit-exact + cycles,
an A2 memory-latency sweep, and host-comm telemetry. Drives the existing replay harness per capsule."""
from __future__ import annotations
import json, re, subprocess, sys, os
from pathlib import Path
import _pbcommon as PB
from merlin.targetgen.rtl.facts import rtl_cache_dir, rtl_facts_path

REPO = PB.REPO
# PIN = committed distilled artifacts (gemmini_arc_replay.c, gemmini_arc_ports.h, arc_results.json);
# CACHE = purgeable arc scratch (gemmini.o input, r.json/replay_active.h/rbin outputs) — never in merlin/.
PIN = rtl_facts_path("gemmini").parent
CACHE = rtl_cache_dir("gemmini", ensure=True)
PYBIN = str(REPO / ".venv/bin/python")
GEN = "merlin.targetgen.rtl.gen_rocc_replay"
H = str(REPO / "merlin/python/merlin/targetgen/rtl/replay_json_to_h.py")
RUNS = REPO / "runs/capsule_bench_v1/runs/gemmini-capsule-bench"
CAPS = REPO / "merlin/contract/capsules"


def cap_yaml(name):
    for p in CAPS.rglob("capsule.yaml"):
        if p.parent.name == name:
            return p
    return None


def build_and_run(cap, trace, env=None):
    subprocess.run([PYBIN, "-m", GEN, str(cap), str(trace), "--out", str(CACHE / "r.json")],
                   cwd=REPO / "merlin/python", capture_output=True)
    subprocess.run([PYBIN, H, str(CACHE / "r.json"), str(CACHE / "replay_active.h")], capture_output=True)
    subprocess.run(["clang", "-O2", "-w", "-I", str(CACHE), "-I", str(PIN),
                    str(PIN / "gemmini_arc_replay.c"), str(CACHE / "gemmini.o"),
                    "-o", str(CACHE / "rbin")], capture_output=True)
    e = dict(os.environ, **(env or {}))
    out = subprocess.run([str(CACHE / "rbin")], capture_output=True, text=True, timeout=300, env=e).stdout
    return out


def parse(out):
    cyc = int(m.group(1)) if (m := re.search(r"cycles=(\d+)", out)) else None
    bit = "BIT-EXACT PASS" in out
    hc = {}
    if (m := re.search(r"rocc_cmds=(\d+).*busy_cyc=(\d+) \((\d+)%\)", out)):
        hc = {"cmds": int(m.group(1)), "busy_pct": int(m.group(3))}
    if (m := re.search(r"mvin: \d+ Get xacts, (\d+) B.*mvout: \d+ Put xacts, (\d+) B", out)):
        hc["mvin_B"], hc["mvout_B"] = int(m.group(1)), int(m.group(2))
    return cyc, bit, hc


def main():
    doc = __import__("yaml").safe_load((PB.KERNELS / "kernel_corpus.yaml").read_text())
    corpus = {k["id"]: k for sec in doc.values() if isinstance(sec, list) for k in sec}
    res = {"capsules": [], "latency_sweep": [], "hostcomm": {}}
    for d in sorted(RUNS.glob("*/")):
        name = d.name
        tr = d / "generated/instruction_trace.json"
        cy = cap_yaml(name)
        if not tr.is_file() or cy is None:
            continue
        out = build_and_run(cy, tr)
        cyc, bit, hc = parse(out)
        # macs: match corpus by capsule (bench names map to G*/K* differently; best-effort)
        res["capsules"].append({"capsule": name, "cycles": cyc, "bitexact": bit})
        if name in ("A2_single_tile_matmul", "C0_mlp_linear1"):
            res["hostcomm"][name] = hc
        print(f"  {name}: cycles={cyc} bitexact={bit}")
    # A2 latency sweep
    a2c = cap_yaml("A2_single_tile_matmul"); a2t = RUNS / "A2_single_tile_matmul/generated/instruction_trace.json"
    for L in [0, 4, 16, 32, 64, 128]:
        out = build_and_run(a2c, a2t, env={"ARC_MEM_LATENCY": str(L)})
        cyc, _, _ = parse(out)
        res["latency_sweep"].append({"latency": L, "cycles": cyc})
        print(f"  latency {L}: cycles={cyc}")
    n = len(res["capsules"]); ok = sum(1 for c in res["capsules"] if c["bitexact"])
    res["summary"] = {"n": n, "bitexact": ok}
    (PIN / "arc_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote arc_results.json: {ok}/{n} bit-exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
