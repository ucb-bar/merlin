#!/usr/bin/env python3
"""Cross-approach Gemmini performance runner.

Drives every kernel in the corpus through each Gemmini code-gen APPROACH on the SAME
ELF->spike(L2)/verilator(L3)->cycles harness, and records cycles + utilization + wall + correctness.
Approaches (extensible; Phase C adds the IREE C++ dialect arm):
  golden          - canonical Gemmini C library `tiled_matmul_auto` (hardware-loop WS), the perf ref;
  baseline        - generated MLIR OOT backend agent_spec_v0_mlir_oot;
  merlin_targetgen- generated MLIR OOT backend agent_spec_v1_mlir_oot;
  merlin_native   - the integrity-exempt Merlin reference lowering (merlin_native_v0).

Correctness = exact-int output == the kernel's shared capsule golden. Utilization is post-hoc
(macs/(cycles*256)); never gates. Per-cell timeout + honest failure recording so the giant LLM
kernels (which generated backends may unroll past feasibility) don't hang the sweep.

Usage:
  run_perf_bench.py [--kernels id1,id2|all] [--approaches golden,baseline,merlin_targetgen,merlin_native]
                    [--sims auto|spike|spike,verilator] [--timeout 900] [--run-id perf_0001]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

import _pbcommon as PB
from merlin.targetgen import baremetalc_corroborate as BMC  # noqa: E402  (golden C build/run reuse)
from merlin.targetgen import capsule_golden as CG  # noqa: E402
from merlin.targetgen import capsule_runner as CR  # noqa: E402

APPROACH_PKG = {
    "baseline": PB.REPO / "generated_targets" / "gemmini" / "agent_spec_v0_mlir_oot",
    "merlin_targetgen": PB.REPO / "generated_targets" / "gemmini" / "agent_spec_v1_mlir_oot",
    "merlin_native": PB.REPO / "generated_targets" / "gemmini" / "merlin_native_v0",
    # --- the 4 agentic capsule-bench backends (abc11/abc9), profiled for perf ---
    "agentic_raw_cpp":      PB.REPO / "merlin" / "experiments" / "gemmini_capsule_bench_v0" / "runs" / "raw_baseline" / "rb_abc11" / "submission",
    "agentic_scaffold_cpp": PB.REPO / "merlin" / "experiments" / "gemmini_capsule_bench_v0" / "runs" / "cpp_merlininfra" / "rbinfra_abc11" / "submission",
    "agentic_python":       PB.REPO / "merlin" / "experiments" / "gemmini_capsule_bench_v0" / "runs" / "merlin_assisted" / "merlin_abc9" / "submission",
    "agentic_circt":        PB.REPO / "merlin" / "experiments" / "gemmini_capsule_bench_v0" / "runs" / "merlin_assisted" / "merlincirct_abc9" / "submission",
}
CONTRACT = str(PB.REPO / "bench_contract")

# ---- golden approach (a): cycle-instrumented tiled_matmul_auto (hardware-loop WS) ----------------
_GOLDEN_C = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#define MI {M}
#define MK {K}
#define MJ {N}
#define SEED_A {seed_a}
#define SEED_B {seed_b}
static elem_t A[MI][MK] row_align(1);
static elem_t B[MK][MJ] row_align(1);
{cdecl}
int main() {{
  for (int i=0;i<MI;i++) for (int k=0;k<MK;k++) {{ long t=(long)SEED_A*(i*MK+k+1)+(long)(i*MK+k)*(i*MK+k); A[i][k]=(elem_t)(t%4); }}
  for (int k=0;k<MK;k++) for (int j=0;j<MJ;j++) {{ long t=(long)SEED_B*(k*MJ+j+1)+(long)(k*MJ+j)*(k*MJ+j); B[k][j]=(elem_t)(t%4); }}
  uint64_t c0 = read_cycles();
  tiled_matmul_auto(MI, MJ, MK, (elem_t*)A, (elem_t*)B, NULL, (void*)C,
      MK, MJ, MJ, MJ,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      {act}, {scale}, 0, false, false, false, {full_C}, false, 0, WS);
  gemmini_fence();
  uint64_t c1 = read_cycles();
  printf("OUT Y0 %d %d", MI, MJ);
  for (int i=0;i<MI;i++) for (int j=0;j<MJ;j++) printf(" %d", (int){celem});
  printf("\n");
  printf("METRIC cycles %llu\n", (unsigned long long)(c1-c0));
  printf("DONE\n");
  exit(0);
}}
"""


# Movement golden: mvin each DIM tile to scratchpad then mvout (identity round-trip), matching the
# capsule `movement` op (Y0 == X). Seeds match the capsule's source leaf name.
_GOLDEN_MOVE_C = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#define MI {M}
#define MJ {N}
#define SEED_X {seed_x}
static elem_t In[MI][MJ] row_align(1);
static elem_t Out[MI][MJ] row_align(1);
int main() {{
  for (long k=0;k<(long)MI*MJ;k++) {{ long t=(long)SEED_X*(k+1)+k*k; ((elem_t*)In)[k]=(elem_t)(t%4); }}
  gemmini_flush(0);
  gemmini_config_ld(MJ * sizeof(elem_t));
  gemmini_config_st(MJ * sizeof(elem_t));
  uint64_t c0 = read_cycles();
  for (int i=0;i<MI;i+=DIM) for (int j=0;j<MJ;j+=DIM) {{
    gemmini_mvin(&In[i][j], i*0 + (i/DIM)*0);  /* spad row reused per tile */
    gemmini_mvout(&Out[i][j], 0);
  }}
  gemmini_fence();
  uint64_t c1 = read_cycles();
  printf("OUT Y0 %d %d", MI, MJ);
  for (long k=0;k<(long)MI*MJ;k++) printf(" %d", (int)((elem_t*)Out)[k]);
  printf("\n");
  printf("METRIC cycles %llu\n", (unsigned long long)(c1-c0));
  printf("DONE\n");
  exit(0);
}}
"""


def _golden_src(M, K, N, epilogue, acc_scale, lhs_name, weight_name) -> str:
    """tiled_matmul_auto golden, seeds matching the capsule's actual lhs/weight leaf names (so
    attention QK/PV [Q,Kt / P,V] fill correctly, not just X/W)."""
    i8out = "acc_scale" in epilogue
    act = "RELU" if "relu" in epilogue else "NO_ACTIVATION"
    scale = f"{acc_scale}f" if (acc_scale and i8out) else "ACC_SCALE_IDENTITY"
    cdecl = "static elem_t C[MI][MJ] row_align(1);" if i8out else "static acc_t C[MI][MJ];"
    full_C = "false" if i8out else "true"
    return _GOLDEN_C.format(M=M, K=K, N=N, seed_a=BMC.det_seed(lhs_name), seed_b=BMC.det_seed(weight_name),
                            act=act, scale=scale, full_C=full_C, cdecl=cdecl, celem="C[i][j]")


def run_golden(k: dict, kdir: Path, sims: list[str], workdir: Path, timeout: int) -> dict:
    """Capsule-driven golden (bareMetalC C lib): dispatch by op. matmul/attention -> tiled_matmul_auto;
    movement -> mvin/mvout identity; conv2d -> deferred (tiled_conv_auto wiring; spike skips conv)."""
    cap = yaml.safe_load((kdir / "capsule.yaml").read_text())
    op = cap["operation"]["op"]
    attrs = cap["operation"].get("attributes", {})
    gold = CG.golden(cap).get("Y0")
    res = {"approach": "golden", "ok_build": False, "per_sim": {}}
    if op in ("matmul", "linear", "attention_qk", "attention_pv"):
        ins = {i["name"]: i["shape"] for i in cap["inputs"]}
        lhs, w = attrs.get("lhs"), attrs.get("weight")
        M, K = ins[lhs]
        N = ins[w][1]
        src = _golden_src(M, K, N, attrs.get("epilogue", []), attrs.get("acc_scale"), lhs, w)
    elif op == "movement":
        src_name = attrs.get("src", "X")
        M, N = next(i["shape"] for i in cap["inputs"] if i["name"] == src_name)
        src = _GOLDEN_MOVE_C.format(M=M, N=N, seed_x=BMC.det_seed(src_name))
    else:  # conv2d (+ any other): golden C-lib wiring pending; honest skip (MLIR arms still run it)
        res["error"] = f"golden({op}) C-lib template not wired (deferred; baseline/merlin/native run it)"
        return res
    try:
        elf = BMC.build(src, f"golden_{k['id']}", workdir)
        res["ok_build"] = True
    except Exception as e:
        res["error"] = str(e)[-300:]
        return res
    for sim in sims:
        t0 = time.time()
        try:
            r = BMC.run(elf, sim, timeout=timeout)
            got = r["outputs"].get("Y0")
            res["per_sim"][sim] = {"cycles": r["cycles"], "wall_s": round(time.time() - t0, 1),
                                   "correct": got == gold,
                                   "util_pct": PB.utilization_pct(k["macs"], r["cycles"])}
        except Exception as e:
            res["per_sim"][sim] = {"error": str(e)[-200:], "wall_s": round(time.time() - t0, 1)}
    return res


def run_mlir(approach: str, k: dict, kdir: Path, sims: list[str], runs_root: Path,
             timeout: int) -> dict:
    """Run a generated MLIR backend package through capsule_runner on this kernel's capsule."""
    pkg = APPROACH_PKG[approach]
    res = {"approach": approach, "ok_build": True, "per_sim": {}}
    cap = CR.load_capsule(kdir, contract=CONTRACT)
    # required tiers per feasibility: always L0/L1/trace; L2 spike always; L3 verilator only if feasible
    tiers = ["L0", "L1", "L2"] + (["L3"] if "verilator" in sims else [])
    cap = dict(cap); cap["required_oracle_tiers"] = tiers
    # run_capsule iterates every tier that HAS an adapter (required_oracle_tiers only marks the
    # integrity gate), so to skip verilator we must drop the L3 adapter — else it runs L3 regardless.
    adapters = CR.default_adapters()
    if "verilator" not in sims:
        adapters = {k: v for k, v in adapters.items() if k != "L3"}
    t0 = time.time()
    try:
        r = CR.run_capsule(cap, str(pkg), runs_root=str(runs_root), run_id=f"{approach}_{k['id']}",
                           contract=CONTRACT, oracle_adapters=adapters, timeout=timeout)
    except Exception as e:
        res["error"] = str(e)[-300:]
        res["wall_s"] = round(time.time() - t0, 1)
        return res
    res["status"] = r.get("status")
    res["numeric"] = (r.get("numeric") or {}).get("status") if isinstance(r.get("numeric"), dict) else r.get("numeric")
    res["wall_s_total"] = round(time.time() - t0, 1)
    rtiers = r.get("tiers", {})
    for sim, tier in (("spike", "L2"), ("verilator", "L3")):
        if sim not in sims:
            continue
        tr = rtiers.get(tier) or {}
        st = tr.get("status") if isinstance(tr, dict) else tr
        cyc = tr.get("cycles") if isinstance(tr, dict) else None
        res["per_sim"][sim] = {"cycles": cyc, "tier_status": st,
                               "correct": st == "pass",
                               "util_pct": PB.utilization_pct(k["macs"], cyc)}
    if r.get("failure"):
        res["failure"] = {kk: r["failure"].get(kk) for kk in ("plane", "category", "detail")}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="all")
    ap.add_argument("--approaches", default="golden,baseline,merlin_targetgen,merlin_native")
    ap.add_argument("--sims", default="auto", help="auto (per kernel sim_hint) | spike | spike,verilator")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--run-id", default="perf_0001")
    a = ap.parse_args(argv)

    corpus_doc = yaml.safe_load((PB.KERNELS / "kernel_corpus.yaml").read_text())
    corpus = [k for sec in ("golden_kernels", "model_kernels", "attention_kernels",
                            "conv_kernels", "movement_kernels")
              for k in (corpus_doc.get(sec) or [])]
    if a.kernels != "all":
        want = set(a.kernels.split(","))
        corpus = [k for k in corpus if k["id"] in want]
    approaches = [s.strip() for s in a.approaches.split(",") if s.strip()]

    # Runs live under the canonical runs/ root (PB.RUNS is re-rooted in _pbcommon.py); the
    # whole perf-bench pipeline (firesim/iree arms, assemble, report) shares PB.RUNS / run_id.
    out_dir = PB.RUNS / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)
    results = []
    for k in corpus:
        if a.sims == "auto":
            sims = ["spike", "verilator"] if k.get("sim_hint") == "L2+L3" else ["spike"]
        else:
            sims = [s.strip() for s in a.sims.split(",")]
        disp = k.get("shape") or (f"{k.get('M')}x{k.get('K')}x{k.get('N')}"
                                   if k.get("M") is not None else "?")
        print(f"\n=== kernel {k['id']}  ({disp} {k.get('output_dtype','')}, "
              f"sims={sims}, macs={k['macs']:,}) ===", flush=True)
        kdir = PB.KERNELS / k["id"]
        cell = {"kernel": k["id"], "shape": disp, "macs": k["macs"],
                "output_dtype": k.get("output_dtype", ""), "source": k["source"],
                "sim_hint": k.get("sim_hint"), "approaches": {}}
        for ap_name in approaches:
            t0 = time.time()
            # FAULT-TOLERANT: a crash in one approach/kernel records an error cell and continues
            # (the user's "if something crashes the run shouldn't stop"). Never aborts the batch.
            try:
                if ap_name == "golden":
                    r = run_golden(k, kdir, sims, work / k["id"], a.timeout)
                else:
                    r = run_mlir(ap_name, k, kdir, sims, out_dir / "_capsule_runs", a.timeout)
            except Exception as e:
                import traceback
                r = {"approach": ap_name, "ok_build": False, "status": "error",
                     "error": f"{type(e).__name__}: {str(e)[:300]}",
                     "traceback": traceback.format_exc()[-1200:], "per_sim": {}}
                print(f"  [{ap_name:16s}] CRASHED (recorded, continuing): {type(e).__name__}: {str(e)[:160]}", flush=True)
            cell["approaches"][ap_name] = r
            summ = {s: (v.get("cycles"), v.get("util_pct"), v.get("correct"))
                    for s, v in r.get("per_sim", {}).items()}
            print(f"  [{ap_name:16s}] {summ}  ({round(time.time()-t0,0)}s)", flush=True)
        results.append(cell)
        (out_dir / f"{k['id']}.json").write_text(json.dumps(cell, indent=2))
    (out_dir / "perf_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_dir}/perf_results.json  ({len(results)} kernels x {len(approaches)} approaches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
