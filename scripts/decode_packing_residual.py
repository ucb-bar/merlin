#!/usr/bin/env python
"""THROWAWAY decode harness — PART A of iteration-3 (packing/memory residual).

Adds the MEMORY-TRAFFIC facet (decode/memory.analyze_memory) on top of the existing
NR/LMUL/residency/vfmacc-form decode, to characterize for the openvla/rdt2 matmul shapes EXACTLY
what the experts (XNNPACK 1x4v, OpenBLAS 8x8) do for DATA MOVEMENT that ours does not:
  (a) pre-packed unit-stride panels vs strided model-layout streams,
  (b) loads / useful-FMA in the K-loop,
  (c) A-broadcast ladder (vslideup/vmv) per FMA (the .vv A-reload cost; 0 for a packed .vf kernel).

HOST + SPIKE-toolchain only, no board. Reuses the EXISTING decode infra + the model.o lowering of
scripts/decode_kernel_breakdown.py (imported, not duplicated).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.kernels.ceiling_drivers import run_expert_gemm as expert
from merlin.kernels.decode.objdump import tokenize
from merlin.kernels.decode.memory import analyze_memory
from merlin.kernels import cca

import decode_kernel_breakdown as dk   # reuse SHAPES, _lower_ours_to_obj, _decode_symbol, _analyze

REPO = Path(repo_root())

# the iteration-2 best whole-model kernel (vfmacc.vv broadcast ladder) and the iteration-3 candidate
# (vfmacc.vf), decoded at the openvla/rdt2 shapes. cube_64 as a sanity baseline.
OURS_FORKS = (
    ("ours_wholemodel", ["accumulator_resident_wholemodel"]),             # iter-1 (.vv, MR=1)
    ("ours_wholemodel_vf", ["accumulator_resident_wholemodel_vf"]),       # iter-2 (.vf, MR=1)
    ("ours_wholemodel_vf_mr4", ["accumulator_resident_wholemodel_vf_mr4"]),  # iter-3 (.vf, MR=4 A-reuse)
)


def _mem_row(stream):
    sp = cca._fma_loop(stream)
    mf = analyze_memory(stream, sp)
    base = dk._analyze(stream, op="matmul", source="x")
    keep = {k: base[k] for k in ("MR", "nr_lanes_vlen256", "accumulator_resident",
                                 "fma_loop_vfmacc_vf", "fma_loop_vfmacc_vv",
                                 "fma_loop_acc_spills")}
    return {**keep, "memory": (mf.to_dict() if mf else None)}


def main():
    rows = []

    # experts (shape-independent ukernel)
    for src, sym in (("xnnpack", "xnn_f32_gemm_ukernel_1x4v__rvv"),
                     ("openblas", "openblas_sgemm_kernel")):
        spec = expert._experts()[src]
        tmp = Path(tempfile.mkdtemp(prefix="memdec_exp_"))
        elf = tmp / f"{src}.riscv"
        err = expert._build(spec["driver"], spec["incs"], elf)
        if err:
            rows.append({"kernel": src, "shape": "ukernel", "blocker": err[:300]})
            continue
        stream, n = dk._decode_symbol(elf, sym)
        rows.append({"kernel": src, "shape": "ukernel(shape-indep)", "symbol": sym,
                     "packed": True, **_mem_row(stream)})

    # ours forks x shapes
    for run_id, feats in OURS_FORKS:
        for sname, (M, N, K) in dk.SHAPES.items():
            tmp = Path(tempfile.mkdtemp(prefix="memdec_ours_"))
            obj, blk = dk._lower_ours_to_obj(feats, M, N, K, tmp)
            if blk:
                rows.append({"kernel": run_id, "shape": sname, "MNK": (M, N, K), "blocker": blk})
                continue
            raws = tokenize(obj)
            secs = {r.section for r in raws}
            fsym = next((s for s in secs if "forward" in s), None)
            stream, n = dk._decode_symbol(obj, fsym)
            rows.append({"kernel": run_id, "shape": sname, "MNK": (M, N, K),
                         "symbol": fsym, "packed": False, **_mem_row(stream)})

    out = REPO / "output" / "kernels" / "ceiling" / "packing_residual_decode.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
