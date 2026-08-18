#!/usr/bin/env python3
"""Approach (d): the hand-written C++ Gemmini MLIR dialect via IREE.

For each matmul kernel we (1) emit a tensor-domain linalg.matmul `.mlir`, (2) compile it through the
real IREE Gemmini plugin (`iree-compile --iree-plugin=gemmini --iree-gemmini-enable
--iree-gemmini-lower-back-to-iree=false`, mirroring models/gemmini_spike.yaml) which recovers
linalg.matmul -> gemmini.* -> custom-3 RoCC inside the dispatch executable, dumping the dispatch
object via `--iree-hal-dump-executable-files-to`, and (3) link that self-contained dispatch `.o`
(verified: zero undefined symbols, RoCC inline) with a tiny harness that builds the IREE
executable-library dispatch ABI (binding_ptrs=[X,W,Y], constants=0, workgroup grid), fills X/W with the
SAME deterministic data the capsule golden uses, times the dispatch with read_cycles, and prints the
shared OUT/METRIC cycles/DONE protocol. Then we run on spike(L2)/verilator(L3) via the same
gemmini.run_elf path and compare to the shared capsule golden.

This is the run-to-cycles path the merlin tree leaves as a downstream concern (tools/spike.py only
produces a .vmfb); we wire it here so approach (d) is measured on the SAME harness as the others.

Usage:
  run_iree_gemmini.py [--kernels id1,id2|all] [--sims auto] [--timeout 900] [--run-id iree_0001]
                      [--validate]   # gate: run G01 + check exact-int == golden, sane cycles
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

import _pbcommon as PB
from merlin.common import stimulus as STIM  # noqa: E402  (one source of truth for leaf data)
from merlin.targetgen import baremetalc_corroborate as BMC  # noqa: E402  (build flags + run reuse)
from merlin.targetgen import capsule_golden as CG  # noqa: E402

IREE = Path("/path/to/merlin-iree/build/host-merlin-release/tools/iree-compile")
# link-embedded=FALSE -> a standard system_elf relocatable .o (objdump-readable, objcopy-globalizable),
# so we can globalize the dispatch fn and link it into our own bare-metal harness. (The embedded-elf
# format is a self-contained IREE-loader module whose symbols are all local/unlinkable.)
GEMMINI_FLAGS = [
    "--iree-hal-target-device=local", "--iree-hal-local-target-device-backends=llvm-cpu",
    "--iree-llvmcpu-target-triple=riscv64-unknown-elf", "--iree-llvmcpu-target-abi=lp64d",
    "--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl256b",
    "--iree-opt-data-tiling=false", "--iree-dispatch-creation-data-tiling=false",
    "--iree-llvmcpu-enable-ukernels=none", "--iree-llvmcpu-link-embedded=false",
    # direct (non-indirect) buffer bindings so our hand-built binding_ptrs[] ABI is what the dispatch
    # reads (indirect bindings add a runtime indirection layer we don't replicate by hand).
    "--iree-hal-indirect-command-buffers=false",
    "--iree-plugin=gemmini", "--iree-gemmini-enable", "--iree-gemmini-lower-back-to-iree=false",
]
_OBJDUMP = "/path/to/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-objdump"
_OBJCOPY = "/path/to/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-objcopy"

# tensor-domain linalg matmul fixture: ins(X[M,K], W[K,N]) -> Y[M,N] (i8xi8->i32)
_FIXTURE = """\
func.func @{name}(%x: tensor<{M}x{K}xi8>, %w: tensor<{K}x{N}xi8>) -> tensor<{M}x{N}xi32> {{
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<{M}x{N}xi32>
  %z = linalg.fill ins(%c0 : i32) outs(%init : tensor<{M}x{N}xi32>) -> tensor<{M}x{N}xi32>
  %y = linalg.matmul ins(%x, %w : tensor<{M}x{K}xi8>, tensor<{K}x{N}xi8>)
                     outs(%z : tensor<{M}x{N}xi32>) -> tensor<{M}x{N}xi32>
  return %y : tensor<{M}x{N}xi32>
}}
"""

# Harness: build the IREE executable-library dispatch ABI + call the self-contained dispatch .o.
# Deterministic X/W match Tensor.deterministic("X"/"W"); the fill loops are EMITTED from
# merlin.common.stimulus so this harness and the Python golden cannot drift. Output Y is i32 [M,N].
# We time the whole workgroup grid with read_cycles.
_HARNESS_C = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#define MI {M}
#define MK {K}
#define MJ {N}
#define SEED_X {seed_x}
#define SEED_W {seed_w}

// --- minimal IREE executable-library v0 ABI (matches iree/hal/local/executable_library.h) ---
typedef struct {{
  uint32_t wsx, wsy; uint16_t wsz; uint16_t constant_count;
  uint32_t wcx, wcy; uint16_t wcz; uint8_t max_concurrency; uint8_t binding_count;
  const uint32_t* constants; void* const* binding_ptrs; const size_t* binding_lengths;
}} dispatch_state_t;
typedef struct {{
  uint32_t wid_x, wid_y; uint16_t wid_z; uint16_t reserved; uint32_t processor_id;
  void* local_memory; uint32_t local_memory_size;
}} workgroup_state_t;

extern int {sym}(const void* env, const dispatch_state_t* st, const workgroup_state_t* wg);

static int8_t X[MI*MK] row_align(1);
static int8_t W[MK*MJ] row_align(1);
static int32_t Y[MI*MJ] row_align(1);
static uint8_t scratch[64*1024] row_align(1);
static uint8_t env_buf[256];  // zeroed iree_hal_executable_environment_v0_t (no imports used)
{mix_fn}
int main() {{
{fill_x}
{fill_w}
  for (long k=0;k<(long)MI*MJ;k++) Y[k]=0;

  void* const binds[3] = {{ (void*)X, (void*)W, (void*)Y }};
  const size_t blen[3] = {{ sizeof(X), sizeof(W), sizeof(Y) }};
  dispatch_state_t st = {{0}};
  st.wsx=1; st.wsy=1; st.wsz=1; st.constant_count=0;
  st.wcx={wcx}; st.wcy={wcy}; st.wcz={wcz}; st.max_concurrency=1; st.binding_count=3;
  st.constants=0; st.binding_ptrs=binds; st.binding_lengths=blen;

  uint64_t c0 = read_cycles();
  for (uint32_t z=0; z<st.wcz; z++) for (uint32_t y=0; y<st.wcy; y++) for (uint32_t x=0; x<st.wcx; x++) {{
    workgroup_state_t wg = {{0}};
    wg.wid_x=x; wg.wid_y=y; wg.wid_z=z; wg.processor_id=0;
    wg.local_memory=scratch; wg.local_memory_size=sizeof(scratch);
    {sym}((const void*)env_buf, &st, &wg);
  }}
  gemmini_fence();
  uint64_t c1 = read_cycles();

  printf("OUT Y0 %d %d", MI, MJ);
  for (long k=0;k<(long)MI*MJ;k++) printf(" %d", (int)Y[k]);
  printf("\n");
  printf("METRIC cycles %llu\n", (unsigned long long)(c1-c0));
  printf("DONE\n");
  exit(0);
}}
"""


def iree_compile(name, M, K, N, workdir: Path) -> tuple[Path, str, tuple]:
    """Compile a linalg matmul fixture through the gemmini IREE plugin; return (dispatch.o, sym, wgc)."""
    workdir.mkdir(parents=True, exist_ok=True)
    fx = workdir / f"{name}.mlir"
    fx.write_text(_FIXTURE.format(name=name, M=M, K=K, N=N))
    dump = workdir / "dump"
    dump.mkdir(exist_ok=True)
    vmfb = workdir / f"{name}.vmfb"
    cmd = [str(IREE), str(fx), *GEMMINI_FLAGS,
           f"--iree-hal-dump-executable-files-to={dump}", "-o", str(vmfb)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError(f"iree-compile failed:\n{p.stderr[-1500:]}")
    objs = list(dump.glob("*_system_elf_riscv_64.o"))
    if not objs:
        raise RuntimeError(f"no system_elf dispatch .o dumped in {dump}")
    obj = objs[0]
    # The dispatch_v0 fn is a LOCAL symbol (.text...i8xi8xi32). Find it via objdump, then objcopy
    # --globalize-symbol so our harness can link against it.
    od = subprocess.run([_OBJDUMP, "-t", str(obj)], capture_output=True, text=True)
    m = re.search(r"\bF\b\s+\S*\s+([A-Za-z0-9_]*dispatch_0_\S*i8xi8xi32)\b", od.stdout)
    if not m:
        m = re.search(r"([A-Za-z0-9_]+dispatch_0[A-Za-z0-9_]*i8xi8xi32)", od.stdout)
    if not m:
        raise RuntimeError("could not find dispatch_v0 function symbol in .o")
    sym = m.group(1)
    gobj = dump / "dispatch_global.o"
    gp = subprocess.run([_OBJCOPY, f"--globalize-symbol={sym}", str(obj), str(gobj)],
                        capture_output=True, text=True)
    if gp.returncode != 0:
        raise RuntimeError(f"objcopy globalize failed:\n{gp.stderr[-600:]}")
    wgc = (1, 1, 1)  # workload [] -> static single workgroup; validated against the golden output
    return gobj, sym, wgc


def build_iree_elf(name, M, K, N, obj: Path, sym: str, wgc: tuple, workdir: Path) -> Path:
    src = workdir / f"{name}_harness.c"
    src.write_text(_HARNESS_C.format(M=M, K=K, N=N, seed_x=BMC.det_seed("X"), seed_w=BMC.det_seed("W"),
                                     mix_fn=STIM.C_MIX_FN,
                                     fill_x=STIM.c_fill_loop("X", "MI", "MK", "SEED_X", cast="int8_t"),
                                     fill_w=STIM.c_fill_loop("W", "MK", "MJ", "SEED_W", cast="int8_t"),
                                     sym=sym, wcx=wgc[0], wcy=wgc[1], wcz=wgc[2]))
    elf = workdir / f"{name}.elf"
    cmd = BMC._build_cmd(src, elf)
    # insert the self-contained dispatch object into the link line (right after the harness .c)
    cmd.insert(cmd.index(str(src)) + 1, str(obj))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if p.returncode != 0:
        raise RuntimeError(f"link failed:\n{p.stderr[-1500:]}")
    return elf


def _i32_matmul_ref(M: int, K: int, N: int) -> list:
    """The exact i32 X@W the IREE fixture computes (pure matmul, no epilogue), on the SAME
    deterministic X/W the harness fills. The IREE arm measures matmul compute; epilogue (acc_scale/
    relu) is not part of the IREE fixture and is a documented caveat for epilogue kernels."""
    from merlin.runtime.tensor import Tensor
    X = Tensor.deterministic("X", (M, K), "i8")
    W = Tensor.deterministic("W", (K, N), "i8")
    return X.matmul(W).to_list()


def run_one(k: dict, sims: list[str], workdir: Path, timeout: int) -> dict:
    name = re.sub(r"[^A-Za-z0-9_]", "_", k["id"])
    res = {"approach": "iree_dialect", "per_sim": {}}
    try:
        obj, sym, wgc = iree_compile(name, k["M"], k["K"], k["N"], workdir)
        elf = build_iree_elf(name, k["M"], k["K"], k["N"], obj, sym, wgc, workdir)
        res["ok_build"] = True
        res["sym"] = sym
    except Exception as e:
        res["error"] = str(e)[-400:]
        return res
    ref = _i32_matmul_ref(k["M"], k["K"], k["N"])  # i32 matmul reference (IREE fixture is epilogue-free)
    import time
    for sim in sims:
        t0 = time.time()
        try:
            r = BMC.run(elf, sim, timeout=timeout)
            got = r["outputs"].get("Y0")
            res["per_sim"][sim] = {"cycles": r["cycles"], "wall_s": round(time.time() - t0, 1),
                                   "correct": got == ref,
                                   "util_pct": PB.utilization_pct(k["macs"], r["cycles"]),
                                   "out_head": (got or [])[:4]}
        except Exception as e:
            res["per_sim"][sim] = {"error": str(e)[-200:], "wall_s": round(time.time() - t0, 1)}
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="all")
    ap.add_argument("--sims", default="auto")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--run-id", default="iree_0001")
    ap.add_argument("--validate", action="store_true",
                    help="gate: compile+run G01_multitile_sq_64x64x64, print output head + cycles")
    a = ap.parse_args(argv)

    if not IREE.is_file():
        print(f"iree-compile not found: {IREE}", file=sys.stderr)
        return 3
    doc = yaml.safe_load((PB.KERNELS / "kernel_corpus.yaml").read_text())
    corpus = (doc.get("golden_kernels", []) or []) + (doc.get("model_kernels", []) or [])
    if a.validate:
        corpus = [k for k in corpus if k["id"] == "G01_multitile_sq_64x64x64"]
    elif a.kernels != "all":
        want = set(a.kernels.split(","))
        corpus = [k for k in corpus if k["id"] in want]

    out = PB.RUNS / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for k in corpus:
        sims = (["spike", "verilator"] if k.get("sim_hint") == "L2+L3" else ["spike"]) \
            if a.sims == "auto" else [s.strip() for s in a.sims.split(",")]
        print(f"\n=== IREE {k['id']} ({k['M']}x{k['K']}x{k['N']}, sims={sims}) ===", flush=True)
        r = run_one(k, sims, out / "_work" / k["id"], a.timeout)
        results.append({"kernel": k["id"], "iree": r})
        print(f"  build={r.get('ok_build')} sym={r.get('sym','-')} "
              f"{ {s: (v.get('cycles'), v.get('util_pct'), v.get('out_head')) for s,v in r.get('per_sim',{}).items()} } "
              f"err={str(r.get('error'))[:120]}", flush=True)
    (out / "iree_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}/iree_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
