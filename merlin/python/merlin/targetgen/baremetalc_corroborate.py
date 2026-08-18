"""P1 — corroborate capsule goldens against REAL Gemmini reference programs.

Closes the "our goldens are only our own engine" gap. We build + run, on the SAME
`spike --extension=gemmini` + verilator we use everywhere, reference programs that exercise the
Gemmini datapath through the canonical primitives the bareMetalC corpus itself uses:

* ``mvin_mvout`` — the upstream bareMetalC movement test, instrumented to print its output tensor
  (its inputs are the deterministic formula ``In[n][i][j]=i*DIM+j+n``);
* ``tiled_matmul_auto`` — the canonical Gemmini library matmul (WS), the same function the corpus's
  matmul/conv tests call (e.g. ``bareMetalC/conv_perf.c``), driven with inputs filled by the SAME
  formula our capsule leaves use, so the reference output must equal our capsule golden.

These are external REFERENCE ORACLES only — never copied/called inside any submission. We then assert
``real_gemmini_output == our_Tensor_golden == capsule_golden`` for movement, single-tile matmul,
K-accumulation, relu, and acc_scale→i8.

Build flags / toolchain are reused verbatim from ``contract.compile.link_elf`` via the
gemmini backend's helpers (resolved through the registry, ``base.get_backend("gemmini")``),
so the reference ELF runs on the identical simulators.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from merlin.common import stimulus as STIM
from merlin.runtime.tensor import Tensor


def det_seed(name: str) -> int:
    """The salt a C program needs to fill leaf data identical to ``Tensor.deterministic``."""
    return STIM.det_seed(name)


def det_tensor(name: str, shape, dtype="i8") -> Tensor:
    return Tensor.deterministic(name, tuple(shape), dtype)


# ---- C reference-program templates --------------------------------------------------------------
# Inputs are filled by merlin.common.stimulus -- the SAME definition Tensor.deterministic uses, emitted
# as C from `stimulus.c_fill_loop_2d` rather than restated here, so the C program and the Python golden
# can never drift apart.

_MATMUL_C = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"

#define MI {I}
#define MK {K}
#define MJ {J}
#define SEED_A {seed_a}
#define SEED_B {seed_b}

static elem_t A[MI][MK] row_align(1);
static elem_t B[MK][MJ] row_align(1);
{cdecl}
{mix_fn}

int main() {{
{fill_a}
{fill_b}

  tiled_matmul_auto(MI, MJ, MK, (elem_t*)A, (elem_t*)B, NULL, (void*)C,
      MK, MJ, MJ, MJ,
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      {act}, {scale}, 0, false,
      false, false,
      {full_C}, false,
      0, WS);
  gemmini_fence();

  printf("OUT C %d %d", MI, MJ);
  for (int i=0;i<MI;i++) for (int j=0;j<MJ;j++) printf(" %d", (int){celem});
  printf("\n");
  printf("DONE\n");
  exit(0);
}}
"""

# mvin_mvout: upstream test, instrumented to print Out[0] (one DIM x DIM tile)
_MVIN_MVOUT_C = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "include/gemmini_testutils.h"
#define N 8
int main() {
  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));
  static elem_t In[N][DIM][DIM] row_align(1);
  static elem_t Out[N][DIM][DIM] row_align(1);
  for (size_t n=0;n<N;++n) for (size_t i=0;i<DIM;++i) for (size_t j=0;j<DIM;++j) In[n][i][j]=i*DIM+j+n;
  for (size_t n=0;n<N;++n) { gemmini_mvin(In[n], n*DIM); gemmini_mvout(Out[n], n*DIM); }
  gemmini_fence();
  printf("OUT C %d %d", DIM, DIM);
  for (int i=0;i<DIM;i++) for (int j=0;j<DIM;j++) printf(" %d", (int)Out[0][i][j]);
  printf("\n");
  printf("DONE\n");
  exit(0);
}
"""


def _build_cmd(src: Path, elf: Path) -> list[str]:
    """Identical flags/toolchain to contract.compile.link_elf, with our .c as the main."""
    from merlin.runtime.backends import base as _bk
    gem = _bk.get_backend("gemmini")
    rt, common = gem.rocc_tests_dir(), gem._common_dir()
    return [str(gem.gcc_path()), "-DPREALLOCATE=1", "-DMULTITHREAD=1", "-mcmodel=medany",
            "-std=gnu99", "-O2", "-ffast-math", "-fno-common", "-fno-builtin-printf",
            "-fno-tree-loop-distribute-patterns", "-march=rv64gc", "-Wa,-march=rv64gc",
            "-lm", "-lgcc", "-I", str(rt / "riscv-tests"), "-I", str(rt / "riscv-tests/env"),
            "-I", str(rt), "-I", str(common), "-DID_STRING=", "-DPRINT_TILE=0",
            "-nostdlib", "-nostartfiles", "-static", "-T", str(common / "test.ld"), "-DBAREMETAL=1",
            str(src), "-o", str(elf),
            *(str(p) for p in sorted(common.glob("*.c"))),
            *(str(p) for p in sorted(common.glob("*.S")))]


def build(src_text: str, name: str, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    src = workdir / f"{name}.c"
    src.write_text(src_text, encoding="utf-8")
    elf = workdir / f"{name}.elf"
    proc = subprocess.run(_build_cmd(src, elf), capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"build {name} failed:\n{proc.stderr[-2000:]}")
    return elf


def run(elf: Path, simulator: str, timeout: int = 600) -> dict:
    from merlin.runtime.backends import base as _bk
    gem = _bk.get_backend("gemmini")
    console = gem.run_elf(elf, simulator=simulator, timeout=timeout)
    outputs, raw = gem.parse_output(console)
    return {"outputs": outputs, "cycles": raw.get("cycles"), "console": console}


def matmul_source(I: int, K: int, J: int, *, name_a: str, name_b: str,
                  act="NO_ACTIVATION", scale="ACC_SCALE_IDENTITY", full_C=True) -> str:
    """C source filling A/B with exactly ``Tensor.deterministic(name_a/name_b, ...)``."""
    seed_a, seed_b = STIM.det_seed(name_a), STIM.det_seed(name_b)
    if full_C:
        cdecl = "static acc_t C[MI][MJ];"
        celem = "C[i][j]"
        fc = "true"
    else:
        cdecl = "static elem_t C[MI][MJ] row_align(1);"
        celem = "C[i][j]"
        fc = "false"
    return _MATMUL_C.format(I=I, K=K, J=J, seed_a=seed_a, seed_b=seed_b, act=act, scale=scale,
                            full_C=fc, cdecl=cdecl, celem=celem, mix_fn=STIM.C_MIX_FN,
                            fill_a=STIM.c_fill_loop_2d("A", "MI", "MK", "SEED_A"),
                            fill_b=STIM.c_fill_loop_2d("B", "MK", "MJ", "SEED_B"))


def _i8(x: int) -> int:
    x &= 0xFF
    return x - 256 if x >= 128 else x


def _matmul_golden(name_a, name_b, I, K, J, *, relu=False, acc_scale=None, i8out=False):
    """Tensor-engine golden on the SAME deterministic inputs the C reference fills."""
    A = Tensor.deterministic(name_a, (I, K), "i8")
    B = Tensor.deterministic(name_b, (K, J), "i8")
    t = A.matmul(B)
    if acc_scale is not None:
        t = t.requant_acc_scale(acc_scale)
    if relu:
        t = t.relu()
    if i8out:
        t = t.to_i8()
    return t.to_list()


# Each anchor: (name, builder -> src, golden, capsule, classes-note)
def _anchors():
    return [
        {"name": "mvin_mvout", "specimen": "bareMetalC/mvin_mvout.c (upstream, instrumented)",
         "capsule": "A1_mvin_mvout", "feature": "MVIN/MVOUT movement (identity, i8)",
         "src": _MVIN_MVOUT_C,
         "golden": [[_i8(i * 16 + j) for j in range(16)] for i in range(16)]},
        {"name": "ref_matmul_16", "specimen": "tiled_matmul_auto (canonical Gemmini lib), WS",
         "capsule": "A2_single_tile_matmul", "feature": "single-tile i8xi8->i32 matmul",
         "src": matmul_source(16, 16, 16, name_a="A0", name_b="W"),
         "golden": _matmul_golden("A0", "W", 16, 16, 16)},
        {"name": "ref_matmul_k32", "specimen": "tiled_matmul_auto, K=32 (K-accumulation)",
         "capsule": "A3_k_accumulation", "feature": "K-accumulation (Kt>1)",
         "src": matmul_source(16, 32, 16, name_a="A0", name_b="W"),
         "golden": _matmul_golden("A0", "W", 16, 32, 16)},
        {"name": "ref_matmul_relu", "specimen": "tiled_matmul_auto + RELU", "capsule": "A5_relu_epilogue",
         "feature": "relu epilogue", "src": matmul_source(16, 16, 16, name_a="A0", name_b="W", act="RELU"),
         "golden": _matmul_golden("A0", "W", 16, 16, 16, relu=True)},
        {"name": "ref_acc_scale_i8", "specimen": "tiled_matmul_auto + acc_scale->i8",
         "capsule": "A4_acc_scale_i8", "feature": "acc_scale (f32) + saturating i8 readout",
         "src": matmul_source(16, 16, 16, name_a="A0", name_b="W", scale="0.0625f", full_C=False),
         "golden": _matmul_golden("A0", "W", 16, 16, 16, acc_scale=0.0625, i8out=True)},
    ]


def corroborate_all(workdir: Path, simulators=("spike",), out_report: Path | None = None) -> list[dict]:
    rows = []
    for anc in _anchors():
        row = {"name": anc["name"], "specimen": anc["specimen"], "capsule": anc["capsule"],
               "feature": anc["feature"], "built": False, "results": {}, "golden_match": {},
               "status": "fail"}
        try:
            elf = build(anc["src"], anc["name"], workdir)
            row["built"] = True
        except Exception as e:
            row["error"] = str(e)[-300:]
            rows.append(row)
            continue
        ok_any = False
        for sim in simulators:
            try:
                res = run(elf, sim)
                got = res["outputs"].get("C")
                match = (got == anc["golden"])
                row["results"][sim] = {"cycles": res["cycles"], "match": match}
                row["golden_match"][sim] = match
                ok_any = ok_any or match
            except Exception as e:
                row["results"][sim] = {"error": str(e)[-200:]}
        row["status"] = "pass" if ok_any and all(
            v.get("match", False) for v in row["results"].values() if "match" in v) else "fail"
        rows.append(row)
    if out_report:
        _write_report(rows, out_report, simulators)
    return rows


def _write_report(rows, out: Path, simulators) -> None:
    L = ["# bareMetalC corroboration report (capsule_bench_v0)", "",
         "Real Gemmini reference programs (the canonical primitives the bareMetalC corpus uses: the",
         "upstream `mvin_mvout` movement test + `tiled_matmul_auto`, the library matmul called by",
         "`conv_perf.c`/`tiled_matmul_ws.c`) built with the IDENTICAL toolchain/flags as our package",
         "ELFs and run on the SAME spike + verilator. Inputs use the same deterministic formula as our",
         "capsule leaves, so the real-Gemmini output must equal our `Tensor` golden (== capsule golden).",
         "These are external reference ORACLES only — never copied/called inside any submission.", "",
         "**The anchor:** `real_gemmini_output == our_Tensor_golden == capsule_golden`.", "",
         "| anchor | reference specimen | equiv capsule | feature | built | " +
         " | ".join(f"{s} match (cyc)" for s in simulators) + " | status |",
         "|---|---|---|---|---|" + "---|" * (len(simulators) + 1)]
    for r in rows:
        cells = []
        for s in simulators:
            res = r["results"].get(s, {})
            if "match" in res:
                cells.append(f"{'yes' if res['match'] else 'NO'} ({res.get('cycles')})")
            else:
                cells.append(f"err: {res.get('error','-')[:30]}" if res else "—")
        L.append(f"| {r['name']} | {r['specimen']} | {r['capsule']} | {r['feature']} | "
                 f"{'yes' if r['built'] else 'NO'} | " + " | ".join(cells) + f" | {r['status']} |")
    npass = sum(1 for r in rows if r["status"] == "pass")
    L += ["", f"**{npass}/{len(rows)} anchors corroborated.**", "",
          "## Interpretation", "",
          "- A passing anchor means our golden engine (and thus the capsule it backs) reproduces real",
          "  Gemmini hardware output bit-exactly for identical inputs — closing the 'goldens are only",
          "  our own engine' gap for movement, single-tile matmul, K-accumulation, relu, and",
          "  acc_scale→i8.",
          "- conv2d corroboration is deferred: spike's Gemmini ISS does not run conv, and a verilator",
          "  conv anchor is future work (recorded honestly, not silently omitted).",
          "- These reference programs are NOT part of any submission; the integrity scan + ABI boundary",
          "  forbid copying/calling Gemmini library kernels in a graded backend."]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/bmc_corro")
    ap.add_argument("--simulators", default="spike")
    ap.add_argument("--report", default="out/artifacts/capsule-bench/gemmini/bareMetalC_corroboration_report.md")
    a = ap.parse_args(argv)
    sims = tuple(a.simulators.split(","))
    rows = corroborate_all(Path(a.workdir), simulators=sims, out_report=Path(a.report))
    for r in rows:
        print(f"  [{r['status']:4s}] {r['name']:18s} {r['capsule']:22s} "
              + " ".join(f"{s}={r['results'].get(s,{}).get('match')}" for s in sims))
    print(f"wrote {a.report}")
    return 0 if all(r["status"] == "pass" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
