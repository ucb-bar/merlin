"""Build one gemmini C kernel, run it on the RTL engine, grade it bit-exactly. JSON on stdout.

A SEPARATE PROCESS ON PURPOSE. AutoComp must run under its own virtualenv (it imports `openai`,
`google-genai`, `boto3` at module scope), while the oracle needs merlin's. Installing AutoComp's
dependencies into merlin's shared venv would change an interpreter several other sessions are using,
so the two interpreters stay separate and this script is the seam between them.

That seam has bitten this repo before in the opposite direction: an evaluator that inherited
`sys.executable` from the caller ran under the wrong interpreter and reported a KNOWN-GOOD reference
kernel as wrong. So the caller resolves merlin's interpreter explicitly and never inherits it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in (here, *here.parents):
        if (cand / "merlin" / "python").is_dir():
            return cand
    raise SystemExit("could not locate repo root")


REPO = _repo_root()
sys.path.insert(0, str(REPO / "merlin" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _track as T                                                    # noqa: E402
from merlin.targetgen import baremetalc_corroborate as BC             # noqa: E402


#: THE AUTOCOMP GEMMINI DIALECT. Its ISA prompt (`agents/gemmini/prompts/isa_prompt_conv.py`)
#: documents the accelerator as bare `mvin`/`mvout`/`config_ld`/`preload`/`compute_*`/`fence` — NOT
#: the `gemmini_*` names our header defines — and its shipped harness files define those aliases
#: locally (e.g. `harnesses/f/test0.c`: `#define fence() gemmini_fence()`). Without them the agent's
#: code is correct AutoComp dialect that simply does not link, which is what produced 16 of 18
#: "compile failures" and would have been reported as the framework's error rate.
#:
#: Every mapping below was verified against the curated `include/gemmini.h`, not guessed:
#:   config_ex(dataflow, act, A_stride, A_tr, B_tr) -> gemmini_extended_config_ex(..., sys_shift=0, ...)
#:   config_ld(stride, scale, spad_block_stride, id) -> gemmini_extended4_config_ld(stride, scale,
#:       shrunk=0, block_mvin_stride=spad_block_stride, id)   [3rd arg is block stride, NOT `shrunk`]
#:   preload / compute_preloaded / compute_accumulated / mvin{,2,3} / mvout -> the `gemmini_extended*`
#:       forms with identical argument order.
#:
#: ⚠️ ONE MAPPING IS NOT DERIVABLE AND IS NOT GUESSED. AutoComp documents `config_st(cols)` as a COLUMN
#: COUNT; our `gemmini_extended_config_st(stride, acc_act, acc_scale)` takes a BYTE STRIDE. Converting
#: needs the output element width, and silently picking one would corrupt every store — the same class
#: of defect as the A7 stride advisory that cost six capsules. So the alias multiplies by
#: `sizeof(acc_t)`, which is the width THIS workload's output actually has (`full_C` acc_t output), and
#: that assumption is stated on the run rather than buried here. A workload with an elem_t output would
#: need a different alias, and this shim must be revisited rather than reused.
_DIALECT = """
#ifndef fence
#define fence() gemmini_fence()
#endif
#define config_ex(dataflow, act, A_stride, A_transpose, B_transpose) \\
    gemmini_extended_config_ex(dataflow, act, 0, A_stride, A_transpose, B_transpose)
#define config_ld(dram_stride, scale_factor, spad_block_stride, id) \\
    gemmini_extended4_config_ld(dram_stride, scale_factor, 0, spad_block_stride, id)
#define config_st(cols) gemmini_extended_config_st((cols) * sizeof(acc_t), 0, MVIN_SCALE_IDENTITY)
#define mvin(dram_addr, spad_acc_addr, cols, rows) \\
    gemmini_extended_mvin(dram_addr, spad_acc_addr, cols, rows)
#define mvin2(dram_addr, spad_acc_addr, cols, rows) \\
    gemmini_extended_mvin2(dram_addr, spad_acc_addr, cols, rows)
#define mvin3(dram_addr, spad_acc_addr, cols, rows) \\
    gemmini_extended_mvin3(dram_addr, spad_acc_addr, cols, rows)
#define mvout(dram_addr, spad_acc_addr, cols, rows) \\
    gemmini_extended_mvout(dram_addr, spad_acc_addr, cols, rows)
#define preload(B_spad_addr, C_acc_addr, B_cols, B_rows, C_cols, C_rows) \\
    gemmini_extended_preload(B_spad_addr, C_acc_addr, B_cols, B_rows, C_cols, C_rows)
#define compute_preloaded(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows) \\
    gemmini_extended_compute_preloaded(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows)
#define compute_accumulated(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows) \\
    gemmini_extended_compute_accumulated(A_spad_addr, bias_spad_addr, A_cols, A_rows, bias_cols, bias_rows)
"""

#: What AutoComp's own harness supplies around an agent's kernel. Nothing else is added: the wrapper
#: must not change the computation, only make a FRAGMENT compilable.
_PROLOGUE = """#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "include/gemmini_testutils.h"
""" + _DIALECT


def _as_program(text: str) -> str:
    """Make a candidate compilable without changing what it computes.

    ⚠️ MEASURED, and it invalidated a whole arm before it was found: AutoComp's gemmini agent returns
    a kernel FUNCTION (`void solution() { ... }`) with no `#include`s, because the framework splices it
    into a harness that provides them. Compiling that as a standalone program failed 16 of 18
    candidates, and reading those failures as AutoComp's compile-failure rate would have blamed the
    framework for this harness's mistake -- the second time that trap fired in this integration
    (see the memory `harness-blames-the-agent`).

    So: if the candidate already defines `main`, it is a complete program and is left EXACTLY as is.
    Otherwise the includes are prepended and a `main` that calls the kernel is appended -- the same
    two things AutoComp's own harness contributes, and nothing more.
    """
    if "int main" in text or "void main" in text:
        return text
    entry = None
    for cand in ("solution", "kernel", "run_kernel", "compute"):
        if f"void {cand}(" in text or f"int {cand}(" in text:
            entry = cand
            break
    if entry is None:
        # No `main` and no recognised entry point: DO NOT guess. Returning it unchanged makes the
        # compiler report the real problem, which is better than inventing a wrapper that hides it.
        return text
    prologue = "" if "#include" in text.split("\n", 1)[0] or "#include" in text[:200] else _PROLOGUE
    return f"{prologue}{text}\n\nint main() {{\n  {entry}();\n  return 0;\n}}\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emit-seed", action="store_true",
                    help="print the cycle-instrumented seed kernel and exit (also crosses the "
                         "interpreter seam: the template lives in merlin's venv)")
    ap.add_argument("--source", help="path to the candidate .c")
    ap.add_argument("--name")
    ap.add_argument("--workdir")
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--name-a", default="A0")
    ap.add_argument("--name-b", default="W")
    ap.add_argument("--simulator", default="gsim")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args(argv)

    if a.emit_seed:
        # The seed AutoComp starts from, cycle-instrumented in the same form gemmini.py uses:
        # read_cycles() either side of the kernel and the METRIC line printed BEFORE the output dump,
        # so a flooded UART cannot lose it. The anchors are checked so a template change fails loudly
        # instead of silently producing an uninstrumented kernel that then scores as unmeasurable.
        src = BC.matmul_source(a.M, a.K, a.N, name_a=a.name_a, name_b=a.name_b)
        anchors = ("  tiled_matmul_auto(", "  gemmini_fence();\n", '  printf("OUT C %d %d"')
        for anc in anchors:
            if anc not in src:
                raise SystemExit(f"seed template changed shape: {anc!r} not found; refusing to "
                                 f"instrument blindly")
        # `unsigned long`, NOT `uint64_t`. MEASURED failure: 8 of 10 AutoComp candidates failed to
        # compile with `unknown type name 'uint64_t'` on this injected line, because the model
        # restructures the file and drops `#include <stdint.h>`. That is a defect in the
        # INSTRUMENTATION, not in the candidate -- and reporting it as an 80% compile-failure rate
        # would have blamed the agent for the harness. `unsigned long` is 64-bit on rv64 and needs no
        # header, so the timing bracket survives any include the agent removes.
        src = src.replace(anchors[0], "  unsigned long _c0 = read_cycles();\n" + anchors[0], 1)
        src = src.replace(anchors[1], anchors[1] + "  unsigned long _c1 = read_cycles();\n"
                          '  printf("METRIC cycles %lu\\n", _c1 - _c0);\n', 1)
        sys.stdout.write(src)
        return 0
    if not (a.source and a.name and a.workdir):
        raise SystemExit("--source, --name and --workdir are required unless --emit-seed")

    import os
    if a.simulator == "gsim":
        os.environ.update(T.gsim_env())

    out: dict = {"name": a.name, "engine": a.simulator, "engine_config": T.GSIM_CONFIG}
    src = _as_program(Path(a.source).read_text(encoding="utf-8"))
    try:
        elf = BC.build(src, a.name, Path(a.workdir))
        out["compiled"] = True
    except Exception as exc:
        print(json.dumps({**out, "compiled": False, "correct": False, "cycles": None,
                          "detail": f"Compile error: {exc}"[-3000:]}))
        return 0
    try:
        res = BC.run(elf, a.simulator, timeout=a.timeout)
    except Exception as exc:
        print(json.dumps({**out, "compiled": True, "correct": False, "cycles": None,
                          "detail": f"Run error: {exc}"[-3000:]}))
        return 0

    golden = BC._matmul_golden(a.name_a, a.name_b, a.M, a.K, a.N)
    got = res.get("outputs")
    if isinstance(got, dict):
        got = next(iter(got.values()), None)
    flat_g = [int(v) for row in golden for v in (row if isinstance(row, list) else [row])]
    if got is None:
        detail, ok = "no outputs were parsed from the console", False
    else:
        flat_c = ([int(v) for row in got for v in (row if isinstance(row, list) else [row])]
                  if got and isinstance(got[0], list) else [int(v) for v in got])
        if len(flat_c) != len(flat_g):
            ok = False
            detail = (f"output has {len(flat_c)} elements, expected {len(flat_g)} "
                      f"({a.M}x{a.N})")
        else:
            bad = next((i for i, (x, y) in enumerate(zip(flat_c, flat_g)) if x != y), None)
            ok = bad is None
            detail = "" if ok else (f"first mismatch at flat index {bad} (row {bad // a.N}, "
                                    f"col {bad % a.N}): got {flat_c[bad]}, expected {flat_g[bad]}")
    cycles = res.get("cycles")
    # A kernel that runs but reports no cycle count is NOT a zero-cycle kernel.
    if ok and not isinstance(cycles, int):
        ok = False
        detail = ("the run produced no 'METRIC cycles' line, so it has no measured cost — the "
                  "read_cycles() bracket and its printf must be preserved")
    print(json.dumps({**out, "compiled": True, "correct": ok,
                      "cycles": cycles if isinstance(cycles, int) else None, "detail": detail}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
