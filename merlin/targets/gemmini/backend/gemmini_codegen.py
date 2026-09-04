"""Command buffer -> bare-metal C driver using low-level Gemmini (libgemmini) intrinsics.

Scope (C0/C1/C4/C5): one RES_PACK of a resident weight W (k x n), then N>=1
MATMUL_RESIDENT (each an m x k activation reusing W -> m x n i32 result) each followed by a
COMMIT (empty epilogue or [relu], i32 output), then EVICT. The explicit weight-stationary
sequence is emitted directly — NOT tiled_matmul_auto — so the command-buffer -> ISA mapping is
transparent and certifiable. Arbitrary m/k/n are supported by tiling into DIM(16) blocks with
K-accumulation and zero-padding the edges (C4). Leaf tensors are embedded exactly as the Merlin
engine materializes them (Tensor.deterministic), so the same bytes feed the reference, the
simulator, and this kernel. Outputs + cycles print as OUT/METRIC/DONE.

Out of scope (-> CodegenError): epilogues beyond relu (C2/C3 requant), transpose (C6),
multiple packs.
"""
from __future__ import annotations

from typing import Any

from merlin.runtime.commandbuffer import materialize_inputs

DIM = 16
# relu is bit-exact (max(0,x)). requant is NOT: Gemmini's acc_scale is ignored on the full-i32
# (full_C) readout, and its i8-downscale path uses float round-to-nearest, whereas merlin's
# requant is an integer round-half-up shift on i32. See docs/gemmini_requant_reconciliation.md.
SUPPORTED_EPILOGUE = {"relu"}

# How many completion barriers this emitter inserts. `RETIRE_ONCE` is the production setting: one
# retire for the whole kernel. `RETIRE_PER_JOB` reproduces what this emitter did before the hoist --
# a barrier at the end of every job -- and exists ONLY so the barrier lever can be measured against
# the arm it replaced. Both arms are bit-exact; they differ in cycles, which is the whole point.
RETIRE_ONCE = "once"
RETIRE_PER_JOB = "per_job"
RETIRE_SETTINGS = frozenset({RETIRE_ONCE, RETIRE_PER_JOB})


class CodegenError(RuntimeError):
    pass


def _ceil_dim(x: int) -> int:
    return ((x + DIM - 1) // DIM) * DIM


def _pad_rowmajor(data: list[int], rows: int, cols: int, prows: int, pcols: int) -> list[int]:
    """Zero-pad a row-major rows x cols matrix into prows x pcols."""
    out = [0] * (prows * pcols)
    for r in range(rows):
        base, pbase = r * cols, r * pcols
        out[pbase:pbase + cols] = data[base:base + cols]
    return out


def _c_array(name: str, ctype: str, data: list[int]) -> str:
    vals = ",".join(str(int(v)) for v in data)
    return f"static const {ctype} {name}[{len(data)}] row_align(1) = {{{vals}}};"


def _parse(cb: dict[str, Any]):
    """Validate the command stream; return (weight, [(lhs, out, epilogue, m)], k, n)."""
    cmds = cb.get("commands", [])
    packs = [c for c in cmds if c["opcode"] == "RES_PACK"]
    matmuls = [c for c in cmds if c["opcode"] in ("MATMUL_RESIDENT", "MATMUL")]
    commits = [c for c in cmds if c["opcode"] == "COMMIT"]
    evicts = [c for c in cmds if c["opcode"] == "EVICT"]
    if len(packs) != 1:
        raise CodegenError(f"expected exactly one RES_PACK, got {len(packs)}")
    if not matmuls or len(matmuls) != len(commits):
        raise CodegenError(f"need matmuls==commits>=1, got {len(matmuls)}/{len(commits)}")
    if len(evicts) != 1:
        raise CodegenError(f"expected exactly one EVICT, got {len(evicts)}")

    weight, res = packs[0]["operands"]["src"], packs[0]["operands"]["dst"]
    tensors = cb.get("tensors", {})
    k, n = tensors[weight]["shape"]
    acc_to_commit = {c["operands"]["src"]: c for c in commits}
    jobs = []
    for mm in matmuls:
        ops = mm["operands"]
        if ops["rhs"] != res:
            raise CodegenError("every matmul must reuse the single resident weight")
        lhs = ops["lhs"]
        m, k2 = tensors[lhs]["shape"]
        if k2 != k:
            raise CodegenError(f"matmul lhs k={k2} != weight k={k}")
        commit = acc_to_commit.get(ops["dst"])
        if commit is None:
            raise CodegenError(f"matmul dst {ops['dst']} has no commit")
        attrs = commit.get("attributes", {})
        if attrs.get("output_dtype", "i8") != "i32":
            raise CodegenError("conformance scope is i32 output")
        epi = list(attrs.get("epilogue", []))
        if any(s not in SUPPORTED_EPILOGUE for s in epi):
            raise CodegenError(f"unsupported epilogue {epi} (supported: {SUPPORTED_EPILOGUE})")
        shift = int(attrs.get("requant_shift", cb.get("params", {}).get("requant_shift", 4)))
        jobs.append((lhs, commit["operands"]["dst"], epi, m, shift))
    return weight, jobs, k, n


def _counter_bracket() -> tuple[list[str], list[str], list[str]]:
    """``(prologue, epilogue, include)`` C lines for this target's combination counters — or empty.

    The sibling MLIR emitter has carried this bracket for a while; the command-buffer driver did not,
    so an A/B run through :func:`generate_driver` produced cycles with no activity decomposition beside
    them and every occupancy consumer refused for want of a source. Same opt-in switch and same
    derivation as the sibling: the counter set and its event codes come from the target's own shipped
    header, and a missing code or an over-full slot set refuses the whole bracket rather than emitting a
    partial one, because a missing combination silently turns realised overlap into a lower bound.

    Off by default. A change that altered every run would make a round's verdicts incomparable with the
    rounds before it, so measuring occupancy is something a caller asks for.
    """
    from .gemmini_codegen_mlir import _COUNTER_SLOTS, _counters_requested
    if not _counters_requested():
        return [], [], []
    try:
        from pathlib import Path as _P

        from merlin.perf import hw_counters as _hc
        found = _hc.counters_for_target("gemmini")
        if found.get("status") != "derived":
            return [], [], []
        text = _P(found["header"]).read_text(encoding="utf-8")
        bracket = _hc.counter_bracket_c(_hc.derive_occupancy_counters(text),
                                        _hc.event_codes(text), slots=_COUNTER_SLOTS)
    except Exception as exc:                      # noqa: BLE001 — never break a graded driver
        return [f"  /* merlin: counters unavailable: {type(exc).__name__} */"], [], []
    return (bracket["prologue"].rstrip("\n").splitlines(),
            bracket["epilogue"].rstrip("\n").splitlines(),
            ['#include "include/gemmini_counter.h"'])


def generate_driver(cb: dict[str, Any], *, mode: str = "explicit",
                    retire: str = RETIRE_ONCE) -> str:
    """Return the C source of the bare-metal Gemmini driver for ``cb``.

    ``retire`` selects how many completion barriers this emitter inserts. The default is the
    production setting and the only one a graded run uses; ``RETIRE_PER_JOB`` exists so the barrier
    lever has a SECOND arm to be measured against (see the call site below and
    :mod:`merlin.perf.barrier_arms`). A lever nobody can emit the other side of is not measurable,
    and the two arms differ by nothing but the inserted barrier statement.
    """
    if mode != "explicit":
        raise CodegenError("only mode='explicit' is supported (no tiled_matmul_auto)")
    if retire not in RETIRE_SETTINGS:
        raise CodegenError(f"unknown retire={retire!r}; supported: {sorted(RETIRE_SETTINGS)}")
    weight, jobs, k, n = _parse(cb)
    leaves = materialize_inputs(cb)
    kp, np_ = _ceil_dim(k), _ceil_dim(n)
    Kt, Nt = kp // DIM, np_ // DIM
    # Derived ONCE: two calls could read a header that changed between them and emit a prologue
    # configuring one slot set with an epilogue reading another.
    _cpro, _cepi, _cinc = _counter_bracket()

    # Embed padded leaf tensors (zero-pad edges -> exact tiled matmul; crop output on print).
    lines = [
        "/* Generated by merlin.runtime.backends.gemmini_codegen — do not edit. */",
        "#include <stdint.h>",
        "#include <stdio.h>",
        '#include "include/gemmini_testutils.h"',
        *_cinc,
        "",
        _c_array(f"T_{weight}", "elem_t", _pad_rowmajor(list(leaves[weight].data), k, n, kp, np_)),
    ]
    job_meta = []
    for lhs, out, epi, m, shift in jobs:
        mp = _ceil_dim(m)
        lines.append(_c_array(f"T_{lhs}", "elem_t",
                              _pad_rowmajor(list(leaves[lhs].data), m, k, mp, kp)))
        lines.append(f"static acc_t T_{out}[{mp * np_}] row_align_acc(1);")
        job_meta.append((lhs, out, epi, m, mp, shift))

    # Scratchpad layout: W tiles resident at rows [0, Kt*Nt*DIM); A tile slot after them.
    a_slot = Kt * Nt * DIM
    lines += [
        "",
        "int main() {",
        "  uint64_t c0, c1;",
        "  /* full-i32 accumulator tile: overwrite address, and the accumulate variant. */",
        "  uint32_t acc_ovw = ((3u<<(ADDR_LEN-2))|(1u<<(ADDR_LEN-3))) & ~(1u<<(ADDR_LEN-2));",
        "  uint32_t acc_acc = acc_ovw | (1u<<(ADDR_LEN-2));",
        "",
        "  gemmini_flush(0);",
        "  gemmini_config_ex(WEIGHT_STATIONARY, NO_ACTIVATION, 0);",
        f"  /* resident weight W ({k}x{n} padded to {kp}x{np_}): mvin all {Kt*Nt} tiles once. */",
        f"  gemmini_config_ld({np_} * sizeof(elem_t));",
    ]
    for kt in range(Kt):
        for nj in range(Nt):
            w_row = (kt * Nt + nj) * DIM
            off = (kt * DIM) * np_ + nj * DIM
            lines.append(f"  gemmini_mvin((void*)&T_{weight}[{off}], {w_row});")
    lines.append("")
    lines += _cpro
    lines.append("  c0 = read_cycles();")

    for job_index, (lhs, out, epi, m, mp, shift) in enumerate(job_meta):
        Mt = mp // DIM
        acc_act = "RELU" if "relu" in epi else "NO_ACTIVATION"
        acc_scale = f"(1.0 / {1 << shift})" if "requant" in epi else "ACC_SCALE_IDENTITY"
        lines.append(f"  /* {lhs} @ {weight} -> {out}  ({m}x{k}x{n}, tiles {Mt}x{Kt}x{Nt}, epi={epi}) */")
        lines.append(f"  gemmini_extended_config_st({np_} * sizeof(acc_t), {acc_act}, {acc_scale});")
        for mi in range(Mt):
            for nj in range(Nt):
                lines.append(f"  gemmini_config_ld({kp} * sizeof(elem_t));")
                for kt in range(Kt):
                    w_row = (kt * Nt + nj) * DIM
                    a_off = (mi * DIM) * kp + kt * DIM
                    cad = "acc_ovw" if kt == 0 else "acc_acc"
                    lines += [
                        f"  gemmini_mvin((void*)&T_{lhs}[{a_off}], {a_slot});",
                        f"  gemmini_preload({w_row}, {cad});",
                        f"  gemmini_compute_preloaded({a_slot}, GARBAGE_ADDR);",
                    ]
                c_off = (mi * DIM) * np_ + nj * DIM
                lines.append(f"  gemmini_mvout((void*)&T_{out}[{c_off}], acc_ovw);")
        # The measured arm, not the shipped one: the trailing retire below already covers the LAST
        # job, so a barrier here for every job but the last gives exactly one barrier per job.
        if retire == RETIRE_PER_JOB and job_index < len(job_meta) - 1:
            lines.append("  gemmini_fence();")
    # ONE retire for the whole kernel, not one per job. Every job stages its activation through the
    # same scratchpad rows, so a second job's mvin is a write-after-read on rows the previous job's
    # compute consumed -- which is why this used to fence after each. The reservation station tracks
    # that hazard, so the barrier bought nothing and cost time.
    #
    # ⚠️ THE SAVING IS PER BARRIER REMOVED, NOT PER KERNEL, and the two-job capsule that first showed
    # this could not tell those apart -- it removes exactly one. Swept over job count at one tile of
    # work per job on the citable engine (`rtl_verilator`), on gemmini_rtl 8c3f9923 plus these bytes
    # (both gemmini pins verify False; nothing here is "pinned"):
    #
    #   jobs   1     2     3     4     6     8
    #   once   215   327   441   560   790   1012
    #   per-job 215  421   644   864   1276  1700
    #   saving 0     94    203   304   486   688      -> 97.76 x (jobs - 1) + 3.07, R^2 = 0.99928
    #
    # A constant-saving explanation has a residual sum of squares of 221876 against 159 for that fit.
    # So the 1.299x a two-job kernel sees is the ONE-BARRIER case and a floor: the ratio grows with the
    # job count (1.29x, 1.46x, 1.54x, 1.62x, 1.68x at 2/3/4/6/8 jobs). The cost is a fixed
    # serialisation, not a drain -- across a 4x reduction depth the one removed barrier costs 94/79/97
    # cycles while the kernel around it grows from 327 to 801. Every output was byte-identical and
    # every arm gated correct; both replicates of every point agreed exactly. The one-job kernel has no
    # redundant barrier and both settings emit the identical program (215 vs 215), which is the
    # negative control. Declared as family `PQ` in merlin/contract/capsules/profiles/_perf.yaml; rows
    # and provenance under out/artifacts/perf-bench/gemmini/.
    #
    # The remaining fence is not optional: it retires the last job before `read_cycles` bounds the
    # measured window and before the readback reads the output rows.
    lines.append("  gemmini_fence();")
    lines += ["  c1 = read_cycles();"] + _cepi + [""]

    # Print outputs cropped to the logical m x n (skip padded rows/cols).
    for lhs, out, epi, m, mp, shift in job_meta:
        lines += [
            f'  printf("OUT {out} {m} {n}");',
            f"  for (long i = 0; i < {m}; i++) for (long j = 0; j < {n}; j++)"
            f" printf(\" %d\", (int)T_{out}[i * {np_} + j]);",
            '  printf("\\n");',
        ]
    lines += [
        '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));',
        '  printf("METRIC cycle_window_gemmini_region 1\\n");',
        '  printf("DONE\\n");',
        "  return 0;",
        "}",
    ]
    return "\n".join(lines) + "\n"
