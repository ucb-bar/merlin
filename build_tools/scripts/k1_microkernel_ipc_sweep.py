#!/usr/bin/env python
"""Sweep the MICRO-KERNEL register block (MR x NR x KC, plus the unroll_m / k_block recipes) on the
real SpacemiT K1 and record, per point, BOTH axes that a wall-clock ranking conflates:

  * ``ticks``    — rdtime on the kernel bracket (what we actually want to minimise), and
  * ``instret``  — retired instructions on the SAME bracket (perf_event_open, ``util.h``).

Wall time alone cannot separate "emits too many instructions" from "stalls on each instruction",
and ranking on wall time alone is exactly what hid the vector-width/IPC tradeoff this sweep exists
to crack: NR=32 (LMUL=4, 32 f32 lanes/insn) reaches ~1.36x of XNNPACK's instruction count but its
IPC collapses, so it LOSES on time to the NR=16 point that issues nearly twice the instructions.

Each point also keeps its EMITTED CODE (``objdump.txt`` of the lowered ``model.o``) and an
``emitted_digest`` over the mnemonic stream, so an INERT lever — a knob that changes the schedule
text and nothing else — is detected instead of being credited with a measurement. Two shipped
levers were inert this way (``KC``; ``MR`` under ``unroll_m``), so the digest is not optional.

Fail-closed by construction: a build failure, a run failure or a missing ``VERIFY PASS`` yields a
``not_run`` row carrying the exact blocker — never a timing.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merlin.common.artifacts import cache_dir
from merlin.common.driver_output import int_after, int_field
from merlin.common.paths import repo_root
from merlin.kernels.microkernel import MicrokernelSpec, UnsupportedAxis
from merlin.mining import k1

import k1_cross_framework_ops as X  # the shared build/deploy/parse path (same protocol as the matrix)

REPO = Path(repo_root())


def _objdump(obj: Path, out: Path) -> str | None:
    """Disassemble the lowered ``model.o`` and keep it beside the measurement (the evidence a
    claim about the inner loop has to be made from). Returns the text, or None if unreadable."""
    from merlin.kernels.decode.objdump import objdump_bin
    try:
        p = subprocess.run([objdump_bin(), "-d", "--mattr=+v", str(obj)],
                           capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    out.write_text(p.stdout, encoding="utf-8")
    return p.stdout


def _digest(text: str) -> str:
    import hashlib
    from merlin.kernels.decode import rvv as _rvv
    stream = _rvv.decode_text(text)
    body = "\n".join(f"{i.raw.mnemonic} {','.join(i.raw.operands)}" for i in stream.insns)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _inner_loop_facts(text: str) -> dict:
    """Structural facts about the INNERMOST loop of the emitted kernel, read from the decoded
    stream (vtype state machine), never guessed from mnemonic substrings.

    These are the numbers a stall hypothesis lives or dies on: how many vector FMAs the loop body
    issues, how many vector loads/stores (a SPILL inside the K loop shows up here, and at LMUL=4 one
    spill moves 128 B in a single instruction — cheap to count, expensive to run), and the effective
    LMUL those instructions actually run under."""
    from merlin.kernels.decode import rvv as _rvv
    stream = _rvv.decode_text(text)
    # Scope to the innermost loop that does VECTOR arithmetic, not simply the tightest loop. A
    # recipe that peels (the VL-agnostic scalable one peels N) leaves a SCALAR remainder loop that
    # is tighter than the vector K loop, and the plain innermost span then reports that tail's mix
    # (0 vfmacc, 0 vle) as though the micro-kernel had not vectorized at all.
    span = stream.innermost_vector_loop() or stream.innermost_loop()
    facts = {"vtype_histogram": stream.vtype_histogram()}
    if span is None:
        return {**facts, "inner_loop": None}
    insns = stream.insns_in(span)
    vt = {}
    for i in insns:
        if i.is_vector and i.vtype:
            vt[str(i.vtype)] = vt.get(str(i.vtype), 0) + 1
    facts["inner_loop"] = {
        "span": [hex(span[0]), hex(span[1])],
        "n_insns": len(insns),
        "n_vector": sum(1 for i in insns if i.is_vector),
        "vfmacc": stream.count_in(span, "vfmacc"),
        "vle": stream.count_in(span, "vle"),
        "vse": stream.count_in(span, "vse"),
        "vlse": stream.count_in(span, "vlse"),
        "flw": stream.count_in(span, "flw"),
        "fsw": stream.count_in(span, "fsw"),
        "vmv": stream.count_in(span, "vmv"),
        "vslide": stream.count_in(span, "vslide"),
        "vset": stream.count_in(span, "vset"),
        "vtype_in_loop": vt,
        "mnemonics": [i.raw.mnemonic for i in insns],
    }
    facts["n_backedges"] = len(stream.loop_spans())
    return facts


def _run_point(spec: MicrokernelSpec, S: int, reps: int, tag: str, keep: Path,
               extra_features: list[str], *, march: str | None = None) -> dict:
    """Lower + build + run ONE micro-kernel point; keep its objdump. Fail-closed.

    ``march`` overrides the codegen march string. VL_DYNAMIC points are compiled WITHOUT the ``_zvl``
    pin (``march=k1.K1_MARCH``) on purpose: pinning would hide whether the emitted loop really sizes
    itself to the hardware at run time (the whole point of the axis), and the cross-cutting finding
    is that the pin may not even take effect at scale. VL_FIXED points keep the pinned default so the
    VL-agnostic loop is measured against the strongest fixed-width baseline."""
    from merlin.mining import workloads
    from merlin.mining.from_strategy import microkernel_features
    from merlin.kernels.microkernel import VL_DYNAMIC

    if march is None:
        march = k1.K1_MARCH if spec.vl_strategy == VL_DYNAMIC else k1.codegen_march()
    base = {"op": "f32_gemm", "dtype": "f32", "M": S, "N": S, "K": S, "target": "k1",
            "mode": "inner_compute", "timer": "rdtime", "timebase_hz": k1.K1_TIMEBASE_HZ,
            "source": "ours_microkernel_sweep", "microkernel": asdict(spec), "march": march}
    try:
        feats = list(extra_features) + microkernel_features(spec.to_knobs())
    except UnsupportedAxis as e:
        return {**base, "ticks": None, "status": "not_run", "blocker": f"UnsupportedAxis: {e}"}
    base["compiler_features"] = feats

    bundle = workloads.gen_matmul_f32(cache_dir("rvv_workloads"), M=S, N=S, K=S)
    work = keep / "work"
    work.mkdir(parents=True, exist_ok=True)
    model_o, cgen, err = X._lower_ours(bundle, tag, feats, int8=False, vectorize=True, work=work,
                                       march=march)
    if err is not None:
        return {**base, "ticks": None, "status": "not_run", "blocker": err}

    text = _objdump(model_o, keep / "objdump.txt")
    if text is not None:
        base["emitted_digest"] = _digest(text)
        facts = _inner_loop_facts(text)
        (keep / "inner_loop.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
        # The full mnemonic list lives in inner_loop.json; the row keeps only the counters so the
        # jsonl stays diffable.
        if facts.get("inner_loop"):
            facts["inner_loop"] = {k: v for k, v in facts["inner_loop"].items() if k != "mnemonics"}
        base.update(facts)

    cc = X._cc()
    rt = REPO / "merlin/runtime/c"
    abi = REPO / "merlin/runtime/abi"
    inc = []
    for d in (X.K1H, X.HERE, cgen, rt):
        inc += ["-I", str(d)]
    binp = keep / f"{tag}.elf"
    srcs = [str(X.HERE / "ours_gemm_driver.c"), str(cgen / "model_call.c"),
            str(rt / "merlin_model.c"), str(abi / "mlir_runtime.c"), str(model_o)]
    cmd = [str(cc), *inc, *X._K1_CFLAGS, f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}",
           "-static", "-o", str(binp), *srcs, "-lm", "-lpthread"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError) as e:
        return {**base, "ticks": None, "status": "not_run", "blocker": f"link exec failed: {e}"}
    if p.returncode != 0 or not binp.is_file():
        return {**base, "ticks": None, "status": "not_run",
                "blocker": f"link failed rc={p.returncode}: {p.stderr.strip()[-500:]}"}

    # The board is SHARED (other agents measure concurrently) and the cross-framework harness's
    # _deploy_run does not take the host-wide lock, so a concurrent run would contend for the 8
    # cores and poison the rdtime measurement. Hold the lock across all reps of one point; blocking
    # here is expected, not a hang.
    best = None
    with k1.board_lock():
        for rep in range(reps):
            console, detail = X._deploy_run(binp, f"{tag}_{rep}", timeout=900)
            r = X._parse(base, console, detail)
            if r["status"] != "pass":
                return r                          # first failure is the honest blocker
            r["instret"] = int_after(console, "INSTRET")
            r["instret_full"] = int_after(console, "INSTRET_FULL")
            r["errors"] = int_field(console, "errors")
            if best is None or r["ticks"] < best["ticks"]:
                best = r
    best["reps"] = reps
    best["timing"] = "min_of_reps"
    if best.get("ticks") and best.get("instret"):
        best["ins_per_tick"] = round(best["instret"] / best["ticks"], 3)
    return best


def _run_xnn(S: int, reps: int, tag: str) -> dict:
    base = {"op": "f32_gemm", "dtype": "f32", "M": S, "N": S, "K": S, "source": "xnnpack",
            "target": "k1", "mode": "inner_compute", "timer": "rdtime",
            "timebase_hz": k1.K1_TIMEBASE_HZ,
            "kernel_file": "tmp/kernels/XNNPACK/src/f32-gemm/gen/f32-gemm-7x4v-rvv.c"}
    with k1.board_lock():                          # same fairness gate as the ours points
        return X._build_run_xnn(tag, X.HERE / "xnnpack_gemm_driver_7x4v.c",
                                [f"-DGEMM_M={S}", f"-DGEMM_N={S}", f"-DGEMM_K={S}"],
                                reps=reps, base=base)


def _parse_specs(text: str) -> list[MicrokernelSpec]:
    """``MR:NR:KC[:flag,flag]`` per comma-free ';'-separated item.

    Boolean flags in {unroll_m, k_block, pack}; ``vl_dynamic`` is the VL-agnostic-loop shorthand for
    ``vl_strategy='dynamic'`` (a scalable N block sized to the runtime VL). Under vl_dynamic, NR is
    the block width in lanes at the RVV MINIMUM VLEN (128 bits): NR=16 -> vector<[8]xf32> == LMUL 4."""
    from merlin.kernels.microkernel import VL_DYNAMIC
    out = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        MR, NR, KC = int(parts[0]), int(parts[1]), int(parts[2])
        flags = parts[3].split(",") if len(parts) > 3 and parts[3] else []
        kw: dict = {}
        for f in flags:
            if not f:
                continue
            if f == "vl_dynamic":
                kw["vl_strategy"] = VL_DYNAMIC
            else:
                kw[f] = True
        out.append(MicrokernelSpec(MR=MR, NR=NR, KC=KC, **kw))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="4:16:16;2:32:16;4:32:16;8:32:16",
                    help="';'-separated MR:NR:KC[:flags] points")
    ap.add_argument("--shapes", default="128")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag-prefix", default="w_", help="unique remote-tag prefix (boards are shared)")
    ap.add_argument("--features", default="erase_self_copy",
                    help="comma-separated extra compiler features applied to every point")
    ap.add_argument("--xnn", action="store_true", help="also measure the XNNPACK reference")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    specs = _parse_specs(a.specs)
    shapes = [int(s) for s in a.shapes.split(",")]
    extra = [f for f in a.features.split(",") if f]
    root = cache_dir("microkernel_ipc_sweep")
    rows: list[dict] = []
    for S in shapes:
        if a.xnn:
            print(f"--- xnnpack f32_gemm {S}^3 ---", flush=True)
            r = _run_xnn(S, a.reps, f"{a.tag_prefix}xnn_{S}")
            print("   ", r["status"], r.get("ticks"), r.get("blocker", ""), flush=True)
            rows.append(r)
        for spec in specs:
            from merlin.kernels.microkernel import VL_DYNAMIC
            flags = "".join(k[0] for k in ("unroll_m", "k_block", "pack") if getattr(spec, k))
            if spec.vl_strategy == VL_DYNAMIC:
                flags += "V"                       # VL-agnostic (scalable) N block
            tag = f"{a.tag_prefix}mk{spec.MR}x{spec.NR}x{spec.KC}{flags}_{S}"
            keep = root / tag
            keep.mkdir(parents=True, exist_ok=True)
            print(f"--- ours {spec.MR}x{spec.NR}x{spec.KC}{'+' + flags if flags else ''} {S}^3 ---",
                  flush=True)
            r = _run_point(spec, S, a.reps, tag, keep, extra)
            r["artifact_dir"] = str(keep)
            il = (r.get("inner_loop") or {})
            print("   ", r["status"], r.get("ticks"), "instret=", r.get("instret"),
                  "ins/tick=", r.get("ins_per_tick"), "digest=", r.get("emitted_digest"),
                  "loop_insns=", il.get("n_insns"), r.get("blocker", ""), flush=True)
            rows.append(r)

    outp = Path(a.out) if a.out else (root / "sweep.jsonl")
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(rows)} rows -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
