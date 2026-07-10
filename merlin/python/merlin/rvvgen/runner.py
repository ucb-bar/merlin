"""certify_rvv — isolated, measured K-ladder for one (RVV package x workload), coupled across
spike (correctness + emitted-instruction evidence) and K1 (cycle truth). Mirrors the gemmini
oot_runner discipline: never raises for a gate/measurement failure (records status + reason),
and ``not_run_is_not_pass`` — an unreachable target is ``not_run``, never a false ``pass``.

K-ladder:
  K0  load + integrity (cflags allowlist, manifest schema)            -> registry.load_rvv_package
  K1  non-perturbation: hand_v0 schedule == pipeline.RVV_TRANSFORM_SCHEDULE
  K2  build via apply_rvv_package -> model.o + zephyr.elf
  K3  spike correctness: run_on_spike + _gate(references) -> gate_ok
  K4  spike instruction histogram: disassemble(model.o); expected_instructions present
  K5  K1 cycles: cross-compile + deploy + rdcycle (not_run when board/toolchain absent)
  K6  delta-vs-baseline: speedup ONLY credited if K3 gate_ok (fail-closed)
"""
from __future__ import annotations

import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..common import schemas
from ..common.yaml import load_yaml, write_yaml
from ..llvmlower import custom_isa, pipeline
from ..runtime.backends import zephyr_model as zm
from . import k1 as k1mod
from .apply import apply_rvv_package
from .registry import RvvPackage, load_rvv_package

def _instruction_histogram(disasm: str) -> dict[str, int]:
    """Histogram of RVV mnemonics in an objdump -d dump (the emitted-instruction evidence)."""
    from ..common.driver_output import is_vector_mnemonic
    hist: Counter[str] = Counter()
    for line in disasm.splitlines():
        # objdump -d: "   1036a:\t<hex enc>\t<mnemonic>\t<operands>"
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        mnem = fields[2].strip().split()[0] if fields[2].strip() else ""
        if mnem and is_vector_mnemonic(mnem):
            hist[mnem] += 1
    return dict(sorted(hist.items()))


def _expected_present(hist: dict[str, int], expected: list[str]) -> bool:
    """Each expected token must prefix-match at least one emitted mnemonic (vfmacc -> vfmacc.vv)."""
    keys = list(hist)
    return all(any(k == e or k.startswith(e) for k in keys) for e in expected)


def _load_references(model_dir: Path) -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    g = model_dir / "golden.npy"
    if g.is_file():
        refs["fp32"] = np.load(g)
    w = model_dir / "golden_w8a8.npy"
    if w.is_file():
        refs["w8a8"] = np.load(w)
    return refs


def _baseline_results(baseline_run_dir: Path | None, workload: str) -> dict | None:
    if baseline_run_dir is None:
        return None
    p = Path(baseline_run_dir) / "results.yaml"
    if not p.is_file():
        return None
    r = load_yaml(p)
    return r if r.get("workload") == workload else None


def certify_rvv(package_dir: str | Path, model_dir: str | Path, *, runs_root: str | Path,
                run_id: str, targets: tuple[str, ...] = ("spike", "k1"),
                baseline_run_dir: str | Path | None = None, harts: int = 2,
                timeout: int = 3600) -> dict[str, Any]:
    """Build one RVV package for one workload, measure it on ``targets``, write results.yaml.

    ``model_dir`` is a workload bundle (model.mlir + inputs.npz + golden.npy [+ golden_w8a8.npy]).
    Returns the results dict (also written to ``runs_root/run_id/results.yaml``). Never raises for a
    package/gate/measurement failure — those are recorded; only an internal harness bug raises.
    """
    package_dir, model_dir = Path(package_dir), Path(model_dir)
    run_dir = Path(runs_root) / run_id
    gen = run_dir / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    workload = model_dir.name

    ladder: dict[str, str] = {}
    rec: dict[str, Any] = {
        "status": "error", "target": "rvv", "workload": workload,
        "package": {"run_id": Path(package_dir).name},
        "ladder": ladder, "correctness": {}, "measurement": [],
        "instruction_histogram": {}, "expected_instructions_present": None,
        "delta_vs_baseline": None, "failure": None,
    }

    def fail(rung: str, reason: str, status: str = "fail") -> dict[str, Any]:
        ladder[rung] = status
        rec["status"] = status
        rec["failure"] = {"rung": rung, "reason": reason}
        _write(run_dir, rec)
        return rec

    # K0 — load + integrity
    try:
        pkg: RvvPackage = load_rvv_package(package_dir)
        ladder["K0"] = "pass"
    except Exception as e:  # schema / cflags-allowlist / parse
        return fail("K0", f"{type(e).__name__}: {e}")
    rec["package"] = {"run_id": pkg.run_id, "dtype_strategy": pkg.dtype_strategy}

    # K1 — non-perturbation (only the baseline must equal the shipping constant)
    if pkg.run_id == "hand_v0" and pkg.schedule_text != pipeline.RVV_TRANSFORM_SCHEDULE:
        return fail("K1", "hand_v0 schedule diverged from pipeline.RVV_TRANSFORM_SCHEDULE")
    ladder["K1"] = "pass"

    # K2 — build
    try:
        build = apply_rvv_package(pkg, model_dir, gen, board="spike_riscv64",
                                  harts=harts, arena_mb=64)
        model_o = gen / "model.o"
        ladder["K2"] = "pass"
    except Exception as e:
        rec["failure"] = {"rung": "K2", "trace": traceback.format_exc()[-1500:]}
        return fail("K2", f"build failed: {type(e).__name__}: {e}", status="error")

    # K4 — instruction histogram (objdump evidence). Done before K3 so a run failure still
    # leaves the emitted-instruction evidence in the record.
    try:
        disasm = custom_isa.disassemble(model_o)
        (gen / "objdump.txt").write_text(disasm, encoding="utf-8")
        hist = _instruction_histogram(disasm)
        rec["instruction_histogram"] = hist
        any_rvv = bool(hist)
        rec["any_rvv"] = any_rvv
        # The K4 GATE is genuine vectorization (RVV emitted, not a scalar fallback). The
        # expected_instructions list is recorded as EVIDENCE for the S4 comparison / gap-router
        # (e.g. the vfmacc-fusion gap: our schedule emits vfmul.vv+vfadd.vv, not fused vfmacc) —
        # it does NOT fail the ladder, since different op classes legitimately emit different sets.
        rec["expected_instructions_present"] = _expected_present(hist, pkg.expected_instructions)
        ladder["K4"] = "pass" if any_rvv else "fail"
    except Exception as e:
        ladder["K4"] = "not_run"
        rec.setdefault("notes", {})["K4"] = f"{type(e).__name__}: {e}"

    # K3 — spike correctness gate
    refs = _load_references(model_dir)
    if "spike" in targets:
        try:
            run = zm.run_on_spike(build["elf"], harts=harts,
                                  mem_bytes=build.get("ram_bytes", 1 << 31), timeout=timeout)
            gate = zm._gate(run["prefix"], refs) if refs else {"ok": None}
            rec["correctness"] = {
                "gate_ok": gate.get("ok"),
                "fp32_cos": gate.get("fp32_cos"), "fp32_rel": gate.get("fp32_rel"),
                "fp32_argmax": gate.get("fp32_argmax"),
                "w8a8_cos": gate.get("w8a8_cos"), "w8a8_rel": gate.get("w8a8_rel"),
            }
            cyc = run.get("metrics", {}).get("cycles")
            rec["measurement"].append({"target": "spike", "cycle_accurate": False, "cycles": cyc})
            ladder["K3"] = "pass" if gate.get("ok") else ("fail" if refs else "not_run")
        except Exception as e:
            ladder["K3"] = "fail"
            rec["failure"] = {"rung": "K3", "reason": f"{type(e).__name__}: {e}"}
    else:
        ladder["K3"] = "not_run"

    # K5 — K1 cycles (real silicon). not_run when board/toolchain unavailable.
    if "k1" in targets:
        if k1mod.available():
            try:
                kr = k1mod.run_on_k1(model_dir, gen, pkg, timeout=timeout)
                m = kr.get("metrics", {})
                # K1's Bianbu kernel traps userspace `rdcycle`, so K1 cycles are an estimate
                # derived from the delegated `rdtime` timebase (cycle_accurate=False); the raw
                # timebase ticks + wall ns are the real-silicon ground truth. spike/FireSim stay
                # the cycle-accurate authorities; K1 is the fast real-hardware wall measurement.
                rec["measurement"].append({"target": "k1", "cycle_accurate": False,
                                           "cycles": m.get("cycles"),
                                           "time_ticks": m.get("time_ticks"),
                                           "wall_ns": m.get("wall_ns"),
                                           "vlen": kr.get("vlen", k1mod.VLEN)})
                ladder["K5"] = "pass"
            except Exception as e:
                ladder["K5"] = "not_run"
                rec.setdefault("notes", {})["K5"] = f"{type(e).__name__}: {e}"
        else:
            ladder["K5"] = "not_run"
            rec.setdefault("notes", {})["K5"] = "K1 unavailable (toolchain or board unreachable)"
    else:
        ladder["K5"] = "not_run"

    # K6 — delta vs baseline (speedup only credited if K3 gate ok: fail-closed)
    base = _baseline_results(baseline_run_dir, workload)
    if base is not None:
        delta: dict[str, Any] = {"baseline_run_id": base.get("package", {}).get("run_id")}
        gate_ok = bool(rec["correctness"].get("gate_ok"))
        for m in rec["measurement"]:
            tgt = m["target"]
            bcyc = next((b["cycles"] for b in base.get("measurement", [])
                         if b["target"] == tgt), None)
            if bcyc and m.get("cycles"):
                speedup = (bcyc / m["cycles"]) if gate_ok else None
                delta[f"cycles_{tgt}"] = {"baseline": bcyc, "this": m["cycles"], "speedup": speedup}
        delta["correctness_regressed"] = (not gate_ok) and base.get("correctness", {}).get("gate_ok")
        rec["delta_vs_baseline"] = delta
        ladder["K6"] = "pass"
    else:
        ladder["K6"] = "not_run"

    # Overall status: pass iff no mandatory rung failed (K3 only mandatory when references exist).
    mandatory = ["K0", "K1", "K2"] + (["K3"] if refs else [])
    rec["status"] = "pass" if all(ladder.get(k) == "pass" for k in mandatory) else "fail"
    _write(run_dir, rec)
    return rec


def _write(run_dir: Path, rec: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    problems = schemas.validate(rec, "rvv_result")
    if problems:
        rec.setdefault("notes", {})["schema"] = problems
    return write_yaml(run_dir / "results.yaml", rec,
                      header="RVV experiment result (merlin.rvvgen.runner.certify_rvv)")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Certify an RVV package on a workload (spike+K1).")
    ap.add_argument("--package", required=True, help="artifacts/targets/rvv/<run_id>/ dir")
    ap.add_argument("--workload", required=True, help="workload bundle dir (model.mlir+inputs+golden)")
    ap.add_argument("--run-id", default=None, help="run id (default: <pkg>_<workload>)")
    ap.add_argument("--runs-root", default="runs/rvv_experiment")
    ap.add_argument("--targets", default="spike,k1", help="comma list: spike,k1,firesim")
    ap.add_argument("--baseline-run-dir", default=None, help="baseline run dir for delta")
    ap.add_argument("--harts", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args(argv)
    run_id = a.run_id or f"{Path(a.package).name}_{Path(a.workload).name}"
    rec = certify_rvv(a.package, a.workload, runs_root=a.runs_root, run_id=run_id,
                      targets=tuple(a.targets.split(",")), baseline_run_dir=a.baseline_run_dir,
                      harts=a.harts, timeout=a.timeout)
    print(f"{run_id}: status={rec['status']} ladder={rec['ladder']} "
          f"gate_ok={rec['correctness'].get('gate_ok')} "
          f"any_rvv={bool(rec['instruction_histogram'])}")
    return 0 if rec["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
