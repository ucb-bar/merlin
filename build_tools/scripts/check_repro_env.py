#!/usr/bin/env python3
"""Reproduction-environment preflight: report which experiment capabilities are runnable here.

The experiment-reproduction roadmap runs across several external toolchains (model2MLIR + clang-23,
spike rv64gcv, the SpacemiT K1 board, Gemmini spike/verilator/VCS/FireSim, CIRCT/firtool, chia agentic
loops, and the Anthropic API). Each degrades fail-closed — its tests skip and its runner records
``not_run`` rather than faking a result. This script asks every capability's OWN availability guard
(``toolchain.available()``, ``spike.available()``, ``k1.available()``, ``gem.available(sim)``,
``chia_available()``, ``vcs_available()``, ``firesim_queue_alive()``) so you can see, before kicking off
real runs, exactly what will really execute vs. skip — and why (which env var to set).

It never runs a workload and has no side effects; it only probes. Import/probe errors are caught
per-capability so one missing dependency can't sink the whole report.

Usage:
  check_repro_env.py                     # human-readable table (always exit 0)
  check_repro_env.py --json              # machine-readable
  check_repro_env.py --require spike_rv64gcv,k1_board   # exit 1 unless all named caps are available
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

# One row per capability: key -> (workstream, human title, env-var hints). The probe is resolved by
# _probe(key) below so a broken import is reported, never fatal.
CAPABILITIES: dict[str, tuple[str, str, list[str]]] = {
    "xdsl": ("all", "xDSL prototyping plane (host dialect descent)", ["(pip extra: .[xdsl])"]),
    "llvm_m2m_toolchain": ("WS-A", "model2MLIR + clang-23 whole-model lowering",
                           ["MERLIN_M2M_DIR", "MERLIN_M2M_VENV", "MERLIN_CLANG", "MERLIN_IREE_BIN"]),
    "spike_rv64gcv": ("WS-A/D", "spike rv64gcv RVV oracle",
                      ["MERLIN_CHIPYARD", "MERLIN_SPIKE", "MERLIN_RISCV_GCC"]),
    "saturn_vec": ("WS-A", "Saturn-vectors RVV spike cert", ["MERLIN_CHIPYARD", "MERLIN_SPIKE"]),
    "k1_board": ("WS-A/D", "SpacemiT K1 board (real cycles)",
                 ["MERLIN_K1_HOST", "MERLIN_K1_SSH_KEY", "MERLIN_K1_TOOLCHAIN"]),
    "gemmini_spike": ("WS-B", "Gemmini spike functional oracle (L2)",
                      ["MERLIN_GEMMINI_SPIKE", "MERLIN_CHIPYARD"]),
    "gemmini_verilator": ("WS-B", "Gemmini verilator cycle-accurate RTL (L3)",
                          ["MERLIN_GEMMINI_VERILATOR", "MERLIN_CHIPYARD"]),
    "gemmini_vcs": ("WS-B", "Gemmini VCS (L4)", ["MERLIN_GEMMINI_SIMV"]),
    "firesim": ("WS-B", "FireSim job queue (L5)", ["MERLIN_EXT_FIRESIM_QUEUE", "FIRESIM_ROOT"]),
    "circt_firtool": ("WS-B", "CIRCT firtool + FileCheck (L4 rtlchecks)",
                      ["MERLIN_CHIPYARD", "(firtool/FileCheck on PATH)"]),
    "chia": ("WS-B/D", "chia agentic-loop framework (build/chia-venv)", ["(uv venv build/chia-venv)"]),
    "llm_api": ("WS-B/D", "Anthropic API for real agentic runs", ["ANTHROPIC_API_KEY", "MERLIN_LLM_MODEL"]),
}


def _probe(key: str) -> tuple[str, str]:
    """Return (status, detail) for a capability. status in {available, unavailable, error}."""
    try:
        if key == "xdsl":
            m = importlib.import_module("merlin.xdsl_dialects._common")
            return ("available" if getattr(m, "HAS_XDSL", False) else "unavailable", "")
        if key == "llvm_m2m_toolchain":
            tc = importlib.import_module("merlin.llvmlower.toolchain")
            return ("available" if tc.available() else "unavailable", "")
        if key == "spike_rv64gcv":
            sp = importlib.import_module("merlin.runtime.backends.spike")
            return ("available" if sp.available() else "unavailable", "")
        if key == "saturn_vec":
            sv = importlib.import_module("merlin.runtime.backends.saturn_vec")
            return ("available" if sv.available() else "unavailable", "")
        if key == "k1_board":
            k1 = importlib.import_module("merlin.rvvgen.k1")
            return ("available" if k1.available() else "unavailable",
                    os.environ.get("MERLIN_K1_HOST", "MERLIN_K1_HOST unset"))
        if key in ("gemmini_spike", "gemmini_verilator"):
            gem = importlib.import_module("merlin.runtime.backends.gemmini")
            sim = "spike" if key == "gemmini_spike" else "verilator"
            return ("available" if gem.available(sim) else "unavailable", sim)
        if key == "gemmini_vcs":
            ho = importlib.import_module("merlin.targetgen.heavy_oracles")
            return ("available" if ho.vcs_available() else "unavailable", "")
        if key == "firesim":
            ho = importlib.import_module("merlin.targetgen.heavy_oracles")
            return ("available" if ho.firesim_queue_alive() else "unavailable", "")
        if key == "circt_firtool":
            ft = shutil.which("firtool")
            fc = shutil.which("FileCheck")
            if ft and fc:
                return ("available", f"{ft}")
            missing = [n for n, v in (("firtool", ft), ("FileCheck", fc)) if not v]
            return ("unavailable", f"missing on PATH: {', '.join(missing)}")
        if key == "chia":
            cb = importlib.import_module("merlin.benchharness.chia_bridge")
            return ("available" if cb.chia_available() else "unavailable",
                    "build/chia-venv" if not cb.chia_available() else "")
        if key == "llm_api":
            has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
            try:
                importlib.import_module("anthropic")
                has_sdk = True
            except ModuleNotFoundError:
                has_sdk = False
            model = os.environ.get("MERLIN_LLM_MODEL", "claude-opus-4-8 (default)")
            if has_key and has_sdk:
                return ("available", f"model={model}")
            reason = []
            if not has_key:
                reason.append("ANTHROPIC_API_KEY unset")
            if not has_sdk:
                reason.append("anthropic SDK not importable")
            return ("unavailable", "; ".join(reason))
    except Exception as e:  # pragma: no cover - a broken guard should report, not crash
        return ("error", f"{type(e).__name__}: {e}")
    return ("error", "unknown capability")


def _interpreters() -> dict[str, str]:
    """Report the three isolated interpreters the reproduction flow keeps separate."""
    from merlin.common.paths import repo_root
    root = repo_root()
    out: dict[str, str] = {}
    venv = root / ".venv" / "bin" / "python"
    out[".venv (driver)"] = str(venv) if venv.exists() else "MISSING (uv sync --all-extras)"
    chia = root / "build" / "chia-venv" / "bin" / "python"
    out["build/chia-venv"] = str(chia) if chia.exists() else "MISSING (uv venv build/chia-venv)"
    m2m_venv = os.environ.get("MERLIN_M2M_VENV") or (
        f"{os.environ.get('MERLIN_M2M_DIR', '')}/.venv" if os.environ.get("MERLIN_M2M_DIR") else "")
    m2m_py = Path(m2m_venv) / "bin" / "python" if m2m_venv else None
    out["$MERLIN_M2M_VENV"] = str(m2m_py) if (m2m_py and m2m_py.exists()) else "unset/MISSING"
    return out


def _out_roots() -> dict[str, str]:
    from merlin.common.paths import artifacts_dir, build_dir, out_dir, runs_dir
    return {"out": str(out_dir()), "runs": str(runs_dir()),
            "artifacts": str(artifacts_dir()), "build": str(build_dir())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--require", default="", help="comma-separated capability keys; exit 1 if any unavailable")
    args = ap.parse_args(argv)

    results = {k: dict(zip(("workstream", "title", "env"), v)) for k, v in CAPABILITIES.items()}
    for k in results:
        status, detail = _probe(k)
        results[k]["status"] = status
        results[k]["detail"] = detail

    interps = _interpreters()
    roots = _out_roots()

    if args.json:
        print(json.dumps({"capabilities": results, "interpreters": interps, "out_roots": roots}, indent=2))
    else:
        mark = {"available": "OK ", "unavailable": "-- ", "error": "ERR"}
        print("== Reproduction environment preflight ==\n")
        print("Interpreters (keep isolated):")
        for name, path in interps.items():
            print(f"  {name:20s} {path}")
        print("\nOutput roots (single out/ convention):")
        for name, path in roots.items():
            print(f"  {name:20s} {path}")
        print("\nCapabilities:")
        for k, r in results.items():
            line = f"  [{mark[r['status']]}] {k:20s} {r['workstream']:8s} {r['title']}"
            if r["status"] != "available" and r["detail"]:
                line += f"\n        -> {r['detail']}"
            if r["status"] == "unavailable" and r["env"]:
                line += f"\n        set: {', '.join(r['env'])}"
            print(line)
        n_ok = sum(1 for r in results.values() if r["status"] == "available")
        print(f"\n{n_ok}/{len(results)} capabilities available.")

    if args.require:
        want = [c.strip() for c in args.require.split(",") if c.strip()]
        missing = [c for c in want if results.get(c, {}).get("status") != "available"]
        if missing:
            print(f"\nREQUIRED but not available: {', '.join(missing)}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
