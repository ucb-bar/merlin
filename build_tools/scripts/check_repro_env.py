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

# CROSS-CUTTING capabilities — toolchains/frameworks NOT tied to any single target's contract. One row
# per capability: key -> (workstream, human title, env-var hints); probed by _probe(key) below (a broken
# import is reported, never fatal). The TARGET-specific execution substrates (spike/verilator/vcs/zephyr
# per target) are NOT listed here — they are DISCOVERED from each target contract's ``runtime.backends``
# by _build_capabilities() and probed generically by _backend_probe(), so a newly-registered target's
# substrates show up with zero edits to this file (no ``if key == "gemmini_spike"`` branching).
_BASE_CAPABILITIES: dict[str, tuple[str, str, list[str]]] = {
    "xdsl": ("all", "xDSL prototyping plane (host dialect descent)", ["(pip extra: .[xdsl])"]),
    "llvm_m2m_toolchain": ("WS-A", "model2MLIR + clang-23 whole-model lowering",
                           ["MERLIN_M2M_DIR", "MERLIN_M2M_VENV", "MERLIN_CLANG", "MERLIN_IREE_BIN"]),
    "spike_rv64gcv": ("WS-A/D", "spike rv64gcv RVV oracle",
                      ["MERLIN_CHIPYARD", "MERLIN_SPIKE", "MERLIN_RISCV_GCC"]),
    "k1_board": ("WS-A/D", "SpacemiT K1 board (real cycles)",
                 ["MERLIN_K1_HOST", "MERLIN_K1_SSH_KEY", "MERLIN_K1_TOOLCHAIN"]),
    "firesim": ("WS-B", "FireSim job queue (L5)", ["MERLIN_EXT_FIRESIM_QUEUE", "FIRESIM_ROOT"]),
    "zephyr_spike": ("WS-A/C", "Zephyr SW build_app path (whole-model spike compile)",
                     ["ZEPHYR_BASE", "MERLIN_ZEPHYR_SW", "ZEPHYR_SDK_INSTALL_DIR", "MERLIN_CHIPYARD"]),
    "zephyr_multicore": ("WS-A/C", "multicore RVV Zephyr image (OpenMP shim over pinned harts)",
                         ["ZEPHYR_BASE", "MERLIN_ZEPHYR_SW", "ZEPHYR_SDK_INSTALL_DIR",
                          "MERLIN_CHIPYARD"]),
    "circt_firtool": ("WS-B", "CIRCT firtool + FileCheck (L4 rtlchecks)",
                      ["MERLIN_CHIPYARD", "(firtool/FileCheck on PATH)"]),
    "chia": ("WS-B/D", "chia agentic-loop framework (build/chia-venv)", ["(uv venv build/chia-venv)"]),
    "llm_api": ("WS-B/D", "Anthropic API for real agentic runs", ["ANTHROPIC_API_KEY", "MERLIN_LLM_MODEL"]),
}

# Cross-cutting env hints keyed on the DECLARED substrate label — the shared toolchain ROOTS a backend
# needs (chipyard/spike/gcc/zephyr), which are target-generic. The per-TARGET substrate var name is NOT
# enumerated here: it is DERIVED from (target, sim) by _backend_env_hints() below. Informational only.
_BACKEND_ENV: dict[str, list[str]] = {
    "simulator": ["(pure-Python model — always available)"],
    "baremetal": ["MERLIN_SPIKE", "MERLIN_RISCV_GCC", "MERLIN_CHIPYARD"],
    "zephyr": ["ZEPHYR_BASE", "MERLIN_ZEPHYR_SW", "ZEPHYR_SDK_INSTALL_DIR", "MERLIN_CHIPYARD"],
}

# Backend label -> the SIM token in the per-target env-var name MERLIN_<TARGET>_<SIM>. This is the same
# derivation convention targetgen/sandbox/toolchain.py uses (``MERLIN_{target.upper()}_...``): a target's
# verilator sim is MERLIN_<TARGET>_VERILATOR (e.g. MERLIN_MUON_VERILATOR / MERLIN_SATURN_VERILATOR), its
# VCS simv binary is MERLIN_<TARGET>_SIMV (e.g. MERLIN_GEMMINI_SIMV), its RTL spike kernel sim is
# MERLIN_<TARGET>_SPIKE. So each newly-registered target's substrate var name derives with zero edits.
_BACKEND_SIM_TOKEN: dict[str, str] = {"verilator": "VERILATOR", "vcs": "SIMV"}


def _backend_env_hints(target: str, backend: str) -> list[str]:
    """Env-var hints for one target's declared execution substrate: the DERIVED per-target substrate var
    ``MERLIN_<TARGET>_<SIM>`` (for the RTL kernel sims that resolve a per-target binary — verilator/vcs
    and any ``spike_*`` kernel sim) followed by the cross-cutting toolchain roots from ``_BACKEND_ENV``.
    No per-target env name is baked in — the target string threads through, matching the toolchain.py
    naming convention, so this stays correct for a target with no bespoke entry here."""
    tok = _BACKEND_SIM_TOKEN.get(backend)
    if tok is None and backend.startswith("spike_"):
        tok = "SPIKE"
    hints: list[str] = []
    if tok:
        hints.append(f"MERLIN_{target.upper()}_{tok}")
        hints.append("MERLIN_CHIPYARD")   # RTL sims are built under the chipyard toolchain root
    return hints + _BACKEND_ENV.get(backend, [])


def _build_capabilities() -> tuple[dict[str, tuple[str, str, list[str]]], dict[str, tuple[str, str]]]:
    """Merge the cross-cutting base rows with per-target substrate rows DISCOVERED from every target
    contract's ``runtime.backends`` (via the target registry). Returns ``(capabilities, derived)`` where
    ``derived`` maps each generated key -> ``(target, backend)`` so _backend_probe can probe it.
    Registry/contract failures degrade to the base rows only (never fatal)."""
    caps: dict[str, tuple[str, str, list[str]]] = dict(_BASE_CAPABILITIES)
    derived: dict[str, tuple[str, str]] = {}
    try:
        from merlin.targetgen import target_registry as tr
        names = tr.all_targets()
    except Exception:
        return caps, derived
    for name in names:
        try:
            backends = list((tr.load_contract(name).get("runtime", {}) or {}).get("backends", []) or [])
        except Exception:
            backends = []
        for b in backends:
            key = f"{name}_{b}"
            caps[key] = ("target", f"{name}: {b} execution substrate", _backend_env_hints(name, b))
            derived[key] = (name, b)
    return caps, derived


def _backend_probe(target: str, backend: str) -> tuple[str, str]:
    """Probe a target's declared execution substrate. Dispatches on the DECLARED backend label (from the
    contract's ``runtime.backends``), never on the target name, so any target declaring the same substrate
    is probed identically. RTL kernel simulators (``verilator`` / ``spike_*``) resolve the target's own
    kernel backend from the runtime backend REGISTRY (``base.get_backend``) — registry-driven, no per-
    target branch."""
    try:
        if backend == "simulator":
            return ("available", "pure-Python model (merlin.runtime.simulator)")
        if backend == "vcs":
            ho = importlib.import_module("merlin.targetgen.heavy_oracles")
            return ("available" if ho.vcs_available() else "unavailable", "")
        if backend == "zephyr":
            zm = importlib.import_module("merlin.runtime.backends.zephyr_model")
            return ("available" if zm.available() else "unavailable", "")
        if backend == "baremetal":
            sp = importlib.import_module("merlin.runtime.backends.spike")
            return ("available" if sp.available() else "unavailable", "rv64gcv spike (baremetal RVV)")
        if backend == "verilator" or backend.startswith("spike_"):
            sim = "verilator" if backend == "verilator" else "spike"
            base = importlib.import_module("merlin.runtime.backends.base")
            if target not in base.list_backends():
                return ("unavailable", f"no kernel backend registered for target '{target}'")
            mod = base.get_backend(target)
            try:
                ok = mod.available(sim)
            except TypeError:                       # backend has no simulator-mode arg
                ok = mod.available()
            return ("available" if ok else "unavailable", sim)
    except Exception as e:  # pragma: no cover - a broken guard should report, not crash
        return ("error", f"{type(e).__name__}: {e}")
    return ("unavailable", f"no preflight probe for backend '{backend}'")


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
        if key == "k1_board":
            k1 = importlib.import_module("merlin.mining.k1")
            return ("available" if k1.available() else "unavailable",
                    os.environ.get("MERLIN_K1_HOST", "MERLIN_K1_HOST unset"))
        if key == "firesim":
            ho = importlib.import_module("merlin.targetgen.heavy_oracles")
            return ("available" if ho.firesim_queue_alive() else "unavailable", "")
        if key == "zephyr_spike":
            zm = importlib.import_module("merlin.runtime.backends.zephyr_model")
            return ("available" if zm.available() else "unavailable", "")
        if key == "zephyr_multicore":
            # Same toolchain as zephyr_spike plus the in-repo OpenMP shim the multicore image
            # links; spike -pN provides V on every hart, so no extra simulator is needed.
            zm = importlib.import_module("merlin.runtime.backends.zephyr_model")
            paths_mod = importlib.import_module("merlin.common.paths")
            shim = paths_mod.runtime_dir() / "c" / "libomp_zephyr.c"
            if not shim.is_file():
                return ("unavailable", f"missing OpenMP shim {shim}")
            return ("available" if zm.available() else "unavailable", "")
        if key == "circt_firtool":
            ft = shutil.which("firtool")
            fc = shutil.which("FileCheck")
            if ft and fc:
                return ("available", f"{ft}")
            missing = [n for n, v in (("firtool", ft), ("FileCheck", fc)) if not v]
            return ("unavailable", f"missing on PATH: {', '.join(missing)}")
        if key == "chia":
            # chia loops run under the ISOLATED out/build/chia-venv (never the main .venv), so probe the
            # loop interpreter's existence, not chia_bridge.chia_available() (which checks importability
            # from THIS main .venv, where chia is intentionally absent -> always False).
            from merlin.common.paths import build_dir
            chia_py = build_dir() / "chia-venv" / "bin" / "python"
            return ("available" if chia_py.is_file() else "unavailable",
                    str(chia_py) if chia_py.is_file() else "uv venv out/build/chia-venv")
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
    from merlin.common.paths import build_dir, repo_root
    root = repo_root()
    out: dict[str, str] = {}
    venv = root / ".venv" / "bin" / "python"
    out[".venv (driver)"] = str(venv) if venv.exists() else "MISSING (uv sync --all-extras)"
    # chia-venv lives under the out/build root (the single generated-output convention), not repo/build.
    chia = build_dir() / "chia-venv" / "bin" / "python"
    out["out/build/chia-venv"] = str(chia) if chia.exists() else "MISSING (uv venv out/build/chia-venv)"
    from merlin.common.paths import env as _env
    m2m_dir = _env("MERLIN_M2M_DIR", "")
    m2m_venv = _env("MERLIN_M2M_VENV") or (f"{m2m_dir}/.venv" if m2m_dir else "")
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

    capabilities, derived = _build_capabilities()
    results = {k: dict(zip(("workstream", "title", "env"), v)) for k, v in capabilities.items()}
    for k in results:
        status, detail = _backend_probe(*derived[k]) if k in derived else _probe(k)
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
