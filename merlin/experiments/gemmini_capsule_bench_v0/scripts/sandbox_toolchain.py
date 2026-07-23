"""The legit TOOLCHAIN made available inside the bwrap sandbox — tools+infra, never answers.

`bwrap_argv` (deny-by-default) tmpfs-masks all of /scratch and /scratch2, so the repo's own python/venv
and the chipyard build+sim toolchain disappear. This module binds the LEGIT tools back over those masks —
exactly the set mined from 27 past runs (clang/mlir, cmake/ninja/make, spike, riscv-gcc, verilator-L3,
python) — and sets the env to find them. It binds NO answer surface:
  • chipyard/generators/gemmini (the full kernel suite + RTL) stays masked; the harness is a CURATED copy
    (linker/crt/headers only, no kernel .c) under contracts/harness_curated/.
  • merlin/runtime oracle + merlin/targets/gemmini lowerings are not bound (only the workspace's curated
    merlin pkg is importable, via PYTHONPATH).
"""
from __future__ import annotations
import os
from pathlib import Path

import _common as C
from merlin.common.paths import ext_path, env

# Chipyard build+sim toolchain — resolved via ext_path('chipyard') (honors .env MERLIN_EXT_CHIPYARD),
# NOT a hard-coded path, so it survives moves. .conda-env carries cmake/ninja/make + spike + riscv-gcc
# + libs (riscv-tools/bin); sims/verilator is the built L3 RTL simulator.
_CHIPYARD = ext_path("chipyard")
CONDA_ENV = str(_CHIPYARD / ".conda-env") if _CHIPYARD else "/path/to/chipyard/.conda-env"
CHIPYARD_VERILATOR = str(_CHIPYARD / "sims" / "verilator") if _CHIPYARD else "/path/to/chipyard/sims/verilator"
UV_PYTHON = os.path.expanduser("~/.local/share/uv")          # the cpython the .venv symlinks point at
VENV = str(C.REPO / ".venv")                                 # third-party deps (xdsl, numpy, jsonschema…)
LLVM = str(C.REPO / "third_party" / "llvm-install")          # clang + mlir-opt/translate (also bundle-allowed)


def _curated_harness() -> str:
    """The curated baremetal C harness dir, from the experiment descriptor (per-target SETUP), resolved
    under the experiment dir. Falls back to the pre-descriptor gemmini default where it exists, so an
    existing gemmini run is byte-identical; returns "" for a target that declares none (arc/cyclotron)."""
    try:
        from merlin.targetgen.target_experiment import load_target_experiment
        te = load_target_experiment(C.EXP / "target_experiment.yaml")
        if te.curated_harness:
            return str(C.EXP / te.curated_harness)
    except Exception:  # noqa: BLE001 — no/invalid descriptor ⇒ fall back, never crash the sandbox
        pass
    default = C.EXP / "contracts/harness_curated/gemmini-rocc-tests"
    return str(default) if default.is_dir() else ""


CURATED_HARNESS = _curated_harness()
# libidn compat shim: the conda cmake transitively loads libidn.so.11 (during project configure), but this
# host only has libidn.so.12. `.compat_lib/libidn.so.11 -> libidn.so.12` bridges it. Under --sandbox none
# (abc4) this was on the ambient LD_LIBRARY_PATH; our explicit env dropped it -> the C++ build failed
# "libidn.so.11: cannot open shared object file". Must be on LD_LIBRARY_PATH wherever cmake configures.
COMPAT_LIB = str(C.REPO / ".compat_lib")
# clang-23 = the ABI's `MERLIN_CLANG` (rv64_compiler: ".ll -> rv64 object"), LLVM-23 ABI-matched to
# llvm-install. The C++ baseline REQUIRES it to build/link its OOT MLIR project (g++ links the clang-built
# LLVM libs with undefined-reference errors). It lives in the IREE merlin install; bind ONLY the compiler
# bin + its resource dir — NOT src/ or python_packages/ (which hold IREE source incl. gemmini lowerings).
CLANG_INSTALL = env("MERLIN_CLANG_INSTALL",
                    "/scratch2/agustin/merlin/build/host-merlin-release/install")
CLANG_BIN = CLANG_INSTALL + "/bin"
CLANG_RESOURCE = CLANG_INSTALL + "/lib/clang"          # clang -print-resource-dir -> lib/clang/23
MERLIN_CLANG = CLANG_INSTALL + "/bin/clang-23"
# Derive the experimenter-memory dir from the CURRENT repo (Claude Code slugifies the project path by
# replacing '/' with '-'). Hard-coding it left the OLD '-scratch-agustin-projects-merlin' path — which
# no longer exists — so the CURRENT memory went UNMASKED (a cheat gap). Deriving it keeps the mask honest.
MEMORY_DIR = os.path.expanduser(f"~/.claude/projects/{str(C.REPO).replace('/', '-')}/memory")
RESOLVE_DIR = "/run/systemd/resolve"   # /etc/resolv.conf -> here; without it DNS fails inside bwrap

# nested-session env vars that must be UNSET for the agent's claude: inherited from THIS Claude Code
# session, they make the spawned claude route through the parent's (now-dead) localhost SSE relay
# (CLAUDE_CODE_SSE_PORT) -> ConnectionRefused. Cleared, the agent runs as a fresh top-level session that
# connects directly to the API with the stored ~/.claude credentials.
NESTED_SESSION_VARS = ["CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SSE_PORT",
                       "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_EXECPATH",
                       "AI_AGENT", "CLAUDE_EFFORT"]


def toolchain_binds() -> list[str]:
    """bwrap args binding the legit toolchain back over the /scratch* tmpfs masks. Append AFTER
    bwrap_argv + claude_runtime_binds so these re-appear; nothing here is an answer surface."""
    b = []
    for p in (CONDA_ENV, CHIPYARD_VERILATOR, UV_PYTHON, VENV, LLVM, CURATED_HARNESS, RESOLVE_DIR,
              CLANG_BIN, CLANG_RESOURCE, COMPAT_LIB):
        if Path(p).exists():
            b += ["--ro-bind", p, p]
    # clear nested-session vars so the agent's claude connects directly to the API (not the dead relay)
    for v in NESTED_SESSION_VARS:
        b += ["--unsetenv", v]
    # defence-in-depth: mask the experimenter memory even though it is also chmod-000 locked
    if Path(MEMORY_DIR).exists():
        b += ["--tmpfs", MEMORY_DIR]
    return b


def sandbox_env(ws: Path) -> str:
    """Shell `export`s prepended to the in-sandbox command. PYTHONPATH points at the WORKSPACE's curated
    merlin pkg (the repo editable-install path is masked by bwrap), so only granted modules import."""
    return (
        # .venv FIRST so `python3` is the 3.13 venv (xdsl/merlin deps) — NOT conda's 3.10. conda still
        # provides cmake/ninja/gcc/g++; riscv-tools provides spike/riscv-gcc; llvm-install provides mlir-opt.
        f'export PATH={VENV}/bin:{LLVM}/bin:{CLANG_BIN}:{CONDA_ENV}/bin:{CONDA_ENV}/riscv-tools/bin:$PATH; '
        f'export MERLIN_CLANG={MERLIN_CLANG}; '   # the ABI's rv64_compiler (.ll -> rv64 object)
        f'export RISCV={CONDA_ENV}/riscv-tools; '
        # NOTE: do NOT put {LLVM}/lib on LD_LIBRARY_PATH — it shadows the system libLLVM and breaks the
        # system C/C++ compilers. mlir-opt/llc find their libs via rpath; the C++ build links via cmake.
        f'export LD_LIBRARY_PATH={COMPAT_LIB}:{CONDA_ENV}/lib:{CONDA_ENV}/riscv-tools/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}; '
        f'export PYTHONPATH={ws}/merlin/python${{PYTHONPATH:+:$PYTHONPATH}}; '
        # Curated baremetal harness: a target-neutral var + the gemmini-named one the gemmini backend and
        # probe still read (back-compat). Skipped entirely for a target that declares no curated harness.
        + (f'export MERLIN_HWBRINGUP_HARNESS_DIR={CURATED_HARNESS}; '
           f'export MERLIN_GEMMINI_HARNESS_DIR={CURATED_HARNESS}; ' if CURATED_HARNESS else '')
    )
