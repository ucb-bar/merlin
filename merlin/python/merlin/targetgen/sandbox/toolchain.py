"""The legit TOOLCHAIN bound back into the deny-by-default sandbox — tools+infra, never answers.

The base :func:`merlin.targetgen.sandbox.bwrap.base_argv` tmpfs-masks all of ``/scratch*``, so the
repo's python/venv and every simulator toolchain disappear. This module binds the LEGIT tools back over
those masks and sets the env to find them, PARAMETERIZED BY THE DESCRIPTOR — never a per-target hand-list:

  * UNIVERSAL tools (venv python, LLVM/MLIR-23, clang-23, the libidn compat shim, DNS) — every target.
  * SIM-FAMILY tools — routed by the descriptor's ``toolchain.sim_via`` through :data:`SIM_TOOLCHAINS`
    (a DECLARATIVE table, no ``if target ==``). ``chipyard`` binds the conda build env + the built
    verilator RTL sim; ``cyclotron`` (SIMT perf-model path) needs no extra binaries beyond the venv.
  * The CURATED baremetal C harness — bound + exported iff the descriptor declares one.

The compute-unit ``kind`` (resolved from the capability manifest via :mod:`merlin.targetgen.families`)
is the cross-check: a ``systolic`` target's RTL tiers imply an RTL-sim toolchain, a ``simt``/``vector``
target does not. It drives :func:`required_tool_probes` so the isolation test asserts exactly the tools
that target's kind needs — again with no target name anywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common.paths import env, ext_path, repo_root
from merlin.targetgen.target_experiment import TargetExperiment

# --------------------------------------------------------------------------- universal toolchain
_REPO = repo_root()
UV_PYTHON = os.path.expanduser("~/.local/share/uv")          # the cpython the .venv symlinks point at
VENV = str(_REPO / ".venv")                                  # third-party deps (xdsl, numpy, jsonschema…)
LLVM = str(_REPO / "third_party" / "llvm-install")           # clang + mlir-opt/translate
COMPAT_LIB = str(_REPO / ".compat_lib")                      # libidn.so.11 -> .12 shim the conda cmake needs
RESOLVE_DIR = "/run/systemd/resolve"                         # /etc/resolv.conf -> here; else DNS fails in bwrap
# clang-23 = the ABI's MERLIN_CLANG (rv64_compiler). LLVM-23 ABI-matched to llvm-install; bind ONLY the
# compiler bin + resource dir (NOT src/python_packages, which carry backend lowerings).
CLANG_INSTALL = env("MERLIN_CLANG_INSTALL",
                    "/scratch2/agustin/merlin/build/host-merlin-release/install")
CLANG_BIN = CLANG_INSTALL + "/bin"
CLANG_RESOURCE = CLANG_INSTALL + "/lib/clang"
MERLIN_CLANG = CLANG_INSTALL + "/bin/clang-23"

# nested-session env vars UNSET for the agent's claude: inherited from THIS Claude Code session they route
# the spawned claude through the parent's dead localhost SSE relay -> ConnectionRefused. Cleared, it runs
# as a fresh top-level session connecting directly to the API with the stored ~/.claude credentials.
NESTED_SESSION_VARS = ("CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SSE_PORT",
                       "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_EXECPATH",
                       "AI_AGENT", "CLAUDE_EFFORT")


@dataclass(frozen=True)
class ToolProbe:
    """One tool the sandbox must provide. ``bind`` = the host path that must be RO-bound in the argv
    (hermetic check); ``cmd`` = a shell command that must exit 0 inside the sandbox (live check)."""
    label: str
    cmd: str
    bind: str | None = None


@dataclass(frozen=True)
class SimToolchain:
    """A simulator FAMILY's toolchain, selected by the descriptor's ``sim_via`` (declarative, not coded)."""
    bind_paths: tuple[str, ...] = ()      # host dirs to RO-bind back over the /scratch* mask
    path_dirs: tuple[str, ...] = ()       # extra PATH entries (after the universal venv/llvm/clang)
    ld_dirs: tuple[str, ...] = ()         # extra LD_LIBRARY_PATH entries (after the compat shim)
    env_extra: dict = field(default_factory=dict)
    probes: tuple[ToolProbe, ...] = ()


def _chipyard() -> SimToolchain:
    ch = ext_path("chipyard")                                # honors .env MERLIN_EXT_CHIPYARD
    conda = str(ch / ".conda-env") if ch else "/nonexistent/chipyard/.conda-env"
    verilator = str(ch / "sims" / "verilator") if ch else "/nonexistent/chipyard/sims/verilator"
    return SimToolchain(
        bind_paths=(conda, verilator),
        path_dirs=(conda + "/bin", conda + "/riscv-tools/bin"),
        ld_dirs=(conda + "/lib", conda + "/riscv-tools/lib"),
        env_extra={"RISCV": conda + "/riscv-tools"},
        probes=(
            ToolProbe("g++", "g++ --version | head -1", conda),
            ToolProbe("cmake>=3.20", "cmake --version | head -1", conda),
            ToolProbe("ninja", "ninja --version", conda),
            ToolProbe("make", "make --version | head -1", conda),
            ToolProbe("spike", "spike --help 2>&1 | head -1", conda),
            ToolProbe("riscv64-unknown-elf-gcc", "riscv64-unknown-elf-gcc --version | head -1", conda),
            # target/config-agnostic: assert SOME built RTL sim exists (the exact config binary is a
            # per-design detail), not a hard-coded config name.
            ToolProbe("verilator RTL sim",
                      f'ls {verilator}/simulator-chipyard.harness-* >/dev/null 2>&1 && echo present',
                      verilator),
        ),
    )


# sim_via string -> the toolchain family it selects. Additive: a new sim backend registers one entry.
SIM_TOOLCHAINS: dict[str, SimToolchain] = {
    "chipyard": _chipyard(),
    # SIMT perf-model path (radiance/cyclotron): the muon oracle + sim are pure-Python (importable from
    # the workspace merlin pkg on PYTHONPATH); no extra host binaries to bind.
    "cyclotron": SimToolchain(),
    "": SimToolchain(),
}

# The universal tool probes every target's sandbox must satisfy, regardless of kind/sim.
UNIVERSAL_PROBES: tuple[ToolProbe, ...] = (
    ToolProbe("python3", "python3 --version", VENV),
    ToolProbe("mlir-opt", "mlir-opt --version | head -1", LLVM),
    ToolProbe("clang-23", "clang-23 --version | head -1", CLANG_BIN),
)


def _sim(te: TargetExperiment) -> SimToolchain:
    return SIM_TOOLCHAINS.get(te.sim_via, SIM_TOOLCHAINS[""])


def curated_harness_dir(te: TargetExperiment) -> str:
    """The curated baremetal C harness dir the descriptor declares (resolved under the experiment dir),
    or "" for a target that declares none (SIMT perf-model targets omit it)."""
    if te.curated_harness:
        p = te.path.parent / te.curated_harness
        if p.is_dir():
            return str(p)
    return ""


def toolchain_binds(te: TargetExperiment) -> list[str]:
    """bwrap args binding the legit toolchain back over the /scratch* masks — universal + the descriptor's
    sim family + the curated harness. Nothing here is an answer surface. Also unsets the nested-session
    vars. Append AFTER the base argv + claude runtime binds so these re-appear over the tmpfs."""
    sim = _sim(te)
    binds: list[str] = []
    universal = (UV_PYTHON, VENV, LLVM, CLANG_BIN, CLANG_RESOURCE, COMPAT_LIB, RESOLVE_DIR)
    harness = curated_harness_dir(te)
    for p in (*universal, *sim.bind_paths, *([harness] if harness else ())):
        if Path(p).exists():
            binds += ["--ro-bind", p, p]
    for v in NESTED_SESSION_VARS:
        binds += ["--unsetenv", v]
    # defence-in-depth: mask the experimenter memory here too (it is also chmod-000 locked). The derived
    # answer-mask pass (bwrap.apply_answer_masks) treats it as already-hidden and adds no redundant mask.
    from merlin.targetgen.sandbox.answer_surfaces import experimenter_memory_dir
    mem = experimenter_memory_dir()
    if mem.is_dir():
        binds += ["--tmpfs", str(mem)]
    return binds


def sandbox_env(te: TargetExperiment, ws: Path) -> str:
    """Shell ``export``s prepended to the in-sandbox command. PYTHONPATH points at the WORKSPACE's curated
    merlin pkg (the repo editable-install path is masked), so only granted modules import. PATH/LD are
    universal (venv/llvm/clang + compat shim) plus the sim family's dirs — derived, not per-target."""
    sim = _sim(te)
    path = ":".join((f"{VENV}/bin", f"{LLVM}/bin", CLANG_BIN, *sim.path_dirs))
    ld = ":".join((COMPAT_LIB, *sim.ld_dirs))
    parts = [
        # venv FIRST so python3 is the 3.13 venv (xdsl/merlin deps), not conda's 3.10.
        f'export PATH={path}:$PATH; ',
        f'export MERLIN_CLANG={MERLIN_CLANG}; ',
    ]
    for k, v in sim.env_extra.items():
        parts.append(f'export {k}={v}; ')
    # NOTE: do NOT put {LLVM}/lib on LD_LIBRARY_PATH — it shadows system libLLVM and breaks the host C/C++
    # compilers. mlir-opt/llc find their libs via rpath.
    parts.append(f'export LD_LIBRARY_PATH={ld}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}; ')
    parts.append(f'export PYTHONPATH={ws}/merlin/python${{PYTHONPATH:+:$PYTHONPATH}}; ')
    harness = curated_harness_dir(te)
    if harness:
        # A target-neutral var + the per-target-named one back-compat consumers read. The per-target name
        # is DERIVED from the target string (gemmini -> MERLIN_GEMMINI_HARNESS_DIR), not hard-coded, so the
        # gemmini backend/probe see the identical var with zero target branch.
        parts.append(f'export MERLIN_HWBRINGUP_HARNESS_DIR={harness}; ')
        parts.append(f'export MERLIN_{te.target.upper()}_HARNESS_DIR={harness}; ')
    return "".join(parts)


def required_tool_probes(te: TargetExperiment) -> list[ToolProbe]:
    """The tools THIS target's sandbox must provide = universal + its sim family's probes. (The compute-
    unit kind cross-checks this: a systolic target's chipyard family carries the RTL-sim probes; a SIMT
    target's cyclotron family carries none beyond the universal set.)"""
    return [*UNIVERSAL_PROBES, *_sim(te).probes]
