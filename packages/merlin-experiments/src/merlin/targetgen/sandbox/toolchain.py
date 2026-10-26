"""The legit TOOLCHAIN bound back into the deny-by-default sandbox — tools+infra, never answers.

The base :func:`merlin.targetgen.sandbox.bwrap.base_argv` tmpfs-masks all of ``/scratch*``, so the
repo's python/venv and every simulator toolchain disappear. This module binds the LEGIT tools back over
those masks and sets the env to find them, PARAMETERIZED BY THE DESCRIPTOR — never a per-target hand-list:

  * UNIVERSAL tools (venv python, LLVM/MLIR-23, clang-23, the libidn compat shim, DNS) — every target.
  * SIM-FAMILY tools — routed by the descriptor's ``toolchain.sim_via`` through :data:`SIM_TOOLCHAINS`
    (a DECLARATIVE table, no ``if target ==``). ``chipyard`` binds the conda build env + the built
    verilator RTL sim. A bespoke engine a target contributes through its own plugin (``plugin.sim_oracle``)
    that needs nothing beyond the universal set has no row: it resolves to the empty family.
  * The CURATED baremetal C harness — bound + exported iff the descriptor declares one.

The compute-unit ``kind`` (resolved from the capability manifest via :mod:`merlin.targetgen.families`)
is the cross-check: a ``systolic`` target's RTL tiers imply an RTL-sim toolchain, a ``simt``/``vector``
target does not. It drives :func:`required_tool_probes` so the isolation test asserts exactly the tools
that target's kind needs — again with no target name anywhere.
"""

from __future__ import annotations

import os
import shlex
import sys
import sysconfig
from collections import UserDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from merlin.common.paths import compat_lib_dir, env, ext_path, python_source_dir, repo_root
from merlin.targetgen.target_experiment import TargetExperiment

# --------------------------------------------------------------------------- universal toolchain
RESOLVE_DIR = "/run/systemd/resolve"  # /etc/resolv.conf -> here; else DNS fails in bwrap


def _interpreter_prefix() -> Path:
    """Retain a selected venv when pre-3.14 Python runs without site init.

    Resolve the lexical executable's environment before following its usual
    symlink to the base interpreter. No ambient VIRTUAL_ENV selects tools.
    """
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix).resolve()
    candidate = Path(sys.executable).absolute().parent.parent
    return candidate.resolve() if (candidate / "pyvenv.cfg").is_file() else Path(sys.prefix).resolve()


# clang-23 = the ABI's MERLIN_CLANG (rv64_compiler). LLVM-23 ABI-matched to llvm-install; bind ONLY the
# compiler bin + resource dir (NOT src/python_packages, which carry backend lowerings).
@dataclass(frozen=True)
class ToolchainPaths:
    """Explicit universal tool locations; construction never discovers a checkout.

    ``from_checkout`` preserves the legacy runtime defaults, not a standalone
    deployment guarantee. Simulator-family and final answer-mask policy are unchanged.
    Explicit ``python_import_roots`` replace inherited PYTHONPATH; an empty tuple
    clears it. Search roots grant no mounts and do not disable Python's site or
    current-directory lookup. Callers must separately admit the source closure.
    """

    repo: Path
    venv: str
    llvm: str
    compat_lib: str
    clang_install: str
    uv_python: str
    python_import_roots: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.python_import_roots is None:
            return
        if not isinstance(self.python_import_roots, tuple):
            raise ValueError("python_import_roots must be an explicit tuple or None")
        for root in self.python_import_roots:
            if not isinstance(root, str) or not root or ":" in root or "\0" in root or not Path(root).is_absolute():
                raise ValueError("python_import_roots entries must be absolute nonempty paths without colon or NUL")

    @classmethod
    def from_checkout(cls) -> ToolchainPaths:
        repo = repo_root()
        from merlin.llvmlower.toolchain import clang_for

        # The whole-model compiler and the agent sandbox must see the same LLVM
        # install. Detached checkouts need not contain third_party/llvm-install;
        # an explicitly selected MERLIN_CLANG may live in a shared toolchain.
        selected_clang = clang_for(repo, os.environ)
        selected_llvm = repo / "third_party/llvm-install"
        if (
            selected_clang.is_absolute()
            and selected_clang.name == "clang-23"
            and selected_clang.parent.name == "bin"
            and selected_clang.is_file()
            and (selected_clang.parent / "mlir-opt").is_file()
        ):
            selected_llvm = selected_clang.parent.parent.resolve()
        checkout_venv = repo / ".venv"
        # An installed core runs from this interpreter's site-packages even when
        # the operator's work root happens to contain an unrelated ``.venv``.
        # Binding that checkout venv would make the frozen, granted Merlin files
        # import through an interpreter missing their declared dependencies.
        active_install = python_source_dir().resolve() == Path(sysconfig.get_path("purelib")).resolve()
        interpreter_prefix = _interpreter_prefix()
        selected_venv = (
            interpreter_prefix
            if active_install or (not checkout_venv.is_dir() and interpreter_prefix != Path(sys.base_prefix).resolve())
            else checkout_venv
        )
        clang_default = (
            selected_llvm
            if selected_llvm != repo / "third_party/llvm-install"
            else repo / "build/host-merlin-release/install"
        )
        # A frozen bundle may name a different checkout from the implementation
        # running this manager. Search that declared checkout's curated source
        # closure as well as the implementation source; bwrap still exposes only
        # files admitted by the bundle.
        legacy_source = repo / "merlin/python"
        import_roots = tuple(
            dict.fromkeys(
                str(path)
                for path in (
                    *((legacy_source,) if legacy_source.is_dir() else ()),
                    repo / "src",
                    python_source_dir(),
                )
            )
        )
        return cls(
            repo=repo,
            venv=str(selected_venv),
            llvm=str(selected_llvm),
            compat_lib=str(compat_lib_dir()),
            clang_install=env("MERLIN_CLANG_INSTALL", str(clang_default)),
            uv_python=os.path.expanduser("~/.local/share/uv"),
            python_import_roots=import_roots,
        )

    @property
    def clang_bin(self) -> str:
        return self.clang_install + "/bin"

    @property
    def clang_resource(self) -> str:
        return self.clang_install + "/lib/clang"

    @property
    def merlin_clang(self) -> str:
        return self.clang_bin + "/clang-23"


# nested-session env vars UNSET for the agent's claude: inherited from THIS Claude Code session they route
# the spawned claude through the parent's dead localhost SSE relay -> ConnectionRefused. Cleared, it runs
# as a fresh top-level session connecting directly to the API with the stored ~/.claude credentials.
NESTED_SESSION_VARS = (
    "CLAUDECODE",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SSE_PORT",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_EXECPATH",
    "AI_AGENT",
    "CLAUDE_EFFORT",
)


def paths_for_bundle(ws: Path, bundle: dict, *, paths: ToolchainPaths | None = None) -> ToolchainPaths:
    """Select an already granted public SDK import root from its frozen profile."""
    from merlin.common.digest import sha256_file
    from merlin.targetgen.tool_profile import verify_tool_profile

    from . import bwrap

    paths = paths if paths is not None else ToolchainPaths.from_checkout()
    profile = verify_tool_profile(bundle)
    resources = (profile or {}).get("shared_scheduling_resources")
    if resources is None:
        return paths
    root = Path(resources[0]["path"]).parent
    namespace = next(row for row in profile["common_sources"] if row["module"] == "merlin.__init__")
    pins = [*resources, {"path": str(root / "merlin/__init__.py"), "sha256": namespace["sha256"]}]
    _, grants = bwrap._snapshot_grants(ws, bundle, paths.repo)
    for pin in pins:
        sources = [source for _, destination, source in grants if destination == Path(pin["path"])]
        if len(sources) != 1 or not sources[0].is_file() or sha256_file(sources[0]) != pin["sha256"]:
            raise ValueError("public SDK import root lacks its exact frozen public grant bytes")
    original = paths.python_import_roots
    if original is None:
        original = (str(paths.repo / "merlin/python"),)
    return replace(paths, python_import_roots=tuple(dict.fromkeys((str(root), *original))))


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

    bind_paths: tuple[str, ...] = ()  # host dirs to RO-bind back over the /scratch* mask
    path_dirs: tuple[str, ...] = ()  # extra PATH entries (after the universal venv/llvm/clang)
    ld_dirs: tuple[str, ...] = ()  # extra LD_LIBRARY_PATH entries (after the compat shim)
    env_extra: dict = field(default_factory=dict)
    probes: tuple[ToolProbe, ...] = ()


def _chipyard() -> SimToolchain:
    ch = ext_path("chipyard")  # honors .env MERLIN_EXT_CHIPYARD
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
            ToolProbe(
                "verilator RTL sim",
                f"ls {verilator}/simulator-chipyard.harness-* >/dev/null 2>&1 && echo present",
                verilator,
            ),
        ),
    )


class _SimToolchainRegistry(UserDict[str, SimToolchain | Callable[[], SimToolchain]]):
    """Resolve configured families on first access, retaining the mapping interface."""

    def __getitem__(self, key: str) -> SimToolchain:
        value = super().__getitem__(key)
        if callable(value):
            value = value()
            self.data[key] = value
        return value

    def get(self, key, default=None):
        # Missing configuration inside a known resolver must NOT look like an
        # unknown family. Mapping.get otherwise catches the resolver's KeyError.
        return self[key] if key in self.data else default


# sim_via string -> the toolchain family it selects. Additive: a sim engine that needs HOST BINARIES
# bound into the sandbox registers one entry. An engine a target contributes through its own plugin
# (plugin.sim_oracle) whose oracle and simulator are importable from the workspace package needs
# nothing beyond the universal set, so it has no row and resolves to the empty family -- core names no
# target's simulator.
SIM_TOOLCHAINS = _SimToolchainRegistry(
    {
        "chipyard": _chipyard,
        "": SimToolchain(),
    }
)


# The universal tool probes every target's sandbox must satisfy, regardless of kind/sim.
def universal_probes(*, paths: ToolchainPaths | None = None) -> tuple[ToolProbe, ...]:
    paths = paths if paths is not None else ToolchainPaths.from_checkout()
    return (
        ToolProbe("python3", "python3 --version", paths.venv),
        ToolProbe("mlir-opt", "mlir-opt --version | head -1", paths.llvm),
        ToolProbe("clang-23", "clang-23 --version | head -1", paths.clang_bin),
    )


def _sim(te: TargetExperiment) -> SimToolchain:
    """The descriptor's sim family, or the empty (universal-only) family for an engine with no row."""
    return SIM_TOOLCHAINS.get(te.sim_via, SIM_TOOLCHAINS[""])


def curated_harness_dir(te: TargetExperiment) -> str:
    """The curated baremetal C harness dir the descriptor declares (resolved under the experiment dir),
    or "" for a target that declares none (SIMT perf-model targets omit it)."""
    if te.curated_harness:
        p = te.resource_path(te.curated_harness)
        if p.is_dir():
            return str(p)
    return ""


def _venv_runtime_paths(venv: str) -> tuple[str, ...]:
    """Include the runtime behind a venv, whose executable may be external.

    Binding the venv alone preserves its symlinks, but not their referents or
    the base interpreter's standard/shared libraries over the scratch masks.
    Read its own pyvenv.cfg rather than borrowing the controller's interpreter.
    """
    root = Path(venv).resolve()
    executable = root / "bin/python"
    runtimes = set()
    if executable.is_file():
        runtimes.add(executable.resolve().parent.parent)
    config = root / "pyvenv.cfg"
    if config.is_file():
        for line in config.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "home":
                home = Path(value.strip())
                if not home.is_absolute():
                    raise ValueError("venv interpreter home must be absolute")
                runtimes.add(home.resolve().parent)
    return tuple(
        str(path)
        for runtime in sorted(runtimes)
        if runtime != root
        for member in ("bin", "lib", "lib64")
        if (path := runtime / member).exists()
    )


def toolchain_binds(
    te: TargetExperiment,
    *,
    paths: ToolchainPaths | None = None,
    sim: SimToolchain | None = None,
    harness: str | None = None,
    memory_dir: str | None = None,
) -> list[str]:
    """bwrap args binding the legit toolchain back over the /scratch* masks — universal + the descriptor's
    sim family + the curated harness. Nothing here is an answer surface. Also unsets the nested-session
    vars. Append AFTER the base argv + claude runtime binds so these re-appear over the tmpfs.

    Explicit ``sim`` bypasses family discovery. ``harness`` and ``memory_dir`` use
    None for legacy discovery and an empty string for explicit absence.
    """
    paths = paths if paths is not None else ToolchainPaths.from_checkout()
    sim = sim if sim is not None else _sim(te)
    binds: list[str] = []
    universal = (
        paths.uv_python,
        *_venv_runtime_paths(paths.venv),
        paths.venv,
        paths.llvm,
        paths.clang_bin,
        paths.clang_resource,
        paths.compat_lib,
        RESOLVE_DIR,
    )
    harness = curated_harness_dir(te) if harness is None else harness
    for p in (*universal, *sim.bind_paths, *([harness] if harness else ())):
        if Path(p).exists():
            # A snapshot checkout (perf_snapshot.create) does not COPY the external toolchain: it
            # links `.venv`/`third_party` at the real host tool it borrows, so `repo_root()` inside a
            # source worker yields `<snapshot>/.venv` while the bytes live at `<origin>/.venv`. Bind
            # the resolved location as well, so a tool referenced by its real path resolves to the
            # SAME bytes already mounted at the link path -- this adds no content to the sandbox, and
            # the answer masks + `coverage_gap` still run afterwards over the assembled argv.
            # Both destinations are created fresh by bwrap because `base_argv` tmpfs-masks all of
            # `/scratch*` first, so neither is the on-host symlink at mount time. That masking is
            # load-bearing, not incidental: bwrap CANNOT mount onto a symlink destination (it fails
            # with "Unable to mount source on destination" even when the link target exists), so an
            # unmasked snapshot path would refuse the whole sandbox rather than this one bind.
            real = str(Path(p).resolve())
            if real != p and Path(real).exists():
                binds += ["--ro-bind", real, real]
            binds += ["--ro-bind", p, p]
    for v in NESTED_SESSION_VARS:
        binds += ["--unsetenv", v]
    # defence-in-depth: mask the experimenter memory here too (it is also chmod-000 locked). The derived
    # answer-mask pass (bwrap.apply_answer_masks) treats it as already-hidden and adds no redundant mask.
    from merlin.targetgen.sandbox.answer_surfaces import experimenter_memory_dir

    mem = experimenter_memory_dir() if memory_dir is None else Path(memory_dir) if memory_dir else None
    if mem is not None and mem.is_dir():
        binds += ["--tmpfs", str(mem)]
    return binds


def sandbox_env(
    te: TargetExperiment,
    ws: Path,
    *,
    paths: ToolchainPaths | None = None,
    sim: SimToolchain | None = None,
    harness: str | None = None,
    bundle: dict | None = None,
) -> str:
    """Shell ``export``s prepended to the in-sandbox command.

    Explicit import roots set exact ``PYTHONPATH`` without inherited entries.
    Otherwise ``PYTHONPATH`` points at this checkout's package root. The checkout itself is tmpfs-masked, then the
    immutable bundle snapshot re-mounts only the granted package children at their original absolute
    destinations, so Python sees exactly the curated module closure and nothing else.  Pointing at
    ``ws/merlin/python`` was illusory: workspace assembly creates friendly basename links, not a package
    tree.  It only appeared to work in the primary checkout because the shared venv's editable-install
    ``.pth`` happened to point back to that same path; a detached launch worktree correctly failed with
    ``ModuleNotFoundError``.  The explicit checkout path removes that ambient dependency.

    PATH/LD are universal (venv/llvm/clang + compat shim) plus the sim family's dirs — derived, not
    per-target. Explicit ``sim`` bypasses family discovery; ``harness=None`` discovers
    the descriptor resource, while an empty string explicitly omits it.
    """
    paths = paths if paths is not None else ToolchainPaths.from_checkout()
    if bundle is not None:
        paths = paths_for_bundle(ws, bundle, paths=paths)
    sim = sim if sim is not None else _sim(te)
    path = ":".join((f"{paths.venv}/bin", f"{paths.llvm}/bin", paths.clang_bin, *sim.path_dirs))
    ld = ":".join((paths.compat_lib, *sim.ld_dirs))
    parts = [
        # venv FIRST so python3 is the 3.13 venv (xdsl/merlin deps), not conda's 3.10.
        f"export PATH={path}:$PATH; ",
        f"export MERLIN_CLANG={paths.merlin_clang}; ",
        # Importing a candidate's Python package writes __pycache__ INTO that candidate. hash_tree
        # skips those directories, so the bytes land inside a content-addressed artifact while its
        # digest does not cover them -- and the seal gate rightly refuses "digest-excluded ephemeral
        # state". Measured 2026-09-03 on perf_stage_20260903T163936Z: 15 cache dirs appeared three
        # minutes in, the moment the broker first ran the candidate's tools, and the sealed round was
        # thrown away. Not writing them is the fix; the gate stays as the check that it worked.
        "export PYTHONDONTWRITEBYTECODE=1; ",
    ]
    for k, v in sim.env_extra.items():
        parts.append(f"export {k}={v}; ")
    # NOTE: do NOT put {LLVM}/lib on LD_LIBRARY_PATH — it shadows system libLLVM and breaks the host C/C++
    # compilers. mlir-opt/llc find their libs via rpath.
    parts.append(f"export LD_LIBRARY_PATH={ld}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}; ")
    if not ws.is_absolute() or ":" in str(ws) or "\0" in str(ws):
        raise ValueError("sandbox workspace must be an absolute Python import root")
    # Carry the exact writable import root to trusted provider import guards.
    parts.append(f"export MERLIN_SANDBOX_WORKSPACE={shlex.quote(str(ws))}; ")
    # The controller sets PYTHONSAFEPATH=1, so python -c never adds the working
    # directory implicitly. The broker clients are staged here, and candidate
    # modules also need this explicit, writable import root inside the box.
    if paths.python_import_roots is None:
        roots = f"{paths.repo}/merlin/python:{ws}"
        parts.append(f"export PYTHONPATH={shlex.quote(roots)}${{PYTHONPATH:+:$PYTHONPATH}}; ")
    else:
        parts.append(f"export PYTHONPATH={shlex.quote(':'.join((*paths.python_import_roots, str(ws))))}; ")
    harness = curated_harness_dir(te) if harness is None else harness
    if harness:
        # A target-neutral var + the per-target-named one back-compat consumers read. The per-target name
        # is DERIVED from the target string (<target> -> MERLIN_<TARGET>_HARNESS_DIR), not hard-coded, so the
        # target's backend/probe see the identical var with zero target branch.
        parts.append(f"export MERLIN_HWBRINGUP_HARNESS_DIR={harness}; ")
        parts.append(f"export MERLIN_{te.target.upper()}_HARNESS_DIR={harness}; ")
    return "".join(parts)


def required_tool_probes(
    te: TargetExperiment,
    *,
    paths: ToolchainPaths | None = None,
    sim: SimToolchain | None = None,
) -> list[ToolProbe]:
    """The tools THIS target's sandbox must provide = universal + its sim family's probes. (The compute-
    unit kind cross-checks this: a systolic target's chipyard family carries the RTL-sim probes; a SIMT
    target's plugin-contributed simulator carries none beyond the universal set.)"""
    sim = sim if sim is not None else _sim(te)
    return [*universal_probes(paths=paths), *sim.probes]
