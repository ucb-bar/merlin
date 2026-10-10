"""Generic runtime backend for a RoCC accelerator built and simulated with chipyard, served from DATA.

A support provider selects this module by reference (``plugin.backend:
merlin.runtime.backends.chipyard_rocc``) instead of shipping its own backend. Discovery
(:mod:`merlin.runtime.backends.base`) then loads THIS FILE as a fresh per-target instance named
``merlin._oot_backends.<target>``, so the instance learns which target it serves from its own
``__name__`` and reads everything else from that target's data:

* the selected capability contract (``runner.toolchain``, ``runner.sim_via``, ``runtime.rtl_sim_config``,
  ``harness_abi``, ``execution_capabilities``, ``readout_semantics``, ``counter_semantics``);
* the provider's ISA-headers spec (``plugin.isa_headers``, :mod:`merlin.targetgen.isa_headers_spec`):
  the upstream header checkout by pin and per-file digest, its include roots, CRT and flags;
* the shared engine resolvers: :mod:`merlin.targetgen.gsim_emulator` (emulator + build receipt),
  :mod:`merlin.targetgen.spike_extension` (functional model), :mod:`merlin.targetgen.runtime_build`
  (memory map, sim config) and :mod:`merlin.targetgen.rtl_engine_policy` (GSIM runtime slots).

Plumbing only. It runs an ELF someone else built (spike = functional bootstrap, verilator/gsim = the
elaborated design), parses the shared ``OUT``/``OUT_ND``/``METRIC``/``DONE`` console, and describes
how a runner-owned harness is compiled and linked. It contains no compiler: the harness renderer is
the generic :func:`merlin.targetgen.contract.harness_render.render_harness`, and RoCC semantics / RTL
checks are whatever generic modules the provider's ``plugin.rocc_semantics`` / ``plugin.rtl_checks``
name. Imported under its own name (in-tree discovery does this) it registers nothing: it is a
template until a provider selects it.
"""

from __future__ import annotations

import hashlib
import os
import resource
import subprocess
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from math import prod
from pathlib import Path
from typing import Any

from merlin.runtime.backends._chipyard_memory_readback import (  # noqa: F401 -- the backend's transport API
    memory_readback_transport,
    prepare_memory_readback,
)
from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register, target_class_for

_OOT_PREFIX = "merlin._oot_backends."

#: The simulator engines this generic backend knows how to drive. A property of THIS implementation
#: (spike with an extension, chipyard's Verilator harness, Merlin's GSIM harness), not of a target.
ENGINES = ("spike", "verilator", "gsim")

#: Merlin's GSIM harness exit status when ``+max-cycles`` elapses before the design's stop condition
#: (chipyard_harness/main.cpp: "GSIM timeout: no RTL/TSI completion after N cycles").
_GSIM_CYCLE_CAP_EXIT = 124

#: What an elaborated-RTL model prints when the DESIGN's own ``assert`` fails. Both RTL engines print
#: it and only one of them stops, so it is matched on the design's text, never on an exit convention.
_RTL_ASSERTION_MARKER = "Assertion failed"


class ChipyardRoccError(RuntimeError):
    """A run or a declaration this backend cannot honor, with the reason.

    A failed simulator run carries its (possibly partial) ``stdout``/``stderr`` so callers can keep the
    transcript and the ``METRIC`` lines printed before the failure."""

    def __init__(self, message: str, *, stdout=None, stderr=None):
        super().__init__(message)
        self.stdout, self.stderr = stdout, stderr


class SimulatorTimeout(ChipyardRoccError):
    """The engine stopped the run at its own hang bound (e.g. GSIM ``+max-cycles``): a budget limit,
    not a verdict on the program."""


def _served_target() -> str | None:
    return __name__[len(_OOT_PREFIX) :] if __name__.startswith(_OOT_PREFIX) else None


#: The target this instance serves (None for the unselected template import).
TARGET_NAME: str | None = _served_target()


def _target() -> str:
    if TARGET_NAME is None:
        raise ChipyardRoccError(
            f"{__name__} is the generic chipyard RoCC template; it serves a target only when a selected "
            "support provider names it as plugin.backend (MERLIN_TARGET_PATH)"
        )
    return TARGET_NAME


def _env_name(what: str) -> str | None:
    if TARGET_NAME is None:
        return None
    from merlin.common.paths import target_env_name

    return target_env_name(TARGET_NAME, what)


#: Per-target overrides, derived from the target name (``MERLIN_<TARGET>_<WHAT>``), never listed.
GSIM_EMU_ENV = _env_name("GSIM_EMU")
VERILATOR_ENV = _env_name("VERILATOR")
GSIM_MAXCYCLES_ENV = _env_name("GSIM_MAXCYCLES")
SPIKE_ENV = _env_name("SPIKE")
VERILATOR_CONFIG_ENV = _env_name("VERILATOR_CONFIG")
HARNESS_DIR_ENV = _env_name("HARNESS_DIR")


# --- data ---------------------------------------------------------------------------------------------
def _contract() -> dict[str, Any]:
    """The selected capability contract (honoring an experiment's observed/selected view)."""
    from merlin.targetgen import target_registry

    contract = target_registry.resolve(_target()).load_contract()
    if not isinstance(contract, dict):
        raise ChipyardRoccError(f"{_target()}: the selected contract is not a mapping")
    return contract


def _block(name: str, *, within: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = _contract() if within is None else within
    block = source.get(name)
    if not isinstance(block, dict):
        where = "contract" if within is None else "block"
        raise ChipyardRoccError(f"{_target()}: the selected {where} declares no {name!r} mapping")
    return block


def _toolchain() -> dict[str, Any]:
    return _block("toolchain", within=_block("runner"))


def _plugin() -> dict[str, Any]:
    from merlin.targetgen.plugins import resolve_support

    return resolve_support(_target()).plugin()


def _provider_root() -> Path:
    from merlin.targetgen.plugins import provider_root, resolve_support

    selected = resolve_support(_target())
    return provider_root(selected.base, selected.plugin().get("path"))


def _isa_headers_spec() -> Path:
    from merlin.targetgen.providers import contained_resource

    reference = _plugin().get("isa_headers")
    if not isinstance(reference, str) or not reference:
        raise ChipyardRoccError(f"{_target()}: the selected support declares no plugin.isa_headers spec")
    return contained_resource(_provider_root(), reference)


def isa_headers(*, verify: bool = True):
    """The provider's content-pinned ISA-header environment (raises when it cannot be produced)."""
    from merlin.targetgen import isa_headers_spec

    try:
        return isa_headers_spec.load(_target(), _isa_headers_spec(), override_env=HARNESS_DIR_ENV, verify=verify)
    except isa_headers_spec.IsaHeadersError as exc:
        raise ChipyardRoccError(str(exc)) from exc


def __getattr__(name: str):
    """Data-derived constants and plugin-selected capabilities, resolved when first asked for."""
    if name == "SPIKE_EXTENSION_NAME":
        value = _toolchain().get("spike_extension_name")
        if not isinstance(value, str) or not value:
            raise ChipyardRoccError(f"{_target()}: runner.toolchain declares no spike_extension_name")
        return value
    if name == "GSIM_MAX_CYCLES":
        value = _toolchain().get("gsim_max_cycles")
        if type(value) is not int or value <= 0:
            raise ChipyardRoccError(f"{_target()}: runner.toolchain declares no positive gsim_max_cycles")
        return value
    if name == "ORACLE":
        return _oracle()
    if name == "EXECUTION_CAPABILITIES":
        return _execution_capabilities()
    if name == "rocc_semantics":
        return _rocc_semantics()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _oracle() -> dict[str, dict[str, Any]]:
    """Per-engine oracle records. ``derived_from_rtl`` is the claim a grade cites; GSIM and Verilator
    both run the elaborated design, the extension-loaded spike does not."""
    extension = __getattr__("SPIKE_EXTENSION_NAME")
    return {
        "spike": {"kind": f"spike_{extension}_functional", "derived_from_rtl": False},
        "verilator": {"kind": "rtl_verilator", "derived_from_rtl": True},
        "gsim": {"kind": "rtl_gsim", "derived_from_rtl": True},
    }


def _execution_capabilities() -> dict[str, str]:
    """Software capabilities the contract declares this runtime's harness path implements."""
    raw = _contract().get("execution_capabilities")
    if raw is None:
        return {}
    if not isinstance(raw, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
        raise ChipyardRoccError(f"{_target()}: execution_capabilities must map a capability to evidence text")
    return dict(raw)


def _rocc_semantics():
    """The MODULE ``plugin.rocc_semantics`` names (absent key: no attribute).

    Served as the module itself: ``rocc.decode`` calls its functions and ``rtl_checks.selected_checks``
    reads ``rocc_semantics.rtl_checks`` from it. A provider that also declares ``plugin.rtl_checks``
    must name the same module the semantics module serves, so the two declarations cannot diverge.
    """
    from merlin.targetgen.plugins import load_declared

    plugin = _plugin()
    if not plugin.get("rocc_semantics"):
        raise AttributeError(f"{_target()}: the selected support declares no plugin.rocc_semantics")
    semantics = load_declared(_target(), "rocc_semantics")
    if plugin.get("rtl_checks"):
        declared = load_declared(_target(), "rtl_checks")
        if getattr(semantics, "rtl_checks", None) is not declared:
            raise ChipyardRoccError(
                f"{_target()}: plugin.rtl_checks names {declared.__name__}, but {semantics.__name__}.rtl_checks "
                "serves a different module; one target cannot have two RTL-check providers"
            )
    return semantics


# --- the harness renderer (generic, owned by contract/harness_render) ---------------------------------
try:
    from merlin.targetgen.contract.harness_render import render_harness  # noqa: F401 -- re-exported
except ModuleNotFoundError as _missing:
    if _missing.name != "merlin.targetgen.contract.harness_render":
        raise
    _RENDER_MISSING = str(_missing)

    def render_harness(cb, *, target, **kwargs):  # type: ignore[no-redef]
        """Placeholder until the generic renderer lands; refuses with the reason rather than guessing."""
        raise NotImplementedError(
            "the generic chipyard RoCC backend renders harnesses with "
            "merlin.targetgen.contract.harness_render.render_harness, which is not installed in this "
            f"checkout ({_RENDER_MISSING}); no target-specific renderer is substituted"
        )


# --- toolchain + engines ------------------------------------------------------------------------------
def chipyard_root() -> Path:
    """The toolchain container checkout, located by the contract's declared variable (or ext key)."""
    from merlin.common.paths import ExternalPathUnset, env, ext_path

    block = _toolchain()
    variable = block.get("root_env")
    if isinstance(variable, str) and variable and env(variable):
        return Path(str(env(variable)))
    key = block.get("root_ext")
    if isinstance(key, str) and key:
        try:
            return ext_path(key)
        except ExternalPathUnset as exc:
            raise ChipyardRoccError(f"{_target()}: neither {variable} nor MERLIN_EXT_{key.upper()} is set") from exc
    raise ChipyardRoccError(f"{_target()}: runner.toolchain.root_env {variable!r} is unset")


def _tool(name: str, *, env_name: str | None) -> Path:
    from merlin.common.paths import env

    entry = _toolchain().get(name)
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"]:
        raise ChipyardRoccError(f"{_target()}: runner.toolchain declares no {name}.path")
    override = entry.get("env") if isinstance(entry.get("env"), str) else env_name
    if override and env(override):
        return Path(str(env(override)))
    relative = entry["path"]
    if "{rtl_sim_config}" in relative:
        relative = relative.replace("{rtl_sim_config}", _rtl_sim_config())
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ChipyardRoccError(f"{_target()}: runner.toolchain.{name}.path must be root-relative")
    return chipyard_root() / relative


def gcc_path() -> Path:
    return _tool("compiler", env_name=None)


def spike_path() -> Path:
    return _tool("spike", env_name=SPIKE_ENV)


def spike_library_dir() -> Path:
    """Where the toolchain's spike extension libraries live (put on ``LD_LIBRARY_PATH``)."""
    relative = _toolchain().get("spike_library_dir")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ChipyardRoccError(f"{_target()}: runner.toolchain declares no root-relative spike_library_dir")
    return chipyard_root() / relative


def spike_extension() -> tuple[tuple[str, ...], Path]:
    """``(spike flags, LD_LIBRARY_PATH dir)`` for this target's functional model.

    A contract ``runner.spike_extension`` (digest-pinned) wins; otherwise the toolchain's library
    directory and the declared extension name, exactly as :mod:`merlin.targetgen.spike_extension`
    defines the additive default.
    """
    from merlin.targetgen.spike_extension import spike_invocation

    return spike_invocation(
        _target(), default_library_dir=spike_library_dir(), default_extension_name=__getattr__("SPIKE_EXTENSION_NAME")
    )


def simulator_identity(simulator: str) -> dict | None:
    """Which build of ``simulator`` runs a program, by content; None for engines it states none for."""

    def digest(path: Path) -> str | None:
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    if simulator != "spike":
        return None
    flags, libdir = spike_extension()
    libraries = {}
    for flag in flags:
        if flag.startswith("--extlib="):
            so = Path(flag.partition("=")[2])
            libraries[so.name] = digest(so)
    if not libraries:
        names = [flag.partition("=")[2] for flag in flags if flag.startswith("--extension=")]
        libraries = {name: digest(Path(libdir) / f"lib{name}.so") for name in names}
    parts = {"binary": digest(spike_path()), "flags": list(flags), "extensions": libraries}
    if parts["binary"] is None or not libraries or None in libraries.values():
        return {**parts, "digest": None}
    return {**parts, "digest": hashlib.sha256(repr(sorted(parts.items())).encode()).hexdigest()}


def platform_dram_base() -> int:
    """The bare-metal load address, derived from the RTL build's memory map for the declared sim."""
    from merlin.targetgen import runtime_build

    return runtime_build.platform_dram_base(_target(), _block("runner").get("sim_via"))


def _rtl_sim_config() -> str:
    from merlin.common.paths import env
    from merlin.targetgen import runtime_build

    if VERILATOR_CONFIG_ENV and env(VERILATOR_CONFIG_ENV):
        return str(env(VERILATOR_CONFIG_ENV))
    config = runtime_build.rtl_sim_config(_target())
    if not config:
        raise ChipyardRoccError(f"{_target()}: the contract declares no runtime.rtl_sim_config")
    return str(config)


def verilator_path() -> Path:
    return _tool("verilator", env_name=VERILATOR_ENV)


def gsim_path() -> Path:
    """The GSIM emulator (env override, else the derived engine home). Existence is not checked."""
    from merlin.targetgen import gsim_emulator

    return gsim_emulator.emulator_path(_target(), env_var=GSIM_EMU_ENV)


def gsim_status() -> tuple[bool, str]:
    """``(available, reason)`` for GSIM, including a refused build receipt's reason."""
    from merlin.targetgen import gsim_emulator

    return gsim_emulator.probe(_target(), env_var=GSIM_EMU_ENV)


def gsim_max_cycles() -> str:
    """The GSIM hang bound as its plusarg value. The per-target variable overrides the contract."""
    from merlin.common.paths import env

    override = (env(GSIM_MAXCYCLES_ENV) or "").strip() if GSIM_MAXCYCLES_ENV else ""
    return override or str(__getattr__("GSIM_MAX_CYCLES"))


def _test_ld() -> Path:
    return isa_headers(verify=False).link_script


def _common_dir() -> Path:
    return _test_ld().parent


def available(simulator: str = "verilator") -> bool:
    """True when the compiler, the declared bare-metal runtime and the engine are present.

    An engine this backend knows but cannot find answers False; an engine it does not know RAISES,
    so an engine-priority probe can tell "absent binary" from "no such engine for this target".
    """
    if simulator not in ENGINES:
        raise ChipyardRoccError(f"unknown simulator {simulator!r} (this backend drives {list(ENGINES)})")
    try:
        base = gcc_path().is_file() and _test_ld().is_file() and _common_dir().is_dir()
        if simulator == "spike":
            return base and spike_path().is_file()
        if simulator == "verilator":
            return base and verilator_path().is_file()
        emu = gsim_path()
        # Executability too: the env override can name a copied-without-mode artifact or the .cpp.
        return base and emu.is_file() and os.access(emu, os.X_OK)
    except ChipyardRoccError:
        return False


# --- pinned runtime selection -------------------------------------------------------------------------
_PINNED_RUNTIME_LOCK = threading.RLock()


def runtime_environment(
    *, binaries: Mapping[str, Path], gsim_max_cycles: int | None, environment: Mapping[str, str]
) -> dict[str, str]:
    """Copy ``environment`` with explicit engine choices; never mutate process state."""
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()
    ):
        raise ValueError("runtime environment must map strings to strings")
    if not isinstance(binaries, Mapping) or set(binaries) != {"gsim", "verilator"}:
        raise ValueError("pinned runtime requires exactly GSIM and Verilator binaries")
    if gsim_max_cycles is not None and (type(gsim_max_cycles) is not int or gsim_max_cycles <= 0):
        raise ValueError("GSIM max cycles must be a positive integer or null")
    _target()
    paths = {engine: Path(path).resolve(strict=True) for engine, path in binaries.items()}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("pinned runtime binaries must be ordinary files")
    result = dict(environment)
    result[GSIM_EMU_ENV] = str(paths["gsim"])
    result[VERILATOR_ENV] = str(paths["verilator"])
    if gsim_max_cycles is None:
        result.pop(GSIM_MAXCYCLES_ENV, None)
    else:
        result[GSIM_MAXCYCLES_ENV] = str(gsim_max_cycles)
    return result


@contextmanager
def pinned_runtime(*, binaries: Mapping[str, Path], gsim_max_cycles: int | None):
    """Apply :func:`runtime_environment` to this process under a lock, restoring only owned keys."""
    keys = (GSIM_EMU_ENV, VERILATOR_ENV, GSIM_MAXCYCLES_ENV)
    with _PINNED_RUNTIME_LOCK:
        configured = runtime_environment(binaries=binaries, gsim_max_cycles=gsim_max_cycles, environment=os.environ)
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                value = configured.get(key)
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            yield sys.modules[__name__]
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


# --- execution ----------------------------------------------------------------------------------------
def _unlimited_stack() -> None:  # pragma: no cover - runs in the child process
    """Lift RLIMIT_STACK for an RTL simulator child (a small stack makes the model warn onto the
    console, corrupting capture)."""
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, OSError):
        pass


def _gsim_argv(elf: str | Path, *, max_cycles: int | None = None) -> list[str]:
    """The Merlin GSIM harness vector: the ELF positionally (symbols) and as ``+loadmem`` (image)."""
    cycles = gsim_max_cycles() if max_cycles is None else str(max_cycles)
    return [str(gsim_path()), str(elf), f"+max-cycles={cycles}", f"+loadmem={elf}"]


def prepare_gsim_command(elf, **kwargs):
    """Describe a pinned GSIM invocation; the caller owns deadline, sandbox and execution."""
    from merlin.runtime.backends._chipyard_gsim_command import prepare_gsim_command as prepare

    return prepare(sys.modules[__name__], elf, **kwargs)


def _spike_argv(elf: str | Path, *extra: str) -> tuple[list[str], dict[str, str]]:
    """The functional-model command for ``elf``: the target's extension, the DRAM span / harts / ISA
    the image states it was laid out for (none stated, none passed), then ``extra`` and the ELF."""
    from merlin.runtime.backends.spike_model import declared_harts, declared_isa, declared_memory

    env = dict(os.environ)
    flags, libdir = spike_extension()
    env["LD_LIBRARY_PATH"] = str(libdir) + ":" + env.get("LD_LIBRARY_PATH", "")
    span = declared_memory(elf)
    memory = [f"-m{hex(span[0])}:{hex(span[1])}"] if span else []
    harts, isa = declared_harts(elf), declared_isa(elf)
    machine = [*([f"-p{harts}"] if harts else []), *([f"--isa={isa}"] if isa else [])]
    return [str(spike_path()), *flags, *machine, *memory, *extra, str(elf)], env


def functional_trace_invocation(elf: str | Path) -> tuple[list[str], dict[str, str]]:
    """``(argv, env)`` running ``elf`` on the functional model with every retired instruction and its
    register writes logged to stderr (``-l --log-commits``) -- the executed-command profile's input."""
    return _spike_argv(elf, "-l", "--log-commits")


def run_elf(
    elf: str | Path,
    simulator: str = "verilator",
    timeout: int = 600,
    *,
    capture_bytes: bool = False,
    memory_readback: Mapping[str, object] | None = None,
) -> str | bytes:
    """Run ``elf`` on ``simulator``; an explicit memory export never changes the default command."""
    preexec = None
    slot = nullcontext()
    export = prepare_memory_readback(elf, simulator, memory_readback) if memory_readback is not None else None
    if simulator == "spike":
        cmd, env = _spike_argv(elf, *(export.argv_suffix if export else ()))
    elif simulator == "verilator":
        env = dict(os.environ)
        # LOADED LIKE GSIM, through the memory backdoor. Loaded over the simulated serial link instead,
        # the same ELF started its kernel window from different cache state (1096 vs gSIM's 1112 cycles on
        # one 16x16 program -- identical once both load by +loadmem) and spent 95% of a small run (103 s
        # against 7.75 s) moving the program in.
        cmd = [str(verilator_path()), str(elf), f"+loadmem={elf}"]
        preexec = _unlimited_stack
    elif simulator == "gsim":
        from merlin.targetgen import rtl_engine_policy

        if getattr(rtl_engine_policy, "GSIM_RUNTIME_SLOT_PROTOCOL", None) != "reentrant_per_thread_v1":
            raise ChipyardRoccError("GSim requires Merlin runtime slot protocol reentrant_per_thread_v1")
        slot = rtl_engine_policy.gsim_runtime_slot(wait_timeout_s=timeout)
        env = dict(os.environ)
        cmd = [*_gsim_argv(elf), *(export.argv_suffix if export else ())]
        preexec = _unlimited_stack
    else:
        raise ChipyardRoccError(f"unknown simulator {simulator!r} (this backend drives {list(ENGINES)})")
    with slot:
        if export is not None:
            export.revalidate()
        proc = subprocess.run(
            cmd, capture_output=True, text=not capture_bytes, timeout=timeout, env=env, preexec_fn=preexec
        )
    # Verilator exits 0 on $finish, spike on htif_exit(0), GSIM when the design's stop condition fires.
    if proc.returncode != 0:
        tail = f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        if simulator == "gsim" and proc.returncode == _GSIM_CYCLE_CAP_EXIT:
            raise SimulatorTimeout(
                f"gsim timed out after its +max-cycles hang bound ({gsim_max_cycles()} cycles; exit "
                f"{proc.returncode}) before the program finished:\n{tail}",
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        raise ChipyardRoccError(
            f"{simulator} exited {proc.returncode}:\n{tail}", stdout=proc.stdout, stderr=proc.stderr
        )
    _refuse_on_rtl_assertion(simulator, proc.stdout, proc.stderr, binary_output=capture_bytes)
    if export is not None:
        export.revalidate(completed=True)
    return proc.stdout


def _refuse_on_rtl_assertion(
    simulator: str, stdout: str | bytes, stderr: str | bytes, *, binary_output: bool = False
) -> None:
    """Fail a run whose DESIGN asserted, whatever the engine did afterwards.

    Verilator turns a failed assertion into ``$stop`` (non-zero exit); the GSIM model prints it and
    keeps going to a clean exit with a complete console. An assertion is the design refusing the
    program, so the numbers would describe a machine that would not have run it.
    """
    if binary_output:
        from merlin.runtime.out_bin import binary_console_diagnostics

        if type(stdout) is not bytes:
            raise ChipyardRoccError("binary readback requires raw stdout bytes")
        stdout = binary_console_diagnostics(stdout)
    for stream in (stdout, stderr):
        if isinstance(stream, bytes):
            stream = stream.decode("utf-8", errors="replace")
        if not stream or _RTL_ASSERTION_MARKER not in stream:
            continue
        rows = [line.rstrip() for line in stream.splitlines()]
        quoted: list[str] = []
        for index, line in enumerate(rows):
            if _RTL_ASSERTION_MARKER not in line:
                continue
            quoted.append(line.strip())
            following = rows[index + 1].strip() if index + 1 < len(rows) else ""
            if following.startswith("at "):
                quoted.append(following)
        raise ChipyardRoccError(
            f"{simulator}: the DESIGN asserted, so this run certifies nothing — the program did "
            "something the hardware does not support:\n  " + "\n  ".join(quoted[:8])
        )


def parse_output(text: str | bytes) -> tuple[dict[str, list], dict[str, int]]:
    """Parse the shared console into ``(outputs, raw metrics)``.

    Bytes are the binary readback frame. Text is ``OUT``/``METRIC``/``DONE`` (Verilator warning
    fragments stripped, malformed METRIC lines tolerated) plus ``OUT_ND <name> <rank> <dims...>
    <values...>`` for a rank-N output whose batch rank a matrix would discard.
    """
    if type(text) is bytes:
        from merlin.runtime.out_bin import parse_binary_console

        return parse_binary_console(text)
    from merlin.runtime.backends.base import _strip_warning_fragments, parse_console

    cleaned = _strip_warning_fragments(text)
    outputs, raw = parse_console(cleaned, error_cls=ChipyardRoccError, tolerant_metric=True)

    def reshape(values: list[int], dimensions: tuple[int, ...]):
        if len(dimensions) == 1:
            return values
        stride = 1
        for extent in dimensions[1:]:
            stride *= extent
        return [reshape(values[index : index + stride], dimensions[1:]) for index in range(0, len(values), stride)]

    for line in cleaned.splitlines():
        parts = line.split()
        if not parts or parts[0] != "OUT_ND":
            continue
        try:
            name = parts[1]
            rank = int(parts[2])
            if rank <= 0:
                raise ValueError("rank must be positive")
            if len(parts) < 3 + rank:
                raise ValueError("dimension list is truncated")
            dimensions = tuple(int(value) for value in parts[3 : 3 + rank])
            if any(extent <= 0 for extent in dimensions):
                raise ValueError("dimensions must be positive")
            values = [int(value) for value in parts[3 + rank :]]
        except (IndexError, ValueError) as exc:
            raise ChipyardRoccError(f"malformed OUT_ND line: {line!r}: {exc}") from exc
        expected = 1
        for extent in dimensions:
            expected *= extent
        if len(values) != expected:
            raise ChipyardRoccError(f"OUT_ND {name}: expected {expected} values, got {len(values)}")
        if name in outputs:
            raise ChipyardRoccError(f"output {name!r} was printed more than once")
        outputs[name] = reshape(values, dimensions)
    return outputs, raw


# --- the runner-owned harness build -------------------------------------------------------------------
def describe_caller_layout(cb: dict, *, target: str, facts: dict) -> dict:
    """The logical harness's physical pointer layout for ``cb``, in argument order; never tensor values.

    The generic harness passes every logical buffer DENSE and row-major (``harness_render``), so the
    projection is the legacy row-major policy with a row alignment of one element. A buffer declaring
    its own ``storage_encodings`` is refused rather than described approximately.
    """
    from merlin.targetgen.contract.harness_render import logical_interface, resolve

    if (
        not isinstance(cb, dict)
        or cb.get("target") != target
        or not isinstance(facts, dict)
        or (facts.get("inputs") or {}).get("target") != target
    ):
        raise ChipyardRoccError("caller layout needs a command buffer and RTL facts for the served target")
    if "storage_encodings" in (cb.get("params") or {}):
        raise ChipyardRoccError("the generic harness describes only dense row-major caller storage")
    rows = []
    for buf in logical_interface(cb, resolve(target)[0]):
        matrix_rows, cols = buf.matrix
        strides = [prod(buf.shape[axis + 1 : -1]) * cols for axis in range(len(buf.shape) - 1)] + [1]
        rows.append(
            {
                "tensor": buf.name,
                "dtype": buf.dtype,
                "logical_shape": list(buf.shape),
                "physical_extents": [matrix_rows, cols],
                "logical_strides_elements": strides if buf.shape else [],
                "storage_elements": buf.elements,
                "offset_elements": 0,
            }
        )
    return {
        "schema": "caller_storage_layout_v1",
        "policy": {"mode": "legacy_aligned_row_major_v1", "row_alignment_elements": 1},
        "tensors": rows,
    }


def caller_layout_source_paths() -> tuple[Path, ...]:
    """The installed core sources whose bytes define :func:`describe_caller_layout`."""
    from merlin.targetgen.contract import harness_render

    return (Path(__file__).resolve(), Path(harness_render.__file__).resolve())


def generated_isa_header() -> Path | None:
    """The minimal hardware-facts ISA header (RTL facts + contract encodings), written once per
    content under ``out/build/isa_headers``; None when the spec declares no generated header."""
    from merlin.targetgen import isa_header_gen

    name = isa_headers(verify=False).generated_header
    if name is None:
        return None
    try:
        return isa_header_gen.materialize(_target(), name)
    except isa_header_gen.IsaHeaderError as exc:
        raise ChipyardRoccError(str(exc)) from exc


def harness_build_recipe():
    """How a runner-owned harness compiles and links against the declared bare-metal environment.

    Every value is data: the compiler from ``runner.toolchain``; include roots, CRT, link script,
    flags and kernel stack budget from the content-verified ISA-headers spec; the entry symbol from
    ``harness_abi``; the load address from the RTL build's memory map. The include path holds the
    bare-metal runtime and the GENERATED facts header only, and the build is refused if any header
    the spec excludes (an upstream kernel library) is reachable from it.
    """
    from merlin.runtime.backends import base as backend_base
    from merlin.targetgen.contract.build_recipe import KernelStackFramePolicy
    from merlin.targetgen.contract.harness_abi import for_target
    from merlin.targetgen.isa_headers_spec import IsaHeadersError

    headers = isa_headers(verify=True)
    generated = generated_isa_header()
    include_roots = headers.include_roots + ((generated.parent,) if generated is not None else ())
    try:
        headers.verify_exclusions(include_roots)
    except IsaHeadersError as exc:
        raise ChipyardRoccError(str(exc)) from exc
    support_sources = headers.crt_sources
    if headers.console_write is not None:
        from merlin.runtime.backends import _chipyard_console

        support_sources += (_chipyard_console.materialize(_target(), headers.console_write),)
    return backend_base.HarnessBuildRecipe(
        compiler=gcc_path(),
        include_roots=include_roots,
        support_sources=support_sources,
        link_script=headers.link_script,
        load_address=platform_dram_base(),
        cflags=headers.cflags,
        ldflags=headers.ldflags,
        error_cls=ChipyardRoccError,
        kernel_stack_frame=KernelStackFramePolicy(
            entry_symbol=for_target(_target()).entry_symbol, max_static_bytes=headers.kernel_max_static_bytes
        ),
        header_dependencies=headers.header_dependencies + ((generated,) if generated is not None else ()),
        link_first=headers.link_first,
    )


# --- readout + counter facts (contract DATA, cross-checked against the header spec) -------------------
def _readout(field: str):
    block = _block("readout_semantics")
    if field not in block:
        raise ChipyardRoccError(f"{_target()}: readout_semantics declares no {field!r}")
    return block[field]


def _readout_provenance() -> dict[str, Any]:
    """The header the readout facts were reviewed against, which must be the spec's pinned bytes."""
    evidence = _readout("evidence_header")
    if not isinstance(evidence, dict) or not evidence.get("file") or not evidence.get("sha256"):
        raise ChipyardRoccError(f"{_target()}: readout_semantics.evidence_header needs file and sha256")
    from merlin.targetgen import isa_headers_spec

    try:
        declared = isa_headers_spec.declared_files(_target(), _isa_headers_spec()).get(str(evidence["file"]))
    except isa_headers_spec.IsaHeadersError as exc:
        raise ChipyardRoccError(str(exc)) from exc
    if declared != evidence["sha256"]:
        raise ChipyardRoccError(
            f"{_target()}: readout facts were reviewed against {evidence['file']} sha256 {evidence['sha256']}, "
            f"but the selected ISA-headers spec pins {declared}; the declared readout no longer describes "
            "the header a harness is built with"
        )
    return {"scope": "contract_declared", "params_header_sha256": evidence["sha256"], "header": evidence["file"]}


def _rows(field: str) -> list[dict[str, Any]]:
    value = _readout(field)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ChipyardRoccError(f"{_target()}: readout_semantics.{field} must be a list of mappings")
    return [dict(row) for row in value]


def readout_epilogue_capability() -> list[dict[str, Any]]:
    """Which epilogue stages each readout selector applies, as the contract declares it."""
    _readout_provenance()
    return _rows("epilogue_capability")


def epilogue_stage_routes() -> list[dict[str, Any]]:
    """Non-readout stage routes (e.g. a bias seeded into the accumulator), as declared."""
    _readout_provenance()
    return _rows("stage_routes")


def _contract_record(field: str) -> dict[str, Any] | None:
    value = _readout(field)
    if value is None:
        return None
    if not isinstance(value, dict) or not value.get("schema"):
        raise ChipyardRoccError(f"{_target()}: readout_semantics.{field} must be a schema-bearing mapping or null")
    return {**value, "provenance": {**dict(value.get("provenance") or {}), **_readout_provenance()}}


def readout_scalar_abi() -> dict[str, Any] | None:
    """The scalar narrowing-readout contract (dtypes, clamp) the declared header commits to."""
    return _contract_record("scalar_abi")


def readout_operand_sum() -> dict[str, Any] | None:
    """What a scaled load does to an operand on its way into the accumulator, or None."""
    return _contract_record("operand_sum")


def counter_engine_kinds() -> dict[str, Any]:
    """Resource kind of each counted engine, as declared (never inferred from a counter's name)."""
    kinds = _block("counter_semantics").get("engine_kinds")
    if not isinstance(kinds, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in kinds.items()):
        raise ChipyardRoccError(f"{_target()}: counter_semantics.engine_kinds must map engine -> kind")
    return dict(kinds)


def counter_partition_inputs() -> dict[str, Any]:
    """The elaborated-artifact boundary for the generic counter occupancy-partition verifier."""
    from merlin.targetgen.rtl import mlc_bridge

    partition = _block("counter_semantics").get("partition")
    if not isinstance(partition, dict) or not partition.get("module") or not partition.get("counter_module"):
        return {"status": "unknown", "why": "counter_semantics.partition declares no module/counter_module"}
    path = mlc_bridge.core_hw_mlir(_target())
    if path is None or not Path(path).is_file():
        return {"status": "unknown", "why": "elaborated CIRCT core HW is unavailable"}
    return {
        "status": "available",
        "hw_text": Path(path).read_text(encoding="utf-8", errors="replace"),
        "module": str(partition["module"]),
        "counter_module": str(partition["counter_module"]),
        "source": str(path),
    }


# --- registration -------------------------------------------------------------------------------------
if TARGET_NAME is not None:
    _CLASS = target_class_for(TARGET_NAME)
    if _CLASS is None:
        raise ChipyardRoccError(
            f"{TARGET_NAME}: its contract declares no compute engine, so its target class is unknown; "
            "refusing to register it under a guessed class"
        )
    register(BackendInfo(TARGET_NAME, TargetClass(_CLASS), BackendKind.KERNEL, __name__))
