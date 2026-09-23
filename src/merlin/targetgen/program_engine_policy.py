"""Read-only program-engine discovery and lineage checks, independent of evaluation.

The execution oracle reexports these exact objects. Already-loaded legacy overrides
remain effective without importing the oracle to inspect an engine's availability.
"""

from __future__ import annotations

import sys
from pathlib import Path

from merlin.common.paths import ext_path  # noqa: F401 — legacy/context resolver export

from .program_values import OracleUnavailable


def _context():
    return sys.modules.get("merlin.targetgen.program_oracle", sys.modules[__name__])


def _model_venv_python(model_ext: str) -> Path:
    """Resolve the model project's own interpreter through the external-project registry."""
    root = _context().ext_path(model_ext)
    py = root / ".venv" / "bin" / "python"
    if not py.is_file():
        raise _context().OracleUnavailable(f"model venv python absent: {py} (run `uv sync` in {root})")
    return py


_DEFAULT_MODEL_PYTHON = _model_venv_python


def selected_model_venv_python(model_ext: str) -> Path:
    """Honor an already-loaded execution module's resolver override without importing it."""
    legacy = sys.modules.get("merlin.targetgen.program_oracle")
    resolver = getattr(legacy, "_model_venv_python", _DEFAULT_MODEL_PYTHON)
    if resolver is not _DEFAULT_MODEL_PYTHON:
        return resolver(model_ext)
    return _model_venv_python(model_ext)


#: Elaborated-RTL engines, each as (ext_path key suffix, conventional wrapper filename). Every engine
#: exposes the SAME ``run_program``; merlin never names a binary or a target. Adding an engine is one row
#: here plus a wrapper beside its build — the priority among them lives in ``rtl_engine_policy``.
#: The engine whose home layout -- and therefore whose build receipt / adoption record --
#: :mod:`gsim_emulator` owns. Named here because the lineage question is only answerable for the homes
#: that module lays out; every other engine is registered and judged by whoever built it.
_LINEAGE_ENGINE = "gsim"

#: The flavour this program-driven path can actually import and call. The other flavour
#: (a standalone binary taking a linked ELF) is a real build of the same engine that this
#: particular route cannot drive -- a distinction the probe must state rather than flatten.
_WRAPPER_FLAVOUR = "wrapper"

_RTL_ENGINES: dict[str, tuple[str, str]] = {
    "vcs": ("vcs", "vcs_run.py"),
    "gsim": ("gsim", "gsim_run.py"),
    "verilator": ("vsim", "verilator_run.py"),
}


def _rtl_engine_dir(target: str, engine: str) -> Path | None:
    """Where ``engine``'s build for ``target`` lives, or None when nothing registers one.

    Two sources, in precedence order:

    1. ``MERLIN_EXT_<TARGET>_<SUFFIX>`` (process env or ``.env``) — the machine-specific registration.
    2. The DERIVED home ``out/build/rtl_engines/<target>/<engine>/``.

    (2) is new and it is the point. Registration by env var alone means an engine that IS BUILT AND
    WORKING on this machine is invisible to the policy until somebody adds a line to a gitignored file
    — which is exactly what happened: a cycle-exact, 32x-faster GSIM engine sat beside its conventional
    wrapper, fully built, and every cert ran on Verilator because no ``MERLIN_EXT_<TARGET>_GSIM`` line
    existed. Nothing reported that; the engine was simply never considered. A derived home makes
    "install it where it belongs" the way to register an engine, and the env var the exception.
    """
    suffix, _ = _context()._RTL_ENGINES[engine]
    try:
        return Path(_context().ext_path(f"{target}_{suffix}"))
    except KeyError:
        pass
    from .gsim_emulator import engine_home

    derived = engine_home(target, engine)
    return derived if derived.is_dir() else None


def _rtl_engine_probe(target: str, engine: str):
    """``() -> (available, reason)`` for one engine: its dir must be registered AND hold its wrapper."""

    def probe():
        from .gsim_emulator import engine_home

        _, fname = _context()._RTL_ENGINES[engine]
        d = _context()._rtl_engine_dir(target, engine)
        if d is None:
            # BOTH places, named. "not registered" alone sent every reader to the env var and none of
            # them to the directory an engine can simply be installed into.
            return False, (
                f"no MERLIN_EXT_{target.upper()}_{_context()._RTL_ENGINES[engine][0].upper()} "
                f"registered and no build at {engine_home(target, engine)}"
            )
        w = d / fname
        if not w.is_file():
            # PRESENCE IS THE HOME-LAYOUT MODULE'S FACT TOO, not just lineage. Statting one filename
            # made a home holding a BUILT engine of the other flavour report the engine as ABSENT --
            # the same two-modules-disagree defect as the blind probe and the bypassed lineage gate,
            # in a third place. The answer stays False, because it must: this path calls
            # ``run_program(words, preload, reads)`` on an imported wrapper, and the binary flavour
            # takes a LINKED ELF instead (``<emu> <elf> +loadmem=<elf> +max-cycles=N``). Assembling an
            # ELF out of program words is the compiler's job upstream, not a probe's, so a home
            # without the wrapper genuinely cannot be driven from here and passing it would trade a
            # clean unavailable for a crash inside the runner import.
            #
            # What changes is the REASON. "absent" sent every reader looking for a missing build;
            # naming the flavour that IS there sends them to the right question -- which is whether
            # this target's own backend drives that engine, as it does for the binary flavour today.
            if engine == _LINEAGE_ENGINE:
                from .gsim_emulator import resolve as _built

                r = _built(target)
                if r.ok and r.flavour and r.flavour != _WRAPPER_FLAVOUR:
                    return False, (
                        f"{engine} IS built for {target}, as the {r.flavour} flavour at {r.path} -- "
                        f"but NOT as {fname}. This program-driven path imports {fname} and calls "
                        f"run_program(words, preload, reads); the {r.flavour} flavour is driven with a "
                        f"linked ELF instead, so it cannot be run from assembled program words here. "
                        f"The engine is not missing: the target's own backend runs it."
                    )
            return False, f"{fname} absent under {d}"
        # FINDING THE WRAPPER PROVES AN ENGINE IS BUILT, NOT THAT ITS BYTES MAY CERTIFY. The home-layout
        # module owns the lineage question for the homes it lays out, and asking it here is what stops
        # two modules disagreeing about one fact with the OPTIMISTIC one having the last word.
        #
        # Measured 2026-09-04: with MERLIN_GSIM_REQUIRE_RECEIPT=1 set, a target whose engine carries only
        # an adoption record -- lineage `adopted`, never built-and-bound -- had gsim_emulator.probe()
        # answer False and STILL certified on it, because this probe asked only whether a file existed.
        # A provenance gate that the selection path routes around is not a gate.
        #
        # Only asked of the engine whose home this module describes, and only when the dir IS that
        # derived home: an engine registered elsewhere, or laid out by someone else, is not this
        # module's to judge, and a refusal it did not author would be worse than none.
        if engine == _LINEAGE_ENGINE:
            from .gsim_emulator import resolve as _resolve

            r = _resolve(target)
            if r.refused:
                return False, r.reason
        return True, f"{fname} at {d}"

    return probe


def select_rtl_engine(target: str) -> dict:
    """Pick this target's elaborated-RTL engine (see :mod:`rtl_engine_policy`). Raises
    :class:`OracleUnavailable` when none can run, so the tier reports unavailable rather than silently
    resolving to a lower-fidelity oracle."""
    from . import rtl_engine_policy as _pol

    try:
        return _pol.select(target, {e: _context()._rtl_engine_probe(target, e) for e in _context()._RTL_ENGINES})
    except _pol.NoEngineAvailable as exc:
        raise OracleUnavailable(str(exc)) from exc
