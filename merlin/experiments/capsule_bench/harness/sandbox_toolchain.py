"""Thin delegator onto the SHARED, descriptor-driven sandbox toolchain.

The real logic now lives in :mod:`merlin.targetgen.sandbox` (target-agnostic, routed by compute-unit
kind / sim family — a new target gets a correct sandbox from its ``target_experiment.yaml`` with no
copied scripts). This module resolves THIS experiment's descriptor and re-exports the toolchain surface
the local harness scripts (``run_baseline_qa_loop``, ``preflight_sandbox``, ``run_rtlchecks_qa_loop``) import,
so their call sites are unchanged. It binds the LEGIT tools back over the /scratch* masks and binds NO
answer surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from merlin_experiments.phase1.context import InvocationContext


def _te(context: InvocationContext | None = None):
    """Resolve an explicit invocation, or initialize the native compatibility edge on use."""
    if context is None:
        import _common as C

        descriptor = C.DESCRIPTOR
    else:
        descriptor = context.descriptor
    from merlin.targetgen.target_experiment import load_target_experiment

    return load_target_experiment(descriptor)


def _toolchain(context: InvocationContext | None = None):
    # The shared module captures tooling configuration at import: initialize invocation first.
    if context is None:
        import _common  # noqa: F401

    from merlin.targetgen.sandbox import toolchain

    return toolchain


def __getattr__(name: str):
    """Retain legacy constants without selecting a target merely by importing the adapter."""
    if name not in {"CHIPYARD_VERILATOR", "MERLIN_CLANG", "MEMORY_DIR", "NESTED_SESSION_VARS", "CURATED_HARNESS"}:
        raise AttributeError(name)
    toolchain = _toolchain()
    if name == "CHIPYARD_VERILATOR":
        value = toolchain.SIM_TOOLCHAINS["chipyard"].bind_paths[1]
    elif name == "MERLIN_CLANG":
        value = toolchain.MERLIN_CLANG
    elif name == "MEMORY_DIR":
        from merlin.targetgen.sandbox.answer_surfaces import experimenter_memory_dir

        value = str(experimenter_memory_dir())
    elif name == "NESTED_SESSION_VARS":
        value = list(toolchain.NESTED_SESSION_VARS)
    else:
        value = toolchain.curated_harness_dir(_te())
    globals()[name] = value
    return value


def toolchain_binds(*, context: InvocationContext | None = None) -> list[str]:
    """bwrap args binding the legit toolchain back over the /scratch* masks (universal + this target's
    sim family + curated harness). Append AFTER the base argv + claude runtime binds."""
    target = _te() if context is None else _te(context)
    return _toolchain(context).toolchain_binds(target)


def sandbox_env(ws: Path, *, context: InvocationContext | None = None) -> str:
    """Shell ``export``s prepended to the in-sandbox command (PATH/LD/PYTHONPATH/harness), derived from
    the descriptor's sim family — PYTHONPATH points at the WORKSPACE's curated merlin pkg."""
    target = _te() if context is None else _te(context)
    return _toolchain(context).sandbox_env(target, ws)


def curated_harness(*, context: InvocationContext) -> str:
    """Explicit replacement for the native CURATED_HARNESS compatibility constant."""
    return _toolchain(context).curated_harness_dir(_te(context))
