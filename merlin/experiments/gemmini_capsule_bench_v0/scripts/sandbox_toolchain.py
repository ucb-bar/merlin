"""Thin delegator onto the SHARED, descriptor-driven sandbox toolchain.

The real logic now lives in :mod:`merlin.targetgen.sandbox` (target-agnostic, routed by compute-unit
kind / sim family — a new target gets a correct sandbox from its ``target_experiment.yaml`` with no
copied scripts). This module resolves THIS experiment's descriptor and re-exports the toolchain surface
the local harness scripts (``run_baseline_qa_loop``, ``test_sandbox``, ``run_rtlchecks_qa_loop``) import,
so their call sites are unchanged. It binds the LEGIT tools back over the /scratch* masks and binds NO
answer surface.
"""
from __future__ import annotations
from pathlib import Path

import _common as C
from merlin.targetgen.sandbox import toolchain as _TC
from merlin.targetgen.target_experiment import load_target_experiment


def _te():
    """This experiment's target descriptor (honors MERLIN_TARGET_EXPERIMENT via C.EXP)."""
    return load_target_experiment(C.EXP / "target_experiment.yaml")


# Re-exported constants some local scripts / the readiness check reference by name.
CHIPYARD_VERILATOR = _TC.SIM_TOOLCHAINS["chipyard"].bind_paths[1]
MERLIN_CLANG = _TC.MERLIN_CLANG
MEMORY_DIR = str(__import__("merlin.targetgen.sandbox.answer_surfaces", fromlist=["x"])
                 .experimenter_memory_dir())
NESTED_SESSION_VARS = list(_TC.NESTED_SESSION_VARS)
CURATED_HARNESS = _TC.curated_harness_dir(_te())


def toolchain_binds() -> list[str]:
    """bwrap args binding the legit toolchain back over the /scratch* masks (universal + this target's
    sim family + curated harness). Append AFTER the base argv + claude runtime binds."""
    return _TC.toolchain_binds(_te())


def sandbox_env(ws: Path) -> str:
    """Shell ``export``s prepended to the in-sandbox command (PATH/LD/PYTHONPATH/harness), derived from
    the descriptor's sim family — PYTHONPATH points at the WORKSPACE's curated merlin pkg."""
    return _TC.sandbox_env(_te(), ws)
