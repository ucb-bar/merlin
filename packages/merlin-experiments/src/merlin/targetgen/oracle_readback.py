"""How a capsule-grading simulator adapter reads a program's outputs back.

An explicit policy is honoured as given. Without one, a serial-console engine uses the fastest exact
transport for the output (:func:`readback_policy.engine_readback`: a coherent memory dump or the binary
frame). With ``MERLIN_GSIM_L3_READBACK=digest`` -- set by the Phase 2 functional regrade, not by
default -- a gSIM run instead reads back one digest per output, tied to full values read on Spike from
the same program (:mod:`merlin_experiments.phase2.gsim_digest_readback`). At gSIM speed, an export of a
few hundred KB costs hours; a digest costs about 3.8 cycles per output byte. Any disagreement falls back
to the full readback.

Every completed run also retains the exact program it executed (:mod:`merlin.targetgen.executed_program`):
tiers that read outputs differently link different callers around one kernel object into the same build
directory, so without the copy only the last tier's ELF would survive the grade.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

GSIM_L3_READBACK_ENV = "MERLIN_GSIM_L3_READBACK"


def _digest_for_gsim() -> bool:
    mode = os.environ.get(GSIM_L3_READBACK_ENV, "").strip().lower()
    if mode not in ("", "auto", "digest"):
        raise ValueError(f"{GSIM_L3_READBACK_ENV} must be 'digest' or unset/'auto', got {mode!r}")
    return mode == "digest"


def _run(cb, llvm_text, *, sim, target, backend, workdir, timeout, policy, before_execution=None):
    from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend
    from merlin.targetgen.contract import compile as oot_compile
    from merlin.targetgen.contract.readback_policy import MEMORY_TRANSPORTS, engine_readback

    if policy is None and type(backend) is BoundRoCCBackend:
        policy = backend.engine_readback_policy(sim, cb=cb)
    if policy is None:  # a serial console: the fastest exact readback, not text
        cb, policy = engine_readback(cb, sim, target=target, backend=backend)
    kwargs: dict[str, Any] = {"readback_policy": policy} if policy is not None else {}
    if before_execution is not None:
        kwargs["before_execution"] = before_execution
    if policy is not None and policy.transport in MEMORY_TRANSPORTS:
        from merlin_experiments.phase1.feedback.native_memory_readback import (
            NativeLogicalMemoryReadback,
            NativeMemoryReadback,
            select_memory_engine,
        )

        from merlin.targetgen.rtl.facts import rtl_facts_path

        facts_path = rtl_facts_path(target)
        citation, revalidate = select_memory_engine(
            target=target, simulator=sim, backend=backend, facts_path=facts_path
        )

        def revalidate_memory_engine():
            revalidate()
            return citation

        reader = NativeLogicalMemoryReadback if type(backend) is BoundRoCCBackend else NativeMemoryReadback
        kwargs["memory_readback"] = reader(facts_path=facts_path, policy=policy)
        kwargs["oracle_revalidate"] = revalidate_memory_engine
    from . import executed_program

    res = oot_compile.run_on_oracle(
        cb, llvm_text, simulator=sim, target=target, workdir=workdir, timeout=timeout, **kwargs
    )
    return executed_program.retain(
        res,
        sim=sim,
        workdir=workdir,
        compiler_artifacts={"command_buffer": cb, "lowered_mlir": llvm_text},
    )


def run_with_readback(
    cb,
    llvm_text,
    *,
    sim: str,
    target: str,
    backend: Any,
    workdir: str | Path,
    timeout: int,
    policy,
    before_execution=None,
):
    """One oracle run of ``cb`` on ``sim`` with the readback selected as the module docstring states.

    Under a self-check certificate probe a probed engine is refused before dispatch
    (:func:`merlin.targetgen.simulation_certificate.probe_observer`).
    """
    from .simulation_certificate import probe_observer

    before_execution = probe_observer(sim, before_execution)
    if sim != "gsim" or policy is not None or not _digest_for_gsim():
        return _run(
            cb,
            llvm_text,
            sim=sim,
            target=target,
            backend=backend,
            workdir=workdir,
            timeout=timeout,
            policy=policy,
            **({"before_execution": before_execution} if before_execution is not None else {}),
        )
    if before_execution is not None:
        raise ValueError("prepared verdict reuse is unavailable for the separate digest corroboration path")
    from merlin_experiments.phase2 import gsim_digest_readback as DIGEST

    def oracle(cb_, llvm_, *, simulator, target, workdir, timeout, readback_policy=None):
        return _run(
            cb_,
            llvm_,
            sim=simulator,
            target=target,
            backend=backend,
            workdir=workdir,
            timeout=timeout,
            policy=readback_policy,
        )

    return DIGEST.run_gsim(cb, llvm_text, target=target, workdir=workdir, timeout=timeout, oracle=oracle)
