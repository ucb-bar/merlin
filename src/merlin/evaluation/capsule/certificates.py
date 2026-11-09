"""Capsule evaluation: certificates."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from merlin.evaluation.capsule.tier import TierResult as TierResult
from merlin.experiments.phase1.feedback import setup_timing as _host_timing
from merlin.runtime.backends import base as _backends_mod
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import completion_order_gate as _CO_GATE
from merlin.targetgen import seeded_grade as _seeded_grade
from merlin.targetgen import tier_cache as _TC
from merlin.targetgen.elf_lanes import PROGRAM_ARTIFACT_SUFFIX

_PREPARED_EXECUTION_OBSERVER = ContextVar("capsule_prepared_execution_observer", default=None)


class _CarriedPreparedTier(Exception):
    def __init__(self, result, built=False):
        self.result, self.built = result, built  # built: this run's build keyed it (a seeded screen)


@contextmanager
def _prepared_tier_reuse(
    capsule_name, tier, mandatory, *, target, generated, shas, from_rtl, simulation_context, first=False
):
    """Look up an exact semantic certificate after this engine builds its ELF.

    A seeded grade (:mod:`.seeded_grade`) also looks up its ``first`` executed tier, the screen."""
    screen = first and _seeded_grade.screen_carry(capsule_name, tier)

    def observe(elf):
        from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME

        if Path(elf).resolve() != (Path(generated) / PACKAGE_ELF_NAME).resolve():
            return
        result, _ = carried_tier_result(
            capsule_name,
            tier,
            mandatory,
            target=target,
            generated=generated,
            shas=shas(),
            from_rtl=from_rtl,
            simulation_context=simulation_context,
        )
        if result is not None:
            raise _CarriedPreparedTier(result, built=screen)

    token = _PREPARED_EXECUTION_OBSERVER.set(observe if simulation_context is not None or screen else None)
    try:
        yield
    finally:
        _PREPARED_EXECUTION_OBSERVER.reset(token)


@_host_timing.measured("order_after_certificate")
def _order_after_certificate(
    target, cfg, tiers, trace, generated, *, gold, policy, gsource, trigger=None, readback_outputs=()
):
    """The opt-in memory-order tier (:mod:`merlin.targetgen.load_order`), judged by this capsule's golden.

    A reordered run is held to exactly the standard the certificate was: the capsule's own comparison
    under its own policy, read through the target backend's console reader when it has one."""
    from merlin.targetgen import elf_lanes as _EL
    from merlin.targetgen import load_order as _LO

    def _parse(text):
        try:
            reader = getattr(_backends_mod.get_backend(target), "parse_output", None)
        except KeyError:
            reader = None
        if callable(reader):
            return reader(text)[0]
        return _backends_mod.parse_console(text, strip_warnings=True, tolerant_metric=True)[0]

    def _compare(outputs):
        return (
            CG.compare_declared_outputs(
                gold, outputs, policy, golden_source=gsource, readback_outputs=readback_outputs
            )["status"]
            == "pass"
        )

    return _LO.after_certificate(
        target=target,
        static=_CO_GATE.sweep_trigger(trigger),
        tiers=tiers,
        rtl_tiers=cfg.rtl_tiers,
        trace=trace,
        elf=Path(generated) / _EL.PACKAGE_ELF_NAME,
        judge=_LO.console_judge(_compare, _parse),
    )


def _clear_stale_executable(generated) -> None:
    """Remove an executable left in this run directory by an EARLIER grade.

    Run directories are reused (a shape-keyed model dir, a re-graded runs root). Every consumer that
    content-addresses "the program this capsule ran" -- the tier cache here, `tier_promote`'s execution
    digest, `oot_runner`'s artifact identity -- reads that file, and a leftover from a previous grade
    answers all of them with the wrong program. Removing it costs nothing: the adapter that needs one
    rebuilds it, deterministically, from this run's own lowering.
    """
    from merlin.targetgen import build_cache as _BC
    from merlin.targetgen import elf_lanes as _EL

    try:
        (Path(generated) / _EL.PACKAGE_ELF_NAME).unlink()
    except OSError:  # absent (the normal case) or not removable: nothing to reuse
        pass
    # The build cache's key marker asserts "this directory holds the build for that key". A directory
    # whose executable has just been deleted asserts nothing, so the marker goes with it -- otherwise a
    # reused run directory could answer THIS grade's first compile out of a PREVIOUS grade's leftovers
    # without the store ever being consulted. Rebuilding from the store is still free when the bytes
    # match; what must not survive is the claim.
    _BC.forget(generated)


def run_executables(generated) -> tuple:
    """Everything this run EXECUTED in ``generated``, as paths, sorted by name.

    DERIVED FROM THE BYTES, NOT FROM A NAME. The operator build path emits one ELF under a known name,
    and the certificate identity used to assume it -- which silently excluded every target whose grade
    links a differently-named executable, or more than one. Measured: one target's grade leaves
    ``kernel.radiance.elf``, ``kernel.radiance.layout.elf`` and ``kernel.soc.elf``, so the assumed name
    never existed and the cache could not fire for it at all, indistinguishably from having nothing to
    carry.

    Two kinds count, and the second is why this is not called ``elf_files``. An ELF identifies itself
    -- the four magic bytes are the format's own, not any target's fact, so asking each file what it is
    needs no per-target table. A target that executes no ELF instead DECLARES what it ran, by writing
    those bytes under :data:`PROGRAM_ARTIFACT_SUFFIX`; nothing is inferred on its behalf, because
    guessing which of a directory's files was the program is the one mistake a certificate identity
    may not make.

    The whole directory is walked rather than short-circuiting on the known name: a target may produce
    both, and listing fifteen files costs nothing next to being wrong about what ran.
    """
    from merlin.targetgen import elf_lanes as _EL

    root = Path(generated)
    found = []
    try:
        for path in sorted(root.iterdir()):
            try:
                if not path.is_file():
                    continue
                if path.name == _EL.PACKAGE_ELF_NAME or path.name.endswith(PROGRAM_ARTIFACT_SUFFIX):
                    found.append(path)
                elif path.open("rb").read(4) == b"\x7fELF":
                    found.append(path)
            except OSError:
                continue
    except OSError:
        return ()
    return tuple(found)


def _tier_certificate_key(
    capsule_name: str, tier: str, *, target, generated, shas, from_rtl: bool, simulation_context=None
):
    """``(execution identity, instrument digest, refusal reason)`` for one (capsule, tier).

    The identity is ``None`` whenever either half cannot be established -- no executable yet, a
    hardware pin nobody could resolve, a grading-path file missing, an RTL tier whose engine is
    undecidable. Every one of those makes the tier re-run, and every one now carries a REASON: they
    were all spelled ``None``, which a caller can only render as "nothing was carried", and that is
    how a cache which had never once fired for two of three targets went unnoticed.
    """
    executables = run_executables(generated)
    if simulation_context is not None:
        from merlin.targetgen.simulation_certificate import key

        identity, instrument = key(simulation_context, executables, expected=(capsule_name, tier, target))
        return identity, instrument, "" if identity else "original simulation request identity is unavailable"
    identity = _TC.execution_identity(target=target, executables=executables, toolchain_shas=shas)
    if identity is None:
        return None, None, _TC.execution_identity_reason(target=target, executables=executables, toolchain_shas=shas)
    instrument = _TC.instrument_digest(target, tier, rtl_tier=from_rtl)
    if instrument is None:
        return (
            None,
            None,
            (
                f"the instrument for tier {tier} could not be established (the grading "
                f"path, or the elaborated-RTL engine that would answer it today)"
            ),
        )
    return identity, instrument, ""


def stamp_cache_refusals(tiers: dict, refusals: dict) -> None:
    """Move a ladder's cache refusals onto the tier records they belong to.

    DEFERRED, not written at refusal time: a tier whose certificate could not be looked up goes on to
    EXECUTE, and that execution replaces its record -- so stamping when the refusal happens is undone
    by the very verdict that proves the tier ran.

    Module-level rather than a closure over the ladder because a closure cannot be tested, and two
    mutations that disabled this entirely went unnoticed for precisely that reason.
    """
    for tier, why in (refusals or {}).items():
        record = (tiers or {}).get(tier)
        if record is None or not why:
            continue
        if not getattr(record, "cache_unavailable", ""):
            record.cache_unavailable = str(why)


def carried_tier_result(
    capsule_name: str, tier: str, mandatory: bool, *, target, generated, shas, from_rtl: bool, simulation_context=None
):
    """A verdict this run may CARRY for ``tier``, as a TierResult, or ``None`` to execute the tier.

    The returned record is the stored one with the act of measurement stripped out: no wall-clock
    ``timing``, no ``concurrency`` (nothing was measured now, and copying a duration forward would be a
    fabricated measurement), and no ``evidence``/console pointers (those name files in the run that
    earned it, not in this one -- they move into the ``carried`` block, where a reader can see where
    they live instead of dereferencing a path that is not here). ``cycles`` are kept: a cycle count is a
    property of the program and the device, which is exactly what the key pins, and it is
    concurrency-invariant where a wall time is not.

    ``mandatory`` is TODAY's, never the stored one. Whether a tier is required is a property of the
    capsule being graded now.

    Returns ``(result_or_None, refusal_reason)``. An empty reason means the question was ASKED and
    missed, which is ordinary; a non-empty one means it could not be asked at all, which is the state
    that used to be invisible.
    """
    identity, instrument, why = _tier_certificate_key(
        capsule_name,
        tier,
        target=target,
        generated=generated,
        shas=shas,
        from_rtl=from_rtl,
        simulation_context=simulation_context,
    )
    if identity is None:
        return None, why
    simulated = simulation_context is not None
    if simulated:
        from merlin.targetgen import simulation_certificate as _SC

        root = _SC.cache_root()
        hit = None if root is None else _TC.lookup(capsule_name, tier, identity, instrument, root=root, ledgers=False)
        if hit is not None and _SC.key(
            simulation_context, run_executables(generated), expected=(capsule_name, tier, target)
        ) != (identity, instrument):
            return None, "simulation certificate identities changed before publication"
    else:
        hit = _TC.lookup(capsule_name, tier, identity, instrument)
    if hit is None:
        return None, ""  # asked and missed: the ordinary case, and not a defect
    stored = dict(hit.get("tier_result") or {})
    carried = _TC.carried_block(hit)
    if simulated:
        carried.update(_SC.scope())
    carried["earned_evidence"] = {k: stored.pop(k) for k in _TC.ARTIFACTS_OF_THE_EARNING_RUN if k in stored} or None
    names = {f.name for f in dataclasses.fields(TierResult)}
    # `submission` is deliberately NOT carried: `_finalize_capsule_result` stamps THIS run's submission
    # on any record that lacks one, and a cycle count earned by a byte-identical executable is a true
    # statement about both submissions. Which package physically earned it stays in the carried block.
    fixed = {"tier", "status", "mandatory", "carried", "timing", "concurrency", "submission"}
    kwargs = {k: v for k, v in stored.items() if k in names and k not in fixed}
    kwargs["derived_from_rtl"] = bool(stored.get("derived_from_rtl", from_rtl))
    stored_reason = kwargs.pop("reason", None)
    return TierResult(
        tier,
        str(hit.get("status")),
        mandatory,
        reason=(
            "verdict carried: this tier was NOT executed in this run; the same "
            "executable was already certified at this tier on this instrument"
            + (f" ({stored_reason})" if stored_reason else "")
        ),
        carried=carried,
        **kwargs,
    ), ""


@_host_timing.measured("tier_certificate")
def record_tier_certificate(
    capsule_name: str,
    tier: str,
    result: TierResult,
    *,
    target,
    generated,
    shas,
    from_rtl: bool,
    run_id: str | None = None,
    simulation_context=None,
) -> None:
    """Store an EXECUTED tier verdict so a later grade need not re-buy it. Never raises.

    Recording is best-effort by construction: a cache that can fail a grade is worse than no cache. It
    is also silent about a miss on purpose -- there is nothing to say when the identity cannot be
    established, because the only consequence is that the tier runs again.
    """
    try:
        identity, instrument, _why = _tier_certificate_key(
            capsule_name,
            tier,
            target=target,
            generated=generated,
            shas=shas,
            from_rtl=from_rtl,
            simulation_context=simulation_context,
        )
        if identity is None:
            return
        arguments = {}
        if simulation_context is not None:
            from merlin.targetgen.simulation_certificate import cache_root

            root = cache_root()
            if root is None:
                return
            arguments["root"] = root
        _TC.record(
            capsule_name,
            tier,
            identity,
            instrument,
            status=result.status,
            tier_result=result.to_dict(),
            run_id=run_id,
            **arguments,
        )
    except Exception:  # noqa: BLE001 -- a cache write may never gate a grade
        return


def suppressed_tier_result(tier: str, mandatory: bool, failed_tier: str, *, from_rtl: bool = False):
    """The TierResult for a tier the ladder did NOT run because a mandatory tier already failed.

    ``skipped``, never ``fail``. ``not_run_is_not_pass`` treats a recorded ``fail`` as evidence the capsule
    was certified at that tier and found wrong; a tier that never executed has no such evidence, and
    fabricating one would put a cycle-accurate verdict on a capsule no RTL ever saw.
    """
    return TierResult(
        tier,
        "skipped",
        mandatory,
        reason=(
            f"not run: mandatory tier {failed_tier} already failed, so this deeper tier "
            f"could not change the verdict and was not worth its cost"
        ),
        derived_from_rtl=from_rtl,
    )
