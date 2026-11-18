"""Capsule evaluation: certificates."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from merlin.development.backend.feedback import setup_timing as _host_timing
from merlin.development.targetgen import capsule_golden as CG
from merlin.development.targetgen import completion_order_gate as _CO_GATE
from merlin.development.targetgen import seeded_grade as _seeded_grade
from merlin.development.targetgen import tier_cache as _TC
from merlin.development.targetgen.elf_lanes import PROGRAM_ARTIFACT_SUFFIX
from merlin.evaluation.capsule.evidence.tiers import TierResult as TierResult
from merlin.runtime.backends import base as _backends_mod
from collections.abc import Callable
from merlin.development.targetgen import oracle_policy as _oracle_policy
from merlin.development.targetgen.oracle_policy import (
    _SIM_ORACLES,
    _bespoke_sim_via,
    _endpoint_of,
    _ensure_sim_oracles_discovered,
)


# ---- certificates ------------------------------------------------------------------------------

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
        from merlin.development.targetgen.elf_lanes import PACKAGE_ELF_NAME

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
    """The opt-in memory-order tier (:mod:`merlin.development.targetgen.load_order`), judged by this capsule's golden.

    A reordered run is held to exactly the standard the certificate was: the capsule's own comparison
    under its own policy, read through the target backend's console reader when it has one."""
    from merlin.development.targetgen import elf_lanes as _EL
    from merlin.development.targetgen import load_order as _LO

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
    from merlin.development.targetgen import build_cache as _BC
    from merlin.development.targetgen import elf_lanes as _EL

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
    from merlin.development.targetgen import elf_lanes as _EL

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
        from merlin.development.targetgen.simulation_certificate import key

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
        from merlin.development.targetgen import simulation_certificate as _SC

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
            from merlin.development.targetgen.simulation_certificate import cache_root

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


# ---- adapters ----------------------------------------------------------------------------------
# Capsule evaluation: adapters.


class OracleUnavailable(Exception):
    pass


def simulator_adapter(sim: str, target: str, selection: dict | None = None, *, readback_policy=None) -> Callable:
    """Run one declared engine, retaining its selected identity and policy record."""

    from merlin.development.targetgen.contract.readback_policy import MEMORY_TRANSPORTS, selected

    readback_policy = selected(readback_policy)
    if readback_policy is not None and readback_policy.transport in MEMORY_TRANSPORTS:
        from merlin.development.targetgen.oracle_readback import validate_engine_policy
        from merlin.runtime.backends import base as backends

        validate_engine_policy(readback_policy, sim=sim, backend=backends.get_backend(target))

    def run(cb, llvm_text, workdir, timeout):
        from merlin.runtime.backends import base as _backends

        backend = _backends.get_backend(target)
        if not backend.available(sim):
            raise OracleUnavailable(f"{sim} not available")
        if sim == "gsim":
            from merlin.development.targetgen import gsim_emulator

            exact, reason = gsim_emulator.selected_firrtl_status(
                target, env_var=getattr(backend, "GSIM_EMU_ENV", None), backend=backend
            )
            if not exact:
                raise OracleUnavailable(reason)
        from merlin.development.targetgen.oracle_readback import run_with_readback

        res = run_with_readback(
            cb,
            llvm_text,
            sim=sim,
            target=target,
            backend=backend,
            workdir=workdir,
            timeout=timeout,
            policy=readback_policy,
            **(
                {"before_execution": _PREPARED_EXECUTION_OBSERVER.get()}
                if _PREPARED_EXECUTION_OBSERVER.get() is not None
                else {}
            ),
        )
        if selection and isinstance(res.get("oracle"), dict):
            res["oracle"]["selection"] = dict(selection)
        return res

    run._merlin_simulator_engine = sim
    run._merlin_readback_policy = readback_policy
    run._merlin_with_readback_policy = lambda policy: simulator_adapter(
        sim, target, selection=selection, readback_policy=policy
    )
    from merlin.development.backend.feedback.operator_child import register_adapter

    register_adapter(run, engine=sim, target=target, selection=selection, readback_policy=readback_policy)
    return run


def _with_readback_policy(adapters: dict[str, Callable], readback_policy) -> dict[str, Callable]:
    """Retarget only declared native simulator adapters; never silently ignore policy."""
    from merlin.development.targetgen.contract.readback_policy import selected

    policy = selected(readback_policy)
    if policy is None:
        return adapters
    if not adapters:
        raise NotImplementedError("no oracle can consume explicit full-value readback policy")
    bound = {}
    for tier, adapter in adapters.items():
        factory = getattr(adapter, "_merlin_with_readback_policy", None)
        if not callable(factory):
            raise NotImplementedError(f"{tier} oracle cannot consume explicit full-value readback policy")
        bound[tier] = factory(policy)
    return bound


def _adapter_readback_policy(adapters: dict[str, Callable]):
    """A model child may inherit only one unanimous, constructed adapter choice."""
    from merlin.development.targetgen.contract.readback_policy import selected

    policies = [selected(getattr(adapter, "_merlin_readback_policy", None)) for adapter in adapters.values()]
    chosen = next((policy for policy in policies if policy is not None), None)
    if chosen is not None and (not policies or any(policy != chosen for policy in policies)):
        raise ValueError("model oracle adapters disagree on explicit full-value readback policy")
    return chosen


def _cb_with_leaf_values(cb: dict) -> dict:
    """A copy of ``cb`` whose leaf tensors carry their operand VALUES inline.

    The command-buffer schema declares a leaf by shape/dtype/role only — values are materialized from
    the declaration, which is how the reference, the simulator and the device harness all end up
    computing on the same stimulus. The mlc arc backend instead indexes ``tensors[name]["data"]``, so
    handing it the raw cb raised ``KeyError: 'data'`` and every capsule failed its RTL tier as a
    ``tool_crash`` (27/28 in both arms of the first saturn_opu agent runs, an infra verdict wearing an
    agent verdict's clothes — the preflight said "mlc arc oracle available", which was true of the
    module and false of the grade).

    Materializing here through the SAME ``materialize_inputs`` the numeric floor uses is the load-bearing
    part: values invented separately for the RTL tier would make it disagree with L0/L1 on stimulus and
    report a mismatch that is really a stimulus difference.
    """
    import copy

    from merlin.runtime.commandbuffer import materialize_inputs

    out = copy.deepcopy(cb)
    leaves = materialize_inputs(cb)
    tensors = out.setdefault("tensors", {})
    for name, spec in tensors.items():
        if name in leaves and "data" not in spec:
            spec["data"] = list(leaves[name].data)
            # A resident handle (RES_PACK dst) is a LAYOUT alias of its source, not a new value — which is
            # exactly how the reference interpreter reads it. The arc backend resolves a matmul's operand names
            # straight against the tensor table, so without the alias entry it raised KeyError on the handle.
            # Only RES_PACK is aliased; any other producer stays absent so the backend fails loudly rather than
            # grading against a value this translation invented.
    for cmd in out.get("commands") or []:
        if cmd.get("opcode") != "RES_PACK":
            continue
        ops = cmd.get("operands") or {}
        src, dst = ops.get("src"), ops.get("dst")
        if src in tensors and dst and dst not in tensors:
            tensors[dst] = {**tensors[src], "resident_of": src}
    return out


def _epilogue_stages_ignored(prepared: dict, res: dict, target: str, mlc_bridge) -> set[str]:
    """Which declared commit-epilogue stages this arc model demonstrably IGNORES (empty set if none).

    Deduced, not declared: run the same buffer again with the epilogue stripped and compare. Identical
    outputs while a stage was declared means the model answered with the pre-epilogue accumulator. Found
    on the first agent runs of a command-buffer OPU target, where the model applies the matmul but not the
    commit stage, so seven epilogue capsules scored numerically "wrong" against a reference that was
    right — a model gap presented as an agent failure. The caller turns this into an UNAVAILABLE tier
    (unknown, not failed); a tier that quietly failed instead would teach the agent to break its epilogue.

    Probing per buffer rather than per target keeps it geometry-free: the shapes are the ones already
    being graded, and only a buffer that actually declares a stage pays the second run.
    """
    import copy

    stages: set[str] = set()
    for cmd in prepared.get("commands") or []:
        if cmd.get("opcode") != "COMMIT":
            continue
        attrs = cmd.get("attributes") or {}
        stages.update(str(x) for x in (attrs.get("epilogue") or []))
    if not stages:
        return set()

    stripped = copy.deepcopy(prepared)
    for cmd in stripped.get("commands") or []:
        if cmd.get("opcode") == "COMMIT":
            (cmd.setdefault("attributes", {}))["epilogue"] = []
    try:
        plain = mlc_bridge.arc_run_command_buffer(stripped, target)
    except Exception:  # noqa: BLE001 — the probe must never turn a working tier into a failure
        return set()
    if (res.get("outputs") or {}) == (plain.get("outputs") or {}):
        return stages
    return set()


def mlc_arc_adapter(target: str) -> Callable:
    """The DEFAULT cross-target RTL oracle: run the command buffer on ``target``'s mlc ARC model (the
    RTL-derived functional model — bit-exact datapath outputs + cycle count from the arc state), for ANY
    target mlc compiled from RTL — NO bespoke sim toolchain needed. This is what makes grading generalize
    across targets. Raises OracleUnavailable if mlc / the arc model is absent for the target."""

    def run(cb, llvm_text, workdir, timeout):
        from merlin.development.targetgen.rtl import mlc_bridge

        if not mlc_bridge.arc_available(target):
            raise OracleUnavailable(f"mlc arc model unavailable for target {target!r}")
        prepared = _cb_with_leaf_values(cb)
        res = mlc_bridge.arc_run_command_buffer(prepared, target)
        ignored = _epilogue_stages_ignored(prepared, res, target, mlc_bridge)
        if ignored:
            raise OracleUnavailable(
                f"the arc model for {target!r} does not model the commit epilogue "
                f"{sorted(ignored)}: it returned identical outputs with the epilogue declared and "
                f"stripped, so its answer is the pre-epilogue accumulator and cannot grade this "
                f"capsule (the hardware does apply it — this is a gap in the model, not in the RTL)"
            )
        _metrics = dict(res.get("metrics") or {})
        # The model reports movement and residency counts beside the cycle total; keeping only
        # `cycles` threw away the direct feedback for the levers a scheduler actually has on this
        # kind of target (how many bytes moved, how often an operand stayed resident).
        _counters = {k: v for k, v in _metrics.items() if k != "cycles" and v is not None}
        return {
            "outputs": res.get("outputs"),
            "cycles": _metrics.get("cycles"),
            "counters": _counters or None,
            "oracle": res.get("oracle"),
            "console": "",
        }

    return run


def _sim_engine_adapters(sim_via: str, target: str) -> dict[str, Callable]:
    """The concrete oracle adapters a DECLARED sim ENGINE provides (additive registry, mirroring
    ``sandbox.toolchain.SIM_TOOLCHAINS``): ``chipyard`` elaborates spike (L2) + an elaborated-RTL engine
    (L3) chosen by :func:`chipyard_l3_selection`. An unknown/absent engine contributes none (the arc RTL
    tier still carries the grade). A new bespoke sim registers one branch here — the engine name is
    DERIVED from the target's contract, never assumed."""
    if sim_via == "chipyard":
        adapters: dict[str, Callable] = {"L2": simulator_adapter("spike", target)}
        plan = _oracle_policy.chipyard_tier_plan(target)
        if plan.selection is None:
            # Reported, not substituted. A tier that cannot run must come back absent; resolving it to
            # the model tier below is how a functional result gets read as an RTL certification.
            print(f"[oracle] {target}: no elaborated-RTL engine for L3 — {plan.unavailable_reason}", flush=True)
            return adapters
        sel = plan.selection
        # The selection is RECORDED, not carried in this dict: every key here is a TIER name and every
        # value a callable (`min(full)` picks the fastest tier, and tier filters compare keys against a
        # capsule's declared tiers), so a metadata entry would be read as a tier.
        from merlin.development.targetgen import rtl_engine_policy as _pol

        print(f"[oracle] {target} L3 engine: {_pol.describe(sel)}", flush=True)
        adapters["L3"] = simulator_adapter(sel["engine"], target, selection=sel)
        return adapters
    return {}


def oracle_adapters(target: str, sim_via: str | None = None, *, readback_policy=None) -> dict[str, Callable]:
    """The oracle adapters per tier for a target. The mlc ARC model is the DEFAULT RTL tier (works for
    ANY mlc target, no bespoke sim); a target that DECLARES a bespoke sim (``sim_via``) additionally gets
    its higher-fidelity sim tiers (chipyard -> spike L2 / verilator L3), preserving that declared sim's path.

    A self-hosted-ISA target (``endpoint_kind == external_backend``) is graded by the generic
    PROGRAM oracle (assemble the emitted `.word`/`.insn` kernel with STOCK LLVM -> its mlc cosim) instead
    of the command_buffer arc path — routed from the contract, no target-name branch.

    ``sim_via=None`` (unspecified) is self-resolved from the target's contract via :func:`_bespoke_sim_via`
    so a bare ``oracle_adapters(target)`` is fully contract-routed — never a silent default target. An
    explicit ``""`` (arc-only) is honored as-is and NOT re-resolved.

    Routing order is DELIBERATE: a declared EXCLUSIVE bespoke sim (a self-hosted SIMT core, ``sim_via=
    cyclotron``) takes precedence over the ``external_backend`` program-oracle default. A SIMT core's
    endpoint is ALSO ``external_backend`` (it emits a kernel, not ``.insn``), but its kernel ELF must be
    graded by its own sim — the arc-cosim program oracle grades the wrong artifact for it. So the sim
    engine is resolved first; only when no exclusive sim is declared does the endpoint select the oracle."""
    _ensure_sim_oracles_discovered()  # wire any target-contributed plugin.sim_oracle
    if sim_via is None:  # unspecified -> derive from contract
        sim_via = _bespoke_sim_via(target)
    so = _SIM_ORACLES.get(sim_via)
    if so is not None and so.exclusive:  # self-hosted SIMT: replaces arc/program default
        return _with_readback_policy(so.adapters(target), readback_policy)
    endpoint_kind, model_ext = _endpoint_of(target)
    if endpoint_kind == "external_backend":
        # Self-hosted-ISA program oracle. ``model_ext`` (the model project that lays out operands + owns
        # the fp8/bf16 dtypes) is REQUIRED from the contract — no ``npu_model`` literal fallback; a target
        # that declares none cannot be program-graded, so we fail closed with an actionable message rather
        # than silently defaulting to one target's model. The DERIVED LADDER here is exactly what mlc
        # provides for an assembled program: a FAST FUNCTIONAL tier (L2 — the model's high-level core, pure
        # Python, no arc .so; the per-round loop tier) and the cycle-exact cosim program-runner (L3 — the
        # gold checkpoint), both resolved by target via mlc inside program_oracle. Each tier guards with
        # its own try/except (OracleUnavailable) so an absent runner reads back as ``unavailable``, never
        # fabricated; the REQUIRED/gold tier stays the cosim (L3) — the functional tier is ADDITIVE.
        plan = _oracle_policy.program_tier_plan(target, model_ext)
        from merlin.development.targetgen.program_oracle import program_oracle_adapter, program_verilator_adapter

        # A tier index is a FIDELITY, not a simulator: L2 is the model tier, L3 is elaborated RTL. The
        # arc cosim is a model compiled FROM the RTL (`rtl_derived_model`, `derived_from_rtl: False`), so
        # it sits at L2 — it used to sit at L3, which on every other target means elaborated RTL, and
        # that collision is how a model result gets read as an RTL certification.
        adapters = {"L2": program_oracle_adapter(target, model_ext=model_ext)}
        # L3: the elaborated-RTL engine, chosen by availability (vcs > gsim > verilator) in
        # program_oracle.select_rtl_engine. None when the target registers no engine -> no L3, reported
        # unavailable rather than silently resolved to the model tier below it.
        if "L3" in plan.tiers:
            rtl = program_verilator_adapter(target, model_ext=model_ext, selection=plan.selection)
            adapters["L3"] = rtl
        return _with_readback_policy(adapters, readback_policy)
    adapters: dict[str, Callable] = {"L3": mlc_arc_adapter(target)}  # arc default (RTL-derived)
    if so is not None:  # optional ADDITIVE bespoke sim (chipyard)
        adapters.update(so.adapters(target))
    return _with_readback_policy(adapters, readback_policy)


# ---- availability ------------------------------------------------------------------------------
# Capsule evaluation: availability.


def oracle_available(target: str, sim_via: str | None = None) -> tuple[bool, str]:
    """Probe whether the target's REQUIRED grading oracle can actually run RIGHT NOW — BEFORE an agent
    run spends anything. Target-agnostic, routed from the contract exactly like :func:`oracle_adapters`:

      * ``external_backend`` (self-hosted-ISA program oracle) -> the mlc arc cosim AND the
        model venv must BOTH be present (the emitted program is assembled/laid-out in the model venv and
        run on the arc cosim);
      * ``command_buffer`` / arc-default target -> the mlc arc model must be present;
      * a DECLARED bespoke sim (``sim_via == "chipyard"``) -> its fastest loop-tier sim
        (spike) binary must be available (else the arc tier, if present, still carries the grade).

    Returns ``(ok, reason)``. ``ok is False`` means grading would only ever emit ``oracle_unavailable``:
    a real (gradeable) run must NOT be launched on it — the sanctioned way to proceed without an oracle
    is an EXPLICIT structure-only smoke (``--no-oracle``). This is the preflight the launcher + the
    GO/NO_GO validator share so a run that cannot be graded aborts before spending tokens.

    Routing order mirrors :func:`oracle_adapters` exactly: a declared EXCLUSIVE bespoke sim
    (``sim_via == "cyclotron"``, a self-hosted SIMT core) is probed FIRST and takes precedence over the
    ``external_backend`` program-oracle default (a SIMT core's endpoint is also external_backend, but the
    arc command-buffer path grades the wrong artifact for it, so arc_available must NOT false-green it)."""
    from merlin.development.targetgen.rtl import mlc_bridge

    _ensure_sim_oracles_discovered()  # wire any target-contributed plugin.sim_oracle
    if sim_via is None:
        sim_via = _bespoke_sim_via(target)
    so = _SIM_ORACLES.get(sim_via)
    if so is not None and so.exclusive:  # self-hosted SIMT: its own sim carries the grade
        return so.available(target)
    endpoint_kind, model_ext = _endpoint_of(target)
    if endpoint_kind == "external_backend":
        if not mlc_bridge.arc_available(target):
            return False, (
                f"external_backend target {target!r}: mlc arc cosim unavailable "
                f"(set MERLIN_MLC_DIR and build the arc model)"
            )
        if not model_ext:
            return False, (
                f"external_backend target {target!r}: contract declares no runner.model_ext — "
                f"cannot resolve the model-venv oracle"
            )
        from merlin.development.targetgen.program_oracle import OracleUnavailable as _PU
        from merlin.development.targetgen.program_oracle import _model_venv_python

        try:
            _model_venv_python(model_ext)
        except _PU as e:
            return False, f"external_backend target {target!r}: {e}"
        return True, (f"external_backend program oracle available (mlc arc cosim + model venv {model_ext!r})")
        # command_buffer / arc-default target (+ an optional ADDITIVE declared bespoke sim, e.g. chipyard).
        # Dispatch through the sim-oracle REGISTRY rather than naming sims here, so a target contributes its
        # own oracle without editing this function.
    if so is not None:
        return so.available(target)
    if mlc_bridge.arc_available(target):
        # "Present" is not "usable": ANSWER a buffer before promising a grade. A target whose arc model
        # imported fine still failed every capsule as a tool_crash because the adapter's input contract
        # was unmet (leaf values, resident aliases, the target argument) — two full agent runs spent
        # against an oracle that could not grade. The probe closes exactly that gap.
        live_ok, live_why = _arc_answers_a_buffer(target)
        if live_ok:
            return True, f"{target!r}: mlc arc oracle available ({live_why})"
        return False, f"{target!r}: mlc arc model present but cannot grade — {live_why}"
    return False, (f"{target!r}: mlc arc model unavailable (set MERLIN_MLC_DIR and build the arc model)")


def _arc_answers_a_buffer(target: str) -> tuple[bool, str]:
    """Run one minimal command buffer through the arc adapter and check it reproduces the reference.

    The buffer is built from DERIVED geometry (the target's own tile edge) in the same schema an agent
    emits, so the probe exercises the real translation path rather than a special case. A mismatch or a
    raise is reported as "cannot grade" with the reason — never swallowed, since a swallowed probe would
    restore the false green it exists to remove.
    """
    from merlin.development.targetgen.corpus_spec import _tile_dim
    from merlin.development.targetgen.target_experiment import load_capability_manifest
    from merlin.runtime.reference import reference_outputs

    try:
        contract = load_capability_manifest(target).contract
    except Exception:  # noqa: BLE001 — no manifest: _tile_dim falls back to the RTL facts / mesh default
        contract = {}
    tile = _tile_dim(target, contract)
    cb = {
        "abi_version": "0.1",
        "target": target,
        "tensors": {
            "probe_w": {"shape": [tile, tile], "dtype": "i8", "role": "weight"},
            "probe_a": {"shape": [tile, tile], "dtype": "i8", "role": "input"},
            "probe_y": {"shape": [tile, tile], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {
                "opcode": "RES_PACK",
                "operands": {"src": "probe_w", "dst": "probe_w_res"},
                "attributes": {"layout": "packed_rhs"},
            },
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "probe_a", "rhs": "probe_w_res", "dst": "probe_acc"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "probe_acc", "dst": "probe_y"},
                "attributes": {"epilogue": [], "output_dtype": "i32"},
            },
        ],
    }
    try:
        got = mlc_arc_adapter(target)(cb, "", None, 120).get("outputs") or {}
    except Exception as e:  # noqa: BLE001 — includes OracleUnavailable: either way it cannot grade
        return False, f"a {tile}x{tile} probe buffer raised {type(e).__name__}: {str(e)[-200:]}"
    want = reference_outputs(cb)
    if not got:
        return False, f"a {tile}x{tile} probe buffer produced no outputs"
    if any(got.get(k) != v for k, v in want.items()):
        return False, (
            f"a {tile}x{tile} probe buffer disagreed with the reference — the model answers, "
            "but not with this buffer's result"
        )
    return True, f"answered a {tile}x{tile} probe buffer matching the reference"


def _full_ladder_enabled() -> bool:
    """Whether a mandatory tier failure completes the ladder before failing the capsule.

    ON by default: a GRADE must be able to say what every declared tier thought, and a tier with no
    record is not evidence. The cost is real -- a failing capsule now pays the tiers ordered after the
    one that refuted it -- so a fast iteration loop can turn it off with MERLIN_FULL_LADDER=0. Never
    turn it off for a run whose numbers will be quoted.
    """
    import os

    return (os.environ.get("MERLIN_FULL_LADDER", "1") or "1").strip().lower() not in ("0", "false", "no")


def codegen_smoke(target: str, *, backend_target: str | None = None) -> tuple[bool | None, str]:
    """Check production code generation through the selected support backend.

    True means the probe ran and passed, False refuses admission, and None records
    unavailable coverage, never success. An experiment may declare backend identity;
    simulator names and ISA properties never select a compiler. The hook receives
    the original experiment target.
    """
    from merlin.runtime.backends import base as backends

    selected = target if backend_target is None else backend_target
    if not isinstance(selected, str) or not selected.strip():
        return False, "codegen smoke requires a non-empty backend target"
    try:
        backend = backends.get_backend(selected)
    except KeyError as exc:
        # Failed plugin loads also raise KeyError; only exact absence permits a skip.
        if backend_target is None and exc.args == (selected,):
            return None, f"n/a (no codegen backend registered for {selected!r})"
        return False, f"codegen backend {selected!r} unavailable: {str(exc)[-200:]}"
    except Exception as exc:  # noqa: BLE001 — selected backend failure refuses admission
        return False, f"codegen backend {selected!r} failed to load: {type(exc).__name__}: {str(exc)[-200:]}"
    hook = getattr(backend, "preflight_codegen_smoke", None)
    if hook is None:
        return None, f"n/a (backend {selected!r} provides no production codegen smoke)"
    if not callable(hook):
        return False, f"backend for {selected!r} exposes a non-callable preflight_codegen_smoke"
    try:
        result = hook(target=target)
    except Exception as exc:  # noqa: BLE001 — a broken production probe refuses admission
        return False, f"production codegen smoke raised {type(exc).__name__}: {str(exc)[-200:]}"
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or (result[0] is not None and type(result[0]) is not bool)
        or not isinstance(result[1], str)
    ):
        return False, f"backend for {selected!r} returned a malformed preflight_codegen_smoke result"
    return result


def _resolve_oracle_adapters(target: str) -> dict[str, Callable]:
    """Contract-routed adapter set for ``target`` — the ``run_capsule`` default when a caller passes no
    adapters. Calls the module :func:`oracle_adapters` (self-resolving sim_via) so an unrouted grade goes
    to the target's OWN endpoint oracle (external_backend->program_oracle, chipyard->spike/verilator,
    else arc), never a fixed target or experiment policy. (Named to dodge the ``run_capsule``
    parameter that shadows ``oracle_adapters`` in that function's local scope.)"""
    return oracle_adapters(target)


def qa_loop_adapters(
    target: str,
    sim_via: str | None = None,
    *,
    declared_tiers: set[str] | None = None,
    readback_policy=None,
) -> dict[str, Callable]:
    """The FAST per-round QA-loop oracle set for ``target`` — resolved from :func:`oracle_adapters`, never
    hardwired. It keeps ONE tier (the fastest) and reserves the slower cycle-accurate tiers for the bounded
    checkpoint (:func:`qa_checkpoint_adapters`). This is a tier-order distinction, not a per-target one; no
    target-name branch — a new accelerator's loop gate falls out of its declared ``sim_via`` with no edit.

    ``declared_tiers`` is the set of tiers the CORPUS BEING GRADED lists in ``required_oracle_tiers``.
    When given, the loop tier is the fastest endpoint tier that the corpus actually declares — so the
    per-round gate always rides a tier the capsule asked for. This matters where an endpoint exposes an
    ADDITIVE cheaper tier below its declared gold tier: picking the merely-fastest tier there makes the
    loop enforce a gate the capsule never declared (and, when that cheap tier's runner misbehaves, every
    capsule fails on it while the declared tier passes). Returns ``{}`` — fail closed, never a
    substitute — when the endpoint reaches none of the declared tiers; the caller reports that.

    Omitting ``declared_tiers`` keeps the legacy "fastest available tier" behavior for callers that have
    no corpus in hand."""
    full = (
        oracle_adapters(target, sim_via)
        if readback_policy is None
        else oracle_adapters(target, sim_via, readback_policy=readback_policy)
    )
    if not full:
        return {}
    if declared_tiers:
        cand = sorted(t for t in full if t in declared_tiers)
        if not cand:
            return {}
        return {cand[0]: full[cand[0]]}
    lowest = min(full)  # tier keys sort lexically (L2 < L3 < L4 …): lowest == fastest
    return {lowest: full[lowest]}


def qa_checkpoint_adapters(target: str, sim_via: str | None = None, *, readback_policy=None) -> dict[str, Callable]:
    """The cycle-accurate QA CHECKPOINT oracle set for ``target`` = its full oracle ladder from
    :func:`oracle_adapters` (chipyard: spike L2 + verilator L3; arc/mlc: the RTL-derived arc model). This
    is the higher-fidelity barrier run once the fast loop (:func:`qa_loop_adapters`) has converged — not
    every round. Kept as a named seam so the loop driver never re-hardwires ``{L2: spike, L3: verilator}``."""
    if readback_policy is None:
        return oracle_adapters(target, sim_via)
    return oracle_adapters(target, sim_via, readback_policy=readback_policy)
