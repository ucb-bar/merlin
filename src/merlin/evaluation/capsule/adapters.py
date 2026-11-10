"""Capsule evaluation: adapters."""

from __future__ import annotations

from collections.abc import Callable

from merlin.evaluation.capsule.certificates import _PREPARED_EXECUTION_OBSERVER
from merlin.targetgen import oracle_policy as _oracle_policy
from merlin.targetgen.oracle_policy import _SIM_ORACLES, _bespoke_sim_via, _endpoint_of, _ensure_sim_oracles_discovered


class OracleUnavailable(Exception):
    pass


def simulator_adapter(sim: str, target: str, selection: dict | None = None, *, readback_policy=None) -> Callable:
    """Run one declared engine, retaining its selected identity and policy record."""

    from merlin.targetgen.contract.readback_policy import MEMORY_TRANSPORTS, selected

    readback_policy = selected(readback_policy)
    if readback_policy is not None and readback_policy.transport in MEMORY_TRANSPORTS:
        from merlin.runtime.backends import base as backends
        from merlin.targetgen.oracle_readback import validate_engine_policy

        validate_engine_policy(readback_policy, sim=sim, backend=backends.get_backend(target))

    def run(cb, llvm_text, workdir, timeout):
        from merlin.runtime.backends import base as _backends

        backend = _backends.get_backend(target)
        if not backend.available(sim):
            raise OracleUnavailable(f"{sim} not available")
        if sim == "gsim":
            from merlin.targetgen import gsim_emulator

            exact, reason = gsim_emulator.selected_firrtl_status(
                target, env_var=getattr(backend, "GSIM_EMU_ENV", None), backend=backend
            )
            if not exact:
                raise OracleUnavailable(reason)
        from merlin.targetgen.oracle_readback import run_with_readback

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
    from merlin.targetgen.contract.readback_policy import selected

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
    from merlin.targetgen.contract.readback_policy import selected

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
        from merlin.targetgen.rtl import mlc_bridge

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
        from merlin.targetgen import rtl_engine_policy as _pol

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
        from merlin.targetgen.program_oracle import program_oracle_adapter, program_verilator_adapter

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
