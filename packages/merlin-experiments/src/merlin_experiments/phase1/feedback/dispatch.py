"""Closed async simulator dispatch shared by broker and certificate promotion."""

from __future__ import annotations

import os

from merlin.targetgen.rtl_engine_policy import ELABORATED_RTL as _ELABORATED_RTL
from merlin_experiments.phase1.context import InvocationContext

_NEUTRAL_SIM = "contract"


def _sim_via(context: InvocationContext) -> str:
    """This target's declared simulator route, from its own descriptor (empty = in-process RTL model)."""
    import yaml

    def _find(node):
        if isinstance(node, dict):
            v = node.get("sim_via")
            if isinstance(v, str):
                return v.strip()
            for x in node.values():
                r = _find(x)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for x in node:
                r = _find(x)
                if r is not None:
                    return r
        return None

    try:
        d = yaml.safe_load((context.descriptor).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 -- an unreadable descriptor means "no bespoke sim", fail closed
        return ""
    return _find(d) or ""


def allowed_sims(context: InvocationContext) -> tuple[str, ...]:
    """The sim names a request may name — STILL a closed allowlist (this is load-bearing isolation:
    an agent must not be able to make the broker run something arbitrary), but DERIVED rather than
    baked to one target's ladder.

    A chipyard target selects along the spike/verilator/vcs ladder. A target that declares no bespoke
    sim grades on its own contract-resolved tier, where ``--sim`` does not apply at all
    (``agent_selfcheck._adapters``) -- so the only accepted token is the neutral sentinel below, and it
    is NOT forwarded to the self-check. Without this, such a target's agent could reach no oracle from
    inside the sandbox while every gate reported the sandbox healthy.
    """
    if _sim_via(context) != "chipyard":
        return (_NEUTRAL_SIM,)
    # DERIVE the elaborated-RTL engines from the engine policy instead of restating one ladder here.
    # The policy is the single place that knows which engines exist and in what order; a second literal
    # tuple in this file is how `gsim` came to be unreachable from inside the sandbox after the backend
    # already supported it. The screen tier (spike) is not an elaborated-RTL engine, so it is named
    # separately. Fail CLOSED: if the policy cannot be imported, offer only what was always offered --
    # never widen the allowlist on an error path (this list is load-bearing isolation).
    try:
        from merlin.targetgen.rtl_engine_policy import ENGINE_PRIORITY
    except Exception:  # noqa: BLE001 -- no policy module: keep the historical ladder, do not widen
        engines = ("vcs", "verilator")
    else:
        engines = tuple(ENGINE_PRIORITY)
    # A campaign-wide engine pin is stronger than this agent-controlled request surface.  Spike remains
    # a correctness-only screen; an RTL request may name only the exact required engine.  Previously the
    # inherited pin reached the child but `_adapters` directly constructed whichever engine the request
    # named, so a GSIM-only run could still launch Verilator from the sandbox broker.
    required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
    if required:
        return ("spike", required) if required in engines else ("spike",)
    return ("spike",) + engines


def cert_sim(cert_tier: str, *, context: InvocationContext) -> str | None:
    """The ``--sim`` token the BROKER will accept for ``cert_tier``, or None when none does.

    ``promote()`` used to write :data:`_NEUTRAL_SIM` unconditionally. That is correct only for a target
    whose ladder comes from its own contract, where ``--sim`` does not apply and the sentinel is the ONLY
    accepted token. A target that declares a bespoke sim ladder accepts that ladder's names and REJECTS
    the sentinel -- so every promotion request such a target wrote was refused, while the capsule had
    already been marked ``pending`` a few lines earlier and therefore stayed pending forever.

    Measured on the live gemmini round merlincirct_arm4_func_20260901_v4: 6 promotion requests, every one
    answered "rejected: --sim 'contract' is not accepted for this target. Use 'spike' or 'verilator' or
    'vcs'", and 2 capsules stranded at L3 pending. Promotion had never fired on a bespoke-sim target.

    Derived from the same allowlist the broker validates against, so the two cannot drift apart again.
    Returns None rather than a guess when nothing serves the tier: the caller must then NOT enqueue, and
    must not mark the capsule pending for a job that can never run.
    """
    try:
        allowed = tuple(allowed_sims(context))
    except Exception:  # noqa: BLE001 -- broker not importable: keep the historical sentinel
        return _NEUTRAL_SIM
    if allowed == (_NEUTRAL_SIM,):
        return _NEUTRAL_SIM
    try:
        from merlin.targetgen.runner_config import runner_config_from_manifest
        from merlin.targetgen.target_experiment import load_capability_manifest, load_target_experiment

        te = load_target_experiment(context.descriptor)
        cfg = runner_config_from_manifest(load_capability_manifest(te.target))
        required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
        if required and cert_tier in set(cfg.rtl_tiers or ()):
            # A tier is a FIDELITY, not a historical binary binding. Under an experiment-wide engine pin
            # the required engine serves every elaborated-RTL tier it implements; consulting `tier_sim`
            # first would recover the manifest's old Verilator label, find it excluded by the broker, and
            # silently disable promotion for the whole of a GSIM-only run.
            return required if required in allowed else None
        sim = (cfg.tier_sim or {}).get(cert_tier)
        # THE CONTRACT NAMES A FIDELITY, NOT A BINARY. `tier_sim` used to read `{L3: verilator}` and the
        # broker's allowlist happened to contain that word, so this returned an engine by accident. Once
        # the contract said what it means -- `{L3: elaborated_rtl}` -- the sentinel matched no `--sim`
        # token, this returned None, and promotion silently switched off for every unpinned run: the
        # caller logs "no --sim serves L3" once and then never enqueues, which is indistinguishable from
        # a round with nothing to promote. Resolve the sentinel through the SAME availability policy the
        # contract comment names, so the fidelity is declared once and the engine chosen once.
        if sim == _ELABORATED_RTL:
            from merlin.targetgen.capsule_runner import chipyard_l3_selection

            sim = str((chipyard_l3_selection(te.target) or {}).get("engine") or "").strip() or None
    except Exception:  # noqa: BLE001 -- unresolvable map: no promotion, and the caller says so
        return None
    return sim if sim in allowed else None
