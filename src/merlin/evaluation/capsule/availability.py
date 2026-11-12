"""Capsule evaluation: availability."""

from __future__ import annotations

from collections.abc import Callable

from merlin.development.targetgen.oracle_policy import (
    _SIM_ORACLES,
    _bespoke_sim_via,
    _endpoint_of,
    _ensure_sim_oracles_discovered,
)
from merlin.evaluation.capsule.adapters import mlc_arc_adapter, oracle_adapters


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
