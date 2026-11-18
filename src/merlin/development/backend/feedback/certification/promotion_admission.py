"""Current-source admission of automatic promotions; never qualification authority.

Obsolete requests and their original source/ledger records remain untouched. A
deferred request becomes eligible again if its source or proven dependencies
return. Explicit caller requests do not use this policy.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from merlin.development.backend.context import InvocationContext
from merlin.development.targetgen.oracles.schedule import CapsuleState, Verdict, valid_execution_digest
from merlin.target.simulators.engine_policy import ELABORATED_RTL as _ELABORATED_RTL

# ---- promotion_admission -----------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    digest: str
    components: dict[str, str]


def read_note(path: Path):
    """Bounded scheduling metadata read; candidate channel links confer no access."""
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 8192:
            return None
        with os.fdopen(descriptor) as stream:
            descriptor = None
            return json.loads(stream.read(8192))
    except (OSError, ValueError):
        return None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def write_note(path: Path, document: dict) -> None:
    """Atomically replace metadata without following an author-created link."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", prefix=".promotion-admission-", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def observe_source(workspace: Path) -> Source:
    from merlin.development.backend.feedback.certification import promotion

    if not (Path(workspace) / "submission").is_dir():
        raise ValueError("promotion source is unavailable")
    digest, components, _ = promotion.submission_digests(workspace)
    return Source(digest, components)


def current_execution(state: dict, name: str, tier: str, digest: str, *, instrument=None) -> str | None:
    """An existing host-recorded loop result explicitly belonging to this source."""
    from merlin.development.backend.feedback.certification import promotion

    if not valid_execution_digest(instrument):
        return None
    for entry in reversed(list(promotion._slots_ro(state, name, tier).values())):
        if (
            isinstance(entry, dict)
            and entry.get("status") == "pass"
            and entry.get("digest") == digest
            and entry.get("instrument") == instrument
            and valid_execution_digest(entry.get("execution_digest"))
        ):
            return entry["execution_digest"]
    return None


def reason(
    name: str,
    *,
    current: Source,
    original: Source,
    current_instrument: str | None,
    original_instrument: str | None,
    original_execution: str | None = None,
    observed_execution: str | None = None,
) -> str | None:
    """None admits work; a string defers it without changing any verdict.

    The dependency union includes newly added/removed components and the
    unattributed bucket. Missing narrower evidence falls through the existing
    scheduler's conservative whole-source comparison.
    """
    if not current.digest or not original.digest:
        return "source identity unavailable"
    if not valid_execution_digest(current_instrument) or not valid_execution_digest(original_instrument):
        return "grading instrument identity unavailable"
    if current_instrument != original_instrument:
        return "grading instrument changed"
    state = CapsuleState(
        name=name,
        digest=current.digest,
        components=current.components,
        depends_on=tuple(sorted(current.components.keys() | original.components.keys())) or None,
        execution_digest=observed_execution,
        verdicts={"cert": Verdict("pending", original.digest, original.components, original_execution)},
    )
    invalid = state.invalidated_by("cert")
    return "superseded source: " + ", ".join(str(item) for item in invalid) if invalid else None


def queued_reason(
    workspace: Path,
    request: dict,
    *,
    loop_tier: str,
    cert_tier: str,
    instrument: str | None,
    loop_instrument: str | None,
    current: Source,
    state: dict,
    snapshots: dict[str, Source],
) -> str | None:
    """Authenticate the protected original snapshot and its outstanding ledger slot."""
    if not request.get("promoted"):
        return None
    from merlin.development.backend.feedback.certification import promotion
    from merlin.development.backend.feedback.execution.snapshots import promotion_snapshot_path

    digest, identity = request.get("submission_digest"), request.get("identity")
    if not isinstance(digest, str) or not digest or not isinstance(identity, str):
        return "enqueue source identity unavailable"
    token = request.get("submission_snapshot")
    frozen = promotion_snapshot_path(workspace, token)
    if frozen is None:
        return "protected enqueue snapshot unavailable"
    try:
        original = snapshots.get(token)
        if original is None:
            original = observe_source(frozen)
            snapshots[token] = original
    except (OSError, ValueError):
        return "protected enqueue snapshot unavailable"
    if original.digest != digest:
        return "protected enqueue snapshot identity changed"
    names = str(request.get("capsules") or "").split(",")
    if not names or any(not name or name == "all" for name in names):
        return "automatic promotion has no explicit capsule identity"
    for name in names:
        entry = promotion._slots_ro(state, name, cert_tier).get(identity)
        if (
            not isinstance(entry, dict)
            or entry.get("status") != "pending"
            or entry.get("digest") != digest
            or promotion.record_identity(entry) != identity
        ):
            return "protected outstanding promotion identity unavailable"
        refusal = reason(
            name,
            current=current,
            original=original,
            current_instrument=instrument,
            original_instrument=entry.get("instrument"),
            original_execution=entry.get("execution_digest"),
            observed_execution=current_execution(state, name, loop_tier, current.digest, instrument=loop_instrument),
        )
        if refusal:
            return refusal
    return None


# ---- pending_retirement ------------------------------------------------------------------------
# Retire a pending certificate record whose own job finished without an attributable result.
#
# ``promote()`` marks a capsule's cert tier ``pending`` when it enqueues the job, and the scheduler treats a
# pending record as known -- an in-flight job must not be bought twice. That is only right while the job is
# in flight. When the job completes and ``record_cert`` cannot attribute its result to the record (the
# boundary join refused it, the result names no execution identity the record can be checked against, or it
# states no verdict), nothing else will ever resolve that record, so the capsule read "pending" for the rest
# of the run and the scheduler never asked for its cert tier again. Measured on the fresh6 campaign: 101
# capsules stranded at L3 pending, and from about +90 min every promotion said "0 capsule(s) want L3".
#
# Retiring is narrow on purpose. It needs the completed job's own request id, so only a completion that
# provably belongs to THIS record retires it; a replay of an older job (``completions.recover``) cannot
# retire a newer record for the same bytes. The retired record keeps every field it had and is not a
# verdict: :data:`~merlin.development.targetgen.oracles.schedule.UNATTRIBUTED_RESULT` reads as "no verdict" to the
# scheduler, so the next loop verdict for those bytes re-enqueues them under a new request id.


def retire(st, name, tier, key, *, request, reason, completed=None, log=None) -> bool:
    """Mark ``st[name][tier]`` record ``key`` unattributed when it is the pending record ``request`` was
    enqueued for. Returns whether it did. ``st`` is modified in place; the caller saves it."""
    from merlin.development.backend.feedback.certification import promotion as P
    from merlin.development.targetgen.oracles.schedule import UNATTRIBUTED_RESULT

    if not request or key is None:
        return False
    slots = P._slots_ro(st, name, tier)
    entry = slots.get(str(key))
    # Legacy records without a request cannot establish completion ownership.
    if not isinstance(entry, dict) or entry.get("status") != "pending" or entry.get("request") != request:
        return False
    retired = dict(
        entry,
        status=UNATTRIBUTED_RESULT,
        unattributed={"request": request, "reason": reason, "completed_execution_digest": completed, "at": time.time()},
    )
    P._slots(st, name, tier)[str(key)] = retired
    mirror = (st.get(name) or {}).get(tier)
    if isinstance(mirror, dict) and P.record_identity(mirror) == str(key):
        st[name][tier] = retired
    if log is not None:
        print(
            f"[promote] {name} {tier} pending record retired for re-queue: job {request} finished "
            f"but its result could not be attributed ({reason})",
            file=log,
            flush=True,
        )
    return True


def completed_request(function):
    """Retire unresolved records after the original recorder, inside its ledger transaction.

    The broker supplies the completed request id; no result field can select it. Recovery of an older
    completion cannot retire a retry because each attempt has a distinct id recorded on its slot.
    """
    from functools import wraps

    @wraps(function)
    def record(ws, verdict, cert_tier, log=None, identity=None, source_identity_verified=False, *, request=None):
        from merlin.development.backend.feedback.certification import promotion as P
        from merlin.development.backend.feedback.verdicts import hydrated

        verdict = hydrated(ws, verdict)  # the full row carries the L3 boundary the join needs

        if request and identity is not None:
            state = P._tier_state(ws)
            rows = []
            for row in (verdict or {}).get("per_capsule") or ():
                entry = P._slots_ro(state, row.get("capsule"), cert_tier).get(str(identity))
                if (
                    isinstance(entry, dict)
                    and entry.get("status") == "pending"
                    and entry.get("request") not in (None, request)
                ):
                    continue
                rows.append(row)
            verdict = {**(verdict or {}), "per_capsule": rows}
        resolved = function(ws, verdict, cert_tier, log, identity, source_identity_verified)
        if not request or identity is None:
            return resolved
        state = P._tier_state(ws)
        changed = False
        for name in state:
            changed |= retire(
                state,
                name,
                cert_tier,
                identity,
                request=request,
                reason="completed job supplied no attributable certificate",
                log=log,
            )
        if changed:
            P._save_tier_state(ws, state)
        return resolved

    return record


def requeue_attempt(slots, key) -> int | None:
    """``None`` when ``key`` already holds a record that must not be re-bought (pending, pass, fail,
    joined), else the attempt number a new request for it carries: 0 for a first request, n+1 after a
    retired one, so its request id differs from every earlier request for the same bytes."""
    from merlin.development.targetgen.oracles.schedule import UNATTRIBUTED_RESULT

    entry = slots.get(key)
    if entry is None:
        return 0
    if isinstance(entry, dict) and entry.get("status") == UNATTRIBUTED_RESULT:
        return int(entry.get("attempt") or 0) + 1
    return None


# ---- dispatch ----------------------------------------------------------------------------------
# Closed async simulator dispatch shared by broker and certificate promotion.

_NEUTRAL_SIM = "contract"

REQUIRED_SIM = "required"


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
        from merlin.target.simulators.engine_policy import ENGINE_PRIORITY
    except Exception:  # noqa: BLE001 -- no policy module: keep the historical ladder, do not widen
        engines = ("vcs", "verilator")
    else:
        engines = tuple(ENGINE_PRIORITY)
    # A campaign-wide engine pin is stronger than this agent-controlled request surface.  Spike remains
    # a correctness-only screen; an RTL request may name only the exact required engine.  Previously the
    # inherited pin reached the child but `_adapters` directly constructed whichever engine the request
    # named, so a GSIM-only run could still launch Verilator from the sandbox broker.
    required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
    # REQUIRED_SIM is always accepted: it selects the grade's own adapters, which already honour the
    # pin, so it can never reach an engine the pin excludes.
    if required:
        return ("spike", required, REQUIRED_SIM) if required in engines else ("spike", REQUIRED_SIM)
    return ("spike",) + engines + (REQUIRED_SIM,)


def grade_selection_for(sim: str | None) -> str | None:
    """An author request naming the experiment's pinned RTL engine asks for the grade's own selection.

    Under a pin the staged in-sandbox self-check shim fills ``--sim`` with the pinned engine by
    default, so an unflagged author request arrives naming it. That engine's single-engine ladder is
    the same tiers (the functional screen plus the engine at L3), so serving it with the grade's own
    adapters changes no tier: the request certifies exactly as the grade does, under the grade's
    certificate instrument. ``spike``, any other name, and an engine whose legacy ladder has another
    shape (VCS's historical L4 rung) keep their own meaning. Promotion jobs are not author requests
    and are never passed through here.
    """
    from merlin.development.backend.campaign import control as joint_control

    if joint_control.selected() is not None:
        return sim
    from merlin.development.backend.feedback.selfcheck.adapters import SIM_TIER

    required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
    if not sim or not required or sim != required or SIM_TIER.get(sim) != "L3":
        return sim
    return REQUIRED_SIM


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
        from merlin.target.registry.descriptor_loading import load_target_experiment
        from merlin.target.registry.experiment_descriptor import load_capability_manifest
        from merlin.target.registry.identity import runner_config_from_manifest

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
            from merlin.target.simulators.oracle_policy import chipyard_l3_selection

            sim = str((chipyard_l3_selection(te.target) or {}).get("engine") or "").strip() or None
    except Exception:  # noqa: BLE001 -- unresolvable map: no promotion, and the caller says so
        return None
    return sim if sim in allowed else None


# ---- certificate_budget ------------------------------------------------------------------------
# Apply an explicitly declared per-capsule certificate cost budget.
#
# Unpriced infrastructure failures retain the existing admission behavior;
# a declared, evaluated ceiling may decline work without changing verdicts.


def affordable_cert_items(want, *, cert_tier, log, context: InvocationContext, capsules_root: Path | None = None):
    """``want`` minus the capsules a DECLARED per-capsule budget says cost too much at ``cert_tier``.

    OFF UNLESS DECLARED. `tier_policy.ceiling_budget_seconds` is opt-in and has no default, so with no
    budget in force this returns ``want`` unchanged before touching a manifest, a cost fit or the disk.
    A budget nobody declared, silently deciding which capsules get certified, is exactly the quiet
    coverage loss the ceiling exists to make visible.

    A COST VERDICT MAY DECLINE; AN INFRASTRUCTURE FAILURE MAY NOT. `oracle_ceiling` fails closed on an
    UNKNOWN cost, which is right -- "not shown to fit" is not "fits". But a manifest that will not load
    is not a statement about cost, and dropping the capsule for it would be a silent coverage loss
    wearing a budget's clothes. Those allow, loudly. This mirrors how promotion is wrapped at its call
    site: an optimisation, never a gate.

    NOT `schedule(cost_s=..., budget_s=...)`: that budget is a TOTAL across the queue and its cost is
    per-TIER, so it answers "how many may we buy" and cannot answer "may THIS capsule be bought".
    """
    from merlin.development.targetgen.oracles.tier_policy import ceiling_budget_seconds, oracle_ceiling

    if ceiling_budget_seconds() is None:
        return want, {}
    try:
        from merlin.development.targetgen.corpus.capsule_io import load_capsule
        from merlin.target.registry.descriptor_loading import load_target_experiment

        te = load_target_experiment(context.descriptor)
        by_name = {
            d.name: d
            for root in ((capsules_root,) if capsules_root is not None else te.graded_roots())
            for d in Path(root).iterdir()
            if (d / "capsule.yaml").is_file()
        }
    except Exception as exc:  # noqa: BLE001 -- cannot read the corpus: certify everything, and say so
        print(
            f"[promote] budget declared but the corpus could not be read ({type(exc).__name__}); "
            f"every capsule admitted to {cert_tier}",
            file=log,
            flush=True,
        )
        return want, {}
    kept, declined = [], {}
    for w in want:
        d = by_name.get(w.capsule)
        if d is None:
            print(
                f"[promote] {w.capsule}: no manifest under the graded roots; admitted to "
                f"{cert_tier} rather than dropped on a budget it was never priced against",
                file=log,
                flush=True,
            )
            kept.append(w)
            continue
        try:
            ceiling = oracle_ceiling(te.target, load_capsule(d), cert_tier)
        except Exception as exc:  # noqa: BLE001 -- an unaskable question is not a NO
            print(
                f"[promote] {w.capsule}: cost ceiling could not be evaluated "
                f"({type(exc).__name__}); admitted to {cert_tier}",
                file=log,
                flush=True,
            )
            kept.append(w)
            continue
        if ceiling.allowed:
            kept.append(w)
        else:
            declined[w.capsule] = {
                "source": ceiling.source,
                "axis": ceiling.axis,
                "reason": ceiling.reason,
                "record": ceiling.record,
            }
            print(f"[promote] {w.capsule}: NOT enqueued for {cert_tier} -- {ceiling.reason}", file=log, flush=True)
    return kept, declined
