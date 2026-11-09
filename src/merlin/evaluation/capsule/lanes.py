"""Capsule evaluation: lanes."""

from __future__ import annotations

from merlin.evaluation.capsule.certificates import run_executables
from merlin.evaluation.capsule.diagnostics import _engine_token as _engine_token
from merlin.evaluation.capsule.diagnostics import _oracle_kind as _oracle_kind
from merlin.evaluation.capsule.lanes_evidence import lane_report as _lane_report
from merlin.targetgen import tier_cache as _TC
from merlin.targetgen.elf_lanes import ACCELERATOR_LANE as _ACCELERATOR_LANE
from merlin.targetgen.elf_lanes import EXECUTED_LANE_EVIDENCE

_LANE_COUNTERS: dict[str, tuple[str, str]] = {
    "on_mesh": ("mesh", "matmul_layers_on_mesh"),
    "scalar_rvv_lane": ("host", "kernels_ran"),
}


def _sim_telemetry(oracle_meta, ev_name: str) -> dict:
    """The simulator-identity fields for a TierResult, from what the adapter reported.

    Kept as ONE extractor because the three tier paths that build a TierResult must not disagree about
    what a record says the engine was — and because a field only some paths populate is worse than one
    none of them do: a consumer reads the absence as "verilator", never as "unrecorded".
    """
    kind = _oracle_kind(oracle_meta) or ev_name
    declared = oracle_meta.get("engine") if isinstance(oracle_meta, dict) else None
    # The adapter's OWN word first. Deriving the engine from the kind string works and is the fallback,
    # but an adapter that states its engine outright is the authority on which one it ran -- the same
    # rule `derived_from_rtl` already follows, for the same reason.
    out: dict = {"engine": str(declared) if declared else (_engine_token(kind) or kind)}
    if isinstance(oracle_meta, dict):
        if isinstance(oracle_meta.get("selection"), dict):
            out["engine_selection"] = dict(oracle_meta["selection"])
        if isinstance(oracle_meta.get("provenance"), dict):
            out["sim_provenance"] = dict(oracle_meta["provenance"])
    return out


def lane_report(
    capsule: dict,
    routing_plan: dict | None,
    mesh_execution: dict | None = None,
    host_execution: dict | None = None,
    lane_execution: dict | None = None,
) -> dict | None:
    """Retain the original counter-map override at the runner boundary."""
    return _lane_report(
        capsule,
        routing_plan,
        mesh_execution,
        host_execution,
        lane_execution,
        lane_counters=_LANE_COUNTERS,
    )


def forbids_accelerator(capsule: dict) -> bool:
    """Does this capsule declare that work must NOT reach the accelerator?"""
    return _ACCELERATOR_LANE in [str(x) for x in ((capsule.get("lanes") or {}).get("forbid") or ())]


def accelerator_lane_violated(capsule: dict, trace) -> bool:
    """True only when a forbidding capsule's emitted stream DECODES accelerator instructions.

    The asymmetry here is the whole reason this is sound on a path that owns no execution record. A
    decoded accelerator instruction proves work went there. Its ABSENCE proves nothing: the decoder
    recognizes the ``.insn r`` form, so a ``.word``-encoded kernel that does drive the device decodes as
    silent, and reading that silence as "the host carried it" would hand a forbidding capsule a free
    pass. Violated, or unmeasured -- never satisfied.
    """
    from merlin.targetgen import trace_check as _TCK

    if not forbids_accelerator(capsule):
        return False
    try:
        return bool(_TCK.drives_accelerator(trace))
    except Exception:  # noqa: BLE001 -- an undecodable trace measures nothing
        return False


def _tier_field(record, key, default=None):
    """One field of a tier record, whichever form it is in -- a :class:`TierResult` or its ``to_dict``."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def carried_verdict_is_attributed(record, execution_identity) -> bool:
    """Does this CARRIED tier verdict belong to the bytes being graded right now?

    A carried verdict is not a guess and not stale by construction. :mod:`merlin.targetgen.tier_cache`
    keys a certificate on :func:`tier_cache.execution_identity` -- a content digest of the executables
    this run produced, the target, and the toolchain pins -- and ``lookup`` re-verifies the stored
    record's own copy of that key before returning it, so a hit is a statement about exactly these
    bytes on exactly this instrument. ``tier_promote.record_cert`` resolves against the same identity
    and never re-hashes, so a certificate is never re-attributed to bytes edited since.

    That is why this asks the question rather than refusing outright: per-capsule promotion is the
    designed behaviour in continuous mode, so most graded capsules CARRY their expensive tier, and
    refusing every carried verdict would make a lane contract flap on cache state rather than on
    compiler quality -- the same class of defect the lane evidence exists to remove.

    It is asked INDEPENDENTLY, though: ``execution_identity`` is recomputed here from the bytes on disk
    (see :func:`graded_execution_identity`) and compared to the identity the carried block records. A
    mismatch, a malformed digest, or no identity at all is refused -- fail closed, because an
    unattributable verdict is exactly the stale-certificate case the strict rule was protecting against.
    """
    from merlin.targetgen.oracle_schedule import valid_execution_digest

    carried = _tier_field(record, "carried")
    if not isinstance(carried, dict):
        return False
    earned = carried.get("execution_identity")
    if not valid_execution_digest(earned) or not valid_execution_digest(execution_identity):
        return False
    return str(earned) == str(execution_identity)


def graded_execution_identity(*, target, generated, shas) -> str | None:
    """The content identity of what this run would execute, or ``None`` when it cannot be formed.

    The same function the certificate cache keys on, called on the bytes present NOW, so comparing it
    with a carried record's identity is an independent check rather than a restatement of the cache's
    own claim. ``None`` on any refusal (no readable executable, an unidentifiable device), which callers
    must treat as "not attributable".
    """
    try:
        return _TC.execution_identity(target=target, executables=run_executables(generated), toolchain_shas=shas)
    except Exception:  # noqa: BLE001 -- an identity we cannot form credits nothing
        return None


def executed_program_tiers(tiers: dict, execution_tiers, *, execution_identity=None) -> list[str]:
    """The oracle tiers that certify THESE bytes on an instrument, and passed.

    ``execution_tiers`` is the runner's own set of tiers that execute the program on an instrument
    (the target's ``tier_sim`` ladder plus any injected adapter) -- the L0/L1 command-buffer
    interpretation is never in it, because interpreting a buffer we built is not running the program.
    A tier is counted only when its record says it passed and that verdict is about this program:

    * ``not_applicable`` / ``budget_deferred`` -- the tier never ran; the first is honestly N/A and the
      second was deliberately not bought. Either way nothing executed, for these bytes or any others.
    * ``carried`` -- executed in an EARLIER run. Counted only when
      :func:`carried_verdict_is_attributed` matches the certificate's identity against
      ``execution_identity``, the identity recomputed from the bytes under grade; refused otherwise, and
      refused outright when the caller supplies no identity to compare against.

    Accepts either a :class:`TierResult` or its ``to_dict`` form, because the model path stores the
    second and the operator path the first.
    """
    out: list[str] = []
    for tier in sorted(str(t) for t in (execution_tiers or ())):
        record = (tiers or {}).get(tier)
        if record is None or _tier_field(record, "status") != "pass":
            continue
        if _tier_field(record, "not_applicable", False) or _tier_field(record, "budget_deferred", False):
            continue
        if _tier_field(record, "carried") and not carried_verdict_is_attributed(record, execution_identity):
            continue
        out.append(tier)
    return out


def operator_path_lane_execution(
    capsule: dict,
    *,
    tiers: dict,
    execution_tiers,
    decoded_trace: dict | None,
    execution_identity=None,
) -> dict | None:
    """Per-lane execution counts the OPERATOR path can honestly hand :func:`lane_report`, or ``None``.

    The op tier ladder owns no routing plan and no dispatch ledger, which is why a capsule requiring a
    lane went ungraded there (``LANE_CONTRACT_NOT_EVALUATED``). It does own the two halves a positive
    claim actually needs, and this function credits the accelerator lane only when it holds BOTH:

    * **Something ran.** An oracle tier executed this capsule's emitted program on an instrument and
      passed -- see :func:`executed_program_tiers`. A skipped, deferred or unavailable tier is not an
      execution, and neither is our own L0/L1 interpretation of the command buffer. A verdict CARRIED
      from an earlier run counts only when its certificate is attributed to the same
      ``execution_identity`` as the bytes under grade.
    * **What ran drove the accelerator.** The decoded instruction stream carries instructions of THIS
      target's accelerator classes, counted by :func:`merlin.targetgen.trace_check.accelerator_instruction_count`
      -- membership is the target's own RTL decode table claiming the opcode, never a list written here.

    Neither half alone says anything. A passing oracle tier with no accelerator instruction is a program
    the host computed; accelerator instructions with no oracle tier are a stream nobody executed. Only
    together do they say that the accelerator lane carried work, which is exactly what the required-lane
    assertion claims. With either half missing this returns ``None`` and the capsule stays unmeasured --
    ``lane_report`` is never handed a fabricated count, and a routing plan is never consulted.
    """
    from merlin.targetgen import trace_check as _TCK

    lanes = capsule.get("lanes") or {}
    if _ACCELERATOR_LANE not in [str(x) for x in (lanes.get("require") or ())]:
        return None
    if not executed_program_tiers(tiers, execution_tiers, execution_identity=execution_identity):
        return None
    if not isinstance(decoded_trace, dict):
        return None
    try:
        n = _TCK.accelerator_instruction_count(decoded_trace)
    except Exception:  # noqa: BLE001 -- an undecodable trace measures nothing, and credits nothing
        return None
    return {_ACCELERATOR_LANE: n} if n > 0 else None


def credit_executed_required_lanes(report: dict | None, capsule: dict, lane_execution: dict | None) -> dict | None:
    """Fold executed-lane evidence into ``report`` for the REQUIRED lanes it settles, leaving the rest alone.

    ``report`` is whatever the negative half already produced -- a linked-ELF or declared-program scan,
    which can refute a ``forbid`` and can never credit a ``require``. That half is copied through
    untouched: its ``violated``/``unmeasured_forbidden``/scan blocks are the artifact-level evidence and
    nothing here may weaken them. What is added is the positive half, and only for lanes that
    :func:`lane_report` itself judged on an executed rung, so the evidence vocabulary has one owner.
    """
    if not lane_execution:
        return report
    positive = lane_report(capsule, None, lane_execution=lane_execution)
    if positive is None:
        return report
    credited = [
        ln
        for ln in positive["required"]
        if ln in positive["observed"] and positive["evidence"].get(ln) in EXECUTED_LANE_EVIDENCE
    ]
    if not credited:
        return report
    # Stated beside the report because `judged_by` is a single label for the whole report and this
    # report now has two halves judged on different rungs -- the per-lane `evidence` mapping is the
    # authority, and this says which lanes the new positive rung answered for, and on how much work.
    provenance = {
        "lane_counts": dict(lane_execution),
        "credited_lanes": sorted(credited),
        "source": "executed oracle tier + decoded instruction trace",
    }
    if report is None:
        # Nothing measured the negative half (no scannable artifact). The positive report stands on its
        # own; any forbidden lane in it is still reported at its own rung and stays unjudged.
        out = dict(positive)
        out["execution_evidence"] = provenance
        return out
    out = dict(report)
    out["evidence"] = {**(report.get("evidence") or {}), **{ln: positive["evidence"][ln] for ln in credited}}
    out["observed"] = sorted(set(report.get("observed") or []) | set(credited))
    out["unexercised"] = [ln for ln in (report.get("unexercised") or []) if ln not in credited]
    still_unmeasured = [ln for ln in (report.get("unmeasured_required") or []) if ln not in credited]
    if report.get("unmeasured_required") is not None:
        if still_unmeasured:
            out["unmeasured_required"] = still_unmeasured
        else:
            out.pop("unmeasured_required", None)
            out.pop("caveat", None)
    out["execution_evidence"] = provenance
    return out
