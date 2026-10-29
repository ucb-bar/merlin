"""Whole-model rows on the Spike-only final route take their L3 from an exact-byte certificate.

The final route runs L0-L2 on Spike and never RTL, so a whole-model row whose only open obligation is
its native L3 tier (``joint_physical_correctness.model_l3_prerequisites``) can never close there. It
used to be filed as an unexplained ``skipped`` row and, through the cohort veto, kept every native
core pass withheld. Each such row is now closed one of two explicit ways:

* ``pass`` with the newest passing L3 certificate the promotion ledger holds for exactly these bytes
  -- the frozen compiler's whole-submission digest (or the row's own execution identity), recorded
  under the instrument a promotion would use today. Nothing is re-attributed: a certificate for
  other bytes or another judge is not evidence here.
* ``l3_evidence_missing``, with the reason, when no such certificate exists. The row is not passed
  and the group cannot pass, but it is reported as what it is rather than as unfinished work.
"""

from __future__ import annotations

from pathlib import Path

MISSING = "l3_evidence_missing"
#: Without continuous promotion (``joint_control.continuous_l3_promotion``) L3 is best-effort: a row
#: with no exact-byte certificate keeps its L0-L2 Spike verdict and says plainly that L3 did not run.
NOT_RUN_REASON = "L3 not run: best-effort final certification; L0-L2 passed on Spike for the frozen compiler"
_TIER = "L3"


def _result(runs: Path, name: str) -> tuple[Path | None, dict | None]:
    import json

    paths = sorted(Path(runs).glob(f"runs/*/{name}/capsule_result.json"))
    if len(paths) != 1:
        return None, None
    try:
        return paths[0], json.loads(paths[0].read_text())
    except (OSError, ValueError):
        return paths[0], None


def _certificate(workspace, name, *, digest, execution, instrument):
    from .feedback import promotion as P

    found = None
    for identity, entry in P._slots_ro(P._tier_state(workspace), name, _TIER).items():
        if (isinstance(entry, dict) and entry.get("status") == "pass" and entry.get("instrument") == instrument
                and (entry.get("digest") == digest
                     or (execution is not None and entry.get("execution_digest") == execution))):
            found = {"identity": identity, "source_digest": entry.get("digest"),
                     "execution_digest": entry.get("execution_digest"), "instrument": instrument,
                     "origin": entry.get("origin", "promotion"), "ledger": "qa/tier_state.json"}
    return found  # ledger order is recording order, so the last match is the newest


def close(score, *, runs, workspace, frozen, compiler_sha256, context) -> dict:
    """``{name: {"certificate": ...} | {"missing": reason}}`` for every model row with only an L3 gap."""
    from .feedback import promotion as P
    from .feedback import qa
    from .feedback.joint_physical_correctness import model_l3_prerequisites

    gaps = {}
    for row in score.get("per_capsule") or ():
        name = row.get("capsule") if isinstance(row, dict) else None
        if not name or row.get("kind") != "model" or row.get("status") != "incomplete":
            continue
        path, result = _result(runs, name)
        if result is not None and model_l3_prerequisites(result, compiler_sha256):
            gaps[name] = path
    if not gaps:
        return {}
    try:
        instrument = P.cert_instrument(_TIER, context=context)
        digest = P.submission_tree_digest(frozen)
    except (OSError, ValueError):
        instrument = digest = None
    from . import joint_control

    closed = {}
    for name, path in gaps.items():
        certificate = None
        if instrument is not None and digest is not None:
            certificate = _certificate(workspace, name, digest=digest, instrument=instrument,
                                       execution=qa._execution_digest_from_result(path))
        if not certificate and not joint_control.continuous_l3_promotion():
            closed[name] = {"not_run": NOT_RUN_REASON}
            continue
        closed[name] = {"certificate": certificate} if certificate else {"missing": (
            "L3 evidence missing: L0-L2 passed on Spike for the frozen compiler, and no passing L3 "
            "certificate for exactly these bytes is recorded"
            + ("" if instrument is not None else " (the L3 certificate instrument is unavailable)"))}
    return closed


def cohort_score(score, certified) -> dict:
    """``score`` with the certified model rows closed, for the cohort-completeness predicate only."""
    if not certified:
        return score
    closed = dict(score)
    closed["per_capsule"] = [dict(row, status="pass") if row.get("capsule") in certified else row
                             for row in score.get("per_capsule") or ()]
    for field, delta in (("n_passed", len(certified)), ("n_incomplete", -len(certified))):
        if type(score.get(field, 0)) is int:
            closed[field] = score.get(field, 0) + delta
    # The native grader withholds ``gradeable`` while a mandatory tier is unavailable. When the
    # certificates close every such row, and nothing else (an empty suite, an infrastructure fault,
    # a withheld oracle) kept it false, the closed score is gradeable.
    if (score.get("gradeable") is not True and closed.get("n_incomplete") == 0
            and type(score.get("n_incomplete")) is int and score["n_incomplete"] > 0
            and type(score.get("n_capsules")) is int and score["n_capsules"] > 0
            and score.get("n_infrastructure_fault", 0) == 0 and score.get("n_not_gradeable_no_oracle", 0) == 0):
        closed["gradeable"] = True
        closed.pop("measurement_incomplete", None)
    return closed
