"""Normal cheap exploration, separate from measured incumbent promotion.

Only original completed observations enter this projection. Model intervals
order experiments, never provide a winning timing score or a plateau result.
"""

from collections import Counter

from .contracts import StageGateError, document_sha256
from .search_inputs import ORDINARY_FAST_KIND

RESUME_ACTION = "search-resume-parent"


def _signature(cells, arm):
    rows = []
    for cell in cells:
        compiled = cell[arm]["compiled"]
        if compiled.get("status") != "observed":
            return None
        rows.append(
            [
                cell["family"],
                cell["capsule"],
                cell["source_sha256"],
                compiled["elf_sha256"],
                compiled["llvm_mlir_sha256"],
            ]
        )
    return document_sha256(sorted(rows))


def _model_order(coordinator, document, weights):
    policy = coordinator.pool.policy
    owner = getattr(coordinator, "calibration", None) or policy._ordinary_screen
    if owner is None:
        return None, [], "no installed live calibrated screen"
    try:
        screen = owner.screen(policy, document)
    except StageGateError:
        return None, [], "original empirical evidence changed; model ordering UNKNOWN"
    if screen.get("status") != "qualified_public_screening":
        return None, [], screen.get("reason", "screen qualification unavailable")
    evidence = screen["evidence"]
    if {(row["family"], row["capsule"]) for row in evidence} != set(weights):
        raise StageGateError("provisional screening changes its complete public objective")
    savings = [row["conservative_saved_cycles"] for row in evidence]
    score = (
        sum(weights[row["family"], row["capsule"]] * value for row, value in zip(evidence, savings, strict=True))
        if all(value is not None for value in savings)
        else None
    )
    return score, screen["order"], "qualified intervals; exploration ordering only"


def project(coordinator, observations):
    """Reopen original source/result selections and retain declared diversity."""
    pool = coordinator.pool
    if not pool.inputs.ordinary:
        return {"status": "UNAVAILABLE", "candidates": [], "history": []}
    counts = Counter(family for family, _ in coordinator.roster)
    weights = {key: 1 / len(counts) / counts[key[0]] for key in coordinator.roster}
    candidates, history, emitted = [], [], set()
    for job in observations:
        if job["kind"] != ORDINARY_FAST_KIND:
            continue
        revision = pool.revisions.get(job["revision_id"])
        row = {
            "revision_id": revision.revision_id,
            "parent_revision": revision.parent_revision,
            "source_sha256": revision.source_sha256,
            "hypothesis": revision.options.get("hypothesis"),
            "edit_family": revision.options.get("edit_family"),
            "job_id": job["job_id"],
            "state": job["state"],
            "promotion": "NONE",
        }
        if job["state"] != "completed":
            history.append({**row, "outcome": "unmeasured", "reason": job.get("refusal")})
            continue
        document = pool.inputs.validate_result(job["result"], revision, ORDINARY_FAST_KIND)
        cells = document["cells"]
        failures = [
            [cell["family"], cell["capsule"]] for cell in cells if cell["candidate"]["functional_status"] != "PASS"
        ]
        if failures:
            history.append({**row, "outcome": "functional_screen_incomplete_or_failure", "members": failures})
            continue
        current, baseline = _signature(cells, "candidate"), _signature(cells, "baseline")
        if current is not None and current == baseline:
            history.append({**row, "outcome": "unchanged_observed_emission"})
            continue
        if current is not None and current in emitted:
            history.append({**row, "outcome": "duplicate_observed_artifacts", "signature_sha256": current})
            continue
        if current is not None:
            emitted.add(current)
        score, order, basis = _model_order(coordinator, document, weights)
        candidate = {
            **row,
            "outcome": "provisional",
            "signature_sha256": current,
            "conservative_model_saved_cycles": score,
            "model_basis": basis,
            "decisive_member_order": order,
            "collection_wall_seconds": sum(cell["candidate"]["elapsed_seconds"] for cell in cells),
            "collection_cost_scope": "sum of observed member work; concurrent route wall time is separate",
            "emission_status": "changed" if current is not None and baseline is not None else "UNKNOWN",
        }
        candidates.append(candidate)
        history.append(candidate)
    candidates.sort(
        key=lambda row: (
            row["conservative_model_saved_cycles"] is None,
            -(row["conservative_model_saved_cycles"] or 0),
            row["source_sha256"],
        )
    )
    shortlisted, sources, families = [], set(), set()
    for distinct in (True, False):
        for row in candidates:
            if len(shortlisted) >= coordinator.settings.beam_width:
                break
            if row["source_sha256"] in sources or distinct and row["edit_family"] in families:
                continue
            shortlisted.append(row)
            sources.add(row["source_sha256"])
            families.add(row["edit_family"])
    from .ordinary_queue_calibration import summary

    return {
        "status": "provisional_exploration",
        "candidates": shortlisted,
        "history": history,
        "ranking_authority": "NONE",
        "diversity_basis": "declared edit families; no exhausted-search claim",
        "required_promotion": "complete correct comparable measured public roster",
        "calibration": summary(coordinator)
        if hasattr(coordinator, "calibration_partition")
        else {"status": "NOT_INSTALLED", "authority": "NONE"},
    }


def resume(coordinator, revision_id):
    """Restore an issued parent while retained evaluation workers stay immutable."""
    coordinator.admit_submission(timeout_s=1)
    if coordinator.selected is not None:
        raise StageGateError("provisional parent selection cannot reopen a finalized compiler")
    revision = coordinator.pool.revisions.get(revision_id)
    answer = coordinator.pool.revisions.restore(revision.revision_id, coordinator.pool.policy.candidate)
    return {"parent": revision.public_record(), "restoration": answer, "promotion": "NONE"}
