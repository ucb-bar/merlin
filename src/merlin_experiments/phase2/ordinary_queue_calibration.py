"""Install a live empirical screen from this normal queue's completed products.

No new simulator is launched. The partition is frozen before observations, and
all executable alternatives of a public member stay in the same group. Saved
reports and author-supplied measurements cannot construct this host owner.
"""

import json
import time

from . import ordinary_calibration as CAL
from .contracts import StageGateError, document_sha256, sha256_file
from .search_inputs import ORDINARY_FAST_KIND, ORDINARY_KIND

ACTION = "search-calibrate"
MAX_INSTALLATIONS = 3


def partition(policy):
    members = sorted(
        policy.validate_search_inputs(),
        key=lambda key: document_sha256([policy.search_corpus.manifest_sha256, list(key)]),
    )
    if len(members) < 2:
        return None
    middle = len(members) // 2
    return {
        "schema": CAL.PARTITION_SCHEMA,
        "calibration": [list(key) for key in members[:middle]],
        "held_validation": [list(key) for key in members[middle:]],
    }


def summary(coordinator):
    owner = coordinator.calibration or getattr(coordinator.pool.policy, "_ordinary_screen", None)
    if owner is None:
        return {
            "status": "NOT_INSTALLED",
            "authority": "NONE",
            "partition": json.loads(coordinator.calibration_partition) if coordinator.calibration_partition else None,
            "installations_remaining": MAX_INSTALLATIONS - coordinator.calibration_installations,
        }
    try:
        owner.verify(coordinator.pool.policy)
    except StageGateError:
        return {"status": "UNKNOWN", "authority": "NONE",
                "reason": "original empirical evidence changed; complete measured selection remains independent"}
    from .ordinary_archive_screen import OrdinaryArchiveScreen

    if type(owner) is OrdinaryArchiveScreen:
        return owner.summary()
    report = json.loads(owner.report_json)
    return {
        "status": "qualified_public_screening" if report["exposable"] else "UNKNOWN",
        "calibration_sha256": owner.sha256,
        "reasons": report["reasons"],
        "absolute_error": report["absolute_error"],
        "interval_coverage": report["interval_coverage"],
        "ranking": report["ranking"],
        "policy": report["policy"],
        "authority": "exploration ordering only",
        "installation_wall_seconds": coordinator.calibration_wall_seconds,
        "installations_remaining": MAX_INSTALLATIONS - coordinator.calibration_installations,
        "screening_net_benefit": "UNKNOWN; measure complete route costs and retained RTL fraction",
    }


def install(coordinator, revision_ids, *, timeout_s):
    """Consume live original completed cheap/RTL jobs, retaining exact evidence pins."""
    from .evaluation_pool import TERMINAL, ComponentEvaluationPool

    coordinator.admit_submission(timeout_s=timeout_s)
    pool, policy = coordinator.pool, coordinator.pool.policy
    if type(pool) is not ComponentEvaluationPool or not pool.inputs.ordinary:
        raise StageGateError("queued calibration requires the original normal issued pool")
    if type(revision_ids) is not str or len(revision_ids) > 8192:
        raise StageGateError("queued calibration needs bounded issued revision identities")
    ids = revision_ids.split(",")
    if not 2 <= len(ids) <= 64 or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise StageGateError("queued calibration needs two or more distinct issued alternatives")
    if coordinator.calibration_partition is None or coordinator.calibration_installations >= MAX_INSTALLATIONS:
        raise StageGateError("queued calibration has no public partition or installation budget")
    selected_partition = json.loads(coordinator.calibration_partition)
    CAL.public_partition(policy, selected_partition)
    started, samples, artifacts = time.monotonic(), [], []
    # Parent fitting owns no simulation slot and starts no process. Serialize
    # queue changes so the source and complete result selections stay fixed.
    with pool._lock:
        observations = pool.observations()
        if any(row["state"] not in TERMINAL for row in observations):
            raise StageGateError("finish or cancel outstanding jobs before installing calibration")
        for revision_id in ids:
            revision = pool.revisions.get(revision_id)
            jobs = {}
            for kind in (ORDINARY_FAST_KIND, ORDINARY_KIND):
                matches = [
                    row
                    for row in observations
                    if row["revision_id"] == revision_id and row["kind"] == kind and row["state"] == "completed"
                ]
                if not matches:
                    raise StageGateError("calibration alternative lacks complete original cheap and RTL jobs")
                # Deterministic original latest replicate; prior replicates are
                # not averaged or used to conceal disagreement.
                row = matches[-1]
                job = pool._get(row["job_id"])
                result = pool.inputs.validate_result(row["result"], revision, kind)
                jobs[kind] = (job, result)
                artifacts.append((str(job.root / "feedback.json"), sha256_file(job.root / "feedback.json")))
            cheap_job, cheap = jobs[ORDINARY_FAST_KIND]
            timing_job, timing = jobs[ORDINARY_KIND]
            values, pins = CAL.collect_sample(
                policy,
                cheap_root=cheap_job.root / "rtl",
                timing_root=timing_job.root / "rtl",
                cheap=cheap,
                timing=timing,
                revision=revision,
                baseline_revision=pool.baseline_revision,
                partition=selected_partition,
            )
            samples.extend(values)
            artifacts.extend(pins)
            if time.monotonic() - started > timeout_s:
                raise StageGateError("queued calibration exceeded its original broker budget")
        root = pool.root / "calibrations" / str(coordinator.calibration_installations)
        if root.exists() or root.resolve() != root:
            raise StageGateError("queued calibration output must be fresh and private")
        root.mkdir(parents=True, mode=0o700)
        coordinator.calibration_installations += 1
        owner = CAL.issue_from_samples(
            policy, partition=selected_partition, samples=samples, artifacts=artifacts, root=root
        )
        if time.monotonic() - started > timeout_s:
            raise StageGateError("queued calibration cannot install a late result")
        if pool._retained_output_bytes() > pool.capacity.disk_bytes:
            raise StageGateError("queued calibration exceeded retained output capacity")
        owner.verify(policy)
        coordinator.calibration = owner
        coordinator.calibration_wall_seconds = time.monotonic() - started
    return summary(coordinator)
