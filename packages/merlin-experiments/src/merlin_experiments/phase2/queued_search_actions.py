"""Broker controls for the normal corpus queue; original final receipts still apply."""

import time

from . import component_workflow as controls
from . import ordinary_measurement_selection as SELECTION
from . import ordinary_queue_calibration as CALIBRATION
from .broker import BrokerAction
from .contracts import StageGateError, canonical_json
from .provisional_search import RESUME_ACTION
from .search_inputs import ORDINARY_FAST_KIND, ORDINARY_KIND


def actions():
    pool = tuple(
        BrokerAction(
            action.name,
            ("__host_corpus_search_pool__", action.name),
            action.placeholders,
            (
                f"Queue retained revision: kind={ORDINARY_FAST_KIND} observes compilation and Spike; "
                f"kind={ORDINARY_KIND} measures certified RTL. Cheap screening cannot select the final source."
                if action.name == controls.EVALUATION_SUBMIT_ACTION
                else action.purpose
            ),
            False,
        )
        for action in controls.evaluation_pool_actions(search=True)
    )
    return (
        *pool,
        *controls.search_actions(),
        BrokerAction(
            SELECTION.ACTION,
            ("__host_corpus_search_pool__", SELECTION.ACTION),
            ("revision_id", "members"),
            "Queue diagnostic RTL for public family/capsule members; partial timing cannot promote",
            False,
        ),
        BrokerAction(
            RESUME_ACTION,
            ("__host_corpus_search_pool__", RESUME_ACTION),
            ("revision_id",),
            "Restore an issued exploration parent; serialize author edits first; measured incumbent is unchanged",
            False,
        ),
        BrokerAction(
            CALIBRATION.ACTION,
            ("__host_corpus_search_pool__", CALIBRATION.ACTION),
            ("revision_ids",),
            "Fit and validate from completed own cheap/RTL jobs; no simulator launch or promotion",
            False,
        ),
    )


def execute(policy, action_name, rendered, timeout_s, started):
    """Return transport data only; selection requires the original final feedback call."""
    policy.validate_search_inputs()
    coordinator, pool = policy.search_coordinator, policy.evaluation_pool
    if coordinator is None or pool is None:
        raise StageGateError("normal search controls require their installed queue and coordinator")
    if action_name == controls.SEARCH_STATUS_ACTION:
        answer = coordinator.status()
    elif action_name == controls.SEARCH_FINALIZE_ACTION:
        answer = coordinator.finalize(timeout_s=timeout_s)
    elif action_name == controls.SEARCH_SELECT_ACTION:
        answer = coordinator.select()
    elif action_name == RESUME_ACTION:
        answer = coordinator.resume_parent(rendered["revision_id"])
    elif action_name == CALIBRATION.ACTION:
        answer = CALIBRATION.install(coordinator, rendered["revision_ids"], timeout_s=timeout_s)
    elif action_name == SELECTION.ACTION:
        coordinator.admit_submission(timeout_s=timeout_s)
        members = SELECTION.parse_members(policy, rendered["members"])
        answer = pool.submit(
            rendered["revision_id"], kind=SELECTION.KIND, selected_members=members, timeout_s=timeout_s
        )
    elif action_name == controls.REVISION_FREEZE_ACTION:
        parent = rendered["parent_revision"]
        answer = pool.revisions.freeze(
            policy.candidate,
            parent_revision=None if parent == "none" else parent,
            options={key: rendered[key] for key in ("hypothesis", "edit_family")},
        ).public_record()
    elif action_name == controls.EVALUATION_SUBMIT_ACTION:
        coordinator.admit_submission(timeout_s=timeout_s)
        answer = pool.submit(rendered["revision_id"], kind=rendered["kind"], timeout_s=timeout_s)
    elif action_name == controls.EVALUATION_STATUS_ACTION:
        answer = pool.status(rendered["job_id"])
    elif action_name == controls.EVALUATION_CANCEL_ACTION:
        answer = pool.cancel(rendered["job_id"])
    else:
        raise StageGateError("normal search action is outside its fixed controls")
    if time.monotonic() - started > timeout_s:
        raise StageGateError("normal search action exceeded its original tool budget")
    return (
        {
            "returncode": 0,
            "stdout": canonical_json(answer).decode(),
            "stderr": "",
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        None,
    )
