"""Continue the original author thread until automatic search finalization.

The existing broker/coordinator owns evaluation, source selection and receipts.
This transport control issues no grade or performance qualification.
"""

import time
from datetime import UTC, datetime
from pathlib import Path

from merlin.benchharness import hash_tree

from .contracts import StageGateError
from .search_guidance import ADAPTIVE_SCOPE
from .transcript_audit import audit_codex_transcript

CONTINUATION = (
    "Continue this Phase2 campaign in the same workspace and thread. Read search-status. "
    "While exploration remains open, implement and measure the next hypothesis; assign helpers "
    "and focused measurements in roofline_priorities.next order, lowest worst-repeat attainment "
    "first. Keep unknown bounds separate and preserve numerical gates. A completed turn is not "
    "campaign completion. Finish one hypothesis, frozen-submission or feedback checkpoint, await "
    "your native tool calls, then emit a concise FINAL turn message. The driver automatically "
    "resumes this same thread; returning a turn does not finalize the search. Read the fixed "
    "search_session deadlines in STAGE_CONTEXT.json and leave time to return before the author cutoff. "
    + ADAPTIVE_SCOPE
    + "When the declared target or a budget boundary stops exploration, join "
    "helpers, stop edits, invoke search-finalize, then search-select when a complete measured "
    "selection exists. Run all original required feedback and analysis on those exact bytes. "
    "Retain diagnostic-only sources and explicit incomplete results when no measured selection "
    "exists. Do not finalize early or ask an operator to choose each iteration."
)


def timing_projection(settings):
    """Describe the fixed automatic campaign cutoffs before author dispatch."""
    if not settings.continue_author_session:
        return None
    settings.verify()
    now, wall = time.monotonic(), time.time()
    hard, exploration = search_deadlines(settings, now)
    cutoffs = {
        "exploration": exploration,
        "author_return": hard - settings.final_reserve_seconds,
        "campaign": hard,
    }
    return {
        "deadlines": {
            name: {"unix": wall + cutoff - now,
                   "utc": datetime.fromtimestamp(wall + cutoff - now, UTC).isoformat(),
                   "remaining_seconds": max(0, int(cutoff - now))}
            for name, cutoff in cutoffs.items()
        },
        "turn_policy": "Finish one hypothesis/submission/feedback checkpoint, await native tool calls, "
                       "then emit a FINAL turn message. The driver automatically resumes the same thread. "
                       "Invoke search-finalize only at the campaign target or exploration boundary; "
                       "finish required final feedback and return before author_return.",
    }


def stage_deadline(settings, wall_seconds):
    """Clip the local stage to the launcher's immutable absolute campaign cap."""
    now = time.monotonic()
    limit = now + wall_seconds
    external = None if settings is None else settings.campaign_deadline_unix
    return min(limit, now + max(0, external - time.time())) if external is not None else limit


def require_stage_budget(settings, wall_seconds, round_seconds):
    required = settings.hard_seconds if settings.continue_author_session else settings.soft_seconds
    if wall_seconds < required or round_seconds < required:
        raise StageGateError("queued corpus search must retain its declared exploration and final reserve")


def search_deadlines(settings, started):
    hard = started + settings.hard_seconds
    if settings.campaign_deadline_unix is not None:
        hard = min(hard, stage_deadline(settings, settings.hard_seconds))
    reserve = settings.final_reserve_seconds
    soft = started + settings.soft_seconds - reserve
    if settings.continue_author_session:
        # One reserve for author-owned drain/selection, one for post-author checks.
        soft = min(soft, hard - 2 * reserve)
    return hard, min(soft, hard - reserve)


def driver_options(broker, *, stage_root, round_index):
    """Bind continuation to the actual original live broker, never saved status."""
    from .broker import Broker
    from .search_coordinator import ComponentSearchCoordinator

    if type(broker) is not Broker or type(broker.workflow.search_coordinator) is not ComponentSearchCoordinator:
        raise StageGateError("automatic continuation requires the original search broker and coordinator")
    coordinator = broker.workflow.search_coordinator
    if not coordinator.settings.continue_author_session:
        raise StageGateError("automatic continuation was not selected in the frozen search settings")
    transcript = Path(stage_root) / "rounds" / f"round_{round_index:02d}.transcript.jsonl"

    def stop():
        status = coordinator.status()
        if not status["finalized"]:
            return None  # A target/budget boundary still needs an author finalization turn.
        if status["selected"] is None:
            if status["ranked"]:
                return None  # Resume to select within the original observed author lineage.
            return "search finalized without complete measured selection; campaign incomplete"
        from .retained_revisions import source_inventory

        store = coordinator.pool.revisions
        selected = store.get(status["selected"]["revision_id"])

        def selected_bytes():
            return source_inventory(
                broker.candidate, max_bytes=store.max_bytes, max_files=store.max_files,
                exclude_generated=True,
            ) == selected.inventory

        if not selected_bytes():
            return None  # Resume to restore the measured selection before required feedback.
        actions = tuple(broker.actions.values())
        audit = audit_codex_transcript(transcript, broker.target_experiment, broker.candidate, actions)
        if not audit["clean"]:
            return "search author audit refused; campaign incomplete"
        try:
            evidence = broker.workflow.verify_receipts(
                broker.receipt_path, actions, audit,
                candidate_sha256=hash_tree(broker.candidate)["sha256"],
            )
        except StageGateError:
            return None  # Let the same author finish the original required feedback.
        return (
            "automatic search finalization complete"
            if evidence.get("all_required_succeeded") and selected_bytes() else None
        )

    return dict(continue_session=True, continuation_prompt=CONTINUATION, continuation_stop=stop)
