"""Explicit original-host source witnesses in the existing diagnostic queue.

No placement or timing scope is inferred. Only selected compile/Spike jobs may
use this live owner; their existing DIAGNOSTIC_ONLY result cannot promote an
incumbent or substitute for original Phase 1/formal qualification.
"""

import math
from time import monotonic

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.checked_source_model_diagnostic import CheckedSourceModelDiagnostic

from .contracts import StageGateError

METADATA_QUEUE_SECONDS = 120


def _selection(selection, passes):
    if type(selection) is not CheckedSourceModelDiagnostic or selection._passes is not passes:
        raise StageGateError("queued source diagnostics require the same original live pass selection")
    return selection


def _deadline(deadline):
    now = monotonic()
    if deadline is None:
        return now + METADATA_QUEUE_SECONDS
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= now:
        raise StageGateError("queued source metadata exhausted its original aggregate deadline")
    # The caller owns the complete operation budget, including qualification.
    # The separate queue wait ceiling must not become another execution deadline.
    return deadline


def _queue_wait(deadline):
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise StageGateError("queued source metadata exhausted its original aggregate deadline")
    return min(remaining, METADATA_QUEUE_SECONDS)


def _completed(deadline):
    if monotonic() > deadline:
        raise StageGateError("queued source metadata exhausted its original aggregate deadline")


def record(selection, passes, *, deadline=None):
    if selection is None:
        return None
    _selection(selection, passes)
    deadline = _deadline(deadline)
    with shared_compute_pool(timeout_s=_queue_wait(deadline)):
        result = selection.record()
    _completed(deadline)
    return result


def for_capsule(policy, capsule, *, deadline=None):
    selection = getattr(policy, "_checked_source_diagnostic", None)
    if selection is None:
        return None
    _selection(selection, policy._compiler_pass_selection)
    deadline = _deadline(deadline)
    with shared_compute_pool(timeout_s=_queue_wait(deadline)):
        # verify_for already revalidates the exact live selector and its owner.
        selection.verify_for(capsule, target=policy.target_experiment.target)
    _completed(deadline)
    return selection
