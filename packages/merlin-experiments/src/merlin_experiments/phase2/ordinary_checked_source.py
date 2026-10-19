"""Explicit original-host source witnesses in the existing diagnostic queue.

No placement or timing scope is inferred. Only selected compile/Spike jobs may
use this live owner; their existing DIAGNOSTIC_ONLY result cannot promote an
incumbent or substitute for original Phase 1/formal qualification.
"""

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen.checked_source_model_diagnostic import CheckedSourceModelDiagnostic

from .contracts import StageGateError


def record(selection, passes):
    if selection is None:
        return None
    if type(selection) is not CheckedSourceModelDiagnostic or selection._passes is not passes:
        raise StageGateError("queued source diagnostics require the same original live pass selection")
    with shared_compute_pool():
        return selection.record()


def for_capsule(policy, capsule):
    selection = getattr(policy, "_checked_source_diagnostic", None)
    if selection is None:
        return None
    record(selection, policy._compiler_pass_selection)
    with shared_compute_pool():
        selection.verify_for(capsule, target=policy.target_experiment.target)
    return selection
