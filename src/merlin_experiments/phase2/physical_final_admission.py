"""Fail closed until a real independent physical final-domain issuer exists.

There is presently no complete live issuer for selected RTL-to-loaded hardware,
runtime/loading/reset/clock semantics and physical complete timers. Independent
functional roles and component screening roles do not supply that authority.
This owner accepts no replacement callback, metadata, saved PASS report or hash.

The future issuer must execute and reopen independently sourced controls, bind
the actual selected HW/runtime/toolchain/timer domain, and join both arm products
to the exact consumed ELF and complete original input/output roster. Its own
source/tool/product/invocation evidence and domain membership must be rechecked
at admission. Private verifier callbacks and numerical checks remain additional
gates; their observation-only qualification cannot discharge this requirement.
"""

from .contracts import StageGateError


def admit_protected_final_comparison(
    *,
    binding,
    original_binding,
    verifier,
    run_dir,
    environment,
    original_budget,
    budget,
    reference,
    candidate,
    reference_build,
    candidate_build,
    original_reference,
    original_reference_build,
    physical_execution_domain=None,
):
    """Refuse production admission without a genuinely issued physical owner.

    The complete numerical/witness lifecycle is available separately through
    observe_protected_final_comparison. No declaration can enable this route:
    implementing and evaluating the missing issuer is still required work.
    """
    raise StageGateError(
        "physical final admission UNKNOWN: no independently qualified physical execution-domain issuer "
        "is implemented; observation qualifications and supplied declarations cannot authorize hardware"
    )
