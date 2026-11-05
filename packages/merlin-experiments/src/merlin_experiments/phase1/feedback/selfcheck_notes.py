"""The note a self-check shows when the author's package ran no test case, chosen by failure plane.

Every package failure used to carry the same note about cmake builds. An integrity refusal (for
example an editable module under a namespace the full-compiler selection does not contain) is not a
build problem, and the build note sent the author looking at cmake instead of the rejected path.
"""

from __future__ import annotations

from . import package_rejection

BUILD_NOTE = (
    "Your package did NOT build/scan, so ZERO capsules ran — this is NOT a stubbed "
    "grader. Fix the build first (see 'detail' for the cmake/compiler stderr). NOTE: "
    "the grade builds an ISOLATED COPY of your package WITHOUT the build/ tree, so your "
    "manifest 'build' must support a CLEAN configure (provide a 'configure' step, e.g. "
    "cmake -S <src> -B <build>, not only 'cmake --build'); build commands run from the "
    "package root, so relative paths and $MLIR_DIR/$LLVM_DIR/$CM env vars are honored."
)

INTEGRITY_NOTE = (
    "Your package was REFUSED by its integrity scan before any build, so ZERO capsules ran — this is "
    "not a build or grader failure. 'detail' names what the scan refused"
)


def package_failure_note(failure) -> str:
    """The note for a package failure: integrity refusals explain the rule, build failures the build."""
    plane = str((failure or {}).get("plane") or "") if isinstance(failure, dict) else ""
    if plane != "integrity":
        return BUILD_NOTE
    hint = package_rejection.namespace_hint(failure)
    if not hint:
        return INTEGRITY_NOTE + "; remove or change it and re-run the check."
    allowed = hint.get("allowed_namespaces")
    return (INTEGRITY_NOTE + f": {hint['rejected_path']}. {hint['rule']}"
            + (f" Allowed: {', '.join(allowed)}." if allowed else ""))
