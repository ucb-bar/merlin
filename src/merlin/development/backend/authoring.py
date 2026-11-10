"""Continue one admitted author lifecycle without changing its evidence authority."""

from __future__ import annotations

from typing import TYPE_CHECKING

from merlin.development.backend.authoring_certification import run_certification
from merlin.development.backend.authoring_completion import complete
from merlin.development.backend.authoring_config import AuthoringRuntime as AuthoringRuntime
from merlin.development.backend.authoring_rounds import run_legacy_continuous, run_rounds
from merlin.development.backend.authoring_session import AuthoringSession

if TYPE_CHECKING:
    from merlin.development.backend.session import PreparedRun


def execute(prepared: PreparedRun, runtime: AuthoringRuntime) -> int:
    """Keep the caller's admitted workspace lease alive through every lifecycle stage."""
    session = AuthoringSession(prepared, runtime)
    if session.options.continuous:
        return run_legacy_continuous(session)
    early_exit = run_rounds(session)
    if early_exit is not None:
        return early_exit
    run_certification(session)
    return complete(session)
