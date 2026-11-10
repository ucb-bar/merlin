"""Source lookup and explicit synthetic inputs for cross-subsystem feedback tests.

This locates authoritative implementations; it does not wrap functions, inject defaults
into implementations, or make retired native imports work.
"""

from pathlib import Path

from merlin.common.paths import checkout_root, module_source_path
from merlin.development.backend.context import InvocationContext

MODULES = {
    "loop_grading": "merlin.development.backend.feedback.loop_grading",
    "grade_agent_run": "merlin.development.backend.feedback.formal",
    "freeze_run": "merlin.development.backend.feedback.freeze",
    "qa_check": "merlin.development.backend.feedback.qa",
    "agent_selfcheck": "merlin.development.backend.feedback.selfcheck",
    "tier_promote": "merlin.development.backend.feedback.promotion",
    "selfcheck_broker": "merlin.development.backend.brokers.selfcheck",
    "simjob_broker": "merlin.development.backend.brokers.simjob",
}


def feedback_source(name: str, fallback: Path | None = None) -> Path:
    if name in MODULES:
        return module_source_path(MODULES[name])
    if fallback is None:
        raise KeyError(name)
    return fallback


def feedback_context(root: Path | None = None, *, target: str = "fixture") -> InvocationContext:
    root = Path(root) if root is not None else checkout_root() / "tests/fixtures/absent_feedback"
    return InvocationContext(
        repo=checkout_root(),
        descriptor=root / "target_experiment.yaml",
        experiment=root,
        target=target,
        runs=root / "runs",
        reports=root / "reports",
        bundles=root / "bundles",
        sourced_environment=(),
    )
