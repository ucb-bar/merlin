"""Source lookup and explicit synthetic inputs for cross-subsystem feedback tests.

This locates authoritative implementations; it does not wrap functions, inject defaults
into implementations, or make retired native imports work.
"""

from pathlib import Path

from merlin.common.paths import checkout_root, module_source_path
from merlin.experiments.phase1.context import InvocationContext

MODULES = {
    "loop_grading": "merlin.experiments.phase1.feedback.loop_grading",
    "grade_agent_run": "merlin.experiments.phase1.feedback.formal",
    "freeze_run": "merlin.experiments.phase1.feedback.freeze",
    "qa_check": "merlin.experiments.phase1.feedback.qa",
    "agent_selfcheck": "merlin.experiments.phase1.feedback.selfcheck",
    "tier_promote": "merlin.experiments.phase1.feedback.promotion",
    "selfcheck_broker": "merlin.experiments.phase1.brokers.selfcheck",
    "simjob_broker": "merlin.experiments.phase1.brokers.simjob",
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
