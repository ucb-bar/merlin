"""Explicit empty implementation origin; sandbox input grants remain separate."""

from pathlib import Path


def validate(options) -> None:
    if getattr(options, "joint_epoch", ""):
        from .joint_epochs import selected
        selected(options)
    if not getattr(options, "fresh_backend", False):
        return
    if options.seed_submission or options.qualify_submission:
        raise ValueError("fresh backend requires an empty implementation, without a seed or qualification submission")
    if options.sandbox != "bwrap" or options.allow_unsandboxed:
        raise ValueError("fresh backend requires the isolated author workspace")


def origin(options, workspace: Path, *, prior: dict | None = None) -> dict | None:
    """Observe after trusted staging; resume retains the original empty-start fact."""
    validate(options)
    selected = getattr(options, "fresh_backend", False)
    if prior is not None:
        recorded = prior.get("backend_origin")
        if selected != (recorded is not None):
            raise ValueError("fresh backend selection differs from the original run")
        if recorded is not None and recorded != {"schema": "merlin.empty_backend_origin.v1", "initial_files": 0}:
            raise ValueError("fresh backend origin record changed")
        return recorded
    if not selected:
        return None
    if getattr(options, "joint_epoch", ""):
        from . import joint_epochs, run_inputs
        record = joint_epochs.selected(options)
        _, inherited = joint_epochs._verify(record)
        if run_inputs._seed_content(workspace / "submission") != run_inputs._seed_content(Path(record["source"])):
            raise ValueError("joint epoch staged compiler differs from its frozen source")
        return inherited
    submission = workspace / "submission"
    if submission.is_symlink() or not submission.is_dir() or any(submission.iterdir()):
        raise ValueError("fresh backend submission must be an existing empty directory before author launch")
    return {"schema": "merlin.empty_backend_origin.v1", "initial_files": 0}
