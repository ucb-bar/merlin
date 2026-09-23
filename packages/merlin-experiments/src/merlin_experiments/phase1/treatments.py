"""Explicit per-invocation treatment choices for the retained native QA controller.

Callbacks remain trusted host code. Their identity refers to the existing implementation
inventory, not a new source seal or a claim about arbitrary in-memory monkeypatches.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Literal, Protocol

from merlin.common.digest import sha256_file


class TaskStager(Protocol):
    def __call__(self, arm: str, ws: Path, run_dir: Path, *, sandbox: str, bundle_dir: Path) -> None: ...


class QARunner(Protocol):
    def __call__(
        self,
        submission: str,
        capsules_root: str,
        runs_root: Path,
        labels: set[str],
        no_oracle: bool,
        timeout: int,
        *,
        contract: Path | None = None,
        additional_forbidden: tuple[str, ...] = (),
    ) -> dict: ...


DurationSink = Callable[[Literal["agent", "qa"], float], None]


class CheckpointFeedback(Protocol):
    def __call__(self, runs_root: Path, *, capsule_roots: tuple[Path, ...]) -> list[dict]: ...


@dataclass(frozen=True)
class Treatment:
    name: str = "baseline"
    capsules_root: Path | None = None
    stage_task: TaskStager | None = None
    qa_runner: QARunner | None = None
    checkpoint_feedback: CheckpointFeedback | None = None
    on_duration: DurationSink | None = None


def callback_reference(callback: Callable, implementation_sources: dict, *, label: str) -> dict:
    """Attribute a host callback to existing source evidence, not arbitrary closure state."""
    inputs = implementation_sources["inputs"]
    if not inspect.isfunction(callback):
        raise ValueError(f"{label} must be an inventoried source function")
    filename = inspect.getsourcefile(callback)
    if filename is None:
        raise ValueError(f"{label} has no source owner")
    source = Path(filename).resolve()
    keys = sorted(key for key, row in inputs.items() if Path(row["path"]).resolve() == source)
    if not keys or not source.is_file():
        raise ValueError(f"{label} is outside the implementation inventory")
    key = keys[0]
    if sha256_file(source) != inputs[key]["sha256"]:
        raise ValueError(f"{label} source changed")
    return {"module": callback.__module__, "qualname": callback.__qualname__, "source_input": key}


def record(treatment: Treatment, implementation_sources: dict) -> dict:
    """Refer to inventoried callback owners, refusing foreign or changed implementation."""
    if not isinstance(treatment.name, str) or not treatment.name.strip():
        raise ValueError("treatment name must be nonempty")
    # Preserve the original eager inventory-shape check even with no callbacks.
    implementation_sources["inputs"]
    callbacks = {}
    for field in ("stage_task", "qa_runner", "checkpoint_feedback", "on_duration"):
        callback = getattr(treatment, field)
        if callback is None:
            callbacks[field] = None
            continue
        callbacks[field] = callback_reference(callback, implementation_sources, label=f"treatment {field}")
    return {
        "name": treatment.name,
        "capsules_root": str(treatment.capsules_root.resolve()) if treatment.capsules_root is not None else None,
        "callbacks": callbacks,
    }


def timed(function: Callable, kind: Literal["agent", "qa"], sink: DurationSink | None) -> Callable:
    """Observe successes/failures; sink errors propagate, as in the legacy finally block."""
    if sink is None:
        return function

    @wraps(function)
    def invoke(*args, **kwargs):
        started = time.time()
        try:
            return function(*args, **kwargs)
        finally:
            sink(kind, round(time.time() - started, 3))

    return invoke
