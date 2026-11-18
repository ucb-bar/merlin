"""Human-readable Phase 1 experiment levels over stable, frozen arm identities.

Levels name comparisons, not compiler capability or an ordering of quality.  In
particular, the optional verification arm branches from EL4.  Existing bundle ids, treatment keys and frozen runs retain
their original machine-readable names.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Literal, Protocol

from merlin.common.digest import sha256_file

# ---- levels ------------------------------------------------------------------------------------

LEVELS = (
    {
        "id": "EL1",
        "name": "Raw baseline",
        "bundle_arm": "raw_baseline",
        "runner_arm": "raw_baseline",
        "treatment": "baseline",
        "parent": None,
    },
    {
        "id": "EL2",
        "name": "C++ infrastructure",
        "bundle_arm": "cpp_merlininfra",
        "runner_arm": "cpp_merlininfra",
        "treatment": "baseline",
        "parent": "EL1",
    },
    {
        "id": "EL3",
        "name": "Merlin-assisted",
        "bundle_arm": "merlin_assisted",
        "runner_arm": "merlin_assisted",
        "treatment": "baseline",
        "parent": "EL2",
    },
    {
        "id": "EL4",
        "name": "RTL-informed Merlin",
        "bundle_arm": "merlin_rtlchecks",
        "runner_arm": "merlin_assisted",
        "treatment": "rtlchecks",
        "parent": "EL3",
    },
    {
        "id": "EL4-V",
        "name": "RTL + verification tools (baseline feedback)",
        "bundle_arm": "merlin_verify",
        "runner_arm": "merlin_assisted",
        "treatment": "baseline",
        "parent": "EL4",
    },
    {
        "id": "EL5",
        "name": "RTL + ISA-derived instruction dialect",
        "bundle_arm": "merlin_isa",
        "runner_arm": "merlin_assisted",
        "treatment": "rtlchecks",
        "parent": "EL4",
    },
    {
        "id": "EL6",
        "name": "ISA generation + interactive Lean proof tooling",
        "bundle_arm": "merlin_lean",
        "runner_arm": "merlin_assisted",
        "treatment": "rtlchecks",
        "parent": "EL5",
    },
)

_BY_ARM = {level["bundle_arm"]: level["id"] for level in LEVELS}

_BY_ID = {level["id"]: level for level in LEVELS}


def materialize_level(config: dict[str, object]) -> None:
    """Expand an authored level into the historical execution keys in place.

    The saved plan continues to carry the real runner arm and treatment. This is
    deliberate: changing either key in an existing run would change its inputs.
    """
    selected = config.get("level")
    if selected is None:
        return
    if not isinstance(selected, str) or selected not in _BY_ID:
        raise ValueError(f"unknown Phase 1 experiment level {selected!r}")
    entry = _BY_ID[selected]
    for key, expected in (("arm", entry["runner_arm"]), ("treatment", entry["treatment"])):
        actual = config.setdefault(key, expected)
        if actual != expected:
            raise ValueError(f"{selected} requires {key}: {expected}, got {actual!r}")
    bundle = config.get("bundle")
    if isinstance(bundle, str):
        from merlin.development.targetgen.authoring.experiment_bundles import _ALL_ARMS

        matches = [name for name, stem in _ALL_ARMS.items() if bundle.startswith(stem + "_")]
        if matches:
            bundle_arm = max(matches, key=lambda name: len(_ALL_ARMS[name]))
            if bundle_arm != entry["bundle_arm"]:
                raise ValueError(f"{selected} contradicts the selected bundle for {_BY_ARM[bundle_arm]}")


def level_for_phase1(config: Mapping[str, object]) -> str | None:
    """Label a declared selection without changing its execution identity.

    The runner's ``arm`` is historically ``merlin_assisted`` for both EL3 and
    EL4.  A concrete bundle stem, when available, distinguishes their tool
    grants; the RTL-checks treatment distinguishes EL4 in authored definitions.
    Unknown or contradictory selections stay unlabeled instead of acquiring a
    plausible but incorrect level.
    """
    declared = config.get("level")
    arm = config.get("arm")
    treatment = config.get("treatment", "baseline")
    bundle = config.get("bundle")
    if not isinstance(arm, str) or not isinstance(treatment, str):
        return None
    if isinstance(bundle, str):
        from merlin.development.targetgen.authoring.experiment_bundles import _ALL_ARMS

        matches = [name for name, stem in _ALL_ARMS.items() if bundle.startswith(stem + "_")]
        if matches:
            selected = max(matches, key=lambda name: len(_ALL_ARMS[name]))
            if selected in {"merlin_rtlchecks", "merlin_isa", "merlin_lean"}:
                inferred = _BY_ARM[selected] if arm == "merlin_assisted" and treatment == "rtlchecks" else None
                return inferred if declared in (None, inferred) else None
            if selected in {"merlin_assisted", "merlin_verify"}:
                inferred = _BY_ARM[selected] if arm == "merlin_assisted" and treatment == "baseline" else None
                return inferred if declared in (None, inferred) else None
            inferred = _BY_ARM[selected] if arm == selected and treatment == "baseline" else None
            return inferred if declared in (None, inferred) else None
    if arm == "merlin_assisted":
        inferred = "EL4" if treatment == "rtlchecks" else "EL3" if treatment == "baseline" else None
        if declared == "EL4-V" and treatment == "baseline":
            inferred = declared
        if declared in ("EL5", "EL6") and treatment == "rtlchecks":
            inferred = declared
    else:
        inferred = _BY_ARM.get(arm) if treatment == "baseline" else None
    return inferred if declared in (None, inferred) else None


# ---- treatments --------------------------------------------------------------------------------
# Explicit per-invocation treatment choices for the retained native QA controller.
#
# Callbacks remain trusted host code. Their identity refers to the existing implementation
# inventory, not a new source seal or a claim about arbitrary in-memory monkeypatches.


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
        defer_pending: Callable[[], str | None] | None = None,
        defer_running: bool = True,
        on_capsule_complete: Callable[[dict], None] | None = None,
        max_workers: int = 0,
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


# ---- spend -------------------------------------------------------------------------------------
# Invocation-time Phase 1 spend-ledger accounting.
#
# The ledger is a soft stop between authoring rounds, not a grading verdict.
# Importing this module does not read credentials, environment, or run state.


def spend_over_cap(this_round_cost) -> tuple[bool, float, float]:
    """Append subagent-inclusive cost to MERLIN_SPEND_LEDGER; return (over_cap, total, cap).

    MERLIN_MAX_SPEND_USD is a soft cap: one in-flight round per arm may overshoot.
    Missing cap/ledger disables it; unknown usage remains a visible lower bound.
    """
    import os as _os

    cap = float(_os.environ.get("MERLIN_MAX_SPEND_USD") or 0)
    ledger = _os.environ.get("MERLIN_SPEND_LEDGER")
    if cap <= 0 or not ledger:
        return False, 0.0, 0.0
    import fcntl

    # Missing usage is unknown, never zero; a timeout may prevent the terminal usage event.
    _unmeasured = this_round_cost is None
    c = None if _unmeasured else float(this_round_cost)
    p = Path(ledger)
    p.parent.mkdir(parents=True, exist_ok=True)
    total, n_unmeasured = 0.0, 0
    with open(p, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"cost": c, "unmeasured": _unmeasured}) + "\n")
        f.flush()
        f.seek(0)
        for line in f:
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 — a malformed ledger line must not defeat the cap
                continue
            if row.get("unmeasured") or row.get("cost") is None:
                n_unmeasured += 1
                continue
            try:
                total += float(row.get("cost") or 0)
            except Exception:  # noqa: BLE001
                continue
        fcntl.flock(f, fcntl.LOCK_UN)
    if n_unmeasured:
        print(
            f"  [spend] ${total:.2f} of ${cap:.2f} measured, plus {n_unmeasured} UNMEASURED round(s) "
            f"whose usage never arrived — the true total is a LOWER BOUND",
            flush=True,
        )
    return total >= cap, total, cap
