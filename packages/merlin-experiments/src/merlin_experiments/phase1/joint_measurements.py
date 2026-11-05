"""Private, question-driven physical measurement admissions for a joint campaign."""
from __future__ import annotations

import fcntl
import json
import time

from . import joint_control


def limits(policy):
    selected = policy.get("operator_measurement_policy", {})
    result = {}
    for name, maximum in (("initial_diagnostics_max", 2), ("bootstrap_observations_max", 8), ("later_batch_max", 4)):
        value = selected.get(name, maximum)
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f"invalid joint measurement limit {name}")
        result[name] = value
    return result


def admit(data, capsule, policy):
    """Consume one permission under the original physical budget's process lock."""
    bounds = limits(policy)
    admissions = data.setdefault("admissions", [])
    pending = data.setdefault("authorizations", [])
    if len(admissions) < bounds["initial_diagnostics_max"]:
        question = (policy.get("operator_measurement_policy", {}).get("witness_questions", {}).get(capsule)
                    or "initial correctness and emitted-mechanism diagnostic")
        row = {"capsule": capsule, "stage": "initial", "question": question}
    else:
        batch = next((row for row in pending if capsule in row["remaining_capsules"]), None)
        if batch is None:
            raise ValueError("gSIM needs an operator-authorized unresolved question after the initial diagnostics")
        if batch["stage"] == "bootstrap" and sum(row["stage"] in {"initial", "bootstrap"} for row in admissions) >= bounds["bootstrap_observations_max"]:
            raise ValueError("joint bootstrap observation budget exhausted")
        batch["remaining_capsules"].remove(capsule)
        row = {"capsule": capsule, "stage": batch["stage"], "question": batch["question"], "batch": batch["batch"]}
    admissions.append({**row, "admitted_at_epoch": time.time()})


def authorize(capsules, *, question, stage="bootstrap"):
    """Operator-only API: explicitly open a bounded batch for a missing model question.

    Caller first sets MERLIN_JOINT_CAMPAIGN_STATE to this run's private state.
    This grants measurement admission only, never calibration qualification.
    """
    selected = joint_control.selected()
    if selected is None:
        raise ValueError("no joint campaign selected")
    policy = selected["policy"]
    bounds = limits(policy)
    if stage not in {"bootstrap", "refinement"} or not isinstance(question, str) or not question.strip():
        raise ValueError("a physical batch needs an explicit stage and unresolved question")
    if (not isinstance(capsules, (list, tuple)) or not capsules or len(capsules) > bounds["later_batch_max"]
            or any(name not in policy["gsim"]["witness_capsules"] for name in capsules)):
        raise ValueError("physical batch must contain at most the selected small witness limit")
    path = joint_control.gsim_ledger()
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        data = json.loads(handle.read() or '{"reserved_s":0,"active":[]}')
        admissions = data.setdefault("admissions", [])
        pending = data.setdefault("authorizations", [])
        issued_bootstrap = sum(row["stage"] in {"initial", "bootstrap"} for row in admissions)
        reserved_bootstrap = sum(len(row["remaining_capsules"]) for row in pending if row["stage"] == "bootstrap")
        # Initial permissions remain reserved even if the operator opens a batch early.
        initial_remaining = max(0, bounds["initial_diagnostics_max"] - len(admissions))
        if stage == "bootstrap" and issued_bootstrap + reserved_bootstrap + initial_remaining + len(capsules) > bounds["bootstrap_observations_max"]:
            raise ValueError("physical batch would exceed the bootstrap observation limit")
        batch = {"batch": len(pending) + 1, "stage": stage, "question": question.strip(),
                 "remaining_capsules": list(capsules), "authorized_at_epoch": time.time()}
        pending.append(batch)
        handle.seek(0)
        handle.truncate()
        json.dump(data, handle)
        return dict(batch)
