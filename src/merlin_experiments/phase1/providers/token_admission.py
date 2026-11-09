"""Opt-in managed author admission using live, lower-bound all-thread usage.

No provider request token cap or native child-spawn interception is claimed.
The owned process supervisor enforces decisions at launch and poll boundaries.
"""

from __future__ import annotations

import fcntl
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..telemetry import author_agents, token_budget

PREFIX = "author token admission: "


def _policy(run_dir):
    path = Path(run_dir) / "environment.yaml"
    if not path.exists():
        return None
    import yaml

    document = yaml.safe_load(path.read_text())
    policy = (document.get("run_config") or {}).get("author_token_budget")
    if policy is None:
        return None
    policy = token_budget.validate(policy)
    return policy if policy["schema"] == token_budget.ADMISSION_SCHEMA else None


class ManagedTokens:
    """One process-round owner, sharing admissions atomically across the run."""

    def __init__(self, run_dir, rnd, *, homes=(), observe=None, clock=time.monotonic):
        self.run_dir, self.rnd = Path(run_dir), rnd
        self.policy = _policy(run_dir)
        self.homes, self.clock = homes, clock
        self.reader = author_agents.UsageRows()
        self.observe = observe or (
            lambda: author_agents.usage_snapshot(self.run_dir, codex_homes=self.homes, reader=self.reader)
        )
        self.key, self.phase = None, None
        self.active, self.owner = False, uuid.uuid4().hex
        self.started = None
        self.next_poll = 0.0
        self.missing_since = None
        self.last = None
        self.path = self.run_dir / "author_token_admission.json"

    @contextmanager
    def _state(self):
        with (self.run_dir / ".author_token_admission.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {
                    "schema": "merlin.managed_author_admission.v1",
                    "policy": self.policy,
                    "observed_high_water": 0,
                    "admissions": 0,
                    "integration_admitted": False,
                    "outstanding": {},
                    "decisions": [],
                    "integration_requested": False,
                }
            )
            if state.get("schema") != "merlin.managed_author_admission.v1" or state.get("policy") != self.policy:
                raise ValueError("managed author admission state differs from selected policy")
            yield state
            temporary = self.path.with_suffix(".pending.json")
            temporary.write_text(json.dumps(state, indent=2) + "\n")
            temporary.replace(self.path)

    def _measurement(self, state, *, bootstrap=False):
        snapshot = self.observe()
        if snapshot.get("available") is not True:
            if bootstrap and state["admissions"] == 0 and snapshot.get("pristine") is True:
                return 0, 0, snapshot
            return None, None, snapshot
        total = snapshot.get("tokens", {}).get("total")
        if type(total) is not int or total < 0:
            return None, None, snapshot
        state["observed_high_water"] = max(total, state["observed_high_water"])
        # This is an explicit planning allowance, NOT a fabricated physical cap.
        # Keep reservations for other managed processes too: do not release them
        # from file age or a stale-looking turn. Conservative double reservation
        # of an unbound owner is preferable to silently spending its allowance.
        allowance = self.policy["admission"]["turn_allowance"]
        other = sum(record["allowance"] for key, record in state["outstanding"].items() if key != self.key)
        pending = max(int(self.active), len(set(snapshot.get("open_threads", ())))) * allowance + other
        return state["observed_high_water"], pending, snapshot

    def _record(self, state, decision, snapshot):
        record = {
            "schema": "merlin.author_token_admission_decision.v1",
            "round": self.rnd,
            "turn": self.key,
            "phase": self.phase,
            "cached_input_and_helpers_included": True,
            "hard_token_cap": False,
            "provider_upper_bound": None,
            "native_child_spawn_intercepted": False,
            "unknown_inflight_usage": True,
            "at_monotonic_s": self.clock(),
            "missing_threads": snapshot.get("missing_leads", []) + snapshot.get("missing_subagents", []),
            **decision,
        }
        self.last = record
        state["decisions"].append(record)
        return record

    def admit(self, turn):
        if self.policy is None:
            return None
        if self.active:
            raise ValueError("managed author turn already has an outstanding reservation")
        self.key = f"{self.rnd}:{turn}"
        with self._state() as state:
            if self.key in state["outstanding"]:
                raise ValueError("managed author turn already has an outstanding reservation")
            total, pending, snapshot = self._measurement(state, bootstrap=True)
            allowance = self.policy["admission"]["turn_allowance"]
            projected = None if total is None else total + pending + allowance
            reason = None
            if total is None:
                reason = "live all-author usage unavailable; admission refused"
            elif state["integration_admitted"]:
                reason = "integration continuation already admitted; preserve checkpoint for review"
            elif projected > self.policy["target"]:
                reason = "outstanding work plus next turn exceeds target planning budget"
            self.phase = (
                "integration"
                if state.get("integration_requested")
                or (projected is not None and projected >= self.policy["target"] - self.policy["reserve"])
                else "exploration"
            )
            if reason is None:
                state["admissions"] += 1
                state["integration_admitted"] = self.phase == "integration"
                state["outstanding"][self.key] = {"allowance": allowance, "phase": self.phase, "owner": self.owner}
                self.active = True
                self.started = None
                self.missing_since = None
                self.next_poll = self.clock() + self.policy["admission"]["poll_s"]
            return self._record(
                state,
                {
                    "admitted": reason is None,
                    "reason": reason,
                    "observed_total": total,
                    "outstanding_allowance": pending,
                    "next_turn_allowance": allowance,
                    "projected_total": projected,
                },
                snapshot,
            )

    def provider_started(self):
        """Start the execution slice after Popen; admission/setup still reserves tokens.

        A turn can start once only. Slow sandbox preparation must not consume
        the integration execution allowance or permit resetting a running clock.
        """
        if self.policy is None:
            return
        if not self.active or self.started is not None:
            raise ValueError("provider start requires an unstarted active admission")
        self.started = self.clock()

    def poll(self):
        if self.policy is None or self.key is None or self.clock() < self.next_poll:
            return None
        self.next_poll = self.clock() + self.policy["admission"]["poll_s"]
        with self._state() as state:
            total, pending, snapshot = self._measurement(state)
            now = self.clock()
            elapsed = None if self.started is None else now - self.started
            if total is None:
                if self.missing_since is None:
                    self.missing_since = now
            else:
                self.missing_since = None
            missing_s = 0.0 if self.missing_since is None else now - self.missing_since
            selected = self.policy["admission"]
            threshold = self.policy["target"] - (self.policy["reserve"] if self.phase == "exploration" else 0)
            projected = None if total is None else total + pending
            reason = None
            if total is None and missing_s >= selected["missing_usage_grace_s"]:
                reason = "live all-author usage unavailable after grace; checkpoint required"
            elif projected is not None and projected >= threshold:
                reason = "development boundary" if self.phase == "exploration" else "integration target boundary"
            elif self.phase == "integration" and elapsed is not None and elapsed >= selected["integration_slice_s"]:
                reason = "integration wall slice"
            if reason == "development boundary":
                state["integration_requested"] = True
            self._record(
                state,
                {
                    "admitted": reason is None,
                    "reason": reason,
                    "observed_total": total,
                    "outstanding_allowance": pending,
                    "projected_total": projected,
                    "missing_usage_duration_s": missing_s,
                    "provider_elapsed_s": elapsed,
                },
                snapshot,
            )
            return None if reason is None else PREFIX + reason

    def finish(self):
        if self.policy is None or self.key is None or not self.active:
            return
        with self._state() as state:
            if state["outstanding"].get(self.key, {}).get("owner") != self.owner:
                raise ValueError("managed author reservation ownership changed")
            state["outstanding"].pop(self.key)
            self.active = False

    def can_resume_checkpoint(self, reason, *, thread_id, continuous, remaining_s):
        return (
            self.policy is not None
            and continuous
            and bool(thread_id)
            and remaining_s > 30
            and reason == PREFIX + "development boundary"
        )

    def prompt(self):
        if self.policy is None:
            return ""
        status = self.last or {}
        if self.phase == "integration":
            return (
                "\n\nManaged token reserve: integration/checkpoint only. Do not start new exploration or helpers. "
                "Collect existing helper work, preserve the best complete compiler, "
                "repair only integration/correctness blockers, "
                "and write a short checkpoint with exact source revisions, results and remaining gaps. "
                "Prior interrupted edits are unqualified until reviewed and checked. "
                "Finish this continuation promptly.\n"
            )
        return (
            "\n\nManaged token admission: observed all-author total "
            f"{status.get('observed_total')}, outstanding allowance {status.get('outstanding_allowance')}; "
            "cached input and helpers count. Use bounded task-specific helper briefs and file ownership. "
            "Read catalog indexes once; retrieve only needed API/source sections. Keep full logs/IR on disk, "
            "return compact findings and artifact paths. Keep a current short checkpoint for budget recovery.\n"
        )
