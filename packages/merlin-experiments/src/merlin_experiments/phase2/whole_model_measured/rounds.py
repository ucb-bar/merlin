"""The authoring round of a ``whole_model_measured`` run: one agent session over the run's candidate.

Each round goes through the Phase 2 owners this mode shares with the others:

1. a FRESH round workspace holding exactly the run's current candidate bytes
   (:func:`..agent_workspace.fresh_round_workspace`), and the task;
2. a :class:`..broker.Broker` serving this mode's host-owned actions -- ``measure`` (request the
   exact current bytes on the screen), ``correctness``, ``wait-for-result``, ``status`` and
   ``check-capsules`` -- through the same credential-free shim, receipt ledger and call budget as
   every mode.  Bytes that exceed the edit authority are refused before they are requested;
3. the agent (Codex in the outer bwrap boundary by default; any callable in tests);
4. after it: the transcript audit (:func:`..transcript_audit.audit_codex_transcript`), the broker
   receipts joined to the audited invocations, the edit authority over the final bytes, and the
   round's status by phase 1's rule (:func:`..authoring.authored_round_status`);
5. an AUTHORED round's final bytes become the run's candidate and the HARNESS requests them on the
   screen, so no authored round ends unmeasured.

A round that is not authored (a crash, a killed agent, an audit finding, a refused edit) is recorded
with why, its bytes are not carried forward, and the next session starts from the previous candidate
(:func:`.sessions.run_sessions`).  A round whose DRIVER was killed (the launcher itself, by SIGKILL or
the host) never writes its record, so each round keeps an OPEN marker naming what it requested while
it runs; the next ``start`` closes any marker no live driver owns (:func:`recover_killed_rounds`) --
its requests are attributed ``unauthored`` instead of staying ``pending`` forever -- and numbers its
own rounds after it (:func:`next_session`), so a relaunch never collides with a used round workspace.

Every byte the agent asked to measure is ``pending`` in the store until its round ends, then
``authored`` or ``unauthored`` (:data:`.jobs.ATTRIBUTION_FILE`); only attributable bytes can become the
best, be tagged ``best`` or be exported as a champion.

The task tells the agent the goal, the rule and the tools -- never how to lower anything.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import broker as PB
from merlin_experiments.phase2.authoring import authored_round_status
from merlin_experiments.phase2.broker_policy import ActionAdmission, WorkflowPolicy
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.transcript_audit import audit_codex_transcript

from . import fast as FAST
from . import jobs as J
from .identity import now, package_digest, read_json, write_json_atomic
from .sessions import ROUND_FAILED

WORKFLOW_ID = "whole_model_measured_v1"
ROUND_SCHEMA = "merlin.phase2.whole_model_measured.round.v1"
OPEN_ROUND_SCHEMA = "merlin.phase2.whole_model_measured.open_round.v1"
#: A round record's suffix, and the marker a round keeps beside it while it runs.
ROUND_SUFFIX = ".round.json"
OPEN_SUFFIX = ".open.json"
#: The status of a round whose driver was killed before it wrote its own record.
ROUND_KILLED = "round_killed"
#: The argv sentinel of a host-owned action: the broker never executes it, the workflow answers it.
HOST_OWNED = "__host_owned_whole_model_measured__"
#: Edit authority modes a launch profile may name.
WHOLE_PACKAGE = "whole-package"
FROZEN_CONTRACT = "frozen-contract"


def _action(name: str, purpose: str, placeholders: tuple[str, ...] = ()) -> PB.BrokerAction:
    return PB.BrokerAction(name, (HOST_OWNED, name, *(f"{{{p}}}" for p in placeholders)), placeholders, purpose, False)


ACTIONS: tuple[PB.BrokerAction, ...] = (
    _action("measure", "request the exact current bytes on the whole-model screen; returns what is known now"),
    _action("correctness", "the capsule screen and the functional model's per-group grade of the current bytes"),
    _action("wait-for-result", "block until what is known about the current bytes changes, then report it"),
    _action("status", "the objective: the bar, the best so far and the measured history"),
    _action("check-capsules", "grade the current bytes on named capsules, or on a failing group's form", ("names",)),
    _action(
        "structure",
        "start (or read) the current bytes' whole program on the functional model: each group's local "
        "correctness and simulator cycles (never the board's); minutes",
    ),
    _action(
        "group-timing",
        "start (or read) the groups the current bytes changed against BASELINE (best, seed or a digest), "
        "each timed alone on the RTL emulator beside the baseline's: the direction of an edit, per group",
        ("baseline",),
    ),
    _action(
        "group-check",
        "start (or read) one GROUP's program on the functional model, on the inputs the model hands it: "
        "where its output is wrong",
        ("group",),
    ),
)


class MeasuredWorkflow(WorkflowPolicy):
    """This mode's broker workflow: every action is answered on the host, by the objective."""

    workflow_id = WORKFLOW_ID

    def __init__(
        self,
        *,
        candidate: Path,
        target_experiment: Any,
        receipt_path: Path,
        objective: Any,
        scratch: Path,
        edit_check: Callable[[Path], Mapping[str, Any]],
        round_index: int = 0,
        open_record: Path | None = None,
    ):
        super().__init__(candidate=candidate, target_experiment=target_experiment, receipt_path=receipt_path)
        self.objective = objective
        self.scratch = Path(scratch)
        self.edit_check = edit_check
        self.round_index = int(round_index)
        #: The digests this round asked the screen for, in order (the round record names them).
        self.requested: list[str] = []
        #: The round's open marker: every request is on disk the moment it is made, so a driver killed
        #: mid-round leaves behind what it asked for (:func:`recover_killed_rounds`).
        self.open_record = Path(open_record) if open_record is not None else None

    def _note_request(self, digest: str) -> None:
        self.requested.append(digest)
        if self.open_record is not None:
            marker = read_json(self.open_record) or {}
            write_json_atomic(self.open_record, {**marker, "requested": list(dict.fromkeys(self.requested))})

    def admission(self, name: str) -> ActionAdmission:
        # No per-action cap: the broker's call budget and the round's deadline bound every action.
        return ActionAdmission(None, False, "", "the round's wall-clock budget is spent")

    def validate_actions(self, actions) -> None:
        if set(actions) != {action.name for action in ACTIONS}:
            raise StageGateError("the measured workflow serves exactly its own declared actions")

    def build_registry(self):
        return ACTIONS

    def answer(self, action: str, rendered: Mapping[str, str], timeout_s: int) -> Mapping[str, Any]:
        candidate = Path(self.candidate)
        objective = self.objective
        if action == "status":
            return objective.summary()
        if action == "wait-for-result":
            return objective.wait(candidate, timeout_s=max(1.0, float(timeout_s) - 5.0))
        # Every action that BUILDS the bytes first holds them to the edit authority.
        self.edit_check(candidate)
        # PENDING until the round ends: never the best, whatever it measures, unless the round is authored.
        # Every action that creates a job carries it: `correctness` requests the same job `measure` does.
        pending = {"state": J.ATTRIBUTION_PENDING, "round": self.round_index, "why": "requested mid-round"}
        if action in ("measure", "correctness"):
            if action == "measure":
                document = objective.measure(candidate, label="agent request", attribution=pending)
            else:
                document = objective.correctness(candidate, attribution=pending)
            self._note_request(str(document.get("package_sha256")))
            return document
        if action in FAST.FAST_KINDS:
            return FAST.request(
                objective,
                action,
                candidate,
                baseline=rendered.get("baseline", "best"),
                group=int(rendered["group"]) if action == "group-check" else None,
            )
        if action == "check-capsules":
            out = self.scratch / f"capsule_check_{time.time_ns()}" / "report.json"
            return objective.check_capsules(candidate, rendered["names"], out=out, timeout_s=float(timeout_s))
        raise StageGateError(f"undeclared action {action!r}")

    def execute(self, request, action_name, rendered, call_index, timeout_s, started):
        try:
            document = self.answer(action_name, rendered, timeout_s)
            text = (document.get("text") or document.get("feedback_text")) if isinstance(document, Mapping) else None
            body = json.dumps(document, indent=1, sort_keys=True, default=str)
            result = {"returncode": 0, "stdout": (f"{text}\n\n" if text else "") + body + "\n", "stderr": ""}
        except Exception as exc:  # noqa: BLE001 -- a refusal the agent reads, never a crashed broker
            result = {"returncode": 125, "stdout": "", "stderr": f"{action_name} refused: {type(exc).__name__}: {exc}"}
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        return result, None


class HostOnlyPolicy:
    """The inner sandbox of a workflow whose every action is answered on the host: nothing executes."""

    argv: tuple[str, ...] = ()
    env_prefix = ""
    process_cwd = None


def render_task(objective: Any, *, round_index: int, rounds: int, round_seconds: int) -> str:
    """The task the agent reads: the goal, the rule and the tools.  Never a lowering hint."""
    rule = objective.rule() or {}
    roles = ", ".join(rule.get("prohibited_roles") or ()) or "none"
    tools = "\n".join(
        f"- `python3 {PB.BROKER_NAME} {a.name}{''.join(f' {p}=...' for p in a.placeholders)}`: {a.purpose}"
        for a in ACTIONS
    )
    derivation = (getattr(objective, "config", None) or {}).get("mechanism_derivation") or {}
    screen = (derivation.get("sections") or {}).get("screen") or {}
    mechanisms = (
        "The selected model's derived closure is closed: package-declared whole-model passes and fused "
        "regions are available through the verified build; inspect `submission/` to author candidates."
        if screen.get("model_closure") == "closed"
        else "The selected model's derived closure is open: package passes and fused regions are unavailable."
        if screen.get("model_closure") == "open"
        else ""
    )
    return f"""# Whole-model performance

Make the compiler package in `submission/` produce a faster whole model, measured end to end, while
every group stays correct.  This is round {round_index + 1} of at most {rounds}; it has {round_seconds} s.

The objective is the whole model's cycles on the screening machine; `status` shows the reference on
the same machine and the best so far.  A wrong program has no cycles that count.

Instruction rule: the linked program may issue no instruction with role(s) `{roles}`, in any group,
library groups and host code included.  The whole program is checked before it runs.

{mechanisms}

Tools (each line is the ENTIRE command, nothing before or after it):
{tools}

Edit only files under `submission/`.  Building and measuring happen on the host through these tools.
Run the package only through them: do not execute any file under `submission/` yourself (not even
`--help`, not as a Python import); read it with `cat`, `sed` or `rg`.  A broker command stands alone:
nothing chained, piped or redirected around it.
"""


def whole_package_edits(seed: Path, candidate: Path) -> dict[str, Any]:
    """WHOLE-PACKAGE authority: any compiler edit, except the manifest's controls (everything but its
    ``optimization_surfaces``), which only the host may change, and no links (a link would carry
    bytes from outside the package into it)."""
    import yaml

    links = sorted(str(p.relative_to(candidate)) for p in Path(candidate).rglob("*") if p.is_symlink())
    if links:
        raise ValueError(f"the package contains link(s) {links[:5]}")

    def controls(root: Path) -> Any:
        path = Path(root) / "manifest.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
        if isinstance(document, dict):
            document = {k: v for k, v in document.items() if k != "optimization_surfaces"}
        return document

    if controls(seed) != controls(candidate):
        raise ValueError("the manifest's controls changed; only the host may change them")
    return {"status": "allowed", "authority": WHOLE_PACKAGE}


def join_receipts(receipt_path: Path, audit: Mapping[str, Any]) -> dict[str, Any]:
    """The host's broker receipts joined, in order, to the audited transcript's invocations: the same
    (action, bindings digest) pairs, gapless.  A mismatch is a refusal of the round."""
    rows: list[Mapping[str, Any]] = []
    if Path(receipt_path).is_file() and not Path(receipt_path).is_symlink():
        for line in Path(receipt_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    # A receipt is written when its call FINISHES, so a call that blocks (`wait-for-result`) lands after
    # the calls the agent made meanwhile; the transcript lists invocations as they START, which is the
    # broker's index order.  Joined in file order, a round with one long wait was refused.
    rows.sort(key=lambda row: row.get("index") if isinstance(row.get("index"), int) else -1)
    gapless = [row.get("index") for row in rows] == list(range(len(rows)))
    observed = [(row.get("action"), row.get("bindings_command_sha256")) for row in rows]
    claimed = [
        (row.get("action"), row.get("bindings_sha256"))
        for row in audit.get("broker_invocations") or ()
        if isinstance(row, Mapping)
    ]
    joined = gapless and observed == claimed
    return {
        "receipts": len(rows),
        "invocations": len(claimed),
        "joined": joined,
        "reason": None if joined else "the host's broker receipts do not match the audited invocations",
    }


def outer_policy_argv(
    workspace: Path, control_dir: Path, target_experiment: Any, runtime_binds: Sequence[str] = ()
) -> list[str]:
    """The agent's filesystem boundary: its workspace, the broker control directory read-only, the
    driver's runtime binds, and every declared answer surface masked.  A coverage gap refuses."""
    from merlin.targetgen.sandbox import bwrap as BW

    inputs = AW._sandbox_inputs(target_experiment, None)
    repo = inputs.paths.repo.resolve()
    argv = AW._strip_claude_home(BW.base_argv(workspace, {}, repo=repo, _policy_test_live_inputs=True))
    argv += ["--clearenv", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin"]
    argv += ["--setenv", "XDG_RUNTIME_DIR", "/tmp/.xdg", *runtime_binds]
    argv += ["--ro-bind", str(control_dir), "/perf-control"]
    surfaces = list(inputs.surfaces)
    argv = BW.apply_answer_masks(argv, surfaces)
    gaps = [str(surface.path) for surface in BW.coverage_gap(argv, surfaces)]
    if gaps:
        raise StageGateError(f"the agent's sandbox exposes answer surfaces: {gaps}")
    return argv


def codex_agent(profile: Mapping[str, Any], *, target_experiment: Any, codex_binary: str | None = None):
    """The production agent: one Codex round inside :func:`outer_policy_argv`, on the profile's model
    (which the launch already checked resolves to itself).

    The executable is resolved to its REAL path here, on the host: the sandbox's PATH is the system's,
    so a bare ``codex`` (an install under the user's home) is not found inside it and the round dies at
    exit 127 before a turn starts."""
    from merlin_experiments.phase2.contracts import require_executable

    resolved = require_executable(codex_binary or os.environ.get("CODEX_BIN") or "codex", label="Codex")

    def run(*, workspace, candidate, control_dir, prompt, round_index, timeout_s, stage_root):
        from merlin.targetgen.sandbox import bwrap as BW
        from merlin_experiments.phase1.providers import codex_agent as CA

        def sandbox_command(inner: str, ws: Path, _bundle: dict, extra_binds: list[str] | None = None) -> str:
            argv = outer_policy_argv(ws, control_dir, target_experiment, extra_binds or ())
            # ``inner`` is already shell-quoted by the driver; quote only the outer payload boundary.
            return BW.compose_command(argv, " bash -c " + shlex.quote(inner), ws)

        return CA.run_round(
            Path(workspace),
            Path(stage_root),
            str(profile["model"]),
            {},
            target_experiment,
            "bwrap",
            int(round_index),
            int(timeout_s),
            effort=str(profile.get("effort") or ""),
            prompt=prompt,
            effective_model=str(profile["model"]),
            sandbox_command=sandbox_command,
            codex_binary=resolved,
            codex_home_root=Path(stage_root) / "codex_homes",
        )

    return run


@dataclass
class RoundDriver:
    """Drives one authoring round per call: ``run_round(session=, stage_root=)``."""

    run_dir: Path
    objective: Any
    target_experiment: Any
    agent: Callable[..., tuple[int, Path]]
    round_seconds: int
    tool_seconds: int
    max_tool_calls: int
    rounds: int
    authority: Any = None
    audit_token_set: Mapping[str, Sequence[str]] | None = None

    def edit_check(self, candidate: Path) -> Mapping[str, Any]:
        if self.authority is not None:
            return self.authority.validate_candidate(Path(candidate))
        return whole_package_edits(self.run_dir / "seed" / "submission", Path(candidate))

    def __call__(self, *, session: int, stage_root: Path) -> dict[str, Any]:
        index = int(session) - 1
        stage_root = Path(stage_root)
        current = self.run_dir / "workspace"
        workspace = stage_root / "agent_workspaces" / f"round_{index:02d}"
        candidate = AW.fresh_round_workspace(current, workspace, package_digest(current))
        prompt = render_task(self.objective, round_index=index, rounds=self.rounds, round_seconds=self.round_seconds)
        (workspace / "TASK.md").write_text(prompt, encoding="utf-8")
        control_dir = stage_root / "control" / f"round_{index:02d}"
        receipt_path = control_dir / "receipts.jsonl"
        rounds_dir = stage_root / "rounds"
        rounds_dir.mkdir(parents=True, exist_ok=True)
        open_marker = rounds_dir / f"round_{index:02d}{OPEN_SUFFIX}"
        write_json_atomic(
            open_marker,
            {
                "schema": OPEN_ROUND_SCHEMA,
                "round": index,
                "run": self.run_dir.name,
                "started_at": now(),
                "pid": os.getpid(),
                "requested": [],
            },
        )
        workflow = MeasuredWorkflow(
            candidate=candidate,
            target_experiment=self.target_experiment,
            receipt_path=receipt_path,
            objective=self.objective,
            scratch=stage_root / "host_checks" / f"round_{index:02d}",
            edit_check=self.edit_check,
            round_index=index,
            open_record=open_marker,
        )
        broker = PB.Broker(
            HostOnlyPolicy(),
            self.target_experiment,
            candidate,
            ACTIONS,
            receipt_path,
            deadline=time.monotonic() + self.round_seconds,
            workflow=workflow,
            max_calls=self.max_tool_calls,
            max_tool_seconds=self.tool_seconds,
        )
        started = time.monotonic()
        rc: int | None = None
        transcript: Path | None = None
        failure: str | None = None
        try:
            with broker.serving() as (host, port):
                PB.stage_broker_shim(
                    control_dir,
                    host=host,
                    port=port,
                    token=broker.token,
                    tool_timeout_s=self.tool_seconds,
                    actions=ACTIONS,
                )
                rc, transcript = self.agent(
                    workspace=workspace,
                    candidate=candidate,
                    control_dir=control_dir,
                    prompt=prompt,
                    round_index=index,
                    timeout_s=self.round_seconds,
                    stage_root=stage_root,
                )
        except Exception as exc:  # noqa: BLE001 -- a driver that dies is this round's failure, recorded
            failure = f"the agent driver raised {type(exc).__name__}: {str(exc)[:500]}"
        finally:
            # The broker's credential leaves with the round; its receipts are sealed read-only.
            config = control_dir / ".perf_broker.json"
            if config.is_file() and not config.is_symlink():
                config.chmod(0o600)
                config.unlink()
            if receipt_path.is_file() and not receipt_path.is_symlink():
                receipt_path.chmod(0o444)
        audit: Mapping[str, Any] = {"clean": None, "hits": [], "commands_seen": None}
        if failure is None:
            try:
                audit = audit_codex_transcript(
                    Path(transcript), self.target_experiment, candidate, ACTIONS, audit_token_set=self.audit_token_set
                )
            except Exception as exc:  # noqa: BLE001 -- an unauditable round is refused, never authored
                failure = f"the round's transcript could not be audited: {type(exc).__name__}: {str(exc)[:500]}"
        receipts = join_receipts(receipt_path, audit)
        refusals = [] if receipts["joined"] else [str(receipts["reason"])]
        try:
            edits = dict(self.edit_check(candidate))
        except ValueError as exc:
            edits = {"status": "refused", "reason": str(exc)}
            refusals.append(f"edit authority: {exc}")
        if failure is None:
            status = authored_round_status(agent_exit_code=int(rc), audit_clean=audit.get("clean"), refusals=refusals)
        else:
            status = {"status": ROUND_FAILED, "stopped_by": None, "why": failure}
        carried = status["status"] == "authored"
        final = None
        if carried:
            staging = self.run_dir / f".workspace_round_{index:02d}"
            shutil.copytree(candidate, staging, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.rmtree(current)
            staging.rename(current)
            # THE HARNESS requests the round's final bytes: an authored round never ends unmeasured.
            final = self.objective.measure(
                current,
                label=f"round {index + 1} final",
                attribution={"state": J.ATTRIBUTION_AUTHORED, "round": index, "why": "the round's final bytes"},
            )
        # What the agent asked for mid-round is attributed by how the round ended: an authored round
        # earned it; any other round did not, and those bytes never become the best or a champion.
        change = {
            "state": J.ATTRIBUTION_AUTHORED if carried else J.ATTRIBUTION_UNAUTHORED,
            "round": index,
            "why": status["why"],
        }
        attribution = {
            digest: self.objective.attribute(digest, change).get("state")
            for digest in dict.fromkeys(workflow.requested)
        }
        record = {
            "schema": ROUND_SCHEMA,
            "round": index,
            "status": status["status"],
            "stopped_by": status.get("stopped_by"),
            "why": status["why"],
            "agent_exit_code": int(rc) if rc is not None else None,
            # A negative exit is the signal that ended the agent (-9: SIGKILL, which leaves no evidence of
            # its own); the round is refused and the next session starts from the previous candidate.
            **({"agent_signal": -int(rc)} if rc is not None and int(rc) < 0 else {}),
            **({"agent_failure": failure} if failure is not None else {}),
            "wall_seconds": round(time.monotonic() - started, 1),
            "audit": {k: audit.get(k) for k in ("clean", "hits", "commands_seen")},
            "receipts": receipts,
            "edits": edits,
            "requested": list(workflow.requested),
            "attribution": attribution,
            "carried_forward": carried,
            "final_package_sha256": (final or {}).get("package_sha256"),
            "final_timing_status": (final or {}).get("timing_status"),
        }
        (rounds_dir / f"round_{index:02d}{ROUND_SUFFIX}").write_text(
            json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8"
        )
        open_marker.unlink(missing_ok=True)  # the record now says everything the marker did
        return {
            "status": status["status"],
            "transcript": str(transcript) if transcript is not None else None,
            "summaries": [str(p) for p in sorted(rounds_dir.glob(f"round_{index:02d}.*summary.json"))],
            "failure": None if carried else status["why"],
        }


def _round_index(name: str) -> int | None:
    """The index of a ``round_<NN>...`` file or directory name, parsed by tokens (None otherwise)."""
    head, sep, rest = name.partition("_")
    if head != "round" or not sep:
        return None
    digits = rest.split(".", 1)[0]
    return int(digits) if digits.isdigit() else None


def next_session(stage_root: Path) -> int:
    """The session number a ``start`` on this stage begins at: one past the highest round it has
    started (a round record, an open marker or a round workspace).  A relaunched ``start`` on the same
    run otherwise begins at session 1 again, finds round 0's workspace taken, and fails every round."""
    stage_root = Path(stage_root)
    seen = [-1]
    for directory in (stage_root / "rounds", stage_root / "agent_workspaces"):
        if directory.is_dir():
            seen += [i for i in (_round_index(p.name) for p in directory.iterdir()) if i is not None]
    return max(seen) + 2


def _driver_alive(pid: Any, run_name: str) -> bool:
    """Whether ``pid`` is a live process other than this one whose command line names ``run_name``
    (pids are reused, so liveness alone names no driver)."""
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        return False
    try:
        return run_name.encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def recover_killed_rounds(
    stage_root: Path, *, attribute: Callable[[str, Mapping[str, Any]], Mapping[str, Any]], run_name: str
) -> list[dict[str, Any]]:
    """Close every round of ``stage_root`` that started and never ended (an open marker without a round
    record) and that no live driver owns: each digest it requested is attributed ``unauthored`` through
    ``attribute`` and the round is recorded :data:`ROUND_KILLED`.  Returns the records written.

    Why: a round's requests are ``pending`` until its driver resolves them; a driver killed mid-round
    (SIGKILL, the host's memory guard) never does, so without this they stay pending forever -- shown,
    never eligible, and never said to be anyone's."""
    rounds_dir = Path(stage_root) / "rounds"
    if not rounds_dir.is_dir():
        return []
    recovered = []
    for marker in sorted(rounds_dir.glob(f"round_*{OPEN_SUFFIX}")):
        index = _round_index(marker.name)
        if index is None or marker.is_symlink():
            continue
        record_path = rounds_dir / f"round_{index:02d}{ROUND_SUFFIX}"
        if record_path.exists():
            marker.unlink(missing_ok=True)  # the round ended; only its marker outlived it
            continue
        opened = read_json(marker) or {}
        if _driver_alive(opened.get("pid"), run_name):
            continue
        why = (
            f"round {index + 1} never ended: its driver (pid {opened.get('pid')}) was killed before it wrote "
            "the round's record, so no authored round earned these bytes"
        )
        change = {"state": J.ATTRIBUTION_UNAUTHORED, "round": index, "why": why}
        attribution: dict[str, Any] = {}
        for digest in dict.fromkeys(str(d) for d in opened.get("requested") or ()):
            try:
                attribution[digest] = attribute(digest, change).get("state")
            except J.ServiceError as exc:
                attribution[digest] = f"not attributed: {exc}"
        record = {
            "schema": ROUND_SCHEMA,
            "round": index,
            "status": ROUND_KILLED,
            "stopped_by": None,
            "why": why,
            "agent_exit_code": None,
            "started_at": opened.get("started_at"),
            "recovered_at": now(),
            "requested": list(attribution),
            "attribution": attribution,
            "carried_forward": False,
        }
        write_json_atomic(record_path, record)
        marker.unlink(missing_ok=True)
        recovered.append(record)
    return recovered


def round_driver(
    *, profile: Mapping[str, Any], run_dir: Path, objective: Any, agent: Any = None, target_experiment: Any = None
) -> RoundDriver:
    """The default ``--round-driver`` (``module:round_driver``).  The target descriptor is the objective
    config's ``descriptor``, else the target's own; the edit authority is the profile's
    ``edit_authority`` (``whole-package``, or ``frozen-contract`` with the config's ``edit_contract``)."""
    config = getattr(objective, "config", None) or {}
    if target_experiment is None:
        from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment

        descriptor = config.get("descriptor") or descriptor_for(str(objective.screen.target))
        if descriptor is None:
            raise StageGateError(f"no target descriptor for {objective.screen.target!r}")
        target_experiment = load_target_experiment(Path(descriptor))
    mode = str(profile.get("edit_authority") or WHOLE_PACKAGE)
    authority = None
    if mode == FROZEN_CONTRACT:
        from merlin_experiments.phase2.edit_authority import FrozenEditAuthority

        if not config.get("edit_contract"):
            raise StageGateError("a frozen-contract edit authority needs the config's edit_contract")
        output = Path(run_dir) / "edit_authority"
        output.mkdir(parents=True, exist_ok=True)
        authority = FrozenEditAuthority(output)
        if authority.contract is None:
            authority.freeze(
                Path(run_dir) / "seed" / "submission",
                json.loads(Path(config["edit_contract"]).read_text(encoding="utf-8")),
                has_iterations=False,
            )
    elif mode != WHOLE_PACKAGE:
        raise StageGateError(f"unknown edit authority {mode!r} (expected {WHOLE_PACKAGE} or {FROZEN_CONTRACT})")
    return RoundDriver(
        run_dir=Path(run_dir),
        objective=objective,
        target_experiment=target_experiment,
        agent=agent or codex_agent(profile, target_experiment=target_experiment),
        round_seconds=int(profile["round_seconds"]),
        tool_seconds=int(profile["iteration_seconds"]),
        max_tool_calls=int(profile["max_tool_calls"]),
        rounds=int(profile["max_sessions"]),
        authority=authority,
    )


__all__ = [
    "ACTIONS",
    "FROZEN_CONTRACT",
    "MeasuredWorkflow",
    "ROUND_KILLED",
    "RoundDriver",
    "WHOLE_PACKAGE",
    "WORKFLOW_ID",
    "codex_agent",
    "join_receipts",
    "next_session",
    "outer_policy_argv",
    "recover_killed_rounds",
    "render_task",
    "round_driver",
    "whole_package_edits",
]
