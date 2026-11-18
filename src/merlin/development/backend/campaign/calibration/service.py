"""Operator-only completed-product registry and sparse calibration supervisor."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import hash_tree
from merlin.development.backend.campaign import control as joint_control
from merlin.development.backend.campaign.calibration import price_fitting as C
from merlin.development.backend.feedback.campaign import timing as PERF
from merlin.development.backend.optimization.contracts import document_sha256
from merlin.runtime.views import native_simulator as F

# ---- joint_calibration_service -----------------------------------------------------------------


def register(row, root, *, result, compiler, capsule):
    """Called only by the native completed-row owner; no directory discovery."""
    state = joint_control.selected()
    if state is None or not state["policy"].get("native_calibration"):
        return
    from merlin.development.backend.campaign.calibration.joint_calibration_eligibility import eligible

    if not eligible(result, capsule):
        return
    profile = PERF._read(state["policy"]["native_calibration"])
    # Descriptive public intake is deliberately separate from the reviewed fit
    # registry below. Failure here must not change legacy registration behavior.
    try:
        from merlin.development.backend.campaign.calibration import joint_calibration_intake

        pin = joint_calibration_intake.register(
            row, root, result=result, compiler=compiler, capsule=capsule, state=state, profile=profile
        )
        if pin:
            row.setdefault("performance", {})["calibration_candidate_intake"] = {
                "status": "pending_review",
                "evidence_sha256": pin["sha256"],
                "dispatch_authorized": False,
                "fit_authorized": False,
            }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        row.setdefault("performance", {})["calibration_candidate_intake"] = "unavailable"
    name = capsule.get("name")
    member = profile["members"].get(name)
    tier = result.get("tiers", {}).get("L2", {})
    if (
        member is None
        or result.get("numeric", {}).get("status") != "pass"
        or tier.get("status") != "pass"
        or tier.get("engine") != "spike"
    ):
        return
    # Only compiler-authored products enter this registry. Merlin-only controls
    # and private operator references cannot acquire an entry via timing alone.
    own = result.get("compiler_source") or result.get("candidate_native_execution")
    if not own:
        return
    compiler = Path(compiler).resolve()
    root = Path(root).resolve()
    source = PERF.capsule_identity(capsule)
    candidate = hash_tree(compiler)["sha256"]
    key = document_sha256([source, candidate, str(root)])
    registry = Path(os.environ[joint_control.ENV]).parent / "calibration_registry"
    directory = registry / key
    if (directory / "entry.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    cap = C._write(directory / "capsule.json", {k: v for k, v in capsule.items() if k != "__dir__"})
    grade = C._write(directory / "result.json", result)
    performance = row.get("performance") or {}
    C._write(
        directory / "entry.json",
        dict(
            id=key,
            capsule=cap,
            result=grade,
            generated_root=str(root),
            compiler=str(compiler),
            compiler_sha256=candidate,
            source_group_sha256=member["source_group_sha256"],
            estimate_status=performance.get("status"),
            outside_calibrated_coverage=performance.get("timing_mode") == "untuned",
            estimated_interval=performance.get("estimated_interval"),
        ),
    )


def step(*, profile_pin, state_path, context, output, request_epoch=True, activation=None):
    """One bounded polling action; no gSIM until legal independent variants exist."""
    profile = PERF._read(profile_pin)
    selected = joint_control.selected()
    chosen = selected["policy"].get("native_calibration", {}) if selected else {}
    if any(chosen.get(key) != profile_pin.get(key) for key in ("path", "sha256")):
        raise ValueError("calibration supervisor profile differs from the immutable campaign policy")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    registry = Path(state_path).parent / "calibration_registry"
    entries = sorted(registry.glob("*/entry.json"))[:2049]
    if len(entries) > 2048:
        raise ValueError("completed calibration registry exceeds bounded review roster")
    # Completed rounds remain immutable; a failed held validation cannot become
    # training data in a retry with the same source group.
    used = set()
    attempted = set()
    for completed in output.glob("round-*/outcome.json"):
        report = json.loads(completed.read_text())
        plan_pin = report.get("plan")
        if plan_pin:
            past = PERF._read(plan_pin)["cases"]
            attempted.update(c["id"] for c in past)
            if report.get("status") in {"not_promoted", "limited_calibrated_provisional"}:
                used.update(c["source_group_sha256"] for c in past if c["split"] == "held")
    rows = [json.loads(p.read_text()) for p in entries]
    rows = [r for r in rows if r["source_group_sha256"] not in used and r["id"] not in attempted]
    # Model-only epochs retain the original policy while changing its separately
    # pinned evaluator selection. Never fit or promote against a stale incumbent.
    if activation == "future_requests":
        from merlin.development.backend.campaign.calibration.activation import current as accepted_selection

        current_pin = accepted_selection(state_path) or profile["performance_selection"]
    else:
        current_pin = selected.get("performance_selection") or profile["performance_selection"]
    current = PERF._read(current_pin)
    initial = PERF._read(profile["performance_selection"])
    from merlin.development.backend.campaign.calibration.joint_runtime_revision import validate

    validate(initial, current)
    from merlin.development.backend.campaign.calibration.joint_calibration_readiness import panel

    readiness = panel(profile, current, rows, witnesses=selected["policy"]["gsim"]["witness_capsules"])
    temporary = output / "panel-readiness.next.json"
    temporary.write_text(json.dumps(readiness, indent=2) + "\n")
    temporary.replace(output / "panel-readiness.json")
    allowed = set(readiness["admissible_registry_ids"])
    rows = [row for row in rows if row["id"] in allowed]
    active = [
        row
        for row in readiness["members"]
        if row["status"] not in {"not_in_current_source_selection", "outside_physical_witness_roster"}
    ]
    if active and all(row["status"] == "blocked_structural_model" for row in active):
        return dict(
            status="blocked_structural_model_admission",
            reason="Repair selected model structure before collecting price-fit variants; no gSIM scheduled.",
            panel_readiness=str(output / "panel-readiness.json"),
            denial_reasons=sorted({row["reason"] for row in active}),
            registered_variants=len(rows),
        )
    from merlin.development.backend.feedback.campaign.timing import choose

    by_runtime = {}
    for row in rows:
        chosen = choose(current, PERF.capsule_identity(PERF.selected_capsule(current, C._capsule(row["capsule"]))))
        by_runtime.setdefault(document_sha256(chosen["runtime"]), []).append(row)
    alternatives = [C.select_variant_cases(group) for group in by_runtime.values()]
    selection = next((v for v in alternatives if v["status"] == "selected"), C.select_variant_cases([]))
    if selection["status"] != "selected":
        return dict(
            selection,
            registered_variants=len(rows),
            panel_readiness=str(output / "panel-readiness.json"),
            reason="need two passing compiler variants in each of four independent reviewed small source groups",
        )
    key = document_sha256(selection["cases"])
    round_root = output / ("round-" + key)
    round_root.mkdir(exist_ok=True)
    plan_path = round_root / "plan-input.json"
    if not plan_path.exists():
        C.freeze_round(
            selection_pin=current_pin,
            cases=selection["cases"],
            parameters=profile["parameters"],
            output=plan_path,
            question=profile["question"],
            stage=profile.get("stage", "bootstrap"),
        )
    plan = F.pin(plan_path, max_bytes=16 << 20)
    if not profile.get("native_view"):
        return dict(
            status="blocked_native_runtime_view",
            round=str(round_root),
            reason="operator profile must pin the selected gSIM namespace inputs before physical measurement",
        )

    with physical_runtime(profile, context=context, directory=round_root) as backend:
        outcome = C.run_round(plan, context=context, backend=backend, output=round_root)

    raise_if_stopping()
    value = PERF._read(outcome)
    if value.get("timing_epochs"):
        if activation == "future_requests":
            from merlin.development.backend.campaign.calibration.activation import accept

            response = accept(
                state_path=state_path,
                selection_pin=outcome,
                incumbent_pin=current_pin,
                review_id="automatic-reviewed-local-native-" + key,
            )
            return dict(status="model_accepted_for_future_requests", selection=outcome, activation=response)
        if activation == "candidate_only" or (activation is None and not request_epoch):
            return dict(
                status="model_candidate_ready",
                selection=outcome,
                reason="Evidence-only lifecycle; selecting a new evaluator requires a separate drained epoch.",
            )
        response = C.request_promotion(
            outcome,
            previous_run=Path(state_path).parent,
            review_id="automatic-reviewed-local-native-" + key,
            output=round_root / "next-epoch.json",
        )
        return dict(status="model_epoch_requested", selection=outcome, epoch=response)
    return dict(status=value["status"], outcome=outcome)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "profile-sha256", "state", "descriptor", "repo", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stop-file", help="Host-owned cooperative watcher shutdown request")
    parser.add_argument(
        "--evidence-only", action="store_true", help="Retain accepted candidates without requesting an evaluator epoch"
    )
    parser.add_argument(
        "--activation",
        choices=("candidate_only", "future_requests", "drained_epoch"),
        help="Prospective model routing; future_requests never stops authoring",
    )
    args = parser.parse_args(argv)
    if not 5 <= args.poll_seconds <= 60:
        parser.error("poll interval must be 5..60 seconds")
    if args.evidence_only and args.activation not in (None, "candidate_only"):
        parser.error("--evidence-only conflicts with model activation")
    os.environ[joint_control.ENV] = str(Path(args.state).resolve())
    from merlin.development.backend.context import load_context

    context = load_context(args.descriptor, repo=args.repo)
    pin = dict(path=str(Path(args.profile).resolve()), sha256=args.profile_sha256)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with stop_scope(args.stop_file):
        while True:
            try:
                raise_if_stopping()
                if joint_control.stop_reason():
                    result = dict(status="native_drain_or_campaign_stop", reason=joint_control.stop_reason())
                else:
                    from merlin.development.backend.campaign.calibration.joint_mechanism_questions import (
                        step as questions_step,
                    )

                    try:
                        diagnostic = questions_step(state_path=args.state, context=context)
                    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                        diagnostic = dict(status="blocked_diagnostic_review", reason=str(error), fit_eligible=False)
                    options = {"activation": args.activation} if args.activation is not None else {}
                    result = step(
                        profile_pin=pin,
                        state_path=args.state,
                        context=context,
                        output=output,
                        request_epoch=not args.evidence_only,
                        **options,
                    )
                    result["mechanism_questions"] = diagnostic
            except StopRequested as error:
                result = dict(status="watcher_host_stopped", reason=str(error))
            except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
                result = dict(status="blocked", reason=str(error), exception=type(error).__name__)
            # Mutable status is descriptive; all accepted evidence stays pinned.
            temporary = output / "status.next.json"
            temporary.write_text(json.dumps(result, indent=2) + "\n")
            temporary.replace(output / "status.json")
            if args.once or result["status"] in {
                "model_epoch_requested",
                "model_candidate_ready",
                "native_drain_or_campaign_stop",
                "watcher_host_stopped",
            }:
                return 0
            # Responsive idle shutdown, without interrupting an owned native call.
            deadline = time.monotonic() + args.poll_seconds
            while time.monotonic() < deadline:
                try:
                    raise_if_stopping()
                except StopRequested:
                    break
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())


# ---- joint_calibration_lifecycle ---------------------------------------------------------------
# Shared host ownership of one optional calibration watcher per campaign.
#
# This does not create pools, start/reset campaign clocks, authorize observations
# or select a model. A stop request is cooperative: the currently owned native
# operation finishes before its next boundary observes the request.

_STOP = ContextVar("calibration_watcher_stop_file", default=None)


class StopRequested(RuntimeError):
    pass


@contextlib.contextmanager
def stop_scope(path):
    token = _STOP.set(Path(path) if path else None)
    try:
        yield
    finally:
        _STOP.reset(token)


def raise_if_stopping():
    path = _STOP.get()
    if path is not None and path.exists():
        raise StopRequested("calibration owner requested cooperative shutdown")


def _read(pin):
    path = F.canonical(Path(pin["path"]))
    before = F.pin(path, max_bytes=16 << 20)
    if before["sha256"] != pin["sha256"]:
        raise ValueError("calibration watcher profile changed")
    value = json.loads(path.read_bytes())
    if F.pin(path, max_bytes=16 << 20) != before:
        raise ValueError("calibration watcher profile changed during read")
    return value


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".next.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@dataclass
class Watcher:
    process: object | None
    report: dict
    directory: Path | None = None
    stop_file: Path | None = None
    lease: object | None = None
    shutdown_s: float = 1235

    @property
    def pid(self):
        return self.process.pid if self.process is not None else None

    def _release(self):
        if self.lease is not None:
            self.lease.close()
            self.lease = None

    def wait(self, *, timeout=None):
        if self.process is None:
            return 0
        result = self.process.wait(timeout=timeout)
        self._release()
        return result

    def poll(self):
        result = self.process.poll() if self.process is not None else 0
        if result is not None:
            self._release()
        return result

    def stop(self):
        """Drain the actual current operation; never kill its simulator child."""
        if self.process is None:
            return self.report
        if self.poll() is None:
            _write(self.stop_file, {"reason": "watcher host lifecycle finished", "at_epoch": time.time()})
            try:
                self.process.wait(timeout=self.shutdown_s)
            except subprocess.TimeoutExpired as error:
                # Keep the original lease/owner. The enclosing native guardian
                # must resolve this ownership failure before tearing down brokers.
                self.report.update(status="shutdown_unknown", reason="current native operation did not drain")
                _write(self.directory / "owner.json", self.report)
                raise RuntimeError("calibration watcher shutdown UNKNOWN; retain broker/native ownership") from error
        self._release()
        self.report.update(status="stopped", returncode=self.process.returncode, stopped_at=time.time())
        _write(self.directory / "owner.json", self.report)
        return self.report


def start(
    *,
    state_path,
    descriptor,
    repo,
    python,
    environment,
    output,
    directory,
    cpus=None,
    profile=None,
    request_epoch=True,
    poll_seconds=30,
    activation=None,
):
    """Spawn against the exact admitted profile and existing shared resources."""
    if state_path is None:
        return Watcher(None, dict(status="skipped_no_campaign", reason="no selected joint campaign"))
    state_path = F.canonical(Path(state_path))
    state = json.loads(state_path.read_bytes())
    selected = state.get("policy", {}).get("native_calibration")
    if not selected:
        return Watcher(None, dict(status="skipped_no_profile", reason="native_calibration is not selected"))
    if profile is not None and any(profile.get(k) != selected.get(k) for k in ("path", "sha256")):
        raise ValueError("watcher profile differs from the admitted native_calibration selection")
    profile_document = _read(selected)
    if state.get("started_at_epoch") is None:
        raise ValueError("calibration watcher requires the already-started admitted campaign clock")
    if not 5 <= poll_seconds <= 60:
        raise ValueError("calibration watcher polling must be five to sixty seconds")
    repo = F.canonical(Path(repo))
    descriptor = F.canonical(Path(descriptor))
    env = dict(environment)
    env["MERLIN_JOINT_CAMPAIGN_STATE"] = str(state_path)
    from merlin.development.backend.campaign.calibration import joint_calibration_admission as admission
    from merlin.development.backend.campaign.calibration import joint_initial_calibration_review as initial

    for key, variable in (
        ("initial_calibration_review", initial.ENV),
        ("native_calibration_admission_review", admission.ENV),
    ):
        pin = profile_document.get(key)
        if pin is not None:
            _read(pin)
            if env.get(variable) and json.loads(env[variable]) != pin:
                raise ValueError("watcher review environment differs from its selected profile")
            env[variable] = json.dumps(pin, sort_keys=True)
    pools = {k: env.get(k) for k in ("MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GSIM_SLOT_ROOT")}
    if any(not value or not Path(value).is_dir() for value in pools.values()):
        raise ValueError("calibration watcher requires the existing shared compute/gSIM pools")
    cpus = sorted(os.sched_getaffinity(0) if cpus is None else cpus)
    if not cpus or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("calibration watcher CPU selection is outside its admitted owner")
    lease = state_path.with_name(".joint_calibration_watcher.lock").open("a+")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return Watcher(None, dict(status="skipped_existing_owner", reason="campaign watcher already has an owner"))
    directory, output = Path(directory), Path(output)
    try:
        directory.mkdir(parents=True, exist_ok=False)
        stop_file = directory / "stop.json"
        command = [
            "/usr/bin/taskset",
            "-c",
            ",".join(map(str, cpus)),
            str(python),
            "-B",
            "-m",
            "merlin.development.backend.campaign.calibration.service",
            "--profile",
            selected["path"],
            "--profile-sha256",
            selected["sha256"],
            "--state",
            str(state_path),
            "--descriptor",
            str(descriptor),
            "--repo",
            str(repo),
            "--output",
            str(output),
            "--poll-seconds",
            str(poll_seconds),
            "--stop-file",
            str(stop_file),
        ]
        if activation is not None:
            if activation not in {"candidate_only", "future_requests", "drained_epoch"}:
                raise ValueError("unknown calibration activation mode")
            command += ["--activation", activation]
        elif not request_epoch:
            command.append("--evidence-only")
        with (directory / "watcher.log").open("xb") as log:
            child = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lease.fileno(),),
            )
        report = dict(
            status="watcher_started",
            pid=child.pid,
            argv=command,
            started_at=time.time(),
            profile=selected,
            state=str(state_path),
            cwd=str(repo),
            shared_pools=pools,
            request_epoch=(activation == "drained_epoch" if activation is not None else request_epoch),
            activation=activation,
            owner=str(directory / "owner.json"),
            status_file=str(output / "status.json"),
        )
        _write(directory / "owner.json", report)
        return Watcher(
            child,
            report,
            directory,
            stop_file,
            lease,
            max(120, state["policy"].get("gsim", {}).get("per_job_s", 1200)) + poll_seconds + 5,
        )
    except BaseException:
        lease.close()
        raise


def start_for_author(*, run_dir, context, round_index):
    """Ordinary launch shares the campaign watcher and pools; failure is advisory.

    The author clock has already started. No clock, model, grant or budget is
    changed here. New ordinary launches use nonstop future-request activation.
    """
    import secrets
    import sys

    from merlin.development.backend.campaign import control as joint_control

    run_dir = Path(run_dir)
    try:
        watcher = start(
            state_path=os.environ.get(joint_control.ENV),
            descriptor=context.descriptor,
            repo=context.repo,
            python=sys.executable,
            environment=dict(os.environ),
            output=run_dir / "calibration_supervisor",
            directory=run_dir / "calibration_watchers" / f"round-{round_index}-{secrets.token_hex(6)}",
            activation="future_requests",
        )
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        watcher = Watcher(None, dict(status="blocked_watcher_start", reason=str(error), changes_authoring=False))
    if os.environ.get(joint_control.ENV):
        _write(run_dir / "calibration_watcher_status.json", watcher.report)
    return watcher


# ---- joint_calibration_launcher ----------------------------------------------------------------
# Explicit operator-pinned launcher corrections; measurement policy stays fixed.

ENV = "MERLIN_NATIVE_CALIBRATION_LAUNCHER_REVIEW"


def selected(view):
    """Keep the selected launcher unless a separately pinned correction is present."""
    original = Path(view["bwrap"])
    raw = os.environ.get(ENV)
    if not raw:
        return original
    receipt = json.loads(raw)
    path = Path(receipt["path"])
    if F.digest(path) != receipt["sha256"]:
        raise ValueError("native calibration launcher review changed")
    review = json.loads(path.read_text())
    if (
        review.get("schema") != "merlin.native_launcher_correction.v1"
        or review.get("original", {}).get("path") != str(original)
        or F.digest(original) != review["original"]["sha256"]
        or review.get("namespace_policy") != "unchanged_strict_kernel_verified"
        or review.get("measurement_policy_changed") is not False
        or not review.get("review_id")
    ):
        raise ValueError("native calibration launcher correction is not applicable")
    replacement = Path(review["replacement"]["path"])
    if F.digest(replacement) != review["replacement"]["sha256"] or not os.access(replacement, os.X_OK):
        raise ValueError("reviewed native calibration launcher changed")
    return replacement


@contextlib.contextmanager
def physical_runtime(profile, *, context, directory):
    """Original selected engine/namespace, shared by fitting and diagnostics."""
    from merlin.runtime.backends.registry import get_backend
    from merlin.runtime.views.native_simulator import executing_native_runtime, prepare_native_runtime_view

    backend = get_backend(context.target)
    view = profile.get("native_view")
    if not view:
        raise ValueError("operator profile must pin native gSIM namespace inputs")
    launcher = selected(view)
    for row in view["readonly_roots"]:
        if (
            "expected_members" in row
            and sorted(p.name for p in Path(row["source"]).iterdir()) != row["expected_members"]
        ):
            raise ValueError("native namespace directory membership changed")
        if row.get("sha256") and F.digest(Path(row["source"])) != row["sha256"]:
            raise ValueError("native namespace source changed")
    roots = tuple((Path(row["source"]), Path(row["target"])) for row in view["readonly_roots"])
    owner = prepare_native_runtime_view(
        Path(directory) / ("native-view-" + str(time.time_ns())),
        backend=backend,
        engine=backend.gsim_path(),
        readonly_roots=roots,
        output_root=Path(view["output_root"]),
        bwrap=launcher,
        max_files=view["max_files"],
        max_bytes=view["max_bytes"],
        max_console_bytes=view["max_console_bytes"],
    )
    try:
        with executing_native_runtime(owner):
            yield backend
    finally:
        owner.close()
