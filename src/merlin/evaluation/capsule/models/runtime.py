"""Capsule evaluation: model inputs."""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from merlin.development.targetgen.packaging.loading import Package
from merlin.evaluation.capsule.evidence.source_custody import (
    _MODEL_NUMERIC_UNSET,
    _SUITE_MODEL_PIN,
    CONTRACT_VERSION,
    _pin_model_capsule,
)
from merlin.evaluation.capsule.models.candidates import _grade_candidate_model_capsule_inline
from merlin.evaluation.capsule.models.grading import _grade_model_capsule_inline

# ---- model_inputs ------------------------------------------------------------------------------

_MODEL_HOST_SNAPSHOT_ROOT_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"


_MODEL_HOST_SNAPSHOT_REQUIRED_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"


_MODEL_HOST_SNAPSHOT_RECORD_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD"


def _resolve_model_host_lane(target: str, dtype: str):
    """Return ``(experiment, package_path, identity)`` for targeted whole-model compilation.

    Kept as a small seam because descriptor resolution, package-content validation and datatype
    compatibility are one fail-closed decision; callers must not independently recreate part of it.
    """
    from merlin.target.registry.descriptor_loading import descriptor_path, load_target_experiment

    snapshot_repo, snapshot_identity = _verified_model_host_lane_snapshot()
    descriptor = _model_frozen_input(descriptor_path(target), snapshot_identity)
    if not descriptor.is_file():
        raise ValueError(f"target {target!r} has no experiment descriptor at {descriptor}")
    experiment = load_target_experiment(descriptor)
    if experiment.target != target:
        raise ValueError(f"descriptor {descriptor} names target {experiment.target!r}, not {target!r}")
        # SELECT the lane by dtype, rather than validating whichever single lane the descriptor happened to
        # declare. The cross-check below already refused a mismatch, so a target needing two precisions had
        # no way to express it and simply failed for one of them; the matrix makes the selection the
        # descriptor's to state, and the check below stays as the belt-and-braces that catches a package
        # whose knobs drifted from the profile it is filed under.
    package, identity = experiment.resolve_host_lane(root=snapshot_repo, dtype=dtype)
    if snapshot_identity is not None:
        identity["run_snapshot"] = snapshot_identity

        # An explicit package bypasses compile_rvv's datatype-aware default selection. Reject a known
        # cross-datatype substitution: otherwise an int8 model with an fp32 package sets
        # ``int8_compute=False`` and the numeric outcome is about a different host program. Dtypes such as
        # fp8 deliberately have no scalar strategy; compile_rvv derives their scalar lane from captured IR,
        # so only a strategy it can decide here is gated here.
    from merlin.cli import compile_workload as _compile_cli

    expected_strategy = _compile_cli._DTYPE_STRATEGY.get(dtype)
    if expected_strategy is not None and identity["dtype_strategy"] != expected_strategy:
        raise ValueError(
            f"descriptor host package declares dtype_strategy={identity['dtype_strategy']!r}, but "
            f"model compile_dtype={dtype!r} requires {expected_strategy!r}"
        )
    return experiment, package, identity


def _resolve_model_numeric_policy(target: str | None) -> tuple[dict | None, dict | None]:
    policy, identity, _ = resolve_numeric_policy(target)
    return policy, identity


def resolve_numeric_policy(
    target: str | None, *, descriptor: Path | None = None
) -> tuple[dict | None, dict | None, Any]:
    """Return policy, byte identity and its descriptor through the already-bound snapshot.

    Consumers deriving a binding must use the returned experiment, not reload its mutable original
    path after policy verification. This does not attest external capability facts or tools.
    """
    if target is None:
        return None, None, None
    from merlin.common.paths import repo_root
    from merlin.experiments.corpus.numeric_policy import load_declared_numeric_policy, numeric_profile_path
    from merlin.target.registry.descriptor_loading import descriptor_path, load_target_experiment

    _, snapshot = _verified_model_host_lane_snapshot()
    source = Path(descriptor) if descriptor is not None else descriptor_path(target)
    experiment = load_target_experiment(_model_frozen_input(source, snapshot))
    if experiment.target != target:
        raise ValueError("model numerical-policy descriptor names a different target")
    original_repo = Path(snapshot["source_repo"]) if snapshot is not None else repo_root()
    declared = numeric_profile_path(experiment.numeric_profile, repo=original_repo)
    frozen = None
    if declared is not None and snapshot is not None:
        frozen = _model_frozen_input(declared, snapshot)
    policy, identity = load_declared_numeric_policy(
        experiment,
        repo=original_repo,
        frozen_profile=frozen,
        frozen_resolver=(lambda source: _model_frozen_input(source, snapshot)) if snapshot is not None else None,
    )
    return policy, identity, experiment


def _model_frozen_input(source: Path, snapshot: dict | None) -> Path:
    """Map only declared frozen bytes after rechecking the host's expected marker."""
    if snapshot is None:
        return source
    from merlin.development.targetgen.sandbox import grants as BW

    root = Path(snapshot["root"])
    marker = root / "snapshot.json"
    import hashlib

    raw = marker.read_bytes()
    if hashlib.sha256(raw).hexdigest() != snapshot["manifest_sha256"]:
        raise ValueError("model host-lane snapshot marker changed during input resolution")
    bundle = _model_snapshot_bundle(json.loads(raw))
    [frozen] = BW.snapshot_input_paths(root.parent / "workspace", bundle, [source], repo=Path(snapshot["source_repo"]))
    return frozen


def _model_snapshot_bundle(manifest: dict) -> dict:
    """Declaration membership from a marker already bound to host-owned provenance."""
    return {key: [{"path": path} for path in manifest.get(key, [])] for key in ("allowed", "host_inputs")}


def _verified_model_host_lane_snapshot() -> tuple[Path | None, dict[str, Any] | None]:
    """Return the verified run-snapshot repo root used for host-lane grading.

    A bwrap experiment sets ``*_REQUIRED`` and points ``*_ROOT`` at its host-only
    ``bundle_inputs`` directory. The aggregate was recomputed immediately before export; resolving and
    hashing the declared host package below joins the bytes mounted for the agent to the bytes passed
    into ``compile_model``. Using the live worktree instead would let a post-launch edit change only the
    grader's compiler.

    Callers outside the bwrap experiment remain compatible: with neither variable set, the descriptor
    resolves against the live repository exactly as before. A declared/required snapshot never falls
    back to that path.
    """
    raw_required = os.environ.get(_MODEL_HOST_SNAPSHOT_REQUIRED_ENV, "").strip().lower()
    if raw_required not in ("", "0", "false", "no", "off", "1", "true", "yes", "on"):
        raise ValueError(f"{_MODEL_HOST_SNAPSHOT_REQUIRED_ENV} has invalid boolean value {raw_required!r}")
    required = raw_required in ("1", "true", "yes", "on")
    raw_root = os.environ.get(_MODEL_HOST_SNAPSHOT_ROOT_ENV, "").strip()
    if not raw_root:
        if required or os.environ.get(_MODEL_HOST_SNAPSHOT_RECORD_ENV):
            raise ValueError(
                f"bwrap model grading requires {_MODEL_HOST_SNAPSHOT_ROOT_ENV}; no verified run snapshot was provided"
            )
        return None, None

    lexical_root = Path(raw_root)
    if not lexical_root.is_absolute() or lexical_root.is_symlink():
        raise ValueError(f"{_MODEL_HOST_SNAPSHOT_ROOT_ENV} must name a non-symlink absolute directory")
    try:
        root = lexical_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"model host-lane run snapshot is missing at {lexical_root}") from exc
    marker = root / "snapshot.json"
    if not root.is_dir() or marker.is_symlink() or not marker.is_file():
        raise ValueError(f"model host-lane run snapshot is incomplete or unsafe at {root}")
    try:
        manifest = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"model host-lane run snapshot marker is unreadable at {marker}") from exc
    from merlin.development.targetgen.sandbox import grants as BW

    BW.require_snapshot_ownership(manifest)
    try:
        expected = json.loads(os.environ.get(_MODEL_HOST_SNAPSHOT_RECORD_ENV, ""))
        source_repo = Path(manifest["repo"])
        if not source_repo.is_absolute() or ".." in source_repo.parts:
            raise ValueError("invalid original snapshot repository")
        BW.verify_snapshot_binding(
            root.parent / "workspace", _model_snapshot_bundle(manifest), expected, repo=source_repo
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("model host-lane snapshot lacks matching host-owned provenance") from exc

        # Preserve the bound aggregate identity here. Mapping declaration/profile bytes through
        # snapshot_input_paths verifies the payload; HostLane.resolve additionally checks the selected
        # compiler package before and after compilation.
    digest = manifest.get("content_sha256")
    n_files = manifest.get("n_files")
    n_bytes = manifest.get("n_bytes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != digest.lower()
        or any(ch not in "0123456789abcdef" for ch in digest)
        or not isinstance(n_files, int)
        or n_files < 1
        or not isinstance(n_bytes, int)
        or n_bytes < 1
    ):
        raise ValueError("model host-lane run snapshot marker has invalid aggregate identity")

    repo = root / "repo"
    if repo.is_symlink() or not repo.is_dir():
        raise ValueError(f"model host-lane run snapshot has no safe repository payload at {repo}")
    resolved_repo = repo.resolve(strict=True)
    if root not in resolved_repo.parents:
        raise ValueError(f"model host-lane run snapshot repository escapes {root}")
    return resolved_repo, {
        "root": str(root),
        "repo_root": str(resolved_repo),
        "content_sha256": digest,
        "n_files": n_files,
        "n_bytes": n_bytes,
        "version": 4,
        "manifest_sha256": expected["manifest_sha256"],
        "source_repo": manifest.get("repo"),
    }


# ---- processes ---------------------------------------------------------------------------------
# Owned descendant discovery and termination for budgeted grading children.


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this process when its parent goes away.

    The budgeted whole-model grade runs in its OWN SESSION so the parent can kill the process group
    when the ceiling expires. That protects against a grade overrunning; it does nothing when the
    PARENT is killed, because the parent's cleanup never runs. MEASURED (2026-08-30): stopping a
    regrade left the model-grade child and its Verilator alive on a shared host, reparented to init,
    with nothing left that knew to reap them.

    Set from inside the CHILD rather than through ``preexec_fn``: the parent grades capsules on a
    ThreadPoolExecutor, and ``preexec_fn`` is documented as unsafe in a threaded process. The cost is a
    small race -- the parent could die during interpreter startup -- which the getppid() re-check
    below closes.
    """
    try:
        import ctypes
        import signal as _sig

        ppid = os.getppid()
        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, _sig.SIGKILL)
        if os.getppid() != ppid:  # already reparented: the signal will never come
            os._exit(1)
    except Exception:  # noqa: BLE001 — a best-effort safeguard, never a blocker
        pass


def _descendants(pid: int) -> list[int]:
    """Every live descendant of *pid*, deepest first, read from /proc.

    Must be collected BEFORE any signal is sent: once a process is orphaned its parent link is gone.
    A process GROUP is not enough. MEASURED (2026-08-30): a budgeted whole-model grade's Verilator
    child had called ``setsid`` of its own, so it sat in neither the grade child's group nor its
    session; ``killpg`` reaped the python and left the simulator running with PPID 1, holding a core
    on a shared host.
    """
    kids: dict[int, list[int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            # /proc/<pid>/stat is "pid (comm) state ppid ..."; comm may itself contain spaces and
            # parentheses, so split after the LAST ')' rather than on whitespace from the left.
            ppid = int((entry / "stat").read_text().rpartition(")")[2].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        kids.setdefault(ppid, []).append(int(entry.name))
    out: list[int] = []
    frontier = list(kids.get(pid, ()))
    while frontier:
        out.extend(frontier)
        frontier = [g for p in frontier for g in kids.get(p, ())]
    out.reverse()  # deepest generation first
    return out


def _running(pid: int) -> bool:
    """True while *pid* is a live process. A ZOMBIE is not: it has exited and merely awaits a wait(),
    so ``/proc/<pid>`` still exists and a liveness check written that way waits out its whole grace."""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _kill_tree(pid: int, *, grace: float = 5.0) -> None:
    """SIGTERM then SIGKILL *pid* and every descendant it had when we looked."""
    import signal as _signal

    victims = _descendants(pid) + [pid]
    for sig in (_signal.SIGTERM, _signal.SIGKILL):
        for victim in victims:
            try:
                os.kill(victim, sig)
            except OSError:  # already gone, or not ours
                pass
        try:  # and the group, for anything spawned since the scan
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not any(_running(v) for v in victims):
                return
            time.sleep(0.1)


# ---- model_runtime -----------------------------------------------------------------------------
# Capsule evaluation: model runtime.

_CHILD_GUARD = (
    "import ctypes,os,signal,sys;"
    "ctypes.CDLL('libc.so.6').prctl(1, signal.SIGKILL);"  # PR_SET_PDEATHSIG
    "os.getppid()==1 and os._exit(1);"  # already orphaned: no signal coming
)


_CHILD_PREAMBLE = _CHILD_GUARD + (
    "__import__('merlin.evaluation.capsule.runner', fromlist=['main']).main(sys.argv[1:])"
)


def model_budget_seconds(*, override: float | None = None) -> float | None:
    """Wall-clock ceiling for ONE whole-model capsule, or ``None`` for unlimited (the default).

    Unlimited is the right default for an operator certification run: a whole model on a
    cycle-accurate oracle legitimately takes hours, and a ceiling that silently truncates it would
    hide coverage rather than buy anything. It is the WRONG default inside an agent loop's per-round
    gate, which is why the loop sets one explicitly.

    The ``timeout`` threaded through this file is a PER-STEP subprocess ceiling. A whole-model grade
    makes many such calls, so their sum is unbounded and a step timeout can never bound the capsule.
    MEASURED (2026-08-29, gemmini arm-4 calibration ``merlincirct_defcal1``): the agent round finished
    in 40 min, the capstone then ran 5 h 30 m past the round's own 4 h timeout, wrote not one byte into
    its run directory, and the round never graded. Its own step timeout was 900 s.
    """
    if override is not None:
        import math

        if type(override) not in (int, float) or not math.isfinite(override) or override <= 0:
            raise ValueError("explicit model budget must be finite and positive")
        inherited = model_budget_seconds()
        return min(float(override), inherited) if inherited else float(override)
    raw = os.environ.get("MERLIN_MODEL_BUDGET_S", "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v > 0 else None


def _grade_model_capsule(
    capsule: dict,
    *,
    target: str | None = None,
    timeout: int,
    package_dir: str | Path | None = None,
    budget_s: float | None = None,
    numeric_policy=_MODEL_NUMERIC_UNSET,
    numeric_policy_identity: dict | None = None,
) -> dict:
    """Serialize whole-model grades whose nested mesh artifacts use shared global run paths.

    ``compile_model`` currently keys ``mesh_run`` and ``mesh_verify`` artifacts by layer shape/index,
    not by the outer capsule/submission.  Two processes can therefore overwrite the exact LLVM/ELF and
    oracle evidence the other is still producing.  A cross-process target lock makes those artifacts
    single-writer until the paths themselves gain an outer-grade namespace.  This deliberately slows
    only model capsules; ordinary op/slice grading remains parallel.
    """
    import fcntl
    import tempfile

    safe_target = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(target or "none"))
    lock_root = Path(tempfile.gettempdir()) / f"merlin-model-grade-locks-{os.getuid()}"
    lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = lock_root / f"{safe_target}.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        result = _grade_model_capsule_unlocked(
            capsule,
            target=target,
            timeout=timeout,
            package_dir=package_dir,
            budget_s=budget_s,
            numeric_policy=numeric_policy,
            numeric_policy_identity=numeric_policy_identity,
        )
        result["model_grade_serialization"] = {
            "scope": f"target:{target}",
            "reason": "nested mesh_run/mesh_verify artifact paths are shape/index keyed",
        }
        return result


def _grade_model_capsule_unlocked(
    capsule: dict,
    *,
    target: str | None = None,
    timeout: int,
    package_dir: str | Path | None = None,
    budget_s: float | None = None,
    numeric_policy=_MODEL_NUMERIC_UNSET,
    numeric_policy_identity: dict | None = None,
    candidate_context: dict | None = None,
    candidate_pkg: Package | None = None,
    candidate_source_pinned: object | None = None,
) -> dict:
    """Grade the original whole-model path under its aggregate capsule budget.

    Budget exhaustion is absence of evidence, never a submission pass/fail.
    An explicit absent/nonpositive budget preserves the original inline route.
    Otherwise a separate child session owns descendant execution so expiry can
    terminate its process tree. The child receives source-pinned JSON data;
    original profile executions restore their sealed host selections in the
    child before any package call; foreign live owners remain inadmissible.
    """
    budget = model_budget_seconds() if budget_s is None else budget_s

    import subprocess as _sp
    import tempfile as _tf

    def _stopped(detail: str, status: str = "incomplete", **extra) -> dict:
        return {
            "capsule": capsule["name"],
            "kind": "model",
            "label": capsule.get("label"),
            "contract_version": CONTRACT_VERSION,
            "status": status,
            **(
                {
                    "legacy_model_diagnostic": {
                        "status": "not_run",
                        "scope": "runner-owned host-dispatch graph; not candidate whole-program evidence",
                    }
                }
                if candidate_context is not None
                else {}
            ),
            **extra,
            "failure": {"plane": "model", "category": "NOT_RUN_IS_NOT_PASS", "detail": detail},
        }

    if candidate_context is None and numeric_policy is _MODEL_NUMERIC_UNSET:
        try:
            numeric_policy, numeric_policy_identity = _resolve_model_numeric_policy(target)
        except Exception as exc:  # noqa: BLE001 — unavailable declared inputs cannot authorize a child
            return _stopped(f"declared model numerical policy unavailable: {type(exc).__name__}: {exc}")
    if not budget or budget <= 0:
        if candidate_context is not None:
            return _grade_candidate_model_capsule_inline(
                capsule,
                target=target,
                timeout=timeout,
                package_dir=package_dir,
                context=candidate_context,
                pkg=candidate_pkg,
            )
        return _grade_model_capsule_inline(
            capsule,
            target=target,
            timeout=timeout,
            package_dir=package_dir,
            numeric_policy=numeric_policy,
            numeric_policy_identity=numeric_policy_identity,
        )

    from merlin.development.backend.feedback.execution import compiler_profile_child as _child_profile
    from merlin.development.backend.feedback.execution import model_children as _merlin_model_children_1

    with _tf.TemporaryDirectory(prefix="model_grade_", dir=_child_profile.model_child_temporary_root()) as _td:
        spec_p, out_p = Path(_td) / "spec.json", Path(_td) / "result.json"
        # The child may outlive the materializer's cache generation. Keep the
        # capsule bytes under the grade's own lifetime before handing it a path.
        child_capsule = (
            capsule
            if candidate_context is not None and candidate_source_pinned is _SUITE_MODEL_PIN
            else _pin_model_capsule(capsule, Path(_td) / "capsule")
        )
        spec_p.write_text(
            json.dumps(
                {
                    "capsule": child_capsule,
                    "target": target,
                    "timeout": timeout,
                    "package_dir": str(package_dir) if package_dir else None,
                    "numeric_policy": numeric_policy if candidate_context is None else None,
                    "numeric_policy_identity": numeric_policy_identity,
                    "candidate_context": candidate_context,
                }
            ),
            encoding="utf-8",
        )
        # `-c` rather than `-m`: the preamble sets PR_SET_PDEATHSIG in the first milliseconds, before
        # importing merlin. Importing first leaves a multi-second window in which the child is
        # unprotected, and a parent killed inside that window orphans it -- measured, and the reason
        # this is a preamble and not a call at the top of main().
        cmd = [sys.executable, "-c", _CHILD_PREAMBLE, "--model-grade", str(spec_p), "--model-grade-out", str(out_p)]
        started = time.monotonic()
        from merlin.common.process.cpu_pool import child_environment

        _child_env = child_environment()  # sealed below: it must already carry the caller's lease class
        # A budgeted model grade runs in another interpreter, so the parent's ContextVar cannot carry
        # the capsule attribution into it. The pass recorder already accepts this explicit environment
        # fallback; set it on the child only so concurrent model grades cannot cross-attribute calls.
        if _child_env.get("MERLIN_PASS_LOG"):
            _child_env["MERLIN_PASS_LOG_CAPSULE"] = str(capsule["name"])
            _child_env["MERLIN_PASS_LOG_REQUIREMENTS"] = json.dumps(list(capsule.get("pass_requirements") or []))
        with _merlin_model_children_1.prepare_model_child(
            spec_path=spec_p, environment=_child_env, package_dir=package_dir
        ) as transport:
            proc = _sp.Popen(
                _merlin_model_children_1._model_child_command(
                    [*cmd, *(transport.argv if transport else ())], environment=_child_env
                ),
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                text=True,
                env=_child_env,
                start_new_session=True,
                **({"pass_fds": transport.pass_fds} if transport else {}),
            )
            if transport is not None:
                transport.launched(proc, proc.args)
            try:
                tail = (proc.communicate(timeout=budget)[0] or "")[-2000:]
            except _sp.TimeoutExpired:
                elapsed = time.monotonic() - started
                _kill_tree(proc.pid)
                try:
                    proc.wait(timeout=10)
                except _sp.TimeoutExpired:
                    pass
                    # `budget_exhausted`, not `incomplete`: nothing about the submission was measured, so it
                    # belongs in neither bucket. Scoring it as a failure would make all_pass unreachable and
                    # cost an agent loop its only early exit -- see NOT_MEASURED_STATUSES.
                return _stopped(
                    f"whole-model grade exceeded its {budget:.0f}s budget (ran {elapsed:.0f}s) and was "
                    f"stopped. Raise MERLIN_MODEL_BUDGET_S, or clear it for no ceiling, to grade it. "
                    f"NOT a verdict on the submission.",
                    status="budget_exhausted",
                    model_budget_s=budget,
                    elapsed_s=round(elapsed, 1),
                )
            if not out_p.is_file():
                return _stopped(f"whole-model grade process exited {proc.returncode} without a result: {tail[-400:]}")
            try:
                result = json.loads(out_p.read_text(encoding="utf-8"))
                return transport.verify_result(result, completed_process=proc) if transport else result
            except (OSError, json.JSONDecodeError) as exc:
                return _stopped(f"whole-model grade result unreadable: {type(exc).__name__}: {exc}")


# ---- model_context -----------------------------------------------------------------------------
# Original submitted-model source custody and grader-context assembly.
#
# This stage selects the original model source and supervision owners. It selects no
# backend or new executor and supplies no numerical, tier or timing authority.


def grade_selected_model(
    capsule,
    *,
    target,
    package_dir,
    paths,
    runs_root,
    run_id,
    cfg,
    contract,
    pkg,
    adapters,
    budget_s,
    timeout,
    model_execution_selection,
    checked_source_diagnostic,
):
    from merlin.evaluation.capsule.evidence.certificates import _adapter_readback_policy
    from merlin.evaluation.capsule.evidence.source_custody import _SUITE_MODEL_PIN, _pin_model_capsule

    suite_pinned = capsule.get("__suite_model_pin") is _SUITE_MODEL_PIN
    readback_policy = _adapter_readback_policy(adapters)
    candidate_capsule = {key: value for key, value in capsule.items() if key != "__suite_model_pin"}
    if not suite_pinned and candidate_capsule.get("__dir__"):
        private_root = paths.run_path / ".private_model_sources"
        if private_root.resolve().is_relative_to(Path(candidate_capsule["__dir__"]).resolve()):
            raise ValueError("candidate model source snapshot cannot be nested inside its source")
        if private_root.exists() or private_root.is_symlink():
            raise ValueError(f"candidate model source snapshot already exists: {private_root}")
        private_root.mkdir(mode=0o700)
        candidate_capsule = _pin_model_capsule(candidate_capsule, private_root / "capsule")
    checked_scope = nullcontext()
    if checked_source_diagnostic is not None:
        from merlin.development.targetgen.execution.host.checked_source_diagnostics import (
            scoped_checked_source_diagnostic,
        )

        checked_scope = scoped_checked_source_diagnostic(checked_source_diagnostic)
    with checked_scope:
        return _grade_model_capsule_unlocked(
            candidate_capsule,
            target=target,
            timeout=timeout,
            package_dir=package_dir,
            budget_s=budget_s if checked_source_diagnostic is None else min(budget_s, timeout),
            candidate_pkg=pkg,
            candidate_source_pinned=_SUITE_MODEL_PIN,
            candidate_context={
                "runs_root": str(Path(runs_root).resolve()),
                "run_id": run_id,
                "suite": cfg.suite,
                "dtype": cfg.dtype,
                "contract": str(contract) if contract is not None else None,
                "fourth_output_name": cfg.fourth_output_name,
                "reachable_oracle_tiers": sorted(adapters),
                "package_prebuilt": pkg is not None,
                "owned_host_build": (
                    json.loads(os.environ["MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION"])
                    if os.environ.get("MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION")
                    else None
                ),
                **(
                    {"public_program_execution": model_execution_selection}
                    if model_execution_selection is not None
                    else {}
                ),
                **({"readback_policy": readback_policy.record()} if readback_policy is not None else {}),
                **(
                    {"checked_source_diagnostic": checked_source_diagnostic.record()}
                    if checked_source_diagnostic is not None
                    else {}
                ),
            },
        )
