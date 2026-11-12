"""Capsule evaluation: model runtime."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from merlin.development.targetgen.package_runtime import Package
from merlin.evaluation.capsule.identity import (
    _MODEL_NUMERIC_UNSET,
    _SUITE_MODEL_PIN,
    CONTRACT_VERSION,
    _pin_model_capsule,
)
from merlin.evaluation.capsule.model import _grade_model_capsule_inline
from merlin.evaluation.capsule.model_candidates import _grade_candidate_model_capsule_inline
from merlin.evaluation.capsule.model_inputs import _resolve_model_numeric_policy
from merlin.evaluation.capsule.processes import _kill_tree as _kill_tree

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

    from merlin.development.backend.feedback import compiler_profile_child as _child_profile

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
        from merlin.common.cpu_pool import child_environment

        _child_env = child_environment()  # sealed below: it must already carry the caller's lease class
        # A budgeted model grade runs in another interpreter, so the parent's ContextVar cannot carry
        # the capsule attribution into it. The pass recorder already accepts this explicit environment
        # fallback; set it on the child only so concurrent model grades cannot cross-attribute calls.
        if _child_env.get("MERLIN_PASS_LOG"):
            _child_env["MERLIN_PASS_LOG_CAPSULE"] = str(capsule["name"])
            _child_env["MERLIN_PASS_LOG_REQUIREMENTS"] = json.dumps(list(capsule.get("pass_requirements") or []))
        with _child_profile.prepare_model_child(
            spec_path=spec_p, environment=_child_env, package_dir=package_dir
        ) as transport:
            proc = _sp.Popen(
                _child_profile._model_child_command(
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
