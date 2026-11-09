"""Capsule evaluation: suite."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from merlin.common.jsonio import readable_json
from merlin.evaluation.capsule.eligibility import _gate_counts, _split_ineligible, suite_for
from merlin.evaluation.capsule.identity import _SUITE_MODEL_PIN, _pin_model_capsule
from merlin.evaluation.capsule.model_suite import (
    _candidate_boundary_readiness,
    _model_gate_operation_passed,
    _original_capsule_worker_scope,
    _run_streaming_model_gate,
)
from merlin.evaluation.capsule.operation import run_capsule
from merlin.targetgen import tier_policy as _tier_policy
from merlin.targetgen.capsule_common import NOT_MEASURED_STATUSES
from merlin.targetgen.oot_runner import build_package, integrity_scan, load_package

_CALIBRATION_CAP = 3


def run_suite(*args, **kwargs) -> list[dict]:
    """One grade, one set of hardware revisions — then :func:`_run_suite`.

    A grade is a single measurement event, so every capsule in it must be attributed to the same
    checkout revisions; observing them per capsule allows two capsules in one grade to disagree, and
    costs four ``git`` subprocesses each (one of them ``status --porcelain`` over a tree with a large
    ignored output directory). Measured on a 2-capsule grade before this: 3.9 s in ``git`` and 6.6 s
    re-parsing the pin registry, out of 20.6 s total. The suite runner owns the measurement, so it is
    the right place to open the scope — a library function opening one would cache beyond the event it
    describes.
    """
    import tempfile as _tf
    from contextlib import nullcontext

    from merlin.common import provenance as PROV

    # A trusted grader may retain its source closure so the result's native
    # file pins remain independently rehashable after this call returns. The
    # default stays temporary for direct/agent-facing suite probes. A caller's
    # post-suite audit runs while even that temporary closure still exists.
    model_snapshot_root = kwargs.pop("model_snapshot_root", None)
    post_suite = kwargs.pop("post_suite", None)

    def finish(results):
        if post_suite is not None:
            post_suite(results)
        return results

    with PROV.observation_scope():
        # Freeze every model BEFORE package build and the op phase. A staged
        # corpus generation can otherwise be collected while those run, long
        # before the first model child starts. The suite owns the copies until
        # all model results have landed, including in the budgeted-child path.
        capsules = args[0] if args else kwargs["capsules"]
        if not any(c.get("kind") == "model" and c.get("__dir__") for c in capsules):
            return finish(_run_suite(*args, **kwargs))
        if model_snapshot_root is None:
            snapshot_scope = _tf.TemporaryDirectory(prefix="merlin-model-suite-")
        else:
            root = Path(model_snapshot_root)
            if not root.is_absolute():
                raise ValueError("model snapshot root must be absolute")
            if root.is_symlink() or root.parent.resolve() != root.parent:
                raise ValueError("model snapshot root may not traverse a symlink")
            root.mkdir(parents=True, mode=0o700, exist_ok=True)
            if root.is_symlink() or root.resolve(strict=True) != root or not root.is_dir():
                raise ValueError("model snapshot root must be an ordinary canonical directory")
            if root.stat().st_uid != os.geteuid():
                raise ValueError("model snapshot root must be owned by the grader")
            root.chmod(0o700)
            snapshot_scope = nullcontext(_tf.mkdtemp(prefix="merlin-model-suite-", dir=root))
        with snapshot_scope as temp:
            frozen = []
            for i, capsule in enumerate(capsules):
                if capsule.get("kind") != "model":
                    frozen.append(capsule)
                    continue
                pinned = _pin_model_capsule(capsule, Path(temp) / str(i))
                if bool((capsule.get("semantic") or {}).get("must_accelerate")):
                    pinned["__suite_model_pin"] = _SUITE_MODEL_PIN
                frozen.append(pinned)
            # CAS model pinning can alter frozen input inode link metadata.
            # Refresh only through the original owner's full native admission,
            # before _run_suite creates any workers. Legacy owners stay fresh.
            from merlin.targetgen.package_runtime import active_package_executor

            executor = active_package_executor()
            refresh = getattr(executor, "staged_suite_snapshot_scope_v1", None)
            with refresh() if callable(refresh) else nullcontext():
                if args:
                    results = _run_suite(frozen, *args[1:], **kwargs)
                else:
                    results = _run_suite(**{**kwargs, "capsules": frozen})
                return finish(results)


def _run_suite(
    capsules: list[dict],
    package_dir: str | Path,
    *,
    runs_root: str | Path,
    contract: str | Path | None = None,
    oracle_adapters: dict[str, Callable] | None = None,
    timeout: int = 600,
    max_workers: int = 1,
    on_capsule_complete: Callable[[dict], None] | None = None,
    defer_pending: Callable[[], str | None] | None = None,
    defer_running: bool = True,
    model_gate_continuations=(),
    model_gate_prerequisites=None,
    target: str | None = None,
    suite: str | None = None,
    dtype: str = "i8xi8_i32",
    config=None,
    perf_extractor: Callable | None = None,
    no_oracle: bool = False,
) -> list[dict]:
    """Run many capsules through one package (building/integrity-scanning it once).

    ``max_workers > 1`` fans the (independent) per-capsule runs out across a ThreadPoolExecutor — each
    capsule has its own ``run_id``/workdir and builds its own ELF, and ``pkg`` is read-only after the
    one-time build, so concurrent simulator instances (verilator/VCS) don't collide. Mirrors
    :func:`heavy_oracles.run_vcs_parallel`. ``max_workers == 1`` preserves the original sequential order.
    """
    # Resolve every path before worker entry: some tool adapters change process
    # cwd. Late resolution can misplace results and exclude failures from the
    # prerequisite denominator; concurrent grades must retain all obligations.
    runs_root = str(Path(runs_root).resolve())
    package_dir = str(Path(package_dir).resolve())
    if contract is not None and Path(contract).exists():
        contract = str(Path(contract).resolve())
    from merlin.experiments.phase1.feedback import operator_child as _operator_child
    from merlin.experiments.phase1.feedback import snapshot_execution

    process_operators = (
        not snapshot_execution.selected()
        and _operator_child.enabled()
        and any(cap.get("kind") != "model" for cap in capsules)
    )
    if process_operators:
        _operator_child.validate(target=target, adapters=oracle_adapters, config=config, perf_extractor=perf_extractor)
    pkg = load_package(package_dir, contract=contract)
    integrity_scan(pkg)
    build_package(pkg)
    snapshot_owner = snapshot_execution.active_snapshot()
    if snapshot_owner is not None:
        snapshot_owner.prepare_result_source(Path(runs_root))
    boundary_readiness = _candidate_boundary_readiness(capsules, pkg, oracle_adapters=oracle_adapters)

    # Pass evidence is opt-in through MERLIN_PASS_LOG. When enabled, install before any capsule work and
    # attribute in-process calls to the capsule that caused them. Package entrypoints remain subprocesses
    # and are certified by the mandatory manifest-command contract; this recorder covers only Merlin's
    # production whole-model boundary plane.
    _pass_recorder = None
    if os.environ.get("MERLIN_PASS_LOG"):
        from merlin.xdsl_dialects.lowering import passes as _pass_recorder

        _pass_recorder.install_pass_recorder()

    def _one_native(cap: dict, workers: int = 1) -> dict:
        # ``workers`` is what THIS capsule ran inside, not what the suite was configured for: the
        # calibration head is serial by construction, and recording the suite's max_workers for it
        # would describe contention its numbers never saw.
        if process_operators and cap.get("kind") != "model":
            from merlin.targetgen.owned_host_leaf_execution import selected_host_leaf

            if not selected_host_leaf(cap):
                return _operator_child.grade(
                    cap,
                    package_dir,
                    runs_root=runs_root,
                    contract=contract,
                    target=target,
                    suite=suite,
                    dtype=dtype,
                    adapters=oracle_adapters,
                    timeout=timeout,
                    workers=workers,
                    no_oracle=no_oracle,
                )
        if _pass_recorder is None:
            return run_capsule(
                cap,
                package_dir,
                runs_root=runs_root,
                run_id=cap["name"],
                contract=contract,
                oracle_adapters=oracle_adapters,
                pkg=pkg,
                timeout=timeout,
                target=target,
                suite=suite,
                dtype=dtype,
                config=config,
                perf_extractor=perf_extractor,
                no_oracle=no_oracle,
                workers=workers,
            )
        with _pass_recorder.pass_run_context(str(cap["name"]), cap.get("pass_requirements") or ()):
            return run_capsule(
                cap,
                package_dir,
                runs_root=runs_root,
                run_id=cap["name"],
                contract=contract,
                oracle_adapters=oracle_adapters,
                pkg=pkg,
                timeout=timeout,
                target=target,
                suite=suite,
                dtype=dtype,
                config=config,
                perf_extractor=perf_extractor,
                no_oracle=no_oracle,
                workers=workers,
            )

    def _one(cap: dict, workers: int = 1) -> dict:
        from merlin.experiments.phase1.feedback import setup_timing

        # Diagnostic identity only. This worker may execute multiple tiers;
        # selecting/requesting L2 does not make every setup stage an L2 event.
        from merlin.experiments.phase1.feedback.compiler_profile_runtime import active_original_execution

        original = active_original_execution()
        source = None if original is None else original.source_sha256
        with setup_timing.capsule_scope(cap["name"], snapshot_id=str(runs_root)):
            with (
                setup_timing.selected(
                    Path(runs_root) / "host-stage-timing", source, operation="native_capsule", queue_wait=True
                ),
                _original_capsule_worker_scope(),
            ):
                with setup_timing.stage("capsule_execution"):
                    result = _one_native(cap, workers)
        gate = cap.get("_driver_model_gate_admission")
        if cap.get("kind") == "model" and isinstance(gate, dict):
            result["model_gate_admission"] = gate
            path = Path(runs_root) / "runs" / (suite or suite_for(target)) / cap["name"] / "capsule_result.json"
            if path.is_file():
                path.write_text(readable_json(result))
        return result

    def _run_all(caps: list[dict]) -> list[dict]:
        # PIPELINED: a capsule starts as soon as a worker is free and its selected `extends` sibling has
        # FINISHED (it cites that sibling's published result), with one short calibration head on the
        # cheapest members while tiers are unpriced. No wave barrier, so one slow member never holds
        # the rest (merlin.targetgen.capsule_scheduling).
        from merlin.experiments.phase1.feedback import snapshot_execution
        from merlin.targetgen.capsule_scheduling import run_pipelined

        return run_pipelined(
            caps,
            lambda c, n: _one(c, n),
            max_workers=max_workers,
            boundary_readiness=boundary_readiness,
            calibrate=lambda: set(_tier_policy.priced_tiers(target or "")),
            calibration_cap=0 if snapshot_execution.selected() else _CALIBRATION_CAP,
            on_complete=on_capsule_complete,
            **({"defer_pending": defer_pending, "defer_running": defer_running} if defer_pending is not None else {}),
        )

    # A whole-model (kind == "model") capsule is the GATED capstone: it is scheduled only after the op
    # suite proves itself (its ``gate.after_op_pass_fraction`` of the graded op capsules passed). Grade the
    # op capsules first; if no model capsule is present this is exactly the original single-pass behavior.
    from merlin.targetgen.capsule_source import model_gate_satisfied

    op_caps = [c for c in capsules if c.get("kind") != "model"]
    model_caps = [c for c in capsules if c.get("kind") == "model"]
    model_names = {str(cap.get("name") or "") for cap in model_caps}
    if any(str(cap.get("extends") or "") in model_names for cap in op_caps):
        raise ValueError("operator capsule extends a gated model; dependency conflicts with the model gate")

    # DO NOT GRADE what this target provably cannot do. A capsule outside the contract's declared
    # capability can never pass, so grading it (a) spends oracle time on a foregone conclusion, (b)
    # misdirects the agent onto unreachable work, and (c) -- the expensive one -- makes ``all_pass``
    # UNREACHABLE, which disables the loop's early exit and turns every run into a fixed-price purchase
    # of its full round budget. Measured: an int8 target scored 22/22 on everything it declares at round
    # 00, then ran 20 rounds because 15 out-of-scope capsules kept all_pass false; a float target had 9
    # reachable failures and burned the same 20 rounds with 4 out-of-scope capsules doing the same thing.
    #
    # They are NOT dropped -- silently shrinking a suite is how a target scores better than it deserves.
    # Each is reported as ``not_graded`` with the derived reason, counted in neither numerator nor
    # denominator, and visible in the result list.
    op_caps, ungradeable = _split_ineligible(op_caps, target)
    # CERTIFY THE COVERING SET FIRST. The derived cover is the fewest capsules whose declared axes span
    # every axis the eligible corpus declares, so a run that is interrupted, times out, or exhausts a
    # certify budget still has every axis represented rather than a lexicographic prefix. Computed over
    # the ELIGIBLE capsules only -- spending the cover on a capsule the hardware cannot execute buys
    # nothing (measured: including the ineligible ones inflated one target's cover from 6 to 17).
    _cover = set(_tier_policy.covering_set(op_caps))
    for _c in op_caps:
        _c["_covering"] = _c.get("name") in _cover
    op_caps = sorted(op_caps, key=lambda c: (not c.get("_covering"), str(c.get("name"))))
    if _cover:
        print(
            f"  tier plan: {len(_cover)}/{len(op_caps)} eligible capsule(s) form the derived covering "
            f"set (certified first); budget="
            f"{_tier_policy.budget_seconds() or 'unlimited'}",
            flush=True,
            file=sys.stderr,
        )
    from merlin.experiments.phase1.feedback import snapshot_execution

    if snapshot_execution.selected():
        # This experimental treatment lets the author choose when a model is
        # ready. Admission is scheduling, never correctness or prior-op credit.
        for cap in model_caps:
            cap["_driver_model_gate_admission"] = {
                "policy": "independent_requests_v1",
                "execution_mode": snapshot_execution.MODE,
                "prerequisite_credit": False,
            }
        return _run_all(op_caps + model_caps) + ungradeable
    if model_caps and not no_oracle and _tier_policy.budget_seconds() is None:
        print(
            "  model gate: streaming over the fixed full selected eligible operator denominator",
            flush=True,
            file=sys.stderr,
        )
        return (
            _run_streaming_model_gate(
                op_caps,
                model_caps,
                _one,
                max_workers=max_workers,
                target=target,
                oracle_adapters=oracle_adapters,
                on_complete=on_capsule_complete,
                calibrate=lambda: set(_tier_policy.priced_tiers(target or "")),
                calibration_cap=_CALIBRATION_CAP,
                model_gate_continuations=model_gate_continuations,
                model_gate_prerequisites=model_gate_prerequisites,
                boundary_readiness=boundary_readiness,
                **(
                    {"defer_pending": defer_pending, "defer_running": defer_running}
                    if defer_pending is not None
                    else {}
                ),
            )
            + ungradeable
        )
    op_results = _run_all(op_caps) + ungradeable
    if ungradeable:
        print(
            f"  {len(ungradeable)} capsule(s) NOT GRADED — outside this target's declared capability: "
            f"{', '.join(r['capsule'] for r in ungradeable[:6])}"
            f"{' ...' if len(ungradeable) > 6 else ''}",
            flush=True,
            file=sys.stderr,
        )
    if not model_caps:
        return op_results
    # The gate fraction is over what the HARDWARE CAN DO, not over everything graded. A capsule the
    # target provably cannot accelerate must not be able to block the whole-model capsules behind it.
    # Measured: an int8 systolic target graded 12 bf16 capsules its contract declares no capability for
    # ("input dtype 'bf16' not in contraction formats ['int8']"). They can never pass, so the best
    # reachable fraction was 23/35 = 0.66 against a 0.8 gate -- the whole-model capsules were
    # MATHEMATICALLY unreachable, and nothing said so. Excluding the ineligible makes the gate mean
    # "the ops this device can do are working", which is what it was for.
    # WHAT COUNTS AS GRADED, for the gate. Previously only pass/fail, which quietly excluded every
    # capsule that ERRORED -- and an error is not evidence the op suite works, it is the absence of
    # evidence. Measured: 18 of 26 op capsules died with a runner crash and left the denominator with
    # them, so the whole-model capstone cleared its 0.8 gate on the surviving 7/8 = 0.88 while two
    # thirds of the suite had not run. `not_graded` (the hardware cannot) and `gated` (not yet
    # attempted) stay excluded; those are the two the exclusion was FOR. This matches the definition
    # the score itself already uses for its denominator.
    # `screened_only` joins the two existing exclusions for the same reason they are excluded: the
    # capsule was not measured against the certifying tier, deliberately and by name, so it is neither
    # evidence for nor against. Unlike an ERROR (which used to slip out of here unexplained), this is an
    # opt-in choice whose coverage is guaranteed by the covering set and whose members are listed.
    graded = [r for r in op_results if r.get("status") not in NOT_MEASURED_STATUSES]
    eligible = [r for r in graded if _gate_counts(r, capsules, target)]
    denom = eligible or graded  # fall back rather than divide by zero if nothing is classifiable
    capsule_by_name = {str(c["name"]): c for c in op_caps}
    frac = (
        (
            sum(
                _model_gate_operation_passed(r, capsule_by_name.get(str(r.get("capsule")), {}), oracle_adapters)
                for r in denom
            )
            / len(denom)
        )
        if denom
        else 0.0
    )
    if oracle_adapters is not None and set(oracle_adapters) == {"L2"}:
        print(
            "  model gate: selected L2 screen evidence only; full certification remains required",
            flush=True,
            file=sys.stderr,
        )
    if len(eligible) != len(graded):
        print(
            f"  model gate: {len(graded) - len(eligible)} graded op capsule(s) excluded from the gate "
            f"denominator as ineligible for this target (the hardware declares no capability for them)",
            flush=True,
            file=sys.stderr,
        )
    # SAY IT WHEN THE GATE GOT CHEAPER. Under a certify budget the denominator is what was CERTIFIED, not
    # what exists, so a capstone can clear 0.8 on the covering set alone. That is the intended trade --
    # the cover spans every declared axis -- but a gate satisfied by 6 capsules instead of 23 is a
    # different statement, and it has to be printed next to the number rather than inferred.
    _screened = [r for r in op_results if r.get("status") == "screened_only"]
    if _screened:
        print(
            f"  model gate: denominator is the {len(denom)} CERTIFIED capsule(s); "
            f"{len(_screened)} more passed the screen and were not certified "
            f"(outside the covering set, budget exhausted) — the gate fraction {frac:.2f} is over "
            f"what was certified, not over the whole suite",
            flush=True,
            file=sys.stderr,
        )
    model_results = []
    for c in model_caps:
        reason = defer_pending() if defer_pending is not None else None
        if reason:
            model_results.append(
                {
                    "capsule": c["name"],
                    "kind": "model",
                    "label": c.get("label"),
                    "status": "gated",
                    "background_deferred": reason,
                    "failure": {"plane": "gate", "category": "BACKGROUND_DEFERRED", "detail": reason},
                }
            )
            continue
        from merlin.targetgen.model_gate_continuation import admits, prerequisite_fraction

        prior = prerequisite_fraction(c, model_gate_prerequisites, target=target)
        value = frac if prior is None else prior
        continuation = (
            prior is None
            and not op_caps
            and admits(c, model_gate_continuations, target=target, adapters=oracle_adapters)
        )
        if model_gate_satisfied(c, value) or continuation:
            c["_driver_model_gate_admission"] = {
                "passed": True,
                "fraction": value,
                "stage": "L2" if set(oracle_adapters or {}) == {"L2"} else "continuation" if continuation else "full",
            }
            result = _one(c)
            model_results.append(result)
            if on_capsule_complete is not None:
                on_capsule_complete(result)
        else:
            thr = (c.get("gate") or {}).get("after_op_pass_fraction")
            model_results.append(
                {
                    "capsule": c["name"],
                    "kind": "model",
                    "label": c.get("label"),
                    "status": "gated",
                    "failure": {
                        "plane": "gate",
                        "category": "GATED",
                        "detail": f"whole-model capsule deferred: op pass fraction {value:.2f} < gate {thr}",
                    },
                }
            )
    return op_results + model_results
