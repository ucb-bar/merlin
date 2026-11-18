"""Host-only sealed model checks, with optional bounded model-kernel RTL replays."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from merlin.common.digest import hash_tree
from merlin.evaluation.capsule import eligibility as capsule_eligibility
from merlin.evaluation.model.bundles.models import default_models_path, target_from_contract
from merlin.evaluation.model.bundles.seal import Bundle
from merlin.experiments.corpus import public_program_suite as S

# ---- model_check_official ----------------------------------------------------------------------

REPORT_SCHEMA = "merlin.evaluation.model.cli.v1"


def _invoke(
    command: str,
    *,
    submission: Path,
    model: str,
    output: Path,
    bundle: Path,
    timeout_s: int,
    program: str | None = None,
    group: str | None = None,
    mode: str = "full",
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    target = target_from_contract()
    argv = [
        sys.executable,
        "-m",
        "merlin.evaluation.model",
        command,
        "--submission",
        str(submission),
        "--model",
        model,
        "--split",
        "sealed",
        "--bundle",
        str(bundle),
        "--models",
        str(default_models_path(target)),
        "--no-cache",
        "--work",
        str(output.with_suffix(".work")),
        "--out",
        str(output),
        "--timeout-s",
        str(timeout_s),
    ]
    if command == "check":
        argv += ["--mode", mode]
    else:
        argv += ["--program", str(program), "--group", str(group), "--engine", "gsim"]
    log = output.with_suffix(".stderr.log")
    from merlin.common.process.cpu_pool import worker_pool_command

    with log.open("wb") as stream:
        process = subprocess.Popen(
            worker_pool_command(argv, dict(os.environ)), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            process.wait(timeout=timeout_s + 30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {"status": "timeout", "report": str(output), "log": str(log)}
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = None
    if not isinstance(report, dict) or any(
        (
            report.get("schema") != REPORT_SCHEMA,
            report.get("command") != command,
            report.get("model") != model,
            report.get("split") != "sealed",
            report.get("exit_code") != process.returncode,
        )
    ):
        return {"status": "unavailable", "report": str(output), "log": str(log)}
    if report.get("status") not in {"pass", "fail", "error"}:
        return {"status": "unavailable", "report": str(output), "log": str(log)}
    return {"status": report["status"], "report": str(output), "log": str(log), "body": report}


def run_gate(
    submission: Path,
    run_dir: Path,
    required_programs: dict[str, tuple[str, ...]],
    *,
    bundle: Path | None = None,
    model_timeout_s: int = 7200,
    rtl_budget_s: int = 7200,
    kernel_timeout_s: int = 300,
    rtl_workers: int = 4,
    replay_model_kernels: bool = False,
    mode: str = "full",
    model_workers: int = 1,
) -> dict:
    """Run the selected sealed check; fast requires the sealed shape variants.

    Capsule L3 qualification and timing collection are separate and stay enabled.
    An unrequested model RTL check is never reported as completed.
    """
    if mode not in {"fast", "full"} or type(model_workers) is not int or model_workers < 1:
        raise ValueError("model check requires an explicit fast/full mode and positive worker count")
    if bundle is None:
        bundle = Bundle.default(target_from_contract()).root
    root = run_dir / "grading_model_check"
    root.mkdir(parents=True, exist_ok=True)
    source_sha256 = hash_tree(submission)["sha256"]
    configured_slots = os.environ.get("MERLIN_GSIM_MAX_SLOTS", "").strip()
    try:
        slot_cap = max(1, int(configured_slots)) if configured_slots else 4
    except ValueError:
        slot_cap = 1
    summary = {
        "schema": "merlin.phase1.model_check_official.v1",
        "passed": False,
        "split": "sealed",
        "bundle": str(bundle),
        "submission_sha256": source_sha256,
        "mode": mode,
        "model_workers": min(model_workers, max(1, len(required_programs))),
        "whole_model_status": "deferred_post_experiment" if mode == "fast" else "required",
        "models": {},
        "rtl": [],
        "unique_kernel_count": 0,
        "rtl_complete": False,
        "rtl_required": replay_model_kernels,
        "rtl_status": "pending" if replay_model_kernels else "not_requested",
        "rtl_workers": max(1, min(int(rtl_workers), 4, slot_cap)),
    }
    selected: dict[tuple, tuple[str, str, str]] = {}
    variant_models = Bundle(bundle).manifest().get("models", {}) if mode == "fast" else {}

    def check_model(model):
        result = _invoke(
            "check",
            submission=submission,
            model=model,
            output=root / "models" / f"{model}.json",
            bundle=bundle,
            timeout_s=model_timeout_s,
            mode=mode,
        )
        return model, result

    with ThreadPoolExecutor(max_workers=summary["model_workers"]) as executor:
        checked = list(executor.map(check_model, required_programs))

    for model, result in checked:
        programs = required_programs[model]
        body = result.pop("body", None)
        counts = {
            name: value.get("variants")
            for name, value in (
                (variant_models.get(model) or {}).get("sealed_variants", {}).get("programs", {}).items()
            )
        }
        observed, problems = assess(body, programs, mode=mode, shape_counts=counts)
        if problems:
            result["status"] = "unavailable" if result["status"] == "pass" else result["status"]
            result["evidence_problems"] = problems
        summary["models"][model] = {
            **result,
            "programs": {
                program: {
                    "build": observed.get(program, {}).get("build"),
                    "census": observed.get(program, {}).get("census"),
                }
                for program in programs
            },
        }
        if result["status"] != "pass":
            continue  # RTL cannot turn an incomplete numerical check into a pass
        for program in programs:
            for kernel in observed.get(program, {}).get("kernels", []):
                if not isinstance(kernel, dict) or not isinstance(kernel.get("source_region_id"), str):
                    continue
                group = kernel["source_region_id"]
                # Cross-program reuse needs an explicit interface and shape identity. With either
                # absent, replay the group again; missing evidence can never reduce RTL coverage.
                interface = kernel.get("interface_sha256")
                shape = kernel.get("shape_sha256")
                key = (interface, shape) if interface and shape else (model, program, group)
                selected.setdefault(key, (model, program, group))
    summary["unique_kernel_count"] = len(selected)
    deadline = time.monotonic() + rtl_budget_s

    def replay(entry: tuple[int, tuple[str, str, str]]) -> dict:
        index, (model, program, group) = entry
        # The host wait may exceed the CLI timeout by 30 seconds, followed by termination grace.
        # Reserve that time inside the global budget so unfinished kernels become pending.
        remaining = int(deadline - time.monotonic()) - 40
        if remaining < 1:
            return {
                "model": model,
                "program": program,
                "group": group,
                "status": "pending",
                "reason": "RTL budget exhausted",
            }
        try:
            result = _invoke(
                "replay-kernel",
                submission=submission,
                model=model,
                program=program,
                group=group,
                output=root / "rtl" / f"kernel_{index:05d}.json",
                bundle=bundle,
                timeout_s=min(kernel_timeout_s, remaining),
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        result.pop("body", None)
        return {"model": model, "program": program, "group": group, **result}

    if replay_model_kernels:
        with ThreadPoolExecutor(max_workers=summary["rtl_workers"]) as executor:
            summary["rtl"] = list(executor.map(replay, enumerate(selected.values())))
    models_complete = bool(required_programs) and all(row["status"] == "pass" for row in summary["models"].values())
    summary["rtl_complete"] = (
        replay_model_kernels
        and models_complete
        and len(summary["rtl"]) == len(selected)
        and all(row["status"] == "pass" for row in summary["rtl"])
    )
    if replay_model_kernels:
        summary["rtl_status"] = "pass" if summary["rtl_complete"] else "incomplete"
    summary["passed"] = models_complete and (not replay_model_kernels or summary["rtl_complete"])
    if hash_tree(submission)["sha256"] != source_sha256:
        summary["passed"] = False
        summary["source_changed_during_grade"] = True
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


# ---- public_programs ---------------------------------------------------------------------------
# Selected public program exploration through the original owned-model evaluator.
#
# No separate model executor, timing estimate, conformance credit or RTL grant is
# introduced. The selfcheck broker retains submission snapshots and concurrency.


def selected(context):
    from merlin.target.registry.descriptor_loading import load_target_experiment

    return S.selected(load_target_experiment(context.descriptor))


def request_policy(sim=None, tiers="", *, inspection=False):
    if inspection or sim not in (None, "", "spike"):
        raise ValueError("public programs accept only ordinary Spike model checks")
    requested = [part.strip() for part in str(tiers or "").split(",") if part.strip()]
    if requested and requested != S.POLICY["tiers"]:
        raise ValueError("public programs require the original complete L0,L1,L2 obligations")
    return "spike"


def _grade(submission, *, context, output, timeout, contract=None, names=None, max_workers=0):
    from merlin.development.backend.campaign.calibration.activation import admit_grade
    from merlin.development.backend.context import readback_kwargs
    from merlin.development.backend.feedback.campaign.timing import (
        attach,
        environment_scope,
        observation_budget,
    )
    from merlin.development.backend.feedback.scalability.public_scalability import _observed_result
    from merlin.development.backend.feedback.selfcheck.diagnostic_adapters import _adapters
    from merlin.development.targetgen.corpus.capsule_io import load_capsule, tier_status
    from merlin.evaluation.capsule import runner as CR
    from merlin.experiments.corpus.source_staging import copy_input
    from merlin.experiments.runs.plan_sources import fingerprint

    choice = selected(context)
    if choice is None:
        raise ValueError("no public whole-program roster is selected by the operator")
    selection_path, record, members = choice
    wanted = set(members) if names is None else set(names)
    if not wanted or not wanted <= set(members):
        raise ValueError("public program requests name selected cases, never source paths")
    candidate = hash_tree(Path(submission))["sha256"]
    source_identity = fingerprint(Path(record["corpus"]["path"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Preserve the complete source/proof dependency tree; the original runner
    # additionally pins each original model source in its own private snapshot.
    private = output / "source_snapshot"
    copy_input(Path(record["corpus"]["path"]), private, private=True)
    if fingerprint(private) != source_identity:
        raise ValueError("public program native source snapshot differs from the selected input")
    adapters, engine = _adapters("spike", context.target, _sim_via(context), **readback_kwargs(context))
    if engine != "spike" or set(adapters) != {"L2"}:
        raise ValueError("public programs require the original selected Spike adapter")
    admission = admit_grade(
        candidate=submission,
        source_sha256=candidate,
        output=output,
        kind="public_programs",
        identity={"source": source_identity, "names": sorted(wanted)},
    )
    deadline = time.monotonic() + timeout
    observed = {}
    previous = os.environ.get("MERLIN_MODEL_NATIVE_SIMULATOR")
    try:
        # One selfcheck child process owns this selection. No ambient RTL model
        # engine can widen this source-selected development endpoint.
        os.environ["MERLIN_MODEL_NATIVE_SIMULATOR"] = "spike"
        for name in sorted(wanted):
            remaining = int(deadline - time.monotonic())
            if remaining < 1:
                raise TimeoutError("public program request exhausted its original aggregate deadline")
            cap = load_capsule(private / members[name].relative_to(Path(record["corpus"]["path"])))
            # One run directory per program: a shared run id made the second program of a
            # request collide on the first one's private model-source snapshot, and left
            # each program's products outside the per-name directory read below.
            row = CR.run_capsule(
                cap,
                submission,
                runs_root=output / "runs",
                run_id=name,
                contract=contract or context.repo / "src/merlin/resources/contracts",
                target=context.target,
                oracle_adapters=adapters,
                adapter_managed_tiers=("L2",),
                timeout=remaining,
                model_budget_s=remaining,
                workers=max_workers,
            )
            # Use the same scoped feedback owner as ordinary selfchecks. Missing
            # coverage remains UNKNOWN; functional instruction counts are not cycles.
            generated = output / "runs" / "runs" / capsule_eligibility.suite_for(context.target) / name / "generated"
            with environment_scope(admission):
                attach(
                    row,
                    generated,
                    result=row,
                    context=context,
                    compiler=Path(submission),
                    capsule=cap,
                    remaining_s=observation_budget(deadline),
                    request_deadline=deadline,
                )
            observed[name] = _observed_result(row)
    finally:
        if previous is None:
            os.environ.pop("MERLIN_MODEL_NATIVE_SIMULATOR", None)
        else:
            os.environ["MERLIN_MODEL_NATIVE_SIMULATOR"] = previous
    if (
        hash_tree(Path(submission))["sha256"] != candidate
        or S.read(selection_path, target=context.target)[0] != record
        or fingerprint(private) != source_identity
    ):
        raise ValueError("public program source or candidate changed during its ordinary grade")
    passed = all(
        row.get("status") == "pass"
        and all(tier_status((row.get("tiers") or {}).get(tier)) == "pass" for tier in S.POLICY["tiers"])
        for row in observed.values()
    )
    result = dict(
        schema="merlin.phase1.public_program_checks.v1",
        policy=S.POLICY,
        candidate_sha256=candidate,
        source_sha256=source_identity,
        requested_cases=sorted(wanted),
        expected_cases=sorted(members),
        passed=passed,
        complete=passed and wanted == set(members),
        observed=observed,
        score={},
        original_conformance_credit=False,
        rtl_qualified=False,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _sim_via(context):
    from merlin.target.registry.descriptor_loading import load_target_experiment

    return load_target_experiment(context.descriptor).sim_via


def cli(args, *, context):
    from merlin.development.backend.campaign.calibration.activation import grade_model_scope
    from merlin.development.backend.feedback.execution.profile_runtime import selfcheck_execution
    from merlin.development.backend.feedback.scalability.public_scalability import public_result

    request_policy(
        args.sim,
        args.tiers,
        inspection=any(
            getattr(args, key, False)
            for key in ("compile_only", "candidate_diagnostics", "caller_layout", "model_layers", "dynamic_census")
        ),
    )
    if not args.out:
        raise ValueError("public program checks require a response output")
    path = Path(args.out)
    from contextlib import redirect_stderr, redirect_stdout

    from merlin.development.backend.feedback.selfcheck.results import create_runs_root

    private_work = create_runs_root()
    with (
        selfcheck_execution(
            args.compiler_bundle_manifest,
            context=context,
            submission=Path(args.submission),
            work_root=private_work / "profile",
        ),
        (private_work / "grader.log").open("w") as log,
        redirect_stdout(log),
        redirect_stderr(log),
    ):
        result = grade_model_scope(_grade)(
            args.submission,
            context=context,
            output=private_work / "programs",
            timeout=args.timeout,
            contract=args.contract,
            names=None if args.capsules == "all" else set(args.capsules.split(",")),
            max_workers=args.workers,
        )
    public = public_result(result)
    public["all_pass"] = result["passed"]
    from merlin.development.backend.feedback.performance.brief import attach as attach_brief

    attach_brief(public, public.get("cases"), debuggable=False)
    text = json.dumps(public, indent=2)
    path.write_text(text)
    print(text)
    return 0 if result["passed"] else 1


def prompt_block(experiment):
    choice = S.selected(experiment)
    if choice is None:
        return ""
    names = ", ".join(sorted(choice[2]))
    return (
        "\n\n## Selected public connected programs\n"
        f"Cases: {names}. Sources: `public_program_sources/suite.json`. "
        "Use `python agent_selfcheck.py --suite programs --capsules NAME "
        "--sim spike --tiers L0,L1,L2 --wait-budget 5`. Only the operator-selected source roster is "
        "available. The original model owner compiles host/device work and checks every output; "
        "all selected programs are required final Spike obligations, tracked separately from original "
        "core conformance. They do not authorize gSIM or earn original-conformance/L3 credit. "
        "The scalability-only idle tuner does not automatically explore this separate endpoint.\n"
    )


# ---- model_check_evidence ----------------------------------------------------------------------
# Validate the evidence required by a selected sealed model-check mode.


def assess(body, programs, *, mode, shape_counts):
    """Keep original programs distinct from the additional sealed variant rows."""
    body = body if isinstance(body, dict) else {}
    rows = [row for row in body.get("programs", []) if isinstance(row, dict)]
    primary = [row for row in rows if not row.get("source")]
    observed = {row.get("program"): row for row in primary}
    problems = []
    if set(observed) != set(programs) or len(primary) != len(programs):
        problems.append("original program roster is missing or duplicated")
    metrics = body.get("model_metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("worst"), dict):
        problems.append("model metrics unavailable")
    if mode == "fast":
        gates = body.get("gates") or {}
        if any(
            (gates.get(name) or {}).get("status") != "pass"
            or ("partial" in (gates.get(name) or {}) and gates[name]["partial"] is not False)
            or (name == "kernel_exact" and (gates.get(name) or {}).get("partial") is not False)
            for name in ("build", "kernel_exact", "model_numerics")
        ):
            problems.append("sealed kernel build, exactness or model gate incomplete")
    for program in programs:
        item = observed.get(program, {})
        build, census = item.get("build") or {}, item.get("census") or {}
        kernels = item.get("kernels") or []
        device, denominator = census.get("device"), census.get("denominator")
        if type(device) is not int or type(denominator) is not int or not 0 <= device <= denominator:
            problems.append(f"{program}: invalid offload census")
        if device and not any(isinstance(k, dict) and isinstance(k.get("source_region_id"), str) for k in kernels):
            problems.append(f"{program}: original kernel identities unavailable")
        if mode == "full":
            if build.get("status") != "linked" or not any(
                isinstance(run, dict) and run.get("engine") == "spike" and run.get("status") == "completed"
                for run in item.get("runs", [])
            ):
                problems.append(f"{program}: linked whole-model Spike execution incomplete")
            continue
        if (
            type(build.get("built")) is not int
            or build["built"] < 0
            or build.get("refused") != 0
            or any(k.get("status") != "pass" for k in kernels)
        ):
            problems.append(f"{program}: kernel compilation or replay incomplete")
        if not device:
            continue
        expected = shape_counts.get(program)
        varied = [row for row in rows if row.get("source") == "sealed_variants" and row.get("program") == program]
        if type(expected) is not int or expected <= 0 or len(varied) != 1:
            problems.append(f"{program}: sealed shape variants unavailable")
            continue
        cases = varied[0].get("kernels") or []
        if (
            not cases
            or any(k.get("status") != "pass" or type(k.get("cases")) is not int or k["cases"] <= 0 for k in cases)
            or sum(k["cases"] for k in cases) != expected
        ):
            problems.append(f"{program}: sealed shape variant replay incomplete")
    return observed, problems


def rtl_requirement_satisfied(record):
    """An explicit unrequested model RTL gate needs its honest empty status."""
    if record.get("rtl_required") is False:
        return (
            record.get("rtl_status") == "not_requested"
            and record.get("rtl") == []
            and record.get("rtl_complete") is False
        )
    return record.get("rtl_complete") is True and record.get("unique_kernel_count") == len(record.get("rtl") or [])
