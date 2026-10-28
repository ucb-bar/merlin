"""Required public functional suite using the ordinary Spike grader, never RTL."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from merlin.common.tree_hash import hash_tree

from ...corpus import public_scalability as S

ENV = "MERLIN_FROZEN_PUBLIC_SCALABILITY"


def install(te, frozen: Path | None):
    """Called only after the ordinary native input snapshot has been verified."""
    from ...runner import fingerprint

    selected = S.selected(te)
    if selected is None:
        os.environ.pop(ENV, None)
        return
    if frozen is None or fingerprint(frozen) != (te.workload_spec or {})[S.KEY]["sha256"]:
        raise ValueError("public scalability frozen snapshot is absent or differs from selected release")
    S.verify(frozen, target=te.target)
    os.environ[ENV] = json.dumps({"root": str(frozen), "descriptor_sha256": te.descriptor_sha256})


def selected(context):
    from merlin.targetgen.target_experiment import load_target_experiment

    from ...runner import fingerprint

    te = load_target_experiment(context.descriptor)
    binding = (te.workload_spec or {}).get(S.KEY)
    if binding is None:
        return None
    raw = os.environ.get(ENV)
    if raw is None:
        raise ValueError("required public scalability has no native frozen input transport")
    from merlin.common.strict_json import loads

    view = loads(raw, max_bytes=16384)
    if (
        type(view) is not dict
        or set(view) != {"root", "descriptor_sha256"}
        or view["descriptor_sha256"] != te.descriptor_sha256
    ):
        raise ValueError("public scalability snapshot belongs to another descriptor")
    root = Path(view["root"])
    if fingerprint(root) != binding["sha256"]:
        raise ValueError("public scalability native input bytes changed")
    return root, S.verify(root, target=context.target)


def request_policy(sim=None, tiers="", *, inspection=False):
    """The server owns this selection: a request can narrow tiers, never widen engines."""
    if inspection or sim not in (None, "", "spike"):
        raise ValueError("public scalability accepts only Spike functional requests")
    requested = [v.strip() for v in str(tiers or "").split(",") if v.strip()] or S.POLICY["tiers"]
    if len(set(requested)) != len(requested) or not set(requested) <= set(S.POLICY["tiers"]):
        raise ValueError("public scalability refuses L3/RTL/unknown tiers")
    return "spike", requested


def complete(score, observed, *, names):
    """A projection can never replace unrun, wrong-engine or partial ordinary evidence."""
    from merlin.targetgen.capsule_common import tier_status

    from .qa import _original_cohort_complete

    if not _original_cohort_complete(score) or score.get("gradeable") is not True:
        return False
    if any(score.get(key) is not True for key in ("numeric_all_exact", "structural_evidence_all_pass")):
        return False
    rows = score.get("per_capsule") or []
    if {r.get("capsule") for r in rows} != set(names) or set(observed) != set(names):
        return False
    if any(score.get("tier_reached", {}).get(t) != len(names) for t in S.POLICY["tiers"]):
        return False
    for name in names:
        row = observed[name]
        tiers = row.get("tiers") or {}
        if row.get("status") != "pass" or set(tiers) != set(S.POLICY["tiers"]):
            return False
        if any(tier_status(tiers[t]) != "pass" for t in S.POLICY["tiers"]):
            return False
        oracle = tiers["L2"]
        if (
            not isinstance(oracle, dict)
            or oracle.get("engine") != "spike"
            or oracle.get("fidelity") != "functional_model"
            or oracle.get("derived_from_rtl") is not False
        ):
            return False
    return True


def _observed_result(row):
    """Keep bounded compiler errors ahead of consequential proof failures."""
    from .selfcheck_results import _public_candidate_emission_failure

    observed = {key: row.get(key) for key in ("capsule", "status", "tiers", "failure", "performance")}
    if row.get("status") != "pass":
        failure = _public_candidate_emission_failure(row)
        if failure is not None:
            observed["failure"] = failure
    return observed


def _grade(
    submission,
    *,
    root,
    record,
    context,
    output,
    timeout,
    contract=None,
    names=None,
    additional_forbidden=(),
    max_workers=0,
):
    """Recompute with the original grader and recheck complete input/candidate custody."""
    from merlin.targetgen import capsule_grade as CG
    from merlin_experiments.corpus.preparation import copy_input

    from ...runner import fingerprint
    from .selfcheck import _adapters

    before, candidate = fingerprint(root), hash_tree(Path(submission))["sha256"]
    if S.verify(root, target=context.target) != record:
        raise ValueError("public scalability selected record changed")
    all_names = set(record["cases"])
    names = all_names if names is None else set(names)
    if not names or not names <= all_names:
        raise ValueError("public scalability request names unknown or empty cases")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = root / "cases"
    if names != all_names:
        cases = output / "subset"
        for name in sorted(names):
            copy_input(root / "cases" / name, cases / name, private=True)
    adapters, engine = _adapters("spike", context.target, _sim_via(context), **_readback(context))
    if (
        engine != "spike"
        or set(adapters) != {"L2"}
        or getattr(adapters["L2"], "_merlin_simulator_engine", None) != "spike"
    ):
        raise ValueError("selected target has no exact Spike-only functional adapter")
    from .joint_performance import deadline as performance_deadline

    observed = {}
    performance_budget = max(0.0, performance_deadline(timeout) - time.monotonic())

    def completed(row):
        nonlocal performance_budget
        name = row.get("capsule")
        if name in observed:
            raise ValueError("duplicate public scalability execution observation")
        from merlin.targetgen import capsule_runner as CR
        from merlin.targetgen.capsule_common import load_capsule

        from .joint_performance import attach as attach_performance

        generated = output / "runs" / "runs" / CR.suite_for(context.target) / name / "generated"
        performance_started = time.monotonic()
        attach_performance(
            row,
            generated,
            result=row,
            context=context,
            compiler=Path(submission),
            capsule=load_capsule(root / "cases" / name),
            remaining_s=performance_budget,
        )
        performance_budget = max(0.0, performance_budget - (time.monotonic() - performance_started))
        observed[name] = _observed_result(row)

    from contextlib import redirect_stderr, redirect_stdout

    from merlin.targetgen.tier_policy import floor_exempt

    # The selected policy is exactly Spike L0/L1/L2: no L3 adapter exists for this suite by construction, so
    # the operator tier floor (MERLIN_CAPSULE_TIER_FLOOR=L3, which raises every scored core capsule) must not
    # add a rung it cannot run. Scoped to these case directories; core capsules keep the floor.
    with (
        (output / "grader.log").open("w") as log,
        redirect_stdout(log),
        redirect_stderr(log),
        floor_exempt([cases / name for name in names]),
    ):
        score = CG.grade(
            submission,
            capsules_root=cases,
            runs_root=output / "runs",
            target=context.target,
            labels={"public", "dev"},
            contract=contract or context.repo / "merlin/contract",
            oracle_adapters=adapters,
            timeout=timeout,
            max_workers=max_workers,
            additional_forbidden=additional_forbidden,
            on_capsule_complete=completed,
        )
    if (
        fingerprint(root) != before
        or S.verify(root, target=context.target) != record
        or hash_tree(Path(submission))["sha256"] != candidate
    ):
        raise ValueError("public scalability source/input/reference or candidate changed during execution")
    passed = names == all_names and complete(score, observed, names=all_names)
    result = {
        "schema": "merlin.phase1.public_scalability.v1",
        "required": True,
        "passed": passed,
        "complete": passed,
        "policy": S.POLICY,
        "suite_sha256": before,
        "candidate_sha256": candidate,
        "expected_cases": sorted(all_names),
        "requested_cases": sorted(names),
        "score": score,
        "observed": observed,
        "rtl_qualified": False,
        "timing_qualified": False,
    }
    (output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def grade(submission, **kwargs):
    from merlin.common import invocation_record as I
    from merlin.common.digest import sha256_file

    from ...runner import fingerprint

    output, root, context = Path(kwargs["output"]), kwargs["root"], kwargs["context"]
    arguments = {
        "candidate_sha256": hash_tree(Path(submission))["sha256"],
        "suite_sha256": fingerprint(root),
        "target": context.target,
        "complete_roster_requested": kwargs.get("names") is None,
    }
    from merlin.targetgen import capsule_grade, capsule_runner

    with I.observe_call(
        output.parent / "issuers",
        stage="public_spike_scalability",
        function=_grade,
        arguments=arguments,
        inputs=(root / "selection.json", context.descriptor),
        dependencies=(Path(__file__), Path(S.__file__), Path(capsule_grade.__file__), Path(capsule_runner.__file__)),
        outputs=(output / "qualification.json",),
    ) as observed:
        result = _grade(submission, **kwargs)
        observed.returned()
    I.verify(observed.path)
    issuance = {"path": str(observed.path), "sha256": sha256_file(observed.path)}
    receipt = {
        "issuer": issuance,
        "result": {"path": str(output / "qualification.json"), "sha256": sha256_file(output / "qualification.json")},
    }
    (output / "issued.json").write_text(json.dumps(receipt, indent=2) + "\n")
    result["issuer"] = issuance
    return result


def verify_issued(path, *, descriptor, candidate):
    from merlin.common import invocation_record as I
    from merlin.common.digest import sha256_file
    from merlin.common.strict_json import loads
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(descriptor)
    selection = S.selected(te)
    if selection is None:
        return True
    root, record = selection
    receipt = loads(Path(path).read_bytes())
    for pin in receipt.values():
        if sha256_file(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("public scalability issuer/result changed")
    call = I.verify(Path(receipt["issuer"]["path"]))
    result = loads(Path(receipt["result"]["path"]).read_bytes())
    expected = {
        "candidate_sha256": hash_tree(Path(candidate))["sha256"],
        "suite_sha256": (te.workload_spec or {})[S.KEY]["sha256"],
        "target": te.target,
        "complete_roster_requested": True,
    }
    if (
        call.get("kind") != "python_call"
        or call.get("stage") != "public_spike_scalability"
        or call.get("callable") != __name__ + "._grade"
        or (call.get("executable") or {}).get("sha256") != sha256_file(Path(__file__))
        or call.get("arguments") != expected
        or receipt["result"] not in call.get("outputs", [])
        or any(result.get(k) != expected[k] for k in ("candidate_sha256", "suite_sha256"))
        or result.get("policy") != S.POLICY
        or result.get("passed") is not True
        or not complete(result["score"], result["observed"], names=record["cases"])
    ):
        raise ValueError("required public scalability has no complete original Spike producer")
    return {
        "required": True,
        "passed": True,
        "n_cases": len(record["cases"]),
        "suite_sha256": expected["suite_sha256"],
        "policy": S.POLICY,
        "producer": {"path": str(Path(path).absolute()), "sha256": sha256_file(Path(path))},
    }


def _sim_via(context):
    from merlin.targetgen.target_experiment import load_target_experiment

    return load_target_experiment(context.descriptor).sim_via


def _readback(context):
    from ..context import readback_kwargs

    return readback_kwargs(context)


def attach(verdict, submission, *, context, output, timeout, contract=None, **kwargs):
    """An unselected legacy run is unchanged; a required refusal cannot converge."""
    selection = selected(context)
    if selection is None:
        return
    from ..joint_control import selected as joint_campaign

    if "all_pass" in verdict and verdict["all_pass"] is not True and joint_campaign() is None:
        verdict["public_scalability"] = {"required": True, "passed": False, "status": "waiting_for_core"}
        return
    try:
        root, record = selection
        result = grade(
            submission,
            root=root,
            record=record,
            context=context,
            output=output,
            timeout=timeout,
            contract=contract,
            **kwargs,
        )
        verdict["public_scalability"] = public_result(result)
    except Exception as exc:
        verdict["public_scalability"] = {
            "required": True,
            "passed": False,
            "reason": f"required suite refused ({type(exc).__name__})",
        }
    if verdict["public_scalability"].get("passed") is not True:
        verdict["all_pass"] = False
        verdict["not_converged_reason"] = "required public Spike scalability suite is incomplete"


def cli(args, *, context):
    selection = selected(context)
    if selection is None:
        raise ValueError("no public scalability suite is selected")
    request_policy(
        args.sim,
        args.tiers,
        inspection=any(
            getattr(args, k, False)
            for k in ("compile_only", "candidate_diagnostics", "caller_layout", "model_layers", "dynamic_census")
        ),
    )
    if args.tiers and [v.strip() for v in args.tiers.split(",")] != S.POLICY["tiers"]:
        raise ValueError("public scalability grading requires all selected L0/L1/L2 tiers")
    if not args.out:
        raise ValueError("public scalability requires a result output path")
    root, record = selection
    names = None if args.capsules == "all" else {v.strip() for v in args.capsules.split(",")}
    from .compiler_profile_runtime import selfcheck_execution

    with selfcheck_execution(
        args.compiler_bundle_manifest,
        context=context,
        submission=Path(args.submission),
        work_root=Path(args.out).with_suffix(".profile"),
    ):
        result = grade(
            args.submission,
            root=root,
            record=record,
            context=context,
            output=Path(args.out).with_suffix(".scalability"),
            timeout=args.timeout,
            contract=args.contract,
            names=names,
            max_workers=args.workers,
        )
    public = public_result(result)
    public["all_pass"] = result["passed"]
    text = json.dumps(public, indent=2)
    Path(args.out).write_text(text)
    print(text)
    return 0 if result["passed"] else 1


def public_result(result):
    """Expose outcomes, never values, golden digests, streams or private artifact paths."""
    from merlin.common.path_scrub import scrub_host_paths
    from merlin.targetgen.capsule_common import tier_status

    from .qa import _redact_detail

    def failure(row):
        raw = row.get("failure") or {}
        return {
            "failure_" + key: _redact_detail(scrub_host_paths(str(raw[key]))) if raw.get(key) else None
            for key in ("plane", "category", "detail")
        }

    public = {k: v for k, v in result.items() if k not in {"score", "observed", "issuer"}}
    public.update(failure(result.get("score") or {}))
    public["cases"] = [
        {
            "capsule": name,
            "status": row.get("status"),
            **failure(row),
            "tiers": {t: tier_status(record) for t, record in (row.get("tiers") or {}).items()},
            **({"performance": row["performance"]} if row.get("performance") is not None else {}),
        }
        for name, row in sorted(result["observed"].items())
    ]
    return public


def prompt_block(te):
    selection = S.selected(te)
    if selection is None:
        return ""
    root, record = selection
    return (
        "\n\n## Required public Spike scalability suite\n"
        f"Mathematical sources and exact source-group policy: {root / 'public_sources/suite.json'}.\n"
        f"This separately selected suite has {len(record['cases'])} cases and is required for completion. "
        "Use `python agent_selfcheck.py --suite scalability --capsules NAME1,NAME2 --wait-budget 5` "
        "for affected cases; collect asynchronous requests with `--attach ID`. "
        "Use `--capsules all` when ready for the complete functional denominator. "
        "Only Spike L0/L1/L2 is permitted here; gSIM and L3 requests are refused. "
        "A subset, unavailable contract or unrun case cannot complete this suite. "
        "Its passes give no RTL or timing credit. Core RTL requirements remain unchanged; "
        "hidden and full-model validation remain separate. "
        "Full-model execution follows the separately selected campaign window. "
        "In a joint campaign, start these cases immediately: their failures and performance "
        "feedback guide compiler architecture while core correctness is still evolving. "
        "Timing estimates, when explicitly selected, carry their own qualification status.\n"
    )
