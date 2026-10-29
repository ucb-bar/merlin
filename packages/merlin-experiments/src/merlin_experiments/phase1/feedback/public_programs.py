"""Selected public program exploration through the original owned-model evaluator.

No separate model executor, timing estimate, conformance credit or RTL grant is
introduced. The selfcheck broker retains submission snapshots and concurrency.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from merlin.common.tree_hash import hash_tree

from ...corpus import public_program_suite as S


def selected(context):
    from merlin.targetgen.target_experiment import load_target_experiment

    return S.selected(load_target_experiment(context.descriptor))


def request_policy(sim=None, tiers="", *, inspection=False):
    if inspection or sim not in (None, "", "spike"):
        raise ValueError("public programs accept only ordinary Spike model checks")
    requested = [part.strip() for part in str(tiers or "").split(",") if part.strip()]
    if requested and requested != S.POLICY["tiers"]:
        raise ValueError("public programs require the original complete L0,L1,L2 obligations")
    return "spike"


def _grade(submission, *, context, output, timeout, contract=None, names=None, max_workers=0):
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen.capsule_common import load_capsule, tier_status

    from ...corpus.preparation import copy_input
    from ...runner import fingerprint
    from ..context import readback_kwargs
    from ..joint_model_activation import admit_grade
    from .joint_performance import attach, environment_scope, observation_budget
    from .public_scalability import _observed_result
    from .selfcheck import _adapters

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
    admission = admit_grade(candidate=submission, source_sha256=candidate, output=output,
                            kind="public_programs", identity={"source": source_identity, "names": sorted(wanted)})
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
                cap, submission, runs_root=output / "runs", run_id=name,
                contract=contract or context.repo / "merlin/contract", target=context.target,
                oracle_adapters=adapters, adapter_managed_tiers=("L2",), timeout=remaining,
                model_budget_s=remaining, workers=max_workers,
            )
            # Use the same scoped feedback owner as ordinary selfchecks. Missing
            # coverage remains UNKNOWN; functional instruction counts are not cycles.
            generated = output / "runs/runs" / CR.suite_for(context.target) / name / "generated"
            with environment_scope(admission):
                attach(row, generated, result=row, context=context, compiler=Path(submission),
                       capsule=cap, remaining_s=observation_budget(deadline), request_deadline=deadline)
            observed[name] = _observed_result(row)
    finally:
        if previous is None:
            os.environ.pop("MERLIN_MODEL_NATIVE_SIMULATOR", None)
        else:
            os.environ["MERLIN_MODEL_NATIVE_SIMULATOR"] = previous
    if (hash_tree(Path(submission))["sha256"] != candidate
            or S.read(selection_path, target=context.target)[0] != record
            or fingerprint(private) != source_identity):
        raise ValueError("public program source or candidate changed during its ordinary grade")
    passed = all(row.get("status") == "pass"
                 and all(tier_status((row.get("tiers") or {}).get(tier)) == "pass" for tier in S.POLICY["tiers"])
                 for row in observed.values())
    result = dict(schema="merlin.phase1.public_program_checks.v1", policy=S.POLICY,
                  candidate_sha256=candidate, source_sha256=source_identity,
                  requested_cases=sorted(wanted), expected_cases=sorted(members),
                  passed=passed, complete=passed and wanted == set(members),
                  observed=observed, score={}, original_conformance_credit=False, rtl_qualified=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _sim_via(context):
    from merlin.targetgen.target_experiment import load_target_experiment

    return load_target_experiment(context.descriptor).sim_via


def cli(args, *, context):
    from ..joint_model_activation import grade_model_scope
    from .compiler_profile_runtime import selfcheck_execution
    from .public_scalability import public_result

    request_policy(args.sim, args.tiers, inspection=any(getattr(args, key, False) for key in (
        "compile_only", "candidate_diagnostics", "caller_layout", "model_layers", "dynamic_census")))
    if not args.out:
        raise ValueError("public program checks require a response output")
    path = Path(args.out)
    from contextlib import redirect_stderr, redirect_stdout

    from .raw_selfcheck_evidence import create_runs_root

    private_work = create_runs_root()
    with selfcheck_execution(args.compiler_bundle_manifest, context=context, submission=Path(args.submission),
                             work_root=private_work / "profile"), (private_work / "grader.log").open("w") as log, \
            redirect_stdout(log), redirect_stderr(log):
        result = grade_model_scope(_grade)(
            args.submission, context=context, output=private_work / "programs", timeout=args.timeout,
            contract=args.contract, names=None if args.capsules == "all" else set(args.capsules.split(",")),
            max_workers=args.workers)
    public = public_result(result)
    public["all_pass"] = result["passed"]
    from .performance_brief import attach as attach_brief

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
    return ("\n\n## Selected public connected programs\n"
            f"Cases: {names}. Sources: `public_program_sources/suite.json`. "
            "Use `python agent_selfcheck.py --suite programs --capsules NAME "
            "--sim spike --tiers L0,L1,L2 --wait-budget 5`. Only the operator-selected source roster is "
            "available. The original model owner compiles host/device work and checks every output; "
            "all selected programs are required final Spike obligations, tracked separately from original "
            "core conformance. They do not authorize gSIM or earn original-conformance/L3 credit. "
            "The scalability-only idle tuner does not automatically explore this separate endpoint.\n")
