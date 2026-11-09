"""Final-drain functional coverage of the immutable native compiler snapshot.

This is an operator-owned completion route, not an author request or certificate.
It never consumes a development verdict, changes a numerical contract, runs RTL,
or resets the campaign clock. Native graders retain their exact artifact caches.
"""
from __future__ import annotations

import contextvars
import json
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from merlin.common.tree_hash import hash_tree

from . import joint_control
from .joint_final_model_l3 import MISSING as MODEL_L3_MISSING


def _write(path, document):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def _rollup(document):
    rows = [row for group in document["groups"].values() for row in group["obligations"].values()]
    document["n_required"] = len(rows)
    document["n_passed"] = sum(row["status"] == "pass" for row in rows)
    document["n_failed"] = sum(row["status"] == "fail" for row in rows)
    # A model row this Spike-only route cannot close for want of L3 evidence is reported as such, not as
    # unfinished work; it still keeps the group (and so the coverage) from passing.
    document["n_l3_evidence_missing"] = sum(row["status"] == MODEL_L3_MISSING for row in rows)
    document["n_l3_not_run"] = sum(row.get("l3") == "not_run" for row in rows)  # reported, never gating
    document["n_unfinished"] = (len(rows) - document["n_passed"] - document["n_failed"]
                                - document["n_l3_evidence_missing"])
    # Rows keep their native verdict; a group's cohort/structural veto lives on the group. Rows are
    # withheld ("skipped" over a native pass) only when the final custody check fails.
    document["n_native_passed"] = sum(row.get("native_status") == "pass" for row in rows)
    document["n_closure_withheld"] = sum(
        row["status"] == "skipped" and row.get("native_status") == "pass" for row in rows)
    document["complete"] = bool(rows) and document["n_unfinished"] == 0
    document["passed"] = document["complete"] and document["n_failed"] == 0 and not document[
        "n_l3_evidence_missing"] and all(
        group.get("passed") is True for group in document["groups"].values())
    document["status"] = "pass" if document["passed"] else "fail" if document["n_failed"] else "incomplete"
    return document


def _capsule_group(capsules):
    names = [cap["name"] for cap in capsules]
    if not names or len(names) != len(set(names)):
        raise ValueError("final coverage requires a nonempty unique native capsule roster")
    return {"passed": False, "obligations": {
        name: {"status": "pending", "reason": "not yet graded"} for name in sorted(names)}}


def _closure_vetoes(score, *, exact, uncertified) -> list[str]:
    """The group-level closure conditions ``score`` (already model-L3 closed) does not meet."""
    from .feedback.qa import _original_cohort_complete

    vetoes = []
    if not exact:
        vetoes.append("native roster differs from the required obligations")
    if uncertified:
        vetoes.append("whole-model rows lack L3 evidence: " + ", ".join(sorted(uncertified)))
    if not _original_cohort_complete(score):
        vetoes.append("native cohort is incomplete")
    for field in ("gradeable", "numeric_all_exact", "structural_evidence_all_pass"):
        if score.get(field) is not True:
            vetoes.append(f"{field} is {score.get(field)!r}")
    if (score.get("functional_public_evidence") or {}).get("claim") != "functional_corroboration":
        vetoes.append("no functional public corroboration claim")
    return vetoes


def _apply_score(group, score, model_l3=None):
    """Only the final native score, after offload/model vetoes, can close a row.

    ``model_l3`` (:func:`.joint_final_model_l3.close`) closes whole-model rows whose only gap is L3:
    with an exact-byte certificate they pass and count toward the cohort; without one they are
    reported as ``l3_evidence_missing``.

    Each row keeps the native grader's own verdict. A cohort, structural or numeric veto belongs to
    the group: it keeps ``group["passed"]`` (and so the coverage) false and is named in
    ``group["closure_vetoes"]``, but it never rewrites another row's pass, so one failing case cannot
    hide every case that passed.
    """
    from .feedback.qa import _original_cohort_complete
    from .joint_final_model_l3 import cohort_score

    model_l3 = model_l3 or {}
    certified = {name for name, closure in model_l3.items() if closure.get("certificate")}
    not_run = {name for name, closure in model_l3.items() if closure.get("not_run")}  # best-effort L3
    rows = score.get("per_capsule") or []
    names = [row.get("capsule") for row in rows]
    expected = set(group["obligations"])
    exact = len(names) == len(set(names)) and set(names) == expected
    for row in rows:
        name = row.get("capsule")
        if name not in expected:
            continue
        status = row.get("status")
        group["obligations"][name] = {
            "status": status if status in {"pass", "fail"} else "skipped",
            "native_status": status, "tiers": row.get("tiers", {}), "numeric": row.get("numeric"),
            "reason": row.get("background_deferred") or row.get("failure_detail") or row.get("tier_reasons"),
        }
        closure = model_l3.get(name)
        if closure is not None:
            obligation = group["obligations"][name]
            if closure.get("certificate"):
                obligation.update(status="pass", l3_certificate=closure["certificate"], reason=None)
            elif closure.get("not_run"):
                obligation.update(status="pass", l3="not_run", reason=closure["not_run"])
            else:
                obligation.update(status=MODEL_L3_MISSING, reason=closure.get("missing"))
    vetoes = _closure_vetoes(cohort_score(score, certified | not_run), exact=exact,
                             uncertified=set(model_l3) - certified - not_run)
    group["passed"] = not vetoes
    group["closure_vetoes"] = vetoes
    group["native_cohort_complete"] = _original_cohort_complete(score)
    if model_l3:
        group["model_l3"] = {"certified": sorted(certified), "not_run": sorted(not_run),
                             "missing": sorted(set(model_l3) - certified - not_run)}
    group["structural_evidence_all_pass"] = score.get("structural_evidence_all_pass")
    if score.get("failure"):
        group["failure"] = score["failure"]


def _apply_program_result(group, result, compiler_sha256):
    """Keep source-selected program obligations separate from core conformance."""
    from merlin.targetgen.capsule_common import tier_status

    rows = result.get("observed") or {}
    expected = set(group["obligations"])
    exact = (set(rows) == expected and set(result.get("expected_cases", [])) == expected
             and set(result.get("requested_cases", [])) == expected
             and result.get("candidate_sha256") == compiler_sha256)
    closed = exact and result.get("passed") is True and result.get("complete") is True
    for name in expected:
        row = rows.get(name) or {}
        tiers = row.get("tiers") or {}
        l2 = tiers.get("L2")
        passed = (closed and row.get("status") == "pass" and tier_status(row.get("numeric")) == "pass"
                  and all(tier_status(tiers.get(t)) == "pass" for t in ("L0", "L1", "L2"))
                  and isinstance(l2, dict) and l2.get("engine") == "spike"
                  and l2.get("fidelity") == "functional_model" and l2.get("derived_from_rtl") is False)
        group["obligations"][name] = {
            "status": "pass" if passed else "fail" if exact and row.get("status") == "fail" else "skipped",
            "native_status": row.get("status"), "numeric": row.get("numeric"), "tiers": tiers,
            "reason": row.get("failure") or (None if passed else "complete same-compiler Spike evidence missing"),
        }
    group["passed"] = closed and all(row["status"] == "pass" for row in group["obligations"].values())


def validate(prepared, *, inputs, frozen: Path, compiler_sha256: str):
    from merlin.common.digest import sha256_file
    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen.target_experiment import load_target_experiment

    from .brokers.model_check import PROGRAMS, execute_visible
    from .context import readback_kwargs
    from .feedback import public_programs as PP
    from .feedback import public_scalability as PS
    from .feedback.loop_grading import compiler_execution
    from .feedback.selfcheck import _adapters
    from .feedback.visible_model_gate import _program_row
    from .joint_model_activation import grade_model_scope

    state = joint_control.selected()
    if state is None or joint_control.remaining(drain=True) > 0:
        raise ValueError("final coverage is admitted only at the joint campaign drain")
    if (prepared.run_dir / "joint_epoch_stop.json").is_file():
        raise ValueError("an intermediate epoch cannot request final coverage")
    prepared.verify_inputs()
    if hash_tree(frozen)["sha256"] != compiler_sha256:
        raise ValueError("final compiler does not match its frozen source pin")
    context = inputs.context
    te = load_target_experiment(context.descriptor)
    contract = inputs.contract or context.repo / "merlin/contract"
    core = inputs.public_root() if callable(inputs.public_root) else inputs.public_root
    # Do not call the smoke selector or development_roots: these intentionally
    # omit the final-only capstones. The native sealed public view contains all.
    core_caps = CR.discover_capsules(core, labels={"public", "dev"}, contract=contract)
    final_only = set(state["policy"].get("final_only_capsules", ()))
    if not final_only <= {cap["name"] for cap in core_caps}:
        raise ValueError("sealed final core view is missing required final-only capsules")
    selected = PS.selected(context)
    if selected is None:
        raise ValueError("joint final coverage has no required scalability selection")
    scale_root, scale_record = selected
    scale_cases = scale_root / "cases"
    scale_caps = CR.discover_capsules(scale_cases, labels={"public", "dev"}, contract=contract)
    if {cap["name"] for cap in scale_caps} != set(scale_record["cases"]):
        raise ValueError("native scalability roster differs from the sealed selection")
    if {cap["name"] for cap in core_caps} & {cap["name"] for cap in scale_caps}:
        raise ValueError("core and scalability obligations overlap")
    output = prepared.run_dir / "joint_final_coverage"
    output.mkdir(exist_ok=False)
    path = output / "coverage.json"
    groups = {"core": _capsule_group(core_caps), "scalability": _capsule_group(scale_caps),
              "models": {"passed": False, "obligations": {
                  f"{model}/{program}": {"status": "pending", "reason": "not yet checked"}
                  for model, programs in PROGRAMS.items() for program in programs}}}
    programs = PP.selected(context)
    if programs is not None:
        if set(programs[2]) & {cap['name'] for cap in core_caps + scale_caps}:
            raise ValueError("selected program and native capsule obligations overlap")
        groups["programs"] = _capsule_group([{"name": name} for name in programs[2]])
    report = {
        "schema": "merlin.joint_final_coverage.v1", "required": True,
        "compiler_sha256": compiler_sha256, "descriptor_sha256": te.descriptor_sha256,
        "policy_sha256": state["policy_sha256"],
        "deadline_epoch": state["started_at_epoch"] + state["policy"]["duration_s"],
        "scope": "all selected public/dev core, scalability and connected programs plus required visible full models",
        "engine": "spike", "tiers": ["L0", "L1", "L2"],
        "rtl_qualified": False, "timing_qualified": False, "final_only": sorted(final_only),
        "reuse": "native exact execution caches only; no development-verdict import",
        "groups": groups,
    }

    # Visible full models run beside the in-process stages (see below); one lock serializes every
    # report mutation and write between the two threads.
    lock = threading.RLock()

    def save():
        with lock:
            _write(path, _rollup(report))

    def expired():
        return "original joint campaign deadline exhausted" if joint_control.remaining() <= 0 else None

    def budget():
        return max(0, int(joint_control.remaining()))

    save()  # Persist the whole required denominator before starting any work.
    adapters, engine = _adapters("spike", context.target, te.sim_via, **readback_kwargs(context))
    if engine != "spike" or set(adapters) != {"L2"}:
        raise ValueError("final cheap coverage requires the exact Spike-only adapter")

    def capsule_stage(name, root):
        group = groups[name]
        try:
            if budget() < 1:
                raise TimeoutError("original joint campaign deadline exhausted")
            prepared.verify_inputs()
            if inputs.verify_grade_inputs is not None:
                inputs.verify_grade_inputs()
            runs = output / name
            with compiler_execution(inputs, candidate=frozen, run_dir=prepared.run_dir):
                with (output / f"{name}.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                    score = CG.grade(
                        frozen, capsules_root=root, runs_root=runs, target=context.target,
                        labels={"public", "dev"}, contract=contract, target_experiment=te,
                        oracle_adapters=adapters, functional_public_tier="L2",
                        timeout=max(1, min(900, budget())), max_workers=CG.default_grade_workers(),
                        additional_forbidden=inputs.additional_forbidden,
                        model_snapshot_root=runs / ".private_model_sources",
                        defer_pending=expired, defer_running=True,
                    )
            score_path = output / f"{name}-score.json"
            _write(score_path, score)
            with lock:
                group["evidence"] = {"path": str(score_path), "sha256": sha256_file(score_path)}
            model_l3 = None
            if name == "core":
                from .joint_final_model_l3 import close as close_model_l3

                try:
                    model_l3 = close_model_l3(score, runs=runs, workspace=prepared.workspace, frozen=frozen,
                                              compiler_sha256=compiler_sha256, context=context)
                except Exception as exc:  # noqa: BLE001 -- the native rows still close as they did
                    with lock:
                        group["model_l3_error"] = f"{type(exc).__name__}: {exc}"
            with lock:
                _apply_score(group, score, model_l3)
        except Exception as exc:
            with lock:
                group["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with lock:
                for row in group["obligations"].values():
                    if row["status"] == "pending":
                        row.update(status="skipped", reason=group.get(
                            "error", "native grader did not close this obligation"))
            save()

    def program_stage():
        group = groups["programs"]
        try:
            if budget() < 1:
                raise TimeoutError("original joint campaign deadline exhausted")
            prepared.verify_inputs()
            if inputs.verify_grade_inputs is not None:
                inputs.verify_grade_inputs()
            with compiler_execution(inputs, candidate=frozen, run_dir=prepared.run_dir):
                with (output / "programs.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                    result = grade_model_scope(PP._grade)(
                        frozen, context=context, output=output / "programs", timeout=budget(),
                        contract=contract, max_workers=CG.default_grade_workers())
            evidence = output / "programs-result.json"
            _write(evidence, result)
            with lock:
                group["evidence"] = {"path": str(evidence), "sha256": sha256_file(evidence)}
                _apply_program_result(group, result, compiler_sha256)
        except Exception as exc:
            with lock:
                group["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with lock:
                for row in group["obligations"].values():
                    if row["status"] == "pending":
                        row.update(status="skipped", reason=group.get(
                            "error", "native program did not close this obligation"))
            save()

    def model_stage():
        for model, programs in PROGRAMS.items():
            try:
                if budget() < 1:
                    raise TimeoutError("original joint campaign deadline exhausted")
                joint_control.require_full_model(engine="spike")
                result = execute_visible(
                    prepared.workspace, {"action": "check", "model": model, "split": "visible", "mode": "full"},
                    snapshot_source=frozen, timeout_s=budget(), stop=lambda: expired() is not None,
                    context=inputs.context,
                    compiler_bundle_manifest=prepared.run_dir / "input_bundle_manifest.yaml",
                )
                if result.get("source_sha256") != compiler_sha256:
                    raise ValueError("full-model check returned another compiler identity")
                result_path = output / f"model-{model}.json"
                _write(result_path, result)
                rows = {}
                for program in programs:
                    row = _program_row(model, program, result)
                    row["evidence"] = {"path": str(result_path), "sha256": sha256_file(result_path)}
                    rows[f"{model}/{program}"] = row
            except Exception as exc:
                rows = {f"{model}/{program}": {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
                        for program in programs}
            with lock:
                groups["models"]["obligations"].update(rows)
            save()

    # Core first (its model-L3 closure reads only its own rows). The visible full models are
    # subprocesses, so they run beside the remaining in-process stages instead of after them: one
    # shared deadline can then no longer skip them, or the selected programs, behind the large
    # scalability suite. The thread carries this context's compiler owners like any other stage.
    capsule_stage("core", core)
    models = threading.Thread(target=contextvars.copy_context().run, args=(model_stage,),
                              name="final-coverage-models", daemon=True)
    models.start()
    try:
        if programs is not None:
            program_stage()
        capsule_stage("scalability", scale_cases)
    finally:
        models.join()
    for row in groups["models"]["obligations"].values():
        if row["status"] == "pending":
            row.update(status="skipped", reason="full-model stage did not close this obligation")
    groups["models"]["passed"] = all(
        row["status"] == "pass" for row in groups["models"]["obligations"].values())
    try:
        prepared.verify_inputs()
        if hash_tree(frozen)["sha256"] != compiler_sha256:
            raise ValueError("frozen compiler changed during final coverage")
    except Exception as exc:
        report["custody_error"] = f"{type(exc).__name__}: {exc}"
        for group in groups.values():
            group["passed"] = False
            for row in group["obligations"].values():
                if row["status"] == "pass":
                    row.update(status="skipped", reason="final input/compiler custody check failed")
    save()  # Coverage is published here; the completion-order pass below can only annotate it.
    report["completion_order"] = _completion_order_after_coverage(groups, output, context.target, budget)
    save()
    return {**report, "artifact": {"path": str(path), "sha256": sha256_file(path)}}


def _completion_order_after_coverage(groups, output, target, remaining):
    """The race check over published native passes, best-effort in the time left (never gating)."""
    from merlin.targetgen import capsule_grade as CG
    from merlin.targetgen import completion_order_gate as G

    from .joint_final_model_l3 import _result

    results = {}
    for name in ("core", "scalability"):
        for case, row in groups[name]["obligations"].items():
            found = _result(output / name, case)[0] if row["status"] == "pass" else None
            if found is not None:
                results[case] = found
    try:
        return G.after_coverage(target, results, remaining=remaining, workers=CG.default_grade_workers())
    except Exception as exc:  # noqa: BLE001 -- an advisory pass never disturbs published coverage
        return {"status": "not_run", "error": f"{type(exc).__name__}: {exc}"}
