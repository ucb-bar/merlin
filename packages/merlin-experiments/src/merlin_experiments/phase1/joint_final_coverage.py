"""Final-drain functional coverage of the immutable native compiler snapshot.

This is an operator-owned completion route, not an author request or certificate.
It never consumes a development verdict, changes a numerical contract, runs RTL,
or resets the campaign clock. Native graders retain their exact artifact caches.
"""
from __future__ import annotations

import json
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
    document["n_unfinished"] = (len(rows) - document["n_passed"] - document["n_failed"]
                                - document["n_l3_evidence_missing"])
    # A group whose cohort/structural closure is unverified withholds its passing rows ("skipped");
    # report what the native grader itself executed and passed so the evidence stays readable.
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


def _apply_score(group, score, model_l3=None):
    """Only the final native score, after offload/model vetoes, can close a row.

    ``model_l3`` (:func:`.joint_final_model_l3.close`) closes whole-model rows whose only gap is L3:
    with an exact-byte certificate they pass and count toward the cohort; without one they are
    reported as ``l3_evidence_missing``.
    """
    from .feedback.qa import _original_cohort_complete
    from .joint_final_model_l3 import cohort_score

    model_l3 = model_l3 or {}
    certified = {name for name, closure in model_l3.items() if closure.get("certificate")}
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
            else:
                obligation.update(status=MODEL_L3_MISSING, reason=closure.get("missing"))
    group["passed"] = (
        exact and not set(model_l3) - certified
        and _original_cohort_complete(cohort_score(score, certified))
        and score.get("gradeable") is True
        and score.get("numeric_all_exact") is True
        and score.get("structural_evidence_all_pass") is True
        and (score.get("functional_public_evidence") or {}).get("claim") == "functional_corroboration"
    )
    if not group["passed"]:
        # A perfect-looking row list cannot hide a package/cohort/structural veto.
        for row in group["obligations"].values():
            if row["status"] == "pass" and (
                not exact or score.get("structural_evidence_all_pass") is not True
                or score.get("numeric_all_exact") is not True or score.get("gradeable") is not True
                or (score.get("functional_public_evidence") or {}).get("claim") != "functional_corroboration"
            ):
                row.update(status="skipped", reason="native cohort or structural closure is unverified")
    group["native_cohort_complete"] = _original_cohort_complete(score)
    if model_l3:
        group["model_l3"] = {"certified": sorted(certified), "missing": sorted(set(model_l3) - certified)}
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

    def save():
        _write(path, _rollup(report))

    def expired():
        return "original joint campaign deadline exhausted" if joint_control.remaining() <= 0 else None

    def budget():
        return max(0, int(joint_control.remaining()))

    save()  # Persist the whole required denominator before starting any work.
    adapters, engine = _adapters("spike", context.target, te.sim_via, **readback_kwargs(context))
    if engine != "spike" or set(adapters) != {"L2"}:
        raise ValueError("final cheap coverage requires the exact Spike-only adapter")
    for name, root in (("core", core), ("scalability", scale_cases)):
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
            group["evidence"] = {"path": str(score_path), "sha256": sha256_file(score_path)}
            model_l3 = None
            if name == "core":
                from .joint_final_model_l3 import close as close_model_l3

                try:
                    model_l3 = close_model_l3(score, runs=runs, workspace=prepared.workspace, frozen=frozen,
                                              compiler_sha256=compiler_sha256, context=context)
                except Exception as exc:  # noqa: BLE001 -- the native rows still close as they did
                    group["model_l3_error"] = f"{type(exc).__name__}: {exc}"
            _apply_score(group, score, model_l3)
        except Exception as exc:
            group["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            for row in group["obligations"].values():
                if row["status"] == "pending":
                    row.update(status="skipped", reason=group.get(
                        "error", "native grader did not close this obligation"))
            save()

    if programs is not None:
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
            group["evidence"] = {"path": str(evidence), "sha256": sha256_file(evidence)}
            _apply_program_result(group, result, compiler_sha256)
        except Exception as exc:
            group["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            for row in group["obligations"].values():
                if row["status"] == "pending":
                    row.update(status="skipped", reason=group.get(
                        "error", "native program did not close this obligation"))
            save()

    for model, programs in PROGRAMS.items():
        try:
            if budget() < 1:
                raise TimeoutError("original joint campaign deadline exhausted")
            joint_control.require_full_model(engine="spike")
            result = execute_visible(
                prepared.workspace, {"action": "check", "model": model, "split": "visible", "mode": "full"},
                snapshot_source=frozen, timeout_s=budget(), stop=lambda: expired() is not None,
                context=inputs.context, compiler_bundle_manifest=prepared.run_dir / "input_bundle_manifest.yaml",
            )
            if result.get("source_sha256") != compiler_sha256:
                raise ValueError("full-model check returned another compiler identity")
            result_path = output / f"model-{model}.json"
            _write(result_path, result)
            for program in programs:
                row = _program_row(model, program, result)
                row["evidence"] = {"path": str(result_path), "sha256": sha256_file(result_path)}
                groups["models"]["obligations"][f"{model}/{program}"] = row
        except Exception as exc:
            for program in programs:
                groups["models"]["obligations"][f"{model}/{program}"] = {
                    "status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
        save()
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
    save()
    return {**report, "artifact": {"path": str(path), "sha256": sha256_file(path)}}
