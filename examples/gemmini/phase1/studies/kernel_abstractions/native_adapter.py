"""Adapt native freezes to ordinary evaluator jobs and the private physical panel.

All paths and effective dependency selections are explicit pinned operator inputs.
There is no author launcher or new numerical grader here. ``stage`` uses ordinary
source intake/admission/public_scalability staging in a private directory;
``plan`` writes commands, never executes them. ``collect`` reopens the ordinary
issuer and rich qualification through evaluate_panels' existing custody join.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml
from physical_panel import pin, read, reopen, write

from merlin.common.tree_hash import hash_tree

SCHEMA = "merlin.study.native_panel_plan.v1"


def _private(path):
    path = Path(path).absolute()
    if path.exists() or path != path.resolve():
        raise ValueError("select an unused canonical private directory")
    path.mkdir(parents=True, mode=0o700)
    return path


def stage(panel_path, descriptor_path, evidence_root, output):
    """Stage all new contracts once through the existing ordinary source owners."""
    from merlin_experiments.corpus import public_scalability as S
    from merlin_experiments.corpus import scalability_intake
    from merlin_experiments.phase0.evidence import load_exported_evidence
    from merlin_experiments.runner import fingerprint

    from merlin.targetgen.target_experiment import load_target_experiment

    panel_pin = pin(panel_path)
    panel = read(panel_pin)
    te = load_target_experiment(descriptor_path)
    output = _private(output)
    evidence = load_exported_evidence(Path(evidence_root))
    for case in panel["cases"]:
        reopen(case["capsule"])
        reopen(case["source"])
    selection = scalability_intake.prepare(
        [Path(c["capsule"]["path"]).parent for c in panel["cases"]], output / "intake",
        target=te.target, evidence=evidence, evidence_root=Path(evidence_root))
    bound = S.stage(selection, output / "suite", te=te)
    experiment = output / "experiment"
    shutil.copytree(Path(descriptor_path).resolve().parent, experiment)
    descriptor = experiment / Path(descriptor_path).name
    document = yaml.safe_load(descriptor.read_text())
    document.setdefault("workload_spec", {})[S.KEY] = bound
    descriptor.chmod(0o600)
    descriptor.write_text(yaml.safe_dump(document, sort_keys=False))
    actual = S.selected(load_target_experiment(descriptor))
    if actual is None or set(actual[1]["cases"]) != {c["case"] for c in panel["cases"]}:
        raise ValueError("ordinary staged suite differs from the complete frozen panel")
    result = {"schema": SCHEMA + ".suite", "panel": panel_pin, "descriptor": pin(descriptor),
              "selection": pin(selection), "root": bound, "suite_sha256": fingerprint(output / "suite"),
              "operator_only": True, "author_visible": False, "executed": False}
    write(output / "suite.json", result)
    return result


def freeze_subject(row):
    """Read a native final summary or declared checkpoint; never rediscover source."""
    from evaluate_panels import _frozen_source

    run = Path(row["run_dir"]).resolve(strict=True)
    if row.get("snapshot", "final") == "final":
        authority = pin(run / "joint_campaign_summary.json")
        summary = read(authority)
        if summary.get("completion_reason") != "freeze_drain":
            raise ValueError("final subject lacks a completed native freeze boundary")
        source, digest = summary["frozen_compiler"], summary["compiler_sha256"]
        frozen = row["frozen_unix"]
    else:
        offset = row["snapshot"]
        if type(offset) is not int or offset < 0:
            raise ValueError("snapshot must be final or a nonnegative declared offset")
        authority = pin(run / "study_snapshots/index.json")
        index = read(authority)
        matches = [s for s in index["snapshots"] if s["offset_s"] == offset]
        if len(matches) != 1 or matches[0].get("status") != "frozen":
            raise ValueError("selected native checkpoint is missing or unavailable")
        snapshot = matches[0]
        source, digest = snapshot["source"], snapshot["compiler_sha256"]
        frozen = index["started_at_epoch"] + snapshot["observed_elapsed_s"]
    if hash_tree(Path(source))["sha256"] != digest:
        raise ValueError("native frozen source changed")
    selected = _frozen_source({"path": source, "rules": row["rules"], "settings": row["settings"],
                               "frozen_unix": frozen})
    return {"id": row["id"], "role": row.get("role", "final_compiler"),
            "replica": row.get("replica"), "treatment": row.get("treatment", row["id"]),
            "baseline_subject": row.get("baseline_subject"), "native_authority": authority,
            "snapshot": row.get("snapshot", "final"), **selected}


def plan(panel_path, subjects_path, artifacts_path, measurement_path, output):
    """Write explicit ordinary selfcheck jobs and a physical selection; do not run."""
    from merlin_experiments.corpus import public_scalability as S
    from merlin_experiments.runner import fingerprint

    from merlin.targetgen.target_experiment import load_target_experiment

    panel_pin, subjects_pin, artifacts_pin, measurement_pin = map(pin, (panel_path, subjects_path, artifacts_path, measurement_path))
    panel, requested, artifacts, measurement = map(read, (panel_pin, subjects_pin, artifacts_pin, measurement_pin))
    if panel.get("schema") != "merlin.study.transfer_panel.v1" or len(panel["cases"]) != 34:
        raise ValueError("native study requires the complete 34-case transfer panel")
    subjects = [freeze_subject(row) for row in requested["subjects"]]
    if not subjects or len({s["id"] for s in subjects}) != len(subjects):
        raise ValueError("select unique frozen subject IDs")
    output = Path(output).absolute()
    for subject in subjects:
        source = Path(subject["source"]["path"])
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("private evaluation cannot modify a compiler tree")
    _private(output)
    cases = panel["cases"]
    names = [c["case"] for c in cases]
    jobs, runtime_pins = [], []
    for row, subject in zip(requested["subjects"], subjects, strict=True):
        config_pin = row["runtime_config"]
        config = read(config_pin)
        runtime_pins.append(config_pin)
        for key in ("python", "descriptor", "environment", "rtl_facts", "dependency_selection", "run_environment"):
            reopen(config[key])
            runtime_pins.append(config[key])
        if read(config["dependency_selection"]).get("compiler_sha256") not in (None, subject["source"]["sha256"]):
            raise ValueError("effective dependency context belongs to another frozen compiler")
        te = load_target_experiment(reopen(config["descriptor"]))
        selected = S.selected(te)
        if selected is None or set(selected[1]["cases"]) != set(names):
            raise ValueError("runtime descriptor must select exactly the frozen transfer panel")
        root, _ = selected
        workers = config["workers"]
        if type(workers) is not int or not 1 <= workers <= 15 or type(config["timeout_s"]) is not int or config["timeout_s"] <= 0:
            raise ValueError("evaluation requires explicit bounded workers and request timeout")
        directory = output / subject["id"]
        directory.mkdir(mode=0o700)
        environment = read(config["environment"])
        native_environment = yaml.safe_load(reopen(config["run_environment"]).read_text())
        full = (native_environment.get("full_compiler_workspace") or {}).get("selection")
        full_env = environment.get("MERLIN_FULL_COMPILER_SELECTION")
        if full is not None:
            from merlin_experiments.phase1.full_compiler_workspace import select

            selected_full = select(**{key: full[key] for key in ("root", "manifest_path", "manifest_sha256")})
            selected_full.verify()
            if full_env and json.loads(full_env) != full:
                raise ValueError("evaluation acquired a different full compiler selection")
            environment["MERLIN_FULL_COMPILER_SELECTION"] = json.dumps(full, sort_keys=True)
        elif full_env:
            raise ValueError("evaluation acquired a full compiler absent from the native run")
        # This operator evaluation has no author or inherited campaign clock.
        # Deployment instruction/primitive audits remain separate mandatory gates.
        environment.pop("MERLIN_JOINT_CAMPAIGN_STATE", None)
        environment["MERLIN_FROZEN_PUBLIC_SCALABILITY"] = json.dumps({"root": str(root), "descriptor_sha256": te.descriptor_sha256})
        environment["TMPDIR"] = str(directory / "tmp")
        (directory / "tmp").mkdir()
        write(directory / "environment.json", environment)
        result_path = directory / "result.json"
        command = [str(reopen(config["python"])), "-B", "-m", "merlin_experiments.phase1.feedback.selfcheck",
                   "--descriptor", str(reopen(config["descriptor"])), "--repo", config["repo"],
                   "--suite", "scalability", "--sim", "spike", "--tiers", "L0,L1,L2",
                   "--capsules", ",".join(names), "--submission", subject["source"]["path"],
                   "--workers", str(workers), "--timeout", str(config["timeout_s"]),
                   "--rtl-facts", str(reopen(config["rtl_facts"])), "--out", str(result_path)]
        for flag, key in (("--contract", "contract"), ("--compiler-bundle-manifest", "compiler_bundle_manifest")):
            if config.get(key):
                selected_path = Path(config[key]["path"])
                if key == "contract" and selected_path.is_dir():
                    if hash_tree(selected_path)["sha256"] != config[key]["sha256"]:
                        raise ValueError("selected grading contract directory changed")
                else:
                    reopen(config[key])
                    runtime_pins.append(config[key])
                command.extend([flag, str(selected_path)])
        binding = native_environment.get("additional_compiler_library")
        if binding is not None:
            from merlin_experiments.phase2.compiler_library_state import restore_selection, verify_binding

            origin = restore_selection(binding["origin"])
            snapshot = verify_binding(binding, origin)
            contract_pin = config.get("library_contract")
            if contract_pin is None or read(contract_pin) != binding["snapshot"]["contract"]:
                raise ValueError("native additional library requires its exact selected contract")
            runtime_pins.append(contract_pin)
            command.extend(["--compiler-library-root", str(snapshot.root),
                            "--compiler-library-contract", str(reopen(contract_pin)),
                            "--compiler-library-contract-sha256", snapshot.contract.sha256])
        subject["runtime_config"] = config_pin
        subject["effective_dependencies"] = config["dependency_selection"]
        jobs.append({"subject": subject["id"], "source": subject["source"], "descriptor": config["descriptor"],
                     "suite_sha256": fingerprint(root), "cases": names, "argv": command,
                     "environment": pin(directory / "environment.json"), "cwd": str(directory), "result": str(result_path)})
    evaluation = {"schema": SCHEMA, "panel": panel_pin, "subjects": subjects, "cases": cases,
                  "inputs": [panel_pin, subjects_pin, artifacts_pin, measurement_pin, *runtime_pins],
                  "jobs": jobs, "executed": False, "author_visible": False,
                  "instruction_policy": "retain selected primitive/no-FSM binary audits before paid execution",
                  "numerical_grader": "ordinary phase1.feedback.selfcheck/public_scalability only"}
    evaluation_path = output / "plan.json"
    write(evaluation_path, evaluation)
    selected_sources = {s["id"]: s["source"]["sha256"] for s in subjects}
    compiled = []
    for artifact in artifacts.get("cells", []):
        key = artifact["subject"], artifact["case"]
        if key[0] not in selected_sources or key[1] not in names:
            raise ValueError("artifact cell is outside the selected subject/panel roster")
        if artifact.get("compiler_sha256") not in (None, selected_sources[key[0]]):
            raise ValueError("compiled cell names another frozen compiler")
        compiled.append({**artifact, "compiler_sha256": selected_sources[key[0]]})
    physical = {"schema": "merlin.study.physical_panel.v1", "evaluation_plan": pin(evaluation_path),
                "panel": panel_pin, "subjects": list(selected_sources), "snapshot": measurement["snapshot"],
                "rounds": 3, "domain": measurement["domain"], "toolchain": measurement["toolchain"],
                "compiled": compiled}
    if measurement.get("cases"):
        physical["cases"] = measurement["cases"]
    write(output / "physical-selection.json", physical)
    return evaluation


def collect(plan_path, *, physical_collections=()):
    """Project original issued results; unissued/missing observations remain unknown."""
    from evaluate_panels import _issued_qualification, _numeric_pass
    from merlin_experiments.phase1.feedback.joint_performance_report import compare
    from physical_panel import apply_physical

    plan_pin = pin(plan_path)
    plan = read(plan_pin)
    if plan.get("schema") != SCHEMA:
        raise ValueError("unsupported native panel plan")
    for item in plan["inputs"]:
        reopen(item)
    subjects = {s["id"]: s for s in plan["subjects"]}
    cases = {c["case"]: c for c in plan["cases"]}
    cells = []
    for job in plan["jobs"]:
        subject = subjects[job["subject"]]
        if hash_tree(subject["source"]["path"])["sha256"] != subject["source"]["sha256"]:
            raise ValueError("frozen compiler changed during evaluation")
        reopen(job["environment"])
        observed, error, result_pin = {}, None, None
        result_path = Path(job["result"])
        if result_path.exists():
            try:
                result_pin = pin(result_path)
                result = read(result_pin)
                rich, _, _ = _issued_qualification(result_path, result,
                    source_sha256=subject["source"]["sha256"], suite_sha256=job["suite_sha256"])
                observed = rich["observed"]
                if not set(observed) <= set(job["cases"]):
                    raise ValueError("ordinary evaluator returned foreign cases")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                observed, error = {}, f"unverified result: {type(exc).__name__}: {exc}"
        for name in job["cases"]:
            row = observed.get(name)
            state = ("missing" if row is None else "pass" if _numeric_pass(row) else
                     "fail" if row.get("status") == "fail" else "incomplete")
            cells.append({"subject": subject["id"], "case": name, "split": cases[name].get("split", "heldout"),
                          "family": cases[name]["family"], "role": subject["role"], "source": subject["source"],
                          "baseline_subject": subject.get("baseline_subject"), "state": state,
                          "numeric": (row or {}).get("numeric"), "performance": (row or {}).get("performance"),
                          "failure": (row or {}).get("failure"), "diagnostic": error, "result": result_pin,
                          "physical": {"status": "UNKNOWN", "cycles": None}})
    lookup = {(c["subject"], c["case"]): c for c in cells}
    for cell in cells:
        other = lookup.get((cell.get("baseline_subject"), cell["case"]))
        cell["model_comparison"] = (compare(other.get("performance") or {}, cell.get("performance") or {})
             if other and other["state"] == cell["state"] == "pass" else
             {"status": "UNKNOWN", "candidate_to_baseline_ratio": None})
    summaries = []
    for name in subjects:
        selected = [c for c in cells if c["subject"] == name]
        summaries.append({"subject": name, "split": "heldout", "cells": len(selected),
                          "states": dict(Counter(c["state"] for c in selected)), "physical_status": "UNKNOWN"})
    collection = {"schema": "private.kernel_abstractions.evaluation_collection.v1", "native_plan": True,
                  "plan": plan_pin, "cells": cells, "summaries": summaries, "physical_timing_qualified": False}
    return apply_physical(collection, physical_collections) if physical_collections else collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("stage")
    for name in ("panel", "descriptor", "evidence-root", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("plan")
    for name in ("panel", "subjects", "artifacts", "measurement", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p = sub.add_parser("collect")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--physical-collection", action="append", default=[], type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "stage":
        stage(args.panel, args.descriptor, args.evidence_root, args.out)
    elif args.action == "plan":
        plan(args.panel, args.subjects, args.artifacts, args.measurement, args.out)
    else:
        write(args.out, collect(args.plan, physical_collections=args.physical_collection))


if __name__ == "__main__":
    main()
