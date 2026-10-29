"""Every frozen ablation retains both complete numerical/performance panels."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


def _module():
    path = repo_root() / "examples/gemmini/phase1/studies/kernel_abstractions/evaluate_panels.py"
    spec = importlib.util.spec_from_file_location("evaluation_panels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path, monkeypatch, count=12, *, training_count=3, heldout_count=6):
    m = _module()
    root = tmp_path / "inputs"
    root.mkdir()
    selected = {
        "train": [f"train{i}" for i in range(training_count)],
        "heldout": [f"hidden{i}" for i in range(heldout_count)],
    }
    cases, suites = [], {}
    for split, names in selected.items():
        directory = root / split
        directory.mkdir()
        descriptor = directory / "descriptor.yaml"
        descriptor.write_text("workload: selected-private-panel\n")
        suites[split] = dict(descriptor=m._pin(descriptor), root=dict(path=str(directory), sha256=split * 16))
        for index, name in enumerate(names):
            source = directory / (name + ".mlir")
            source.write_text("module { /* " + name + " */ }\n")
            cases.append(
                dict(
                    case=name,
                    family=str(index % 3),
                    split=split,
                    source=m._pin(source),
                    contract={"source_shape": [index + 1, 16, 16]},
                )
            )
    manifest = root / "manifest.json"
    m._write(manifest, dict(cases=cases, suites=suites))
    panel = root / "small-panel.json"
    m._write(
        panel,
        dict(
            inputs_manifest=m._pin(manifest),
            selection=dict(training_case_ids=selected["train"], heldout_case_ids=selected["heldout"]),
        ),
    )
    contract = root / "target_contract.yaml"
    contract.write_text("target: selected\n")
    facts = root / "facts.json"
    m._write(facts, {"derived": True})
    performance = root / "performance.json"
    m._write(performance, {"runtime": "same-reviewed-model"})
    runner = root / "run.py"
    runner.write_text("# selected ordinary runner\n")
    rows, frozen = [], {}
    for index in range(count):
        private = root / f"run-{index}"
        private.mkdir()
        env = dict(
            MERLIN_TARGET_CONTRACT=str(contract),
            MERLIN_JOINT_PERFORMANCE_SELECTION=json.dumps(m._pin(performance)),
            TMPDIR=str(private / "tmp"),
        )
        m._write(private / "evaluator-environment.json", env)
        config = dict(
            private=str(private),
            workspace=str(root),
            framework=str(root),
            python="/usr/bin/python3",
            facts=str(facts),
            environment=env,
            runner=m._pin(runner),
            tooling=f"treatment{index // 2}",
            training_cases=selected["train"],
            reference_guidance="on" if index % 2 else "off",
            library_guidance=True,
            schedule_guidance=True,
        )
        path = private / "config.json"
        m._write(path, config)
        rows.append(dict(config=str(path), config_sha256=m._pin(path)["sha256"], replica=index % 2))
        source = root / f"frozen-{index}"
        source.mkdir()
        (source / "compiler.py").write_text(f"# immutable candidate{index}\n")
        frozen[str(path)] = str(source)
    baseline = root / "frozen-baseline"
    baseline.mkdir()
    (baseline / "compiler.py").write_text("# current dependency environment baseline\n")
    frozen_path = root / "frozen.json"
    m._write(frozen_path, dict(schema=m.FROZEN, subjects=frozen, baseline=str(baseline)))
    runs = root / "runs.json"
    m._write(runs, dict(inputs=m._pin(manifest), runs=rows))
    from merlin_experiments.corpus import public_scalability

    from merlin.targetgen import target_experiment

    monkeypatch.setattr(
        target_experiment, "load_target_experiment", lambda path: SimpleNamespace(descriptor=Path(path))
    )
    monkeypatch.setattr(
        public_scalability,
        "selected",
        lambda te: (te.descriptor.parent, {"cases": selected[te.descriptor.parent.name]}),
    )
    args = (runs, panel, manifest, frozen_path, tmp_path / "evaluation")
    return m, args


def _responses(m, plan, *, cycles_by_subject=None):
    for job in plan["jobs"]:
        cycles = (cycles_by_subject or {}).get(job["subject"], 100 if job["subject"].startswith("baseline") else 50)
        rows = []
        for name in job["cases"]:
            perf = dict(
                status="PROVISIONAL",
                estimated_whole_soc_cycles=cycles,
                capsule_sha256=name,
                selection_sha256="same-selection",
                model_sha256="same-model",
                comparison_domain_sha256="same-domain",
                scope_sha256="same-complete-scope",
            )
            rows.append(
                dict(
                    capsule=name,
                    status="pass",
                    numeric={"status": "pass"},
                    tiers={
                        "L2": {
                            "status": "pass",
                            "engine": "spike",
                            "fidelity": "functional_model",
                            "derived_from_rtl": False,
                        }
                    },
                    performance=perf,
                )
            )
        # A requested subset is deliberately not a full-suite pass. Cell evidence remains useful.
        m._write(
            job["result"],
            dict(
                schema="merlin.phase1.public_scalability.v1",
                all_pass=False,
                candidate_sha256=job["source"]["sha256"],
                suite_sha256=job["suite_sha256"],
                requested_cases=job["cases"],
                cases=rows,
            ),
        )


def test_plan_twelve_arms_and_one_common_baseline_without_execution(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch)
    result = m.plan(*args)
    assert result["executed"] is False
    assert len(result["jobs"]) == 26
    assert len([j for j in result["jobs"] if j["subject"] == "baseline"]) == 2
    for subject in result["subjects"]:
        jobs = [j for j in result["jobs"] if j["subject"] == subject["id"]]
        assert [len(j["cases"]) for j in jobs] == [3, 6]
        assert {j["split"] for j in jobs} == {"train", "heldout"}
        for job in jobs:
            assert "evaluate" in job["argv"] and "launch" not in job["argv"]
            assert job["argv"][-2:] == ["--tiers", "L0,L1,L2"]
            assert not Path(job["result"]).exists()
            config = m._read(job["config"]["path"])
            assert Path(config["descriptor"]).parent.name == job["split"]
            assert config["reference_guidance"] == "off" and config["reference_map"] is None


def test_small_pair_retains_every_selected_case_without_execution(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch, count=2, training_count=1, heldout_count=3)
    result = m.plan(*args)
    assert result["executed"] is False
    assert len(result["jobs"]) == 6
    _responses(m, result)
    cells = m.collect(args[-1] / "plan.json")["cells"]
    assert len(cells) == 12  # common seed and both authors, each on all four shapes
    assert all(cell["state"] == "pass" for cell in cells)
    assert all(cell["physical"]["cycles"] is None for cell in cells)


@pytest.mark.parametrize("training_count,heldout_count", [(0, 3), (1, 0)])
def test_panel_cannot_drop_an_entire_split(tmp_path, monkeypatch, training_count, heldout_count):
    m, args = _inputs(tmp_path, monkeypatch, count=2, training_count=training_count, heldout_count=heldout_count)
    with pytest.raises(ValueError, match="nonempty training and heldout"):
        m.plan(*args)
    assert not args[-1].exists()


def test_evaluation_cannot_relabel_an_author_training_case_as_hidden(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch, count=2)
    panel = m._read(args[1])
    panel["selection"]["heldout_case_ids"].append(panel["selection"]["training_case_ids"].pop())
    m._write(args[1], panel)
    # The manifest split check rejects this before any evaluator is prepared.
    with pytest.raises(ValueError, match="panel split differs"):
        m.plan(*args)
    assert not args[-1].exists()


def test_evaluation_training_selection_must_match_the_author(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch, count=2)
    panel = m._read(args[1])
    panel["selection"]["training_case_ids"].pop()
    m._write(args[1], panel)
    with pytest.raises(ValueError, match="training panel differs from the author"):
        m.plan(*args)
    assert not args[-1].exists()


def test_plan_requires_all_final_frozen_paths_and_same_model(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch, count=2)
    doc = m._read(args[3])
    key, value = doc["subjects"].popitem()
    m._write(args[3], doc)
    with pytest.raises(ValueError, match="every run"):
        m.plan(*args)
    doc["subjects"][key] = value
    m._write(args[3], doc)
    runs = m._read(args[0])
    row = runs["runs"][1]
    config = m._read(row["config"])
    env_path = Path(config["private"]) / "evaluator-environment.json"
    env = m._read(env_path)
    other = tmp_path / "other-model.json"
    m._write(other, {"different": True})
    env["MERLIN_JOINT_PERFORMANCE_SELECTION"] = json.dumps(m._pin(other))
    m._write(env_path, env)
    with pytest.raises(ValueError, match="same selected execution environment"):
        m.plan(*args)
    assert not args[-1].exists()


def test_collect_complete_panel_pairs_and_physical_unknown(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch)
    plan = m.plan(*args)
    _responses(m, plan)
    result = m.collect(args[-1] / "plan.json")
    assert len(result["cells"]) == 117
    candidate = [c for c in result["cells"] if c["subject"] != "baseline"]
    assert len(candidate) == 108
    assert all(c["state"] == "pass" and c["model_comparison"]["candidate_to_baseline_ratio"] == 0.5 for c in candidate)
    assert all(c["physical"]["cycles"] is None for c in result["cells"])
    assert not result["physical_timing_qualified"]
    assert all(
        s["model_candidate_to_baseline_geomean"] == 0.5 for s in result["summaries"] if s["subject"] != "baseline"
    )


def test_collect_missing_timeout_and_mismatched_domain_are_not_speedups(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch, count=2)
    plan = m.plan(*args)
    _responses(m, plan)
    jobs = [j for j in plan["jobs"] if j["subject"] == "run-00"]
    Path(jobs[0]["result"]).unlink()
    response = m._read(jobs[1]["result"])
    response["cases"][0].update(status="fail", numeric={"status": "fail"}, failure={"category": "timeout"})
    response["cases"][1]["performance"]["comparison_domain_sha256"] = "other-domain"
    response["cases"][2]["performance"]["estimated_whole_soc_cycles"] = 0
    m._write(jobs[1]["result"], response)
    result = m.collect(args[-1] / "plan.json")
    cells = [c for c in result["cells"] if c["subject"] == "run-00"]
    assert [c["state"] for c in cells[:3]] == ["missing"] * 3
    assert cells[3]["state"] == "timeout"
    assert all(cells[i]["model_comparison"]["candidate_to_baseline_ratio"] is None for i in range(6))
    assert all(
        s["aggregate_status"] == "UNKNOWN" and s["model_candidate_to_baseline_geomean"] is None
        for s in result["summaries"]
        if s["subject"] == "run-00"
    )


def test_collect_rejects_stale_result_and_changed_frozen_source(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch, count=1)
    plan = m.plan(*args)
    _responses(m, plan)
    job = next(j for j in plan["jobs"] if j["subject"] == "run-00")
    response = m._read(job["result"])
    response["candidate_sha256"] = "old-source"
    m._write(job["result"], response)
    result = m.collect(args[-1] / "plan.json")
    assert all(
        c["state"] == "invalid_result" for c in result["cells"] if c["subject"] == "run-00" and c["split"] == "train"
    )
    (Path(job["source"]["path"]) / "compiler.py").write_text("# edited after freeze\n")
    with pytest.raises(ValueError, match="frozen input changed"):
        m.collect(args[-1] / "plan.json")


def _progress_fixture(m, args, plan, rid, *, fail=False, stale=False, other_domain=False):
    """Retain an actual owner call receipt around a tiny synthetic result fixture."""
    import shutil

    from merlin.common import invocation_record as I

    config = m._read(m._read(args[0])["runs"][0]["config"])
    private, channel = Path(config["private"]), Path(config["workspace"]) / ".qa_channel"
    channel.mkdir(exist_ok=True)
    job = private / "checks" / rid
    job.mkdir(parents=True)
    source = job / "submission"
    candidate = next(j for j in plan["jobs"] if j["subject"] == "run-00" and j["split"] == "train")
    shutil.copytree(candidate["source"]["path"], source)
    source_sha = m._tree(source)["sha256"]
    result = m._read(candidate["result"])
    result["candidate_sha256"] = source_sha
    if fail:
        result["cases"][0].update(status="fail", numeric={"status": "fail"})
    if other_domain:
        for row in result["cases"]:
            row["performance"]["comparison_domain_sha256"] = "another-model-domain"
    qualification = dict(
        candidate_sha256=source_sha,
        suite_sha256=result["suite_sha256"],
        observed={row["capsule"]: row for row in result["cases"]},
    )
    qualification_path = job / "result.scalability" / "qualification.json"
    qualification_path.parent.mkdir()
    with I.observe_call(
        job / "issuers",
        stage="public_spike_scalability",
        function=_progress_fixture,
        arguments={"candidate_sha256": source_sha, "suite_sha256": result["suite_sha256"]},
        outputs=(qualification_path,),
    ) as observed:
        m._write(qualification_path, qualification)
        observed.returned()
    m._write(
        qualification_path.parent / "issued.json", dict(issuer=m._pin(observed.path), result=m._pin(qualification_path))
    )
    m._write(job / "result.json", result)
    (job / "worker.log").write_text("ordinary evaluator fixture\n")
    request = dict(submission_sha256="broker-source-" + rid, capsules=",".join(candidate["cases"]))
    m._write(channel / ("req_" + rid + ".json"), request)
    m._write(
        channel / ("resp_" + rid + ".json"),
        dict(
            result,
            selfcheck_request_id=rid,
            submission_sha256=request["submission_sha256"],
            requested_submission_sha256="stale" if stale else request["submission_sha256"],
            graded_at_unix_ns=1,
        ),
    )  # An author-writable timestamp cannot set the retrospective clock.
    (channel / ("done_" + rid)).write_text("ok\n")
    return config, job


def _baseline_collection(m, args, plan):
    _responses(m, plan)
    path = args[-1] / "collection.json"
    m._write(path, m.collect(args[-1] / "plan.json"))
    return path


def test_progress_verified_source_numeric_and_model_receipts_drive_improvement(tmp_path, monkeypatch):
    import time

    m, args = _inputs(tmp_path, monkeypatch, count=1)
    plan = m.plan(*args)
    baseline = _baseline_collection(m, args, plan)
    config, job = _progress_fixture(m, args, plan, "checked")
    _progress_fixture(m, args, plan, "stale", stale=True)
    _progress_fixture(m, args, plan, "wrong", fail=True)
    private = Path(config["private"])
    m._write(private / "status.json", {"status": "FINISHED"})
    (private / "author-1.jsonl").write_text(
        json.dumps({"type": "item.completed", "text": "PRIVATE_REASONING"})
        + "\n"
        + json.dumps(
            {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 30}}
        )
        + "\n"
    )
    audit = tmp_path / "launch-audit.json"
    m._write(
        audit,
        dict(
            schema="private.kernel_abstractions.launch_audit.v1",
            runs=[dict(config=m._read(args[0])["runs"][0]["config"], started_unix=time.time() - 30)],
        ),
    )
    result = m.progress(args[0], baseline_collection=baseline, launch_audit=audit)
    report = result["reports"][0]
    requests = {r["request_id"]: r for r in report["requests"]}
    assert requests["checked"]["evaluation_wall_s"] >= 0
    assert requests["checked"]["source"] == m._tree(job / "submission")
    assert len(requests["checked"]["accepted_numeric_cases"]) == 3
    assert requests["stale"]["status"] == "unverified_result"
    assert requests["stale"]["accepted_numeric_cases"] == []
    assert len(requests["wrong"]["accepted_numeric_cases"]) == 2
    first = report["trajectories"][0]
    assert first["first_correct_improvement"]["request_id"] == "checked"
    assert first["first_correct_improvement"]["candidate_to_baseline_ratio"] == 0.5
    assert 20 < first["time_to_first_correct_improvement_s"] < 40
    assert first["physical_confirmation"] == "UNKNOWN"
    assert report["usage"]["totals_by_format"]["turn.completed"]["cached_input_tokens"] == 30
    assert "PRIVATE_REASONING" not in json.dumps(result)


def test_progress_missing_usage_start_and_model_domain_remain_unknown(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch, count=1)
    plan = m.plan(*args)
    baseline = _baseline_collection(m, args, plan)
    _progress_fixture(m, args, plan, "correct", other_domain=True)
    report = m.progress(args[0], baseline_collection=baseline)["reports"][0]
    assert report["started_unix"] is None and report["start_authority"] == "UNKNOWN"
    assert report["usage"]["status"] == "UNKNOWN" and report["usage"]["totals_by_format"] is None
    assert all(
        t["first_correct_improvement"] is None and t["timing_status"] == "UNKNOWN" for t in report["trajectories"]
    )
    # Even a valid correct implementation alone has no paired timing authority.
    report = m.progress(args[0])["reports"][0]
    assert all(t["accepted_model_observations"] == [] for t in report["trajectories"])


def test_progress_failed_worker_without_result_preserves_actual_snapshot(tmp_path, monkeypatch):
    import shutil

    m, args = _inputs(tmp_path, monkeypatch, count=1)
    plan = m.plan(*args)
    config = m._read(m._read(args[0])["runs"][0]["config"])
    private, channel = Path(config["private"]), Path(config["workspace"]) / ".qa_channel"
    channel.mkdir()
    job = private / "checks" / "failed"
    job.mkdir(parents=True)
    shutil.copytree(plan["jobs"][2]["source"]["path"], job / "submission")
    (job / "worker.log").write_text("error: selected ordinary compiler refused\n")
    m._write(channel / "req_failed.json", {"submission_sha256": "source", "capsules": "train0"})
    m._write(channel / "resp_failed.json", {"status": "offline_request_failed", "error": "no result"})
    request = m.progress(args[0])["reports"][0]["requests"][0]
    assert request["status"] == "offline_request_failed"
    assert request["source"] == m._tree(job / "submission")
    assert request["failure"] == "no result" and request["accepted_numeric_cases"] == []
    assert request["files"]["worker_log"] == m._pin(job / "worker.log")


def test_rules_and_settings_are_separately_frozen_and_reopened(tmp_path, monkeypatch):
    import time

    m, args = _inputs(tmp_path, monkeypatch, count=1)
    rules, settings = tmp_path / "RULES.md", tmp_path / "settings.json"
    rules.write_text("A source-pinned rule with explicit shape and refusal scope.\n")
    m._write(settings, {"tile": 16})
    frozen = m._read(args[3])
    key = next(iter(frozen["subjects"]))
    frozen["subjects"][key] = dict(
        path=frozen["subjects"][key], rules=m._pin(rules), settings=m._pin(settings), frozen_unix=time.time()
    )
    m._write(args[3], frozen)
    plan = m.plan(*args)
    subject = next(s for s in plan["subjects"] if s["id"] == "run-00")
    assert subject["freeze"]["requirements_complete"]
    assert subject["freeze"]["rules"] == m._pin(rules)
    assert plan["subjects"][0]["freeze"]["status"] == "UNKNOWN"  # Plain path never implied RULES custody.
    rules.write_text("Changed after compiler/RULES freeze.\n")
    with pytest.raises(ValueError, match="frozen input changed"):
        m.collect(args[-1] / "plan.json")


def _per_arm_baselines(m, args):
    frozen, runs = m._read(args[3]), m._read(args[0])
    common = frozen.pop("baseline")
    frozen["baselines"] = {row["config"]: common for row in runs["runs"]}
    for index, row in enumerate(runs["runs"]):
        config = m._read(row["config"])
        config["full_compiler"] = {"manifest_sha256": str(index) * 64}
        m._write(row["config"], config)
        row["config_sha256"] = m._pin(row["config"])["sha256"]
    m._write(args[0], runs)
    m._write(args[3], frozen)


def test_per_arm_baselines_separate_tool_migration_from_author_improvement(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch, count=2)
    _per_arm_baselines(m, args)
    plan = m.plan(*args)
    assert plan["baseline_mode"] == "per_arm" and len(plan["jobs"]) == 8
    assert set(plan["baseline_bindings"].values()) == {"baseline-00", "baseline-01"}
    assert len(plan["full_compiler_selections"]) == 2
    assert not plan["migration"]["included_in_agent_improvement"]
    _responses(m, plan, cycles_by_subject={"baseline-00": 200, "baseline-01": 100, "run-00": 160, "run-01": 100})
    collected = m.collect(args[-1] / "plan.json")
    for cell in collected["cells"]:
        if cell["subject"] == "run-00":
            assert cell["baseline_subject"] == "baseline-00"
            assert cell["model_comparison"]["candidate_to_baseline_ratio"] == 0.8
        elif cell["subject"] == "run-01":
            assert cell["baseline_subject"] == "baseline-01"
            assert cell["model_comparison"]["candidate_to_baseline_ratio"] == 1.0
        else:
            assert cell["model_comparison"] is None
    assert not collected["migration"]["included_in_agent_improvement"]


@pytest.mark.parametrize("failure", ["missing", "ambiguous", "model", "contract"])
def test_per_arm_baselines_require_exact_mapping_and_common_instrument(tmp_path, monkeypatch, failure):
    m, args = _inputs(tmp_path, monkeypatch, count=2)
    _per_arm_baselines(m, args)
    frozen, runs = m._read(args[3]), m._read(args[0])
    if failure == "missing":
        frozen["baselines"].popitem()
    elif failure == "ambiguous":
        frozen["baseline"] = next(iter(frozen["baselines"].values()))
    else:
        config = m._read(runs["runs"][1]["config"])
        env_path = Path(config["private"]) / "evaluator-environment.json"
        env = m._read(env_path)
        other = tmp_path / "different-instrument.json"
        m._write(other, {"different": failure})
        key = "MERLIN_JOINT_PERFORMANCE_SELECTION" if failure == "model" else "MERLIN_TARGET_CONTRACT"
        env[key] = json.dumps(m._pin(other)) if failure == "model" else str(other)
        m._write(env_path, env)
    m._write(args[3], frozen)
    with pytest.raises(ValueError, match="baseline mapping|same selected execution environment"):
        m.plan(*args)
    assert not args[-1].exists()


@pytest.mark.parametrize("field", ["model_sha256", "comparison_domain_sha256"])
def test_per_arm_model_comparisons_still_require_same_domain(tmp_path, monkeypatch, field):
    m, args = _inputs(tmp_path, monkeypatch, count=2)
    _per_arm_baselines(m, args)
    plan = m.plan(*args)
    _responses(m, plan)
    job = next(j for j in plan["jobs"] if j["subject"] == "run-00" and j["split"] == "train")
    result = m._read(job["result"])
    for row in result["cases"]:
        row["performance"][field] = "different"
    m._write(job["result"], result)
    result = m.collect(args[-1] / "plan.json")
    rejected = [c for c in result["cells"] if c["subject"] == "run-00" and c["split"] == "train"]
    assert all(c["state"] == "pass" and c["model_comparison"]["candidate_to_baseline_ratio"] is None for c in rejected)


def test_progress_uses_own_initial_baseline_and_rejects_absent_mapping(tmp_path, monkeypatch):
    m, args = _inputs(tmp_path, monkeypatch, count=2)
    _per_arm_baselines(m, args)
    plan = m.plan(*args)
    _responses(m, plan, cycles_by_subject={"baseline-00": 200, "baseline-01": 100})
    baseline = args[-1] / "collection.json"
    m._write(baseline, m.collect(args[-1] / "plan.json"))
    _progress_fixture(m, args, plan, "checked")
    reports = m.progress(args[0], baseline_collection=baseline)["reports"]
    first = reports[0]["trajectories"][0]["accepted_model_observations"][0]
    assert first["candidate_to_baseline_ratio"] == 0.25
    assert reports[0]["baseline_subject"] == "baseline-00"
    assert reports[1]["baseline_subject"] == "baseline-01"
    collection = m._read(baseline)
    collection["baseline_bindings"].popitem()
    m._write(baseline, collection)
    with pytest.raises(ValueError, match="baseline mapping omits"):
        m.progress(args[0], baseline_collection=baseline)


@pytest.mark.parametrize("qualification_state", ["valid", "missing", "wrong_source"])
def test_collect_real_compact_response_requires_issued_numerical_rows(tmp_path, monkeypatch, qualification_state):
    from merlin.common import invocation_record as I

    m, args = _inputs(tmp_path, monkeypatch, count=1)
    plan = m.plan(*args)
    _responses(m, plan)
    for job in plan["jobs"]:
        response = m._read(job["result"])
        directory = Path(job["result"]).parent
        qualified_path = Path(job["result"]).with_suffix(".scalability") / "qualification.json"
        qualified_path.parent.mkdir()
        source = "other-source" if qualification_state == "wrong_source" else response["candidate_sha256"]
        qualification = dict(
            candidate_sha256=source,
            suite_sha256=response["suite_sha256"],
            observed={row["capsule"]: row for row in response["cases"]},
        )
        if qualification_state != "missing":
            with I.observe_call(
                directory / "issuers",
                stage="public_spike_scalability",
                function=test_collect_real_compact_response_requires_issued_numerical_rows,
                arguments={"candidate_sha256": response["candidate_sha256"], "suite_sha256": response["suite_sha256"]},
                outputs=(qualified_path,),
            ) as issued:
                m._write(qualified_path, qualification)
                issued.returned()
            m._write(
                qualified_path.parent / "issued.json", dict(issuer=m._pin(issued.path), result=m._pin(qualified_path))
            )
        # This is the actual public_scalability.public_result projection, not the private row schema.
        response["cases"] = [
            dict(capsule=row["capsule"], status="pass", tiers={"L2": "pass"}, performance=row["performance"])
            for row in response["cases"]
        ]
        m._write(job["result"], response)
    cells = m.collect(args[-1] / "plan.json")["cells"]
    if qualification_state == "valid":
        assert all(c["state"] == "pass" and c["qualification"] for c in cells)
        assert all(
            c["model_comparison"]["candidate_to_baseline_ratio"] == 0.5 for c in cells if c["subject"] == "run-00"
        )
    else:
        assert all(c["state"] == "invalid_result" for c in cells)
        assert all(not (c.get("model_comparison") or {}).get("candidate_to_baseline_ratio") for c in cells)
