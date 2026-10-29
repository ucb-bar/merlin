"""Matched preparation and endpoint isolation, without launching an author/evaluator."""

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


def module(name):
    path = repo_root() / "examples/gemmini/phase1/studies/kernel_abstractions" / (name + ".py")
    spec = importlib.util.spec_from_file_location("study_" + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(tmp_path):
    prep = module("prepare_runs")
    seed, library, bank, framework = (tmp_path / name for name in ("seed", "library", "references", "framework"))
    for directory in (seed, library, bank, framework, tmp_path / "facts"):
        directory.mkdir()
    (seed / "backend.py").write_text("original Fresh6 source\n")
    (seed / "compiler_library").mkdir()
    (seed / "compiler_library/old.py").write_text("old library\n")
    (library / "new.py").write_text("reviewed current library\n")
    (bank / "reference-manifest.json").write_text("{}")
    (bank / "archive.zip").write_bytes(bytes(range(256)))
    shim = framework / "packages/merlin-experiments/src/merlin_experiments/phase1/tools/selfcheck.py"
    shim.parent.mkdir(parents=True)
    original = repo_root() / "packages/merlin-experiments/src/merlin_experiments/phase1/tools/selfcheck.py"
    shim.write_bytes(original.read_bytes())
    skill_module = shim.parent.parent / "skill_staging.py"
    skill_module.write_bytes((original.parent.parent / "skill_staging.py").read_bytes())
    skill = skill_module.parent / "author_skills/pass-library/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: pass-library\ndescription: Reviewed generic example skill.\n---\nUse the source contract.\n"
    )
    source = tmp_path / "source.mlir"
    source.write_text("module {}")
    contract = tmp_path / "target.yaml"
    contract.write_text("target: selected")
    facts = tmp_path / "facts/facts.json"
    facts.write_text("{}")
    workspace = tmp_path / "old-workspace"
    workspace.mkdir()
    (workspace / "RULES.md").write_text("DO NOT INHERIT WINNING RULES")
    base = dict(
        source_seed=str(seed),
        workspace=str(workspace),
        descriptor=str(contract),
        suite=str(tmp_path),
        facts=str(facts),
        python="python3",
        environment={
            "MERLIN_TARGET_CONTRACT": str(contract),
            "MERLIN_REFERENCE_MAP": "old selection",
            "MERLIN_REFERENCE_EXPERIMENTS": "old tasks",
        },
    )
    base_path = tmp_path / "base.json"
    prep.write(base_path, base)
    row = dict(case="training", split="train", source={"path": str(source), "sha256": prep.digest(source)})
    inputs = dict(
        reference_root=str(bank),
        reference_manifest={
            "path": str(bank / "reference-manifest.json"),
            "sha256": prep.digest(bank / "reference-manifest.json"),
        },
        early_slot={"case": "training", "split": "train"},
        cases=[row, {**row, "case": "secret-heldout", "split": "heldout"}],
    )
    inputs_path = tmp_path / "inputs.json"
    prep.write(inputs_path, inputs)
    return prep, base_path, inputs_path, framework, library, bank


def test_prepare_four_matched_fresh_workspaces(tmp_path):
    prep, base, inputs, framework, library, bank = fixture(tmp_path)
    result = prep.prepare(base, inputs, tmp_path / "runs", framework=framework, library=library)
    assert result["author_launched"] is False
    assert [row["cpu"] for row in result["runs"]] == [0, 1, 2, 3]
    for row in result["runs"]:
        config = json.loads(Path(row["config"]).read_bytes())
        workspace = Path(config["workspace"])
        assert config["workers"] == 1 and config["wall_s"] == 7200
        assert config["early_timeout_s"] == 180
        assert list((workspace / "public-cases").iterdir()) == [workspace / "public-cases/training.mlir"]
        assert (workspace / "RULES.md").read_text() == ""
        assert not (workspace / "submission/compiler_library/old.py").exists()
        assert (workspace / "submission/compiler_library/new.py").read_bytes() == (library / "new.py").read_bytes()
        assert (workspace / "baseline/backend.py").read_bytes() == (workspace / "submission/backend.py").read_bytes()
        assert (workspace / "reference-kernels/archive.zip").stat().st_ino == (bank / "archive.zip").stat().st_ino
        assert not any(key.startswith("MERLIN_REFERENCE") for key in config["environment"])
        assert row["config_sha256"] == prep.digest(row["config"])


def continuation_fixture(tmp_path, prep, base, framework):
    from merlin.common.tree_hash import hash_tree

    config = json.loads(base.read_bytes())
    source, previous = Path(config["source_seed"]), Path(config["workspace"])
    (source / "REPORT.md").write_text("prior successful and rejected work\n")
    (source / "docs").mkdir()
    (source / "docs/iteration_notes.md").write_text("the prior session stopped after checking retention\n")
    (previous / "OPTIMIZATION_RULES.json").write_text('{"rules":["keep K completion"]}\n')
    rejected = previous / "ab/rejected"
    rejected.mkdir(parents=True)
    (rejected / "README.md").write_text("rejected: this order was slower\n")
    variant = previous / "variants/rejected"
    variant.mkdir(parents=True)
    (variant / "backend.py").write_text("whole prior implementation\n")
    (variant / "build").mkdir()
    (variant / "build/private.log").write_text("not a source artifact\n")
    run = tmp_path / "previous-run"
    (run / "rounds").mkdir(parents=True)
    (run / "rounds/last.final.txt").write_text("continue from the accepted incumbent\n")
    selection = tmp_path / "continuation.json"
    prep.write(
        selection,
        {
            "from": {"path": str(source), "sha256": hash_tree(source)["sha256"]},
            "run": {
                "run_id": "earlier-run",
                "run_dir": str(run),
                "workspace": str(previous),
                "framework": str(framework),
            },
        },
    )
    return selection


def test_explicit_continuation_keeps_full_seed_and_common_history_with_fresh_clock(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    selected = continuation_fixture(tmp_path, prep, base, framework)
    configs = prep.prepare(
        base,
        inputs,
        tmp_path / "runs",
        framework=framework,
        library=library,
        seed_continuation=selected,
        seconds=37,
        replicas=1,
        treatments=["library_schedule", "cca_library_schedule"],
    )
    values = [json.loads(Path(row["config"]).read_bytes()) for row in configs["runs"]]
    assert len({row["initial_submission_sha256"] for row in values}) == 1
    evidence = []
    for row in values:
        workspace = Path(row["workspace"])
        for name in ("baseline", "submission"):
            assert (workspace / name / "compiler_library/old.py").read_text() == "old library\n"
            assert (workspace / name / "compiler_library/new.py").read_text() == "reviewed current library\n"
            assert (workspace / name / "backend.py").read_text() == "original Fresh6 source\n"
        assert (workspace / "OPTIMIZATION_RULES.json").read_text() == '{"rules":["keep K completion"]}\n'
        assert "slower" in (workspace / "SEED_EXPERIMENTS/ab/rejected/README.md").read_text()
        assert "prior successful" in (workspace / "SEED_NOTES/REPORT.md").read_text()
        assert "accepted incumbent" in (workspace / "PREVIOUS_SESSION.md").read_text()
        assert (workspace / "SEED_VARIANTS/rejected/backend.py").read_text() == "whole prior implementation\n"
        assert not (workspace / "SEED_VARIANTS/rejected/build").exists()
        assert (workspace / "SEED_PERFORMANCE/baseline.json").is_file()
        assert not (workspace / "RUN_PRIVATE").exists()
        assert (Path(row["private"]) / "seed_baseline.json").is_file()
        assert "No previous author conversation or experiment history" not in (workspace / "TASK.md").read_text()
        assert row["wall_s"] == 37 and row["cpu_quota_percent"] == 200
        assert row["library_guidance"] and row["schedule_guidance"] and not row["sdk_readonly"]
        record = row["seed_continuation"]
        assert record["selection"]["sha256"] == prep.digest(selected)
        assert record["effective_source_sha256"] == row["initial_submission_sha256"]
        assert "none" in record["certificate_reuse"]
        assert record["carry"] == dict(certificates=False, results=False, rules=True, handoff=True,
                                      variants=True, performance=True)
        evidence.append(record["evidence"]["pin"]["sha256"])
    assert evidence[0] == evidence[1]
    assert values[0]["reference_guidance"] == "off" and values[1]["reference_guidance"] == "on"


def test_continuation_source_hash_mismatch_refuses_before_output(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    selected = continuation_fixture(tmp_path, prep, base, framework)
    source = Path(json.loads(selected.read_bytes())["from"]["path"])
    (source / "backend.py").write_text("changed after selection\n")
    with pytest.raises(ValueError, match="source digest differs"):
        prep.prepare(base, inputs, tmp_path / "runs", framework=framework, library=library, seed_continuation=selected)
    assert not (tmp_path / "runs").exists()


def study_fixture(tmp_path):
    import shutil

    from merlin_experiments.phase1 import full_compiler_workspace as F

    prep, base, inputs, framework, library, bank = fixture(tmp_path)
    repo = tmp_path / "compiler"
    for relative, text in {
        "merlin/__init__.py": "",
        "merlin/compile/__init__.py": "",
        "merlin/compile/old.py": "value=1\n",
    }.items():
        path = repo / "src" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    inherited = F.prepare(repo, tmp_path / "inherited", review_id="Fresh6")
    source = Path(json.loads(base.read_bytes())["source_seed"])
    shutil.rmtree(source / "compiler_library")
    shutil.copytree(inherited.root, source / "compiler_library")
    (repo / "src/merlin/compile/epochs.py").write_text("new_checked_capability=True\n")
    (repo / "src/merlin/compile/old.py").write_text("value=2\n")
    current = F.prepare(repo, tmp_path / "current", review_id="current")
    fields = ("root", "manifest_path", "manifest_sha256")
    settings = json.loads(base.read_bytes())
    (library / "shared-library.json").write_text('{"capability":"current"}\n')
    (Path(settings["workspace"]) / "shared-library.json").write_text('{"capability":"inherited"}\n')
    settings["study_tools"] = {
        key: {name: owner.record()[name] for name in fields}
        for key, owner in (
            ("inherited", inherited),
            ("current", current),
            ("library", current),
            ("scheduling", current),
        )
    }
    prep.write(base, settings)
    continuation = continuation_fixture(tmp_path, prep, base, framework)
    return prep, base, inputs, framework, library, bank, continuation


@pytest.mark.parametrize("study", ["cca", "library", "scheduling"])
def test_named_study_grants_implementations_without_restricting_control_edits(tmp_path, study):
    prep, base, inputs, framework, library, _, continuation = study_fixture(tmp_path)
    result = prep.prepare(
        base,
        inputs,
        tmp_path / "runs",
        framework=framework,
        library=library,
        seed_continuation=continuation,
        study=study,
        seconds=1800,
        replicas=1,
    )
    control, treatment = [json.loads(Path(row["config"]).read_bytes()) for row in result["runs"]]
    for config in (control, treatment):
        workspace = Path(config["workspace"])
        assert not config["sdk_readonly"] and config["cpu_quota_percent"] == 200
        assert config["task_sha256"] == prep.digest(workspace / "TASK.md")
        assert not (workspace / "submission/compiler_library/new.py").exists()  # no silent SDK fill
        assert (workspace / "submission/backend.py").read_text() == "original Fresh6 source\n"
        assert json.loads((workspace / "shared-library.json").read_bytes())["capability"] == (
            "current" if study == "cca" else "inherited"
        )
    old = Path(control["workspace"]) / "submission/compiler_library/merlin/compile"
    new = Path(treatment["workspace"]) / "submission/compiler_library/merlin/compile"
    assert (new / "epochs.py").is_file()
    assert (old / "epochs.py").exists() == (study == "cca")
    assert (old / "old.py").read_text() == ("value=2\n" if study == "cca" else "value=1\n")
    assert (new / "old.py").read_text() == "value=2\n"
    assert control["reference_guidance"] == "off"
    assert treatment["reference_guidance"] == ("on" if study == "cca" else "off")
    assert (control["initial_submission_sha256"] == treatment["initial_submission_sha256"]) == (study == "cca")


def test_library_migration_never_overwrites_authored_changes(tmp_path):
    from merlin_experiments.phase1 import full_compiler_workspace as F

    prep, base, _, _, _, _, _ = study_fixture(tmp_path)
    settings = json.loads(base.read_bytes())
    source = Path(settings["source_seed"])
    (source / "compiler_library/merlin/compile/old.py").chmod(0o644)
    (source / "compiler_library/merlin/compile/old.py").write_text("authored_customization=True\n")
    with pytest.raises(ValueError, match="authored seed change"):
        prep.migrate_library(
            source, F.select(**settings["study_tools"]["inherited"]), F.select(**settings["study_tools"]["current"])
        )
    assert "authored_customization" in (source / "compiler_library/merlin/compile/old.py").read_text()


def test_pinned_cca_report_is_staged_only_to_the_enabled_arm(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    report = tmp_path / "report.md"
    report.write_text("reference mechanism evidence; performance remains unmeasured\n")
    settings = json.loads(base.read_bytes())
    settings["cca_guidance"] = {"path": str(report), "sha256": prep.digest(report)}
    prep.write(base, settings)
    result = prep.prepare(base, inputs, tmp_path / "runs", framework=framework, library=library, replicas=1)
    for row in result["runs"]:
        config = json.loads(Path(row["config"]).read_bytes())
        workspace = Path(config["workspace"])
        assert (workspace / "CCA_REFERENCE.md").exists() == (config["reference_guidance"] == "on")
    report.write_text("changed after pin\n")
    with pytest.raises(ValueError, match="guidance pin changed"):
        prep.prepare(base, inputs, tmp_path / "refused", framework=framework, library=library)


def capture_fixture(tmp_path, prep):
    source = tmp_path / "capture-selection"
    source.mkdir()
    entry = {}
    for key in ("capture", "facts", "contract", "observer"):
        path = source / (key + ".txt")
        path.write_text(key)
        entry[key] = dict(path="cca-evidence/" + path.name, sha256=prep.digest(path))
    (source / "operator-only.txt").write_text("not selected for the author")
    manifest = source / "manifest.json"
    prep.write(manifest, dict(schema="merlin.reference_capture_selection.v1", entries=[entry]))
    return dict(path=str(manifest), sha256=prep.digest(manifest))


def test_captured_lifting_is_separate_from_identical_raw_banks(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    pin = capture_fixture(tmp_path, prep)
    settings = json.loads(base.read_bytes())
    settings["cca_lifting"] = pin
    prep.write(base, settings)
    result = prep.prepare(base, inputs, tmp_path / "runs", framework=framework, library=library, replicas=1)
    for row in result["runs"]:
        config = json.loads(Path(row["config"]).read_bytes())
        workspace = Path(config["workspace"])
        enabled = config["reference_guidance"] == "on"
        assert (workspace / "cca-evidence").exists() == enabled
        assert ("MERLIN_REFERENCE_CAPTURE_SELECTION" in config["environment"]) == enabled
        if enabled:
            assert config["cca_lifting"]["sha256"] == pin["sha256"]
            assert not (workspace / "cca-evidence/operator-only.txt").exists()
            assert (workspace / "cca-evidence/capture.txt").read_text() == "capture"
            assert not (workspace / "cca-evidence/manifest.json").stat().st_mode & 0o222


@pytest.mark.parametrize("mutation", ["bytes", "escape"])
def test_capture_grant_rejects_changed_or_escaping_members(tmp_path, mutation):
    prep = module("prepare_runs")
    pin = capture_fixture(tmp_path, prep)
    manifest = Path(pin["path"])
    if mutation == "bytes":
        (manifest.parent / "capture.txt").write_text("changed trace")
    else:
        value = json.loads(manifest.read_bytes())
        value["entries"][0]["capture"]["path"] = "cca-evidence/../outside.txt"
        prep.write(manifest, value)
        pin["sha256"] = prep.digest(manifest)
    with pytest.raises(ValueError, match="pin changed|reviewed evidence grant"):
        prep.stage_capture(pin, tmp_path / "workspace")


def test_pinned_source_change_refuses_before_preparation(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    doc = json.loads(inputs.read_bytes())
    Path(doc["cases"][0]["source"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="source pin changed"):
        prep.prepare(base, inputs, tmp_path / "runs", framework=framework, library=library)
    assert not (tmp_path / "runs").exists()


def test_off_rejects_cca_request_without_calling_endpoint(tmp_path):
    run = module("run")
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    run.CH, run.WS = channel, tmp_path
    run.C = {"reference_guidance": "off"}
    run.shim = SimpleNamespace(_submission_digest=lambda _: "actual-source")
    run.request_label = lambda _: "submission"
    run.pinned_source = lambda *_: nullcontext(tmp_path)
    run.selection_path = None
    request = channel / "req_trial.json"
    request.write_text(json.dumps({"action": "reference-analysis", "submission_sha256": "actual-source"}))
    run.serve(request)
    response = json.loads((channel / "resp_trial.json").read_bytes())
    assert "disabled" in response["error"]
    assert response["official"] is False
    assert (channel / "done_trial").read_text() == "ok\n"


def test_raw_bank_links_refuse_symlink(tmp_path):
    prep = module("prepare_runs")
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="no symlinks"):
        prep.link_bank(bank, tmp_path / "linked")


def test_budget_stop_is_distinct_from_provider_failure():
    run = module("run")
    assert run.completion_status(failed=False, budget_stop=True) == "BUDGET_COMPLETE"
    assert run.completion_status(failed=True, budget_stop=True) == "FAILED"
    assert run.completion_status(failed=False, budget_stop=False) == "FINISHED"


def test_only_enabled_environment_selects_reference_map(tmp_path):
    prep = module("prepare_runs")
    original = {"MERLIN_REFERENCE_MAP": "stale", "MERLIN_REFERENCE_EXPERIMENTS": "prior"}
    selected = {"path": "/selected-map.json", "sha256": "a" * 64}
    for mode in ("off", "on"):
        env = prep.environment(
            original, framework=tmp_path, private=tmp_path, cpu=2, guidance=mode, reference_map=selected
        )
        assert "MERLIN_REFERENCE_EXPERIMENTS" not in env
        assert ("MERLIN_REFERENCE_MAP" in env) == (mode == "on")
        assert env["MERLIN_GRADE_WORKERS"] == "1"
        assert env["MERLIN_COMPUTE_CPUS"] == "2"


def test_agent_only_control_matches_source_with_shared_cpu_budget(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    result = prep.prepare(
        base, inputs, tmp_path / "runs", framework=framework, library=library, agent_only_control=True, replicas=2
    )
    assert len(result["runs"]) == 6
    configs = [json.loads(Path(row["config"]).read_bytes()) for row in result["runs"]]
    assert {row["tooling"] for row in configs} == {"agent_only", "library_schedule", "cca_library_schedule"}
    assert len({row["initial_submission_sha256"] for row in configs}) == 1
    for row in configs:
        workspace = Path(row["workspace"])
        assert row["cpus"] == [0, 1, 2, 3] and row["cpu_quota_percent"] == 66
        assert row["environment"]["MERLIN_COMPUTE_CPUS"] == "0,1,2,3"
        assert (workspace / ".agents/skills").exists() == (row["tooling"] != "agent_only")
        assert row["workers"] == 1
        if row["tooling"] == "agent_only":
            assert row["reference_guidance"] == "off" and row["reference_map"] is None
            assert "agent-only inherited-F6 dependency baseline" in (workspace / "TASK.md").read_text()


def test_agent_only_mounts_inherited_sdk_readonly(tmp_path):
    run = module("run")
    run.WS, run.F = tmp_path, repo_root()
    run.C = {"python": "/usr/bin/python3", "tooling": "agent_only"}
    run.base_argv = lambda *args, **kwargs: ["bwrap"]
    argv, _ = run.sandbox(["/usr/bin/true"], tmp_path / "home")
    library = str(tmp_path / "submission/compiler_library")
    assert any(argv[position : position + 3] == ["--ro-bind", library, library] for position in range(len(argv)))


def test_case_subset_and_independent_assistance_gates(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    document = json.loads(inputs.read_bytes())
    document["cases"].append({**document["cases"][0], "case": "unselected-large"})
    document["early_slot"] = {"case": "unselected-large", "split": "train"}
    prep.write(inputs, document)
    result = prep.prepare(
        base,
        inputs,
        tmp_path / "runs",
        framework=framework,
        library=library,
        cases=["training"],
        cca=False,
        library_guidance=False,
        schedule_guidance=True,
    )
    for row in result["runs"]:
        config = json.loads(Path(row["config"]).read_bytes())
        assert config["training_cases"] == ["training"] and config["early_slot"] is None
        assert config["reference_guidance"] == "off" and config["reference_map"] is None
        assert config["library_guidance"] is False and config["schedule_guidance"] is True
        assert not (Path(config["workspace"]) / ".agents/skills/pass-library").exists()


def test_all_six_treatments_have_independent_effective_flags_and_equal_bytes(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    result = prep.prepare(
        base, inputs, tmp_path / "runs", framework=framework, library=library, treatments=list(prep.PRESETS), replicas=2
    )
    assert len(result["runs"]) == 12
    configs = [json.loads(Path(row["config"]).read_bytes()) for row in result["runs"]]
    assert len({row["initial_submission_sha256"] for row in configs}) == 1
    for row in configs:
        assert row["cpu_quota_percent"] == 33 and row["cpus"] == [0, 1, 2, 3]
        cca, library, schedule = prep.PRESETS[row["tooling"]]
        assert (row["reference_guidance"] == "on", row["library_guidance"], row["schedule_guidance"]) == (
            cca,
            library,
            schedule,
        )
        assert row["sdk_readonly"] == (not (library or schedule))


def test_actual_quota_check_refuses_unbounded_or_wrong_limit(tmp_path):
    run = module("run")
    group = tmp_path / "study.service"
    group.mkdir()
    proc = tmp_path / "cgroup"
    proc.write_text("0::/study.service\n")
    for value in ("max 100000", "66000 100000"):
        (group / "cpu.max").write_text(value)
        with pytest.raises(ValueError, match="quota differs"):
            run.verify_quota(33, proc=proc, root=tmp_path)
    (group / "cpu.max").write_text("33000 100000")
    run.verify_quota(33, proc=proc, root=tmp_path)


def test_launcher_reuses_one_reservation_and_independent_lifetime_bound_services(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    root = tmp_path / "runs"
    prep.prepare(base, inputs, root, framework=framework, library=library, treatments=list(prep.PRESETS), replicas=2)
    launcher = module("launch")
    selected = launcher.plan(root / "runs.json", "merlinphase1studytest", user="fixture")
    assert selected["quota_percent"] == 33 and selected["cpus"] == [0, 1, 2, 3]
    outer = selected["reservation_argv"]
    assert outer[1].endswith("/launch/resource_reservation.py")
    assert outer[outer.index("--cpus") + 1] == "0-3"
    assert len(selected["children"]) == 12
    assert "-" not in selected["slice"]  # root-level slice; no implicit ancestor hierarchy
    for child in selected["children"]:
        argv = child["argv"]
        assert "--property=CPUQuota=33%" in argv and "--property=AllowedCPUs=0-3" in argv
        assert "--property=BindsTo=merlinphase1studytest.service" in argv
        assert "--property=After=merlinphase1studytest.service" in argv
        assert "--property=KillMode=control-group" in argv
    assert not (root / "cpu-reservation.json").exists()  # planning reserves or launches nothing


def test_three_parallel_pairs_share_one_reservation_with_disjoint_pools(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    pools = [[4, 5, 6, 7], [48, 49, 50, 51], [52, 53, 54, 55]]
    rows = []
    for index, cpus in enumerate(pools):
        settings = json.loads(base.read_bytes())
        settings["cpus"] = cpus
        prep.write(base, settings)
        runs = prep.prepare(base, inputs, tmp_path / f"pair-{index}", framework=framework,
                            library=library, replicas=1,
                            treatments=["library_schedule", "cca_library_schedule"])
        for row in runs["runs"]:
            c = json.loads(Path(row["config"]).read_bytes())
            assert c["cpus"] == cpus and c["cpu_quota_percent"] == 200
            assert c["environment"]["MERLIN_COMPUTE_CPUS"] == ",".join(map(str, cpus))
        rows.extend(runs["runs"])
    path = tmp_path / "combined/runs.json"
    prep.write(path, dict(runs=rows, author_launched=False))
    selected = module("launch").plan(path, "merlinphase1paralleltest", user="fixture")
    assert selected["cpus"] == [cpu for pool in pools for cpu in pool]
    assert selected["quota_percent"] == 200 and len(selected["children"]) == 6
    outer = selected["reservation_argv"]
    assert outer.count("--cpus") == 1
    assert outer[outer.index("--cpus") + 1] == "4,5,6,7,48,49,50,51,52,53,54,55"
    for index, child in enumerate(selected["children"]):
        pool = pools[index // 2]
        assert f"--property=AllowedCPUs={pool[0]}-{pool[-1]}" in child["argv"]
        assert "--property=CPUQuota=200%" in child["argv"]
        assert "--property=BindsTo=merlinphase1paralleltest.service" in child["argv"]
    assert len({json.loads(Path(row["config"]).read_bytes())["workspace"] for row in rows}) == 6
    assert not (path.parent / "cpu-reservation.json").exists()


def test_parallel_launcher_refuses_partially_overlapping_pools(tmp_path):
    prep, base, inputs, framework, library, _ = fixture(tmp_path)
    rows = []
    for index, cpus in enumerate(([4, 5, 6, 7], [6, 7, 8, 9])):
        settings = json.loads(base.read_bytes())
        settings["cpus"] = cpus
        prep.write(base, settings)
        rows.extend(prep.prepare(base, inputs, tmp_path / f"pair-{index}", framework=framework,
                                 library=library, replicas=1,
                                 treatments=["library_schedule", "cca_library_schedule"])["runs"])
    path = tmp_path / "combined/runs.json"
    prep.write(path, dict(runs=rows))
    with pytest.raises(ValueError, match="must be disjoint"):
        module("launch").plan(path, "merlinphase1overlaptest", user="fixture")
