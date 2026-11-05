"""A real spec renderer and prepared-plan guards; no author/evaluator launches."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

SPEC = importlib.util.spec_from_file_location(
    "fresh10_study_test", Path(__file__).with_name("fresh10_tooling_study.py")
)
S = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(S)


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


@pytest.fixture
def fixture(tmp_path):
    policy = dict(
        version=1,
        duration_s=43200,
        drain_s=3600,
        final_model_s=7200,
        gsim=dict(max_jobs=15, worker_budget_s=864000),
        gsim_correctness=dict(capsules=["core", "calibration"], max_jobs=15, worker_budget_s=432000),
        native_calibration=dict(path="/pinned/calibration.json", sha256="c" * 64),
        final_only_capsules=[],
        prohibited_instruction_roles=["loop_descriptor"],
        workload_roles=dict(members={"program": dict(role="connected", source_sha256="b" * 64)}),
    )
    baseline_policy = json_file(tmp_path / "baseline-policy.json", dict(policy=policy))
    base_policy = json_file(tmp_path / "base-policy.json", policy)
    manifest = json_file(
        tmp_path / "reference-manifest.json",
        dict(
            schema="operator.allowed_reference_sources.v2",
            origins=[dict(name="reference", files=[dict(destination=f"file{i}") for i in range(23)])],
        ),
    )
    mapping = json_file(
        tmp_path / "REFERENCE_MAP.json",
        dict(entries=[dict(id=f"entry{i}") for i in range(155)], reference_manifest_sha256=S.pin(manifest)["sha256"]),
    )
    performance = json_file(tmp_path / "baseline-performance.json", dict(model="common-runtime"))
    audit = json_file(
        tmp_path / "baseline-audit.json",
        dict(
            source_revision=S.BASELINE_REVISION,
            origin=dict(initial_files=0),
            launch_capsules=S.COUNTS,
            task_scope=dict(held_out_capsules=14),
            pins=dict(
                reference_manifest=S.pin(manifest),
                reference_map=S.pin(mapping),
                joint_policy=S.pin(baseline_policy),
                performance_selection=S.pin(performance),
            ),
        ),
    )
    selected = {
        flag: json_file(tmp_path / (flag.removeprefix("--") + ".json"), {"selected": flag})
        for flag in S.FILE_FLAGS
        if not flag.endswith("-sha256")
    }
    argv = ["python", "-m", "merlin_experiments.phase1"]
    settings = dict(
        zip(
            S.COMMON_FLAGS,
            (
                "gpt-6.1-sol",
                "high",
                "codex",
                "/public/codex",
                "bwrap",
                "full",
                "EL4",
                "snapshot_external_v1",
                "full_compiler",
                "full",
                "continuous",
                "1200",
                "3600",
                "1200",
                "120",
                "rtl",
            ),
            strict=True,
        )
    )
    for flag, value in settings.items():
        argv += [flag, value]
    for flag in S.FILE_FLAGS:
        argv += [flag, "f" * 64 if flag.endswith("-sha256") else str(selected[flag])]
    argv += [
        "--compiler-library-root",
        "${INPUTS}/library",
        "--compiler-library-contract",
        "${ARTIFACTS}/contract.json",
        "--compiler-library-contract-sha256",
        "a" * 64,
        "--full-compiler-manifest",
        "${INPUTS}/full-compiler/manifest.json",
        "--codex-max-subagents",
        "8",
        "--author-max-subagents",
        "8",
        "--sim-max-jobs",
        "15",
        "--max-wall-s",
        "${DURATION_S}",
        "--round-timeout",
        "${DURATION_S}",
        "--operator-errata",
        "${ARTIFACTS}/author-guidance.md",
        "--joint-campaign",
        "${ARTIFACTS}/policy.json",
        "--fresh-backend",
        "--author-token-unbounded",
        "--live-conformance-feedback",
    ]
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            dict(
                schema=S.campaign_module().SCHEMA,
                variables=dict(TASK=str(tmp_path), REPO=str(tmp_path / "repo")),
                run=dict(run_id="original-fresh10cca", cpus="23,24,25", max_wall_s=43200),
                inputs=dict(public_references=str(manifest.parent)),
                plan=dict(
                    policy=dict(base=S.pin(base_policy), defaults={}),
                    argv=argv,
                    public_reference_manifest=str(manifest),
                    reference_map=S.pin(mapping),
                    timing_selection=str(performance),
                    environment={},
                    required_flags=["--fresh-backend", "--author-token-unbounded"],
                ),
            )
        )
    )
    coordinator = tmp_path / "control/studies/kernel_abstractions/study.py"
    coordinator.parent.mkdir(parents=True)
    coordinator.write_text("def run(*args, **kwargs):\n    raise AssertionError('not an author launch fixture')\n")
    helper = tmp_path / "control/launch/campaign_capacity.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("# reviewed capacity fixture\n")
    return dict(
        base=base,
        audit=audit,
        coordinator=coordinator,
        root=tmp_path,
        revision="a" * 40,
        manifest=manifest,
        mapping=mapping,
        policy=policy,
    )


def render(fixture, **overrides):
    kwargs = dict(
        baseline_audit=fixture["audit"],
        source_revision=fixture["revision"],
        export_id=fixture["revision"][:12],
        coordinator=fixture["coordinator"],
    )
    kwargs.update(overrides)
    output = fixture["root"] / "study"
    record = S.specs(fixture["base"], output, **kwargs)
    return output / "study.json", record


def prepared_fixture(fixture):
    study, record = render(fixture)
    common = S.reopen(record["common_workloads"])
    paths = []
    for row in record["arms"]:
        spec = yaml.safe_load(Path(row["spec"]["path"]).read_text())
        campaign = S.campaign_module()
        context = campaign.Context(row["spec"]["path"], export_id=record["export_id"])
        context.computed["DURATION_S"] = str(S.AUTHOR_S + S.DRAIN_S)
        argv = [context.x(item) for item in spec["plan"]["argv"]]
        argv += ["--author-profile", row["profile"], "--author-helper-effort", "medium"]
        arm_root = study.parent / row["arm"]
        policy_path = json_file(arm_root / "policy.json", S.reopen(record["common_base_policy"]))
        argv = S.set_option(argv, "--joint-campaign", policy_path)
        guide = arm_root / "guide.md"
        guide.write_text(Path(record["common_guidance"]["path"]).read_text())
        argv = S.set_option(argv, "--operator-errata", guide)
        timing = json_file(arm_root / "timing.json", dict(model="common-runtime"))
        framework = study.parent / "frameworks" / record["export_id"]
        json_file(framework / ".merlin-source-revision.json", dict(revision=record["source_revision"]))
        capabilities = dict(
            reference_analysis=False,
            automatic_kernel_learning=False,
            raw_kernel_references=row["arm"] != "no-reference-kernels",
            scheduling_semantics=row["arm"] != "infra",
            transform_library=row["arm"] == "no-reference-kernels",
            catalog_advice=row["arm"] == "no-reference-kernels",
            compiler_integration=True,
            performance_diagnostics=True,
        )
        plan = dict(
            source_revision=record["source_revision"],
            execution_started=False,
            run_id=row["run_id"],
            treatment=row["arm"],
            author_profile=dict(name=row["profile"], live_cca=False, capabilities=capabilities),
            framework_repository=str(framework),
            environment=dict.fromkeys(
                ("MERLIN_COMPUTE_CPUS", "MERLIN_CODEX_CPUS", "MERLIN_PREPARATION_CPUS"), ",".join(map(str, row["cpus"]))
            ),
            launch_spec=dict(sha256=row["spec"]["sha256"], unit=row["unit"]),
            cpu_reservation=dict(cpus=row["cpus"], currently_acquired=False),
            capsules=copy.deepcopy(S.COUNTS),
            argv=argv,
            reference_guidance="off",
            timing_selection=S.pin(timing),
            reference_manifest=None if row["arm"] == "no-reference-kernels" else S.pin(fixture["manifest"]),
            reference_map=None if row["arm"] == "no-reference-kernels" else S.pin(fixture["mapping"]),
        )
        path = json_file(arm_root / "proposed-launch.json", plan)
        paths.append(path)
    evidence = json_file(study.parent / "native-evidence.json", dict(all_passed=True, model_calls=0))
    review = dict(
        schema=S.REVIEW_SCHEMA,
        source_revision=record["source_revision"],
        export_id=record["export_id"],
        common_workloads_sha256=record["common_workloads"]["sha256"],
        baseline_sha256=record["baseline"]["sha256"],
        arms=[
            dict(
                arm=row["arm"],
                plan_sha256=S.pin(path)["sha256"],
                preflight_returncode=0,
                all_passed=True,
                initial_files=0,
                public_abi_files_read=23,
                evidence=[S.pin(evidence)],
            )
            for row, path in zip(record["arms"], paths, strict=True)
        ],
        **dict.fromkeys(S.REVIEW_CHECKS, True),
    )
    review_path = json_file(study.parent / "launch-review.json", review)
    return study, record, paths, review_path, common


def update_plan(path, review_path, transform):
    plan = json.loads(path.read_text())
    transform(plan)
    json_file(path, plan)
    review = json.loads(review_path.read_text())
    for row in review["arms"]:
        if row["arm"] == plan["treatment"]:
            row["plan_sha256"] = S.pin(path)["sha256"]
    json_file(review_path, review)


def test_render_preserves_native_flags_and_full_reference_set(fixture):
    _, record = render(fixture)
    assert record["references"]["map_entries"] == 155
    assert record["references"]["map_new_elf_subset"] == 68
    assert record["references"]["raw_members"] == 23
    assert record["references"]["public_abi_files_separate"] == 23
    assert record["panel"] == record["common_workloads"]
    assert S.reopen(record["common_workloads"])["protected_capsules"] == 14
    assert S.reopen(record["common_workloads"])["full_models"] == 6
    policy = S.reopen(record["common_base_policy"])
    assert "study_protocol" not in policy
    assert policy["gsim_correctness"]["continuous_promotion"] is True
    assert policy["native_calibration"] == fixture["policy"]["native_calibration"]
    for index, row in enumerate(record["arms"]):
        document = yaml.safe_load(Path(row["spec"]["path"]).read_text())
        assert document["plan"]["author_profile"] == S.PROFILES[index]
        assert document["inputs"]["author_profile_common_inputs"] == "fresh10_common"
        assert document["plan"]["reference_guidance"] == "off"
        assert document["plan"]["reference_experiments"] is None
        assert document["run"]["preparation_cpus"] == "0"
        assert document["run"]["max_wall_s"] == 15300
        assert document["variables"]["FRAMEWORK"].endswith("a" * 12)
        for flag in S.FILE_FLAGS:
            assert flag in document["plan"]["argv"]
        if index == 2:
            assert document["inputs"]["public_references"] is None
            assert document["plan"]["reference_map"] is None
            assert document["plan"]["public_reference_manifest"] is None
        else:
            assert document["plan"]["reference_map"] == S.pin(fixture["mapping"])
        assert "seed" not in document


def test_no_models_or_reservations_in_spec_generation(fixture, monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("process launched"))
    _, record = render(fixture)
    assert record["author_jobs_launched"] == record["model_calls"] == 0


def test_dispatch_envelope_retains_the_actual_external_cpu_reservations(fixture):
    _, record = render(fixture)
    assert record["reserved_external_cpus"] == list(range(4, 38)) + list(range(48, 56))
    assert not set(record["reserved_external_cpus"]) & {cpu for row in record["arms"] for cpu in row["cpus"]}


@pytest.mark.parametrize(
    "kwargs", [dict(source_revision="HEAD"), dict(export_id="b" * 12), dict(groups=[range(15)] * 3)]
)
def test_render_refuses_implicit_source_or_overlapping_pools(fixture, kwargs):
    with pytest.raises(ValueError):
        render(fixture, **kwargs)
    assert not (fixture["root"] / "study").exists()


def test_render_new_export_before_native_reference_output_exists(fixture):
    base = yaml.safe_load(fixture["base"].read_text())
    base["variables"]["FRAMEWORK"] = "${REPO}/out/build/run-frameworks/${EXPORT_ID}"
    base["plan"]["public_reference_manifest"] = "${FRAMEWORK}/out/new-inputs/public-references/reference-manifest.json"
    fixture["base"].write_text(yaml.safe_dump(base))
    assert not (fixture["root"] / "repo/out/build/run-frameworks" / fixture["revision"][:12]).exists()
    _, record = render(fixture)
    assert record["references"]["manifest_sha256"] == S.pin(fixture["manifest"])["sha256"]
    for row in record['arms']:
        context = S.campaign_module().Context(Path(row['spec']['path']), export_id=record['export_id'])
        inputs = context.path(context.section('inputs').get('output', '${INPUTS}'))
        assert inputs.is_relative_to(context.path('${FRAMEWORK}') / 'out/artifacts')
    assert not (fixture["root"] / "repo/out/build/run-frameworks" / fixture["revision"][:12]).exists()


def test_render_checks_source_reference_bank_before_preparation(fixture):
    base = yaml.safe_load(fixture["base"].read_text())
    bank = fixture["root"] / "wrong-bank"
    bank.mkdir()
    (bank / "reference-manifest.json").write_text("{}\n")
    base["inputs"]["public_references"] = str(bank)
    fixture["base"].write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match="actual fresh10 raw reference manifest"):
        render(fixture)


def test_refuses_map_truncation_to_68_new_entries(fixture):
    mapping = json.loads(fixture["mapping"].read_text())
    mapping["entries"] = mapping["entries"][:68]
    json_file(fixture["mapping"], mapping)
    audit = json.loads(fixture["audit"].read_text())
    audit["pins"]["reference_map"] = S.pin(fixture["mapping"])
    json_file(fixture["audit"], audit)
    with pytest.raises(ValueError, match="155"):
        render(fixture)


def test_prepare_accepts_realistic_unequal_reference_profiles(fixture):
    study, _, plans, review, _ = prepared_fixture(fixture)
    record = S.prepare(study, plans, review)
    assert record["state"] == "prepared"
    assert record["prepared_arms"] == list(S.ARMS)
    assert len({row["policy"]["sha256"] for row in record["arms"]}) == 1


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--validation-scope", "kernels"),
        ("--schedule", "phase"),
        ("--private-full-model-spec", None),
        ("--instruction-selection", None),
        ("--compiler-library-root", None),
        ("--codex-max-subagents", "9"),
        ("--sim-max-jobs", "16"),
        ("--author-helper-effort", "high"),
        ("--max-wall-s", "14400"),
        ("--model", "another-model"),
    ],
)
def test_prepare_rejects_silent_native_obligation_or_resource_changes(fixture, flag, value):
    study, _, plans, review, _ = prepared_fixture(fixture)

    def change(plan):
        if value is None:
            position = plan["argv"].index(flag)
            del plan["argv"][position : position + 2]
        else:
            plan["argv"] = S.set_option(plan["argv"], flag, value)

    update_plan(plans[1], review, change)
    with pytest.raises(ValueError):
        S.prepare(study, plans, review)
    assert json.loads(study.read_text())["state"] == "specifications"


@pytest.mark.parametrize("alter", ["deferred", "promotion_off", "calibration", "workloads"])
def test_prepare_rejects_deferred_evaluation_or_model_drift(fixture, alter):
    study, _, plans, review, _ = prepared_fixture(fixture)
    plan = json.loads(plans[1].read_text())
    path = Path(S.option(plan["argv"], "--joint-campaign"))
    policy = json.loads(path.read_text())
    if alter == "deferred":
        policy["study_protocol"] = dict(evaluation="deferred")
    elif alter == "promotion_off":
        policy["gsim_correctness"]["continuous_promotion"] = False
    elif alter == "calibration":
        policy["native_calibration"]["sha256"] = "0" * 64
    else:
        policy["gsim_correctness"]["capsules"] = ["subset"]
    json_file(path, policy)
    with pytest.raises(ValueError, match="normal full-model"):
        S.prepare(study, plans, review)


@pytest.mark.parametrize("arm,field", [(0, "reference_map"), (1, "reference_manifest"), (2, "reference_manifest")])
def test_prepare_enforces_reference_asymmetry_exactly(fixture, arm, field):
    study, _, plans, review, _ = prepared_fixture(fixture)
    update_plan(plans[arm], review, lambda plan: plan.update({field: S.pin(fixture["manifest"]) if arm == 2 else None}))
    with pytest.raises(ValueError, match="reference"):
        S.prepare(study, plans, review)


def test_prepare_does_not_accept_attestation_without_native_evidence(fixture):
    study, _, plans, review, _ = prepared_fixture(fixture)
    data = json.loads(review.read_text())
    data["arms"][0]["evidence"] = []
    json_file(review, data)
    with pytest.raises(ValueError, match="pinned evidence"):
        S.prepare(study, plans, review)


def test_launch_requires_confirmation_and_unchanged_coordinator(fixture):
    study, _, plans, review, _ = prepared_fixture(fixture)
    with pytest.raises(ValueError, match="confirm"):
        S.run(study)
    S.prepare(study, plans, review)
    fixture["coordinator"].write_text("# changed coordinator\n")
    with pytest.raises(ValueError, match="input changed"):
        S.run(study, confirm=True)


def runtime_fixture(fixture):
    study, record, plans, review, _ = prepared_fixture(fixture)
    record = S.prepare(study, plans, review)
    runtime = dict(schema="merlin.study.fresh10_runtime_index.v1", source_revision=record["source_revision"], arms=[])
    for arm in record["arms"]:
        root = study.parent / "runtime" / arm["arm"]
        workspace = root / "workspace"
        submission = workspace / "submission"
        submission.mkdir(parents=True)
        (submission / "compiler.py").write_text("# one coherent candidate\n")
        run_dir = root / "run"
        json_file(run_dir / "joint_campaign.json", dict(started_at_epoch=1000, policy=S.reopen(arm["policy"])))
        runtime["arms"].append(dict(arm=arm["arm"], workspace=str(workspace), run_dir=str(run_dir)))
    index = json_file(study.parent / "runtime-index.json", runtime)
    return study, index


def test_operator_snapshots_are_immutable_and_never_rewrite_author_policy(fixture):
    study, runtime = runtime_fixture(fixture)
    result = S.observe(study, runtime, once=True, now=1001)
    assert result["author_visible"] is False
    assert result["automatic_grade_credit"] is False
    item = result["arms"]["infra"]["snapshots"][0]
    assert item["status"] == "frozen"
    copied = Path(item["source"]) / "compiler.py"
    assert copied.stat().st_mode & 0o222 == 0
    again = S.observe(study, runtime, once=True, now=1002)
    assert again["arms"]["infra"]["snapshots"] == [item]


def test_late_observer_cannot_fabricate_historical_snapshots(fixture):
    study, runtime = runtime_fixture(fixture)
    result = S.observe(study, runtime, once=True, now=2901)
    for arm in S.ARMS:
        assert [item["offset_s"] for item in result["arms"][arm]["snapshots"]] == [0, 1800]
        assert all(item["status"] == "unavailable" for item in result["arms"][arm]["snapshots"])


def test_snapshot_clock_cannot_reset(fixture):
    study, runtime = runtime_fixture(fixture)
    S.observe(study, runtime, once=True, now=1001)
    data = json.loads(runtime.read_text())
    path = Path(data["arms"][0]["run_dir"]) / "joint_campaign.json"
    campaign = json.loads(path.read_text())
    campaign["started_at_epoch"] = 2000
    json_file(path, campaign)
    with pytest.raises(ValueError, match="clock changed"):
        S.observe(study, runtime, once=True, now=2001)


@pytest.mark.parametrize("kind", ["cca", "learning", "passes", "mask", "framework", "model"])
def test_prepare_rejects_capability_or_native_binding_drift(fixture, kind):
    study, _, plans, review, _ = prepared_fixture(fixture)

    def change(plan):
        if kind == "cca":
            plan["author_profile"]["live_cca"] = True
        elif kind == "learning":
            plan["author_profile"]["capabilities"]["automatic_kernel_learning"] = True
        elif kind == "passes":
            plan["author_profile"]["capabilities"]["transform_library"] = True
        elif kind == "mask":
            plan["environment"]["MERLIN_COMPUTE_CPUS"] = "0,1"
        elif kind == "framework":
            marker = Path(plan["framework_repository"]) / ".merlin-source-revision.json"
            json_file(marker, dict(revision="b" * 40))
        else:
            timing = Path(plan["timing_selection"]["path"])
            json_file(timing, dict(model="changed-runtime"))
            plan["timing_selection"] = S.pin(timing)

    update_plan(plans[1], review, change)
    with pytest.raises(ValueError):
        S.prepare(study, plans, review)


def test_run_rechecks_native_policy_after_preparation(fixture):
    study, _, plans, review, _ = prepared_fixture(fixture)
    record = S.prepare(study, plans, review)
    path = Path(record["arms"][0]["policy"]["path"])
    policy = json.loads(path.read_text())
    policy["drain_s"] = 1
    json_file(path, policy)
    with pytest.raises(ValueError, match="policy pin changed"):
        S.run(study, confirm=True)
