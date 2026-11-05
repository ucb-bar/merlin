"""Synthetic lifecycle proof, not an agent or hardware qualification.

Real controller/checkpoint sequencing, snapshots, candidate copy/hash and regrade
admission, paired CLI/plans/raw writer/evidence admission, statistics and final seal.
Injected: Chia/preflight/treatment, admitted functional cohort and candidate handoff,
certificate/RTL provenance, commit/reveal materializer, and engine execution.
Every process launch is forbidden; no native controller imports are used.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import checkpoint_admission as AD
from merlin_experiments.phase2 import checkpoint_controller as CTRL
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import corpus as CORPUS
from merlin_experiments.phase2 import form_holdout as FORM
from merlin_experiments.phase2 import functional_cohort as FC
from merlin_experiments.phase2 import gsim_gate as GATE
from merlin_experiments.phase2 import gsim_workload as WORKLOAD
from merlin_experiments.phase2 import holdout_corpus as HOLDOUT
from merlin_experiments.phase2 import measurement_support as MS
from merlin_experiments.phase2 import paired_cli as pair
from merlin_experiments.phase2 import paired_inputs as PI
from merlin_experiments.phase2 import paired_measurement as PME

from merlin.benchharness import hash_tree


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    return build_lifecycle(tmp_path, monkeypatch)


def _synthetic_reveal(root, *, target_name, descriptor, name, family, cohort, shape):
    """A frozen v2 reveal of one member, read back through the paired-input loader."""
    member = root / "_perf" / name
    member.mkdir(parents=True)
    m, k, n = shape
    member_descriptor = {
        **descriptor,
        "name": name,
        "performance": {**descriptor["performance"], "family": family},
        "inputs": [
            {"name": "X", "role": "input", "shape": [m, k], "dtype": "i8"},
            {"name": "W", "role": "weight", "shape": [k, n], "dtype": "i8"},
        ],
    }
    (member / "capsule.yaml").write_text(json.dumps(member_descriptor))
    tree = PI.holdout_tree_record(root)
    manifest = root / "holdout_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "generated_performance_holdout_reveal",
                "domain": {"target": target_name},
                "cohorts": {cohort: {"family": family, "member_count": 1}},
                "members": [
                    {"name": name, "path": f"_perf/{name}", "family": family, "cohort": cohort, "M": m, "N": n, "K": k}
                ],
                "corpus": tree,
            }
        )
    )
    for path in reversed([root, *root.rglob("*")]):
        path.chmod(0o555 if path.is_dir() else 0o444)
    return PI.load_holdout_corpus(
        root,
        manifest,
        manifest_sha256=C.sha256_file(manifest),
        capsules_sha256=tree["sha256"],
        expected_target=target_name,
    )


def build_lifecycle(
    tmp_path,
    monkeypatch,
    *,
    functional_run=None,
    target_name="fixture",
    published_compiler_root=None,
    form_holdout=False,
    form_shape=(32, 17, 16),
    form_generated_root=None,
):
    """Run the real checkpoint owner, optionally consuming a Phase-1 formal handoff.

    With ``form_holdout`` the campaign also commits and reveals a synthetic form-scale cohort of
    one member of ``form_shape`` (M, K, N); its private generated root is ``form_generated_root``.
    """

    def forbidden(*args, **kwargs):
        pytest.fail("synthetic lifecycle attempted a native process")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setenv(CTRL.FANOUT_ENVIRONMENT_VARIABLE, "1")
    descriptor = {
        "name": "case",
        "label": "dev",
        "source_role": "derived_sweep",
        "performance": {"family": "PK", "claim": "RECOVERS"},
        "operation": {"op": "matmul", "attributes": {"lhs": "X", "weight": "W", "out": "Y", "output_dtype": "i32"}},
        "inputs": [
            {"name": "X", "role": "input", "shape": [16, 17], "dtype": "i8"},
            {"name": "W", "role": "weight", "shape": [17, 16], "dtype": "i8"},
        ],
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
    }
    public = tmp_path / "live/public"
    public.mkdir(parents=True)
    member = public.parent / "_tuning/case"
    member.mkdir(parents=True)
    (member / "capsule.yaml").write_text(json.dumps(descriptor))
    (public.parent / "MANIFEST.yaml").write_text(
        json.dumps(
            {
                "generated": ["_tuning/case"],
                "hand_authored": [],
                "performance_generation": {
                    target_name: {
                        "errors": [],
                        "phase": {
                            "category": "_tuning",
                            "label": "dev",
                            "included_in_functional_grade": False,
                        },
                    }
                },
            }
        )
    )
    target = SimpleNamespace(
        target=target_name, descriptor_sha256="d" * 64, capsule_corpus=public, graded_roots=lambda: [public]
    )
    tuning = CORPUS.freeze_performance_corpus(CORPUS.discover_performance_corpus(target), tmp_path / "tuning")
    holdout_root = tmp_path / "heldout"
    holdout_member = holdout_root / "_perf/case"
    holdout_member.mkdir(parents=True)
    (holdout_member / "capsule.yaml").write_text(json.dumps(descriptor))
    holdout_tree = PI.holdout_tree_record(holdout_root)
    holdout_manifest = holdout_root / "holdout_manifest.json"
    holdout_manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "generated_performance_holdout_reveal",
                "domain": {"target": target_name},
                "cohorts": {"PK_predictor": {"family": "PK", "member_count": 1}},
                "members": [
                    {
                        "name": "case",
                        "path": "_perf/case",
                        "family": "PK",
                        "cohort": "PK_predictor",
                        "M": 16,
                        "N": 16,
                        "K": 17,
                    }
                ],
                "corpus": holdout_tree,
            }
        )
    )
    for path in reversed([holdout_root, *holdout_root.rglob("*")]):
        path.chmod(0o555 if path.is_dir() else 0o444)
    holdout = PI.load_holdout_corpus(
        holdout_root,
        holdout_manifest,
        manifest_sha256=C.sha256_file(holdout_manifest),
        capsules_sha256=holdout_tree["sha256"],
        expected_target=target_name,
    )
    if functional_run is None:
        baseline = tmp_path / "baseline"
        baseline.mkdir()
        (baseline / "compiler.py").write_text("# frozen functional bytes\n")
        baseline_sha = hash_tree(baseline)["sha256"]
        functional = SimpleNamespace(
            run_id="functional",
            digest=baseline_sha,
            submission_dir=baseline,
            frozen_at=None,
            public_score={"per_capsule": [{"capsule": "public", "tiers": {"L3": "pass"}}]},
            hidden_score={"per_capsule": [{"capsule": "hidden", "tiers": {"L3": "pass"}}]},
        )
        functional_runs_root = tmp_path / "functional-runs"
    else:
        functional = functional_run
        baseline = functional.submission_dir
        baseline_sha = functional.digest
        assert hash_tree(baseline)["sha256"] == baseline_sha
        functional_runs_root = functional.run_dir.parents[1]
    workload = PME.gsim_workload(tuning.capsules[0])
    certificate = SimpleNamespace(
        sha256="c" * 64,
        path=tmp_path / "certificate.json",
        unresolved={},
        target=target_name,
        pins={name: {"sha256": "a" * 64} for name in GATE.REQUIRED_PINS},
        members={GATE.workload_sha256(workload): {}},
        to_dict=lambda: {"sha256": "c" * 64},
    )
    form_corpus = form_certificate = None
    if form_holdout:
        form_corpus = _synthetic_reveal(
            tmp_path / "heldout-form",
            target_name=target_name,
            descriptor=descriptor,
            name="form_case",
            family="PW",
            cohort=FORM.COHORT,
            shape=form_shape,
        )
        form_workload = WORKLOAD.derive_workload(form_corpus.root / "_perf/form_case/capsule.yaml")
        # The form cohort's own same-build extension: the tuning envelope plus its one revealed workload.
        form_certificate = SimpleNamespace(
            **{
                **vars(certificate),
                "sha256": "f" * 64,
                "path": tmp_path / "experiment/heldout_gsim_qualification_form/certificate.json",
                "members": {**certificate.members, GATE.workload_sha256(form_workload): {}},
                "to_dict": lambda: {"sha256": "f" * 64},
            }
        )

    def load_certificate(path, *_a, **_k):
        if form_certificate is not None and Path(path).resolve() == form_certificate.path.resolve():
            return form_certificate
        return certificate

    monkeypatch.setattr(pair, "load_target_experiment", lambda _, **kw: target)
    monkeypatch.setattr(GATE, "load_certificate", load_certificate)
    monkeypatch.setattr(MS, "load_rtl_identity", lambda *_a: {"fixture": "external RTL facts"})
    monkeypatch.setattr(MS, "roofline_auxiliary_requirements", lambda *_a: {"fixture": "not qualified"})
    runs = tmp_path / "runs"
    runs.mkdir()

    execute = PME.execute_schedule

    def engine(spec, *_args, **_kwargs):
        return {
            "execution": spec.as_dict(),
            "measurement": {
                "status": "pass",
                "numeric": "pass",
                "gsim_qualification": {"admitted": True},
                "per_sim": {
                    "spike": {"correct": True},
                    "gsim": {
                        "correct": True,
                        "cycles": 100 if spec.arm == "baseline" else 80,
                        "provenance": {
                            "tier": "L3",
                            "simulator": "gsim",
                            "oracle_kind": "rtl_gsim",
                            "derived_from_rtl": True,
                            "cycle_accurate": True,
                            "elf_sha256": "e" * 64,
                            "reused_measurement": False,
                        },
                    },
                },
            },
        }

    monkeypatch.setattr(PME, "execute_schedule", lambda *a, **kw: execute(*a, **kw, executor=engine))
    events = []
    interruptions = {}
    resources = tmp_path / "resources"
    resources.mkdir()
    for name in ("target.yaml", "facts.json", "profile.json", "prices.json"):
        (resources / name).write_text("{}")
    context = AD.ExecutionContext(
        source_root=tmp_path,
        contract_root=resources / "contract",
        functional_runs_root=functional_runs_root,
        stage_root=tmp_path / "stages",
        measurement_root=runs,
        holdout_sources=HOLDOUT.HoldoutSourceContext(
            tmp_path,
            resources / "catalog.yaml",
            tmp_path / "core",
            tmp_path / "experiments",
            tmp_path / "namespace",
        ),
        chia_wrapper=resources / "wrapper.py",
        invocation=("synthetic-python", "synthetic-driver.py"),
        suite="synthetic-lifecycle",
    )
    config = AD.Config(
        context=context,
        experiment_id="lifecycle",
        root=tmp_path / "experiment",
        functional_run_id=functional.run_id,
        functional_submission_sha256=baseline_sha,
        descriptor=resources / "target.yaml",
        rtl_facts=resources / "facts.json",
        perf_profile=resources / "profile.json",
        telemetry_price_table=resources / "prices.json",
        gsim_certificate=certificate.path,
        gsim_certificate_sha256=certificate.sha256,
        model="synthetic",
        effort="high",
        wall_budget_seconds=60,
        rounds=1,
        round_timeout_seconds=60,
        max_tool_calls=1,
        tool_timeout_seconds=60,
        smoke_replicates=1,
        holdout_count=1,
        measurement_timeout=60,
        waive_functional_gsim_certificate=True,
        published_compiler_root=published_compiler_root,
    )
    if form_holdout:
        if form_generated_root is None:
            form_generated_root = tmp_path / "private-form-run"
            form_generated_root.mkdir()
        config = dataclasses.replace(
            config,
            form_holdout_generated_root=form_generated_root,
            form_holdout_applications=("private_application",),
        )
    treatment = {"identity": "synthetic-external-agent"}
    contracts = {trial: {"trial": trial, "treatment_identity": treatment} for trial in AD.TRIALS}
    declaration = {
        "status": "GO",
        "agent_treatment": treatment,
        "trial_contracts": contracts,
        "agent_telemetry": {},
        "orchestration": {
            "required_entrypoint_sha256": "w" * 64,
            "chia_trace_sha256": "t" * 64,
        },
    }
    if form_holdout:
        declaration["form_holdout"] = AD.form_holdout_declaration(config)
    if published_compiler_root is not None:
        from merlin_experiments.phase2 import published_payload

        declaration["published_compiler"] = published_payload.inspect(
            functional, published_compiler_root, target=target_name
        ).identity()
    launch = {"wrapper": {"sha256": "w" * 64}, "chia_trace": {"sha256": "t" * 64}}
    monkeypatch.setattr(CTRL.CHIA, "verify_launch_receipt", lambda **kw: launch)
    monkeypatch.setattr(AD, "preflight", lambda *a, **kw: declaration)
    monkeypatch.setattr(AD, "_verify_live_agent_treatment", lambda *a: None)
    monkeypatch.setattr(AD, "child_environment", lambda *a: {})

    def load_target(path, *, source_root):
        assert source_root == context.source_root
        assert path == config.descriptor
        return target

    monkeypatch.setattr(CTRL, "load_target_experiment", load_target)
    monkeypatch.setattr(CTRL.FI, "inspect_stage_functional_run", lambda *a, **kw: functional)
    cohort = FC.FunctionalGradeCohort((), (), 0, 0, admission_descriptor_sha256=target.descriptor_sha256)
    monkeypatch.setattr(FC, "functional_grade_cohort_from_run", lambda *a, **kw: cohort)
    monkeypatch.setattr(FC, "declined_names", lambda _: ())
    monkeypatch.setattr(AD, "_frozen_functional_regrade_inputs", lambda *a: ("public", "hidden", context.contract_root))
    handoffs = {}

    def read_handoff(path, **kwargs):
        handoff = handoffs[Path(path).parent.name.split("__")[-1]]
        return SimpleNamespace(**{**vars(handoff), "record_sha256": C.sha256_file(Path(path))})

    monkeypatch.setattr(CTRL.VERIFY, "verify_candidate_handoff", read_handoff)

    def paired_inputs(record, *args, **kwargs):
        handoff = read_handoff(record)
        corpus = tuning if kwargs["phase"] == "tuning" else holdout
        if form_corpus is not None and Path(kwargs["corpus_root"]) == form_corpus.root:
            corpus = form_corpus
        selected = load_certificate(kwargs["gsim_certificate"])
        assert Path(kwargs["functional_runs_root"]) == context.functional_runs_root
        return PME.PairedInputs(
            functional,
            handoff,
            corpus,
            kwargs["phase"],
            baseline,
            baseline_sha,
            handoff.candidate_path,
            handoff.candidate_sha256,
            selected,
        )

    monkeypatch.setattr(PI, "load_paired_inputs", paired_inputs)

    def stages():
        paths = sorted((config.root / "state").glob("checkpoint.*.json"))
        return [json.loads(path.read_bytes())["stage"] for path in paths]

    def runner(command, cwd, environment):
        assert cwd == context.source_root
        assert command[1] == "-m"
        module = command[2]

        def value(flag):
            return command[command.index(flag) + 1]

        if module == "merlin_experiments.phase2.authoring_cli":
            assert "holdout_committed" in stages()
            directory = Path(value("--stage-root"))
            trial = directory.name.split("__")[-1]
            events.append("author:" + trial)
            candidate = directory / "candidate"
            candidate.mkdir(parents=True)
            (candidate / "compiler.py").write_text("# synthetic candidate " + trial)
            (candidate / "compiler.py").chmod(0o444)
            candidate.chmod(0o555)
            record = directory / "performance_candidate.json"
            record.write_text(json.dumps({"synthetic": trial}))
            record.chmod(0o444)
            artifacts = {
                key: {"sha256": "a" * 64}
                for key in ("combined_raw", "trajectory", "aet_metrics_log", "cost_time_toolcalls", "activity_share")
            }
            handoffs[trial] = SimpleNamespace(
                record_path=record,
                record_sha256=C.sha256_file(record),
                candidate_path=candidate,
                candidate_sha256=hash_tree(candidate)["sha256"],
                corpus_root=tuning.root,
                corpus_manifest_sha256=tuning.manifest_sha256,
                corpus_sha256=tuning.capsules_sha256,
                functional_run_id=functional.run_id,
                functional_submission_sha256=baseline_sha,
                agent_contract=contracts[trial],
                telemetry_evidence={
                    "preflight_sha256": AD._sha_bytes(AD._canonical({})),
                    "artifacts": artifacts,
                    "tool_call_count": 0,
                    "subagent_tool_calls_tracked": True,
                },
            )
        elif module == "merlin_experiments.phase1.feedback.formal":
            assert all("candidate:" + trial in stages() for trial in AD.TRIALS)
            directory = Path(value("--run-dir"))
            trial = directory.name
            events.append("regrade:" + trial)
            (directory / "run_manifest.yaml").write_text(
                json.dumps(
                    {
                        "submission_sha256": handoffs[trial].candidate_sha256,
                        "completion": {"formal_grade_complete": True},
                        "public_dev": {"formal_complete": True},
                        "hidden": {"formal_complete": True},
                    }
                )
            )
        elif module == "merlin_experiments.phase2.paired_cli":
            assert "statistics_predeclared" in stages()
            assert Path(value("--source-root")) == context.source_root
            if interruptions.get("measurement") == value("--run-id"):
                del interruptions["measurement"]
                events.append("interrupted:" + value("--run-id"))
                raise AD.ExperimentError("synthetic interruption before measurement output")
            events.append("measure:" + value("--run-id"))
            assert pair.main(list(command[3:])) == 0
        else:
            pytest.fail("unexpected external boundary " + module)
        return AD.CommandResult(0)

    def commit(public_path, private_path, **kwargs):
        assert stages() == ["predeclared"]
        failure = interruptions.pop("commit", None)
        if failure is not None:
            raise failure
        events.append("commit")
        public_path.write_text("synthetic holdout commitment")
        private_path.mkdir()
        return HOLDOUT.HoldoutPaths(public_path, private_path, private_path / "seed", private_path / "state")

    def reveal(public_path, private_path, destination, **kwargs):
        assert all("functional_regrade:" + trial in stages() for trial in AD.TRIALS)
        assert set(kwargs["candidate_seals"]) == set(AD.TRIALS)
        events.append("reveal")
        return holdout.manifest_path

    def commit_form(generated_root, public_path, private_path, **kwargs):
        assert stages() == ["predeclared", "holdout_committed"]
        assert generated_root == config.form_holdout_generated_root
        assert kwargs["applications"] == config.form_holdout_applications
        assert kwargs["candidate_ids"] == AD.TRIALS
        assert kwargs["agent_view_root"] == config.root / "agent_visible"
        assert kwargs["tuning_root"] == public.parent.resolve()
        assert public_path.parent == kwargs["agent_view_root"]
        assert private_path.parent == config.root
        failure = interruptions.pop("commit_form", None)
        if failure is not None:
            raise failure
        events.append("commit_form")
        public_path.write_text("synthetic form-scale commitment")
        private_path.mkdir()
        return {"public_commitment": public_path, "host_private_dir": private_path, "state": private_path / "state"}

    def reveal_form(public_path, private_path, destination, **kwargs):
        assert stages()[-1] == "holdout_revealed"
        assert set(kwargs["candidate_seals"]) == set(AD.TRIALS)
        assert destination.parent == config.root
        failure = interruptions.pop("reveal_form", None)
        if failure is not None:
            raise failure
        events.append("reveal_form")
        return form_corpus.manifest_path

    def qualify(manifest, destination, tuning_certificate):
        form = form_corpus is not None and Path(manifest) == form_corpus.manifest_path
        assert stages()[-1] == (
            "heldout_gsim_certificate" if form else "form_holdout_revealed" if form_holdout else "holdout_revealed"
        )
        events.append("qualify_form" if form else "qualify")
        destination.mkdir()
        path = destination / "certificate.json"
        path.write_text("synthetic external certificate")
        return path, (form_certificate if form else certificate).sha256

    def run():
        return CTRL.run(
            config,
            command_runner=runner,
            commit_holdout=commit,
            reveal_holdout=reveal,
            heldout_certificate_provider=qualify,
            commit_form_holdout=commit_form,
            reveal_form_holdout=reveal_form,
        )

    return SimpleNamespace(
        run=run,
        config=config,
        events=events,
        stages=stages,
        handoffs=handoffs,
        interruptions=interruptions,
        declaration=declaration,
    )


def test_real_checkpoint_lifecycle_and_resume_without_relaunch(lifecycle):
    path = lifecycle.run()
    document = json.loads(path.read_bytes())
    expected = ["predeclared", "holdout_committed"]
    expected += ["candidate:" + trial for trial in AD.TRIALS]
    expected += ["functional_regrade:" + trial for trial in AD.TRIALS]
    expected += ["holdout_revealed", "heldout_gsim_certificate", "statistics_predeclared"]
    expected += [f"measurement:{trial}:{phase}" for trial in AD.TRIALS for phase in ("tuning", "held_out")]
    assert lifecycle.stages() == expected
    assert lifecycle.events[:9] == ["commit"] + ["author:" + trial for trial in AD.TRIALS] + [
        "regrade:" + trial for trial in AD.TRIALS
    ] + ["reveal", "qualify"]
    assert len(document["measurement_manifests"]) == 6
    assert document["statistics"]["status"] == "admitted"
    assert document["statistics"]["aggregate"]["median_speedup"] == 1.25
    assert path.stat().st_mode & 0o222 == 0
    events = list(lifecycle.events)
    assert lifecycle.run() == path
    assert lifecycle.events == events


@pytest.mark.parametrize("tamper", ["checkpoint", "candidate", "candidate_bytes", "measurement", "regrade"])
def test_lifecycle_resume_refuses_tampered_evidence_without_relaunch(lifecycle, tamper):
    manifest = json.loads(lifecycle.run().read_bytes())
    if tamper == "checkpoint":
        path = next((lifecycle.config.root / "state").glob("checkpoint.0000.*"))
    elif tamper == "candidate":
        path = lifecycle.handoffs[AD.TRIALS[0]].record_path
    elif tamper == "candidate_bytes":
        path = lifecycle.handoffs[AD.TRIALS[0]].candidate_path / "compiler.py"
    elif tamper == "measurement":
        path = Path(manifest["measurement_manifests"][0]["path"])
    else:
        path = Path(manifest["functional_regrades"][AD.TRIALS[0]]["path"])
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b"\n ")
    events = list(lifecycle.events)
    with pytest.raises(AD.ExperimentError):
        lifecycle.run()
    assert lifecycle.events == events


def test_partial_authoring_child_is_not_rerun_in_place(lifecycle):
    stage = lifecycle.config.context.stage_root / (lifecycle.config.experiment_id + "__" + AD.TRIALS[0])
    stage.mkdir(parents=True)
    (stage / "partial.txt").write_text("incomplete synthetic authoring output")
    with pytest.raises(AD.ExperimentError, match="partial; refusing in-place rerun"):
        lifecycle.run()
    assert lifecycle.events == ["commit"]
    assert lifecycle.stages() == ["predeclared", "holdout_committed"]
    assert (stage / "partial.txt").read_text() == "incomplete synthetic authoring output"


def test_interrupted_measurement_resumes_without_repeating_completed_phases(lifecycle):
    first = "lifecycle__trial_00__tuning"
    second = "lifecycle__trial_00__held_out"
    lifecycle.interruptions["measurement"] = second
    with pytest.raises(AD.ExperimentError, match="synthetic interruption"):
        lifecycle.run()
    # Child launches finish before ordered commits. The first cell exists but is
    # deliberately uncheckpointed until the complete child batch succeeds.
    assert lifecycle.stages()[-1] == "statistics_predeclared"
    assert not (lifecycle.config.context.measurement_root / second).exists()
    earlier = list(lifecycle.events)
    first_manifest = lifecycle.config.context.measurement_root / first / "campaign_manifest.json"
    first_bytes = first_manifest.read_bytes()
    final = json.loads(lifecycle.run().read_bytes())
    assert lifecycle.events[: len(earlier)] == earlier
    resumed = lifecycle.events[len(earlier) :]
    assert len(resumed) == 5
    assert all(event.startswith("measure:") for event in resumed)
    assert resumed[0] == "measure:" + second
    assert lifecycle.events.count("measure:" + first) == 1
    assert first_manifest.read_bytes() == first_bytes
    assert len(final["measurement_manifests"]) == 6
    assert final["statistics"]["status"] == "admitted"


# --- Form-scale holdout beside the PK holdout -------------------------------------------------------

_FORM_LABELS = ("tuning", "held_out", AD.FORM_HOLDOUT_MEASUREMENT_LABEL)
_FORM_STAGES = (
    ["predeclared", "holdout_committed", "form_holdout_committed"]
    + ["candidate:" + trial for trial in AD.TRIALS]
    + ["functional_regrade:" + trial for trial in AD.TRIALS]
    + ["holdout_revealed", "form_holdout_revealed", "heldout_gsim_certificate", "form_heldout_gsim_certificate"]
    + ["statistics_predeclared"]
    + [f"measurement:{trial}:{label}" for trial in AD.TRIALS for label in _FORM_LABELS]
)


@pytest.fixture
def form_lifecycle(tmp_path, monkeypatch):
    return build_lifecycle(tmp_path, monkeypatch, form_holdout=True)


def test_form_holdout_is_committed_revealed_qualified_and_measured_beside_pk(form_lifecycle):
    path = form_lifecycle.run()
    document = json.loads(path.read_bytes())
    assert form_lifecycle.stages() == _FORM_STAGES
    assert form_lifecycle.events[:12] == ["commit", "commit_form"] + ["author:" + trial for trial in AD.TRIALS] + [
        "regrade:" + trial for trial in AD.TRIALS
    ] + ["reveal", "reveal_form", "qualify", "qualify_form"]
    measured = [event for event in form_lifecycle.events if event.startswith("measure:")]
    assert measured == [f"measure:lifecycle__{trial}__{label}" for trial in AD.TRIALS for label in _FORM_LABELS]
    assert len(document["measurement_manifests"]) == 9
    assert document["statistics"]["status"] == "admitted"
    assert document["holdout_verdict"] == "complete_pk_and_form_scale"
    form = document["form_holdout"]
    assert form["measurement_label"] == AD.FORM_HOLDOUT_MEASUREMENT_LABEL
    assert form["heldout_gsim_certificate"]["sha256"] == "f" * 64
    assert document["heldout_gsim_certificate"]["sha256"] == "c" * 64
    assert Path(form["commitment"]["public"]).parent == form_lifecycle.config.root / "agent_visible"
    assert Path(form["commitment"]["private"]).parent == form_lifecycle.config.root
    assert document["declaration"]["form_holdout"] == AD.form_holdout_declaration(form_lifecycle.config)
    declared = (form_lifecycle.config.root / "statistics_predeclaration.json").read_text()
    assert "held_out:PW" in declared and "form_case" in declared
    events = list(form_lifecycle.events)
    assert form_lifecycle.run() == path
    assert form_lifecycle.events == events


def test_pk_only_campaign_records_carry_no_form_holdout(lifecycle):
    document = AD._config_document(lifecycle.config)
    assert not any(key.startswith("form_holdout") for key in document)
    assert AD.form_holdout_declaration(lifecycle.config) is None
    final = json.loads(lifecycle.run().read_bytes())
    assert "form_holdout" not in final and "holdout_verdict" not in final
    assert not any("form" in stage for stage in lifecycle.stages())


def test_form_holdout_binds_the_campaign_identity_by_roster_digest(form_lifecycle):
    document = AD._config_document(form_lifecycle.config)
    assert "private_application" not in json.dumps(document)
    assert document["form_holdout"]["roster_sha256"] == FORM.roster_sha256(["private_application"])
    other = dataclasses.replace(form_lifecycle.config, form_holdout_applications=("another_application",))
    assert AD._canonical(AD._config_document(other)) != AD._canonical(document)


def test_a_half_declared_form_holdout_is_refused(lifecycle):
    with pytest.raises(AD.ExperimentError, match="both its private generated root and its private roster"):
        AD.form_holdout_configured(dataclasses.replace(lifecycle.config, form_holdout_applications=("a",)))
    with pytest.raises(AD.ExperimentError, match="absolute"):
        AD.form_holdout_configured(
            dataclasses.replace(
                lifecycle.config, form_holdout_generated_root=Path("relative"), form_holdout_applications=("a",)
            )
        )


@pytest.mark.parametrize("stage", ["commit_form", "reveal_form"])
def test_a_failed_form_holdout_refuses_the_run_instead_of_reporting_pk_alone(form_lifecycle, stage):
    form_lifecycle.interruptions[stage] = HOLDOUT.HoldoutError("synthetic form-scale refusal")
    with pytest.raises(HOLDOUT.HoldoutError, match="synthetic form-scale refusal"):
        form_lifecycle.run()
    assert not list(form_lifecycle.config.root.glob("experiment_manifest.*.json"))
    if stage == "commit_form":
        # Refused before authoring: no agent ran against a half-committed holdout.
        assert form_lifecycle.stages() == ["predeclared", "holdout_committed"]
        assert not any(event.startswith("author:") for event in form_lifecycle.events)
    else:
        assert form_lifecycle.stages()[-1] == "holdout_revealed"
        assert "qualify" not in form_lifecycle.events
        assert not any(event.startswith("measure:") for event in form_lifecycle.events)
        # A resume after the cause is gone completes BOTH cohorts; nothing PK-only was ever sealed.
        document = json.loads(form_lifecycle.run().read_bytes())
        assert form_lifecycle.stages() == _FORM_STAGES
        assert document["holdout_verdict"] == "complete_pk_and_form_scale"


def test_form_holdout_must_match_its_predeclaration(form_lifecycle):
    form_lifecycle.declaration.pop("form_holdout")
    with pytest.raises(AD.ExperimentError, match="differs from its predeclaration"):
        form_lifecycle.run()
    assert form_lifecycle.events == []
    assert form_lifecycle.stages() == ["predeclared"]


def test_private_form_root_inside_an_agent_reachable_root_commits_nothing(tmp_path, monkeypatch):
    private = tmp_path / "stages" / "private-form-run"
    private.mkdir(parents=True)
    lifecycle = build_lifecycle(tmp_path, monkeypatch, form_holdout=True, form_generated_root=private)
    with pytest.raises(AD.ExperimentError, match="agent-reachable"):
        lifecycle.run()
    assert lifecycle.events == []
    assert not (lifecycle.config.root / "agent_visible" / "holdout_commitment.json").exists()


def test_form_cohort_sharing_a_pk_workload_is_refused_at_reveal(tmp_path, monkeypatch):
    lifecycle = build_lifecycle(tmp_path, monkeypatch, form_holdout=True, form_shape=(16, 17, 16))
    with pytest.raises(AD.ExperimentError, match="not disjoint: 0 shared name"):
        lifecycle.run()
    assert lifecycle.stages()[-1] == "holdout_revealed"
    assert "qualify" not in lifecycle.events
