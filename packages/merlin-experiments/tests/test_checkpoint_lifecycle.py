"""Synthetic lifecycle proof, not an agent or hardware qualification.

Real controller/checkpoint sequencing, snapshots, candidate copy/hash and regrade
admission, paired CLI/plans/raw writer/evidence admission, statistics and final seal.
Injected: Chia/preflight/treatment, admitted functional cohort and candidate handoff,
certificate/RTL provenance, commit/reveal materializer, and engine execution.
Every process launch is forbidden; no native controller imports are used.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import checkpoint_admission as AD
from merlin_experiments.phase2 import checkpoint_controller as CTRL
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import corpus as CORPUS
from merlin_experiments.phase2 import functional_cohort as FC
from merlin_experiments.phase2 import gsim_gate as GATE
from merlin_experiments.phase2 import holdout_corpus as HOLDOUT
from merlin_experiments.phase2 import measurement_support as MS
from merlin_experiments.phase2 import paired_cli as pair
from merlin_experiments.phase2 import paired_inputs as PI
from merlin_experiments.phase2 import paired_measurement as PME

from merlin.benchharness import hash_tree


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    return build_lifecycle(tmp_path, monkeypatch)


def build_lifecycle(tmp_path, monkeypatch, *, functional_run=None, target_name="fixture"):
    """Run the real checkpoint owner, optionally consuming a Phase-1 formal handoff."""

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
    monkeypatch.setattr(pair, "load_target_experiment", lambda _, **kw: target)
    monkeypatch.setattr(GATE, "load_certificate", lambda *_a, **_k: certificate)
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
            certificate,
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
        events.append("commit")
        public_path.write_text("synthetic holdout commitment")
        private_path.mkdir()
        return HOLDOUT.HoldoutPaths(public_path, private_path, private_path / "seed", private_path / "state")

    def reveal(public_path, private_path, destination, **kwargs):
        assert all("functional_regrade:" + trial in stages() for trial in AD.TRIALS)
        assert set(kwargs["candidate_seals"]) == set(AD.TRIALS)
        events.append("reveal")
        return holdout.manifest_path

    def qualify(manifest, destination, tuning_certificate):
        assert stages()[-1] == "holdout_revealed"
        events.append("qualify")
        destination.mkdir()
        path = destination / "certificate.json"
        path.write_text("synthetic external certificate")
        return path, certificate.sha256

    def run():
        return CTRL.run(
            config,
            command_runner=runner,
            commit_holdout=commit,
            reveal_holdout=reveal,
            heldout_certificate_provider=qualify,
        )

    return SimpleNamespace(
        run=run, config=config, events=events, stages=stages, handoffs=handoffs, interruptions=interruptions
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
