"""Real reviewed derivation→frozen Phase-1 inputs and formal receipt handoff.

Phase-0 generation and the installed Phase-1 authoring owner execute, with a
synthetic in-process provider. No compiler, simulator, paid agent or listener runs.
Joined formal/Phase-2 tests use synthetic oracle, sandbox and measurement
observations, not native qualification.
"""

import importlib.util
import json
import shutil
import socket
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus import release
from merlin_experiments.phase1 import controller as PHASE1
from merlin_experiments.phase1 import corpus_inputs as CI
from merlin_experiments.phase1 import session as PHASE1_SESSION
from merlin_experiments.phase1 import task_staging as TASK_STAGING
from merlin_experiments.phase1.context import load_context
from merlin_experiments.phase1.feedback import loop_grading as LOOP_GRADING
from merlin_experiments.phase1.feedback import qa as PHASE1_QA
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.providers import execution as PROVIDER
from merlin_experiments.phase2 import campaign
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.runner import fingerprint
from merlin_experiments.spec import SpecError

from merlin.benchharness import hash_tree
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.target_experiment import load_target_experiment

CHILD_GUARD = """
import importlib.abc, pathlib, socket, subprocess, sys
class DenyNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"_common", "run_agent_experiment", "run_baseline_qa_loop",
                        "run_global_perf_experiment", "perf_agent_stage"}:
            raise AssertionError("native controller import forbidden: " + fullname)
sys.meta_path.insert(0, DenyNative())
def denied(*args, **kwargs):
    raise AssertionError("qualification fixture cannot create listeners or native jobs")
socket.socket.bind = denied
_popen = subprocess.Popen
def generation_only(argv, *args, **kwargs):
    if not (isinstance(argv, list) and pathlib.Path(argv[0]).resolve() == pathlib.Path(sys.executable).resolve()
            and "-m" in argv and argv[argv.index("-m") + 1] == "merlin_experiments.phase0"):
        raise AssertionError("unexpected child command: " + repr(argv))
    return _popen(argv, *args, **kwargs)
subprocess.Popen = generation_only
"""


@pytest.fixture
def bridge(tmp_path, monkeypatch, request):
    def forbidden(*args, **kwargs):
        pytest.fail("reviewed input admission cannot launch tools or listeners")

    monkeypatch.setattr(socket.socket, "bind", forbidden)
    spec = importlib.util.spec_from_file_location(
        "reviewed_corpus_fixture", Path(__file__).with_name("reviewed_corpus_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    fixture = helper.build_phase0_handoff(tmp_path, copy_sources=False)
    hook = fixture["hooks"] / "sitecustomize.py"
    hook.write_text(CHILD_GUARD + hook.read_text())
    result = fixture["cli"]("run", fixture["definition"], "--phase", "0", "--run-dir", fixture["run"])
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads((fixture["run"] / "resolved-plan.json").read_text())
    assert plan["phases"]["0"]["argv"][1:3] == ["-m", "merlin_experiments.phase0"]
    record_before = json.loads((fixture["run"] / "orchestration.json").read_text())
    derived_sha = record_before["attempts"][0]["output_sha256"]
    resumed_phase0 = fixture["cli"]("resume", fixture["run"])
    assert resumed_phase0.returncode == 0, resumed_phase0.stdout + resumed_phase0.stderr
    assert json.loads((fixture["run"] / "orchestration.json").read_text())["attempts"] == record_before["attempts"]
    assert fingerprint(fixture["run"] / "phase0/capsules") == derived_sha
    golden = fixture["run"] / "phase0/capsules/isa/generated_member/golden.yaml"
    assert yaml.safe_load(golden.read_text())["golden_source"] == "merlin_tensor_int"
    # From this point onward every admission step is in-process and process-free.
    original_popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    for name in (
        "MERLIN_REPO_ROOT",
        "MERLIN_OUT_ROOT",
        "MERLIN_CONTRACT_DIR",
        "MERLIN_SCHEMAS_DIR",
        "MERLIN_BUNDLE_CAS",
    ):
        monkeypatch.setenv(name, fixture["environment"][name])
    report = release.prepare(fixture["run"], fixture["release"])
    assert report["state"] == "awaiting_operator_review"
    descriptor = Path(report["descriptor"])
    seal_path = fixture["release"] / "private/seal.json"
    with pytest.raises((SpecError, FileNotFoundError)):
        release.verify(seal_path, descriptor)
    sealed = release.seal(
        fixture["release"],
        expected_digest=report["review_digest"],
        reviewed_by="synthetic-test-operator",
        review_note="Explicit test-only review of derived bytes",
    )
    identity = release.verify(Path(sealed["seal"]), descriptor)
    target = load_target_experiment(descriptor)
    bundle_id = getattr(request, "param", "raw_baseline_public_v0")
    manifest = descriptor.parent / "input_bundles" / bundle_id / "input_bundle_manifest.yaml"
    bundle = yaml.safe_load(manifest.read_text())
    run = fixture["workspace"] / "phase1-inputs"
    run.mkdir()
    contract = Path(fixture["environment"]["MERLIN_CONTRACT_DIR"])
    shutil.copytree(contract, fixture["workspace"] / "merlin/contract")
    toolchain = fixture["workspace"] / "third_party/llvm-install"
    toolchain.mkdir(parents=True)
    (toolchain / "FIXTURE_ONLY.txt").write_text("No compiler is installed or executed by this admission test.\n")
    # Declare the fixture's L0 endpoint without executing it or claiming a grade.
    monkeypatch.setattr(CR, "qa_loop_adapters", lambda *_args, **_kwargs: {"L0": forbidden})
    monkeypatch.setattr(CR, "oracle_adapters", lambda *_args, **_kwargs: {"L0": forbidden})
    prepared = CI.prepare_bundle(run, target, manifest, bundle, contract=contract)
    workspace = run / "workspace"
    workspace.mkdir()
    BW.materialize_bundle_inputs(workspace, prepared.bundle, repo=fixture["workspace"])
    BW.require_snapshot_ownership(BW.verify_bundle_snapshot(workspace, prepared.bundle, repo=fixture["workspace"]))
    assert (
        release.verify_snapshot(Path(sealed["seal"]), descriptor, workspace, prepared.bundle, repo=fixture["workspace"])
        == identity
    )
    view = CI.resolve(
        workspace,
        prepared.bundle,
        prepared.corpus_record,
        repo=fixture["workspace"],
        reviewed_roots=tuple(target.graded_roots()),
    )
    return SimpleNamespace(
        fixture=fixture,
        target=target,
        descriptor=descriptor,
        seal=Path(sealed["seal"]),
        identity=identity,
        run=run,
        workspace=workspace,
        prepared=prepared,
        manifest=manifest,
        authored_bundle=bundle,
        view=view,
        original_popen=original_popen,
    )


def test_reviewed_derivation_reaches_real_v4_phase1_views_and_resume(bridge):
    b = bridge
    record = BW.snapshot_record(b.workspace)
    assert record["version"] == 4 and record["n_files"] > 0
    assert not b.view.public.is_relative_to(b.fixture["release"])
    assert sorted(path.parent.name for path in b.view.public.rglob("capsule.yaml")) == [
        "generated_member",
        "retained_member",
    ]
    before = fingerprint(b.view.public)
    # Resume retains run-owned corpus bytes, not the mutable pre-snapshot staging tree.
    staged = Path(b.prepared.corpus_record["staging_path"])
    staged.rename(staged.with_name("retired-staging"))
    environment = {
        "authored_bundle_manifest_sha256": b.prepared.authored_sha256,
        "bundle_manifest_sha256": b.prepared.effective_sha256,
        "public_corpus_input": b.prepared.corpus_record,
    }
    resumed = CI.prepare_bundle(
        b.run,
        b.target,
        b.manifest,
        b.authored_bundle,
        contract=Path(b.fixture["environment"]["MERLIN_CONTRACT_DIR"]),
        environment=environment,
    )
    view = CI.resolve(
        b.workspace,
        resumed.bundle,
        resumed.corpus_record,
        repo=b.fixture["workspace"],
        reviewed_roots=tuple(b.target.graded_roots()),
    )
    assert fingerprint(view.public) == before
    assert (
        release.verify_snapshot(b.seal, b.descriptor, b.workspace, resumed.bundle, repo=b.fixture["workspace"])
        == b.identity
    )


@pytest.mark.parametrize("mutation", ["reviewed", "frozen", "manifest"])
def test_reviewed_to_frozen_handoff_refuses_changed_bytes(bridge, mutation):
    b = bridge
    if mutation == "reviewed":
        path = next((b.fixture["release"] / "payload/corpus").rglob("capsule.yaml"))
    elif mutation == "frozen":
        path = next(b.view.public.rglob("capsule.yaml"))
    else:
        path = b.run / "input_bundle_manifest.yaml"
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b"\n# changed admission bytes\n")
    if mutation == "reviewed":
        with pytest.raises(SpecError):
            release.verify_snapshot(b.seal, b.descriptor, b.workspace, b.prepared.bundle, repo=b.fixture["workspace"])
    elif mutation == "frozen":
        with pytest.raises((ValueError, RuntimeError)):
            CI.resolve(
                b.workspace,
                b.prepared.bundle,
                b.prepared.corpus_record,
                repo=b.fixture["workspace"],
                reviewed_roots=tuple(b.target.graded_roots()),
            )
    else:
        environment = {
            "authored_bundle_manifest_sha256": b.prepared.authored_sha256,
            "bundle_manifest_sha256": b.prepared.effective_sha256,
            "public_corpus_input": b.prepared.corpus_record,
        }
        with pytest.raises(RuntimeError, match="bundle.*changed"):
            CI.prepare_bundle(
                b.run,
                b.target,
                b.manifest,
                b.authored_bundle,
                contract=Path(b.fixture["environment"]["MERLIN_CONTRACT_DIR"]),
                environment=environment,
            )


def test_reviewed_snapshot_is_not_sandbox_or_functional_qualification(bridge):
    # These are actual observed limits: no bwrap execution and no answer-mask probe.
    diagnostic = {
        "sandbox": "none",
        "isolation_violations": [],
        "bundle_input_snapshot": BW.snapshot_record(bridge.workspace),
    }
    predicates = {row.predicate for row in campaign._validate_clean_run(diagnostic, {})}
    assert {"sandbox_not_bwrap", "answer_mask_vacuous"} <= predicates
    with pytest.raises(campaign.CampaignGateError, match="UNWAIVABLE"):
        FI.inspect_stage_functional_run(
            bridge.run, "diagnostic", "a" * 64, waive=("sandbox_not_bwrap", "answer_mask_vacuous")
        )


@pytest.mark.parametrize("bridge", ["merlin_assisted_rtlchecks_public_v0"], indirect=True)
def test_reviewed_assisted_grant_executes_from_frozen_installed_source_in_bwrap(bridge, monkeypatch):
    """Exercise the actual mount boundary without launching an agent or grader."""
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is unavailable on this host")
    allowed = [entry["path"] for entry in bridge.prepared.bundle["allowed"]]
    tool_dir = next(path for path in allowed if path.endswith("/targetgen/rtl/"))
    tool = str(Path(tool_dir) / "gen_iface_irdl.py")
    assert Path(tool_dir).is_absolute() and Path(tool).is_file()
    hidden = bridge.fixture["release"] / "payload/corpus/hidden/private_member_identity/golden.yaml"
    assert hidden.is_file()
    argv = BW.base_argv(bridge.workspace, bridge.prepared.bundle, repo=bridge.fixture["workspace"])
    argv += [
        "--",
        "/bin/bash",
        "-c",
        'test -r "$1" && test -s "$1" && ! test -e "$2"',
        "qualification",
        tool,
        str(hidden),
    ]
    with monkeypatch.context() as scoped:
        scoped.setattr(subprocess, "Popen", bridge.original_popen)
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("bridge", ["merlin_assisted_rtlchecks_public_v0"], indirect=True)
def test_reviewed_derivation_formal_freeze_and_phase2_admission_share_exact_bytes(bridge, tmp_path, monkeypatch):
    from merlin_experiments.phase2.global_inputs import FrozenPhase1

    # Fresh assisted releases select the active installed Python package, not
    # the historical checkout-only merlin/python/merlin spelling.  The same
    # selected paths must remain visible to the task's tool declaration reader.
    selected = TASK_STAGING.granted_merlin_tools(bridge.manifest.parent)
    assert any(path.endswith("/targetgen/rtl/") for path in selected)
    assert not any(path.startswith("merlin/python/merlin/") for path in selected)

    authored_submission = _execute_reviewed_authoring(bridge, tmp_path, monkeypatch)
    spec = importlib.util.spec_from_file_location(
        "joined_formal_handoff", Path(__file__).with_name("test_phase1_formal_handoff.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    handoff = helper.build_handoff(
        tmp_path / "formal", monkeypatch, reviewed=bridge, authored_submission=authored_submission
    )
    assert handoff.events == ["public", "hidden"]
    assert hash_tree(handoff.run / "submission") == hash_tree(authored_submission)
    frozen = campaign.inspect_functional_run(handoff.runs, handoff.run.name, handoff.digest)
    assert not frozen.deviations
    assert frozen.public_score["n_capsules"] == 2
    assert {row["capsule"] for row in frozen.public_score["per_capsule"]} == {
        "generated_member",
        "retained_member",
    }
    binding = FrozenPhase1(handoff.runs, handoff.run.name, handoff.digest, (), 2, 2, ())
    admitted = binding.verify(handoff.run / "submission")
    assert admitted["public_passed"] == admitted["public_total"] == 2
    assert handoff.environment["bundle_input_snapshot"] == BW.snapshot_record(bridge.workspace)
    assert handoff.environment["public_corpus_input"] == bridge.prepared.corpus_record
    assert (
        release.verify_snapshot(
            bridge.seal,
            bridge.descriptor,
            bridge.workspace,
            bridge.prepared.bundle,
            repo=bridge.fixture["workspace"],
        )
        == bridge.identity
    )
    # Re-admission consumes the same frozen evidence, not a new certification.
    assert binding.verify(handoff.run / "submission") == admitted

    # The checkpoint owner now consumes this exact formal submission as its
    # optimization baseline. Agent, simulator and OS-isolation observations above
    # remain synthetic and must never be presented as hardware qualification.
    lifecycle_spec = importlib.util.spec_from_file_location(
        "joined_checkpoint_lifecycle", Path(__file__).with_name("test_checkpoint_lifecycle.py")
    )
    lifecycle_helper = importlib.util.module_from_spec(lifecycle_spec)
    lifecycle_spec.loader.exec_module(lifecycle_helper)
    lifecycle = lifecycle_helper.build_lifecycle(
        tmp_path / "phase2", monkeypatch, functional_run=frozen, target_name=bridge.target.target
    )
    result_path = lifecycle.run()
    result = json.loads(result_path.read_bytes())
    assert len(result["measurement_manifests"]) == 6
    assert result["statistics"]["status"] == "admitted"
    assert lifecycle.config.functional_run_id == handoff.run.name
    assert lifecycle.config.functional_submission_sha256 == handoff.digest
    events = list(lifecycle.events)
    assert lifecycle.run() == result_path
    assert lifecycle.events == events


def _execute_reviewed_authoring(bridge, tmp_path, monkeypatch):
    """Run the installed authoring owner over a real reviewed bundle with an inert provider."""
    calls = []

    def local_provider(workspace, run, *_args, **_kwargs):
        calls.append(workspace)
        submission = workspace / "submission"
        submission.mkdir(exist_ok=True)
        (submission / "compiler.py").write_text("# authored by the synthetic local provider\n")
        (submission / "manifest.yaml").write_text(
            json.dumps(
                {
                    "artifact_type": "mlir_oot_target_backend",
                    "target": bridge.target.target,
                    "language": "python",
                    "authoring": {"mode": "hand_curated", "generated_by_agent": False},
                    "integrity_exempt": False,
                    "entrypoints": {"tool": "compiler.py"},
                    "commands": {
                        name: {"argv": ["{tool}", "{input_mlir}"]}
                        for name in (
                            "parse",
                            "lower_interface_to_target",
                            "emit_command_buffer",
                            "lower_target_to_llvm",
                        )
                    },
                }
            )
        )
        transcript = run / "rounds/round_00.transcript.jsonl"
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_text(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {"model": "fixture", "content": [{"type": "text", "text": "Wrote fixture compiler."}]},
                }
            )
            + "\n"
            + json.dumps({"type": "result", "num_turns": 1, "usage": {"input_tokens": 1, "output_tokens": 1}})
            + "\n"
        )
        return 0, transcript

    options = parse_options(
        [
            "--run-id",
            "reviewed-local-authoring",
            "--arm",
            "merlin_assisted",
            "--model",
            "fixture",
            "--driver",
            "codex",
            "--sandbox",
            "none",
            "--allow-unsandboxed",
            "--no-oracle",
            "--skip-hidden",
            "--schedule",
            "rounds",
            "--max-rounds",
            "1",
            "--experiment",
            "realistic",
        ]
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(PROVIDER, "launch", local_provider)
        scoped.setattr(PHASE1_SESSION, "repo_sha", lambda **_kwargs: "synthetic-unversioned-workspace")

        def unavailable_fast_grade(*_args, **_kwargs):
            raise RuntimeError("synthetic fixture has no external fast-grade oracle")

        scoped.setattr(LOOP_GRADING, "fast_grade", unavailable_fast_grade)
        scoped.setattr(
            PHASE1_QA,
            "run",
            lambda *_args, **_kwargs: {
                "all_pass": False,
                "n_passed": 0,
                "n_capsules": 0,
                "per_capsule": [],
                "note": "synthetic fixture has no external oracle; no conformance verdict",
            },
        )
        context = load_context(bridge.descriptor, repo=bridge.fixture["workspace"])
        result = PHASE1.run(
            context,
            options,
            bundle_manifest=bridge.manifest,
            bundle_id=bridge.prepared.bundle["bundle_id"],
            oracle_timing=tmp_path / "timing.json",
            public_root=bridge.view.public,
        )
        submitted = hash_tree(context.runs / options.arm / options.run_id / "submission")
        resumed = PHASE1.run(
            context,
            replace(options, resume=True),
            bundle_manifest=bridge.manifest,
            bundle_id=bridge.prepared.bundle["bundle_id"],
            oracle_timing=tmp_path / "timing.json",
            public_root=bridge.view.public,
        )
    assert result == 1
    assert resumed == 1
    assert len(calls) == 1
    diagnostic_run = context.runs / options.arm / options.run_id
    assert hash_tree(diagnostic_run / "submission") == submitted
    summary = yaml.safe_load((diagnostic_run / "qa_loop_summary.yaml").read_text())
    assert summary["n_rounds"] == 1
    assert summary["authoring_complete"] is False
    assert summary["numeric_all_pass"] is False
    submission = calls[0] / "submission"
    assert (submission / "compiler.py").is_file()
    assert (submission / "manifest.yaml").is_file()
    return submission
