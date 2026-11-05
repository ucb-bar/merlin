"""Generated guidance drives real checked scheduling through the existing knob transport.

The local owner executes neutral commands independently; there is no simulator,
agent, elapsed-cycle observation, or target-hardware qualification in this test.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.kernels.reconstruction import reconstruct
from merlin.mining.reconstruction_cli import main
from merlin.mining.reconstruction_experiments import workspace_tuning
from merlin.transforms.certificates import ir_digest


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def run_operator_steering_smoke(workspace):
    """Exercise a bounded generated request; retain actual plans and numerical evidence."""
    from merlin_experiments.phase1.brokers.selfcheck import _decorate_response, _submission_digest
    from merlin_experiments.phase1.brokers.selfcheck_tuning import apply_request
    from merlin_experiments.phase1.feedback.idle_tuning import IdleTuner
    from merlin_experiments.phase1.feedback.reference_experiments import (
        invocation,
        pending_requests,
        project,
        stage_guidance,
    )

    fixture_path = repo_root() / "merlin/tests/kernels/test_reconstruction.py"
    spec = importlib.util.spec_from_file_location("steering_neutral_owner_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    workspace = Path(workspace)
    workspace.mkdir(parents=True)
    archive = workspace / "run"
    archive.mkdir()
    for root in (workspace, archive):
        (root / "TASK.md").write_text("Use the ordinary compiler and selfcheck workflow.\n")
    compiler = workspace / "submission"
    compiler.mkdir()
    (compiler / "settings.json").write_text(json.dumps(dict(unroll=False, unrelated=3)))
    (compiler / "TUNING_KNOBS.json").write_text(json.dumps(dict(
        schema="merlin.tuning_knobs.v1", settings_file="settings.json",
        knobs=dict(unroll=dict(values=[False, True], mechanism="schedule.loop_transformed")),
    )))
    shape = (8, 6, 8)

    def compile_copy(root, geometry=shape):
        settings = json.loads((root / "settings.json").read_text())
        transforms = (fixture.LoopTransform("unroll", ("outer_m",), factor=2),) if settings["unroll"] else ()
        adapter = fixture.adapter(geometry, block_tiles=((1, 1, 1),), outer_orders=(("m", "n"),),
                                  loop_transform_sets=(transforms,))
        result = reconstruct(adapter, ("schedule.loop_transformed", "schedule.consumed"), max_candidates=1)
        path = result.paths[0]
        assert path.candidate.emission is not None
        checked = path.candidate.evaluation.result.artifact
        checked.revalidate()
        fixture.execute(checked.kernel, geometry)
        return adapter, path

    original, control = compile_copy(compiler)
    context = dict(
        compiler_sha256=_submission_digest(compiler), compiler_identity_basis="phase1_submission_sha256",
        source_sha256=ir_digest(original.source), environment_sha256=_sha(fixture_path.read_bytes()),
        reference_sha256=_sha(control.candidate.emission.data), visibility="reference_guided",
        evaluator=dict(identity="neutral-independent-command-owner", sha256=_sha(Path(__file__).read_bytes())),
        baseline=dict(submission="submission"), tuning=workspace_tuning(workspace),
        capsules=[dict(id="neutral-dense", goals=["schedule.loop_transformed"], split="training",
                       selfcheck_argv=["python3", "agent_selfcheck.py", "--submission", "{submission}",
                                       "--capsules", "neutral-dense", "--suite", "scalability", "--sim", "spike"])],
    )
    index = dict(gaps=[dict(
        kind="bounded_search_exhaustion", goals=["schedule.loop_transformed"],
        reason="the actual control schedule contains no consumed loop transformation",
        next_action="explore_registered_loop_transform", witnesses=[control.summary()],
    )])
    for name, value in (("context.json", context), ("input.json", index)):
        (workspace / name).write_text(json.dumps(value))
    out = workspace / "steering"
    assert main(["steer", "--input", str(workspace / "input.json"), "--context", str(workspace / "context.json"),
                 "--out", str(out)]) == 0
    selection = out / "selection.json"
    with invocation(selection, _sha(selection.read_bytes())):
        stage_guidance(workspace, archive)
        guidance = (workspace / "REFERENCE_EXPERIMENTS.json").read_bytes()
        assert guidance == (archive / "REFERENCE_EXPERIMENTS.json").read_bytes()
        assert b"schedule.loop_transformed" in guidance
        ready = pending_requests(context["compiler_sha256"])
        assert any(row["role"] == "variant" for row in ready)
        tuner = IdleTuner(workspace)
        candidates = tuner.candidates(context["compiler_sha256"], None)
        candidate = next(row for row in candidates if "reference_experiment" in row)
        channel = workspace / ".qa_channel"
        channel.mkdir()
        job = dict(candidate=candidate, model=None, bundle="neutral dense region", suggested=[])
        rid = tuner.submit(channel, job, ["neutral-dense"], context["compiler_sha256"])
        request = json.loads((channel / f"req_{rid}.json").read_text())
        assert request["tuning_overrides"] == {"unroll": True}
        copied = workspace / "private-snapshot" / "submission"
        shutil.copytree(compiler, copied)
        assert _submission_digest(copied) == context["compiler_sha256"]
        applied = apply_request(copied, request, digest=_submission_digest)
        _, changed = compile_copy(copied)
        assert changed.satisfied
        assert changed.candidate.emission.data != control.candidate.emission.data
        assert json.loads((copied / "settings.json").read_text())["unrelated"] == 3
        assert json.loads((compiler / "settings.json").read_text())["unroll"] is False
        assert applied["_tuning_applied"]["sha256"] == _submission_digest(copied)
        response = _decorate_response(json.dumps(dict(
            all_pass=False, sim="spike", n_capsules=1,
            per_capsule=[dict(capsule="neutral-dense", status="incomplete", numerical_execution_passed=True)],
        )), applied, rid, context["compiler_sha256"], started_ns=1, completed_ns=2)
        observed = response["reference_experiment_result"]
        assert observed["numerical_status"] == "pass"
        assert observed["mechanism_status"] == "unknown" and observed["task_complete"] is False
        assert not any(row["request_id"] == observed["request_id"]
                       for row in pending_requests(context["compiler_sha256"]))
        assert not any("reference_experiment" in row
                       for row in tuner.candidates(context["compiler_sha256"], None))
        (workspace / "feedback.json").write_text(json.dumps(response, indent=2) + "\n")
        # Source drift cannot apply an old packet to a newly edited compiler.
        assert pending_requests(_submission_digest(copied)) == ()
        prior = project({}, current_compiler_sha256=_submission_digest(copied))
        assert prior["status"] == "prior_reference_guidance" and prior["required_selfchecks"] == []
        assert "schedule.loop_transformed" in prior["pending_goals"]
        feedback = project({}, current_compiler_sha256=context["compiler_sha256"])
        assert feedback is not None
    outputs = {}
    for name, path in (("control", control), ("changed", changed)):
        directory = workspace / "compiled" / name
        directory.mkdir(parents=True)
        (directory / "commands.txt").write_bytes(path.candidate.emission.data)
        plan = path.candidate.evaluation.result.artifact.realization["schedule_ir"]
        (directory / "schedule.mlir").write_text(plan["text"])
        outputs[name] = str(directory)
    # Freeze the selected settings before checking new, independently compiled geometries.
    generalization = []
    for split, geometry in (("validation", (9, 7, 9)), ("held_out", (4, 9, 5))):
        _, path = compile_copy(copied, geometry)
        assert path.satisfied
        generalization.append(dict(split=split, shape_mkn=geometry,
                                   program_sha256=_sha(path.candidate.emission.data), numerical="pass"))
    report = dict(
        schema="operator.reference_steering_smoke.v1", request_id=rid,
        actual_broker_request=str(channel / f"req_{rid}.json"),
        control_program_sha256=_sha(control.candidate.emission.data),
        changed_program_sha256=_sha(changed.candidate.emission.data),
        changed_schedule_sha256=changed.candidate.evaluation.result.artifact.kernel.digest(),
        schedule_dialect_consumed="schedule.consumed" in changed.observation.established,
        joined_observation=observed, repeat_request_suppressed=True,
        author_guidance_staged=True, source_edits_retain_prior_goals=True,
        numerical="independent neutral-command execution equals dense integer reference",
        timing="unknown", hardware_qualification=False, original_submission_unchanged=True,
        compiled_artifacts=outputs, generalization=generalization,
    )
    (workspace / "operator-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def test_generated_request_consumes_schedule_and_changes_actual_numerically_checked_emission(tmp_path):
    report = run_operator_steering_smoke(tmp_path / "operator")
    assert report["schedule_dialect_consumed"]
    assert report["control_program_sha256"] != report["changed_program_sha256"]
    assert report["timing"] == "unknown"
