"""Actual installed CLI and programmatic admission share objective policy."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import __main__ as entrypoint
from merlin_experiments.phase1 import context, controller, session
from merlin_experiments.phase1.options import RunOptions, parse_options

INVALID = (
    ({"max_wall_s": 0}, ["--max-wall-s", "0"]),
    ({"max_wall_s": -1}, ["--max-wall-s", "-1"]),
    ({"schedule": "rounds"}, ["--schedule", "rounds"]),
    ({"continuous": True}, ["--continuous"]),
    ({"plateau_rounds": 2}, ["--plateau-rounds", "2"]),
    ({"qualify_submission": "/selected"}, ["--qualify-submission", "/selected"]),
    ({"seal_current": True}, ["--seal-current"]),
)


COVERAGE = "--objective device-coverage --schedule continuous --max-wall-s 10 --plateau-rounds 0".split()


def cli_inputs(tmp_path):
    manifest = tmp_path / "bundle.yaml"
    manifest.write_text("bundle_id: synthetic\n")
    paths = [tmp_path / "descriptor.yaml", tmp_path, manifest, tmp_path / "timing.json"]
    flags = ["--descriptor", "--repo", "--bundle-manifest", "--oracle-timing"]
    return ["--run-id", "probe", "--bundle", "synthetic"] + [a for f, p in zip(flags, paths) for a in (f, str(p))]


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("readback", [None, "out_bin_v1"])
def test_installed_entrypoint_binds_candidate_transport_before_treatment_capture(
    tmp_path, monkeypatch, selected, readback
):
    from merlin_experiments.phase1 import candidate_execution
    from merlin_experiments.phase1.feedback import qa, rtlchecks

    class FactoryBoundary(BaseException):
        pass

    policy = tmp_path / "candidate-policy.json"
    transport = object()
    loaded = []

    def load(descriptor, *, repo, **kwargs):
        loaded.append(kwargs)
        kwargs = dict(kwargs)
        if "candidate_policy" in kwargs:
            kwargs["candidate_policy"] = Path(kwargs["candidate_policy"]).absolute()
        return context.InvocationContext(repo, descriptor, repo, "synthetic", repo, repo, repo, (), **kwargs)

    monkeypatch.setattr(context, "load_context", load)
    monkeypatch.setattr(rtlchecks, "prepare_arguments", lambda args, **kwargs: args)
    monkeypatch.setattr(qa, "_loop_target_sim_via", lambda ctx: (ctx.target, "chipyard"))
    capsules = tmp_path / "public"
    (capsules / "one").mkdir(parents=True)
    (capsules / "one/capsule.yaml").write_text("required_oracle_tiers: [L2]\n")

    def selected_transport(ctx):
        assert ctx.candidate_policy == policy
        return transport

    monkeypatch.setattr(candidate_execution, "transport", selected_transport)

    def factory(target, sim_via, **kwargs):
        assert target == "synthetic" and sim_via == "chipyard"
        assert kwargs.get("candidate_transport") is (transport if selected else None)
        assert ("candidate_transport" in kwargs) == selected
        if readback:
            assert kwargs["readback_policy"].record()["transport"] == readback
        else:
            assert "readback_policy" not in kwargs
        raise FactoryBoundary

    monkeypatch.setattr(qa.CR, "qa_checkpoint_adapters", factory)

    def admit(ctx, options, *, treatment, **kwargs):
        assert ctx.candidate_policy == (policy if selected else None)
        # Execute the real treatment callback and real normal QA path, rather than the initial
        # fast-grade path which uses the controller's separately replaced context.
        treatment.qa_runner("submission", str(capsules), tmp_path / "runs", {"public"}, False, 900)
        pytest.fail("normal QA did not reach the selected oracle factory")

    monkeypatch.setattr(controller, "run", admit)
    command = cli_inputs(tmp_path) + ["--treatment", "rtlchecks"]
    if selected:
        command += ["--candidate-policy", str(policy)]
    if readback:
        command += ["--readback-policy", readback]
    with pytest.raises(FactoryBoundary):
        entrypoint.main(command)
    assert loaded == [
        {
            **({"readback_policy": readback} if readback else {}),
            **({"candidate_policy": str(policy)} if selected else {}),
        }
    ]


@pytest.mark.parametrize("updates,args", INVALID)
def test_real_installed_cli_refuses_before_context_or_treatment_inputs(tmp_path, monkeypatch, capsys, updates, args):
    from merlin_experiments.phase1.feedback import rtlchecks

    def forbidden(*a, **k):
        pytest.fail("invalid objective reached resource discovery or admission")

    monkeypatch.setattr(context, "load_context", forbidden)
    monkeypatch.setattr(controller, "run", forbidden)
    monkeypatch.setattr(rtlchecks, "prepare_arguments", forbidden)
    command = cli_inputs(tmp_path) + COVERAGE + ["--treatment", "rtlchecks"] + args
    with pytest.raises(SystemExit) as error:
        entrypoint.main(command)
    assert error.value.code == 2
    assert "device-coverage requires" in capsys.readouterr().err


COVERAGE_OPTIONS = dict(objective="device-coverage", schedule="continuous", max_wall_s=10, plateau_rounds=0)


@pytest.mark.parametrize(
    "updates,match",
    [(dict(COVERAGE_OPTIONS, **updates), "device-coverage requires") for updates, _ in INVALID]
    + [
        (dict(COVERAGE_OPTIONS, max_wall_s=budget), "device-coverage requires")
        for budget in (True, 1.5, "10", float("nan"), float("inf"), -float("inf"))
    ]
    + [(dict(objective=objective), "unknown authoring objective") for objective in ("unknown", True, [], {})],
)
def test_programmatic_construction_and_session_refuse_identical_invalid_objectives(updates, match):
    """Invalid coverage objectives, non-positive-integer budgets and unknown objectives, both entry points."""
    values = dataclasses.asdict(parse_options(["--run-id", "probe"], environ={}))
    values.update(updates)
    with pytest.raises(ValueError, match=match):
        RunOptions(**values)
    with pytest.raises(ValueError, match=match):
        session.validate_options(SimpleNamespace(**values))


@pytest.mark.parametrize("coverage", [False, True])
def test_real_installed_cli_preserves_legacy_default_and_forwards_valid_coverage(tmp_path, monkeypatch, coverage):
    observed = []
    monkeypatch.setattr(context, "load_context", lambda *a, **k: SimpleNamespace(target="synthetic"))

    def admit(selected_context, options, **kwargs):
        observed.append(options)
        return 77

    monkeypatch.setattr(controller, "run", admit)
    args = cli_inputs(tmp_path)
    if coverage:
        args += COVERAGE
    assert entrypoint.main(args) == 77
    assert len(observed) == 1 and observed[0].objective == ("device-coverage" if coverage else "correctness")
    if not coverage:
        defaults = parse_options(["--run-id", "probe"], environ={})
        assert observed[0].schedule == defaults.schedule and observed[0].max_wall_s == defaults.max_wall_s
