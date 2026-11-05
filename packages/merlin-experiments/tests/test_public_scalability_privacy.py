"""Synthetic checks keep private references outside the author response channel."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import (
    case_declarations,
    compiler_profile_runtime,
    performance_brief,
    scratch_retention,
)
from merlin_experiments.phase1.feedback import (
    public_scalability as suite,
)
from merlin_experiments.phase1.feedback import (
    raw_selfcheck_evidence as evidence,
)


@pytest.fixture
def staged_request(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    channel = workspace / ".qa_channel"
    channel.mkdir(parents=True)
    private = tmp_path / "operator" / "evidence"
    environment = evidence.broker_environment(private, workspace, environ={})
    for key in (evidence.ROOT_ENV, evidence.WORKSPACE_ENV):
        monkeypatch.setenv(key, environment[key])
    root = tmp_path / "operator" / "suite"
    (root / "cases").mkdir(parents=True)
    context = SimpleNamespace(target="fixture-device")
    args = SimpleNamespace(
        out=str(channel / "response.json"), submission=str(workspace / "submission"),
        sim="spike", tiers="L0,L1,L2", capsules="case0", timeout=30,
        workers=2, contract=None, compiler_bundle_manifest=None,
    )
    monkeypatch.setattr(suite, "selected", lambda selected: (root, {"cases": ["case0"]}))
    monkeypatch.setattr(case_declarations, "register", lambda *a, **kw: None)
    monkeypatch.setattr(performance_brief, "attach", lambda *a, **kw: None)
    retained = []
    monkeypatch.setattr(scratch_retention, "raw_runs", lambda path: retained.append(path))
    return args, context, workspace, private, retained


@pytest.mark.parametrize("passed", [True, False])
def test_public_response_never_contains_reference_scratch(staged_request, monkeypatch, capsys, passed):
    args, context, workspace, private, retained = staged_request
    calls = []

    @contextmanager
    def profile(manifest, **kwargs):
        path = kwargs["work_root"]
        path.mkdir()
        (path / "private-profile-input").write_text("PROFILE_INPUT_CANARY")
        print("PROFILE_LOG_CANARY")
        yield
        print("PROFILE_EXIT_CANARY")

    def grade(submission, **kwargs):
        calls.append(kwargs)
        output = kwargs["output"]
        subset = output / "subset" / "case0"
        subset.mkdir(parents=True)
        (subset / "golden.npz").write_bytes(b"PRIVATE_REFERENCE_CANARY")
        (subset / "golden.yaml").write_text("output: PRIVATE_REFERENCE_CANARY\n")
        print("GRADER_LOG_CANARY")
        return {
            "passed": passed, "complete": passed, "observed": {
                "case0": {"status": "pass" if passed else "fail", "tiers": {"L2": "pass" if passed else "fail"},
                          "numeric": {"private_reference": "PRIVATE_REFERENCE_CANARY"}},
            },
            "score": {},
        }

    monkeypatch.setattr(compiler_profile_runtime, "selfcheck_execution", profile)
    monkeypatch.setattr(suite, "grade", grade)
    assert suite.cli(args, context=context) == (0 if passed else 1)
    public = json.loads(Path(args.out).read_text())
    assert public["all_pass"] is passed
    assert public["cases"][0]["tiers"] == {"L2": "pass" if passed else "fail"}
    assert calls[0]["names"] == {"case0"}
    assert calls[0]["max_workers"] == 2
    assert calls[0]["timeout"] == 30
    assert retained == [calls[0]["output"].parent / "profile", calls[0]["output"]]
    assert calls[0]["output"].parent.parent == private
    assert not calls[0]["output"].is_relative_to(workspace)
    assert list(workspace.rglob("golden.npz")) == []
    assert sorted(p.name for p in (workspace / ".qa_channel").iterdir()) == ["response.json"]
    assert list(private.rglob("golden.npz"))[0].read_bytes() == b"PRIVATE_REFERENCE_CANARY"
    log = list(private.rglob("grader.log"))[0].read_text()
    assert all(value in log for value in ["PROFILE_LOG_CANARY", "GRADER_LOG_CANARY", "PROFILE_EXIT_CANARY"])
    printed = capsys.readouterr()
    assert json.loads(printed.out) == public
    assert printed.err == ""
    assert "CANARY" not in Path(args.out).read_text()


@pytest.mark.parametrize("selection", ["absent", "partial", "visible", "symlink"])
def test_private_staging_refuses_before_grading(staged_request, monkeypatch, selection):
    args, context, workspace, private, retained = staged_request
    monkeypatch.setattr(suite, "grade", lambda *a, **kw: pytest.fail("must refuse before grading"))
    monkeypatch.setattr(
        compiler_profile_runtime, "selfcheck_execution", lambda *a, **kw: pytest.fail("no profile entry"),
    )
    if selection in ("absent", "partial"):
        monkeypatch.delenv(evidence.ROOT_ENV)
        if selection == "absent":
            monkeypatch.delenv(evidence.WORKSPACE_ENV)
    elif selection == "visible":
        monkeypatch.setenv(evidence.ROOT_ENV, str(workspace / "scratch"))
    else:
        private.parent.mkdir(parents=True, exist_ok=True)
        private.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(ValueError, match="private|canonical|disjoint"):
        suite.cli(args, context=context)
    assert not Path(args.out).exists()
    assert retained == []
    assert list(workspace.rglob("golden.npz")) == []
    assert not (workspace / "scratch").exists()


def test_failed_private_grade_retains_evidence_without_public_partial_result(staged_request, monkeypatch, capsys):
    args, context, workspace, private, retained = staged_request

    @contextmanager
    def profile(*a, **kw):
        yield

    def fail(*a, **kw):
        root = kw["output"]
        root.mkdir()
        (root / "golden.npz").write_bytes(b"FAILED_PRIVATE_REFERENCE")
        print("FAILED_PRIVATE_LOG")
        raise RuntimeError("synthetic grade failure")

    monkeypatch.setattr(compiler_profile_runtime, "selfcheck_execution", profile)
    monkeypatch.setattr(suite, "grade", fail)
    with pytest.raises(RuntimeError, match="synthetic grade failure"):
        suite.cli(args, context=context)
    assert not Path(args.out).exists()
    assert list(workspace.rglob("golden.npz")) == []
    assert len(list(private.rglob("golden.npz"))) == 1
    assert list(private.rglob("grader.log"))[0].read_text() == "FAILED_PRIVATE_LOG\n"
    assert capsys.readouterr().out == ""
