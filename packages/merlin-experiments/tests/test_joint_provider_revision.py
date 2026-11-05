"""Synthetic source/ancestry tests; no timing accuracy or target execution credit."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_calibration_admission as A
from merlin_experiments.phase1 import joint_provider_revision as P


@pytest.fixture
def case(tmp_path):
    def put(name, value):
        p = tmp_path / name
        p.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

    provider = put("provider.so", b"synthetic provider")
    source = put("source.json", {"sha256": "a" * 64})
    before = dict(
        domain={"feature_engine_sha256": "a" * 64},
        libraries=["functional", "old-provider"],
        target={"prices": {"ws": 3}, "facts": {"dim": 16}},
        scope={"kind": "counter"},
        diagnostics={"scoped_resources": {"enabled": True}},
    )
    after = copy.deepcopy(before)
    after["libraries"][1] = provider["sha256"]
    old_mp, new_mp = put("old-model.json", before), put("model.json", after)
    old_spec = dict(
        model=old_mp["path"],
        model_sha256=old_mp["sha256"],
        libraries=["functional.so", "old.so"],
        observer="same-spike",
        source_manifest=source["path"],
        functional_argv=["functional"],
        timing_selection={"window": "same"},
    )
    new_spec = dict(
        old_spec, model=new_mp["path"], model_sha256=new_mp["sha256"], libraries=["functional.so", provider["path"]]
    )
    old, new = put("old-runtime.json", old_spec), put("runtime.json", new_spec)
    build = put(
        "build.json",
        dict(
            provider=provider,
            parameters_unchanged=True,
            calibration_promotion_authorized=False,
            source_commit="f" * 40,
            sources=[put("source.hpp", b"synthetic source")],
        ),
    )
    params = [{"path": ["target", "prices", "ws"]}]
    profile = put("profile.json", {"parameters": params})
    denial_row = dict(
        model_sha256=old_mp["sha256"], price_paths=[params[0]["path"]], reason="public structural gap", evidence=[build]
    )
    previous_denial = put(
        "previous-denial.json", dict(schema="merlin.native_calibration_admission_review.v1", deny=[denial_row])
    )
    denial = put(
        "denial.json",
        dict(
            schema="merlin.native_calibration_admission_review.v1",
            deny=[denial_row, dict(denial_row, model_sha256=new_mp["sha256"])],
        ),
    )
    elf = put("case.elf", b"synthetic ELF")
    stdout = put("case.stdout", b"synthetic outputs")

    def report(mp, cycles):
        return dict(
            model_sha256=mp["sha256"],
            execution_completed=True,
            timing_complete=True,
            unresolved=[],
            functional_status="PASS",
            raw_functional_exit_code=0,
            elf_sha256=elf["sha256"],
            input_sha256="inputs",
            scope_sha256="scope",
            scope_binding_sha256="binding",
            estimated_cycles=cycles,
        )

    row = dict(
        baseline_report=put("old-report.json", report(old_mp, 146)),
        report=put("report.json", report(new_mp, 129)),
        elf=elf,
        baseline_stdout=stdout,
        stdout=stdout,
        physical_evidence=put("physical.json", {"synthetic": True}),
    )
    replay = put(
        "replays.json",
        dict(
            schema="merlin.provider_public_replays.v1",
            private_labels_used=False,
            qualification_issued=False,
            rows=[row],
        ),
    )
    evidence = dict(
        schema=P.SCHEMA,
        status="provisional_untuned",
        previous_runtime=old,
        runtime=new,
        fit_authorized=False,
        promotion_authorized=False,
        previous_model=old_mp,
        model=new_mp,
        observer_source=source,
        provider_build=build,
        public_replays=replay,
        previous_denial=previous_denial,
        continued_denial=denial,
        calibration_profile=profile,
    )
    return SimpleNamespace(**locals())


def verify(c):
    P.verify(
        SimpleNamespace(model_json=json.dumps(c.before)),
        SimpleNamespace(model_json=json.dumps(c.after)),
        c.evidence,
        previous=c.old,
        current=c.new,
    )


def test_exact_provider_correction_is_admissible_but_not_fit_authority(case):
    verify(case)


@pytest.mark.parametrize("field", ["prices", "facts", "scope", "diagnostics"])
def test_other_model_changes_refused(case, field):
    if field in ("prices", "facts"):
        case.after["target"][field]["new"] = 1
    else:
        case.after[field]["changed"] = True
    with pytest.raises(ValueError, match="prices, facts or model scope"):
        verify(case)


@pytest.mark.parametrize("field", ["build", "source", "replay", "denial"])
def test_missing_or_changed_evidence_refused(case, field):
    if field == "build":
        Path(case.provider["path"]).write_bytes(b"mutated binary")
    elif field == "source":
        Path(case.source["path"]).write_bytes(b"{}")
    elif field == "replay":
        Path(case.row["report"]["path"]).unlink()
    else:
        case.evidence["continued_denial"] = case.previous_denial
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        verify(case)


def test_admission_follows_correction_and_requires_current_denial(case, monkeypatch):
    from merlin_experiments.phase1 import joint_scoped_revision as S

    # Origin validation itself is covered by the original observer-source tests.
    calls = []
    monkeypatch.setattr(S, "verify", lambda *args, **kwargs: calls.append(kwargs))
    origin_review = case.put(
        "origin.json",
        dict(
            schema=S.SCHEMA,
            runtime=case.old,
            model=case.old_mp,
            continued_denial=case.previous_denial,
            calibration_profile=case.profile,
        ),
    )
    origin = dict(
        kind="scoped_observer_diagnostics",
        previous={"path": "original", "sha256": "original"},
        runtime=case.old,
        evidence=origin_review,
    )
    revision = dict(
        schema="merlin.runtime_implementation_revision.v1",
        kind=P.KIND,
        previous=case.old,
        runtime=case.new,
        evidence=case.put("review.json", case.evidence),
        review_id="synthetic-review",
    )
    selection = dict(runtime=case.new, runtime_revisions=[origin, revision])
    plan = dict(runtime=case.new, selection_document=selection, parameters=case.params)
    monkeypatch.setattr(
        P.PERF,
        "_runtime",
        lambda pin: SimpleNamespace(model_json=json.dumps(case.after if pin == case.new else case.before)),
    )
    monkeypatch.setenv(A.ENV, json.dumps(case.previous_denial))
    with pytest.raises(ValueError, match="active admission review differs"):
        A.assert_allowed(plan)
    monkeypatch.setenv(A.ENV, json.dumps(case.denial))
    with pytest.raises(ValueError, match="blocked_structural_model_admission"):
        A.assert_allowed(plan)
    assert len(calls) == 2
    revision["kind"] = "implementation_correction"
    with pytest.raises(ValueError, match="unreviewed provider ancestry"):
        A.assert_allowed(plan)


@pytest.mark.parametrize("field", ["observer", "functional_argv", "timing_selection", "source_manifest"])
def test_runtime_fact_and_boundary_changes_refused(case, field):
    changed = dict(case.new_spec)
    changed[field] = "foreign"
    case.new = case.put("runtime.json", changed)
    case.evidence["runtime"] = case.new
    with pytest.raises(ValueError, match="observer, facts or runtime scope"):
        verify(case)


def test_runtime_ancestry_dispatches_evidenced_provider_kind(case, monkeypatch):
    from merlin_experiments.phase1 import joint_runtime_revision as R

    original = dict(runtime=case.old, members={}, timing_epochs=[])
    current = dict(
        original,
        runtime=case.new,
        runtime_revisions=[
            dict(
                schema="merlin.runtime_implementation_revision.v1",
                kind=P.KIND,
                previous=case.old,
                runtime=case.new,
                evidence=case.put("review.json", case.evidence),
                review_id="synthetic-review",
            )
        ],
    )
    monkeypatch.setattr(
        P.PERF,
        "_runtime",
        lambda pin: SimpleNamespace(model_json=json.dumps(case.after if pin == case.new else case.before)),
    )
    R.validate(original, current)
