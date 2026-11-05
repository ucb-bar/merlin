"""Public model exploration uses exact selected sources and the original owner."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus import public_program_suite as S
from merlin_experiments.phase1.capability_inventory import _digest
from merlin_experiments.phase1.feedback import public_programs as F
from merlin_experiments.runner import fingerprint

from merlin.common.digest import sha256_file


def fixture(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import operator_exclusion as E
    from merlin_experiments.phase0 import public_programs as P

    root = tmp_path / "corpus"
    directory = root / "_perf" / "public-test"
    directory.mkdir(parents=True)
    source = directory / "capsule.interface.mlir"
    source.write_text("source-only-interface")
    (directory / "golden.yaml").write_text("private expected output")
    (directory / "source-capture-integer-reference.json").write_text("private independent reference")
    cap = dict(name="public-test", kind="model", label="dev", source_role="materialized_performance_capture",
               required_oracle_tiers=S.POLICY["tiers"], materialized_capture={},
               interface_mlir=source.name, performance=dict(gate=dict(source_execution_basis=dict(
                   schema="merlin.phase0.public_program_source_basis.v1", kind="public_program",
                   boundary="complete_program_call"))))
    cap["software_screen"] = dict(
        status="admitted", program_sha256=sha256_file(source), placements={"accelerator": 1},
        decisions=[dict(status="admitted", placement="accelerator", mlir_operation="linalg.matmul", count=1)],
    )
    (directory / "capsule.yaml").write_text(yaml.safe_dump(cap))
    protection = tmp_path / "synthetic-unit-only-exclusion.json"
    protection.write_text(json.dumps(dict(schema=E.SCHEMA, excluded_source_sha256=["a" * 64],
                                         excluded_contraction_sha256=[])))
    protection.chmod(0o600)
    # Isolate selection/transport tests from the independently tested projection
    # semantics. Real capture publication executes both unchanged owners.
    checks = []
    monkeypatch.setattr(P, "verify_written", lambda c, p: checks.append(("source", Path(p))))
    monkeypatch.setattr(E.OperatorPrivateExclusion, "check_member",
                        lambda self, p: checks.append(("protection", Path(p))))
    document = dict(schema=S.SCHEMA, target="fixture", policy=S.POLICY,
                    corpus=dict(path=str(root), sha256=fingerprint(root)),
                    members=[dict(member="_perf/public-test", capsule_sha256=_digest(cap))],
                    operator_exclusion=dict(path=str(protection), sha256=sha256_file(protection)))
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document))
    return path, document, cap, checks


def test_selected_public_source_and_answer_free_view(tmp_path, monkeypatch):
    path, document, _, checks = fixture(tmp_path, monkeypatch)
    assert S.read(path, target="fixture")[0] == document
    view = tmp_path / "public_program_sources"
    S.public_view(path, view, target="fixture")
    S.public_view(path, view, target="fixture", verify=True)
    assert {p.name for p in view.rglob("*") if p.is_file()} == {"suite.json", "capsule.interface.mlir"}
    assert {stage for stage, _ in checks} == {"source", "protection"}
    (view / "unexpected-reference.json").write_text("must refuse")
    with pytest.raises(ValueError, match="answer-bearing"):
        S.public_view(path, view, target="fixture", verify=True)


@pytest.mark.parametrize("change", ["path", "declaration", "tree", "tiers", "private", "count"])
def test_selection_refuses_ambiguous_or_mutated_source(tmp_path, monkeypatch, change):
    path, doc, cap, _ = fixture(tmp_path, monkeypatch)
    if change == "path":
        doc["members"][0]["member"] = "../outside"
    elif change == "declaration":
        doc["members"][0]["capsule_sha256"] = "b" * 64
    elif change == "tree":
        doc["corpus"]["sha256"] = "b" * 64
    elif change == "count":
        doc["members"] *= 10
    else:
        if change == "tiers":
            cap["required_oracle_tiers"] = ["L0", "L1", "L2", "L3"]
        else:
            cap["label"] = "hidden"
        (Path(doc["corpus"]["path"]) / "_perf/public-test/capsule.yaml").write_text(yaml.safe_dump(cap))
        doc["corpus"]["sha256"] = fingerprint(Path(doc["corpus"]["path"]))
        doc["members"][0]["capsule_sha256"] = _digest(cap)
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        S.read(path, target="fixture")


def test_program_requests_never_select_rtl_or_partial_obligations():
    assert F.request_policy() == "spike"
    assert F.request_policy("spike", "L0,L1,L2") == "spike"
    for sim, tiers, inspection in (("gsim", "", False), ("spike", "L2", False), (None, "", True)):
        with pytest.raises(ValueError):
            F.request_policy(sim, tiers, inspection=inspection)


@pytest.mark.parametrize("change", ["absent", "unknown", "source", "host_only", "counts"])
def test_source_publication_without_bound_device_placement_cannot_be_selected(tmp_path, monkeypatch, change):
    path, document, cap, _ = fixture(tmp_path, monkeypatch)
    screen = cap["software_screen"]
    if change == "absent":
        cap.pop("software_screen")
    elif change == "unknown":
        screen["status"] = "unknown"
    elif change == "source":
        screen["program_sha256"] = "a" * 64
    elif change == "host_only":
        screen["placements"] = {"host": 1}
        screen["decisions"][0]["placement"] = "host"
    else:
        screen["placements"]["accelerator"] = 2
    (Path(document["corpus"]["path"]) / "_perf/public-test/capsule.yaml").write_text(yaml.safe_dump(cap))
    document["members"][0]["capsule_sha256"] = _digest(cap)
    document["corpus"]["sha256"] = fingerprint(Path(document["corpus"]["path"]))
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        S.read(path, target="fixture")


def test_placement_producer_uses_original_screen_and_preserves_source_contract(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import program_admission as A

    path, document, cap, _ = fixture(tmp_path, monkeypatch)
    directory = Path(document["corpus"]["path"]) / "_perf/public-test"
    screen = cap.pop("software_screen")
    evidence = object()
    calls = []

    def original(c, d, *, target, evidence):
        calls.append((c, d, target, evidence))
        return screen

    monkeypatch.setattr(A, "screen_written", original)
    admitted = S.admit_placement(cap, directory, target="fixture", evidence=evidence)
    assert admitted == dict(cap, software_screen=screen)
    assert "software_screen" not in cap
    assert calls == [(cap, directory, "fixture", evidence)]
    screen["status"] = "unknown"
    with pytest.raises(ValueError, match="not admitted"):
        S.admit_placement(cap, directory, target="fixture", evidence=evidence)


def test_exact_owner_request_preserves_model_source_and_complete_tiers(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_model_activation as A
    from merlin_experiments.phase1.feedback import joint_performance as J
    from merlin_experiments.phase1.feedback import selfcheck

    from merlin.targetgen import capsule_runner as CR

    path, document, _, _ = fixture(tmp_path, monkeypatch)
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: fixture\ntoolchain: {sim_via: chipyard}\n")
    context = SimpleNamespace(descriptor=descriptor, target="fixture", repo=tmp_path, readback_policy=None)
    monkeypatch.setattr(F, "selected", lambda c: (path, *S.read(path, target="fixture")))
    monkeypatch.setattr(selfcheck, "_adapters", lambda *a, **k: ({"L2": object()}, "spike"))
    monkeypatch.setattr(A, "admit_grade", lambda **k: {})
    monkeypatch.setattr(J, "attach", lambda *a, **k: None)
    calls = []

    def owner(cap, submission, **kwargs):
        calls.append((cap, kwargs))
        assert cap["kind"] == "model" and cap["required_oracle_tiers"] == S.POLICY["tiers"]
        assert cap["__dir__"].startswith(str(tmp_path / "work/source_snapshot"))
        assert set(kwargs["oracle_adapters"]) == {"L2"}
        assert kwargs["adapter_managed_tiers"] == ("L2",)
        # Each program owns its run directory; the feedback reader opens runs/<suite>/<name>.
        assert kwargs["run_id"] == cap["name"]
        return dict(capsule=cap["name"], status="pass", tiers={t: {"status": "pass"} for t in S.POLICY["tiers"]})

    monkeypatch.setattr(CR, "run_capsule", owner)
    # The normal reader's full schema owner is separately covered; this fixture
    # only tests routing, not an invented numerical receipt.
    monkeypatch.setattr("merlin.targetgen.capsule_common.load_capsule", lambda d: dict(
        yaml.safe_load((d / "capsule.yaml").read_bytes()), __dir__=str(d)))
    submission = tmp_path / "submission"
    submission.mkdir()
    result = F._grade(submission, context=context, output=tmp_path / "work", timeout=10)
    assert result["passed"] and result["complete"] and len(calls) == 1
    assert result["original_conformance_credit"] is False and result["rtl_qualified"] is False
    assert document["corpus"]["sha256"] == fingerprint(Path(document["corpus"]["path"]))


def test_public_view_uses_verified_local_copies_never_historical_absolute_paths(tmp_path, monkeypatch):
    path, doc, cap, _ = fixture(tmp_path, monkeypatch)
    directory = Path(doc['corpus']['path']) / '_perf/public-test'
    (directory / 'capsule.pytorch.py').write_text('def get_model_and_inputs(): pass\n')
    cap['materialized_capture'] = dict(loader_path='/unselected/private.py',
                                      loader_sha256=sha256_file(directory / 'capsule.pytorch.py'))
    (directory / 'capsule.yaml').write_text(yaml.safe_dump(cap))
    doc['members'][0]['capsule_sha256'] = _digest(cap)
    doc['corpus']['sha256'] = fingerprint(Path(doc['corpus']['path']))
    path.write_text(json.dumps(doc))
    view = tmp_path / 'public'
    S.public_view(path, view, target='fixture')
    assert (view / 'public-test/capsule.pytorch.py').read_bytes() == (directory / 'capsule.pytorch.py').read_bytes()
    assert 'private.py' not in (view / 'suite.json').read_text()


def test_native_staging_retains_selector_custody_and_preserves_selected_bytes(tmp_path, monkeypatch):
    path, doc, _, _ = fixture(tmp_path, monkeypatch)
    staged = tmp_path / 'native-staged'
    binding = S.stage(path, staged, target='fixture')
    assert binding == dict(path=str(staged / 'selection.json'), sha256=sha256_file(staged / 'selection.json'))
    selected, members = S.read(Path(binding['path']), target='fixture')
    assert selected['corpus']['sha256'] == doc['corpus']['sha256']
    pin = selected['operator_exclusion']
    assert pin['path'] == str(staged / 'private/operator_exclusion.json')
    assert pin['sha256'] == doc['operator_exclusion']['sha256']
    assert 'private' not in Path(selected['corpus']['path']).parts
    assert list(members) == ['public-test']
    with pytest.raises(ValueError, match='mutate'):
        S.stage(path, Path(doc['corpus']['path']) / 'nested', target='fixture')
