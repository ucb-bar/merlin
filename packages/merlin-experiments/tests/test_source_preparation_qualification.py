"""Versioned domain transport with real native records and isolated authorities.

Real source preparation remains incomplete and refuses before candidate work.
Positive qualification wiring uses explicitly synthetic origin/runtime/static
and preparation facets; it is not experiment or hardware qualification.
"""

import copy
import importlib
import importlib.util
import inspect
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import component_coverage as COVERAGE
from merlin_experiments.phase0 import source_preparation_release as P
from merlin_experiments.phase0.component_generation import digest
from merlin_experiments.phase1 import component_generation_admission as A
from merlin_experiments.phase1 import component_qualification as Q
from merlin_experiments.phase1 import component_qualification_domain as D
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common.jsonio import canonical_json

ACTUAL_VERIFY_ORIGIN = Q._verify_origin
ACTUAL_VERIFY_RUNTIME = Q._verify_runtime


def _fixtures(name):
    spec = importlib.util.spec_from_file_location("qualification_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _fixtures("test_component_qualification")
S = _fixtures("test_source_preparation_release")
automatic, independent, selected, inputs, prepared = S.automatic, S.independent, S.selected, S.inputs, S.prepared


@pytest.fixture
def domain(tmp_path, monkeypatch):
    owner = tmp_path / "isolated-qualification"
    owner.mkdir()
    return inspect.unwrap(F.domain)(owner, monkeypatch)


def _origin(preparation):
    return SimpleNamespace(
        inputs=SimpleNamespace(
            source_preparation=preparation, hardware=preparation.hardware, software=preparation.software
        )
    )


def test_actual_incomplete_original_source_preparation_refuses_before_grader(prepared, domain, monkeypatch):
    monkeypatch.setitem(sys.modules, COVERAGE.__name__, COVERAGE)
    values, _, _ = domain
    values.update(corpus_root=prepared.root, compiler_origin=_origin(prepared))
    calls = F._grade(domain, monkeypatch)
    original = prepared.record()
    assert original["mandatory_source_blockers"]
    with pytest.raises(StageGateError, match="unresolved original mandatory"):
        Q.qualify_component_compiler(**values)
    assert not calls and not values["evidence_root"].exists()
    assert prepared.record() == original
    assert all(row["candidate_verdict"] == "not_evaluated" for row in original["candidate_predicates"])


@pytest.mark.parametrize("defect", ["copied", "json", "root", "hardware", "software"])
def test_actual_source_domain_requires_original_live_selection(prepared, tmp_path, defect):
    preparation = copy.copy(prepared) if defect == "copied" else prepared
    origin = _origin(preparation)
    root = prepared.root
    if defect == "json":
        origin.inputs.source_preparation = prepared.record()
    elif defect == "root":
        root = tmp_path
    elif defect in {"hardware", "software"}:
        setattr(origin.inputs, defect, object())
    with pytest.raises(StageGateError):
        D.reopen_domain(root, origin=origin)


def _diagnostic_selection(domain, monkeypatch):
    """Isolate domain wiring; this deliberately unissued owner cannot admit a run."""
    values, _, report = domain
    development = copy.deepcopy(report["obligations"][0])
    development.update(id="development", cohort="development")
    member = development["members"][0]
    member.update(name="development", member="cases/development")
    shutil.copytree(values["corpus_root"] / "cases/functional_guard", values["corpus_root"] / member["member"])
    capsule_path = values["corpus_root"] / member["member"] / "capsule.yaml"
    capsule = json.loads(capsule_path.read_bytes())
    capsule["name"] = "development"
    capsule_path.write_text(json.dumps(capsule))
    report["obligations"].insert(0, development)
    report.update(status="source_prepared", plan={"sha256": "6" * 64})
    for row in report["obligations"]:
        row["state"] = "source_generated"
        for original_member in row["members"]:
            original_member["state"] = "source_generated"
    monkeypatch.setattr(sys.modules[COVERAGE.__name__], "build_guard_link", COVERAGE.build_guard_link)
    record = {
        "schema": P.SCHEMA,
        "coverage_sha256": report["sha256"],
        "original_required_ids": [row["id"] for row in report["obligations"]],
        "original_mandatory_ids": [row["id"] for row in report["obligations"] if row["mandatory"]],
        "mandatory_source_blockers": [],
        "requirements": [
            {
                "original_id": row["id"],
                "original_requirement_sha256": digest(row),
                "mandatory": row["mandatory"],
                "cohort": row["cohort"],
            }
            for row in report["obligations"]
        ],
        "candidate_predicates": [
            {
                "original_id": row["id"],
                "candidate_verdict_phase": 1,
                "candidate_predicate": "original_compiler_requirement",
                "candidate_verdict": "not_evaluated",
            }
            for row in report["obligations"]
        ],
    }
    preparation = P.SourcePreparation(
        values["corpus_root"],
        values["corpus_root"] / "synthetic-coverage.json",
        object(),
        object(),
        None,
        P.SourcePreparationBudget(10000),
        (),
        (),
        canonical_json(record),
        values["corpus_root"] / "synthetic-preparation.json",
    )
    with pytest.raises(ValueError, match="actual live"):
        preparation.verify()
    values["compiler_origin"] = _origin(preparation)

    def isolated_inputs(root, *, preparation, hardware, software):
        assert root == values["corpus_root"]
        assert preparation is values["compiler_origin"].inputs.source_preparation
        assert hardware is preparation.hardware and software is preparation.software
        return report

    # Only this positive transport fixture bypasses the actual issuance leg.
    # The native records below prove its evidence plumbing, not runtime roles.
    monkeypatch.setattr(A, "verify_generation_inputs", isolated_inputs)
    return preparation, record


def test_v2_grades_and_replays_every_original_mandatory_member_with_native_records(domain, monkeypatch):
    preparation, record = _diagnostic_selection(domain, monkeypatch)
    calls = F._grade(domain, monkeypatch)
    qualification = Q.qualify_component_compiler(**domain[0])
    document = qualification.verify()
    assert len(calls) == 1 and document["schema"] == D.SOURCE_RECEIPT_SCHEMA
    assert document["source_domain"]["source_preparation_sha256"] == preparation.sha256
    assert document["source_domain"]["original_required_ids"] == record["original_required_ids"]
    assert document["source_domain"]["candidate_predicates"] == record["candidate_predicates"]
    assert {row["member_name"] for row in document["stage_witnesses"]} == {
        "development",
        "functional_guard",
        "withheld_transfer",
    }
    assert document["compile_roles"] == {"unit_facet": "isolated"}
    assert len(document["invocation_evidence"]) == 6
    assert document["guard_link"]["status"] == "not_established" and not document["guard_link"]["guards"]
    with pytest.raises(ValueError, match="actual live"):
        preparation.verify()


@pytest.mark.parametrize("missing", ["score", "invocations", "stage"])
def test_v2_cannot_omit_the_mandatory_development_candidate_obligation(domain, monkeypatch, missing):
    _diagnostic_selection(domain, monkeypatch)
    F._grade(domain, monkeypatch)
    grade = domain[1].grade

    def incomplete(package, **kwargs):
        score = grade(package, **kwargs)
        if missing == "score":
            score["per_capsule"] = [row for row in score["per_capsule"] if row["capsule"] != "development"]
        elif missing == "invocations":
            shutil.rmtree(kwargs["runs_root"] / "development/invocations")
        return score

    monkeypatch.setattr(domain[1], "grade", incomplete)
    if missing == "stage":
        verifier = domain[0]["runtime_authority"].stage_verifier

        def absent_stage(**kwargs):
            if kwargs["member"]["name"] == "development":
                raise StageGateError("original development effects remain UNKNOWN")
            return verifier(**kwargs)

        monkeypatch.setattr(domain[0]["runtime_authority"], "stage_verifier", absent_stage)
    qualification = Q.qualify_component_compiler(**domain[0])
    document = Q.C.mapping_file(qualification.receipt)
    assert qualification.status == "refused" and document["source_domain"]["original_mandatory_ids"]
    with pytest.raises(StageGateError):
        qualification.verify()


@pytest.mark.parametrize("defect", ["optional", "predicate", "missing", "member"])
def test_v2_replays_full_original_membership_and_unevaluated_candidate_contract(domain, monkeypatch, defect):
    preparation, record = _diagnostic_selection(domain, monkeypatch)
    F._grade(domain, monkeypatch)
    qualification = Q.qualify_component_compiler(**domain[0])
    qualification.verify()
    if defect == "predicate":
        changed = copy.deepcopy(record)
        changed["candidate_predicates"][0]["candidate_verdict"] = "passed"
        domain[0]["compiler_origin"].inputs.source_preparation = replace(
            preparation, receipt_json=canonical_json(changed)
        )
    elif defect == "missing":
        domain[2]["obligations"].pop()
    elif defect == "member":
        domain[2]["obligations"][0]["members"][0]["output_roster"].append("missing-output")
    else:
        domain[2]["obligations"][0]["mandatory"] = False
    with pytest.raises(StageGateError, match="original requirements|domain selection"):
        qualification.verify()


def test_v2_live_selection_cannot_be_removed_after_qualification(domain, monkeypatch):
    _diagnostic_selection(domain, monkeypatch)
    F._grade(domain, monkeypatch)
    qualification = Q.qualify_component_compiler(**domain[0])
    domain[0]["compiler_origin"].inputs.source_preparation = None
    with pytest.raises(StageGateError, match="domain selection changed"):
        qualification.verify()


def test_v2_source_completion_keeps_original_unresolved_compile_static_obligations(domain, monkeypatch):
    _diagnostic_selection(domain, monkeypatch)
    calls = F._grade(domain, monkeypatch)
    unresolved = ["required source-only role unresolved: original:resource_legality"]
    original_roles = {"static_denominator": {"required": 4, "proved": 0, "unknown": 4}}
    monkeypatch.setattr(Q, "qualification_compile_roles", lambda *_args, **_kwargs: (original_roles, unresolved))
    qualification = Q.qualify_component_compiler(**domain[0])
    document = Q.C.mapping_file(qualification.receipt)
    assert len(calls) == 1 and qualification.status == "refused"
    assert document["compile_roles"] == original_roles and document["failures"] == unresolved
    assert document["schema"] == D.SOURCE_RECEIPT_SCHEMA
    with pytest.raises(StageGateError, match="no evaluated domain qualification"):
        qualification.verify()


def test_legacy_qualification_keeps_original_v1_receipt_and_scope(domain, monkeypatch):
    F._grade(domain, monkeypatch)
    document = Q.qualify_component_compiler(**domain[0]).verify()
    assert document["schema"] == D.LEGACY_RECEIPT_SCHEMA and "source_domain" not in document
    assert [row["member_name"] for row in document["stage_witnesses"]] == ["functional_guard", "withheld_transfer"]


def test_new_domain_comparisons_preserve_exact_json_scalar_types(domain, monkeypatch):
    preparation, record = _diagnostic_selection(domain, monkeypatch)
    _, binding = D.reopen_domain(domain[0]["corpus_root"], origin=domain[0]["compiler_origin"])
    changed = copy.deepcopy(binding)
    changed["original_requirements"][0]["mandatory"] = 1
    assert not D.unchanged_domain(binding, changed)
    with pytest.raises(StageGateError, match="domain selection changed"):
        D.verify_receipt_domain(D.receipt_domain(changed), binding)
    changed = copy.deepcopy(record)
    changed["candidate_predicates"][0]["candidate_verdict_phase"] = True
    domain[0]["compiler_origin"].inputs.source_preparation = replace(preparation, receipt_json=canonical_json(changed))
    with pytest.raises(StageGateError, match="pending candidate predicates"):
        D.reopen_domain(domain[0]["corpus_root"], origin=domain[0]["compiler_origin"])


def test_domain_and_preparation_readers_are_in_complete_ordinary_source_closure():
    sources = Q.E.component_sources()
    paths = {row["path"]: row["sha256"] for members in sources.values() for row in members.values()}
    for module in (D, P, A):
        path = Path(module.__file__).resolve()
        assert paths[str(path)] == Q.C.sha256_file(path)


def test_private_domain_reader_source_installed_and_bytecode_aliases_are_masked(tmp_path, monkeypatch):
    from merlin.common import access
    from merlin.targetgen.sandbox import bwrap

    surfaces_owner = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    relative = Path("merlin_experiments/phase1/component_qualification_domain.py")
    source = tmp_path / "packages/merlin-experiments/src" / relative
    site = tmp_path / "installed/site-packages"
    installed = site / relative
    bytecode = installed.parent / "__pycache__/component_qualification_domain.cpython-314.pyc"
    alternate_site = tmp_path / "alternate/site-packages"
    alternate = alternate_site / relative
    public = site / "merlin_experiments/phase1/context.py"
    for path in (source, installed, bytecode, alternate, public):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic owned module; never imported\n")
    monkeypatch.setattr(
        access,
        "sys",
        SimpleNamespace(path=[str(site), str(alternate_site)], prefix=str(tmp_path / "python"), modules={}),
    )
    monkeypatch.setattr(surfaces_owner, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(surfaces_owner, "artifacts_dir", lambda: tmp_path / "out/artifacts")
    monkeypatch.setattr(surfaces_owner, "_evicted_oracle_modules", lambda: [])
    monkeypatch.setattr(surfaces_owner, "_support_package_dirs", lambda: [])
    monkeypatch.setattr(surfaces_owner, "experimenter_memory_dir", lambda: tmp_path / "absent-memory")
    policy = SimpleNamespace(
        target="synthetic",
        capsule_corpus=None,
        corpus_siblings=lambda: (),
        hidden_corpus=lambda: None,
        prior_backends=(),
        backend_package=None,
    )
    assert D.__name__ in access.declared_modules("grader")
    surfaces = surfaces_owner.answer_surfaces(policy)
    expected = {source, installed, bytecode, alternate}
    assert expected <= {surface.path for surface in surfaces if surface.origin == "grader"}
    assert all(surface.path != public and surface.path not in public.parents for surface in surfaces)
    exposed = ["--ro-bind", str(tmp_path), str(tmp_path)]
    assert {surface.path for surface in bwrap.coverage_gap(exposed, surfaces)} == expected
    assert bwrap.coverage_gap(bwrap.apply_answer_masks(exposed, surfaces), surfaces) == []


@pytest.mark.parametrize("gate", ["origin", "runtime"])
def test_source_preparation_cannot_create_an_origin_or_independent_runtime(prepared, domain, monkeypatch, gate):
    monkeypatch.setitem(sys.modules, COVERAGE.__name__, COVERAGE)
    domain[0]["compiler_origin"] = _origin(prepared)
    calls = F._grade(domain, monkeypatch)
    monkeypatch.setattr(Q, "_verify_" + gate, ACTUAL_VERIFY_ORIGIN if gate == "origin" else ACTUAL_VERIFY_RUNTIME)
    with pytest.raises(StageGateError, match="issued fresh Phase 1|independently issued target runtime"):
        Q.qualify_component_compiler(**domain[0])
    assert not calls and not domain[0]["evidence_root"].exists()
    assert prepared.record()["mandatory_source_blockers"]
