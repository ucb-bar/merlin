"""Original typed public input owners with neutral target execution fixtures."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import test_public_diagnostic_corpus as CORPUS_TEST
import test_public_source_diagnostic as POLICY_TEST
from merlin_experiments.phase2 import capsule_ordinary_scope as C
from merlin_experiments.phase2 import capsule_source_development as D
from merlin_experiments.phase2 import public_diagnostic_corpus as PDC
from merlin_experiments.phase2 import public_source_diagnostic as P
from merlin_experiments.phase2 import search_inputs as S
from merlin_experiments.phase2 import stage_prompt as SP
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from test_capsule_certification import selected as selected
from test_capsule_ordinary_scope import scoped as scoped
from test_capsule_qualification_handoff import qualified as qualified


@pytest.fixture
def development(scoped, tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("typed-public", "typed-profile")]
    for root in roots:
        root.mkdir()
    published = CORPUS_TEST.published.__wrapped__(roots[0], monkeypatch)
    corpus = PDC.freeze_public_diagnostic_corpus(**published[0])
    args, original, cap, issuer = POLICY_TEST.inputs.__wrapped__(roots[1], monkeypatch)
    import yaml

    declaration = yaml.safe_load(scoped.owner.descriptor.read_text())
    declaration["capsule_corpus"] = str(corpus.capsules_root)
    descriptor = tmp_path / "development.yaml"
    descriptor.write_text(yaml.safe_dump(declaration))
    target = replace(
        args["target_experiment"],
        target=declaration["target"],
        path=descriptor,
        descriptor_sha256=sha256_file(descriptor),
        capsule_corpus=corpus.capsules_root,
    )
    inputs = D.CapsuleSourceDevelopment(corpus, args["service"], target)
    _, scope = C.select(
        scoped.owner,
        functional_runs_root=scoped.base.run_dir.parent,
        functional_run_id=scoped.base.run_dir.name,
        functional_submission_sha256=scoped.owner.submission_sha256,
        target_experiment=target,
        required_cohort_declaration=scoped.pin,
        development_inputs=inputs,
    )
    import shutil

    from merlin.targetgen.contract.schemas import contract_dir

    shutil.copytree(contract_dir() / "schemas", args["contract_root"] / "schemas")
    args.update(corpus=corpus, target_experiment=target)
    return scoped, inputs, scope, args


def test_separate_descriptors_keep_original_qualification_and_real_public_corpus(development):
    original, inputs, scope, _ = development
    record = C.verify(scope)
    assert record["phase1"]["descriptor"]["sha256"] == sha256_file(original.owner.descriptor)
    assert record["development"]["development_descriptor"]["sha256"] == inputs.target.descriptor_sha256
    assert record["development"]["performance_cohort_admitted"] is False
    assert D.freeze(inputs.corpus, Path("unused")) is inputs.corpus
    claim, replicates = D.formal_claim(inputs.corpus, 10)
    assert claim["formal_cohort_admitted"] is False and replicates == 0


@pytest.mark.parametrize("field", ["hardware_spec", "numeric_profile", "isa_headers", "toolchain"])
def test_development_view_cannot_change_original_target_declarations(development, field):
    original, inputs, _, _ = development
    import yaml

    path = inputs.target.path
    doc = yaml.safe_load(path.read_text())
    doc[field] = {"changed": True}
    path.write_text(yaml.safe_dump(doc))
    changed = replace(inputs, target=replace(inputs.target, descriptor_sha256=sha256_file(path)))
    with pytest.raises(StageGateError, match="declarations"):
        changed.verify(original.owner)


def test_typed_inputs_refuse_saved_record_and_late_corpus_profile_or_target_change(development):
    original, inputs, scope, _ = development
    saved = inputs.verify(original.owner)
    with pytest.raises(StageGateError):
        D.prepare_policy(
            saved,
            scope,
            candidate=None,
            baseline=None,
            corpus=inputs.corpus,
            target_experiment=inputs.target,
            edit_authority=None,
            compiler_library_selection=None,
            compiler_pass_selection=None,
            checked_source_diagnostic=None,
            contract_root=None,
            work_root=None,
        )
    inputs.service.bundle_manifest.write_text("changed profile")
    with pytest.raises((StageGateError, ValueError)):
        C.verify(scope)


def test_real_public_policy_is_used_and_scope_has_no_RTL_or_certificate(development):
    _, inputs, scope, args = development
    policy = D.prepare_policy(inputs, scope, **{k: v for k, v in args.items() if k != "service"})
    assert type(policy) is P.PublicSourceDiagnosticPolicy
    assert policy.binding()["capsule_development"]["promotion_authority"] == "NONE"
    selected = S.SearchInputs(policy)
    assert selected.kinds == ("corpus-selected-compile-spike-feedback",)
    assert selected.identities()["certificate_sha256"] is None
    assert C.enabled(policy)
    assert "evaluation-submit-functional-selected" in {a.name for a in policy.build_registry()}
    assert all("RTL" not in a.name for a in policy.build_registry())
    with pytest.raises(StageGateError):
        D.broker_execute(policy, None, "evaluation-submit", {"kind": "corpus-rtl-feedback"}, 0, 10, 0)
    with pytest.raises(StageGateError):
        D.broker_execute(policy, None, "tuning-gsim-feedback", {}, 0, 10, 0)


def test_actual_large_cohort_prompt_is_bounded_and_answer_free():
    members = [
        {"identity": f"public:{i}", "old": {"complete_golden": "/private/answer", "records": "x" * 15000}}
        for i in range(243)
    ]
    cohort = {"members": members}
    assert len(json.dumps(cohort)) > 3_000_000
    summary = SP._capsule_cohort_summary(cohort)
    assert summary["member_count"] == 243 and len(summary["identities"]) == 243
    payload = json.dumps(summary)
    assert len(payload) < 16 << 10 and "golden" not in payload and "/private" not in payload
    assert cohort["members"][0]["old"]["complete_golden"] == "/private/answer"


def test_diagnostic_configuration_roundtrips_real_original_policy_and_pool(development, tmp_path):
    import os

    from merlin_experiments.phase2.evaluation_pool import AssignedResources, JobResources
    from merlin_experiments.phase2.search_configuration import prepare_search_configuration, read_search_configuration
    from merlin_experiments.phase2.search_coordinator import SearchSettings

    _, inputs, scope, args = development
    policy = D.prepare_policy(inputs, scope, **{k: v for k, v in args.items() if k != "service"})
    config = prepare_search_configuration(
        policy,
        tmp_path / "diagnostic-search.json",
        resources=AssignedResources(
            tuple(sorted(os.sched_getaffinity(0))), 128 << 20, 128 << 20, 1, author_cores=0, reserve_fraction=0
        ),
        job_resources=JobResources(1, 64 << 20, 32 << 20),
        settings=SearchSettings(beam_width=1, soft_seconds=60, hard_seconds=90, final_reserve_seconds=10),
        source_bytes=1 << 20,
        source_files=100,
        retained_source_bytes=4 << 20,
        max_pending=1,
        max_jobs=1,
        author_agents=1,
    )
    assert config.document["schema"] == "merlin.capsule_diagnostic_search_configuration.v1"
    assert config.document["certificate_sha256"] is None
    pool = config.install(policy, edit_authority=policy.edit_authority, root=tmp_path / "actual-pool")
    try:
        status = policy.search_coordinator.status()
        assert status["ranked"] == []
        assert pool.observations() == ()
        assert pool.inputs.kinds == ("corpus-selected-compile-spike-feedback",)
    finally:
        pool.close()
    document = copy.deepcopy(config.document)
    document["schema"] = "merlin.component_search_configuration.v1"
    path = tmp_path / "forged-formal.json"
    path.write_text(json.dumps(document))
    with pytest.raises(StageGateError, match="immutable identity"):
        read_search_configuration(path)


def test_unprepared_source_selection_has_no_mechanical_preparation_authority(development):
    from merlin.targetgen.checked_source_model_diagnostic import MECHANISM, select_checked_source_diagnostic
    from merlin.targetgen.compiler_pass_selection import CompilerPassSelection

    _, inputs, _, args = development
    old = args["compiler_pass_selection"]
    source = CompilerPassSelection(old.registry, old.binding, mechanisms=(MECHANISM,), factory=old.factory)
    checked = select_checked_source_diagnostic(
        source,
        target=args["checked_source_diagnostic"].record()["target"],
        rtl_facts=args["checked_source_diagnostic"].record()["rtl_facts"]["path"],
        board_config="fixture",
    )
    D.validate_preparation(
        inputs, source, None, args["compiler_library_selection"], True, object(), Path("edit.json"), checked
    )
    with pytest.raises(StageGateError):
        D.validate_preparation(
            None, source, None, args["compiler_library_selection"], True, object(), Path("edit.json"), checked
        )
    with pytest.raises(StageGateError):
        D.validate_preparation(
            inputs, old, None, args["compiler_library_selection"], True, object(), Path("edit.json"), checked
        )
