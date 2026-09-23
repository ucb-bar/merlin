"""Synthetic installed mechanism authority; no engine, process or hardware execution."""

import copy
import json
import socket
import subprocess

import pytest
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.mechanism_evidence import verify_checkpoint_catalog, verify_checkpoint_work_order
from merlin_experiments.phase2.mechanism_program import MechanismProgram

from merlin.perf import phase2_edit_contract


@pytest.fixture(autouse=True)
def refuse_execution(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("mechanism metadata must not launch processes or bind sockets")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


def seal(document):
    body = {key: value for key, value in document.items() if key != "sha256"}
    return {**body, "sha256": contracts.document_sha256(body)}


def write_document(path, document):
    path.write_bytes(contracts.canonical_json(document))
    path.chmod(0o444)
    return path, contracts.sha256_file(path)


@pytest.fixture
def program(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("def schedule():\n    return 1\ndef protected():\n    return 2\n")
    (candidate / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    output = tmp_path / "output"
    output.mkdir()
    authority = FrozenEditAuthority(output)
    contract = phase2_edit_contract.seal(
        {
            "schema": "compiler_edit_contract_v1",
            "existing_symbols": [{"surface_id": "schedule", "path": "compiler.py", "symbol": "schedule"}],
            "helper_extensions": [],
        }
    )
    authority.freeze(candidate, contract, has_iterations=False)
    members = [{"capsule": name, "capsule_sha256": digest * 64} for name, digest in [("first", "a"), ("second", "b")]]
    portfolio = {"schema": "full_model_optimization_portfolio_v1", "members": members}
    owner = MechanismProgram(
        output,
        authority,
        portfolio_identity=portfolio,
        portfolio_identity_sha256=contracts.document_sha256(portfolio),
    )
    catalog = seal(
        {
            "schema": "compiler_mechanism_catalog_v1",
            "contract_sha256": contract["sha256"],
            "mechanisms": [
                {"id": "schedule", "selectors": [{"kind": "function", "path": "compiler.py", "symbol": "schedule"}]}
            ],
        }
    )
    return owner, authority, catalog, portfolio, tmp_path


def freeze_catalog(program):
    owner, _, catalog, _, root = program
    source, digest = write_document(root / "catalog.json", catalog)
    return owner.freeze_catalog(source, digest, has_iterations=False)


def work_order(program):
    _, authority, catalog, portfolio, _ = program
    candidate = authority.binding["initial_candidate_sha256"]
    return seal(
        {
            "schema": "host_prepared_mechanism_work_order_v1",
            "mechanism_id": "schedule",
            "catalog_sha256": catalog["sha256"],
            "contract_sha256": authority.contract["sha256"],
            "initial_candidate_sha256": candidate,
            "round_start_candidate_sha256": candidate,
            "portfolio_sha256": contracts.document_sha256(portfolio),
            "ordered_portfolio": portfolio["members"],
            "status": "ready_for_authoring",
            "source_operation_ids": [],
            "portfolio_site_bindings": [
                {
                    **member,
                    "compiler_sha256": candidate,
                    "source_sha256": "c" * 64,
                    "plan_digest": "d" * 64,
                    "candidate_command_buffer_sha256": "e" * 64,
                    "candidate_lowered_sha256": "f" * 64,
                    "status": "ready",
                    "inventory": {},
                    "source_operation_ids": [0, "tail"],
                    "chains": [],
                }
                for member in portfolio["members"]
            ],
        }
    )


def freeze_work(program):
    owner, authority, _, _, root = program
    freeze_catalog(program)
    source, digest = write_document(root / "work.json", work_order(program))
    prepared = owner.prepare_work_order(source, digest, has_iterations=False)
    return owner.freeze_work_order(prepared, candidate_sha256=authority.binding["initial_candidate_sha256"])


def analysis_record(program):
    _, authority, _, portfolio, _ = program
    candidate = authority.binding["initial_candidate_sha256"]
    analyses = [
        {
            "candidate_sha256": candidate,
            "workload": {"capsule_sha256": member["capsule_sha256"]},
            "diagnostics": {"verified_global_plan_emission": {"source_sha256": "c" * 64, "plan_digest": "d" * 64}},
            "emission": {"candidate_command_buffer_sha256": "e" * 64, "candidate_lowered_sha256": "f" * 64},
        }
        for member in portfolio["members"]
    ]
    return {
        "schema": "global_perf_iteration_v1",
        "iteration": 0,
        "candidate_sha256": candidate,
        "analysis": analyses[0],
        "analysis_reuse_binding": {"identity": "synthetic"},
        "portfolio": {
            "portfolio_sha256": contracts.document_sha256(portfolio),
            "members": [
                {"identity": member, "analysis": analysis}
                for member, analysis in zip(portfolio["members"], analyses, strict=True)
            ],
        },
    }


def test_catalog_freeze_is_detached_readonly_and_once(program):
    owner, authority, catalog, _, root = program
    binding = freeze_catalog(program)
    assert binding["contract_document_sha256"] == authority.binding["contract_document_sha256"]
    assert owner.catalog == catalog
    binding["catalog"]["mechanisms"].clear()
    assert owner.catalog["mechanisms"]
    owner.check_integrity()
    with pytest.raises(ValueError, match="once"):
        owner.freeze_catalog(root / "catalog.json", contracts.sha256_file(root / "catalog.json"), has_iterations=False)


def test_unconfigured_analysis_binding_is_noop(program):
    owner, _, _, _, root = program
    before = sorted((root / "output").iterdir())
    assert owner.bind_analysis({}) is None
    assert not owner.has_work_order
    assert sorted((root / "output").iterdir()) == before


def test_portfolio_is_detached_but_its_owned_identity_cannot_drift(program):
    owner, _, _, portfolio, _ = program
    original = copy.deepcopy(owner.portfolio_identity)
    portfolio["members"].reverse()
    assert owner.portfolio_identity == original
    owner.check_integrity()
    owner.portfolio_identity["members"].reverse()
    with pytest.raises(ValueError, match="portfolio identity changed"):
        owner.check_integrity()


@pytest.mark.parametrize(
    "field",
    [
        "catalog",
        "catalog_binding",
        "catalog_binding_sha256",
        "work_order",
        "work_order_binding",
        "work_order_binding_sha256",
        "analysis_binding",
    ],
)
def test_cleared_metadata_cannot_disable_configured_program(program, field):
    owner, _, _, _, root = program
    freeze_work(program)
    record = analysis_record(program)
    write_document(root / "output/iteration_0000.json", record)
    owner.bind_analysis(record)
    if field.endswith("_sha256"):
        with pytest.raises(AttributeError):
            setattr(owner, field, None)
        owner.check_integrity()
        return
    setattr(owner, field, None)
    assert owner.has_work_order
    with pytest.raises(ValueError, match="incomplete or changed"):
        owner.check_integrity()
    with pytest.raises(ValueError, match="incomplete or changed"):
        owner.bind_analysis(record, immutable_iteration=record)


@pytest.mark.parametrize(
    "artifact",
    ["catalog.json", "output/compiler_mechanism_catalog.json", "output/compiler_mechanism_catalog_receipt.json"],
)
def test_catalog_mutation_is_refused(program, artifact):
    owner, _, _, _, root = program
    freeze_catalog(program)
    path = root / artifact
    path.chmod(0o644)
    path.write_text("{}")
    path.chmod(0o444)
    with pytest.raises(ValueError, match="catalog changed"):
        owner.check_integrity()


def test_catalog_cannot_expand_edit_authority(program):
    owner, _, catalog, _, root = program
    catalog["mechanisms"][0]["selectors"][0]["symbol"] = "protected"
    source, digest = write_document(root / "expanded.json", seal(catalog))
    with pytest.raises(ValueError, match="exceeds host-frozen authority"):
        owner.freeze_catalog(source, digest, has_iterations=False)


@pytest.mark.parametrize("kind", ["catalog", "work"])
def test_candidate_owned_metadata_is_refused(program, kind):
    owner, authority, catalog, _, _ = program
    if kind == "work":
        freeze_catalog(program)
    source, digest = write_document(
        authority.initial_source / "metadata.json", catalog if kind == "catalog" else work_order(program)
    )
    with pytest.raises(ValueError, match="candidate-editable"):
        if kind == "catalog":
            owner.freeze_catalog(source, digest, has_iterations=False)
        else:
            owner.prepare_work_order(source, digest, has_iterations=False)


def test_work_order_preserves_graph_local_ids_and_identity(program):
    owner, authority, _, _, _ = program
    binding = freeze_work(program)
    assert [row["source_operation_ids"] for row in owner.work_order["portfolio_site_bindings"]] == [
        [0, "tail"],
        [0, "tail"],
    ]
    binding["work_order"]["portfolio_site_bindings"].clear()
    assert len(owner.work_order["portfolio_site_bindings"]) == 2
    assert owner.validate_work_order(
        work_order(program), candidate_sha256=authority.binding["initial_candidate_sha256"]
    ) == work_order(program)
    owner.check_integrity()


def test_prepared_work_order_rechecks_source_after_admission(program):
    owner, authority, _, _, root = program
    freeze_catalog(program)
    source, digest = write_document(root / "work.json", work_order(program))
    prepared = owner.prepare_work_order(source, digest, has_iterations=False)
    source.chmod(0o644)
    source.write_text("{}")
    source.chmod(0o444)
    with pytest.raises(ValueError):
        owner.freeze_work_order(prepared, candidate_sha256=authority.binding["initial_candidate_sha256"])
    assert not (root / "output/compiler_mechanism_work_order.json").exists()


def test_work_order_requires_owner_issued_preflight(program):
    owner, authority, _, _, root = program
    freeze_catalog(program)
    source, digest = write_document(root / "work.json", work_order(program))
    prepared = owner.prepare_work_order(source, digest, has_iterations=False)
    with pytest.raises(ValueError):
        owner.freeze_work_order(copy.copy(prepared), candidate_sha256=authority.binding["initial_candidate_sha256"])
    owner.freeze_work_order(prepared, candidate_sha256=authority.binding["initial_candidate_sha256"])


@pytest.mark.parametrize("kind", ["catalog", "work"])
def test_program_must_freeze_before_iterations(program, kind):
    owner, _, catalog, _, root = program
    if kind == "work":
        freeze_catalog(program)
    source, digest = write_document(root / "late.json", catalog if kind == "catalog" else work_order(program))
    with pytest.raises(ValueError, match="before candidate execution"):
        if kind == "catalog":
            owner.freeze_catalog(source, digest, has_iterations=True)
        else:
            owner.prepare_work_order(source, digest, has_iterations=True)


@pytest.mark.parametrize("mutation", ["flat", "order", "capsule", "compiler", "duplicate", "bool", "negative", "empty"])
def test_work_order_rejects_ambiguous_or_substituted_sites(program, mutation):
    owner, authority, _, _, _ = program
    freeze_catalog(program)
    work = work_order(program)
    row = work["portfolio_site_bindings"][0]
    if mutation == "flat":
        work["source_operation_ids"] = [0]
    elif mutation == "order":
        work["portfolio_site_bindings"].reverse()
    elif mutation in {"capsule", "compiler"}:
        row["capsule_sha256" if mutation == "capsule" else "compiler_sha256"] = "0" * 64
    elif mutation == "empty":
        for site in work["portfolio_site_bindings"]:
            site["source_operation_ids"] = []
    else:
        row["source_operation_ids"] = {"duplicate": [0, 0], "bool": [True], "negative": [-1]}[mutation]
    with pytest.raises(ValueError):
        owner.validate_work_order(seal(work), candidate_sha256=authority.binding["initial_candidate_sha256"])


def test_analysis_binding_and_continuation_preserve_static_identity(program):
    owner, _, _, _, root = program
    freeze_work(program)
    record = analysis_record(program)
    write_document(root / "output/iteration_0000.json", record)
    binding = owner.bind_analysis(record)
    assert len(binding["members"]) == 2
    assert binding["members"][0]["analysis_sha256"] == contracts.document_sha256(record["analysis"])
    continuation = copy.deepcopy(record)
    continuation["elapsed_seconds"] = 12
    continuation["probe_evidence"] = {"synthetic": True}
    assert owner.bind_analysis(continuation, immutable_iteration=record) == binding
    continuation["portfolio"]["members"][1]["analysis"]["emission"]["candidate_lowered_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="immutable current portfolio"):
        owner.bind_analysis(continuation, immutable_iteration=record)
    with pytest.raises(ValueError, match="immutable current portfolio"):
        owner.bind_analysis(record)


@pytest.mark.parametrize("mutation", ["missing_receipt", "receipt", "member", "plan", "candidate", "portfolio"])
def test_analysis_binding_refuses_unbound_evidence(program, mutation):
    owner, _, _, _, root = program
    freeze_work(program)
    record = analysis_record(program)
    if mutation == "member":
        record["portfolio"]["members"][1]["analysis"]["workload"]["capsule_sha256"] = "0" * 64
    elif mutation == "plan":
        record["analysis"]["diagnostics"]["verified_global_plan_emission"]["plan_digest"] = "0" * 64
    elif mutation == "candidate":
        record["candidate_sha256"] = "0" * 64
    elif mutation == "portfolio":
        record["portfolio"]["portfolio_sha256"] = "0" * 64
    if mutation != "missing_receipt":
        write_document(root / "output/iteration_0000.json", {} if mutation == "receipt" else record)
    with pytest.raises(ValueError):
        owner.bind_analysis(record)


@pytest.fixture
def checkpoint(program):
    owner, _, _, _, root = program
    freeze_work(program)
    record = analysis_record(program)
    write_document(root / "output/iteration_0000.json", record)
    owner.bind_analysis(record)
    return program


def verify_checkpoint(program):
    owner, authority, _, portfolio, root = program
    verify_checkpoint_catalog(root / "output", owner.catalog_binding, authority.binding)
    verify_checkpoint_work_order(
        root / "output",
        owner.work_order_binding,
        owner.analysis_binding,
        owner.catalog_binding,
        authority.binding,
        portfolio,
    )


def replace_document(path, document):
    path.chmod(0o644)
    write_document(path, document)


def test_checkpoint_uses_local_evidence_without_original_host_sources(checkpoint):
    owner, authority, _, _, root = checkpoint
    # Preserve recorded local paths; this does not qualify arbitrary archive relocation.
    (root / "catalog.json").rename(root / "retired-catalog.json")
    (root / "work.json").rename(root / "retired-work.json")
    before = copy.deepcopy(authority.binding)
    verify_checkpoint(checkpoint)
    assert authority.binding == before
    assert "grants no edit or measurement authority" in owner.work_order_binding["permission_scope"]
    with pytest.raises(ValueError, match="catalog changed"):
        owner.check_integrity()


@pytest.mark.parametrize(
    "artifact",
    [
        "compiler_mechanism_catalog.json",
        "compiler_mechanism_catalog_receipt.json",
        "compiler_mechanism_work_order.json",
        "compiler_mechanism_work_order_receipt.json",
        "compiler_mechanism_work_order_analysis.json",
    ],
)
@pytest.mark.parametrize("mutation", ["bytes", "writable", "missing", "link"])
def test_checkpoint_refuses_changed_frozen_artifacts(checkpoint, artifact, mutation):
    _, _, _, _, root = checkpoint
    path = root / "output" / artifact
    if mutation == "bytes":
        replace_document(path, {})
    elif mutation == "writable":
        path.chmod(0o644)
    else:
        retained = path.with_suffix(".retained")
        path.rename(retained)
        if mutation == "link":
            path.symlink_to(retained)
    with pytest.raises(ValueError, match="checkpoint compiler mechanism"):
        verify_checkpoint(checkpoint)


@pytest.mark.parametrize("parent", ["contract_document_sha256", "initial_candidate_sha256", "sha256"])
def test_checkpoint_catalog_refuses_parent_identity_substitution(checkpoint, parent):
    owner, authority, _, _, root = checkpoint
    binding = copy.deepcopy(owner.catalog_binding)
    binding[parent] = "0" * 64
    if parent != "sha256":
        binding = seal(binding)
    replace_document(root / "output/compiler_mechanism_catalog_receipt.json", binding)
    with pytest.raises(ValueError, match="catalog changed"):
        verify_checkpoint_catalog(root / "output", binding, authority.binding)


@pytest.mark.parametrize(
    "mutation", ["parent", "portfolio", "member", "initial_bytes", "initial_missing", "initial_link"]
)
def test_checkpoint_analysis_binds_parent_members_and_exact_initial_bytes(checkpoint, mutation):
    owner, _, _, _, root = checkpoint
    if mutation in {"parent", "portfolio", "member"}:
        binding = copy.deepcopy(owner.analysis_binding)
        if mutation == "parent":
            binding["work_order_binding_sha256"] = "0" * 64
        elif mutation == "portfolio":
            binding["portfolio_sha256"] = "0" * 64
        else:
            binding["members"][1]["analysis_sha256"] = "0" * 64
        owner.analysis_binding = seal(binding)
        replace_document(root / "output/compiler_mechanism_work_order_analysis.json", owner.analysis_binding)
    else:
        path = root / "output/iteration_0000.json"
        if mutation == "initial_bytes":
            # Identical JSON meaning is insufficient: the initial receipt pins exact bytes.
            document = json.loads(path.read_text())
            path.chmod(0o644)
            path.write_text(json.dumps(document, indent=2))
            path.chmod(0o444)
        else:
            retained = path.with_suffix(".retained")
            path.rename(retained)
            if mutation == "initial_link":
                path.symlink_to(retained)
    with pytest.raises(ValueError, match="checkpoint compiler mechanism work-order"):
        verify_checkpoint(checkpoint)


def test_checkpoint_requires_analysis_exactly_when_work_order_present(checkpoint):
    owner, authority, _, portfolio, root = checkpoint
    with pytest.raises(ValueError, match="lacks analyzed site bindings"):
        verify_checkpoint_work_order(
            root / "output", owner.work_order_binding, None, owner.catalog_binding, authority.binding, portfolio
        )
    with pytest.raises(ValueError, match="analysis without a work order"):
        verify_checkpoint_work_order(
            root / "output", None, owner.analysis_binding, owner.catalog_binding, authority.binding, portfolio
        )
    assert verify_checkpoint_work_order(root / "output", None, None, None, None, None) is None
    assert verify_checkpoint_catalog(root / "output", None, None) is None


@pytest.mark.parametrize("operation_ids", [[0, 0], [True, -1, {}], []])
def test_checkpoint_preserves_historical_site_policy_without_new_admission(checkpoint, operation_ids):
    owner, authority, _, _, root = checkpoint
    # Simulate a consistently sealed historical receipt. Historical decoding never
    # enforced the live admission validator's typed, unique, nonempty site policy.
    work = copy.deepcopy(owner.work_order)
    for row in work["portfolio_site_bindings"]:
        row["source_operation_ids"] = operation_ids
    work = seal(work)
    work_path = root / "output/compiler_mechanism_work_order.json"
    replace_document(work_path, work)
    binding = copy.deepcopy(owner.work_order_binding)
    binding.update(
        work_order=work,
        work_order_document_sha256=contracts.document_sha256(work),
        work_order_declared_sha256=work["sha256"],
        canonical_bytes_sha256=contracts.sha256_file(work_path),
    )
    owner.work_order_binding = seal(binding)
    replace_document(root / "output/compiler_mechanism_work_order_receipt.json", owner.work_order_binding)
    analysis = copy.deepcopy(owner.analysis_binding)
    analysis["work_order_binding_sha256"] = owner.work_order_binding["sha256"]
    for bound, site in zip(analysis["members"], work["portfolio_site_bindings"], strict=True):
        bound["site_binding_sha256"] = contracts.document_sha256(site)
    owner.analysis_binding = seal(analysis)
    replace_document(root / "output/compiler_mechanism_work_order_analysis.json", owner.analysis_binding)
    verify_checkpoint(checkpoint)
    with pytest.raises(ValueError):
        owner.validate_work_order(work, candidate_sha256=authority.binding["initial_candidate_sha256"])
