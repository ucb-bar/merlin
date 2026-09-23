"""Installed round attribution uses tiny AST packages, without executing candidates."""

import json
import socket
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.mechanism_program import MechanismProgram
from merlin_experiments.phase2.mechanism_rounds import MechanismRounds

from merlin.perf import phase2_edit_contract


@pytest.fixture(autouse=True)
def refuse_execution(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("mechanism rounds must not launch processes or bind sockets")

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
    edit(candidate, schedule=1)
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


@pytest.fixture
def rounds(program):
    owner, authority, _, _, root = program
    freeze_catalog(program)
    shared = root / "shared"
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    return MechanismRounds(owner), authority.initial_source, shared, root / "output"


def begin(fixture, index=0):
    owner, candidate, shared, _ = fixture
    prepared = owner.capture(candidate, round_index=index, compiler_shared_source_root=shared)
    return owner.publish_start(prepared)


def inspect(fixture, *, require_semantic_edit=False):
    owner, candidate, shared, _ = fixture
    return owner.inspect(candidate, require_semantic_edit=require_semantic_edit, compiler_shared_source_root=shared)


def finalize(fixture, index=0):
    owner, candidate, shared, _ = fixture
    return owner.finalize(candidate, round_index=index, compiler_shared_source_root=shared)


def edit(candidate, *, schedule=3, protected=2):
    (candidate / "compiler.py").write_text(
        f"def schedule():\n    return {schedule}\ndef protected():\n    return {protected}\n"
    )


def test_seed_inspection_has_no_round_delta(rounds):
    owner, _, _, output = rounds
    before = sorted(output.iterdir())
    result = inspect(rounds)
    assert owner.configured
    assert result["status"] == "initial_seed"
    assert result["schema"] == "global_compiler_mechanism_seed_analysis_v1"
    assert owner.active is None
    assert sorted(output.iterdir()) == before


def test_capture_precedes_publication_and_freezes_start(rounds):
    owner, candidate, shared, output = rounds
    prepared = owner.capture(candidate, round_index=0, compiler_shared_source_root=shared)
    assert prepared.snapshot.is_dir()
    assert not prepared.snapshot.stat().st_mode & 0o222
    assert not (prepared.snapshot / "compiler.py").stat().st_mode & 0o222
    assert owner.active is None
    assert not (output / "mechanism_round_start_0000.json").exists()
    record = owner.publish_start(prepared)
    assert record["closed"] is False
    receipt = Path(record["round_start_receipt"]["path"])
    assert contracts.sha256_file(receipt) == record["round_start_receipt"]["sha256"]
    assert json.loads(receipt.read_text())["candidate_sha256"] == record["candidate_sha256"]
    record["closed"] = True
    assert owner.active["closed"] is False
    owner.require_open(0)
    with pytest.raises(ValueError):
        owner.publish_start(prepared)


@pytest.mark.parametrize("index", [-1, True, 1.5, "0", None])
def test_invalid_round_refused_before_snapshot(rounds, index):
    owner, candidate, shared, output = rounds
    before = sorted(output.iterdir())
    with pytest.raises(ValueError, match="nonnegative integer"):
        owner.capture(candidate, round_index=index, compiler_shared_source_root=shared)
    assert sorted(output.iterdir()) == before


def test_open_round_blocks_another_start_and_wrong_finalization(rounds):
    owner, candidate, shared, _ = rounds
    begin(rounds)
    with pytest.raises(ValueError, match="not closed"):
        owner.capture(candidate, round_index=1, compiler_shared_source_root=shared)
    with pytest.raises(ValueError, match="matching open round"):
        finalize(rounds, 1)


def test_semantic_noop_refused_and_cannot_be_analyzed(rounds):
    begin(rounds)
    candidate = rounds[1]
    with (candidate / "compiler.py").open("a") as stream:
        stream.write("# formatting-only authoring round\n")
    result = finalize(rounds)
    assert result["status"] == "refused"
    assert result["semantic_noop"] is True
    assert any("no semantic" in row["reason"] for row in result["violations"])
    with pytest.raises(ValueError, match="refused compiler mechanism round"):
        inspect(rounds)


def test_single_mechanism_delta_finalizes_exact_candidate_once(rounds):
    begin(rounds)
    edit(rounds[1])
    result = finalize(rounds)
    assert result["status"] == "allowed"
    assert result["semantic_noop"] is False
    assert result["mechanism_ids"] == ["schedule"]
    receipt = Path(result["receipt"]["path"])
    assert contracts.sha256_file(receipt) == result["receipt"]["sha256"]
    assert inspect(rounds)["candidate_sha256"] == result["candidate_sha256"]
    with pytest.raises(ValueError, match="matching open round"):
        finalize(rounds)
    edit(rounds[1], schedule=4)
    with pytest.raises(ValueError, match="changed after final"):
        inspect(rounds)


def test_repeated_round_index_cannot_overwrite_start(rounds):
    record = begin(rounds)
    edit(rounds[1])
    finalize(rounds)
    owner, candidate, shared, _ = rounds
    receipt = Path(record["round_start_receipt"]["path"])
    before = receipt.read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        owner.capture(candidate, round_index=0, compiler_shared_source_root=shared)
    assert receipt.read_bytes() == before


def test_out_of_scope_delta_refused(rounds):
    begin(rounds)
    edit(rounds[1], protected=9)
    result = finalize(rounds)
    assert result["status"] == "refused"
    assert result["violations"]


@pytest.mark.parametrize("artifact", ["snapshot", "receipt"])
def test_round_start_tamper_refused(rounds, artifact):
    record = begin(rounds)
    path = (
        Path(record["round_start_path"]) / "compiler.py"
        if artifact == "snapshot"
        else Path(record["round_start_receipt"]["path"])
    )
    path.chmod(0o644)
    path.write_text("{}\n" if artifact == "receipt" else "def schedule():\n    return 999\n")
    path.chmod(0o444)
    with pytest.raises(ValueError, match="round-start binding changed"):
        inspect(rounds)


def test_missing_start_refuses_edited_analysis_and_finalization(rounds):
    edit(rounds[1])
    with pytest.raises(ValueError, match="no immutable mechanism round start"):
        inspect(rounds)
    with pytest.raises(ValueError, match="matching open round"):
        finalize(rounds)


@pytest.mark.parametrize("operation", ["inspect", "finalize", "capture"])
def test_cleared_catalog_binding_cannot_disable_rounds(rounds, program, operation):
    begin(rounds)
    program[0].catalog_binding = None
    owner, candidate, shared, _ = rounds
    assert owner.configured
    with pytest.raises(ValueError):
        if operation == "inspect":
            inspect(rounds)
        elif operation == "finalize":
            finalize(rounds)
        else:
            owner.capture(candidate, round_index=1, compiler_shared_source_root=shared)


def test_active_view_cannot_remove_current_round(rounds):
    begin(rounds)
    owner = rounds[0]
    view = owner.active
    view.clear()
    assert owner.active["round"] == 0
    with pytest.raises(AttributeError):
        owner.active = None
    owner.require_open(0)


def test_foreign_publication_token_refused_without_receipt(rounds, program):
    owner, candidate, shared, output = rounds
    prepared = owner.capture(candidate, round_index=0, compiler_shared_source_root=shared)
    other = MechanismRounds(program[0])
    with pytest.raises(ValueError):
        other.publish_start(prepared)
    assert other.active is None
    assert not (output / "mechanism_round_start_0000.json").exists()
    owner.publish_start(prepared)


def test_multiple_individually_authorized_mechanisms_refused(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    edit(candidate, schedule=1)
    (candidate / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    output = tmp_path / "output"
    output.mkdir()
    authority = FrozenEditAuthority(output)
    symbols = ("schedule", "protected")
    contract = phase2_edit_contract.seal(
        {
            "schema": "compiler_edit_contract_v1",
            "existing_symbols": [{"surface_id": symbol, "path": "compiler.py", "symbol": symbol} for symbol in symbols],
            "helper_extensions": [],
        }
    )
    authority.freeze(candidate, contract, has_iterations=False)
    portfolio = {"schema": "full_model_optimization_portfolio_v1", "members": []}
    program_owner = MechanismProgram(
        output, authority, portfolio_identity=portfolio, portfolio_identity_sha256=contracts.document_sha256(portfolio)
    )
    catalog = seal(
        {
            "schema": "compiler_mechanism_catalog_v1",
            "contract_sha256": contract["sha256"],
            "mechanisms": [
                {"id": symbol, "selectors": [{"kind": "function", "path": "compiler.py", "symbol": symbol}]}
                for symbol in symbols
            ],
        }
    )
    source, digest = write_document(tmp_path / "catalog.json", catalog)
    program_owner.freeze_catalog(source, digest, has_iterations=False)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    fixture = MechanismRounds(program_owner), candidate, shared, output
    begin(fixture)
    edit(candidate, schedule=3, protected=4)
    assert authority.validate_candidate(candidate)["status"] == "allowed"
    result = finalize(fixture)
    assert result["status"] == "refused"
    assert result["mechanism_ids"] == ["protected", "schedule"]
    assert any("more than one" in row["reason"] for row in result["violations"])
