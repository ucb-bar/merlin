"""Host-frozen edit authority uses real AST checks on tiny synthetic compiler packages."""

import json
import shutil
import socket
import subprocess

import pytest
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority

from merlin.benchharness import hash_tree
from merlin.perf import compiler_edit_scope, phase2_edit_contract
from merlin.perf.agent_guidance import inspect_compiler_package


@pytest.fixture(autouse=True)
def refuse_processes(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("edit authority must not launch a process")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.fixture
def package(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text(
        "LIMIT = 16\ndef schedule():\n    return 1\ndef protected():\n    return 2\n"
    )
    (candidate / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    output = tmp_path / "authority"
    output.mkdir()
    contract = phase2_edit_contract.seal(
        {
            "schema": "compiler_edit_contract_v1",
            "existing_symbols": [{"surface_id": "overlap", "path": "compiler.py", "symbol": "schedule"}],
            "helper_extensions": [],
        }
    )
    compiler_edit_scope.validate_edit_contract(contract, candidate)
    return FrozenEditAuthority(output), candidate, contract, output


def declaration():
    return {
        "id": "overlap",
        "scope": "codegen",
        "path": "compiler.py",
        "symbol": "schedule",
        "effects": ["latency_hiding", "movement"],
        "cca_axes": ["dispatch.dma_overlap"],
        "mechanism": "Schedule independent transfers",
        "emitted_delta": "Changed issue ordering",
        "validation": "Source dependency proof",
        "abandonment": "Invalid dependency",
    }


@pytest.fixture
def selected_guidance(package, tmp_path):
    _, candidate, edit_contract, output = package
    resource_root = tmp_path / "selected_contract"
    (resource_root / "schemas").mkdir(parents=True)
    schema = resource_root / "schemas/manifest.schema.json"
    schema.write_text(
        json.dumps(
            {
                "properties": {
                    "optimization_surfaces": {
                        "items": {
                            "properties": {
                                "scope": {"enum": ["selected_scope"]},
                                "effects": {"items": {"enum": ["selected_effect"]}},
                            }
                        }
                    }
                }
            }
        )
    )
    guidance = {**declaration(), "scope": "selected_scope", "effects": ["selected_effect"]}
    authority = FrozenEditAuthority(output, guidance_contract=resource_root)
    return authority, candidate, edit_contract, output, resource_root, schema, guidance


def test_selected_guidance_schema_controls_unconfigured_inspection(selected_guidance, monkeypatch):
    from merlin.common import paths

    authority, candidate, _, _, _, _, guidance = selected_guidance
    (candidate / "manifest.yaml").write_text(
        json.dumps({"components": {"emit": ["compiler.py"]}, "optimization_surfaces": [guidance]})
    )
    monkeypatch.setattr(paths, "data_path", lambda *_: pytest.fail("selected guidance used ambient schema"))
    result = authority.inspect_optimization_surfaces(candidate)
    assert result["surfaces"][0]["scope"] == "selected_scope"
    assert result["surfaces"][0]["effects"] == ["selected_effect"]
    assert not authority.configured


def test_selected_guidance_used_for_freeze_and_current_inspection(selected_guidance, monkeypatch):
    from merlin.common import paths

    authority, candidate, edit_contract, output, resource_root, schema, guidance = selected_guidance
    monkeypatch.setattr(paths, "data_path", lambda *_: pytest.fail("selected guidance used ambient schema"))
    binding = authority.freeze(candidate, edit_contract, has_iterations=False, host_surface_declarations=[guidance])
    assert binding["guidance_contract_sha256"] == contracts.sha256_file(schema)
    assert "guidance_contract" not in binding
    assert str(resource_root) not in json.dumps(binding)
    assert json.loads((output / "compiler_edit_authority.json").read_text()) == binding
    assert binding["guidance_inventory"]["surfaces"][0]["effects"] == ["selected_effect"]
    source = candidate / "compiler.py"
    source.write_text("\n" + source.read_text().replace("return 1", "return 3"))
    result = authority.inspect_optimization_surfaces(candidate)
    assert result["surfaces"][0]["scope"] == "selected_scope"
    assert result["surfaces"][0]["line"] == binding["guidance_inventory"]["surfaces"][0]["line"] + 1
    authority.check_integrity()


@pytest.mark.parametrize("operation", ["check", "inspect", "validate"])
@pytest.mark.parametrize("mutation", ["bytes", "missing", "linked", "selection"])
def test_selected_guidance_schema_drift_refused(selected_guidance, tmp_path, mutation, operation):
    authority, candidate, edit_contract, _, resource_root, schema, guidance = selected_guidance
    authority.freeze(candidate, edit_contract, has_iterations=False, host_surface_declarations=[guidance])
    if mutation == "bytes":
        schema.write_text(schema.read_text() + "\n")
    elif mutation == "missing":
        schema.unlink()
    elif mutation == "linked":
        retained = tmp_path / "retained_schema.json"
        schema.rename(retained)
        schema.symlink_to(retained)
    else:
        alternate = tmp_path / "alternate_contract"
        shutil.copytree(resource_root, alternate)
        authority.guidance_contract = alternate
    with pytest.raises(ValueError, match="guidance contract|edit authority changed"):
        if operation == "check":
            authority.check_integrity()
        elif operation == "inspect":
            authority.inspect_optimization_surfaces(candidate)
        else:
            authority.validate_candidate(candidate)


def test_default_guidance_binding_and_inspection_stay_unchanged(package):
    authority, candidate, edit_contract, _ = package
    binding = authority.freeze(
        candidate, edit_contract, has_iterations=False, host_surface_declarations=[declaration()]
    )
    assert "guidance_contract" not in binding
    assert "guidance_contract_sha256" not in binding
    assert authority.inspect_optimization_surfaces(candidate)["surfaces"][0]["scope"] == "codegen"


def test_schema_change_during_freeze_cannot_publish_authority(selected_guidance, monkeypatch):
    from merlin_experiments.phase2 import edit_authority

    authority, candidate, edit_contract, output, _, schema, guidance = selected_guidance
    original = edit_authority.inspect_compiler_package

    def mutate_schema_after_inspection(*args, **kwargs):
        result = original(*args, **kwargs)
        schema.write_text(schema.read_text() + "\n")
        return result

    monkeypatch.setattr(edit_authority, "inspect_compiler_package", mutate_schema_after_inspection)
    with pytest.raises(ValueError, match="guidance contract"):
        authority.freeze(candidate, edit_contract, has_iterations=False, host_surface_declarations=[guidance])
    assert not (output / "compiler_edit_authority.json").exists()


def test_freeze_retains_readonly_seed_and_detached_binding_once(package):
    authority, candidate, contract, output = package
    binding = authority.freeze(candidate, contract, has_iterations=False)
    assert authority.initial_source == candidate.resolve()
    assert authority.seed == output / "edit_scope_seed"
    assert all(path.stat().st_mode & 0o222 == 0 for path in [authority.seed, *authority.seed.rglob("*")])
    receipt = output / "compiler_edit_authority.json"
    original = receipt.read_bytes()
    assert receipt.stat().st_mode & 0o222 == 0
    assert json.loads(original) == binding
    binding["contract"]["existing_symbols"].clear()
    contract["existing_symbols"].clear()
    assert authority.contract["existing_symbols"]
    authority.check_integrity()
    with pytest.raises(ValueError, match="once"):
        authority.freeze(candidate, contract, has_iterations=False)
    assert receipt.read_bytes() == original


def test_freeze_after_iterations_is_refused_without_publication(package):
    authority, candidate, contract, output = package
    with pytest.raises(ValueError, match="before"):
        authority.freeze(candidate, contract, has_iterations=True)
    assert authority.contract is None
    assert list(output.iterdir()) == []


def test_reviewed_declaration_rejects_unowned_required_decision(package):
    authority, candidate, contract, output = package
    reviewed = phase2_edit_contract.seal(
        {
            **contract,
            "target": "synthetic",
            "package_id": "tiny",
            "required_decisions": [{"decision": "protected policy", "owner": "compiler.py:protected"}],
        }
    )
    # Existing-symbol validation alone accepts this declaration; semantic routing must refuse it.
    compiler_edit_scope.validate_edit_contract(reviewed, candidate)
    with pytest.raises(phase2_edit_contract.Phase2EditContractError, match="protected policy"):
        authority.freeze(candidate, reviewed, has_iterations=False)
    assert list(output.iterdir()) == []


def test_reviewed_declaration_accepts_owned_required_decision(package):
    authority, candidate, contract, _ = package
    reviewed = phase2_edit_contract.load(
        "synthetic",
        "tiny",
        body={
            **contract,
            "target": "synthetic",
            "package_id": "tiny",
            "required_decisions": [{"decision": "schedule policy", "owner": "compiler.py:schedule"}],
        },
    )
    result = authority.freeze(candidate, reviewed, has_iterations=False)
    assert result["contract"] == reviewed
    authority.check_integrity()


@pytest.mark.parametrize("mutation", [None, "missing", "changed", "extra"])
def test_source_pin_membership_and_bytes_must_match_exactly(package, mutation):
    authority, candidate, contract, output = package
    pins = {path.name: contracts.sha256_file(path) for path in candidate.iterdir()}
    if mutation == "missing":
        pins.pop("manifest.yaml")
    elif mutation == "changed":
        pins["compiler.py"] = "a" * 64
    elif mutation == "extra":
        pins["absent.py"] = "a" * 64
    if mutation is None:
        assert authority.freeze(candidate, contract, has_iterations=False, source_pins=pins)["source_pins_checked"]
    else:
        with pytest.raises(ValueError, match="source-file pins"):
            authority.freeze(candidate, contract, has_iterations=False, source_pins=pins)
        assert list(output.iterdir()) == []


def test_initial_symlink_is_refused(package):
    authority, candidate, contract, output = package
    (candidate / "linked.py").symlink_to(candidate / "compiler.py")
    with pytest.raises(ValueError, match="symlink"):
        authority.freeze(candidate, contract, has_iterations=False)
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("mutation", ["id", "symbol"])
def test_host_guidance_cannot_expand_edit_authority(package, mutation):
    authority, candidate, contract, output = package
    guidance = declaration()
    guidance[mutation] = "protected" if mutation == "symbol" else "new authority"
    with pytest.raises(ValueError, match="outside frozen edit authority"):
        authority.freeze(candidate, contract, has_iterations=False, host_surface_declarations=[guidance])
    assert list(output.iterdir()) == []


def test_inspection_keeps_frozen_semantics_and_current_source_locations(package):
    authority, candidate, contract, _ = package
    guidance = declaration()
    authority.freeze(candidate, contract, has_iterations=False, host_surface_declarations=[guidance])
    before = authority.inspect_optimization_surfaces(candidate)
    guidance["mechanism"] = "unapproved caller mutation"
    source = candidate / "compiler.py"
    source.write_text("\n\n" + source.read_text().replace("return 1", "return 3"))
    authority.check_integrity()
    assert authority.validate_candidate(candidate)["status"] == "allowed"
    after = authority.inspect_optimization_surfaces(candidate)
    assert after["surfaces"][0]["mechanism"] == declaration()["mechanism"]
    assert after["surfaces"][0]["line"] == before["surfaces"][0]["line"] + 2
    assert after["host_guidance_binding"]["permission_scope"] == "unchanged host-frozen edit contract"
    assert after["host_guidance_binding"]["inventory_sha256"] == authority.binding["guidance_inventory_sha256"]


def test_allowed_edits_do_not_reset_cumulative_unauthorized_edit_baseline(package):
    authority, candidate, contract, output = package
    authority.freeze(candidate, contract, has_iterations=False)
    source = candidate / "compiler.py"
    source.write_text(source.read_text().replace("return 1", "return 3"))
    assert authority.validate_candidate(candidate)["status"] == "allowed"
    seed_bytes = (authority.seed / "compiler.py").read_bytes()
    source.write_text(source.read_text().replace("return 2", "return 9"))
    with pytest.raises(ValueError, match="exceeds host-frozen authority"):
        authority.validate_candidate(candidate)
    refusals = list(output.glob("edit_scope_refusal_*.json"))
    assert len(refusals) == 1
    assert refusals[0].stat().st_mode & 0o222 == 0
    refusal = json.loads(refusals[0].read_bytes())
    assert refusal["status"] == "refused"
    assert refusal["violations"]
    assert len(refusal["candidate_sha256"]) == 64
    assert (authority.seed / "compiler.py").read_bytes() == seed_bytes


@pytest.mark.parametrize("mutation", ["contract", "seed", "guidance_missing", "guidance_binding"])
def test_integrity_refuses_mutation_of_frozen_authority(package, mutation):
    authority, candidate, contract, _ = package
    authority.freeze(candidate, contract, has_iterations=False, host_surface_declarations=[declaration()])
    if mutation == "contract":
        authority.contract["existing_symbols"][0]["symbol"] = "protected"
    elif mutation == "seed":
        source = authority.seed / "compiler.py"
        source.chmod(0o644)
        source.write_text(source.read_text() + "CHANGED = True\n")
        source.chmod(0o444)
    elif mutation == "guidance_missing":
        authority.guidance_inventory = None
    elif mutation == "guidance_binding":
        authority.binding["guidance_inventory"]["surfaces"][0]["mechanism"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        authority.check_integrity()


def test_unexpected_checker_error_is_not_recorded_as_policy_refusal(package, monkeypatch):
    authority, candidate, contract, output = package
    authority.freeze(candidate, contract, has_iterations=False)

    def broken_checker(*args, **kwargs):
        raise RuntimeError("synthetic checker implementation failure")

    monkeypatch.setattr(compiler_edit_scope, "inspect_compiler_edits", broken_checker)
    with pytest.raises(RuntimeError, match="implementation failure"):
        authority.validate_candidate(candidate)
    assert list(output.glob("edit_scope_refusal_*.json")) == []


def test_unconfigured_authority_retains_development_only_status(package):
    authority, candidate, _, output = package
    assert authority.validate_candidate(candidate) == {"status": "unconfigured_development_only"}
    authority.check_integrity()
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("operation", ["check", "validate", "inspect"])
@pytest.mark.parametrize(
    "mutation",
    [
        "contract_none",
        "binding_none",
        "contract_rehashed",
        "seed_rehashed",
        "guidance_removed",
        "guidance_injected",
        "seed_writable",
        "seed_file_writable",
        "seed_linked",
        "seed_missing",
        "receipt_writable",
        "receipt_linked",
        "receipt_missing",
        "receipt_changed",
    ],
)
def test_corrupted_frozen_authority_is_refused_at_every_public_use(package, mutation, operation):
    authority, candidate, contract, output = package
    guidance = None if mutation == "guidance_injected" else [declaration()]
    authority.freeze(candidate, contract, has_iterations=False, host_surface_declarations=guidance)
    receipt = output / "compiler_edit_authority.json"
    if mutation == "contract_none":
        authority.contract = None
    elif mutation == "binding_none":
        authority.binding = None
    elif mutation == "contract_rehashed":
        authority.contract["existing_symbols"].append(
            {"surface_id": "unapproved", "path": "compiler.py", "symbol": "protected"}
        )
        authority.contract = phase2_edit_contract.seal(authority.contract)
        authority.binding["contract"] = authority.contract
        authority.binding["contract_document_sha256"] = contracts.document_sha256(authority.contract)
    elif mutation == "seed_rehashed":
        source = authority.seed / "compiler.py"
        source.chmod(0o644)
        source.write_text(source.read_text().replace("return 2", "return 7"))
        source.chmod(0o444)
        authority.binding["initial_candidate_sha256"] = hash_tree(authority.seed)["sha256"]
    elif mutation == "guidance_removed":
        authority.binding.pop("guidance_inventory")
        authority.binding.pop("guidance_inventory_sha256")
        authority.guidance_inventory = None
    elif mutation == "guidance_injected":
        authority.guidance_inventory = inspect_compiler_package(candidate, host_surface_declarations=[declaration()])
    elif mutation == "seed_writable":
        authority.seed.chmod(0o755)
    elif mutation == "seed_file_writable":
        (authority.seed / "compiler.py").chmod(0o644)
    elif mutation in ("seed_linked", "seed_missing"):
        saved = authority.seed.with_name("retained_seed")
        authority.seed.rename(saved)
        if mutation == "seed_linked":
            authority.seed.symlink_to(saved, target_is_directory=True)
    elif mutation == "receipt_writable":
        receipt.chmod(0o644)
    elif mutation in ("receipt_linked", "receipt_missing"):
        saved = receipt.with_suffix(".retained")
        receipt.rename(saved)
        if mutation == "receipt_linked":
            receipt.symlink_to(saved)
    elif mutation == "receipt_changed":
        receipt.chmod(0o644)
        receipt.write_bytes(receipt.read_bytes() + b" ")
        receipt.chmod(0o444)
    with pytest.raises(ValueError):
        if operation == "check":
            authority.check_integrity()
        elif operation == "validate":
            authority.validate_candidate(candidate)
        else:
            authority.inspect_optimization_surfaces(candidate)
    assert list(output.glob("edit_scope_refusal_*.json")) == []


def test_cleared_contract_does_not_reopen_freeze_lifecycle(package):
    authority, candidate, contract, output = package
    authority.freeze(candidate, contract, has_iterations=False)
    original = (output / "compiler_edit_authority.json").read_bytes()
    authority.contract = None
    authority.binding = None
    with pytest.raises(ValueError, match="once"):
        authority.freeze(candidate, contract, has_iterations=False)
    assert (output / "compiler_edit_authority.json").read_bytes() == original


def test_publication_failure_never_promotes_authority_or_reopens_freeze(package):
    authority, candidate, contract, output = package
    receipt = output / "compiler_edit_authority.json"
    original = b"preexisting authority evidence"
    receipt.write_bytes(original)
    with pytest.raises(FileExistsError):
        authority.freeze(candidate, contract, has_iterations=False)
    assert receipt.read_bytes() == original
    for operation in (
        authority.check_integrity,
        lambda: authority.validate_candidate(candidate),
        lambda: authority.inspect_optimization_surfaces(candidate),
    ):
        with pytest.raises(ValueError):
            operation()
    authority.contract = None
    authority.binding = None
    with pytest.raises(ValueError, match="once"):
        authority.freeze(candidate, contract, has_iterations=False)
    assert receipt.read_bytes() == original
    assert list(output.glob("edit_scope_refusal_*.json")) == []


def assert_all_authority_uses_refused(authority, candidate):
    for operation in (
        authority.check_integrity,
        lambda: authority.validate_candidate(candidate),
        lambda: authority.inspect_optimization_surfaces(candidate),
    ):
        with pytest.raises(ValueError):
            operation()


def test_rewriting_contract_binding_and_receipt_cannot_replace_original_authority(package):
    authority, candidate, contract, output = package
    authority.freeze(candidate, contract, has_iterations=False)
    authority.contract["existing_symbols"].append(
        {"surface_id": "unapproved", "path": "compiler.py", "symbol": "protected"}
    )
    authority.contract = phase2_edit_contract.seal(authority.contract)
    authority.binding["contract"] = authority.contract
    authority.binding["contract_document_sha256"] = contracts.document_sha256(authority.contract)
    receipt = output / "compiler_edit_authority.json"
    receipt.chmod(0o644)
    receipt.write_bytes(contracts.canonical_json(authority.binding))
    receipt.chmod(0o444)
    assert json.loads(receipt.read_bytes()) == authority.binding
    compiler_edit_scope.validate_edit_contract(authority.contract, candidate)
    assert_all_authority_uses_refused(authority, candidate)


@pytest.mark.parametrize("entry", ["build", ".git", "new_empty_directory"])
def test_seed_integrity_includes_digest_excluded_files_and_empty_directories(package, entry):
    authority, candidate, contract, _ = package
    if entry != "new_empty_directory":
        (candidate / entry).mkdir()
        (candidate / entry / "retained.txt").write_text("original retained bytes")
    authority.freeze(candidate, contract, has_iterations=False)
    original_compiler_digest = hash_tree(authority.seed)["sha256"]
    if entry == "new_empty_directory":
        authority.seed.chmod(0o755)
        (authority.seed / entry).mkdir(mode=0o555)
        authority.seed.chmod(0o555)
    else:
        source = authority.seed / entry / "retained.txt"
        source.chmod(0o644)
        source.write_text("changed retained bytes")
        source.chmod(0o444)
    assert hash_tree(authority.seed)["sha256"] == original_compiler_digest
    assert all(path.stat().st_mode & 0o222 == 0 for path in [authority.seed, *authority.seed.rglob("*")])
    assert_all_authority_uses_refused(authority, candidate)


@pytest.mark.parametrize("field", ["output", "seed", "initial_source"])
def test_equal_byte_directory_substitution_cannot_change_frozen_locations(package, tmp_path, field):
    authority, candidate, contract, _ = package
    authority.freeze(candidate, contract, has_iterations=False)
    original = getattr(authority, field)
    replacement = tmp_path / f"substituted-{field}"
    shutil.copytree(original, replacement)
    assert contracts.exact_tree_record(replacement)["sha256"] == contracts.exact_tree_record(original)["sha256"]
    setattr(authority, field, replacement)
    assert_all_authority_uses_refused(authority, candidate)
