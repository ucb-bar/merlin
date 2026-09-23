"""Frozen qualification resources cannot fall back to mutable checkout policy."""

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2 import functional_cohort as cohort
from merlin_experiments.phase2 import functional_qualification as qualification


def sealed_contract(root):
    directory = root / "inputs" / "contract"
    directory.mkdir(parents=True)
    schema = directory / "schema.json"
    schema.write_text("{}\n")
    schema.chmod(0o444)
    directory.chmod(0o555)
    return directory, {
        "schema": qualification.SCHEMA,
        "contract_snapshot": {"path": str(directory), **contracts.exact_tree_record(directory)},
    }


@pytest.mark.parametrize("mutation", ["bytes", "membership", "writable", "linked", "legacy", "owner"])
def test_contract_snapshot_refuses_drift(tmp_path, mutation):
    directory, declaration = sealed_contract(tmp_path)
    assert qualification.validate_contract_snapshot(tmp_path, declaration) == directory
    schema = directory / "schema.json"
    if mutation == "bytes":
        schema.chmod(0o644)
        schema.write_text('{"changed":true}')
        schema.chmod(0o444)
    elif mutation == "membership":
        directory.chmod(0o755)
        (directory / "extra.json").write_text("{}")
        (directory / "extra.json").chmod(0o444)
        directory.chmod(0o555)
    elif mutation == "writable":
        schema.chmod(0o644)
    elif mutation == "linked":
        directory.chmod(0o755)
        (directory / "alias.json").symlink_to(schema)
    elif mutation == "legacy":
        declaration["schema"] = "merlin.functional-gsim-qualification.v1"
    else:
        declaration["contract_snapshot"]["path"] = str(tmp_path / "outside")
    with pytest.raises(qualification.FunctionalQualificationError):
        qualification.validate_contract_snapshot(tmp_path, declaration)


def test_builtin_lowerer_receives_explicit_contract(tmp_path, monkeypatch):
    selected = tmp_path / "selected-policy"
    observed = {}
    selected_inputs = object()

    def sandbox(*args, **kwargs):
        assert kwargs == {"inputs": selected_inputs}
        return object()

    def lowerer(*args, **kwargs):
        observed.update(kwargs)
        return args[2]

    monkeypatch.setattr(qualification.HQUAL, "lower_with_functional_baseline", lowerer)
    monkeypatch.setattr(qualification.CAMPAIGN, "package_sandbox_policy", sandbox)
    monkeypatch.setattr(qualification.CAMPAIGN, "boxed_entrypoints", lambda policy: nullcontext())
    monkeypatch.setattr(qualification.WORKLOAD, "derive_workload", lambda manifest: {})
    case = SimpleNamespace(capsule_names=("case",), manifest=tmp_path / "capsule.yaml", identity="a" * 64)
    qualification._lower_case(
        functional_base=tmp_path / "base",
        case=case,
        attempt=tmp_path / "attempt",
        timeout=1,
        target_experiment=object(),
        lowerer=lowerer,
        contract_root=selected,
        policy_inputs=selected_inputs,
    )
    assert observed == {"contract_root": selected}


def test_live_cohort_uses_selected_contract_for_both_views(tmp_path, monkeypatch):
    observed = []
    selected = tmp_path / "policy"

    def discover(roots, *, labels, contract):
        observed.append(contract)
        return [{"name": next(iter(labels)), "kind": "model"}]

    monkeypatch.setattr(cohort, "discover_capsules", discover)
    monkeypatch.setattr(cohort, "functional_capsule", lambda cap: cap)
    monkeypatch.setattr(cohort.CAPSULES, "_split_ineligible", lambda *args: ([], []))
    target = SimpleNamespace(target="fixture", graded_roots=lambda: [], hidden_roots=lambda: [])
    cohort.functional_grade_cohort(target, contract_root=selected)
    assert observed == [selected, selected]


@pytest.mark.parametrize("kind", ["declaration", "completion"])
def test_receipt_parser_uses_the_bytes_it_hashes(tmp_path, monkeypatch, kind):
    document = (
        {"original": True}
        if kind == "declaration"
        else {
            "schema": qualification.SCHEMA,
            "status": "complete",
            "declaration_sha256": "d" * 64,
            "source_certificate": {"sha256": "s" * 64},
            "selected_captures": [],
            "functional_certificate": {"path": str(tmp_path / "certificate.json")},
        }
    )
    payload = json.dumps(document).encode()
    digest = hashlib.sha256(payload).hexdigest()
    receipt = tmp_path / f"{kind}.{digest}.json"
    receipt.write_bytes(payload)
    original = Path.read_bytes
    reads = []

    def read(path):
        data = original(path)
        if path == receipt:
            reads.append(path)
            path.write_text('{"changed":true}')
        return data

    monkeypatch.setattr(Path, "read_bytes", read)
    if kind == "declaration":
        assert qualification._load_sealed_declaration(tmp_path)[2] == document
    else:
        # The admitted original reaches certificate validation; a later reread
        # would instead reject its missing schema/status before this point.
        with pytest.raises(qualification.FunctionalQualificationError, match="functional GSIM certificate"):
            qualification._completion(tmp_path, "d" * 64, SimpleNamespace(sha256="s" * 64), set())
    assert reads == [receipt]
