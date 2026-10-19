"""Synthetic protected inventories test issuance; no private workload is opened."""

import hashlib
import json
from pathlib import Path

import pytest
import test_connected_integer_bounds as B
import test_operator_private_exclusion as H
from merlin_experiments.phase0 import operator_exclusion as E
from merlin_experiments.phase0 import operator_exclusion_intake as I
from merlin_experiments.phase0 import operator_exclusion_issuer as U
from merlin_experiments.phase0 import operator_typed_exclusion as T

LIMIT = 1024 * 1024


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path, data):
    path.write_text(json.dumps(data))
    return pin(path)


def test_same_typed_projection_ignores_names_and_provenance_but_checks_arithmetic_and_maps():
    source = B.program().encode()
    expected = T.signatures(source, max_bytes=LIMIT)
    assert len(expected) == 1
    assert T.signatures(source.replace(b"%wide", b"%renamed"), max_bytes=LIMIT) == expected
    # Exclusion is conservative: changed initialization also remains excluded.
    assert T.signatures(source.replace(b"constant 0 : i32", b"constant 1 : i32"), max_bytes=LIMIT) == expected
    assert T.signatures(source.replace(b"arith.muli", b"arith.addi"), max_bytes=LIMIT) != expected
    square = B.program(k=2).encode()
    assert T.signatures(square.replace(b"(m,k)", b"(k,m)"), max_bytes=LIMIT) != T.signatures(square, max_bytes=LIMIT)
    with pytest.raises(ValueError, match="parse budget"):
        T.signatures(source, max_bytes=1)
    with pytest.raises(Exception):
        T.signatures(source.replace(b"linalg.yield %sum", b"linalg.yield %missing"), max_bytes=LIMIT)


def test_issuer_creates_only_hashes_and_rechecks_complete_original_pins(monkeypatch, tmp_path):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    source = tmp_path / "synthetic-original.mlir"
    source.write_text(B.program())
    original = {str(source): pin(source)["sha256"]}
    calls = []
    monkeypatch.setattr(I, "resolve", lambda p, sha: calls.append((p, sha)) or dict(original))
    destination = private / "selection.json"
    selection, receipt = U.issue(tmp_path / "selected-original.json", "a" * 64, destination,
                                 forbidden_roots=[tmp_path / "public"], max_ir_bytes=LIMIT, max_total_ir_bytes=LIMIT)
    assert len(calls) == 2
    assert destination.stat().st_mode & 0o777 == 0o600
    data = json.loads(destination.read_bytes())
    assert set(data) == {"schema", "excluded_source_sha256", "excluded_contraction_sha256", "excluded_typed_ir_sha256"}
    assert data["schema"] == E.SCHEMA_V2 and data["excluded_contraction_sha256"] == []
    assert str(source) not in destination.read_text() and "tensor" not in destination.read_text()
    assert receipt["protected_source_count"] == 1 and "numerical" in receipt["qualification"]
    assert repr(selection) == "<OperatorPrivateExclusion>"
    with pytest.raises(U.IssuanceRefusal):
        U.issue(tmp_path / "original", "a" * 64, destination, forbidden_roots=[tmp_path / "public"],
                max_ir_bytes=LIMIT, max_total_ir_bytes=LIMIT)


@pytest.mark.parametrize("failure", ["missing", "changed", "empty", "budget", "malformed", "inventory_drift"])
def test_issuance_has_no_partial_file_on_any_missing_or_failed_member(monkeypatch, tmp_path, failure):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    source = tmp_path / "original.mlir"
    source.write_text("not MLIR" if failure == "malformed" else B.program())
    sha = "f" * 64 if failure == "changed" else pin(source)["sha256"]
    calls = []

    def resolve(*args):
        calls.append(args)
        if failure == "missing":
            raise ValueError("sensitive fixture name must not escape")
        if failure == "empty" or (failure == "inventory_drift" and len(calls) > 1):
            return {}
        return {str(source): sha}

    monkeypatch.setattr(I, "resolve", resolve)
    destination = private / "selection.json"
    with pytest.raises(U.IssuanceRefusal) as caught:
        U.issue(tmp_path / "p", "a" * 64, destination, forbidden_roots=[tmp_path / "public"],
                max_ir_bytes=1 if failure == "budget" else LIMIT, max_total_ir_bytes=LIMIT)
    assert not destination.exists()
    assert "sensitive" not in str(caught.value) and str(source) not in str(caught.value)


def test_issuer_refuses_public_destination_symlinks_and_world_readable_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "resolve", lambda *_: pytest.fail("must refuse before protected intake"))
    root = tmp_path / "public"
    root.mkdir(mode=0o700)
    tmp_path.chmod(0o755)
    for destination, excluded in [(root / "selection.json", [root]), (tmp_path / "selection.json", [root])]:
        with pytest.raises(U.IssuanceRefusal):
            U.issue(tmp_path / "p", "a" * 64, destination, forbidden_roots=excluded,
                    max_ir_bytes=LIMIT, max_total_ir_bytes=LIMIT)


def validation_fixture(tmp_path):
    root = tmp_path / "protected"
    root.mkdir()
    program = root / "synthetic/programs/unit"
    (program / "dev").mkdir(parents=True)
    (program / "catalog.json").write_text("{}")
    (program / "census.json").write_text("{}")
    (program / "dev/model.prepared.mlir").write_text(B.program())
    (root / "answer.npy").write_bytes(b"unread synthetic answer")
    members = {str(path.relative_to(root)): pin(path)["sha256"] for path in root.rglob("*") if path.is_file()}
    sums = root / "SHA256SUMS"
    sums.write_text("".join(f"{sha}  {name}\n" for name, sha in sorted(members.items())))
    manifest = root / "MANIFEST.json"
    manifest_pin = write_json(manifest, {
        "schema": "merlin.model_check.bundle.v1", "models": {"synthetic": {
            "model": "synthetic", "programs": {"unit": {"consistent": False}},
        }}, "selected_capture_roots": {"synthetic": "original operator scope"},
        "seal": {"sums": sums.name, "sums_sha256": pin(sums)["sha256"], "files": len(members)},
    })
    producer = tmp_path / "producer.json"
    producer_pin = write_json(producer, {"status": "sealed_verified_and_protected", "finished_at": "original",
                                         "manifest_sha256": manifest_pin["sha256"]})
    return {"manifest": manifest_pin, "sums": pin(sums), "producer_receipt": producer_pin}, root


def test_source_inventory_checks_complete_namespace_and_source_pins_without_reading_answers(tmp_path, monkeypatch):
    section, root = validation_fixture(tmp_path)
    read = Path.read_bytes

    def restricted(path):
        if path.suffix == ".npy":
            pytest.fail("private answers must remain unread")
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", restricted)
    actual_root, sources = I.validation_sources(section)
    assert actual_root == root and len(sources) == 1
    # Numerical consistency is deliberately not inferred from inventory custody.
    (root / "new-leaf.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="namespace"):
        I.validation_sources(section)


@pytest.mark.parametrize("mutation", ["producer_failed", "missing_source", "duplicate_sums", "unsafe_sums"])
def test_protected_source_intake_preserves_failure_and_complete_roster(mutation, tmp_path):
    section, root = validation_fixture(tmp_path)
    if mutation == "producer_failed":
        section["producer_receipt"] = write_json(tmp_path / "producer.json", {"status": "failed"})
    elif mutation == "missing_source":
        (root / "synthetic/programs/unit/dev/model.prepared.mlir").unlink()
    else:
        sums = root / "SHA256SUMS"
        old = sums.read_text()
        sums.write_text(
            old + old.splitlines()[0] + "\n" if mutation == "duplicate_sums" else "f" * 64 + "  ../escape\n"
        )
        section["sums"] = pin(sums)
    with pytest.raises(ValueError):
        I.validation_sources(section)


def test_v2_public_match_uses_same_typed_ir_projection_and_old_v1_still_works(tmp_path):
    cap, source = H.H.capsule(H.original())
    directory = H.writer(tmp_path / "public", cap=cap, source=source)
    hashes = T.signatures(source.encode(), max_bytes=LIMIT)
    assert len(hashes) == 2
    path = tmp_path / "selection.json"
    pin_data = write_json(path, {"schema": E.SCHEMA_V2, "excluded_source_sha256": [],
                                "excluded_contraction_sha256": [], "excluded_typed_ir_sha256": [hashes[-1]]})
    path.chmod(0o600)
    with pytest.raises(E.PublicationRefusal, match=E.REFUSAL):
        E.select(path, pin_data["sha256"]).check_member(directory)
