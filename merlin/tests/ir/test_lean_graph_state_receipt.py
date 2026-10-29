"""Strict actual-chain byte binding and consistent fresh metadata forgeries."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
import test_lean_graph_state_chain as fixture

from merlin.verify import lean_graph_state as stable
from merlin.verify import lean_graph_state_receipt as receipt
from merlin.verify import lean_graph_state_receipt_schema as schema


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    root = tmp_path_factory.mktemp("strict-chain")
    edges, contract, produced = fixture.actual_chain(root)
    claim = receipt.issue(
        edges,
        root / "measured",
        reference_root=fixture.fixture.REFERENCE,
        contract=contract,
        original_contract_identity=produced["contract_sha256"],
    )
    return root, edges, contract, claim


def forged(measured, name, change):
    root, _, _, claim = measured
    value = copy.deepcopy(claim)
    change(value)
    path = root / "measured" / (name + ".json")
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def rename_untouched(data):
    if 'sym_name = "untouched"' in data:
        return data.replace('sym_name = "untouched"', 'sym_name = "changed"')
    assert "@untouched" in data
    return data.replace("@untouched", "@changed")


def test_actual_strict_receipt_live_bindings(measured):
    root, edges, contract, claim = measured
    checked = receipt.validate_live(root / "measured/receipt.json", edges, contract=contract)
    assert checked == claim and len(claim["proof_inventory"]) == 63
    assert all(claim["claims"][key] is False for key in claim["claims"] if key != "status")


@pytest.mark.parametrize("endpoint", [0, 1, 2, 3])
def test_fresh_substituted_original_fully_rehashed_still_refused(endpoint, measured, tmp_path):
    _, edges, contract, _ = measured
    paths = [Path(path) for edge in edges for path in edge]
    replaced = []
    for index, source in enumerate(paths):
        path = tmp_path / f"original-{index}.mlir"
        path.write_bytes(source.read_bytes())
        if index == endpoint:
            path.write_text(rename_untouched(path.read_text()))
        replaced.append(path)
    # Metadata is freshly and consistently bound to the SUBSTITUTED originals;
    # the already checked snapshots are unchanged. Byte binding must reject it.
    path = forged(
        measured,
        f"fresh-substituted-{endpoint}",
        lambda data: data.update(originals=[stable.file_ref(path) for path in replaced]),
    )
    with pytest.raises(ValueError, match="snapshot byte buffers differ"):
        receipt.validate_live(path, [(replaced[0], replaced[1]), (replaced[2], replaced[3])], contract=contract)


@pytest.mark.parametrize("scope", ["middle", "all"])
def test_consistent_fresh_chain_originals_cannot_reuse_unchanged_snapshot(scope, measured, tmp_path):
    _, edges, contract, _ = measured
    originals = [Path(path) for edge in edges for path in edge]
    replaced = []
    for index, source in enumerate(originals):
        target = tmp_path / f"fresh-{index}.mlir"
        text = source.read_text()
        target.write_text(rename_untouched(text) if scope == "all" or index in (1, 2) else text)
        replaced.append(target)
    selected = [stable.file_ref(path) for path in replaced]
    assert selected[1]["sha256"] == selected[2]["sha256"]  # unchanged exact handoff requirement still holds
    path = forged(measured, "coherent-fresh-" + scope, lambda value: value.update(originals=selected))
    with pytest.raises(ValueError, match="snapshot byte buffers differ"):
        receipt.validate_live(path, [(replaced[0], replaced[1]), (replaced[2], replaced[3])], contract=contract)


@pytest.mark.parametrize("endpoint", [0, 1, 2, 3])
def test_fresh_snapshot_metadata_cannot_hide_unchanged_snapshot(endpoint, measured, tmp_path):
    _, edges, contract, _ = measured
    originals = [Path(path) for edge in edges for path in edge]
    replacement = tmp_path / "fresh-original.mlir"
    replacement.write_text(rename_untouched(originals[endpoint].read_text()))
    originals[endpoint] = replacement

    def change(data):
        data["originals"] = [stable.file_ref(path) for path in originals]
        # Forge a matching new hash/length for the old unchanged snapshot file.
        data["snapshots"][endpoint].update({key: data["originals"][endpoint][key] for key in ("sha256", "bytes")})

    path = forged(measured, f"fresh-snapshot-forgery-{endpoint}", change)
    with pytest.raises(ValueError, match="actual saved bytes differ"):
        receipt.validate_live(path, [(originals[0], originals[1]), (originals[2], originals[3])], contract=contract)


@pytest.mark.parametrize(
    "kind",
    [
        "bool-version",
        "float-version",
        "bool-length",
        "float-length",
        "negative-length",
        "empty-proofs",
        "missing-proof",
        "duplicate-proof",
        "approval",
        "native",
        "integer-flag",
        "extra",
        "missing-middle",
        "tool",
        "semantic",
        "source",
        "empty-dependencies",
    ],
)
def test_strict_receipt_schema_and_inventory_forgeries(kind, measured):
    root, edges, contract, _ = measured

    def change(data):
        if kind == "bool-version":
            data["version"] = True
        elif kind == "float-version":
            data["version"] = 1.0
        elif kind == "bool-length":
            data["originals"][0]["bytes"] = True
        elif kind == "float-length":
            data["originals"][0]["bytes"] = 359.0
        elif kind == "negative-length":
            data["originals"][0]["bytes"] = -1
        elif kind == "empty-proofs":
            data["proof_inventory"] = []
        elif kind == "missing-proof":
            data["proof_inventory"].pop()
        elif kind == "duplicate-proof":
            data["proof_inventory"].append(data["proof_inventory"][0])
        elif kind == "approval":
            data["claims"]["source_contract_approved"] = True
        elif kind == "native":
            data["claims"]["native_fidelity_proved"] = True
        elif kind == "integer-flag":
            data["claims"]["full_model_claim"] = 0
        elif kind == "extra":
            data["qualification"] = "approved by candidate"
        elif kind == "missing-middle":
            data["originals"].pop(1)
        elif kind == "tool":
            data["selected_tool"]["sha256"] = "0" * 64
        elif kind == "semantic":
            data["original_contract_identity"] = "0" * 64
        elif kind == "source":
            data["sources"][0]["original"]["sha256"] = "0" * 64
        else:
            data["dependency_inventory"] = []

    path = forged(measured, "forged-" + kind, change)
    with pytest.raises(ValueError):
        receipt.validate_live(path, edges, contract=contract)


def test_duplicate_json_fields_refused(measured):
    root, _, _, _ = measured
    path = root / "duplicate.json"
    path.write_text('{"schema":"old","schema":"new"}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        schema.load(path)


def rehash_all_metadata(root, previous):
    """Adversary updates every reachable file record after a proof substitution."""

    def rewrite(value):
        if type(value) is str:
            return value.replace(str(previous), str(root))
        if type(value) is list:
            return [rewrite(item) for item in value]
        if type(value) is dict:
            changed = {key: rewrite(item) for key, item in value.items()}
            if set(changed) == {"path", "sha256", "bytes"} and Path(changed["path"]).is_file():
                return stable.file_ref(changed["path"])
            return changed
        return value

    files = [*root.rglob("*.json"), *root.rglob("*.stdout")]
    for _ in range(20):
        changed = False
        for path in files:
            try:
                value = json.loads(path.read_text())
            except (ValueError, UnicodeError):
                continue
            value = rewrite(value)
            if path.name == "receipt.json":
                for key in ("proof_inventory", "dependency_inventory"):
                    value[key].sort(key=lambda row: row["path"])
            data = json.dumps(value, indent=2) + "\n"
            if path.read_text() != data:
                path.write_text(data)
                changed = True
        if not changed:
            return
    raise AssertionError("coherent metadata forgery did not converge")


def test_consistent_rehashed_proof_object_forgery_is_refused(measured, tmp_path):
    original, edges, contract, _ = measured
    clone = tmp_path / "coherent-forgery"
    shutil.copytree(original / "measured", clone)
    proof = clone / "checks/direct/stage-000/GraphStateInstance.olean"
    proof.write_bytes(proof.read_bytes() + b"coherently rehashed substitution")
    rehash_all_metadata(clone, original / "measured")
    # Hashes, paths, buffers, source/tool selections and all recursive report
    # metadata are consistent. The LIVE proof reproduction must still reject.
    receipt.validate_live(clone / "receipt.json", edges, contract=contract)
    with pytest.raises(ValueError, match="does not reproduce"):
        receipt.replay(
            clone / "receipt.json",
            edges,
            tmp_path / "proof-refused",
            reference_root=fixture.fixture.REFERENCE,
            contract=contract,
        )


@pytest.mark.parametrize("kind", ["float-count", "boolean-count", "empty-roots", "report-approval", "missing-import"])
def test_consistent_rehashed_audit_forgery_is_refused(kind, measured, tmp_path):
    original, edges, contract, _ = measured
    clone = tmp_path / "coherent-audit-forgery"
    shutil.copytree(original / "measured", clone)
    folder = clone / "checks/direct/stage-000"
    report = json.loads((folder / "report.json").read_text())
    audit = report["transitive_audit"]
    raw = json.loads((folder / "audit.stdout").read_text())
    if kind in ("float-count", "boolean-count"):
        value = float(raw["theorems_audited"]) if kind == "float-count" else True
        raw["theorems_audited"] = value
        audit["raw"]["theorems_audited"] = value
    elif kind == "empty-roots":
        audit["statements"] = {}
        (folder / "statements.stdout").write_text('{"statements":{}}\n')
    elif kind == "report-approval":
        report["public_acceptance_issued"] = True
    else:
        removed = raw["imports"].pop(0)
        audit["raw"]["imports"] = raw["imports"]
        audit["imported_objects"] = [row for row in audit["imported_objects"] if row["path"] != removed["olean"]]
    (folder / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (folder / "audit.stdout").write_text(json.dumps(raw) + "\n")
    rehash_all_metadata(clone, original / "measured")
    if kind == "missing-import":
        # All byte hashes/report inventories are consistent; only an actual
        # fresh import audit establishes the omitted transitive dependency.
        receipt.validate_live(clone / "receipt.json", edges, contract=contract)
        with pytest.raises(ValueError, match="do not reproduce"):
            receipt.replay(
                clone / "receipt.json",
                edges,
                tmp_path / "import-refused",
                reference_root=fixture.fixture.REFERENCE,
                contract=contract,
            )
    else:
        with pytest.raises(ValueError):
            receipt.validate_live(clone / "receipt.json", edges, contract=contract)
