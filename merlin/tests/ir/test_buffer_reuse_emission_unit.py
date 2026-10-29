"""Bounded source-only factoring/authority controls; no kernel result is mocked."""

import json
from dataclasses import replace

import pytest
from buffer_test_inputs import case_root

from merlin.verify import buffer_reuse_gate as gate
from merlin.verify.buffer_reuse_emit import MODULES, factor, theorem_roster
from merlin.verify.buffer_reuse_intake import ALLOCATOR, ROOT, canonical, prepare_chain


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    base = case_root(tmp_path_factory) / "epoch-cases-v1/two_epochs"
    paths = [base / name for name in ("source.mlir", "shape/after.mlir", "memory/middle.mlir", "memory/after.mlir")]
    if not all(path.is_file() for path in paths):
        pytest.fail("selected two_epochs raw4 data unavailable")
    return prepare_chain(tuple(path.read_text() for path in paths), allocator=ALLOCATOR)


def forbidden(*args, **kwargs):
    pytest.fail("native proof or consumer invoked by forged request")


def test_complete_eight_module_and_declaration_inventory(candidate):
    parts = json.loads(candidate.query_parts_json)
    assert tuple(part["module"] for part in parts) == MODULES
    assert parts == factor(candidate.query, candidate.roots)
    assert set(theorem_roster(candidate.query)) == {root for root in candidate.roots if not root.startswith("Merlin.")}
    assert "GraphMemoryAfterMaterialization.expanded_to_actual" in candidate.roots
    assert candidate.roots[-1] == ROOT
    sources = json.loads(candidate.sources_json)
    assert sum(row["path"].endswith("/buffer_reuse_emit.py") for row in sources) == 1
    assert sum(row["path"].endswith("/buffer_reuse_gate.py") for row in sources) == 1
    assert len(gate._ordered_sources(candidate)) == 111
    assert not candidate.describe()["authority"]


@pytest.mark.parametrize("alteration", ["missing", "order", "imports", "source_rehashed"])
def test_mutated_bundle_never_launches_native(candidate, alteration, tmp_path, monkeypatch):
    parts = json.loads(candidate.query_parts_json)
    if alteration == "missing":
        del parts[3]
    elif alteration == "order":
        parts[2], parts[3] = parts[3], parts[2]
    elif alteration == "imports":
        parts[-1]["imports"] = ["WholeFinal"]
    else:
        from merlin.verify.buffer_reuse_emit import digest

        parts[4]["source"] += "\n-- rehashed source is still a changed request\n"
        parts[4]["sha256"] = digest(parts[4]["source"])
    forged = replace(candidate, query_parts_json=canonical(parts))
    monkeypatch.setattr(gate, "_run", forbidden)
    result = gate.check(forged, lean=tmp_path / "unused", destination=tmp_path / "check")
    assert result.status == "unavailable" and result.completion is None
    assert not (tmp_path / "check").exists()


def test_coherently_refactored_but_stale_full_source_is_not_authority(candidate, tmp_path, monkeypatch):
    # Even a consistent fresh hash/import/part roster cannot replace original raw4.
    query = candidate.query
    marker = "namespace GraphMemoryAfterEntries\n"
    query = query.replace(marker, marker + "-- changed canonical query\n", 1)
    assert query != candidate.query
    forged = replace(candidate, query=query, query_parts_json=canonical(factor(query, candidate.roots)))
    monkeypatch.setattr(gate, "_run", forbidden)
    result = gate.check(forged, lean=tmp_path / "unused", destination=tmp_path / "check")
    assert result.status == "unavailable" and result.completion is None
    assert not (tmp_path / "check").exists()


@pytest.mark.parametrize(
    "alteration", ["dropped_helper_root", "extra_global", "changed_options", "missing_scope", "hidden_import"]
)
def test_incomplete_or_unaccounted_declarations_refuse(candidate, alteration):
    query, roots = candidate.query, candidate.roots
    if alteration == "dropped_helper_root":
        roots = tuple(root for root in roots if root != "GraphMemoryAfterMaterialization.expanded_to_actual")
    elif alteration == "extra_global":
        query += "def silentlyIgnored : Nat := 0\n"
    elif alteration == "changed_options":
        query = query.replace("set_option maxHeartbeats 8000000", "set_option maxHeartbeats 8000001")
    elif alteration == "missing_scope":
        query = query.replace("end ShapeBoundary\n", "")
    else:
        query = query.replace(
            "import Merlin.GraphLoopsMemorySCFAfterChain\n",
            "import Merlin.GraphLoopsMemorySCFAfterChain Merlin.Hidden\n",
            1,
        )
    with pytest.raises(ValueError):
        factor(query, roots)


def test_last_module_only_audit_is_not_complete(candidate):
    audit = {
        "violations": [],
        "assurance": "lean_kernel_std3_no_native_reflection",
        "theorems_audited": 1,
        "axioms_used": ["propext"],
        "imports": [{"module": "RawChainEntries", "olean": "/test/fake.olean"}],
    }
    with pytest.raises(gate.Unavailable, match="actual proof imports"):
        gate._audit(
            audit,
            {root: "SYNTHETIC TYPE; NOT KERNEL EVIDENCE" for root in candidate.roots},
            candidate,
            {name for name, _ in gate._ordered_sources(candidate)} | set(MODULES),
        )


@pytest.mark.parametrize("field", ["domain", "allocator", "index_bits", "roots"])
def test_wrong_relation_domain_or_root_set_has_no_authority(candidate, field, monkeypatch, tmp_path):
    changes = {"domain": "full-error-world", "allocator": "finite", "index_bits": 32, "roots": (ROOT,)}
    forged = replace(candidate, **{field: changes[field]})
    monkeypatch.setattr(gate, "_run", forbidden)
    result = gate.check(forged, lean=tmp_path / "unused", destination=tmp_path / "check")
    assert result.status == "unavailable" and result.completion is None
