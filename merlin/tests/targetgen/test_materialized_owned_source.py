"""Public projection preserves hash-bound original-owner compiler documents."""

import hashlib

import pytest
import yaml

from merlin.compile.owned_spatial_source import _local_document
from merlin.targetgen.contract.materialize import materialize_public_capsules


def fixture(tmp_path):
    root = tmp_path / "policy" / "case"
    root.mkdir(parents=True)
    documents = {"source": ("original-source.json", b'{"source":"original"}'),
                 "plan": ("original-plan.json", b'{"plan":"original"}')}
    selected = {"schema": "merlin.owned_spatial_source.v1"}
    for key, (name, raw) in documents.items():
        (root / name).write_bytes(raw)
        selected[key] = {"file": name, "sha256": hashlib.sha256(raw).hexdigest()}
    cap = {"name": "case", "label": "public", "required_oracle_tiers": ["L2", "L3"],
           "owned_host_source": selected}
    (root / "capsule.yaml").write_text(yaml.safe_dump(cap))
    (root / "capsule.interface.mlir").write_text("original interface")
    (root / "undeclared.json").write_text("not a selected compiler input")
    return root, cap


def test_relocated_public_capsule_retains_original_owner_documents(tmp_path):
    root, original = fixture(tmp_path)
    public = tmp_path / "run" / "private_corpus_input" / "public"
    assert materialize_public_capsules(public, corpus_roots=[root.parent]) == ["case"]
    relocated = public / "case"
    cap = yaml.safe_load((relocated / "capsule.yaml").read_bytes())
    assert cap["owned_host_source"] == original["owned_host_source"]
    assert cap["required_oracle_tiers"] == ["L2"]
    for key in ("source", "plan"):
        declaration = cap["owned_host_source"][key]
        path, raw = _local_document(relocated, declaration)
        assert path.parent == relocated and not path.is_symlink()
        assert raw == (root / declaration["file"]).read_bytes()
    assert not (relocated / "undeclared.json").exists()


@pytest.mark.parametrize("defect", [None, "missing_graph", "changed_rewrite"])
def test_current_ssa_region_documents_survive_native_style_relocation_or_refuse(tmp_path, defect):
    root, cap = fixture(tmp_path)
    selected = cap["owned_host_source"]
    selected["schema"] = "merlin.owned_spatial_source.v2"
    for key, name, raw in (
        ("source_graph", "source-region.mlir", b"original actual region"),
        ("source_rewrite", "source-rewrite.json", b'{"original":"group statements"}'),
    ):
        (root / name).write_bytes(raw)
        selected[key] = {"file": name, "sha256": hashlib.sha256(raw).hexdigest()}
    if defect == "missing_graph":
        (root / selected["source_graph"]["file"]).unlink()
    elif defect == "changed_rewrite":
        (root / selected["source_rewrite"]["file"]).write_text("different numerical statements")
    (root / "capsule.yaml").write_text(yaml.safe_dump(cap))
    public = tmp_path / "run" / "private_corpus_input" / "public"
    if defect is not None:
        with pytest.raises(ValueError):
            materialize_public_capsules(public, corpus_roots=[root.parent])
        assert not (public / "case" / "capsule.yaml").exists()
        return
    assert materialize_public_capsules(public, corpus_roots=[root.parent]) == ["case"]
    relocated = public / "case"
    for key in ("source", "plan", "source_graph", "source_rewrite"):
        path, raw = _local_document(relocated, selected[key])
        assert path.parent == relocated and raw == (root / path.name).read_bytes()


@pytest.mark.parametrize("defect", ["missing", "symlink", "changed", "oversized", "external", "collision", "schema"])
def test_invalid_original_owner_documents_fail_before_publication(tmp_path, defect):
    root, cap = fixture(tmp_path)
    selected = cap["owned_host_source"]
    path = root / selected["source"]["file"]
    if defect == "missing":
        path.unlink()
    elif defect == "symlink":
        other = root / "other.json"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    elif defect == "changed":
        path.write_bytes(b"changed")
    elif defect == "oversized":
        path.write_bytes(b" " * ((1 << 20) + 1))
    elif defect == "external":
        selected["source"]["file"] = "../original-source.json"
    elif defect == "collision":
        path = root / "capsule.interface.mlir"
        selected["source"] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        selected["schema"] = "unknown"
    (root / "capsule.yaml").write_text(yaml.safe_dump(cap))
    with pytest.raises(ValueError):
        materialize_public_capsules(tmp_path / "public", corpus_roots=[root.parent])
    assert not (tmp_path / "public" / "case" / "capsule.yaml").exists()


def test_original_owner_document_mutation_during_projection_is_rejected(tmp_path, monkeypatch):
    from merlin.targetgen.contract import materialize

    root, cap = fixture(tmp_path)
    original_copy = materialize.shutil.copyfile

    def mutate(*args, **kwargs):
        result = original_copy(*args, **kwargs)
        (root / cap["owned_host_source"]["source"]["file"]).write_text("changed during publication")
        return result

    monkeypatch.setattr(materialize.shutil, "copyfile", mutate)
    with pytest.raises(ValueError, match="differs"):
        materialize_public_capsules(tmp_path / "public", corpus_roots=[root.parent])
