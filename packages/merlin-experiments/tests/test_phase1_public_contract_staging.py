"""Future-source fix verification; native readability is a separate real sandbox receipt."""
import hashlib
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import public_contract_staging as P
from merlin_experiments.phase1 import workspace_transport as W


@pytest.mark.parametrize("profile", ("xs-el-v1/el1", "xs-el-v1/el2", "xs-el-v1/el3",
                                     "fresh10-tooling-v1/infra", "fresh10-tooling-v1/scheduling",
                                     "fresh10-tooling-v1/no-reference-kernels",
                                     "xs-el-no-expert-v1/el1", "xs-el-no-expert-v1/el2",
                                     "xs-el-no-expert-v1/el3"))
def test_real_frozen_grant_copies_use_pinned_bytes_not_changed_checkout(tmp_path, monkeypatch, profile):
    repo, ws = tmp_path / "repo", tmp_path / "workspace"
    contract = repo / "merlin/contract"
    schemas = contract / "schemas"
    schemas.mkdir(parents=True)
    expected = {name: ("frozen public " + name).encode() for name in sorted(P.PUBLIC_FILES)}
    for name, data in expected.items():
        (contract / name).write_bytes(data)
    schema_bytes = b'{"type":"object"}\n'
    (schemas / "manifest.schema.json").write_bytes(schema_bytes)
    descriptor = repo / "descriptor.yaml"
    descriptor.write_text("target: fixture\ncapsule_corpus: merlin/contract/capsules\n")
    context = SimpleNamespace(repo=repo, descriptor=descriptor)
    bundle = {"author_profile": {"name": profile}, "allowed": [
        *({"path": P.PREFIX + name, "mode": "ro"} for name in sorted(P.PUBLIC_FILES)),
        {"path": P.SCHEMAS, "mode": "ro"},
    ]}
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    W.BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    try:
        (contract / "mlir_oot_backend_contract.yaml").write_bytes(b"new unadmitted checkout bytes")
        (schemas / "late.schema.json").write_bytes(b"unadmitted")
        W.assemble_workspace(bundle, ws, context=context)
        for name, data in expected.items():
            copied = ws / name
            assert copied.is_file() and not copied.is_symlink()
            assert copied.read_bytes() == data
            assert hashlib.sha256(copied.read_bytes()).digest() == hashlib.sha256(data).digest()
            assert copied.stat().st_mode & 0o222 == 0
        assert not (ws / "schemas").is_symlink()
        assert sorted(p.name for p in (ws / "schemas").iterdir()) == ["manifest.schema.json"]
        assert (ws / "schemas/manifest.schema.json").read_bytes() == schema_bytes
        assert (ws / "schemas").stat().st_mode & 0o222 == 0
    finally:
        W.BW.remove_bundle_snapshot(ws)


@pytest.mark.parametrize("bad", [
    "merlin/contract", "merlin/contract/capsules/hidden", "merlin/contract/../private", "/private/answer",
])
def test_unreviewed_input_path_fails_closed(tmp_path, bad):
    source = tmp_path / "source"
    source.write_bytes(b"not copied")
    destination = tmp_path / "target"
    with pytest.raises(ValueError, match="unexpected"):
        P.copy_grant({"path": bad, "mode": "ro"}, source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["symlink", "hidden", "nested", "golden.yaml", "unexpected.txt"])
def test_schema_grant_rejects_unsafe_members_before_any_copy(tmp_path, kind):
    source = tmp_path / "schemas-source"
    source.mkdir()
    (source / "manifest.schema.json").write_bytes(b"{}")
    if kind == "symlink":
        private = tmp_path / "private.schema.json"
        private.write_bytes(b"private")
        (source / "alias.schema.json").symlink_to(private)
    elif kind in {"hidden", "nested"}:
        (source / kind).mkdir()
    else:
        (source / kind).write_bytes(b"private")
    destination = tmp_path / "schemas"
    with pytest.raises(ValueError, match="unexpected member"):
        P.copy_grant({"path": P.SCHEMAS, "mode": "ro"}, source, destination)
    assert not destination.exists()


def test_file_grant_refuses_source_alias_and_destination_alias(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"public")
    source_alias = tmp_path / "source-alias"
    source_alias.symlink_to(source)
    entry = {"path": P.PREFIX + "VERSION", "mode": "ro"}
    destination = tmp_path / "VERSION"
    with pytest.raises(ValueError, match="canonical"):
        P.copy_grant(entry, source_alias, destination)
    destination.symlink_to(source)
    with pytest.raises(ValueError, match="fresh destination"):
        P.copy_grant(entry, source, destination)
    assert source.read_bytes() == b"public"


def test_other_grants_and_historical_profiles_keep_existing_transport():
    entry = {"path": P.PREFIX + "VERSION", "mode": "ro"}
    assert not P.selected({}, entry)
    assert not P.selected({"author_profile": {"name": "xs-el-v1/el4"}}, entry)
    assert not P.selected({"author_profile": {"name": "xs-el-v1/el3"}}, {"path": "/public/sources"})
    assert P.selected({"author_profile": {"name": "xs-el-v1/el3"}}, {"path": "merlin/contract"})
