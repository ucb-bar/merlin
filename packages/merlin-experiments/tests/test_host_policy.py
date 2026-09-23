"""Installed host-policy identities use real module origins and explicit resource owners."""

import copy
import importlib
import importlib.util
import socket
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2 import host_policy as HP
from merlin_experiments.phase2 import static_identity as SI


@pytest.fixture(autouse=True)
def refuse_processes_and_listeners(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("host-policy capture and admission must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


@pytest.fixture
def policy_inputs(tmp_path):
    controller = tmp_path / "controller.py"
    controller.write_text("POLICY = 'synthetic controller'\n")
    resources = tmp_path / "contract"
    for relative in HP.RESOURCE_FILES:
        path = resources / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"synthetic_resource": "' + relative + '"}\n')
    return controller, resources


@pytest.fixture
def captured(policy_inputs):
    controller, resources = policy_inputs
    return HP.build_record(controller_source=controller, contract_root=resources)


def reseal_location(record):
    record["location_sha256"] = contracts.document_sha256(
        {name: record[name] for name in ("sources", "identities", "closures")}
    )


def relocate(record, root, *, split=False):
    """Copy all captured bytes, preserving each explicitly declared closure mapping."""
    copied = copy.deepcopy(record)
    destinations = {}
    for namespace, closure in copied["closures"].items():
        directory = root / "closures" / namespace
        directories = [directory / "first", directory / "second"] if split else [directory]
        closure["roots"] = sorted(str(path) for path in directories)
        for index, (relative, identity) in enumerate(closure["members"].items()):
            destinations[identity] = directories[index % len(directories)] / relative
    for identity in copied["identities"]:
        if identity not in destinations:
            destinations[identity] = root / "individual" / identity
    copied["sources"] = {}
    for identity, original in record["identities"].items():
        destination = destinations[identity]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(original).read_bytes())
        copied["identities"][identity] = str(destination)
        copied["sources"][str(destination)] = record["sources"][original]
    reseal_location(copied)
    return copied


@pytest.fixture
def snapshot(captured, tmp_path):
    root = tmp_path / "snapshot"
    record = relocate(captured, root)
    assert HP.content_sha256(record, source_root=root) == captured["sha256"]
    return root, record


def test_capture_binds_real_installed_sources_and_explicit_inputs(captured, policy_inputs):
    controller, resources = policy_inputs
    assert captured["schema"] == HP.SCHEMA == "global_host_verification_policy_v3"
    assert captured["identities"]["controller/global"] == str(controller.resolve())
    for relative in HP.RESOURCE_FILES:
        assert captured["identities"][f"resource/contract/{relative}"] == str((resources / relative).resolve())
    for module in HP.V3_MODULES:
        identity = f"python/{module}"
        assert Path(captured["identities"][identity]).resolve() == HP.module_source_path(module).resolve()
    closure = captured["closures"]["merlin_experiments.phase2"]
    assert "host_policy.py" in closure["members"]
    assert "contracts.py" in closure["members"]
    for relative, identity in closure["members"].items():
        assert Path(captured["identities"][identity]) == Path(closure["roots"][0]) / relative
    perf = captured["closures"]["merlin.perf"]
    assert perf["members"]["lane_migration_qualifier.py"] == "python/merlin.perf.lane_migration_qualifier"
    namespace = importlib.import_module("merlin.perf")
    assert perf["roots"] == sorted(str(Path(path).resolve()) for path in namespace.__path__)
    for relative, identity in perf["members"].items():
        assert sum(Path(captured["identities"][identity]) == Path(root) / relative for root in perf["roots"]) == 1
    assert set(captured["sources"]) == set(captured["identities"].values())
    for path, digest in captured["sources"].items():
        assert contracts.sha256_file(Path(path)) == digest


def test_portable_identity_survives_different_snapshot_layouts(captured, tmp_path):
    first_root, second_root = tmp_path / "first", tmp_path / "nested" / "second"
    first, second = relocate(captured, first_root), relocate(captured, second_root, split=True)
    assert first["location_sha256"] != second["location_sha256"]
    assert first["sha256"] == second["sha256"] == captured["sha256"]
    assert HP.content_sha256(first, source_root=first_root) == HP.content_sha256(second, source_root=second_root)


def test_recorded_admission_never_discovers_live_modules(snapshot, monkeypatch):
    root, record = snapshot

    def forbidden(*args, **kwargs):
        pytest.fail("recorded policy admission must not discover live implementations")

    monkeypatch.setattr(HP, "module_source_path", forbidden)
    assert HP.content_sha256(record, source_root=root) == record["sha256"]


@pytest.mark.parametrize("mutation", ["changed", "missing", "linked", "escaped"])
def test_recorded_source_changes_are_refused(snapshot, tmp_path, mutation):
    root, record = snapshot
    identity = "controller/global"
    original = Path(record["identities"][identity])
    if mutation == "changed":
        original.write_text("POLICY = 'substituted'\n")
    elif mutation == "missing":
        original.unlink()
    else:
        outside = tmp_path / "outside.py"
        outside.write_bytes(original.read_bytes())
        if mutation == "linked":
            original.unlink()
            original.symlink_to(outside)
        else:
            record["identities"][identity] = str(outside)
            record["sources"][str(outside)] = record["sources"].pop(str(original))
            reseal_location(record)
    with pytest.raises(ValueError):
        HP.content_sha256(record, source_root=root)


@pytest.mark.parametrize("relative", HP.RESOURCE_FILES)
def test_each_selected_resource_is_part_of_recorded_identity(snapshot, relative):
    root, record = snapshot
    resource = Path(record["identities"][f"resource/contract/{relative}"])
    resource.write_bytes(resource.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="bytes changed"):
        HP.content_sha256(record, source_root=root)


@pytest.mark.parametrize(
    "namespace,filename",
    [("merlin_experiments.phase2", "contracts.py"), ("merlin.perf", "lane_migration_qualifier.py")],
)
@pytest.mark.parametrize("mutation", ["added", "removed", "changed", "linked_member", "linked_root", "escaped_member"])
def test_recorded_closure_membership_is_exact(snapshot, tmp_path, mutation, namespace, filename):
    root, record = snapshot
    closure = record["closures"][namespace]
    directory = Path(closure["roots"][0])
    member = directory / filename
    if mutation == "added":
        (directory / "unrecorded.py").write_text("UNRECORDED = True\n")
    elif mutation == "removed":
        member.unlink()
    elif mutation == "changed":
        member.write_bytes(member.read_bytes() + b"\n")
    elif mutation == "linked_root":
        saved = directory.with_name("retained-closure")
        directory.rename(saved)
        directory.symlink_to(saved, target_is_directory=True)
    elif mutation == "escaped_member":
        identity = closure["members"].pop(filename)
        closure["members"]["../" + filename] = identity
        reseal_location(record)
    else:
        outside = tmp_path / "outside-member.py"
        outside.write_bytes(member.read_bytes())
        member.unlink()
        member.symlink_to(outside)
    with pytest.raises(ValueError):
        HP.content_sha256(record, source_root=root)


@pytest.mark.parametrize("field", ["sha256", "location_sha256"])
def test_recorded_digest_cannot_be_substituted(snapshot, field):
    root, record = snapshot
    record[field] = "0" * 64
    with pytest.raises(ValueError):
        HP.content_sha256(record, source_root=root)


def test_selected_source_must_match_loaded_module_origin(policy_inputs, tmp_path, monkeypatch):
    controller, resources = policy_inputs
    module = "merlin.common.digest"
    importlib.import_module(module)
    original_resolver = HP.module_source_path
    shadow = tmp_path / "shadow-digest.py"
    shadow.write_bytes(original_resolver(module).read_bytes())
    monkeypatch.setattr(HP, "module_source_path", lambda name: shadow if name == module else original_resolver(name))
    with pytest.raises(ValueError):
        HP.build_record(controller_source=controller, contract_root=resources)


@pytest.mark.parametrize(
    "name", ["merlin.common.digest", "merlin_experiments.phase2.contracts", "merlin.perf.lane_migration_qualifier"]
)
def test_loaded_source_shadow_cannot_claim_the_selected_module(policy_inputs, tmp_path, monkeypatch, name):
    controller, resources = policy_inputs
    module = importlib.import_module(name)
    shadow = tmp_path / "shadow.py"
    shadow.write_bytes(Path(module.__file__).read_bytes())
    monkeypatch.setattr(module, "__file__", str(shadow))
    with pytest.raises(ValueError, match="loaded source origin"):
        HP.build_record(controller_source=controller, contract_root=resources)


@pytest.mark.parametrize("mutation", ["missing_identity", "unknown_identity", "member_rebound"])
def test_recorded_logical_ownership_cannot_be_substituted(snapshot, mutation):
    root, record = snapshot
    if mutation == "missing_identity":
        record["identities"].pop("controller/global")
    elif mutation == "unknown_identity":
        record["identities"]["controller/unapproved"] = record["identities"].pop("controller/global")
    else:
        closure = record["closures"]["merlin_experiments.phase2"]
        closure["members"]["contracts.py"] = closure["members"]["host_policy.py"]
    reseal_location(record)
    with pytest.raises(ValueError):
        HP.content_sha256(record, source_root=root)


@pytest.mark.parametrize("kind", ["controller", "resource"])
def test_capture_requires_real_explicit_inputs(policy_inputs, tmp_path, kind):
    controller, resources = policy_inputs
    path = controller if kind == "controller" else resources / HP.RESOURCE_FILES[0]
    saved = tmp_path / "saved-input"
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises(ValueError):
        HP.build_record(controller_source=controller, contract_root=resources)


def test_namespace_roots_cannot_shadow_even_identical_bytes(snapshot):
    root, record = snapshot
    closure = record["closures"]["merlin.perf"]
    original = Path(closure["roots"][0]) / "lane_migration_qualifier.py"
    shadow = root / "shadow-perf"
    shadow.mkdir()
    (shadow / original.name).write_bytes(original.read_bytes())
    closure["roots"].append(str(shadow))
    closure["roots"].sort()
    reseal_location(record)
    with pytest.raises(ValueError):
        HP.content_sha256(record, source_root=root)


@pytest.mark.parametrize("mutation", ["duplicate", "relative", "escape"])
def test_namespace_root_ownership_refuses_malformed_roots(snapshot, tmp_path, mutation):
    root, record = snapshot
    roots = record["closures"]["merlin.perf"]["roots"]
    if mutation == "duplicate":
        roots.append(roots[0])
    elif mutation == "relative":
        roots[0] = "relative/perf"
    else:
        outside = tmp_path / "outside-perf"
        outside.mkdir()
        roots.append(str(outside))
        roots.sort()
    reseal_location(record)
    with pytest.raises(ValueError):
        HP.content_sha256(record, source_root=root)


def test_frozen_historical_v2_decodes_without_live_discovery(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "historical_host_policy", Path(__file__).with_name("host_policy_fixtures.py")
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    record = helper.historical_v2(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("historical V2 must not discover current implementations")

    monkeypatch.setattr(HP, "module_source_path", forbidden)
    assert HP.content_sha256(record, source_root=tmp_path) == helper.V2_CONTENT_SHA256
    assert SI.host_policy_content_sha256(record, source_root=tmp_path) == helper.V2_CONTENT_SHA256


@pytest.mark.parametrize("schema", ["global_host_verification_policy_v4", "unknown", ""])
def test_unknown_schema_cannot_fall_back_to_historical_decoding(snapshot, schema):
    root, record = snapshot
    record["schema"] = schema
    for verify in (HP.content_sha256, SI.host_policy_content_sha256):
        with pytest.raises(ValueError):
            verify(record, source_root=root)


@pytest.mark.parametrize("duplicate", [False, True])
def test_capture_covers_every_active_perf_root(policy_inputs, tmp_path, monkeypatch, duplicate):
    controller, resources = policy_inputs
    before = HP.build_record(controller_source=controller, contract_root=resources)
    namespace = importlib.import_module("merlin.perf")
    additional = tmp_path / "additional-perf"
    additional.mkdir()
    name = "lane_migration_qualifier" if duplicate else "new_qualifier"
    (additional / f"{name}.py").write_text("POLICY = 'newly active root'\n")
    monkeypatch.setattr(namespace, "__path__", [*namespace.__path__, str(additional)])
    if duplicate:
        with pytest.raises(ValueError):
            HP.build_record(controller_source=controller, contract_root=resources)
    else:
        after = HP.build_record(controller_source=controller, contract_root=resources)
        assert after["sha256"] != before["sha256"]
        assert after["identities"]["python/merlin.perf.new_qualifier"] == str(additional / "new_qualifier.py")


@pytest.mark.parametrize("descendant", ["__init__.py", "child.py"])
def test_live_namespace_refuses_module_package_collisions(policy_inputs, tmp_path, monkeypatch, descendant):
    controller, resources = policy_inputs
    namespace = importlib.import_module("merlin.perf")
    additional = tmp_path / "conflicting-perf"
    package = additional / "collision"
    package.mkdir(parents=True)
    (additional / "collision.py").write_text("MODULE = True\n")
    (package / descendant).write_text("PACKAGE = True\n")
    monkeypatch.setattr(namespace, "__path__", [*namespace.__path__, str(additional)])
    with pytest.raises(ValueError, match="shadow|collision|duplicate"):
        HP.build_record(controller_source=controller, contract_root=resources)
