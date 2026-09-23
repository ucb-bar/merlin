"""Selected provider bytes/resources use the existing performance source seal.

Synthetic cold Python children only: no services, hardware, or live providers.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments import frozen_python
from merlin_experiments import source_snapshot as SNAP
from merlin_experiments.phase2 import portfolio_launch as PL

from merlin.common.paths import module_source_path, repo_root

VERIFIER = module_source_path("merlin_experiments.source_snapshot")


def write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def writable(root):
    for path in (root, *root.rglob("*")):
        if not path.is_symlink():
            path.chmod(0o700 if path.is_dir() else 0o600)


@pytest.fixture
def selected(tmp_path):
    source, provider, frozen = (tmp_path / name for name in ("source", "provider", "frozen"))
    write(source, "src/merlin/__init__.py", "")
    write(source, "src/merlin/targetgen/__init__.py", "")
    write(source, "src/merlin/targetgen/plugins.py", module_source_path("merlin.targetgen.plugins").read_text())
    for name in (
        "common.paths",
        "common.digest",
        "targetgen.providers",
        "targetgen.target_registry",
        "targetgen.rtl.facts",
    ):
        write(source, "src/merlin/" + name.replace(".", "/") + ".py", module_source_path("merlin." + name).read_text())
    write(source, "src/merlin/common/__init__.py", "")
    write(source, "src/merlin/targetgen/rtl/__init__.py", "")
    write(source, "packages/merlin-experiments/src/merlin_experiments/__init__.py", "")
    write(
        source,
        "packages/merlin-experiments/src/merlin_experiments/frozen_python.py",
        Path(frozen_python.__file__).read_text(),
    )
    write(
        provider,
        "provider.yaml",
        "schema: merlin.provider.v1\nid: selected\nrole: support\ntarget: fixture\ncontract: contracts/custom.yaml\n",
    )
    write(provider, "contracts/custom.yaml", "name: fixture\n")
    write(
        provider,
        "pkg/__init__.py",
        "from importlib.resources import files\nVALUE=files(__package__).joinpath('data.txt').read_text()\n",
    )
    write(provider, "pkg/data.txt", "selected bytes")
    write(provider, "shadow_dependency.py", "raise AssertionError('global provider lookup')\n")
    selection = {"target": "fixture", "resolved_target": "fixture", "kind": "external", "source": str(provider)}
    SNAP.create(
        source,
        frozen,
        output_root=tmp_path / "output",
        source_roots=("src", "packages/merlin-experiments/src"),
        target_name="fixture",
        provider=selection,
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
    )
    yield source, provider, frozen
    writable(frozen)


def test_guarded_provider_import_resources_survive_source_removal(selected, monkeypatch):
    source, provider, frozen = selected
    record = SNAP.verify(frozen)
    original_contract = provider / "contracts/custom.yaml"
    provider.rename(provider.with_name("removed-provider"))
    source.rename(source.with_name("removed-source"))
    monkeypatch.setenv("MERLIN_TARGET_PATH", "/nonexistent-live-provider")
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", "/wrong-contract")
    monkeypatch.setenv("MERLIN_RTL_FACTS", "/wrong-facts")
    monkeypatch.setenv("MERLIN_TARGETS_DIR", "/wrong-targets")
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", "/wrong-schema-contract")
    monkeypatch.setenv("MERLIN_SCHEMAS_DIR", "/wrong-schemas")
    command = frozen_python.python_command(
        frozen,
        [
            sys.executable,
            "-c",
            """
import os, sys, importlib.util
from pathlib import Path
from merlin.targetgen.plugins import load_module
from merlin.targetgen.target_registry import resolve
root=Path(os.environ['MERLIN_TARGET_PATH'])
assert root.name == '_selected_provider'
assert Path(os.environ['MERLIN_TARGETS_DIR']) == root.parent / 'merlin/targets'
assert Path(os.environ['MERLIN_CONTRACT_DIR']) == root.parent / 'merlin/contract'
assert Path(os.environ['MERLIN_SCHEMAS_DIR']) == root.parent / 'merlin/schemas'
assert not os.environ['MERLIN_TARGET_CONTRACT'] and not os.environ['MERLIN_RTL_FACTS']
module=load_module(root, 'pkg', package_name='fixture')
assert module.VALUE == 'selected bytes'
assert str(root) not in sys.path
assert importlib.util.find_spec('shadow_dependency') is None
selected=resolve('fixture')
assert selected.base == root
assert selected.contract_path == root / 'contracts/custom.yaml'
print(module.__file__)
""",
        ],
        verifier_source=VERIFIER,
    )
    child = subprocess.run(command, capture_output=True, text=True, timeout=20, env=dict(os.environ))
    assert child.returncode == 0, child.stderr
    assert str(frozen / "_selected_provider/pkg/__init__.py") in child.stdout
    assert SNAP.remap_input(frozen, record, original_contract).read_text() == "name: fixture\n"


@pytest.mark.parametrize("mutation", ["edit", "add", "remove"])
def test_provider_payload_drift_is_refused(selected, mutation):
    _, _, frozen = selected
    writable(frozen)
    path = frozen / "_selected_provider/pkg/data.txt"
    if mutation == "edit":
        path.write_text("different")
    elif mutation == "add":
        (path.parent / "extra.txt").write_text("new")
    else:
        path.unlink()
    for p in (frozen, *frozen.rglob("*")):
        if not p.is_symlink():
            p.chmod(p.stat().st_mode & ~0o222)
    with pytest.raises(SNAP.SnapshotError, match="source"):
        SNAP.verify(frozen)


@pytest.mark.parametrize("schema", ["merlin.performance-source-snapshot.v2", "merlin.performance-source-snapshot.v1"])
def test_legacy_selection_requires_newly_frozen_ownership(schema, monkeypatch, tmp_path):
    for external in ("", str(tmp_path / "external")):
        monkeypatch.setenv("MERLIN_TARGET_PATH", external)
        with pytest.raises(SNAP.SnapshotError, match="newly frozen"):
            SNAP.provider_environment(tmp_path, {"schema": schema})


def test_actual_v2_launch_refused_without_modifying_original(selected, monkeypatch):
    _, _, frozen = selected
    old_marker, receipt = SNAP.load_seal(frozen, "snapshot")
    writable(frozen)
    old_marker.unlink()
    receipt["schema"] = "merlin.performance-source-snapshot.v2"
    receipt.pop("selected_provider")
    marker = SNAP.seal(frozen, "snapshot", receipt)
    for path in (frozen, *frozen.rglob("*")):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)
    original = marker.read_bytes()
    for external in ("", "/external-provider"):
        monkeypatch.setenv("MERLIN_TARGET_PATH", external)
        with pytest.raises(RuntimeError, match="newly frozen"):
            frozen_python.python_command(frozen, [sys.executable, "-c", "pass"], verifier_source=VERIFIER)
    assert marker.read_bytes() == original
    assert "selected_provider" not in json.loads(marker.read_text())


def test_native_selection_survives_hostile_resource_environment(selected, monkeypatch, tmp_path):
    source, _, _ = selected
    write(source, "merlin/targets/fixture/contracts/target_contract.yaml", "name: fixture\n")
    native = source / "merlin/targets/fixture"
    destination = tmp_path / "native"
    SNAP.create(
        source,
        destination,
        output_root=tmp_path / "output",
        source_roots=("src", "packages/merlin-experiments/src", "merlin/targets"),
        target_name="fixture",
        provider={"target": "fixture", "resolved_target": "fixture", "kind": "reference", "source": str(native)},
        python_roots=("packages/merlin-experiments/src", "src"),
        legacy_roots=(),
    )
    monkeypatch.setenv("MERLIN_TARGET_PATH", "/wrong-provider")
    monkeypatch.setenv("MERLIN_TARGETS_DIR", "/wrong-targets")
    try:
        command = frozen_python.python_command(
            destination,
            [
                sys.executable,
                "-c",
                "from merlin.targetgen.target_registry import resolve; import os; from pathlib import Path; "
                "p=resolve('fixture'); assert p.kind == 'reference'; "
                "assert p.base == Path(os.environ['MERLIN_REPO_ROOT'])/'merlin/targets/fixture'",
            ],
            verifier_source=VERIFIER,
        )
        child = subprocess.run(command, capture_output=True, text=True, timeout=20)
        assert child.returncode == 0, child.stderr
    finally:
        writable(destination)


def test_selection_precedence_is_captured_not_rediscovered(selected, monkeypatch, tmp_path):
    _, provider, _ = selected
    from merlin.targetgen.target_registry import TargetCollisionError

    other = tmp_path / "other"
    write(other, "provider.yaml", (provider / "provider.yaml").read_text())
    write(other, "contracts/custom.yaml", "name: fixture\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", os.pathsep.join(map(str, (provider, other))))
    assert SNAP.selected_provider("fixture")["source"] == str(provider)
    monkeypatch.setenv("MERLIN_TARGET_PATH", os.pathsep.join(map(str, (other, provider))))
    assert SNAP.selected_provider("fixture")["source"] == str(other)
    shelf = tmp_path / "shelf"
    shelf.mkdir()
    (shelf / "one").symlink_to(provider, target_is_directory=True)
    (shelf / "two").symlink_to(other, target_is_directory=True)
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(shelf))
    with pytest.raises(TargetCollisionError):
        SNAP.selected_provider("fixture")


@pytest.mark.parametrize("escape", [False, True])
def test_provider_aliases_are_copied_only_within_declared_root(selected, tmp_path, escape):
    source, provider, _ = selected
    target = write(tmp_path, "outside.txt", "external") if escape else provider / "pkg/data.txt"
    (provider / "alias.txt").symlink_to(target)
    destination = tmp_path / "alias-snapshot"
    selection = {"target": "fixture", "resolved_target": "fixture", "kind": "external", "source": str(provider)}
    try:
        if escape:
            with pytest.raises(SNAP.SnapshotError, match="escapes"):
                SNAP.create(
                    source,
                    destination,
                    output_root=tmp_path / "out",
                    source_roots=("src",),
                    target_name="fixture",
                    provider=selection,
                    python_roots=("src",),
                    legacy_roots=(),
                )
        else:
            SNAP.create(
                source,
                destination,
                output_root=tmp_path / "out",
                source_roots=("src",),
                target_name="fixture",
                provider=selection,
                python_roots=("src",),
                legacy_roots=(),
            )
            SNAP.verify(destination)
            alias = destination / "_selected_provider/alias.txt"
            assert not alias.is_symlink() and alias.read_text() == "selected bytes"
    finally:
        if destination.exists():
            writable(destination)


def test_declared_external_inputs_survive_unlink_and_internal_inputs_are_not_duplicated(selected, tmp_path):
    source, provider, _ = selected
    external = write(tmp_path, "upstream/descriptor.yaml", "target: fixture\n")
    internal = provider / "contracts/custom.yaml"
    destination = tmp_path / "input-snapshot"
    selection = {"target": "fixture", "resolved_target": "fixture", "kind": "external", "source": str(provider)}
    try:
        SNAP.create(
            source,
            destination,
            output_root=tmp_path / "out",
            source_roots=("src",),
            target_name="fixture",
            provider=selection,
            declared_inputs={"descriptor": external, "contract": internal},
            python_roots=("src",),
            legacy_roots=(),
        )
        receipt = SNAP.verify(destination)
        assert receipt["declared_inputs"]["contract"]["snapshot"] == "_selected_provider/contracts/custom.yaml"
        external.unlink()
        assert SNAP.remap_input(destination, receipt, external, name="descriptor").read_text() == "target: fixture\n"
        with pytest.raises(SNAP.SnapshotError, match="declaration"):
            SNAP.remap_input(destination, receipt, internal, name="descriptor")
        writable(destination)
        SNAP.remap_input(destination, receipt, external, name="descriptor").write_text("tampered")
        for path in (destination, *destination.rglob("*")):
            if not path.is_symlink():
                path.chmod(path.stat().st_mode & ~0o222)
        with pytest.raises(SNAP.SnapshotError, match="source changed"):
            SNAP.verify(destination)
    finally:
        writable(destination)


@pytest.mark.parametrize("invalid", ["escaping_name", "symlink", "directory", "changing_source"])
def test_declared_inputs_refuse_invalid_or_unstable_sources(selected, tmp_path, monkeypatch, invalid):
    source, _, _ = selected
    external = write(tmp_path, "upstream/data.json", "{}")
    name = "../escape" if invalid == "escaping_name" else "facts"
    if invalid == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(external)
        external = alias
    elif invalid == "directory":
        external = external.parent
    elif invalid == "changing_source":
        original = SNAP.shutil.copy2

        def mutate(src, dst):
            result = original(src, dst)
            if Path(src) == external:
                external.write_text('{"changed":true}')
            return result

        monkeypatch.setattr(SNAP.shutil, "copy2", mutate)
    destination = tmp_path / "bad-inputs"
    try:
        with pytest.raises(SNAP.SnapshotError):
            SNAP.create(
                source,
                destination,
                output_root=tmp_path / "out",
                source_roots=("src",),
                declared_inputs={name: external},
                python_roots=("src",),
                legacy_roots=(),
            )
        assert not list(destination.glob("snapshot.*.json"))
    finally:
        if destination.exists():
            writable(destination)


@pytest.mark.parametrize("creator", ["suite", "global"])
def test_actual_creators_freeze_selected_provider_and_declared_inputs(selected, tmp_path, monkeypatch, creator):
    """Exercise creator wiring; scientific admission is explicitly prequalified.

    Stop the global launcher immediately after its real snapshot is sealed, before
    transport, authoring or resource-managed process execution.
    """
    import importlib
    from types import SimpleNamespace

    source, provider, _ = selected
    monkeypatch.syspath_prepend(str(repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"))
    suite = importlib.import_module("perf_suite")
    layout_owner = importlib.import_module("source_snapshot_layout")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    roots = ("src", "packages/merlin-experiments/src", "merlin/contract", str(suite.SCRIPTS))
    for relative in suite.EVALUATION_FILES.values():
        write(source, str(relative), "shared evaluator fixture\n")
    layout = {
        "source_roots": roots,
        "python_roots": ("packages/merlin-experiments/src", "src"),
        "legacy_roots": (str(suite.SCRIPTS),),
        "internal_aliases": {},
        "exclude_paths": (),
    }
    monkeypatch.setattr(layout_owner, "snapshot_layout", lambda *a, **kw: layout)
    monkeypatch.setattr(suite, "snapshot_layout", lambda *a, **kw: layout)
    descriptor = write(tmp_path, "upstream/descriptor.yaml", "target: fixture\n")
    price = write(tmp_path, "upstream/prices.json", "{}")
    output = tmp_path / "outputs"
    output.mkdir()
    config = {"descriptor": str(descriptor), "telemetry_price_table": str(price)}
    if creator == "suite":
        monkeypatch.setattr(suite, "repo_root", lambda: source)
        monkeypatch.setattr(suite, "out_dir", lambda: output)
        monkeypatch.setattr(suite, "artifacts_dir", lambda: output)
        monkeypatch.setattr(suite, "load_target_experiment", lambda _: SimpleNamespace(target="fixture"))
        monkeypatch.setattr(suite, "claim_groups", lambda *_: [{"family": "fixture", "members": ["one"]}])
        for name in ("rtl_facts", "perf_profile", "gsim_certificate", "functional_gsim_certificate"):
            config[name] = str(write(tmp_path, f"upstream/{name}.json", "{}"))
        config.update(
            suite_id="fixture_suite",
            members=["one"],
            functional_run_id="qualified-fixture",
            functional_submission_sha256="a" * 64,
        )
        root = output / "suite"
        suite.prepare(root, config)
        snapshot = root / "source"
        manifest = suite.load(root)
        assert manifest["evaluation_boundary"]["selected_provider"]["source"] == str(provider)
        assert manifest["campaigns"][0]["config"]["descriptor"].startswith(str(snapshot))
    else:
        launcher = importlib.import_module("launch_global_agent_experiment")
        monkeypatch.setattr(launcher, "repo_root", lambda: source)
        monkeypatch.setattr(launcher, "__file__", str(source / suite.SCRIPTS / "launch_global_agent_experiment.py"))
        monkeypatch.setattr(PL, "acquire_host_resource_lease", lambda _, **kwargs: (tmp_path / "lease").open("w"))
        monkeypatch.setattr(PL, "sample_host_memory", lambda: {})
        monkeypatch.setattr(PL, "violations", lambda *_: [])
        original = SNAP.create

        class StopAfterSnapshot(Exception):
            pass

        def stop(*args, **kwargs):
            original(*args, **kwargs)
            raise StopAfterSnapshot

        monkeypatch.setattr(SNAP, "create", stop)
        config_file = write(tmp_path, "config.json", json.dumps(config))
        before = config_file.read_bytes()
        with pytest.raises(StopAfterSnapshot):
            launcher.main(
                [
                    "--campaign-config",
                    str(config_file),
                    "--candidate",
                    str(tmp_path / "candidate"),
                    "--output",
                    str(output / "global"),
                ]
            )
        assert config_file.read_bytes() == before
        snapshot = output / "global.source"
    try:
        record = SNAP.verify(snapshot)
        descriptor.unlink()
        price.unlink()
        assert SNAP.remap_input(snapshot, record, descriptor, name="descriptor").read_text() == "target: fixture\n"
        assert SNAP.remap_input(snapshot, record, price, name="telemetry_price_table").read_text() == "{}"
    finally:
        writable(snapshot)
