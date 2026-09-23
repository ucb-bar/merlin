"""Explicit installed portfolio deployment admission without launching a worker."""

import copy
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from merlin_experiments.phase2 import portfolio_cli as CLI


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("CLI admission tests cannot launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


@pytest.fixture
def deployment(tmp_path):
    sandbox = tmp_path / "sandbox-declaration.json"
    sandbox.write_text('{"execution_policy": {}}')
    document = {
        "schema": "merlin.portfolio-deployment.v1",
        "target": "synthetic",
        "source_root": str(tmp_path / "source"),
        "output_root": str(tmp_path / "storage"),
        "lease_path": str(tmp_path / "lease/compiler.lock"),
        "source_roots": ["python", "contract"],
        "python_roots": ["python"],
        "legacy_roots": [],
        "internal_aliases": {},
        "exclude_paths": [],
        "provider_root": None,
        "functional_runs_root": str(tmp_path / "functional"),
        "contract_root": "contract",
        "compiler_shared_source_root": "python/merlin",
        "sandbox_root": str(tmp_path / "sandbox"),
        "sandbox_declaration": str(sandbox),
        "sandbox_declaration_sha256": hashlib.sha256(sandbox.read_bytes()).hexdigest(),
    }
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(document))
    return path, document


def test_deployment_decodes_without_rediscovering_original_source_tree(deployment):
    path, document = deployment
    assert not Path(document["source_root"]).exists()
    assert CLI.load_deployment(path) == document


@pytest.mark.parametrize("collision", ["candidate", "output", "lease", "source_snapshot", "transport"])
def test_direct_cli_refuses_writable_inputs(deployment, collision):
    path, document = deployment
    args = SimpleNamespace(output=path.parent / "stage", candidate=path.parent / "candidate")
    declared = {"deployment": path}
    if collision == "candidate":
        args.candidate = Path(document["source_root"]) / "python/candidate"
    elif collision == "output":
        args.output = Path(document["sandbox_root"]) / "stage"
    elif collision == "lease":
        document["lease_path"] = str(path)
    elif collision == "source_snapshot":
        declared["resource"] = path.parent / "stage.source/resource.json"
    else:
        declared["resource"] = path.parent / "stage.transport/resource.json"
    with pytest.raises(ValueError, match="overlaps immutable"):
        CLI._admit_destinations(args, document, declared)


def test_common_output_container_can_also_hold_retained_inputs(deployment):
    path, document = deployment
    document["output_root"] = str(path.parent)
    args = SimpleNamespace(output=path.parent / "stage", candidate=path.parent / "candidate")
    CLI._admit_destinations(args, document, {"deployment": path})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "unknown"),
        ("target", ""),
        ("source_root", "relative"),
        ("output_root", "/tmp/../escape"),
        ("lease_path", "relative.lock"),
        ("source_roots", []),
        ("python_roots", []),
        ("source_roots", ["../escape"]),
        ("python_roots", ["/absolute"]),
        ("legacy_roots", ["a/../b"]),
        ("contract_root", "../contract"),
        ("compiler_shared_source_root", "/shared"),
        ("internal_aliases", {"../alias": "python"}),
        ("exclude_paths", ["../outside"]),
        ("provider_root", "relative"),
        ("sandbox_declaration_sha256", "not-a-digest"),
    ],
)
def test_malformed_or_escaping_deployment_selections_refuse(deployment, field, value):
    path, document = deployment
    document[field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        CLI.load_deployment(path)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_deployment_schema_fields_are_exact(deployment, mutation):
    path, document = deployment
    if mutation == "missing":
        document.pop("sandbox_root")
    else:
        document["worker_module"] = "operator_selected_code"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        CLI.load_deployment(path)


def arguments(deployment, *, worker=False):
    path, _document = deployment
    args = [
        "--deployment",
        str(path),
        "--campaign-config",
        str(path.parent / "campaign.json"),
        "--candidate",
        str(path.parent / "candidate"),
        "--output",
        str(path.parent / "stage"),
    ]
    return [*args, "--source-worker"] if worker else args


def test_worker_requires_guarded_startup_not_an_ambient_snapshot_claim(deployment, monkeypatch):
    monkeypatch.setattr(CLI.FP, "active_source_identity", lambda: None)
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(deployment[0].parent))
    monkeypatch.setattr(CLI.WORKER, "run", lambda *a, **k: pytest.fail("unguarded worker reached execution"))
    with pytest.raises((ValueError, RuntimeError)):
        CLI.main(arguments(deployment, worker=True))


@pytest.mark.parametrize("with_provider", [False, True])
def test_parent_composes_fixed_worker_and_explicit_input_ownership(deployment, monkeypatch, with_provider):
    path, document = deployment
    descriptor = path.parent / "descriptor.yaml"
    descriptor.write_text("target: synthetic\n")
    campaign = path.parent / "campaign.json"
    campaign.write_text(json.dumps({"config": {"descriptor": str(descriptor)}}))
    if with_provider:
        document["provider_root"] = str(path.parent / "selected-support")
        path.write_text(json.dumps(document))
    admitted = []

    def admission(selected, decoded):
        assert selected == path and decoded == document
        admitted.append(selected)
        return {"deployment": str(path)}

    def provider(selected):
        assert selected == document["provider_root"]
        return SimpleNamespace(role=CLI.ProviderRole.SUPPORT, target="synthetic", root=Path(selected))

    def launch(invocation, *, deployment):
        assert admitted == [path]
        assert deployment.worker_entrypoint == (sys.executable, "-m", CLI.MODULE)
        assert deployment.source_root == Path(document["source_root"])
        assert deployment.output_root == Path(document["output_root"])
        assert deployment.worker_python_roots == ("python",)
        assert deployment.declared_inputs == {
            "deployment": path,
            "campaign_config": campaign,
            "sandbox_declaration": Path(document["sandbox_declaration"]),
            "descriptor": descriptor,
        }
        assert deployment.declared_input_sha256 == {
            name: hashlib.sha256(selected.read_bytes()).hexdigest()
            for name, selected in deployment.declared_inputs.items()
        }
        if with_provider:
            assert deployment.selected_provider == {
                "target": "synthetic",
                "resolved_target": "synthetic",
                "kind": "external",
                "source": document["provider_root"],
            }
        else:
            assert deployment.selected_provider is None
        return 7

    monkeypatch.setattr(CLI, "_source_inputs", admission)
    monkeypatch.setattr(CLI, "read_provider", provider)
    monkeypatch.setattr(CLI.LAUNCH, "launch", launch)
    assert CLI.main(arguments((path, document))) == 7


@pytest.mark.parametrize("corrupt_sandbox", [False, True])
def test_worker_uses_sealed_deployment_and_campaign_not_mutated_live_files(deployment, monkeypatch, corrupt_sandbox):
    live_path, document = deployment
    Path(document["sandbox_root"]).mkdir()
    snapshot = live_path.parent / "snapshot"
    archived = snapshot / "inputs"
    archived.mkdir(parents=True)
    sealed_deployment = archived / "deployment.json"
    sealed_deployment.write_text(json.dumps(copy.deepcopy(document)))
    campaign = archived / "campaign.json"
    campaign.write_text(json.dumps({"descriptor": str(live_path.parent / "descriptor.yaml"), "model": "sealed-model"}))
    descriptor = archived / "descriptor.yaml"
    descriptor.write_text("target: synthetic\n")
    declaration = archived / "sandbox.json"
    declaration.write_bytes(Path(document["sandbox_declaration"]).read_bytes())
    if corrupt_sandbox:
        declaration.write_bytes(b"changed")
    for root in ("contract", "python/merlin"):
        (snapshot / root).mkdir(parents=True)
    worker = snapshot / "python/merlin_experiments/phase2/portfolio_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# sealed synthetic worker owner\n")
    receipt = {
        "schema": CLI.SNAP.SCHEMA,
        "python_roots": ["python"],
        "source_root": document["source_root"],
        "source_roots": document["source_roots"],
        "legacy_roots": [],
        "internal_aliases": {},
        "selected_provider": None,
        "directories": ["contract", "python/merlin"],
        "files": {str(worker.relative_to(snapshot)): hashlib.sha256(worker.read_bytes()).hexdigest()},
    }
    seal = CLI.SNAP.seal(snapshot, "snapshot", receipt)
    identity = {"path": str(seal), "sha256": hashlib.sha256(seal.read_bytes()).hexdigest()}
    monkeypatch.setattr(CLI.FP, "active_source_identity", lambda: identity)
    monkeypatch.setattr(CLI.SNAP, "verify", lambda root: receipt)
    mapping = {
        "deployment": sealed_deployment,
        "campaign_config": campaign,
        "sandbox_declaration": declaration,
        "descriptor": descriptor,
    }
    remapped = []

    def remap(root, observed, original, *, name):
        assert root == snapshot and observed is receipt
        remapped.append(name)
        return mapping[name]

    sandbox_inputs = object()
    restored = []

    def restore(root, policy):
        restored.append(root)
        assert root == Path(document["sandbox_root"]) and policy == {}
        return sandbox_inputs

    target = SimpleNamespace(target="synthetic")

    def load_target(path, *, source_root):
        assert path == descriptor and source_root == snapshot
        return target

    def run(invocation, config, *, context):
        assert not corrupt_sandbox
        assert config["model"] == "sealed-model"
        assert invocation.args.campaign_config == campaign
        assert context.snapshot_root == snapshot
        assert context.controller_source == worker
        assert context.contract_root == snapshot / "contract"
        assert context.compiler_shared_source_root == snapshot / "python/merlin"
        assert context.sandbox_inputs is sandbox_inputs
        assert context.target_experiment is target
        return 9

    monkeypatch.setattr(CLI.SNAP, "remap_input", remap)
    monkeypatch.setattr(CLI.Q, "restore", restore)
    monkeypatch.setattr(CLI, "load_target_experiment", load_target)
    monkeypatch.setattr(CLI.WORKER, "__file__", str(worker))
    monkeypatch.setattr(CLI.WORKER, "run", run)
    # The worker must not decode these replaced live records at all.
    live_path.write_text("invalid live deployment")
    (live_path.parent / "campaign.json").write_text("invalid live campaign")
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", "/unselected")
    monkeypatch.setenv("MERLIN_SCHEMAS_DIR", "/unselected")
    if corrupt_sandbox:
        with pytest.raises(ValueError, match="sandbox declaration differs"):
            CLI.main(arguments(deployment, worker=True))
        assert restored == []
    else:
        assert CLI.main(arguments(deployment, worker=True)) == 9
        assert remapped == ["deployment", "campaign_config", "sandbox_declaration", "descriptor"]


@pytest.fixture
def synthetic_owners(deployment, monkeypatch):
    path, document = deployment
    source = Path(document["source_root"])
    owners = {}
    names = (
        "merlin",
        "merlin.common.paths",
        "merlin.common.source_membership",
        "merlin_experiments",
        CLI.MODULE,
        CLI.WORKER.__name__,
    )
    loaded = {}
    for name in names:
        suffix = name.replace(".", "/")
        relative = suffix + "/__init__.py" if name in {"merlin", "merlin_experiments"} else suffix + ".py"
        selected = source / "python" / relative
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text("# synthetic ordinary selected owner\n")
        owners[name] = selected
        module = ModuleType(name)
        module.__file__ = str(selected)
        module.__spec__ = SimpleNamespace(origin=str(selected))
        loaded[name] = module
    (source / "contract").mkdir()
    Path(document["sandbox_root"]).mkdir()
    document["source_roots"] = ["python/merlin", "python/merlin_experiments", "contract"]
    path.write_text(json.dumps(document))
    # Replace only observations of loaded modules; discovery walks real files.
    monkeypatch.setattr(CLI, "sys", SimpleNamespace(executable=sys.executable, modules=loaded))
    monkeypatch.setattr(CLI, "module_source_path", lambda name: owners[name])
    return path, document, owners


def test_real_source_admission_accepts_parent_import_root_with_selected_package_children(synthetic_owners):
    path, document, owners = synthetic_owners
    inventory = CLI.deployment_source_inputs(path)
    assert inventory["deployment"] == str(path)
    assert inventory["sandbox_declaration"] == document["sandbox_declaration"]
    assert inventory["source_root:python/merlin"] == str(Path(document["source_root"]) / "python/merlin")
    assert inventory["source_root:python/merlin_experiments"] == str(
        Path(document["source_root"]) / "python/merlin_experiments"
    )
    assert document["output_root"] not in inventory.values()
    assert document["lease_path"] not in inventory.values()


def test_real_source_admission_refuses_missing_required_owner(synthetic_owners):
    path, document, owners = synthetic_owners
    selected = owners[CLI.MODULE]
    selected.rename(selected.with_suffix(".removed"))
    with pytest.raises(ValueError, match="module owner"):
        CLI.deployment_source_inputs(path)


def test_real_source_admission_refuses_duplicate_module_ownership(synthetic_owners):
    path, document, owners = synthetic_owners
    duplicate = Path(document["source_root"]) / "duplicate/merlin/__init__.py"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(owners["merlin"].read_bytes())
    document["source_roots"].append("duplicate/merlin")
    document["python_roots"].append("duplicate")
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="duplicate Python ownership"):
        CLI.deployment_source_inputs(path)
