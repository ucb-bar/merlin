"""Frozen fusion checks retain only their declared private exclusion metadata."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments import source_snapshot
from merlin_experiments.phase0 import composition_generation as generation
from merlin_experiments.phase0 import freeze, fusion_cases, fusion_validation
from merlin_experiments.phase1.source_inputs import fingerprint

from merlin.targetgen.composition_corpus import held_out_shapes


@pytest.fixture
def selected(tmp_path, monkeypatch):
    # This fixture owns only synthetic exclusion metadata, with no host RTL selections.
    from merlin.common import provenance
    from merlin.targetgen.rtl import semantic_facts

    monkeypatch.setattr(provenance, "load_pins", lambda: {})
    monkeypatch.setattr(provenance, "load_artifacts", lambda: {})
    monkeypatch.setattr(semantic_facts, "load_probes", lambda _: {})
    root = tmp_path / "checkout"
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(root))
    module = root / "src/merlin/__init__.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    manifest = root / "merlin/contract/capsules/MANIFEST.yaml"
    capsule = manifest.parent / "_model_layers/private/capsule.yaml"
    capsule.parent.mkdir(parents=True)
    capsule.write_text(
        yaml.safe_dump(
            dict(
                operation=dict(op="matmul", attributes=dict(lhs="A", weight="W")),
                inputs=[dict(name="A", role="input", shape=[3, 5]), dict(name="W", role="weight", shape=[5, 7])],
            )
        )
    )
    manifest.write_text(yaml.safe_dump(dict(generated=["_model_layers/private", "unrelated"])))
    (manifest.parent / "unrelated").mkdir()
    (manifest.parent / "unrelated/secret.bin").write_bytes(b"must not be frozen")
    (capsule.parent / "golden.npy").write_bytes(b"must not be frozen")
    descriptor = root / "descriptor.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            dict(
                workload_spec=dict(
                    synthesize_compositions=True,
                    fusion_validation=dict(path="record.json", sha256="a" * 64),
                )
            )
        )
    )
    software = root / "software.yaml"
    software.write_text("{}\n")
    command = dict(
        inputs=dict(descriptor=str(descriptor), software_spec=str(software)),
        entrypoint=str(module),
        argv=[str(module)],
        env={},
    )
    evidence = SimpleNamespace(
        software_spec={},
        contract={},
        source_paths=[],
        source_snapshots=[],
        views_json=b"{}",
        raw_facts_sha256=None,
        status="fixture",
        diagnostics=[],
    )
    monkeypatch.setattr(freeze, "selection", lambda *args: evidence)
    paths, command["phase0_evidence"] = freeze.selected_inputs(command, "fixture")
    paths.update(
        {"phase0:descriptor": str(descriptor), "phase0:software_spec": str(software), "phase0:entrypoint": str(module)}
    )
    plan = dict(
        phases={"0": command},
        input_paths=paths,
        inputs={name: dict(path=path, sha256=fingerprint(path)) for name, path in paths.items()},
        target="fixture",
        run_dir=str(tmp_path / "run"),
        storage_root=str(tmp_path / "out"),
    )
    provider = tmp_path / "provider"
    provider.mkdir()
    (provider / "__init__.py").write_text("")
    monkeypatch.setattr(
        source_snapshot,
        "selected_provider",
        lambda _: dict(
            target="fixture",
            resolved_target="fixture",
            kind="external",
            source=str(provider),
        ),
    )
    monkeypatch.setattr(freeze, "_source_layout", lambda: (root, ("src/merlin", "merlin/contract"), ("src",)))
    monkeypatch.setattr(freeze, "module_source_path", lambda _: module)
    monkeypatch.setattr(freeze.frozen_python, "python_command", lambda *args, **kwargs: [])
    from merlin_experiments import runner

    monkeypatch.setattr(runner, "_phase0_operator_inputs", lambda _: {})

    def export(_, destination):
        destination.mkdir(parents=True)
        return dict(sources=[])

    monkeypatch.setattr(freeze, "export_evidence", export)
    return SimpleNamespace(
        root=root,
        manifest=manifest,
        capsule=capsule,
        descriptor=descriptor,
        command=command,
        evidence=evidence,
        plan=plan,
    )


def test_frozen_private_exclusions_survive_live_removal_and_bind_suite(selected, monkeypatch, tmp_path):
    frozen = freeze.stage(selected.plan)
    snapshot = Path(frozen["phase0_source_snapshot"])
    receipt = source_snapshot.verify(snapshot)
    private = [row for name, row in frozen["input_paths"].items() if name.startswith("phase0:private_exclusion:")]
    assert len(private) == 2
    assert all(Path(path).is_relative_to(snapshot / source_snapshot.INPUT_ROOT) for path in private)
    canonical = snapshot / "merlin/contract/capsules/MANIFEST.yaml"
    assert canonical.resolve() == Path(private[0])
    assert len(receipt["internal_aliases"]) == 1
    assert not (canonical.parent / "unrelated").exists()
    assert not (canonical.parent / "_model_layers/private/golden.npy").exists()
    shutil.rmtree(selected.root)
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(snapshot))
    assert held_out_shapes() == frozenset({(3, 5, 7)})
    context = fusion_cases.exclusion_context()
    from merlin_experiments.corpus.publication import _excluded_cores

    assert len(_excluded_cores(context)) == 1
    source_snapshot.verify(snapshot)

    # Stub campaign execution only; read_bound_validation must check the real
    # frozen manifest against the selected suite's exclusion digest.
    members = {}
    for name in ("candidates", "cases", "facts", "spec", "extension", "provider", "backend"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(dict(enumeration_complete=True, campaign={})))
        members[name] = path
    identity = {
        key: fusion_cases.file_digest(members[name])
        for key, name in (
            ("candidate_manifest_sha256", "candidates"),
            ("case_manifest_sha256", "cases"),
            ("facts_sha256", "facts"),
            ("spec_sha256", "spec"),
            ("simulator_extension_sha256", "extension"),
            ("reference_backend_sha256", "backend"),
        )
    }
    identity["private_exclusion_sha256"] = fusion_cases.digest(sorted(held_out_shapes()))
    suite = dict(identity=identity, profile={})
    ledger = dict(identity=identity.copy(), target="fixture", execution_complete=True, rows=[], scope=dict(campaign={}))
    record = tmp_path / "record.json"
    record.write_text(json.dumps(ledger))
    binding = {name: str(path) for name, path in members.items()}
    binding.update(path=str(record), sha256=fusion_cases.file_digest(record))
    evidence = SimpleNamespace(
        target="fixture",
        raw_facts_sha256=identity["facts_sha256"],
        source_snapshots=[SimpleNamespace(role="software-spec", sha256=identity["spec_sha256"])],
    )
    monkeypatch.setattr(fusion_validation, "verify_seal", lambda _: None)
    monkeypatch.setattr(fusion_validation, "_inventory", fusion_cases.file_digest)
    monkeypatch.setattr(fusion_validation, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(fusion_validation, "read_suite", lambda *args, **kwargs: suite)
    monkeypatch.setattr(fusion_validation, "execution_identity", lambda *args: {})
    monkeypatch.setattr(fusion_validation, "read_validation", lambda *args, **kwargs: ledger)
    from merlin.compile import fusion_enumeration

    monkeypatch.setattr(fusion_enumeration, "verify_stratification", lambda *args, **kwargs: None)

    def frozen_shapes(manifest):
        assert manifest == Path(private[0])
        return held_out_shapes(manifest)

    monkeypatch.setattr(fusion_cases.corpus, "held_out_shapes", frozen_shapes)
    assert generation.read_bound_validation(binding, base=tmp_path, evidence=evidence)[0] == ledger
    identity["private_exclusion_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="held-out selection changed"):
        generation.read_bound_validation(binding, base=tmp_path, evidence=evidence)


@pytest.mark.parametrize("change", ["bytes", "membership"])
def test_private_exclusion_drift_fails_before_snapshot(selected, change):
    if change == "bytes":
        selected.capsule.write_text(selected.capsule.read_text() + "\n# changed\n")
    else:
        selected.manifest.write_text(yaml.safe_dump(dict(generated=["_model_layers/private", "_model_layers/new"])))
        new = selected.manifest.parent / "_model_layers/new/capsule.yaml"
        new.parent.mkdir()
        new.write_bytes(selected.capsule.read_bytes())
    with pytest.raises(ValueError, match="private exclusion (bytes|membership) changed"):
        freeze.stage(selected.plan)


def test_compositions_off_without_binding_does_not_read_capsule_shelf(selected):
    selected.descriptor.write_text("workload_spec: {synthesize_compositions: false}\n")
    shutil.rmtree(selected.manifest.parent)
    paths, _ = freeze.selected_inputs(selected.command, "fixture")
    assert not any(name.startswith("phase0:private_exclusion:") for name in paths)
