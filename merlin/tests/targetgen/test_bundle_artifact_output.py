"""Bundle generation defaults to tracked products, never authored example siblings."""

import hashlib
from pathlib import Path

import pytest
import yaml

from merlin.targetgen import generate_bundles as bundles


@pytest.fixture
def generation(tmp_path, monkeypatch):
    descriptor = tmp_path / "example/target/descriptor.yaml"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("target: synthetic_device\n")
    output = tmp_path / "out"
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(output))
    destinations = []

    def materialize(te, dest, **kwargs):
        assert te.target == "synthetic_device"
        destinations.append(Path(dest))
        manifest = Path(dest) / "synthetic_public/input_bundle_manifest.yaml"
        manifest.parent.mkdir(parents=True)
        manifest.write_text("bundle_id: synthetic_public\n")
        return [manifest]

    monkeypatch.setattr(bundles, "materialize_bundles", materialize)
    return descriptor, output, destinations


def test_default_output_has_manifest_and_descriptor_provenance(generation):
    descriptor, output, destinations = generation
    before = descriptor.read_bytes()
    assert bundles._main(["--descriptor", str(descriptor)]) == 0
    (destination,) = destinations
    assert destination.name == "input_bundles"
    assert destination.is_relative_to(output / "artifacts")
    product = yaml.safe_load((destination.parent / "manifest.yaml").read_text())
    assert product["topic"] == "capsule-bench"
    assert product["version"] == 1
    assert product["target"] == "synthetic_device"
    assert product["artifacts"] == ["input_bundles/synthetic_public/input_bundle_manifest.yaml"]
    assert any(
        source.get("path") == str(descriptor) and source.get("sha256") == hashlib.sha256(before).hexdigest()
        for source in product["sources"]
    )
    assert descriptor.read_bytes() == before
    assert list(descriptor.parent.iterdir()) == [descriptor]
    assert not list(output.rglob("latest"))


def test_repeated_default_invocation_preserves_distinct_products(generation):
    descriptor, output, destinations = generation
    for _ in range(2):
        assert bundles._main(["--descriptor", str(descriptor)]) == 0
    assert len(set(destinations)) == 2
    assert all((dest.parent / "manifest.yaml").is_file() for dest in destinations)
    assert all((dest / "synthetic_public/input_bundle_manifest.yaml").is_file() for dest in destinations)
    assert not (descriptor.parent / "input_bundles").exists()
    assert not list(output.rglob("latest"))


def test_explicit_destination_remains_exact_without_product_creation(generation, tmp_path):
    descriptor, output, destinations = generation
    destination = tmp_path / "legacy/input_bundles"
    assert bundles._main(["--descriptor", str(descriptor), "--dest", str(destination)]) == 0
    assert destinations == [destination]
    assert (destination / "synthetic_public/input_bundle_manifest.yaml").is_file()
    assert not (destination.parent / "manifest.yaml").exists()
    assert not output.exists()


def test_generation_exception_does_not_publish_latest_or_register_outputs(generation, monkeypatch):
    descriptor, output, _ = generation

    def fail(te, dest, **kwargs):
        raise RuntimeError("generation interrupted")

    monkeypatch.setattr(bundles, "materialize_bundles", fail)
    with pytest.raises(RuntimeError, match="generation interrupted"):
        bundles._main(["--descriptor", str(descriptor)])
    assert not list(output.rglob("latest"))
    # new_product records creation immediately; no completed artifacts may be claimed.
    for manifest in output.rglob("manifest.yaml"):
        assert yaml.safe_load(manifest.read_text())["artifacts"] == []
    assert not (descriptor.parent / "input_bundles").exists()


@pytest.mark.parametrize("selection", ["legacy", "relative", "absolute"])
def test_task_grant_tracks_actual_authored_resources(tmp_path, monkeypatch, selection):
    from merlin.targetgen.target_experiment import load_target_experiment

    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: synthetic_device\ncapsule_corpus: corpus/isa\n")
    task = tmp_path / "task"
    if selection != "legacy":
        task = tmp_path / "examples/device/phase1/task"
        task.parent.mkdir(parents=True)
        declared = str(task) if selection == "absolute" else "examples/device/phase1/task"
        descriptor.write_text(descriptor.read_text() + f"task_root: {declared}\n")
        # A populated historical directory cannot become a fallback grant.
        (tmp_path / "task").mkdir()
        (tmp_path / "task/TASK.md").write_text("wrong source")
    te = load_target_experiment(descriptor)
    task_grant = te.experiment_resource("task") + "/"
    if selection != "legacy":
        assert task_grant == "examples/device/phase1/task/"
        assert "task/" not in {row["path"] for row in bundles._shared_allow(te, "public_v0")}
    assert task_grant not in {row["path"] for row in bundles._shared_allow(te, "public_v0")}
    task.mkdir()
    grants = {row["path"] for row in bundles._shared_allow(te, "public_v0")}
    assert task_grant in grants
    if selection != "legacy":
        assert "task/" not in grants
    task.rmdir()
    task.write_text("not a directory")
    with pytest.raises(ValueError, match="task resource"):
        bundles._shared_allow(te, "public_v0")


@pytest.mark.parametrize("target", ["..", "../escape", "/absolute", "nested/device", "a\\b", "."])
def test_unsafe_target_refused_before_output_creation(generation, target, capsys):
    descriptor, output, destinations = generation
    descriptor.write_text(yaml.safe_dump({"target": target}))
    with pytest.raises(SystemExit) as caught:
        bundles._main(["--descriptor", str(descriptor)])
    assert caught.value.code == 2
    assert "target" in capsys.readouterr().err.lower()
    assert not destinations
    assert not output.exists()
    assert not (descriptor.parent / "input_bundles").exists()
