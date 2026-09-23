"""Freeze the native descriptor's corpus closure without running any grader."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from merlin_experiments import SpecError, load_spec
from merlin_experiments.adapters import ADAPTERS
from merlin_experiments.runner import preflight, resolve_plan, resume, run, status


def _capsule(root: Path, category: str, name: str) -> Path:
    member = root / category / name
    member.mkdir(parents=True)
    (member / "capsule.yaml").write_text(yaml.safe_dump({"name": name, "label": "public"}))
    payload = member / "payload.bin"
    payload.write_bytes(b"original bytes")
    return payload


@pytest.fixture
def corpus_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    engine = tmp_path / "engine.py"
    engine.write_text(
        "import pathlib, sys\n"
        "receipt=pathlib.Path(__file__).with_name('calls')\n"
        "count=int(receipt.read_text())+1 if receipt.exists() else 1\n"
        "receipt.write_text(str(count))\n"
        "sys.exit(7 if count==1 else 0)\n"
    )
    # These cases replay historical script plans; installed commands have separate coverage.
    monkeypatch.setitem(ADAPTERS, "capsule_bench", replace(ADAPTERS["capsule_bench"], script="engine.py", module=None))
    corpus = tmp_path / "corpus"
    payloads = {
        "primary": _capsule(corpus, "isa", "primary"),
        "sibling": _capsule(corpus, "layers", "sibling"),
        "hidden": _capsule(corpus, "hidden", "private-identity"),
        "performance": _capsule(corpus, "_perf", "performance"),
        "foreign": _capsule(corpus / "another-target", "isa", "foreign"),
    }
    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: fixture-target\ncapsule_corpus: corpus/isa\n")
    definition = tmp_path / "experiment.yaml"
    definition.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "corpus-test",
                "target": "fixture-target",
                "phases": {
                    "1": {
                        "adapter": "capsule_bench",
                        "config": {
                            "descriptor": descriptor.name,
                            "arm": "raw_baseline",
                            "model": "fixture-model",
                            "effort": "high",
                            "max_wall_s": 60,
                            "round_timeout": 60,
                        },
                    }
                },
            }
        )
    )
    return definition, tmp_path / "run", corpus, payloads


def test_closure_uses_native_descriptor_selection_and_keeps_private_names_out(corpus_run):
    from merlin.targetgen.target_experiment import load_target_experiment

    definition, destination, _, payloads = corpus_run
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    descriptor = load_target_experiment(definition.parent / "target.yaml")
    assert plan["corpus_closures"]["1"] == {
        "graded": [str(path.resolve()) for path in descriptor.graded_roots()],
        "hidden": [str(path.resolve()) for path in descriptor.hidden_roots()],
    }
    check = preflight(plan)
    assert check["configuration_ready"], check["errors"]
    corpus_pins = {name: pin for name, pin in check["inputs"].items() if ":corpus:" in name}
    # Corpus data and the cross-phase corpus implementation are distinct inputs.
    # Keep the privacy assertions below over both, but count only data closures.
    assert len([name for name in corpus_pins if name.startswith("phase1:corpus:")]) == 3
    assert any(name.startswith("phase1:startup:corpus:") for name in corpus_pins)
    assert payloads["hidden"].parent.name not in json.dumps(plan)
    assert payloads["hidden"].parent.name not in json.dumps(corpus_pins)
    assert "original bytes" not in json.dumps(corpus_pins)
    assert not destination.exists()


def test_plan_pins_exact_public_clients_and_mapping_owner(corpus_run):
    from merlin.common.paths import module_source_path
    from merlin.targetgen.tool_registry import public_client_modules

    definition, destination, _, _ = corpus_run
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    actual = {name: path for name, path in plan["input_paths"].items() if name.startswith("phase1:client:")}
    assert actual == {
        f"phase1:client:{module}": str(module_source_path(module).resolve())
        for module in (*public_client_modules(), "merlin.targetgen.tool_registry")
    }
    assert len(actual) == 6


@pytest.mark.parametrize("change", ["bytes", "resolution", "missing_pin", "membership"])
def test_public_client_drift_refuses_resume_before_launch(corpus_run, monkeypatch, change):
    from merlin.common import paths
    from merlin.targetgen import tool_registry

    definition, destination, _, _ = corpus_run
    source = definition.parent / "public-client.py"
    source.write_bytes(paths.module_source_path("merlin_experiments.phase1.tools.simjob").read_bytes())
    original = paths.module_source_path
    monkeypatch.setattr(
        paths,
        "module_source_path",
        lambda name: source if name == "merlin_experiments.phase1.tools.simjob" else original(name),
    )
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    if change == "missing_pin":
        del plan["input_paths"]["phase1:client:merlin_experiments.phase1.tools.simjob"]
        check = preflight(plan)
        assert not check["configuration_ready"]
        assert any("public client source closure" in message for message in check["errors"])
        assert not (definition.parent / "calls").exists()
        return
    assert run(plan) == 7
    if change == "bytes":
        source.write_text(source.read_text() + "\n# changed client\n")
    elif change == "resolution":
        monkeypatch.setattr(paths, "module_source_path", original)
    else:
        previous = tool_registry.public_client_modules()
        monkeypatch.setattr(tool_registry, "public_client_modules", lambda: previous[:-1])
    with pytest.raises(SpecError, match="public client source closure|frozen input changed: phase1:client"):
        resume(destination)
    assert (definition.parent / "calls").read_text() == "1"


@pytest.fixture
def extracted_phase1_sources(corpus_run, monkeypatch):
    from merlin.common import paths

    definition, _, _, _ = corpus_run
    installed = definition.parent / "installed"
    installed.mkdir()
    original = paths.module_source_path
    owner = original("merlin_experiments")
    for relative in ("__init__.py", "spec.py", "resources/legacy_entrypoints.json"):
        destination = installed / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(owner.parent / relative, destination)
    shutil.copytree(
        original("merlin_experiments.phase1").parent, installed / "phase1", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copyfile(original("merlin.targetgen.corpora"), installed / "corpora.py")

    def resolve(module):
        if module == "merlin_experiments":
            return installed / "__init__.py"
        if module == "merlin_experiments.spec":
            return installed / "spec.py"
        if module == "merlin_experiments.phase1":
            return installed / "phase1/__init__.py"
        if module.startswith("merlin_experiments.phase1."):
            return installed / "phase1" / (module.removeprefix("merlin_experiments.phase1.").replace(".", "/") + ".py")
        if module == "merlin.targetgen.corpora":
            return installed / "corpora.py"
        return original(module)

    monkeypatch.setattr(paths, "module_source_path", resolve)
    return installed


@pytest.mark.parametrize(
    "change",
    [
        "phase1/context.py",
        "phase1/run_inputs.py",
        "phase1/providers/codex_agent.py",
        "__init__.py",
        "spec.py",
        "resources/legacy_entrypoints.json",
        "corpora.py",
        "new_member",
        "missing_member",
        "new_nested_member",
        "missing_nested_member",
    ],
)
def test_extracted_phase1_source_closure_drift_refuses_resume(corpus_run, extracted_phase1_sources, change):
    definition, destination, _, _ = corpus_run
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    assert run(plan) == 7
    installed = extracted_phase1_sources
    if change == "new_member":
        (installed / "phase1/additional.py").write_text("VALUE = 1\n")
    elif change == "missing_member":
        (installed / "phase1/context.py").unlink()
    elif change == "new_nested_member":
        (installed / "phase1/providers/additional.py").write_text("VALUE = 1\n")
    elif change == "missing_nested_member":
        (installed / "phase1/providers/resource_sampler.py").unlink()
    else:
        path = installed / change
        path.write_text(path.read_text() + "\n")
    with pytest.raises(SpecError, match="phase-1 implementation|frozen input changed: phase1:"):
        resume(destination)
    assert (definition.parent / "calls").read_text() == "1"


def test_native_phase1_startup_pins_require_exact_declared_entrypoint(corpus_run):
    from merlin_experiments.runner import _phase1_source_inputs

    definition, destination, _, _ = corpus_run
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    command = plan["phases"]["1"]
    assert not any(key.startswith("phase1:startup:native:") for key in _phase1_source_inputs(command))
    harness = definition.parent / "merlin/experiments/capsule_bench/harness"
    harness.mkdir(parents=True)
    entry = harness / "run_baseline_qa_loop.py"
    entry.write_text("# synthetic transport-only native entrypoint\n")
    for name in ("_common.py", "sandbox_toolchain.py"):
        (harness / name).write_text("# synthetic startup identity\n")
    command = dict(command, entrypoint=str(entry))
    pins = _phase1_source_inputs(command)
    assert pins["phase1:startup:native:_common.py"] == str(harness / "_common.py")
    assert pins["phase1:startup:native:sandbox_toolchain.py"] == str(harness / "sandbox_toolchain.py")
    (harness / "_common.py").unlink()
    with pytest.raises(SpecError, match="native startup input is absent"):
        _phase1_source_inputs(command)


def test_client_drift_during_successful_child_fails_only_outer_attribution(corpus_run, monkeypatch):
    from merlin.common import paths

    definition, destination, _, _ = corpus_run
    client = definition.parent / "client.py"
    original = paths.module_source_path
    client.write_bytes(original("merlin_experiments.phase1.tools.simjob").read_bytes())
    monkeypatch.setattr(
        paths,
        "module_source_path",
        lambda module: client if module == "merlin_experiments.phase1.tools.simjob" else original(module),
    )
    evidence = definition.parent / "engine-evidence.json"
    (definition.parent / "engine.py").write_text(
        "from pathlib import Path\n"
        f"client = Path({str(client)!r})\n"
        "client.write_text(client.read_text() + '\\n# mutation during child execution\\n')\n"
        f"Path({str(evidence)!r}).write_text('{{\"transport_finished\": true}}\\n')\n"
    )
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 1
    record = status(destination)
    assert record["state"] == "execution_failed"
    assert record["attempts"][-1]["engine_returncode"] == 0
    assert "input identity changed during execution" in record["attempts"][-1]["error"]
    assert evidence.read_text() == '{"transport_finished": true}\n'


@pytest.mark.parametrize("visibility", ["primary", "sibling", "hidden"])
def test_resume_rejects_changed_public_or_hidden_bytes_before_process_launch(corpus_run, visibility):
    definition, destination, _, payloads = corpus_run
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    payloads[visibility].write_bytes(b"different bytes")
    with pytest.raises(SpecError, match="frozen input changed: phase1:corpus:"):
        resume(destination)
    assert (definition.parent / "calls").read_text() == "1"
    assert len(status(destination)["attempts"]) == 1


@pytest.mark.parametrize("mutation", ["add_category", "remove_category", "add_hidden"])
def test_resume_rediscovers_category_membership(corpus_run, mutation):
    definition, destination, corpus, _ = corpus_run
    if mutation == "add_hidden":
        (corpus / "hidden").rename(corpus / "_private-fixture")
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    if mutation == "add_category":
        _capsule(corpus, "new-category", "new")
    elif mutation == "remove_category":
        (corpus / "layers").rename(corpus / "_retired-fixture")
    else:
        (corpus / "_private-fixture").rename(corpus / "hidden")
    with pytest.raises(SpecError, match="corpus category membership changed"):
        resume(destination)
    assert (definition.parent / "calls").read_text() == "1"


def test_membership_drift_between_inspection_and_run_is_rejected(corpus_run):
    definition, destination, corpus, _ = corpus_run
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    _capsule(corpus, "new-category", "new")
    check = preflight(plan)
    assert not check["configuration_ready"]
    assert "corpus category membership changed" in " ".join(check["errors"])
    with pytest.raises(SpecError, match="corpus category membership changed"):
        run(plan)
    assert not destination.exists()
    assert not (definition.parent / "calls").exists()


def test_native_excluded_categories_do_not_invalidate_functional_resume(corpus_run):
    definition, destination, _, payloads = corpus_run
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    payloads["performance"].write_bytes(b"performance-only update")
    payloads["foreign"].write_bytes(b"other target update")
    assert resume(destination) == 0
    assert (definition.parent / "calls").read_text() == "2"


def test_legacy_functional_plan_without_closure_is_not_silently_upgraded(corpus_run):
    definition, destination, _, _ = corpus_run
    plan = resolve_plan(load_spec(definition), run_dir=destination)
    del plan["corpus_closures"]
    with pytest.raises(SpecError, match="no frozen corpus closure"):
        run(plan)
    assert not destination.exists()


def test_corpus_output_overlap_is_rejected_before_writing(corpus_run):
    definition, _, corpus, _ = corpus_run
    with pytest.raises(SpecError, match="overlaps frozen input"):
        resolve_plan(load_spec(definition), run_dir=corpus / "isa" / "run")
    assert not (corpus / "isa" / "run").exists()


def test_missing_primary_corpus_fails_preflight(corpus_run):
    definition, destination, corpus, _ = corpus_run
    (corpus / "isa").rename(corpus / "_missing-primary")
    check = preflight(resolve_plan(load_spec(definition), run_dir=destination))
    assert not check["configuration_ready"]
    assert "input is absent" in " ".join(check["errors"])


def test_missing_descriptor_corpus_is_a_clear_configuration_error(corpus_run):
    definition, destination, _, _ = corpus_run
    (definition.parent / "target.yaml").write_text("target: fixture-target\n")
    with pytest.raises(SpecError, match="descriptor does not declare capsule_corpus"):
        resolve_plan(load_spec(definition), run_dir=destination)


def test_resume_cannot_reinterpret_descriptor_under_another_repository(corpus_run, monkeypatch, tmp_path):
    definition, destination, _, _ = corpus_run
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "another-repository"))
    with pytest.raises(SpecError, match="corpus repository root changed"):
        resume(destination)
    assert (definition.parent / "calls").read_text() == "1"


def test_resume_rejects_malformed_descriptor_without_launching(corpus_run):
    definition, destination, _, _ = corpus_run
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    (definition.parent / "target.yaml").write_text("target: [unterminated\n")
    with pytest.raises(SpecError, match="cannot discover descriptor-owned corpus closure"):
        resume(destination)
    assert (definition.parent / "calls").read_text() == "1"


def test_resume_rejects_category_symlink_retargeting_even_with_identical_bytes(corpus_run):
    import shutil

    definition, destination, corpus, _ = corpus_run
    first = corpus / "_first"
    second = corpus / "_second"
    (corpus / "layers").rename(first)
    shutil.copytree(first, second)
    (corpus / "layers").symlink_to(first, target_is_directory=True)
    assert run(resolve_plan(load_spec(definition), run_dir=destination)) == 7
    (corpus / "layers").unlink()
    (corpus / "layers").symlink_to(second, target_is_directory=True)
    with pytest.raises(SpecError, match="corpus category membership changed"):
        resume(destination)
    assert (definition.parent / "calls").read_text() == "1"
