"""Independent platform inputs preserve exact native joins and private boundaries."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin_experiments.phase1.feedback import public_owned_host_inputs as inputs
from merlin_experiments.phase1.feedback import public_owned_host_selection as session


def pin(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def selected(tmp_path, monkeypatch):
    from merlin.targetgen import target_experiment
    from merlin.runtime import modelinputs as model_execution_inputs

    descriptor, catalog, dts, facts = [tmp_path / name for name in ("descriptor.yaml", "catalog.yaml", "cpu.dts", "facts.json")]
    descriptor.write_text("selected descriptor")
    dts.write_text("selected DTS")
    facts.write_text("selected private facts")
    board = dict(target="fixture", rtl_sim_config="selected_native", host_dts_sha256=pin(dts)["sha256"],
                 dram_bytes="256 MiB", dram_base=2147483648, harts=1, console="htif", flow="baremetal",
                 code_reserve="64 MiB")
    catalog.write_text(yaml.safe_dump({"schema_version": 1, "boards": {"native_board": board}}))
    experiment = SimpleNamespace(target="fixture", selected_board_catalog=lambda: catalog)
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path, **kw: experiment)
    observed = []

    def native(path, *, target, config):
        assert (path, target, config) == (facts, "fixture", "selected_native")
        observed.append((path, target, config))
        return {"path": str(facts), "target": target, "config": config}

    monkeypatch.setattr(model_execution_inputs, "selected_firrtl", native)
    record = dict(schema=inputs.SCHEMA, target="fixture", descriptor=pin(descriptor),
                  board_catalog=pin(catalog), dts=pin(dts), rtl_facts=pin(facts), board="native_board", arena_mb=32)
    request = tmp_path / "operator-inputs.json"

    def write(document=None):
        request.write_text(json.dumps(record if document is None else document))
        return pin(request)

    return SimpleNamespace(record=record, write=write, descriptor=descriptor, catalog=catalog, dts=dts, facts=facts,
                           request=request, observed=observed, output=tmp_path / "private/selection.json")


def test_answer_free_leaf_selection_without_private_models(selected):
    result = inputs.prepare(selected.write(), target="fixture", descriptor=selected.descriptor, output=selected.output)
    assert result == pin(selected.output) and selected.output.stat().st_mode & 0o777 == 0o600
    record = json.loads(selected.output.read_bytes())
    assert set(record) == {"board_catalog", "dts", "rtl_facts", "board", "arena_mb"}
    assert record["arena_mb"] == 32 and len(selected.observed) == 1
    assert not ({"models", "capture", "golden", "outputs", "instruction_declaration"} & set(record))
    with pytest.raises(ValueError, match="fresh"):
        inputs.prepare(selected.write(), target="fixture", descriptor=selected.descriptor, output=selected.output)


@pytest.mark.parametrize("mutation", ["target", "board", "arena_bool", "arena_large", "extra", "schema", "different_catalog", "dts_hash"])
def test_wrong_join_or_model_record_refuses_without_publication(selected, tmp_path, mutation):
    record = dict(selected.record)
    if mutation == "target": record["target"] = "other"
    elif mutation == "board": record["board"] = "unknown"
    elif mutation == "arena_bool": record["arena_mb"] = True
    elif mutation == "arena_large": record["arena_mb"] = 97
    elif mutation == "extra": record["models"] = []
    elif mutation == "schema": record["schema"] = "private_full_models"
    elif mutation == "different_catalog":
        alternate = tmp_path / "alternate.yaml"; alternate.write_bytes(selected.catalog.read_bytes()); record["board_catalog"] = pin(alternate)
    else:
        selected.dts.write_text("a different DTS"); record["dts"] = pin(selected.dts)
    with pytest.raises(ValueError):
        inputs.prepare(selected.write(record), target="fixture", descriptor=selected.descriptor, output=selected.output)
    assert not selected.output.exists()


@pytest.mark.parametrize("role", ["descriptor", "board_catalog", "dts", "rtl_facts"])
def test_changed_pin_refuses(selected, role):
    chosen = selected.write()
    Path(selected.record[role]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        inputs.prepare(chosen, target="fixture", descriptor=selected.descriptor, output=selected.output)
    assert not selected.output.exists()


def test_symlink_and_unpinned_request_refuse(selected, tmp_path):
    chosen = selected.write()
    alias = tmp_path / "linked.json"; alias.symlink_to(selected.request)
    with pytest.raises(ValueError, match="ordinary"):
        inputs.prepare({**chosen, "path": str(alias)}, target="fixture", descriptor=selected.descriptor, output=selected.output)
    with pytest.raises(ValueError, match="changed"):
        inputs.prepare({**chosen, "sha256": "0" * 64}, target="fixture", descriptor=selected.descriptor, output=selected.output)


def test_configuration_validation_and_postcheck_cannot_be_skipped(selected, monkeypatch):
    from merlin.runtime import modelinputs as model_execution_inputs

    def mutate(*a, **kw): selected.facts.write_text("changed during validation")
    monkeypatch.setattr(model_execution_inputs, "selected_firrtl", mutate)
    with pytest.raises(ValueError, match="changed"):
        inputs.prepare(selected.write(), target="fixture", descriptor=selected.descriptor, output=selected.output)
    assert not selected.output.exists()


@pytest.mark.parametrize("valid", [True, False])
def test_optional_instruction_selection_uses_original_reader(selected, tmp_path, monkeypatch, valid):
    from merlin_experiments.phase1.feedback import private_instruction_declaration as declarations

    declaration = tmp_path / "instruction.json"; declaration.write_text("selected declaration")
    selected.record["instruction_declaration"] = pin(declaration)
    observed = []

    def read(path, *, target):
        assert (path, target) == (declaration, "fixture")
        if not valid: raise ValueError("original instruction reader refusal")
        return SimpleNamespace(verify=lambda: observed.append(True))
    monkeypatch.setattr(declarations, "read", read)
    if not valid:
        with pytest.raises(ValueError, match="original instruction"):
            inputs.prepare(selected.write(), target="fixture", descriptor=selected.descriptor, output=selected.output)
        assert not selected.output.exists()
    else:
        inputs.prepare(selected.write(), target="fixture", descriptor=selected.descriptor, output=selected.output)
        assert json.loads(selected.output.read_bytes())["instruction_declaration"] == pin(declaration)
        assert observed == [True]


def test_session_optional_explicit_selection_does_not_enable_models(selected, tmp_path, monkeypatch):
    monkeypatch.setenv(session.ENV, "untrusted inherited selection")
    result = session.install_for_session(None, None, target="fixture", run_dir=tmp_path / "run",
                                         descriptor=selected.descriptor, host_inputs=selected.write())
    assert json.loads(__import__("os").environ[session.ENV]) == result
    assert "models" not in json.loads(Path(result["path"]).read_bytes())


def test_session_does_not_inherit_missing_or_invalid_inputs(selected, tmp_path, monkeypatch):
    monkeypatch.setenv(session.ENV, "untrusted inherited selection")
    assert session.install_for_session(None, None, target="fixture", run_dir=tmp_path) is None
    assert session.ENV not in __import__("os").environ
    monkeypatch.setenv(session.ENV, "untrusted inherited selection")
    with pytest.raises(ValueError):
        session.install_for_session("private-model-spec", None, target="fixture", run_dir=tmp_path,
                                    descriptor=selected.descriptor, host_inputs=selected.write())
    assert session.ENV not in __import__("os").environ


def test_preparation_cli_only_emits_private_pin(selected, capsys):
    request = selected.write()
    assert inputs.main(["--inputs", request["path"], "--inputs-sha256", request["sha256"], "--target", "fixture",
                        "--descriptor", str(selected.descriptor), "--output", str(selected.output)]) == 0
    assert json.loads(capsys.readouterr().out) == pin(selected.output)
