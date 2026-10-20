"""Canonical partitions retain the full denominator and join only matching host workers."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1.feedback import private_full_model_execution as execution
from merlin_experiments.phase1.feedback import private_full_models as models
from merlin_experiments.phase1.feedback import private_model_partitions as partitions


def test_selected_model_validates_full_roster_and_never_passes_full_gate(tmp_path, monkeypatch):
    spec = tmp_path / "spec.yaml"
    names = ("first", "second", "third")
    spec.write_text(
        yaml.safe_dump({"schema": models.SCHEMA, "target": "fixture", "models": [{"id": name} for name in names]})
    )
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "source").write_text("synthetic package; no compiler executed")
    monkeypatch.setattr(models.source_freeze_api, "resolve_optional", lambda *a, **kw: {})
    monkeypatch.setattr(models.source_freeze_api, "claim_binding", lambda *a, **kw: {})
    kwargs = dict(
        target="fixture",
        required_models=names,
        required_programs={n: ("model",) for n in names},
        loader_env_requirements={n: {} for n in names},
        out=tmp_path / "out",
    )
    result = models.run(package, spec, selected_model="second", **kwargs)
    assert result["required_models"] == list(names)
    assert [row["model"] for row in result["models"]] == ["second"]
    assert result["mode"] == "canonical_model_partition" and result["passed"] is False
    assert result["partition_passed"] is False
    assert len(list(kwargs["out"].glob("private_full_model_partition-*.json"))) == 1
    with pytest.raises(ValueError, match="required target roster"):
        models.run(package, spec, selected_model="second", **{**kwargs, "required_models": ("second",)})
    for selected in ("unknown", 1):
        with pytest.raises(ValueError, match="canonical model partition"):
            models.run(package, spec, selected_model=selected, **kwargs)
    for incompatible in ({"diagnostic_model": "second"}, {"prebuilt_receipts": {}}):
        with pytest.raises(ValueError, match="canonical model partition"):
            models.run(package, spec, selected_model="second", **kwargs, **incompatible)
    assert not models.complete(
        {**result, "passed": True},
        required_models=("second",),
        required_programs={"second": ("model",)},
        candidate_sha256="a" * 64,
    )


def test_execution_partition_runs_every_selected_program_without_upgrading_scope(tmp_path, monkeypatch):
    gate = {"required": True, "programs": {"first": {"x": {}, "y": {}}, "second": {"z": {}}}}
    static = dict(
        schema=models.RESULT_SCHEMA,
        target="fixture",
        mode="canonical_model_partition",
        selected_model="first",
        passed=False,
        partition_passed=True,
        required_models=list(gate["programs"]),
        required_programs={"first": ["x", "y"], "second": ["z"]},
        candidate_tree_sha256="a" * 64,
        private_spec_sha256="b" * 64,
        authored_source_freeze=dict(
            schema=models.source_freeze_api.SCHEMA, spec_sha256="b" * 64, record_sha256="c" * 64, root=str(tmp_path)
        ),
        models=[
            dict(
                model="first",
                status="pass",
                checks={
                    "build": {
                        "programs": [
                            {"program": "x", "elf_sha256": "d" * 64},
                            {"program": "y", "elf_sha256": "e" * 64},
                        ]
                    }
                },
            )
        ],
    )
    observed = []

    def executed(model, program, *a, **kw):
        observed.append((model, program))
        return {"status": "pass"}

    monkeypatch.setattr(execution, "_execute_program", executed)
    result = execution.run(static, gate, target="fixture", static_out=tmp_path, selected_model="first")
    assert observed == [("first", "x"), ("first", "y")]
    assert result["passed"] is False and result["partition_passed"] is True
    assert result["required_models"] == ["first", "second"]
    assert not execution.complete({**result, "passed": True}, gate, static=static, candidate_sha256="a" * 64)
    for changed in ({"required_models": ["first"]}, {"partition_passed": False}, {"mode": "prebuilt_diagnostic"}):
        with pytest.raises(execution.ExecutionGateError, match="complete original static roster"):
            execution.run({**static, **changed}, gate, target="fixture", static_out=tmp_path, selected_model="first")


@pytest.fixture
def workers(tmp_path, monkeypatch):
    names = ("first", "second")
    programs = {name: ("model",) for name in names}
    gate = {"required": True, "programs": {name: {"model": {"engine": "gsim"}} for name in names}}
    instruction = tmp_path / "instruction.json"
    instruction.write_text("original synthetic declaration")
    digest = partitions.file_sha256(instruction)
    expected = dict(
        run_dir=str(tmp_path),
        candidate_tree_sha256="a" * 64,
        private_spec_sha256="b" * 64,
        authored_source_freeze={},
        compiler_profile={"profile_sha256": "c" * 64},
        fact_reader_binding={"private_input": "selected"},
        instruction_declaration_sha256=digest,
    )
    records, observations, verified = {}, {}, {}
    for name in names:
        path = tmp_path / f"{name}.json"
        record = dict(
            schema=partitions.SCHEMA,
            selected_model=name,
            **expected,
            instruction_selection={"declaration_sha256": digest},
            private_full_models=dict(
                schema=models.RESULT_SCHEMA,
                target="fixture",
                mode="canonical_model_partition",
                selected_model=name,
                passed=False,
                partition_passed=True,
                required_models=list(names),
                required_programs={n: list(programs[n]) for n in names},
                candidate_tree_sha256="a" * 64,
                private_spec_sha256="b" * 64,
                authored_source_freeze={},
                models=[dict(model=name, status="pass")],
            ),
            private_full_model_execution=dict(
                schema=execution.SCHEMA,
                mode="canonical_model_partition",
                selected_model=name,
                passed=False,
                partition_passed=True,
                candidate_tree_sha256="a" * 64,
                required_models=list(names),
                required_programs={n: list(programs[n]) for n in names},
                programs=[dict(model=name, program="model", status="pass", engine="gsim")],
            ),
        )
        static_report = tmp_path / f"{name}.static.json"
        static_report.write_text(json.dumps(record["private_full_models"]))
        record["static_partition_report"] = str(static_report)
        record["static_partition_report_sha256"] = partitions.file_sha256(static_report)
        path.write_text(json.dumps(record))
        observed = tmp_path / f"{name}.invocation.json"
        observed.write_text("synthetic observation transport; no worker launched")
        verified[str(observed)] = dict(
            kind="subprocess", outputs=[dict(path=str(path), sha256=partitions.file_sha256(path))]
        )
        records[name], observations[name] = path, observed
    monkeypatch.setattr(partitions.invocation_record, "verify", lambda path: deepcopy(verified[str(path)]))
    monkeypatch.setattr(partitions, "_selection", lambda *a, **kw: None)
    checked = []
    monkeypatch.setattr(models, "complete", lambda record, **kw: checked.append((record, kw)) or True)
    monkeypatch.setattr(execution, "complete", lambda record, *a, **kw: checked.append((record, kw)) or True)
    kwargs = dict(
        expected=expected,
        required_models=names,
        required_programs=programs,
        gate=gate,
        target="fixture",
        instruction_path=instruction,
    )
    return records, observations, kwargs, verified, checked


def test_join_calls_original_full_static_and_execution_validators(workers):
    records, observations, kwargs, _, checked = workers
    result = partitions.join(records, observations, **kwargs)
    assert result["passed"] is True and len(checked) == 2
    assert checked[0][1]["required_models"] == ["first", "second"]
    assert [row["model"] for row in checked[0][0]["models"]] == ["first", "second"]
    assert [(row["model"], row["program"]) for row in checked[1][0]["programs"]] == [
        ("first", "model"),
        ("second", "model"),
    ]


@pytest.mark.parametrize(
    "change",
    [
        "candidate_tree_sha256",
        "private_spec_sha256",
        "compiler_profile",
        "authored_source_freeze",
        "fact_reader_binding",
        "instruction_declaration_sha256",
    ],
)
def test_join_refuses_changed_source_owner_or_declaration(workers, change):
    records, observations, kwargs, verified, checked = workers
    path = records["second"]
    record = json.loads(path.read_bytes())
    record[change] = "different"
    path.write_text(json.dumps(record))
    verified[str(observations["second"])]["outputs"][0]["sha256"] = partitions.file_sha256(path)
    with pytest.raises(ValueError, match="original owner, source, facts or declaration"):
        partitions.join(records, observations, **kwargs)
    assert checked == []


def test_join_refuses_missing_model_changed_receipt_and_narrow_program_roster(workers):
    records, observations, kwargs, _, checked = workers
    with pytest.raises(ValueError, match="complete original declaration"):
        partitions.join({"first": records["first"]}, observations, **kwargs)
    record = json.loads(records["second"].read_bytes())
    record["private_full_models"]["required_models"] = ["second"]
    records["second"].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="completed protected host worker"):
        partitions.join(records, observations, **kwargs)
    assert checked == []


def test_worker_issuer_and_each_actual_source_pin_are_reopened(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import private_instruction_coordinator

    from merlin.common import invocation_record

    instruction = tmp_path / "instruction.json"
    instruction.write_text("synthetic declaration; no policy or hardware issued")
    source = tmp_path / "original-source.py"
    source.write_text("original selected synthetic bytes")
    selected = {"declaration_sha256": partitions.file_sha256(instruction)}
    receipt = tmp_path / "selection.json"
    receipt.write_text(json.dumps(selected))
    monkeypatch.setattr(partitions.declaration, "read", lambda *a, **kw: SimpleNamespace(verify=lambda: None))
    with invocation_record.observe_call(
        tmp_path,
        stage="fresh_instruction_coordinator_selection",
        function=private_instruction_coordinator._issue,
        arguments={"target": "fixture", "declaration_sha256": selected["declaration_sha256"]},
        inputs=(instruction,),
        outputs=(receipt,),
    ) as observed:
        observed.returned()
    record = dict(
        instruction_selection=selected,
        instruction_selection_receipt=str(receipt),
        instruction_selection_invocation=str(observed.path),
        instruction_source_pins=[[str(source), partitions.file_sha256(source)]],
    )
    partitions._selection(record, target="fixture", instruction_path=instruction)
    source.write_text("changed original source bytes")
    with pytest.raises(ValueError, match="instruction source selection changed"):
        partitions._selection(record, target="fixture", instruction_path=instruction)
