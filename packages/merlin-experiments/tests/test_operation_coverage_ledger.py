import json

import pytest
import yaml
from merlin_experiments.phase0 import operation_coverage_ledger as L
from merlin_experiments.phase0.component_generation import digest

SPEC = {
    "schema": "merlin.software_spec.v1",
    "target": "toy",
    "status": "reviewed",
    "numerical_semantics": {
        "model": {"engine": "integer_reference"},
        "operand_dtype": "int8",
        "accumulator_dtype": "int32",
        "readout_dtype": "int8",
        "subnormal_operand_flush": False,
    },
    "operations": [
        {"id": "contraction", "families": ["contraction"], "placement": "accelerator", "signature": {"ranks": [2]}},
        {"id": "relu_map", "ops": ["relu"], "placement": "accelerator", "signature": {"ranks": [2]}},
        {
            "id": "pending_reduce",
            "families": ["reduction"],
            "placement": "accelerator",
            "status": "unreviewed",
            "signature": {"ranks": [2]},
        },
        {"id": "views", "families": ["movement"], "placement": "host", "signature": {"ranks": [2]}},
        {"id": "unused_map", "ops": ["gelu"], "placement": "accelerator", "signature": {"ranks": [2]}},
    ],
    "compositions": {
        "schema": "merlin.software_spec.compositions.v1",
        "status": "reviewed",
        "grading": "placement_and_instruction_audit",
        "chains": [
            {
                "id": "scaled_operand_sum",
                "producer": "operand_sum",
                "readout_stages": "from_output_path",
                "exits": ["requant", "accumulator"],
                "constraints": [{"operand_multiplier_in": "(0, 1]"}, "load_minimum_not_negated"],
            },
            {
                "id": "lonely_chain",
                "producer": "operand_sum",
                "readout_stages": "from_output_path",
                "exits": ["accumulator"],
                "constraints": [{"operand_multiplier_in": "(0, 1]"}, "load_minimum_not_negated"],
            },
        ],
    },
}


def _step(declaration, family, line):
    return {"declaration_id": declaration, "family": family, "source": {"path": "spec.yaml", "line": line}}


def _case(case_id, overload, verdict, *, placement="accelerator", steps=()):
    return {
        "case_id": case_id,
        "overload": overload,
        "verdict": verdict,
        "spec_expectation": {"placement": placement, "steps": list(steps)},
        "observed": {"lane": "device" if verdict == "agree_device" else "host"},
    }


def _record(spec_path, cases, unmodeled=()):
    import hashlib

    record = {
        "schema": L.SPEC_VALIDATION_SCHEMA,
        "software_spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "verdict": "validated",
        "cases": cases,
        "unmodeled": list(unmodeled),
    }
    record["sha256"] = digest(record)
    return record


CASES = [
    _case("c0", "aten.mm.default", "agree_device", steps=[_step("contraction", "contraction", 10)]),
    _case("c1", "mm.default", "agree_device", steps=[_step("contraction", "contraction", 10)]),
    _case("c2", "relu.default", "agree_device", steps=[_step("relu_map", "elementwise_map", 20)]),
    _case("c3", "sum.dim_IntList", "agree_device", steps=[_step("pending_reduce", "reduction", 30)]),
    _case("c4", "view.default", "agree_host", placement="host", steps=[_step("views", "movement", 40)]),
    _case("c5", "addmm.default", "MISMATCH_admitted_not_device", steps=[_step("contraction", "contraction", 10)]),
    _case("c6", "sort.default", "agree_host", placement="host"),
    _case("c7", "cumsum.default", "no_work", placement="host"),
]


@pytest.fixture
def inputs(tmp_path):
    spec = tmp_path / "software-spec.yaml"
    spec.write_text(yaml.safe_dump(SPEC, sort_keys=False))
    record = tmp_path / "record.json"
    record.write_text(json.dumps(_record(spec, CASES, unmodeled=["sort.default"])))
    model = tmp_path / "model.mlir"
    model.write_text(
        "\n".join(
            f'%{i} = linalg.generic {{prov.aten = "aten.{op}"}}'
            for i, op in enumerate(["mm.default", "relu.default", "convolution.default", "view.default"])
        )
    )
    capsules = [
        {"name": "SY_source_mlp", "op": "model", "model": "mlp", "source_role": "materialized_iteration_capture"},
        {
            "name": "SY_contraction_int8_aligned",
            "op": "matmul",
            "source_role": "derived_sweep",
            "lanes": {"require": ["on_mesh"]},
            "source_reference": "synthesized for conformance cell c",
        },
        {
            "name": "SY_composition_scaled_operand_sum_q_tail",
            "op": "composition_program",
            "generalization": {
                "composition_chain": SPEC["compositions"]["chains"][0],
                "quantization_scheme": {"id": "q"},
            },
            "source_role": "derived_sweep",
            "lanes": {"require": ["on_mesh"]},
        },
        {"name": "SY_orphan", "op": "no_such_op", "source_role": "derived_sweep"},
    ]
    return dict(spec_validation=record, software_spec=spec, capsules=capsules, captures={"mlp": model})


def _rows(document):
    return {row["overload"]: row for row in document["operations"]}


def test_every_overload_has_exactly_one_row_with_distinct_states(inputs):
    document = L.prepare(**inputs).record()
    names = [row["overload"] for row in document["operations"]]
    assert len(names) == len(set(names))
    rows = _rows(document)
    assert set(rows) == {
        "mm.default",
        "relu.default",
        "sum.dim_IntList",
        "view.default",
        "addmm.default",
        "sort.default",
        "cumsum.default",
        "convolution.default",
    }
    assert {name: row["evidence_state"] for name, row in rows.items()} == {
        "mm.default": "executed_spike",
        "relu.default": "executed_spike",
        "sum.dim_IntList": "unreviewed_declaration",
        "view.default": "executed_host",
        "addmm.default": "mismatch",
        "sort.default": "unmodeled",
        "cumsum.default": "executed_host",
        "convolution.default": "generated",
    }
    assert set(document["state_counts"]) <= set(L.STATES)
    assert "rtl_qualified" not in document["state_counts"]


def test_row_follows_coverage_table(inputs):
    row = _rows(L.prepare(**inputs).record())["mm.default"]
    assert row["source_requirement"]["demanded_by"] == ["workload:mlp"]
    assert row["declaration"]["status"] == "screened"
    assert row["declaration"]["rows"][0]["id"] == "contraction"
    assert row["declaration"]["rows"][0]["sources"] == ["spec.yaml:10"]
    assert row["declaration"]["rows"][0]["operation_domain"]["ranks"] == [2]
    assert row["reference"]["spec_validation"]["cases"] == 2
    assert row["test_identity"]["validation_cases"] == ["c0", "c1"]
    assert row["test_identity"]["capsules"] == ["SY_source_mlp"]
    assert row["test_identity"]["family_capsules"] == ["SY_contraction_int8_aligned"]
    assert row["required_engine"] == "accelerator"
    assert "SY_source_mlp" in row["justification"]


def test_evidence_only_and_host_rows_are_justified(inputs):
    rows = _rows(L.prepare(**inputs).record())
    assert rows["relu.default"]["source_requirement"]["demanded_by"] == ["workload:mlp"]
    assert rows["sum.dim_IntList"]["source_requirement"]["demanded_by"] == ["core_aten_only"]
    assert rows["view.default"]["required_engine"] == "host"
    assert rows["view.default"]["justification"].startswith("host:")
    assert rows["cumsum.default"]["declaration"]["status"] == "unsupported"
    assert all(row["justification"] for row in rows.values())


def test_evidence_only_row_states_why_it_has_no_capsule(inputs, tmp_path):
    spec = dict(SPEC, operations=[dict(row, status="reviewed") for row in SPEC["operations"]])
    path = tmp_path / "reviewed.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    record = tmp_path / "reviewed.json"
    record.write_text(json.dumps(_record(path, CASES)))
    row = _rows(L.prepare(**dict(inputs, software_spec=path, spec_validation=record)).record())["sum.dim_IntList"]
    assert row["evidence_state"] == "executed_spike"
    assert row["justification"].startswith("evidence-only: absent from captured workloads")


def test_omissions_are_listed(inputs):
    document = L.prepare(**inputs).record()
    omissions = document["omissions"]
    # convolution is captured and family-admitted; it has capsules, so it is not an omission.
    assert omissions["admitted_without_evidence"] == []
    assert omissions["compositions_without_capsule"] == ["lonely_chain"]
    assert omissions["accelerator_declarations_unexercised"] == ["unused_map"]
    assert omissions["unmodeled"] == ["sort.default"]
    assert omissions["mismatches"] == ["addmm.default"]
    assert omissions["unjustified_capsules"] == ["SY_orphan"]
    assert document["status"] == "diagnostic_incomplete"
    assert document["release_authority"] == "not_issued"


def test_admitted_capture_without_capsule_or_case_is_an_omission(inputs):
    inputs["capsules"] = [c for c in inputs["capsules"] if c["name"] != "SY_source_mlp"]
    inputs["capsules"] = [c for c in inputs["capsules"] if c["name"] != "SY_contraction_int8_aligned"]
    rows = _rows(L.prepare(**inputs).record())
    assert rows["convolution.default"]["evidence_state"] == "not_evidenced"
    assert rows["convolution.default"]["justification"].startswith("omission:")
    assert "convolution.default" in L.prepare(**inputs).record()["omissions"]["admitted_without_evidence"]


def test_every_capsule_records_what_it_adds(inputs):
    capsules = {row["name"]: row for row in L.prepare(**inputs).record()["capsules"]}
    assert "overload:convolution.default" in capsules["SY_source_mlp"]["adds"]
    assert capsules["SY_contraction_int8_aligned"]["adds"] == ["family:contraction"]
    assert capsules["SY_composition_scaled_operand_sum_q_tail"]["covers"] == ["composition:scaled_operand_sum"]
    assert capsules["SY_composition_scaled_operand_sum_q_tail"]["required_engine"] == "accelerator"
    assert capsules["SY_orphan"]["justification"] is None
    assert all(row["justification"] for name, row in capsules.items() if name != "SY_orphan")


def test_operation_domain_is_reported_when_declared(inputs, tmp_path):
    spec = dict(
        SPEC,
        operation_domains={"schema": "merlin.software_spec.operation_domains.v1", "required": ["ranks"]},
    )
    path = tmp_path / "domains.yaml"
    spec["operations"] = [dict(row, unconstrained={"layouts": "all layouts"}) for row in SPEC["operations"]]
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    record = tmp_path / "domains.json"
    record.write_text(json.dumps(_record(path, CASES)))
    document = L.prepare(**dict(inputs, software_spec=path, spec_validation=record)).record()
    domain = _rows(document)["relu.default"]["declaration"]["rows"][0]["operation_domain"]
    assert domain["ranks"] == [2]
    assert domain["layouts"] == {"unconstrained": "all layouts"}
    assert _rows(document)["view.default"]["declaration"]["rows"][0]["operation_domain"] is None


@pytest.mark.parametrize(
    "block",
    [
        {"relu_map": {"ranks": [2]}},
        [{"id": "relu_map"}],
        {"schema": "merlin.software_spec.operation_domains.v1", "required": ["layouts"]},
    ],
)
def test_invalid_operation_domains_are_refused(inputs, block):
    path = inputs["software_spec"]
    path.write_text(yaml.safe_dump(dict(SPEC, operation_domains=block)))
    inputs["spec_validation"].write_text(json.dumps(_record(path, CASES)))
    with pytest.raises(ValueError, match="operation_domains|domain axes"):
        L.prepare(**inputs)


def test_spec_validation_is_read_by_schema_and_digest(inputs, tmp_path):
    record = json.loads(inputs["spec_validation"].read_text())
    assert L.read_spec_validation(inputs["spec_validation"], sha256=record["sha256"])["sha256"] == record["sha256"]
    with pytest.raises(ValueError, match="pinned digest"):
        L.read_spec_validation(inputs["spec_validation"], sha256="0" * 64)
    tampered = dict(record, verdict="failed")
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="recorded digest"):
        L.read_spec_validation(path)
    foreign = dict(record, schema="merlin.phase0.spec_validation.v0")
    foreign["sha256"] = digest({k: v for k, v in foreign.items() if k != "sha256"})
    path.write_text(json.dumps(foreign))
    with pytest.raises(ValueError, match="schema"):
        L.read_spec_validation(path)


def test_stale_spec_is_refused(inputs, tmp_path):
    inputs["software_spec"].write_text(inputs["software_spec"].read_text() + "\n# edited\n")
    with pytest.raises(ValueError, match="different software spec"):
        L.prepare(**inputs)


def test_source_requirement_ledger_is_linked_by_digest(inputs):
    linked = {"schema": "merlin.phase0.source_requirement_ledger.v1", "requirements": []}
    linked["sha256"] = digest(linked)
    document = L.prepare(**inputs, source_requirement_ledger=linked).record()
    assert document["source_requirement_ledger"] == {"schema": linked["schema"], "sha256": linked["sha256"]}
    with pytest.raises(ValueError, match="intact ledger"):
        L.prepare(**inputs, source_requirement_ledger=dict(linked, requirements=[1]))


def test_verify_recomputes_and_detects_change(inputs):
    ledger = L.prepare(**inputs)
    assert L.prepare(**inputs).document_json == ledger.document_json
    assert L.verify(ledger, **inputs)["sha256"] == ledger.record()["sha256"]
    inputs["capsules"] = inputs["capsules"][:-1]
    with pytest.raises(ValueError, match="changed"):
        L.verify(ledger, **inputs)


def test_validation_case_identity_is_suite_local(inputs):
    path = inputs["spec_validation"]
    cases = [dict(CASES[0], source=source) for source in ("corpus_193", "bounded")]
    path.write_text(json.dumps(_record(inputs["software_spec"], cases)))
    assert len(L.read_spec_validation(path)["cases"]) == 2
    path.write_text(json.dumps(_record(inputs["software_spec"], cases + [cases[0]])))
    with pytest.raises(ValueError, match="repeats a"):
        L.read_spec_validation(path)


@pytest.mark.parametrize("input_kind", ["descriptor", "descriptor_path", "workload_spec"])
def test_disabled_composition_synthesis_is_out_of_scope(inputs, tmp_path, input_kind):
    workload_spec = {"synthesize_compositions": False}
    if input_kind == "descriptor_path":
        descriptor = tmp_path / "descriptor.yaml"
        descriptor.write_text(yaml.safe_dump({"workload_spec": workload_spec}))
        scope = {"descriptor": descriptor}
    elif input_kind == "descriptor":
        scope = {"descriptor": {"workload_spec": workload_spec}}
    else:
        scope = {"workload_spec": workload_spec}
    document = L.prepare(**inputs, **scope).record()
    assert {row["composition"] for row in document["compositions"]} == {"scaled_operand_sum", "lonely_chain"}
    assert {row["evidence_state"] for row in document["compositions"]} == {"out_of_scope"}
    assert all("synthesize_compositions is false" in row["justification"] for row in document["compositions"])
    assert document["omissions"]["compositions_without_capsule"] == []
    default = L.prepare(**inputs).record()
    assert document["operations"] == default["operations"]
    assert default["omissions"]["compositions_without_capsule"] == ["lonely_chain"]


@pytest.mark.parametrize(
    "scope",
    [{"descriptor": {}}, {"descriptor": {"workload_spec": {}}}, {"workload_spec": {"synthesize_compositions": True}}],
)
def test_composition_synthesis_defaults_to_enabled(inputs, scope):
    assert L.prepare(**inputs, **scope).record() == L.prepare(**inputs).record()
