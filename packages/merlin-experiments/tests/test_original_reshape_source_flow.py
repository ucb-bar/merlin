"""Connected source-v10 construction with explicitly substituted native seams."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_factory_prerequisites as F
from merlin_experiments.phase0 import source_requirement_ledger as L
from merlin_experiments.phase0.component_semantic_basis import PROVENANCE, SCHEMA, BasisSource, ComponentSemanticBasis
from test_original_reshape_sources import declarations
from test_original_scalar_binary_plan import budget

from merlin.common.jsonio import canonical_json


def independent_sources(monkeypatch, tmp_path, *, version=10, limits=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    documents = [declarations((2, 3, 4), (3, -1)), declarations(dtype="float16"), declarations(shape=(True, 24))]
    members, rows, schema_rows = [], [], []
    for index, document in enumerate(documents):
        graph = tmp_path / f"graph-{index}.json"
        graph.write_bytes(canonical_json(document[0]))
        members.append(
            {
                "id": f"original-{index}",
                "kind": "model2mlir_frontend_trace",
                "path": str(graph),
                "sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
                "schema": document[0]["schema"],
                "operation_semantics": ["aten.reshape.default"],
                "effect_semantics": [],
            }
        )
        directory = tmp_path / "sources" / str(index)
        directory.mkdir(parents=True)
        rows.append(
            {"graph_path": str(graph), "request": {}, "observation": str(directory / "defaults.json"), "invocation": {}}
        )
        schema_rows.append({"graph_path": str(graph), "tensor_bindings": []})
    path = tmp_path / "basis.json"
    path.write_bytes(
        canonical_json({"schema": SCHEMA, "status": "reviewed", "provenance": PROVENANCE, "members": members})
    )
    basis = ComponentSemanticBasis.load(
        path.read_bytes(),
        source=BasisSource(str(path), hashlib.sha256(path.read_bytes()).hexdigest(), "semantic-basis-roster"),
        parent=tmp_path,
        routing={},
    )
    monkeypatch.setattr(C.D, "observe_members", lambda **kwargs: copy.deepcopy(rows))
    monkeypatch.setattr(
        C.D,
        "verify_member",
        lambda row, **kwargs: copy.deepcopy(
            documents[[item.path for item in basis.graph_sources].index(row["graph_path"])]
        ),
    )
    numerical = {
        "model": {"engine": "integer_reference"},
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "subnormal_operand_flush": False,
        "overflow": "bounded_exact",
    }
    schemas = {"schema": "merlin.independent_operator_schema_intake.v2", "members": schema_rows}
    record = C.observe(
        schema_record=schemas,
        basis=basis,
        numerical_semantics=numerical,
        budget=budget() if limits is None else limits,
        destination=tmp_path / "sources",
        version=version,
    )
    hardware = SimpleNamespace(sha256="1" * 64)

    class Software:
        sha256 = "2" * 64
        receipt_json = json.dumps({"semantic_basis_sha256": basis.source.sha256})

        def public_facts(self):
            return {"numerical_semantics": copy.deepcopy(numerical)}

    software = Software()
    software.hardware = hardware

    class SchemaOwner:
        sha256 = "3" * 64

        def record(self):
            return copy.deepcopy(schemas)

    owner = SchemaOwner()
    owner.software = software
    monkeypatch.setattr(F, "IndependentOperatorSchemaIntake", SchemaOwner)
    monkeypatch.setattr(F, "IndependentSoftwareIntake", Software)
    return dict(
        source_record=record,
        basis=basis,
        numerical=numerical,
        schema_intake=owner,
        schemas=schemas,
        software=software,
        hardware=hardware,
    )


def test_opt_in_v10_constructs_all_supported_cohorts_and_keeps_numeric_and_unsupported_rows(monkeypatch, tmp_path):
    original = independent_sources(monkeypatch, tmp_path)
    record = original["source_record"]
    slots = [row for graph in record["members"] for row in graph["source_members"]]
    assert record["schema"] == C.RESHAPE_SCHEMA and len(slots) == 9
    assert [row["status"] for row in slots] == ["source_constructed"] * 3 + ["unknown"] * 6
    assert [row["metadata"]["inputs"][0]["shape"] for row in slots[:3]] == [[1, 1, 3], [2, 3, 1], [3, 1, 3]]
    assert all(row["metadata"]["parameters"] == {"shape": [3, -1]} for row in slots[:3])
    assert all(row["metadata"]["source_numerical_semantics"] == original["numerical"] for row in slots[:3])
    missing = C.required_unknowns(record, basis=original["basis"], unknown=A.P._unknown)
    assert sum(row["kind"] == "original_operator_factory" for row in missing) == 6
    assert sum(row["kind"] == "original_operator_admission" for row in missing) == 3
    assert all(form["status"] == "unknown" for graph in record["members"] for form in graph["policy_compatibility"])
    assert C.reader_modules(10) == C.reader_modules(9) + ("merlin.targetgen.original_reshape_sources",)


def test_prior_v9_keeps_all_nine_factory_gaps_and_every_prior_reader(monkeypatch, tmp_path):
    original = independent_sources(monkeypatch, tmp_path, version=9)
    source = original["source_record"]
    assert source["schema"] == C.TRIANGULAR_SCHEMA
    assert all(graph["forms"] == [] for graph in source["members"])
    assert sum(len(graph["source_members"]) for graph in source["members"]) == 9
    assert all(row["status"] == "unknown" for graph in source["members"] for row in graph["source_members"])


@pytest.mark.parametrize(
    "key,limit,count",
    [
        ("max_sources", 8, 0),
        ("max_tensor_elements", 5, 0),
        ("max_total_tensor_elements", 20, 2),
        ("max_total_source_bytes", 1, 0),
    ],
)
def test_complete_reservations_refuse_before_omitting_any_required_original_slot(
    monkeypatch, tmp_path, key, limit, count
):
    limits = budget()
    limits[key] = limit
    original = independent_sources(monkeypatch, tmp_path, limits=limits)
    owner = F.prepare(**{key: original[key] for key in ("schema_intake", "basis", "source_record")}, version=4)
    rows = owner.record()["factory_prerequisites"]
    assert len(rows) == 9 and sum(row["factory_state"] == "source_constructed" for row in rows) == count
    assert all(row["candidate_verdict"] == "not_evaluated" for row in rows)


@pytest.mark.parametrize("change", ["missing", "extra", "order", "source", "literal", "geometry", "dtype", "version"])
def test_complete_source_members_types_literal_bindings_and_bytes_are_replayed(monkeypatch, tmp_path, change):
    original = independent_sources(monkeypatch, tmp_path)
    record = original["source_record"]
    graph = record["members"][0]
    if change == "missing":
        graph["source_members"].pop()
    elif change == "extra":
        graph["source_members"].append(copy.deepcopy(graph["source_members"][0]))
    elif change == "order":
        graph["source_members"].reverse()
    elif change == "source":
        Path(graph["source_members"][0]["source"]["path"]).write_text("return X\n")
    elif change == "literal":
        graph["forms"][0]["parameters"]["shape"][0] = 1
    elif change == "geometry":
        graph["forms"][0]["original_geometry"]["input_strides"] = [1, 1, 1]
    elif change == "dtype":
        graph["source_members"][0]["metadata"]["outputs"][0]["dtype"] = "int8"
    else:
        record["schema"] = C.TRIANGULAR_SCHEMA
    with pytest.raises(ValueError):
        C.verify(
            record,
            schema_record=original["schemas"],
            basis=original["basis"],
            numerical_semantics=original["numerical"],
        )


def test_stable_factory_and_call_ids_survive_new_construction_and_unavailable_formats(monkeypatch, tmp_path):
    before = independent_sources(monkeypatch, tmp_path / "before", version=9)
    prior = F.prepare(**{key: before[key] for key in ("schema_intake", "basis", "source_record")}, version=3).record()
    after = independent_sources(monkeypatch, tmp_path / "after")
    current = F.prepare(**{key: after[key] for key in ("schema_intake", "basis", "source_record")}, version=4).record()
    assert current["schema"] == F.RESHAPE_SCHEMA
    assert current["original_factory_prerequisite_ids"] == prior["original_factory_prerequisite_ids"]
    assert current["original_call_prerequisite_ids"] == prior["original_call_prerequisite_ids"]
    assert len(current["factory_prerequisites"]) == 9
    assert current["admission"] == "not_issued"
    with pytest.raises(ValueError):
        F.prepare(**{key: after[key] for key in ("schema_intake", "basis", "source_record")}, version=3)


def test_new_ledger_retains_every_original_requirement_and_completed_factory_prerequisite(monkeypatch, tmp_path):
    original = independent_sources(monkeypatch, tmp_path)
    coverage_rows = C.required_unknowns(original["source_record"], basis=original["basis"], unknown=A.P._unknown)
    requirements = [
        {
            "original_id": row["id"],
            "kind": row["kind"],
            "original_selector": row["selector"],
            "mandatory": True,
            "source_state": "unavailable",
        }
        for row in coverage_rows
    ]
    monkeypatch.setattr(
        L,
        "prepare_requirement_ledger",
        lambda **kwargs: SimpleNamespace(
            record=lambda: {
                "schema": L.SCHEMA,
                "original_required_ids": [row["id"] for row in coverage_rows],
                "requirements": requirements,
            }
        ),
    )
    result = L.prepare_reshape_prerequisite_ledger(
        root=tmp_path,
        coverage={
            "automatic_derivation": {"original_call_sources": original["source_record"]},
            "obligations": coverage_rows,
        },
        purpose="performance_campaign",
        **{key: original[key] for key in ("hardware", "software", "schema_intake")},
        semantic_basis=original["basis"],
    ).record()
    assert result["schema"] == L.RESHAPE_PREREQUISITE_SCHEMA
    assert result["requirements"] == requirements and result["original_required_ids"] == [
        row["id"] for row in coverage_rows
    ]
    assert set(result["original_required_ids"]) <= set(result["original_prerequisite_ids"])
    assert len(result["original_factory_prerequisites"]["factory_prerequisites"]) == 9
    assert len(result["original_factory_prerequisites"]["call_prerequisites"]) == 3
