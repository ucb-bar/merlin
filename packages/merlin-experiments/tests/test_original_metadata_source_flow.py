"""Ordinary source-v8/owner controls with explicitly substituted native observations."""

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
from merlin_experiments.phase0 import zero_return_intake as Z
from merlin_experiments.phase0.component_semantic_basis import PROVENANCE, SCHEMA, BasisSource, ComponentSemanticBasis
from test_original_metadata_sources import declarations
from test_original_scalar_binary_plan import budget

from merlin.common.jsonio import canonical_json
from merlin.targetgen import original_metadata_sources as M


def independent_sources(monkeypatch, tmp_path, *, version=8, limits=None, zero_bridge=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    documents = [
        declarations(),
        declarations(M.ASSERTION),
        declarations(result_dtype="float32", conditions={"copy": 1}),
        declarations(M.ASSERTION, conditions={"size": [101, 103]}),
    ]
    members, rows, schema_rows = [], [], []
    for index, document in enumerate(documents):
        graph_path = tmp_path / f"graph-{index}.json"
        graph_path.write_bytes(canonical_json(document[0]))
        target = document[0]["graphs"]["original"]["nodes"][1]["target"]
        members.append(
            {
                "id": f"original-{index}",
                "kind": "model2mlir_frontend_trace",
                "path": str(graph_path),
                "sha256": hashlib.sha256(graph_path.read_bytes()).hexdigest(),
                "schema": document[0]["schema"],
                "operation_semantics": [target],
                "effect_semantics": [],
            }
        )
        directory = tmp_path / "sources" / str(index)
        directory.mkdir(parents=True)
        rows.append(
            {
                "graph_path": str(graph_path),
                "request": {},
                "observation": str(directory / "defaults.json"),
                "invocation": {},
            }
        )
        schema_rows.append(
            {"graph_path": str(graph_path), "tensor_bindings": [], "zero_returns": {"original_index": index}}
        )
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
            documents[[item.path for item in basis.graph_sources].index(row["graph_path"])][:3]
        ),
    )
    zero_calls = []

    def verify_zero(**kwargs):
        zero_calls.append(kwargs)
        index = kwargs["member"]["original_index"]
        assert canonical_json(kwargs["trace"]) == canonical_json(documents[index][0])
        assert canonical_json(kwargs["schema_observation"]) == canonical_json(documents[index][1])
        assert kwargs["getter"] == {"diagnostic_only": "no native getter selected"}
        return copy.deepcopy(documents[index][3])

    monkeypatch.setattr(Z, "verify_returns", verify_zero)
    schemas = {
        "schema": "merlin.independent_operator_schema_intake.v3",
        "members": schema_rows,
        "zero_return_getter": {"diagnostic_only": "no native getter selected"},
    }
    if not zero_bridge:
        schemas["schema"] = "merlin.independent_operator_schema_intake.v2"
        schemas.pop("zero_return_getter")
        for row in schemas["members"]:
            row.pop("zero_returns")
    numerical = {
        "model": {"engine": "integer_reference"},
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "subnormal_operand_flush": False,
        "overflow": "bounded_exact",
    }
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
        schema_intake=owner,
        software=software,
        hardware=hardware,
        schemas=schemas,
        numerical=numerical,
        zero_calls=zero_calls,
    )


@pytest.mark.parametrize("limit,count", [("max_sources", 0), ("max_total_tensor_elements", 3)])
def test_budget_denials_keep_every_stable_prerequisite_and_all_original_admission_blockers(
    monkeypatch, tmp_path, limit, count
):
    before = independent_sources(monkeypatch, tmp_path / "before")
    ids = F.prepare(**{key: before[key] for key in ("schema_intake", "basis", "source_record")}, version=2).record()[
        "original_factory_prerequisite_ids"
    ]
    limits = budget()
    limits[limit] = 11 if limit == "max_sources" else 13
    after = independent_sources(monkeypatch, tmp_path / "after", limits=limits)
    record = F.prepare(**{key: after[key] for key in ("schema_intake", "basis", "source_record")}, version=2).record()
    assert record["original_factory_prerequisite_ids"] == ids
    assert len(record["factory_prerequisites"]) == 12
    assert sum(row["factory_state"] == "source_constructed" for row in record["factory_prerequisites"]) == count
    assert all(row["candidate_verdict"] == "not_evaluated" for row in record["factory_prerequisites"])


def test_v7_reproduces_all_twelve_missing_cast_assertion_source_slots_without_changing_legacy(monkeypatch, tmp_path):
    original = independent_sources(monkeypatch, tmp_path, version=7)
    source = original["source_record"]
    assert source["schema"] == C.INTEGER_SCALAR_SCHEMA and original["zero_calls"] == []
    assert all(row["forms"] == [] for row in source["members"])
    assert sum(len(row["source_members"]) for row in source["members"]) == 12
    assert all(member["status"] == "unknown" for row in source["members"] for member in row["source_members"])


def test_connected_v8_replays_zero_bridge_and_retains_complete_unsupported_and_admission_roster(monkeypatch, tmp_path):
    original = independent_sources(monkeypatch, tmp_path)
    source = original["source_record"]
    slots = [member for graph in source["members"] for member in graph["source_members"]]
    assert source["schema"] == C.METADATA_SCHEMA and len(slots) == 12
    assert [row["status"] for row in slots] == ["source_constructed"] * 6 + ["unknown"] * 6
    assert len(original["zero_calls"]) == 8  # fresh observation plus complete verifier replay
    assert all(row["status"] == "unknown" for graph in source["members"] for row in graph["policy_compatibility"])
    missing = C.required_unknowns(source, basis=original["basis"], unknown=A.P._unknown)
    assert sum(row["kind"] == "original_operator_factory" for row in missing) == 6
    assert sum(row["kind"] == "original_operator_admission" for row in missing) == 4
    assert original["numerical"]["readout_dtype"] == "i32" and original["numerical"]["overflow"] == "bounded_exact"
    assert C.reader_modules(8)[: len(C.reader_modules(7))] == C.reader_modules(7)
    assert "merlin_experiments.phase0.zero_return_intake" in C.reader_modules(8)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "order", "dtype", "none", "scalar_alias", "source", "bridge", "version"]
)
def test_complete_original_slots_source_bytes_types_and_native_zero_member_cannot_change(monkeypatch, tmp_path, change):
    original = independent_sources(monkeypatch, tmp_path)
    source = original["source_record"]
    if change == "missing":
        source["members"][0]["source_members"].pop()
    elif change == "extra":
        source["members"][0]["source_members"].append(copy.deepcopy(source["members"][0]["source_members"][0]))
    elif change == "order":
        source["members"][0]["source_members"].reverse()
    elif change == "dtype":
        source["members"][0]["forms"][0]["result_dtypes"] = ["int32"]
    elif change == "none":
        source["members"][1]["source_members"][0]["metadata"]["outputs"] = []
    elif change == "scalar_alias":
        source["members"][1]["source_members"][0]["metadata"]["dispatcher_result_count"] = False
    elif change == "source":
        Path(source["members"][1]["source_members"][0]["source"]["path"]).write_text("return None\n")
    elif change == "bridge":
        original["schemas"]["members"][1]["zero_returns"]["original_index"] = 0
    else:
        source["schema"] = C.INTEGER_SCALAR_SCHEMA
    with pytest.raises((AssertionError, ValueError)):
        C.verify(
            source,
            schema_record=original["schemas"],
            basis=original["basis"],
            numerical_semantics=original["numerical"],
        )


def test_new_factory_owner_preserves_status_independent_ids_and_refuses_old_vocabulary(monkeypatch, tmp_path):
    original = independent_sources(monkeypatch, tmp_path)
    inputs = {key: original[key] for key in ("schema_intake", "basis", "source_record")}
    with pytest.raises(ValueError):
        F.prepare(**inputs)
    owner = F.prepare(**inputs, version=2)
    record = owner.record()
    assert record["schema"] == F.METADATA_SCHEMA and len(record["factory_prerequisites"]) == 12
    assert record["admission"] == "not_issued"
    assert all(row["candidate_verdict"] == "not_evaluated" for row in record["factory_prerequisites"])
    expected = [
        A.P._unknown(
            "original_operator_factory",
            {
                "member": f"original-{i}",
                "node": call["node"],
                "target": call["target"],
                "cohort": cohort,
                "extent": extent,
            },
            "unused",
        )["id"]
        for i, graph in enumerate(original["source_record"]["members"])
        for call in graph["calls"]
        for cohort, extent in C.required_source_cohorts()
    ]
    assert record["original_factory_prerequisite_ids"] == expected


def test_exact_zero_return_join_keeps_the_original_binding_identity_without_admission(monkeypatch, tmp_path):
    before = independent_sources(monkeypatch, tmp_path / "before", version=7)
    missing = C.required_unknowns(before["source_record"], basis=before["basis"], unknown=A.P._unknown)
    binding_ids = [row["id"] for row in missing if row["kind"] == "original_call_binding"]
    assert len(binding_ids) == 2
    after = independent_sources(monkeypatch, tmp_path / "after")
    selected = F.prepare(**{key: after[key] for key in ("schema_intake", "basis", "source_record")}, version=2)
    bound = selected.record()
    assert set(binding_ids) <= set(bound["original_call_prerequisite_ids"])
    assert all(row["binding_state"] == "bound" for row in bound["call_prerequisites"])
    assert all(row["candidate_verdict"] == "not_evaluated" for row in bound["call_prerequisites"])
    denied = independent_sources(monkeypatch, tmp_path / "denied", zero_bridge=False)
    unavailable = F.prepare(
        **{key: denied[key] for key in ("schema_intake", "basis", "source_record")}, version=2
    ).record()
    assert unavailable["original_call_prerequisite_ids"] == bound["original_call_prerequisite_ids"]
    assert [
        row["id"] for row in unavailable["call_prerequisites"] if row["binding_state"] == "unavailable"
    ] == binding_ids
    assert unavailable["admission"] == bound["admission"] == "not_issued"
    assert denied["zero_calls"] == []


@pytest.mark.parametrize("change", ["missing", "extra", "order", "status", "scalar_alias"])
def test_binding_roster_cannot_be_edited_to_replace_complete_live_replay(monkeypatch, tmp_path, change):
    original = independent_sources(monkeypatch, tmp_path)
    selected = F.prepare(**{key: original[key] for key in ("schema_intake", "basis", "source_record")}, version=2)
    record = selected.record()
    if change == "missing":
        record["call_prerequisites"].pop()
    elif change == "extra":
        record["call_prerequisites"].append(copy.deepcopy(record["call_prerequisites"][0]))
    elif change == "order":
        record["call_prerequisites"].reverse()
    elif change == "status":
        record["call_prerequisites"][0]["binding_state"] = "admitted"
    else:
        record["call_prerequisites"][0]["source_producer_phase"] = False
    changed = F.OriginalFactoryPrerequisites(
        selected.schema_intake, selected.basis, selected.source_json, canonical_json(record), selected.version
    )
    with pytest.raises(ValueError, match="identities, slots or outcomes changed"):
        changed.record()


def test_v4_ledger_keeps_original_blockers_and_all_twelve_factory_prerequisites(monkeypatch, tmp_path):
    original = independent_sources(monkeypatch, tmp_path)
    unknowns = C.required_unknowns(original["source_record"], basis=original["basis"], unknown=A.P._unknown)
    call = original["source_record"]["members"][1]["calls"][0]
    old_binding = A.P._unknown(
        "original_call_binding",
        {"member": "original-1", "node": call["node"], "target": call["target"]},
        "old missing join",
    )
    unknowns.append(old_binding)
    unknowns.append(A.P._unknown("effect_domain", {"independent": "all effects still missing"}, "missing"))
    document = {
        "schema": L.SCHEMA,
        "original_required_ids": [row["id"] for row in unknowns],
        "requirements": [
            {
                "original_id": row["id"],
                "kind": row["kind"],
                "original_selector": row["selector"],
                "mandatory": True,
                "candidate_verdict": "not_evaluated",
            }
            for row in unknowns
        ],
        "mandatory_source_blockers": [row["id"] for row in unknowns],
        "status": "diagnostic_incomplete",
        "sha256": "old",
    }
    monkeypatch.setattr(
        L, "prepare_requirement_ledger", lambda **kwargs: L.SourceRequirementLedger(json.dumps(document))
    )
    inputs = dict(
        root=tmp_path,
        coverage={
            "automatic_derivation": {"original_call_sources": original["source_record"]},
            "obligations": [{"id": row["id"]} for row in unknowns],
        },
        purpose="source_preparation",
        schema_intake=original["schema_intake"],
        semantic_basis=original["basis"],
        software=original["software"],
        hardware=original["hardware"],
    )
    with pytest.raises(ValueError):
        L.prepare_prerequisite_ledger(**inputs)
    ledger = L.prepare_metadata_prerequisite_ledger(**inputs)
    result = L.verify_metadata_prerequisite_ledger(ledger, **inputs)
    assert result["schema"] == L.METADATA_PREREQUISITE_SCHEMA
    for key in ("original_required_ids", "requirements", "mandatory_source_blockers", "status"):
        assert result[key] == document[key]
    assert set(document["original_required_ids"]) < set(result["original_prerequisite_ids"])
    assert len(result["original_factory_prerequisites"]["factory_prerequisites"]) == 12
    assert old_binding["id"] in result["original_required_ids"]
    assert old_binding["id"] in result["original_prerequisite_ids"]
    bindings = result["original_factory_prerequisites"]["call_prerequisites"]
    assert next(row for row in bindings if row["id"] == old_binding["id"])["binding_state"] == "bound"
