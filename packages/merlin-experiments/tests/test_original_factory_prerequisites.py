"""Real source reconstruction with mocked native ownership, never admission."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_factory_prerequisites as F
from merlin_experiments.phase0.component_semantic_basis import PROVENANCE, SCHEMA, BasisSource, ComponentSemanticBasis
from test_original_integer_scalar_binary_plan import independent_sources
from test_original_scalar_binary_sources import declarations

from merlin.common.jsonio import canonical_json


@pytest.fixture
def original(monkeypatch, tmp_path):
    source, draft, numerical, schemas = independent_sources(monkeypatch, tmp_path)
    declaration = tmp_path / "basis.json"
    members = []
    for index, (row, scalar) in enumerate(zip(draft.graph_sources, (1.0, 1, 2**63), strict=True)):
        path = Path(row.path)
        trace = declarations(scalar=scalar)[0]
        path.write_text(json.dumps(trace))
        members.append(
            {
                "id": f"declared-{index}",
                "kind": "model2mlir_frontend_trace",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "schema": trace["schema"],
                "operation_semantics": ["aten.mul.Tensor"],
                "effect_semantics": [],
            }
        )
    declaration.write_text(
        json.dumps({"schema": SCHEMA, "status": "reviewed", "provenance": PROVENANCE, "members": members})
    )
    basis = ComponentSemanticBasis.load(
        declaration.read_bytes(),
        source=BasisSource(
            str(declaration), hashlib.sha256(declaration.read_bytes()).hexdigest(), "semantic-basis-roster"
        ),
        parent=tmp_path,
        routing={},
    )
    hardware = SimpleNamespace(sha256="1" * 64)

    class Software:
        sha256 = "2" * 64
        receipt_json = json.dumps({"semantic_basis_sha256": basis.source.sha256})

        def public_facts(self):
            return {"numerical_semantics": copy.deepcopy(numerical)}

    software = Software()
    software.hardware = hardware

    class Schema:
        sha256 = "3" * 64

        def record(self):
            return copy.deepcopy(schemas)

    schema = Schema()
    schema.software = software
    # The native issuer boundaries are explicitly substituted in these pure
    # controls. Actual C.verify / typed factories / source bytes still replay.
    monkeypatch.setattr(F, "IndependentOperatorSchemaIntake", Schema)
    monkeypatch.setattr(F, "IndependentSoftwareIntake", Software)
    return {
        "schema_intake": schema,
        "basis": basis,
        "source_record": source,
        "schemas": schemas,
        "numerical": numerical,
        "hardware": hardware,
        "software": software,
    }


def prepare(original):
    return F.prepare(**{key: original[key] for key in ("schema_intake", "basis", "source_record")})


def test_complete_roster_keeps_fulfilled_and_unavailable_factories_without_admission(original):
    record = prepare(original).record()
    rows = record["factory_prerequisites"]
    assert len(rows) == 9
    assert [row["factory_state"] for row in rows] == ["source_constructed"] * 6 + ["unavailable"] * 3
    expected = [
        F.A._unknown(
            "original_operator_factory",
            {
                "member": member["id"],
                "node": call["node"],
                "target": call["target"],
                "cohort": cohort,
                "extent": extent,
            },
            "any reason",
        )["id"]
        for member, graph in zip(
            json.loads(original["basis"].declaration_json)["members"], original["source_record"]["members"], strict=True
        )
        for call in graph["calls"]
        for cohort, extent in C.required_source_cohorts()
    ]
    assert record["original_factory_prerequisite_ids"] == expected
    assert all(row["mandatory"] is True and row["candidate_verdict"] == "not_evaluated" for row in rows)
    assert record["admission"] == "not_issued"
    assert record["unavailable_factory_prerequisite_ids"] == expected[-3:]
    assert rows[0]["metadata"] == original["source_record"]["members"][0]["source_members"][0]["metadata"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "reordered",
        "cohort",
        "extent_bool",
        "node",
        "target",
        "duplicate_call",
        "graph",
        "member_id",
        "old_schema",
    ],
)
def test_original_membership_refuses_before_reading_factory_status(original, monkeypatch, change):
    source = original["source_record"]
    rows = source["members"][0]["source_members"]
    if change == "missing":
        rows.pop()
    elif change == "extra":
        rows.append(copy.deepcopy(rows[-1]))
    elif change == "reordered":
        rows.reverse()
    elif change in {"cohort", "extent_bool", "node", "target"}:
        key, value = {
            "cohort": ("cohort", "development"),
            "extent_bool": ("extent", True),
            "node": ("node", "substituted"),
            "target": ("target", "substituted"),
        }[change]
        rows[0][key] = value
    elif change == "duplicate_call":
        source["members"][0]["calls"].append(copy.deepcopy(source["members"][0]["calls"][0]))
    elif change == "graph":
        source["members"].reverse()
    elif change == "member_id":
        raw = json.loads(original["basis"].declaration_json)
        raw["members"][1]["id"] = raw["members"][0]["id"]
        original["basis"] = replace(original["basis"], declaration_json=json.dumps(raw))
    else:
        source["schema"] = C.SCALAR_BINARY_SCHEMA
    monkeypatch.setattr(
        C, "verify", lambda *args, **kwargs: pytest.fail("status replay preceded complete identity derivation")
    )
    with pytest.raises(ValueError):
        prepare(original)


@pytest.mark.parametrize(
    "change", ["source_bytes", "source_alias", "graph_bytes", "basis_bytes", "native_binding", "basis_selection"]
)
def test_live_original_sources_and_selected_native_bindings_cannot_drift(original, tmp_path, change):
    owner = prepare(original)
    if change.startswith("source_"):
        path = Path(original["source_record"]["members"][0]["source_members"][0]["source"]["path"])
        if change == "source_bytes":
            path.write_text("substituted source")
        else:
            path.unlink()
            path.symlink_to(tmp_path / "absent")
    elif change == "graph_bytes":
        Path(original["basis"].graph_sources[0].path).write_text("substituted graph")
    elif change == "basis_bytes":
        Path(original["basis"].source.path).write_text("substituted original roster")
    elif change == "native_binding":
        original["schemas"]["members"][1]["tensor_bindings"][0]["native"]["wrapped_number"] = False
    else:
        original["software"].receipt_json = json.dumps({"semantic_basis_sha256": "f" * 64})
    with pytest.raises(ValueError):
        owner.record()


@pytest.mark.parametrize(
    "change", ["drop", "extra", "order", "source_state", "mandatory_int", "candidate_pass", "selector_bool"]
)
def test_saved_factory_rows_or_statuses_are_never_replay_authority(original, change):
    owner = prepare(original)
    document = json.loads(owner.document_json)
    rows = document["factory_prerequisites"]
    if change == "drop":
        rows.pop()
    elif change == "extra":
        rows.append(copy.deepcopy(rows[-1]))
    elif change == "order":
        rows.reverse()
    elif change == "source_state":
        rows[-1]["factory_state"] = "source_constructed"
    elif change == "mandatory_int":
        rows[0]["mandatory"] = 1
    elif change == "candidate_pass":
        rows[0]["candidate_verdict"] = "accepted"
    else:
        rows[0]["selector"]["extent"] = True
    with pytest.raises(ValueError):
        replace(owner, document_json=canonical_json(document)).record()


def test_real_construction_budget_transition_keeps_all_prerequisite_ids(original):
    before = prepare(original)
    denied = copy.deepcopy(original["source_record"])
    denied["budget"]["max_sources"] = 8  # All nine slots survive the real factory reservation refusal.
    totals = dict.fromkeys(("tensor_elements", "scalar_products", "source_bytes"), 0)
    for graph in denied["members"]:
        graph["source_members"] = [
            row
            for row, loader in C._sources(
                graph["calls"], graph["forms"], budget=denied["budget"], total=totals, requested=9, version=7
            )
        ]
    original["source_record"] = denied
    after = prepare(original).record()
    assert before.record()["original_factory_prerequisite_ids"] == after["original_factory_prerequisite_ids"]
    assert len(after["unavailable_factory_prerequisite_ids"]) == 9
    assert all(
        row["source"] is None and row["candidate_verdict"] == "not_evaluated" for row in after["factory_prerequisites"]
    )
    with pytest.raises(ValueError):
        replace(before, source_json=canonical_json(denied)).record()


def test_plain_saved_inputs_are_not_live_schema_ownership(original):
    original["schema_intake"] = {}
    with pytest.raises(ValueError, match="actual live"):
        prepare(original)
