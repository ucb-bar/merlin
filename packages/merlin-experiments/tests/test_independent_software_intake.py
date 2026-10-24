"""Protected minimal semantic content, actual graph replay and grant exclusions."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import software_intake as S
from merlin_experiments.phase0.evidence import EvidenceSource
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal, issue_independent_hardware_intake
from merlin_experiments.phase2.component_experiment import ComponentView
from test_component_semantic_basis import select_basis
from test_independent_rtl_intake import selected  # noqa: F401 -- actual native hardware replay fixture


def _write(path, document):
    path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def selection(selected, tmp_path):  # noqa: F811 -- registered actual native replay fixture
    hardware = issue_independent_hardware_intake(**selected)
    recipe = tmp_path / "independent-recipe.json"
    _write(recipe, {})
    _, roster, _, graph, _ = select_basis(recipe)
    numerics = {
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "model": {"engine": "integer_reference"},
        "subnormal_operand_flush": False,
        "overflow": "bounded_exact",
    }
    source = tmp_path / "minimal-spec.json"
    spec = {
        "schema": "merlin.software_spec.v1",
        "target": hardware.target,
        "status": "reviewed",
        "numerical_semantics": numerics,
        "operations": {"movement": {"families": ["movement"], "hardware": "standalone"}},
    }
    source_pin = _write(source, spec)
    review = tmp_path / "protected-review.json"
    _write(
        review,
        {
            "schema": S.REVIEW_SCHEMA,
            "target": hardware.target,
            "source": source_pin,
            "semantic_basis": {"path": str(roster), "sha256": hashlib.sha256(roster.read_bytes()).hexdigest()},
            "numerical_choices": numerics,
            "operation_basis": [
                {"owner": "movement", "member": "independent-copy", "operations": ["aten.clone.default"]}
            ],
        },
    )
    return dict(
        hardware=hardware,
        source=source,
        review=review,
        forbidden_roots=selected["forbidden_roots"],
        output_root=tmp_path / "software-intake",
    )


def _repin(selection, spec):
    review = json.loads(selection["review"].read_bytes())
    review["source"] = _write(selection["source"], spec)
    _write(selection["review"], review)


def test_actual_minimal_schema_original_graph_replay_and_exact_public_projection(selection):
    authority = S.issue_independent_software_intake(**selection)
    facts = authority.public_facts()
    assert set(facts) == {"schema", "target", "status", "numerical_semantics", "operations"}
    assert facts["operations"][0]["id"] == "movement"
    assert "43" not in json.dumps(facts) and "79" not in json.dumps(facts)
    receipt = json.loads(authority.receipt_json)
    assert "hardware_numerical_support" in receipt["unknowns"]
    assert any(pin.role == "independent-example-graph" for pin in authority.source_pins)
    authority.verify_selected_source(selection["source"])
    authority.verify_public_facts(selection["output_root"] / "software-spec.json")
    facts["operations"][0]["shape_bounds"] = [1, 22]
    assert "shape_bounds" not in authority.public_facts()["operations"][0]


def test_saved_or_reconstructed_software_grant_is_not_live(selection):
    authority = S.issue_independent_software_intake(**selection)
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        replace(authority).verify()
    object.__setattr__(authority, "facts_json", b"{}")
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        authority.verify()


@pytest.mark.parametrize(
    "extra", ["component_performance", "capture_recipe", "capability_contract", "profiling", "schedule"]
)
def test_legacy_extra_fields_refuse_before_graph_access_even_with_repin_and_reviewed_status(
    selection, monkeypatch, extra
):
    spec = json.loads(selection["source"].read_bytes())
    spec[extra] = {"target_cycles": 22, "reviewed": True}
    _repin(selection, spec)
    monkeypatch.setattr(S, "_basis", lambda *args: pytest.fail("legacy spec reached independent graph reader"))
    with pytest.raises(RtlIntakeRefusal, match="minimal semantic schema"):
        S.issue_independent_software_intake(**selection)
    assert not selection["output_root"].exists()


@pytest.mark.parametrize(
    "location,key,value",
    [
        ("operation", "description", "use a fast host schedule"),
        ("operation", "evidence", "old model capture"),
        ("signature", "shape_bounds", {"min": 22, "max": 50}),
        ("signature", "layouts", ["packed_workload_rows"]),
        ("model", "source_root_path", "/old/backend"),
        ("numeric", "target_cycles", 22),
    ],
)
def test_nested_history_or_implementation_metadata_cannot_enter_minimal_projection(selection, location, key, value):
    spec = json.loads(selection["source"].read_bytes())
    target = {
        "operation": spec["operations"]["movement"],
        "signature": spec["operations"]["movement"].setdefault("signature", {}),
        "model": spec["numerical_semantics"]["model"],
        "numeric": spec["numerical_semantics"],
    }[location]
    target[key] = value
    _repin(selection, spec)
    with pytest.raises(RtlIntakeRefusal):
        S.issue_independent_software_intake(**selection)


def test_status_and_source_hash_without_actual_review_correspondences_refuse(selection):
    review = json.loads(selection["review"].read_bytes())
    _write(
        selection["review"],
        {"schema": S.REVIEW_SCHEMA, "target": review["target"], "source": review["source"], "status": "reviewed"},
    )
    with pytest.raises(RtlIntakeRefusal, match="closed source"):
        S.issue_independent_software_intake(**selection)


def test_declared_links_must_match_actual_source_calls_and_every_owner(selection):
    review = json.loads(selection["review"].read_bytes())
    review["operation_basis"][0]["operations"] = ["aten.matmul.default"]
    _write(selection["review"], review)
    with pytest.raises(RtlIntakeRefusal, match="actual original example calls"):
        S.issue_independent_software_intake(**selection)
    review["operation_basis"] = []
    _write(selection["review"], review)
    with pytest.raises(RtlIntakeRefusal, match="actual independent source-operation"):
        S.issue_independent_software_intake(**selection)


def test_numeric_review_and_actual_source_drift_refuse(selection):
    review = json.loads(selection["review"].read_bytes())
    changed = copy.deepcopy(review)
    changed["numerical_choices"]["overflow"] = "modular_wrap"
    _write(selection["review"], changed)
    with pytest.raises(RtlIntakeRefusal, match="numerical choices"):
        S.issue_independent_software_intake(**selection)
    _write(selection["review"], review)
    authority = S.issue_independent_software_intake(**selection)
    selection["source"].write_bytes(selection["source"].read_bytes() + b"\n")
    with pytest.raises(RtlIntakeRefusal, match="changed"):
        authority.verify()


def test_protected_source_refuses_before_read(selection, monkeypatch):
    source = selection["forbidden_roots"][0] / "old-spec.json"
    source.write_text("legacy private sentinel")
    selection["source"] = source
    read = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda path: pytest.fail("protected legacy spec read") if path == source else read(path)
    )
    with pytest.raises(RtlIntakeRefusal, match="protected implementation"):
        S.issue_independent_software_intake(**selection)


def test_component_binding_keeps_source_and_reviewed_numeric_identity(selection):
    authority = S.issue_independent_software_intake(**selection)
    evidence = SimpleNamespace(
        source_snapshots=(EvidenceSource(selection["source"], "software-spec", selection["source"].read_bytes()),),
        software_spec=authority.public_facts(),
    )
    assert S.bind_component_software(evidence, authority) == {"software_intake_sha256": authority.sha256}
    evidence.software_spec["numerical_semantics"]["overflow"] = "modular_wrap"
    with pytest.raises(RtlIntakeRefusal, match="numerics differ"):
        S.bind_component_software(evidence, authority)
    evidence.software_spec = authority.public_facts()
    evidence.software_spec["operations"][0]["families"] = ["contraction"]
    with pytest.raises(RtlIntakeRefusal, match="semantic selectors differ"):
        S.bind_component_software(evidence, authority)


@pytest.mark.parametrize(
    "original,changed",
    [
        ({"aliasing": "forbid"}, {"aliasing": "allow"}),
        ({"ordered_operand_dtypes": ["int8"]}, {"ordered_operand_dtypes": ["i32"]}),
        ({"quantization_parameters": {"zero_point": 0}}, {"quantization_parameters": {"zero_point": 7}}),
        ({"aliasing": "forbid"}, {}),
    ],
)
def test_effective_binding_cannot_weaken_original_signature_constraints(selection, original, changed):
    spec = json.loads(selection["source"].read_bytes())
    spec["operations"]["movement"]["signature"] = original
    _repin(selection, spec)
    authority = S.issue_independent_software_intake(**selection)
    evidence = SimpleNamespace(
        source_snapshots=(EvidenceSource(selection["source"], "software-spec", selection["source"].read_bytes()),),
        software_spec=authority.public_facts(),
    )
    evidence.software_spec["operations"][0]["signature"].update({"dtypes": ["i8"]})
    evidence.software_spec["operations"][0]["placement"] = "accelerator"
    assert S.bind_component_software(evidence, authority) == {"software_intake_sha256": authority.sha256}
    evidence.software_spec["operations"][0]["signature"] = changed
    with pytest.raises(RtlIntakeRefusal, match="original signature constraints"):
        S.bind_component_software(evidence, authority)


def test_resolver_record_carries_an_unmarked_form_but_never_a_substituted_one(selection):
    authority = S.issue_independent_software_intake(**selection)
    evidence = SimpleNamespace(
        source_snapshots=(EvidenceSource(selection["source"], "software-spec", selection["source"].read_bytes()),),
        software_spec=authority.public_facts(),
    )
    row = evidence.software_spec["operations"][0]
    form = row.pop("hardware")
    row["placement"] = "unknown"  # the facts established no form: unresolved rows carry no marker
    with pytest.raises(RtlIntakeRefusal, match="hardware form selection"):
        S.bind_component_software(evidence, authority)
    evidence.software_spec_derivation = {"declarations": [{"id": row["id"], "form": form, "status": "unresolved"}]}
    assert S.bind_component_software(evidence, authority) == {"software_intake_sha256": authority.sha256}
    row["derived_from_facts"] = {"form": f"not-{form}"}
    with pytest.raises(RtlIntakeRefusal, match="hardware form selection"):
        S.bind_component_software(evidence, authority)
    del row["derived_from_facts"]
    evidence.software_spec_derivation["declarations"][0]["form"] = f"not-{form}"
    with pytest.raises(RtlIntakeRefusal, match="hardware form selection"):
        S.bind_component_software(evidence, authority)


def test_effective_binding_cannot_duplicate_an_owner_to_drop_an_original(selection):
    spec = json.loads(selection["source"].read_bytes())
    spec["operations"]["second-copy"] = copy.deepcopy(spec["operations"]["movement"])
    _repin(selection, spec)
    review = json.loads(selection["review"].read_bytes())
    review["operation_basis"].append({**review["operation_basis"][0], "owner": "second-copy"})
    _write(selection["review"], review)
    authority = S.issue_independent_software_intake(**selection)
    evidence = SimpleNamespace(
        source_snapshots=(EvidenceSource(selection["source"], "software-spec", selection["source"].read_bytes()),),
        software_spec=authority.public_facts(),
    )
    assert S.bind_component_software(evidence, authority) == {"software_intake_sha256": authority.sha256}
    evidence.software_spec["operations"][1] = copy.deepcopy(evidence.software_spec["operations"][0])
    with pytest.raises(RtlIntakeRefusal, match="every original owner exactly once"):
        S.bind_component_software(evidence, authority)


def test_ordinary_fresh_evidence_route_binds_minimal_source_and_refuses_legacy_selection_before_read(selection):
    from merlin_experiments.phase0.evidence import select_evidence

    authority = S.issue_independent_software_intake(**selection)
    common = dict(
        target=selection["hardware"].target,
        hardware_intake=selection["hardware"],
        facts_path=next(
            Path(pin.path) for pin in selection["hardware"].source_pins if Path(pin.path).name == "facts.json"
        ),
        software_intake=authority,
        capability_contract_path={"name": selection["hardware"].target},
    )
    actual = select_evidence(**common, software_spec=selection["source"])
    assert S.bind_component_software(actual, authority) == {"software_intake_sha256": authority.sha256}
    other = selection["source"].with_name("legacy-unopened.json")
    other.write_text("an invalid legacy document that must never be parsed")
    with pytest.raises(RtlIntakeRefusal, match="source differs"):
        select_evidence(**common, software_spec=other)


def test_exact_public_view_member_cannot_replace_minimal_projection_with_legacy_spec(selection, tmp_path):
    authority = S.issue_independent_software_intake(**selection)
    root = tmp_path / "public-view"
    (root / "contract").mkdir(parents=True)
    member = root / "contract/software_spec.json"
    member.write_bytes(authority.facts_json)
    manifest = {
        "schema": "merlin.component_agent_view.v1",
        "generation_sha256": "a" * 64,
        "library_sha256": "b" * 64,
        "members": [
            {
                "path": "contract/software_spec.json",
                "sha256": hashlib.sha256(member.read_bytes()).hexdigest(),
                "n_bytes": member.stat().st_size,
                "role": "contract",
            }
        ],
    }
    _write(root / "manifest.json", manifest)
    view = ComponentView(root, hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(), "a" * 64, "b" * 64)
    authority.verify_public_fact_view(view)
    manifest["members"][0]["path"] = "contract/other.json"
    member.rename(root / "contract/other.json")
    _write(root / "manifest.json", manifest)
    view = replace(view, manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())
    with pytest.raises(RtlIntakeRefusal, match="exact protected minimal software"):
        authority.verify_public_fact_view(view)
