"""Actual diagnostic processes retain joins; synthetic grades grant no roles."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase2 import component_runtime_stage_products as S
from merlin_experiments.phase2 import component_runtime_support as R
from merlin_experiments.phase2.contracts import StageGateError
from test_component_decode_products import prepared as decode_fixture
from test_component_runtime_support import prepared as prepared

from merlin.common import invocation_record as I


def fixture_source(source):
    return source.read_text()


def finite_report(path, *, missing=None):
    # Intentionally all OBSERVED strings. These diagnostics cannot turn unit
    # fixture bookkeeping into a full stage witness or runtime qualification.
    def rows(names):
        return [{"name": name, "outcome": "OBSERVED", "observation": "unit data only"} for name in names]

    data = {
        "emission_facets": rows(S.EMISSION_FACETS),
        "execution_effects": rows(S.REQUIRED_EXECUTION_EFFECTS),
        "unresolved_prerequisites": [],
    }
    if missing:
        data[missing].pop()
    path.write_text(json.dumps(data))


def registered(prepared, tmp_path):
    root, result_path, data, _, _ = decode_fixture(tmp_path)
    capsule, candidate = tmp_path / "original-capsule", tmp_path / "original-candidate"
    capsule.mkdir()
    candidate.mkdir()
    source = capsule / "source.mlir"
    source.write_text("fixture original source; no compiler semantic authority\n")
    (candidate / "driver.py").write_text("# synthetic inventory only\n")
    data["numeric_report"] = {"status": "pass", "scope": "diagnostic fixture only"}
    result_path.write_text(json.dumps(data))
    with I.observe_call(
        root,
        stage="primitive_source_verification",
        function=fixture_source,
        arguments={"scope": "fixture attribution only"},
        inputs=(source,),
    ) as invoked:
        invoked.returned(stdout=fixture_source(source))
    context = replace(
        prepared, source_pins=(*prepared.source_pins, (Path(__file__).resolve(), S.sha256_file(Path(__file__))))
    )
    products = S.collect(
        ordinary_result=result_path,
        capsule=capsule,
        candidate=candidate,
        target_descriptor=context.target_descriptor,
        coherent=True,
        grade_owner=root.parent,
    )
    summary = root / "capsule_result.json"
    summary.write_text(json.dumps({"status": "pass", "numeric": "pass", "ordinary_products": products}))
    R._GRADED[context][summary] = {"summary": S.member(summary), "products": products, "finite": ()}
    return context, root, summary, products


def produced(context, root, products, *, missing=None, omit_input=False):
    product = root / "finite-observation.json"
    inputs = [
        products["original_source"],
        *(products["coherent_products"][name] for name in ("elf", "decoder_product", "payload")),
    ]
    with I.observe_call(
        root,
        stage="actual_unit_facet_producer",
        function=finite_report,
        arguments={"scope": "synthetic diagnostic only"},
        inputs=tuple(Path(pin["path"]) for pin in inputs[1 if omit_input else 0 :]),
        outputs=(product,),
    ) as observed:
        finite_report(product, missing=missing)
        observed.returned()
    return product, observed.path


def test_registered_products_reopen_original_compiler_and_actual_decode(prepared, tmp_path):
    context, _, summary, products = registered(prepared, tmp_path)
    actual = context._graded_state(summary)
    assert actual["products"] == products and len(products["required_execution_effects"]) == 11
    assert "observer_integrity" in products["unknown"]
    candidate = Path(products["original_candidate_root"]) / "driver.py"
    candidate.write_text("changed compiler\n")
    with pytest.raises(StageGateError, match="compiler changed"):
        context._graded_state(summary)


def test_source_owned_finite_data_is_retained_but_all_observed_labels_cannot_issue_witness(prepared, tmp_path):
    context, root, summary, products = registered(prepared, tmp_path)
    product, record = produced(context, root, products)
    actual = context.attach_stage_observation(result_path=summary, product_path=product, producer_record=record)
    assert len(actual["execution_effects"]) == 11
    assert all(row["outcome"] == "OBSERVED" for row in actual["execution_effects"])
    with pytest.raises(StageGateError, match="remain UNKNOWN"):
        context.stage_verifier(result_path=summary)
    report = next(root.glob("stage_refusal_*.json"))
    data = json.loads(report.read_text())
    assert len(data["required_controls"]) == 14 and "physical_timing" in data["unknown"]
    assert data["finite_observations"] == [actual]
    stage = next(
        path
        for path in root.glob("invocations/*/invocation.json")
        if I.verify(path)["stage"] == "ordinary_stage_refusal_observation"
    )
    assert S.member(report) in I.verify(stage)["outputs"]


@pytest.mark.parametrize("defect", ["emission_facets", "execution_effects", "source_input", "source_selection"])
def test_disconnected_or_incomplete_finite_observation_cannot_attach(prepared, tmp_path, defect):
    context, root, summary, products = registered(prepared, tmp_path)
    product, record = produced(
        context,
        root,
        products,
        missing=defect if defect in ("emission_facets", "execution_effects") else None,
        omit_input=defect == "source_input",
    )
    if defect == "source_selection":
        context = replace(context, source_pins=prepared.source_pins)
        R._GRADED[context][summary] = {"summary": S.member(summary), "products": products, "finite": ()}
    with pytest.raises(StageGateError):
        context.attach_stage_observation(result_path=summary, product_path=product, producer_record=record)


def test_saved_numeric_summary_cannot_supply_context_owned_products(prepared, tmp_path):
    summary = tmp_path / "saved-summary.json"
    summary.write_text('{"status":"pass","numeric":"pass"}')
    with pytest.raises(StageGateError, match="no actual context-owned"):
        prepared.stage_verifier(result_path=summary)


def test_changed_finite_observation_cannot_hide_behind_original_attachment(prepared, tmp_path):
    context, root, summary, products = registered(prepared, tmp_path)
    product, record = produced(context, root, products)
    context.attach_stage_observation(result_path=summary, product_path=product, producer_record=record)
    product.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        context.stage_verifier(result_path=summary)


@pytest.mark.parametrize("field", ["required_execution_effects", "required_emission_facets", "unknown"])
def test_original_denominator_and_unknowns_cannot_be_dropped(prepared, tmp_path, field):
    context, _, summary, products = registered(prepared, tmp_path)
    products[field].pop()
    with pytest.raises(StageGateError, match="denominator or unknown"):
        context._graded_state(summary)


def test_finite_alias_is_refused_before_excluded_data_read(prepared, tmp_path):
    context, root, summary, _ = registered(prepared, tmp_path)
    excluded = tmp_path / "not-a-finite-observation"
    original = b"not JSON; must not be parsed through an alias"
    excluded.write_bytes(original)
    alias = root / "aliased-observation.json"
    alias.symlink_to(excluded)
    with pytest.raises(StageGateError, match="canonical admitted membership"):
        context.attach_stage_observation(result_path=summary, product_path=alias, producer_record=alias)
    assert excluded.read_bytes() == original


def test_returned_finite_data_cannot_mutate_the_retained_context_state(prepared, tmp_path):
    context, root, summary, products = registered(prepared, tmp_path)
    product, record = produced(context, root, products)
    returned = context.attach_stage_observation(result_path=summary, product_path=product, producer_record=record)
    returned["execution_effects"].clear()
    returned["unknown"].clear()
    with pytest.raises(StageGateError, match="remain UNKNOWN"):
        context.stage_verifier(result_path=summary)
    report = json.loads(next(root.glob("stage_refusal_*.json")).read_text())
    assert len(report["finite_observations"][0]["execution_effects"]) == 11
    assert "observer_integrity" in report["finite_observations"][0]["unknown"]
