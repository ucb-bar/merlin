"""Must-accelerate lanes follow the selected software spec's admission, not an operation table."""

import copy
import json
import os
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.functional_truth import SPEC_DECLINED, expectation, select_admission
from merlin.targetgen.software_spec import load_software_spec
from merlin.targetgen.spec_fact_drift import resolve_spec

ROOT = repo_root()
SPEC = ROOT / "examples/gemmini/target/software-spec.yaml"
EVIDENCE = json.loads((ROOT / "merlin/tests/data/exactness/core_aten_193_lane_evidence.json").read_text())


def _signature_only(spec):
    # These shape/alias tests isolate source signatures. Concrete execution
    # conditions are tested against the original declarations separately.
    for row in spec["operations"]:
        row.pop("admission_conditions", None)
    return spec


def _case(overload):
    return copy.deepcopy(next(case for case in EVIDENCE["cases"] if case["overload"] == overload))


@pytest.fixture(scope="module")
def authored():
    # The authored contraction declaration needs no fact resolution; fact-derived rows stay unresolved.
    return {"spec": resolve_spec(_signature_only(load_software_spec(SPEC)), {})[0]}


def test_native_lane_requires_a_selected_admission():
    with pytest.raises(ValueError, match="software spec admission"):
        expectation(_case("aten.mm.default"))
    assert expectation(_case("aten.abs.default"))["lane"] == "host-guard"


def test_admitted_contraction_is_a_device_obligation(authored):
    commitment = expectation(_case("aten.mm.default"), admission=authored)
    assert commitment["lane"] == "device"
    assert commitment["admission"]["declaration"] == "contraction"


def _int32_sub_tensor():
    case = _case("aten.sub.Tensor")
    case["case_id"] = "aten.sub.Tensor::03c9f285d4fbdd15"
    tensor = dict(kind="tensor", dtype="int32", shape=[1], stride=[1], storage_offset=1, values=[0])
    case["arguments"] = dict(args=dict(kind="tuple", items=[tensor, copy.deepcopy(tensor)]), kwargs=dict(alpha=0))
    case["expected"] = {**tensor, "storage_offset": 0}
    return case


def test_int32_sub_tensor_observes_component_arithmetic():
    from merlin.targetgen.functional_truth_admission import source_signature
    from merlin.targetgen.functional_truth_types import Case

    case = _int32_sub_tensor()
    admission = {"spec": resolve_spec(load_software_spec(SPEC), {})[0]}

    commitment = expectation(case, admission=admission)

    assert commitment["lane"] == "device"
    assert commitment["admission"]["declaration"] == "component_linear"
    signature = source_signature(Case(case, "test"), case, "N3", admission["spec"])[1]
    assert signature["arithmetic"] == "integer_components"
    assert signature["numerical_contract"] == {
        "representation": "signed_radix_256_components",
        "producer": "closed_static_linear_integer",
    }


@pytest.mark.parametrize(
    "field, value",
    [
        ("arithmetic", "integer_affine"),
        ("arithmetic", "integer_multiply_accumulate"),
        ("arithmetic", "readout_stages"),
        ("arithmetic", "integer_sum"),
        ("arithmetic", "none"),
        ("arithmetic", "not_a_representation"),
        ("representation", "signed_radix_16_components"),
        ("representation", "opaque_little_endian_bytes"),
        ("representation", "definitely_unproved"),
        ("representation", None),
        ("producer", "closed_static_bilinear_integer"),
        ("producer", None),
        ("declaration", "remove"),
        ("declaration", "rename"),
    ],
)
def test_component_declaration_cannot_change_the_observed_representation(field, value):
    from merlin.targetgen.functional_truth_admission import admit
    from merlin.targetgen.functional_truth_types import Case

    case = _int32_sub_tensor()
    # Nonzero alpha also exercises the ordinary two-input subtraction form.
    case["arguments"]["kwargs"]["alpha"] = 1
    spec = resolve_spec(load_software_spec(SPEC), {})[0]
    source = Case(case, "test")
    original = admit(source, case, "N3", {"spec": spec})
    assert original["status"] == "admitted"
    row = next(row for row in spec["operations"] if row["id"] == "component_linear")
    if field == "declaration":
        if value == "remove":
            spec["operations"].remove(row)
        else:
            row["id"] = "other_linear"
    elif field == "arithmetic":
        row[field] = value
    else:
        row["numerical_contract"][field] = value

    decision = admit(source, case, "N3", {"spec": spec})

    assert decision["status"] == "unsupported"
    assert decision["observed_signature"] == original["observed_signature"]


@pytest.mark.parametrize("overload", ["aten.add.Tensor", "aten.sub.Tensor"])
@pytest.mark.parametrize(
    "shapes, broadcasting",
    [
        ([[2, 3], [2, 3]], "none"),
        ([[2, 3], [1, 3]], "outer"),
        ([[2, 3], [3]], "outer"),
        ([[2, 4, 3], [2, 1, 3]], "outer"),
        ([[2, 3], [2, 1]], "inner"),
        ([[2, 1], [1, 3]], "inner"),
        ([[1, 3], [2, 1]], "inner"),
    ],
)
def test_broadcast_admission_preserves_the_innermost_axis(overload, shapes, broadcasting):
    from merlin.targetgen.functional_truth_admission import source_signature
    from merlin.targetgen.functional_truth_types import Case

    spec = _signature_only(load_software_spec(SPEC))
    row = next(row for row in spec["operations"] if row["id"] == "elementwise_map_standalone")
    row.pop("hardware")
    row["placement"] = "accelerator"
    row["signature"]["dtypes"] = ["int8"]
    admission = {"spec": resolve_spec(spec, {})[0]}
    case = _case(overload)
    for tensor, shape in zip(case["arguments"]["args"]["items"], shapes, strict=True):
        tensor["shape"] = shape
    case["expected"]["shape"] = [max(extents) for extents in zip(*shapes)] if len(shapes[1]) > 1 else shapes[0]
    commitment = expectation(case, admission=admission)
    assert source_signature(Case(case, "test"), case, "N3", admission["spec"])[1]["broadcasting"] == broadcasting
    assert commitment["lane"] == ("host-guard" if broadcasting == "inner" else "device")


def _in_place(case):
    case["mutated_arguments"] = ["args[0]"]


def _rank_five(case):
    for tensor in [*case["arguments"]["args"]["items"], case["expected"]]:
        tensor["shape"], tensor["stride"] = [1, 1, *tensor["shape"]], [0, 0, *tensor["stride"]]


@pytest.mark.parametrize("mutate", [_in_place, _rank_five], ids=["in_place", "undeclared_rank"])
def test_construction_outside_the_declared_domain_is_a_host_commitment(authored, mutate):
    case = _case("aten.bmm.default")
    mutate(case)
    commitment = expectation(case, admission=authored)
    assert commitment["truth"] == SPEC_DECLINED and commitment["lane"] == "host-guard"
    assert commitment["admission"]["status"] == "unsupported"


@pytest.mark.parametrize("changed_input", ["software_spec", "rtl_facts"])
def test_lane_selection_reopens_authority_at_the_same_paths(tmp_path, monkeypatch, changed_input):
    import yaml

    from merlin.targetgen import core_aten_spec_int8
    from merlin.targetgen.core_aten_capsules import _truth_admission

    # Supply capability derivation at its boundary; selection still reads the real spec and facts.
    def select_spec(target, facts, software_spec, host_capabilities):
        return {
            "spec": resolve_spec(_signature_only(load_software_spec(software_spec)), facts)[0],
            "identities": {},
            "capability_contract": {},
            "host": {},
        }

    monkeypatch.setattr(core_aten_spec_int8, "select_spec", select_spec)
    spec_path, facts_path = tmp_path / "software.yaml", tmp_path / "facts.json"
    spec_path.write_bytes(SPEC.read_bytes())
    facts_path.write_text(
        json.dumps(
            {
                "declared_unchecked_forms": {
                    "elementwise_map": {"standalone": {"placement": ["accelerator"], "dtypes": ["int8"]}}
                }
            }
        )
    )
    flow = dict(
        target="fixture", software_spec=spec_path, rtl_facts=facts_path, host_capabilities=tmp_path / "host.yaml"
    )
    case = _case("aten.add.Tensor")
    assert expectation(case, admission=_truth_admission(flow))["lane"] == "device"
    if changed_input == "software_spec":
        document = yaml.safe_load(spec_path.read_bytes())
        document["operations"]["elementwise_map_standalone"]["broadcasting"] = ["inner"]
        spec_path.write_text(yaml.safe_dump(document))
    else:
        facts_path.write_text("{}")
    assert expectation(case, admission=_truth_admission(flow))["lane"] == "host-guard"


def test_all_fixture_movement_signatures_are_admitted_without_command_observations(authored, monkeypatch):
    from merlin.targetgen.operator_review import ACCEPT_ENV

    monkeypatch.setenv(ACCEPT_ENV, "1")
    # Use the selected declaration rather than an op-name movement roster.
    movements = [
        case
        for case in EVIDENCE["cases"]
        if expectation(case, admission=authored).get("admission", {}).get("declaration") == "movement_bitwise"
    ]
    assert len(movements) == 31
    assert all(expectation(case, admission=authored)["lane"] == "device" for case in movements)


@pytest.mark.skipif(
    not os.environ.get("MERLIN_RTL_FACTS") or not Path(os.environ["MERLIN_RTL_FACTS"]).is_file(),
    reason="the fact-resolved spec needs the selected RTL facts (MERLIN_RTL_FACTS)",
)
@pytest.mark.parametrize("enabled", ["1", "0"])
def test_spec_admission_uses_source_observations_for_recorded_cases(monkeypatch, enabled):
    from merlin.targetgen.operator_review import ACCEPT_ENV

    monkeypatch.setenv(ACCEPT_ENV, enabled)
    admission = select_admission(
        "gemmini",
        rtl_facts=Path(os.environ["MERLIN_RTL_FACTS"]),
        software_spec=SPEC,
        host_capabilities=ROOT / "examples/gemmini/platform/host-capabilities-core-aten.yaml",
    )
    decisions = {case["overload"]: expectation(case, admission=admission) for case in EVIDENCE["cases"]}
    device = {op for op, result in decisions.items() if result["lane"] == "device"}
    recorded = set(EVIDENCE["device_lane"])
    assert device == recorded
    assert len(device) == 57


@pytest.mark.skipif(
    not os.environ.get("MERLIN_RTL_FACTS") or not Path(os.environ["MERLIN_RTL_FACTS"]).is_file(),
    reason="the fact-resolved spec needs the selected RTL facts (MERLIN_RTL_FACTS)",
)
@pytest.mark.parametrize("enabled", ["1", "0"])
def test_reviewed_fixture_signature_eligibility_is_preserved(monkeypatch, enabled):
    from merlin.targetgen.operator_review import ACCEPT_ENV

    monkeypatch.setenv(ACCEPT_ENV, enabled)
    admission = select_admission(
        "gemmini",
        rtl_facts=Path(os.environ["MERLIN_RTL_FACTS"]),
        software_spec=SPEC,
        host_capabilities=ROOT / "examples/gemmini/platform/host-capabilities-core-aten.yaml",
    )
    # Removing optional source conditions does not alter fixture eligibility.
    _signature_only(admission["spec"])
    device = {
        case["overload"] for case in EVIDENCE["cases"] if expectation(case, admission=admission)["lane"] == "device"
    }
    assert device == set(EVIDENCE["device_lane"])
    assert len(device) == 57


@pytest.mark.skipif(
    not os.environ.get("MERLIN_RTL_FACTS") or not Path(os.environ["MERLIN_RTL_FACTS"]).is_file(),
    reason="the fact-resolved spec needs the selected RTL facts (MERLIN_RTL_FACTS)",
)
@pytest.mark.parametrize("enabled,count", [("1", 57), ("0", 51)])
def test_fixture_admission_preserves_pending_row_gating(monkeypatch, enabled, count):
    from merlin.targetgen.operator_review import ACCEPT_ENV, PENDING

    monkeypatch.setenv(ACCEPT_ENV, enabled)
    admission = select_admission(
        "gemmini",
        rtl_facts=Path(os.environ["MERLIN_RTL_FACTS"]),
        software_spec=SPEC,
        host_capabilities=ROOT / "examples/gemmini/platform/host-capabilities-core-aten.yaml",
    )
    # Packet 2 reviews the live row. A pending fixture isolates packet 1's
    # acceptance-policy regression from the new execution-observation checks.
    _signature_only(admission["spec"])
    contraction = next(row for row in admission["spec"]["operations"] if row["id"] == "contraction")
    contraction["status"] = PENDING
    device = {
        case["overload"] for case in EVIDENCE["cases"] if expectation(case, admission=admission)["lane"] == "device"
    }
    assert len(device) == count
    assert len(EVIDENCE["device_lane"]) == EVIDENCE["lane_run"]["device"] == 57
    expected_drops = (
        {
            "aten.addmm.default",
            "aten.bmm.default",
            "aten.convolution.default",
            "aten.convolution_backward.default",
            "aten.cumsum.default",
            "aten.mm.default",
        }
        if enabled == "0"
        else set()
    )
    assert set(EVIDENCE["device_lane"]) - device == expected_drops
    for case in EVIDENCE["cases"]:
        if case["overload"] in expected_drops:
            decision = expectation(case, admission=admission)["admission"]
            assert decision["declaration"] == "contraction"
            assert decision["status"] == "unknown" and "operation review" in decision["reason"]
