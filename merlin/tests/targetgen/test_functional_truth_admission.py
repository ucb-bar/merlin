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


@pytest.mark.parametrize("dtype,element", [("float16", "f16"), ("bfloat16", "bf16"), ("int8", "i8")])
def test_source_component_form_admits_inner_broadcast_without_widening_the_byte_affine_form(dtype, element):
    from mlir_fixtures import pointwise

    from merlin.targetgen.functional_truth_admission import admit
    from merlin.targetgen.functional_truth_types import Case

    case = _case("aten.add.Tensor")
    for tensor, shape in zip(case["arguments"]["args"]["items"], [[2, 1], [1, 3]], strict=True):
        tensor.update(dtype=dtype, shape=shape)
    case["expected"].update(dtype=dtype, shape=[2, 3])
    spec = {"spec": resolve_spec(load_software_spec(SPEC), {})[0]}
    scalar = "addi" if element == "i8" else "addf"
    module = pointwise([f"tensor<2x3x{element}>"] * 2, f"tensor<2x3x{element}>", f"%v=arith.{scalar} %a,%b : {element}")
    decision = admit(Case(case, "test"), case, "N3", spec, source_module=module)
    assert decision["status"] == "admitted"
    assert decision["declaration"] == "component_linear"
    assert decision["observed_signature"]["broadcasting"] == "inner"
    assert decision["observed_signature"]["arithmetic"] == "integer_components"


def test_complex_arithmetic_is_not_relabelled_as_a_validated_real_quantized_operand(authored):
    from merlin.targetgen.functional_truth_admission import admit
    from merlin.targetgen.functional_truth_types import Case

    case = _case("aten.add.Scalar")
    case["arguments"]["args"]["items"][0]["dtype"] = "complex64"
    case["expected"]["dtype"] = "complex64"
    decision = admit(Case(case, "test"), case, "N3c", authored)
    assert decision["status"] != "admitted"
    assert decision["observed_signature"]["operand_dtype"] == "complex<f32>"


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


def _full_spec():
    return {"spec": resolve_spec(load_software_spec(SPEC), {})[0]}


def _int_case(overload, dtype="int32"):
    case = _case(overload)
    for tensor in [*case["arguments"]["args"]["items"], case["expected"]]:
        if isinstance(tensor, dict) and tensor.get("kind") == "tensor":
            tensor["dtype"] = dtype
    return case


_THRESHOLD = "%z=arith.constant 0 : i32\n%c=arith.cmpi slt,%a,%z : i32\n%v=arith.select %c,%z,%a : i32"
_SELF_TEST = (
    "%z=arith.constant 1 : i32\n%s=arith.subi %a,%z : i32\n%c=arith.cmpi eq,%a,%a : i32\n%v=arith.select %c,%s,%a : i32"
)
_LINEAR = "%z=arith.constant 1 : i32\n%v=arith.subi %a,%z : i32"


@pytest.mark.parametrize(
    "body,compares",
    [
        (_LINEAR, False),
        (_SELF_TEST, False),
        (_THRESHOLD, True),
        ("%z=arith.constant 0 : i32\n%v=arith.maxsi %a,%z : i32", True),
    ],
    ids=["linear", "self", "threshold", "maxsi"],
)
def test_component_form_does_not_carry_a_data_compare_select(body, compares):
    from mlir_fixtures import pointwise

    from merlin.targetgen.functional_truth_admission import admit
    from merlin.targetgen.functional_truth_types import Case

    case = _int_case("aten.relu.default")
    module = pointwise(["tensor<2x3xi32>"], "tensor<2x3xi32>", body)
    decision = admit(Case(case, "test"), case, "N3", _full_spec(), source_module=module)
    assert decision["observed_signature"]["arithmetic"] == "integer_components"
    assert decision["status"] == ("unsupported" if compares else "admitted")
    assert decision["declaration"] == "component_linear"


@pytest.mark.parametrize(
    "operation,element",
    [(name, "i32") for name in ("maxsi", "minsi", "maxui", "minui")]
    + [(name, "f32") for name in ("maximumf", "minimumf", "maxnumf", "minnumf")],
)
@pytest.mark.parametrize("distinct", [False, True])
def test_minmax_orders_only_distinct_data_words(operation, element, distinct):
    from mlir_fixtures import pointwise

    from merlin.targetgen.functional_truth_admission import _data_comparisons

    other = "%b" if distinct else "%a"
    module = pointwise(
        [f"tensor<2x3x{element}>"] * 2,
        f"tensor<2x3x{element}>",
        f"%v=arith.{operation} %a,{other} : {element}",
    )
    assert _data_comparisons(module) is distinct


def test_index_minmax_is_addressing_and_division_is_not_a_data_comparison():
    from mlir_fixtures import pointwise

    from merlin.targetgen.functional_truth_admission import _data_comparisons

    module = pointwise(
        ["tensor<2x3xi32>"] * 2,
        "tensor<2x3xi32>",
        "%i=linalg.index 0 : index\n%j=linalg.index 1 : index\n"
        "%bound=arith.maxsi %i,%j : index\n%v=arith.divsi %a,%b : i32",
    )
    assert not _data_comparisons(module)


def test_contraction_rank_is_the_rank_the_source_contracts():
    from mlir_fixtures import forward

    from merlin.targetgen.functional_truth_admission import admit
    from merlin.targetgen.functional_truth_types import Case

    case = _int_case("aten.convolution.default")
    module = forward(
        "%x:tensor<9x9xi32>,%y:tensor<9x2xi32>",
        "tensor<9x2xi32>",
        "%zero=arith.constant 0 : i32\n%init=tensor.splat %zero : tensor<9x2xi32>\n"
        "%r=linalg.matmul ins(%x,%y:tensor<9x9xi32>,tensor<9x2xi32>) outs(%init:tensor<9x2xi32>) -> tensor<9x2xi32>",
    )
    operator = admit(Case(case, "test"), case, "N1", _full_spec())
    source = admit(Case(case, "test"), case, "N1", _full_spec(), source_module=module)
    assert operator["observed_signature"]["rank"] == 4 and operator["status"] != "admitted"
    assert source["observed_signature"]["rank"] == 2
    assert source["status"] == "admitted" and source["declaration"] == "component_bilinear"


@pytest.mark.parametrize("with_matmul", [False, True])
@pytest.mark.parametrize("product", ["data", "literal", "index", "none"])
def test_unclassified_product_reduction_preserves_operator_result_rank(with_matmul, product):
    from mlir_fixtures import forward, generic

    from merlin.targetgen.functional_truth_admission import admit
    from merlin.targetgen.functional_truth_types import Case

    case = _int_case("aten.convolution.default")
    bodies = {
        "data": "%m=arith.muli %a,%b : i32\n%v=arith.addi %unused,%m : i32",
        "literal": "%c=arith.constant 2 : i32\n%m=arith.muli %a,%c : i32\n%v=arith.addi %unused,%m : i32",
        "index": "%i=linalg.index 0 : index\n%j=linalg.index 1 : index\n"
        "%m=arith.muli %i,%j : index\n%v=arith.addi %unused,%a : i32",
        "none": "%v=arith.addi %unused,%a : i32",
    }
    reduction = generic(
        [("%a4", "tensor<2x3x4x5xi32>"), ("%b4", "tensor<2x3x5x6xi32>")],
        "tensor<2x3x4x6xi32>",
        bodies[product],
        name="%r4",
        iterators=["parallel"] * 4 + ["reduction"],
        maps=[
            "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2,d4)>",
            "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d4,d3)>",
            "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2,d3)>",
        ],
    )
    matmul = (
        "\n%zero=arith.constant 0 : i32\n%init=tensor.splat %zero : tensor<4x6xi32>\n"
        "%r=linalg.matmul ins(%x,%y:tensor<4x5xi32>,tensor<5x6xi32>) outs(%init:tensor<4x6xi32>) -> tensor<4x6xi32>"
    )
    module = forward(
        "%a4:tensor<2x3x4x5xi32>,%b4:tensor<2x3x5x6xi32>,%x:tensor<4x5xi32>,%y:tensor<5x6xi32>",
        "(tensor<2x3x4x6xi32>, tensor<4x6xi32>)" if with_matmul else "tensor<2x3x4x6xi32>",
        reduction + (matmul if with_matmul else ""),
        ret="%r4, %r" if with_matmul else "%r4",
    )
    decision = admit(Case(case, "test"), case, "N1", _full_spec(), source_module=module)
    fallback = product == "data" or not with_matmul
    assert decision["observed_signature"]["rank"] == (4 if fallback else 2)
    # Additive scans retain their existing linear reduction admission.
    unsupported = product == "data" or (product == "literal" and not with_matmul)
    assert decision["status"] == ("unsupported" if unsupported else "admitted")


_FOLD = {"authority": "static_capture_input_descriptors", "inputs": [], "queries": [{"value": 3}]}


@pytest.mark.parametrize("folds,reads", [((_FOLD,), False), ((), False), ((_FOLD,), True)])
def test_descriptor_fold_without_tensor_reads_has_no_device_placement(folds, reads):
    from mlir_fixtures import forward, generic

    from merlin.targetgen.functional_truth_admission import admit
    from merlin.targetgen.functional_truth_types import Case

    case = _case("aten.sym_size.int")
    if reads:
        body = generic([("%x", "tensor<2x3xf32>")], "tensor<2x3xf32>", "%v=arith.negf %a : f32")
        module = forward("%x:tensor<2x3xf32>", "(tensor<2x3xf32>, tensor<2x3xf32>)", body, ret="%r, %x")
    else:
        body = "%c=arith.constant 3.0 : f32\n%r=tensor.splat %c : tensor<f32>"
        module = forward("%x:tensor<2x3xf32>", "(tensor<f32>, tensor<2x3xf32>)", body, ret="%r, %x")
    decision = admit(Case(case, "test"), case, "T0", _full_spec(), source_module=module, descriptor_folds=folds)
    folded = decision.get("reason") == "result folds from static input descriptors and reads no tensor data"
    assert folded == (bool(folds) and not reads)


@pytest.mark.parametrize("literal,admitted", [("1.25", True), ("0x7FC00000", False)])
def test_storage_only_fill_is_movement_only_for_finite_literals(literal, admitted):
    from mlir_fixtures import forward

    from merlin.targetgen.functional_truth_admission import admit
    from merlin.targetgen.functional_truth_types import Case

    case = _case("aten.full.default")
    module = forward("", "tensor<2x3xf32>", f"%c=arith.constant {literal} : f32\n%r=tensor.splat %c : tensor<2x3xf32>")
    decision = admit(Case(case, "test"), case, "T0", _full_spec(), source_module=module)
    assert (decision["status"] == "admitted") == admitted
    if admitted:
        assert decision["declaration"] == "movement_bitwise"
