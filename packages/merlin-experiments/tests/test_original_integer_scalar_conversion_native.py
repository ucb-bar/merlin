"""Fresh original wrapped int -> registered typed source; no numeric admission.

The unchanged six declared calls request all eighteen artificial fixture slots.
This roster never substitutes for protected mandatory coverage. Actual direct
and boxed dispatch readouts establish typed storage only, not value equivalence.
"""

import copy
from dataclasses import replace

import original_scalar_conversion_fixtures as F
import pytest
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_scalar_conversion as V
from xdsl.dialects.builtin import IntegerAttr, i64

from merlin.common.jsonio import canonical_json
from merlin.common.strict_json import loads
from merlin.targetgen import original_scalar_binary_correspondence as B
from merlin.targetgen import original_scalar_binary_sources as S

integer_native_originals = F.integer_native_originals


def receipt(owner):
    return loads(owner.receipt_json)


def source_products(owner, row):
    selected = loads(owner.selection.read_bytes())
    return {
        "form": row["form"],
        "source": S.scalar_binary_source(
            row["form"], extent=row["extent"], max_tensor_elements=selected["budget"]["max_tensor_elements"]
        ),
        "extent": row["extent"],
        "text": V._plain(row["products"]["source"]["path"]).read_text(),
        "trace": loads(V._plain(row["products"]["trace"]["path"]).read_bytes()),
        "registry": loads(V._plain(row["products"]["registry"]["path"]).read_bytes()),
        "source_inventory": {pin["path"]: pin["sha256"] for pin in receipt(owner)["source_pins"]},
        "limits": {key: selected["budget"][key] for key in B._LIMITS},
    }


def check_products(products):
    values = dict(products)
    form, source = values.pop("form"), values.pop("source")
    return B.verify(form, source, **values)


def test_actual_v2_public_getter_registered_importer_all_original_calls_and_slots(integer_native_originals):
    owner = integer_native_originals
    record = owner.record()
    assert record["schema"] == V.INTEGER_SCHEMA
    assert loads(owner.source_json)["schema"] == C.INTEGER_SCALAR_SCHEMA
    assert len(record["members"]) == 18
    assert all(row["state"] == "registered_source_checked" for row in record["members"])
    assert {row["original_member_id"] for row in record["members"]} == {
        "independent-" + str(index) for index in range(6)
    }
    assert [(row["original_member_id"], row["cohort"], row["extent"]) for row in record["members"]] == [
        ("independent-" + str(index), cohort, extent)
        for index in range(6)
        for cohort, extent in C.required_source_cohorts()
    ]
    assert record["totals"]["tensor_elements"] == 240
    assert record["totals"]["promotion_tensor_elements"] == 63
    assert record["unavailable_requirements"] == list(V._UNKNOWN)
    assert loads(owner.numerical_json) == {
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "model": {"engine": "integer_reference"},
        "subnormal_operand_flush": False,
        "overflow": "bounded_exact",
    }


def test_actual_original_int1_wrapper_promotion_complete_storage_and_direct_cast(integer_native_originals):
    owner = integer_native_originals
    rows = [row for row in receipt(owner)["members"] if row["original_member_id"] == "independent-5"]
    assert len(rows) == 3
    assert [row["cost"]["promotion_tensor_elements"] for row in rows] == [7, 19, 37]
    for row in rows:
        products = source_products(owner, row)
        assert check_products(products)["coefficient_f32_bits"] == "3f800000"
        assert products["form"]["parameters"]["other"] == {"kind": "int", "value": 1}
        binding = products["form"]["tensor_binding"]
        assert binding["request"]["literal"] == {"type": "int", "value": "1"}
        assert binding["request"]["argument_index"] == 1
        assert binding["request"]["argument_path"] == "args/1"
        native = binding["native"]
        assert native["shape"] == [] and native["dtype"] == "torch.int64" and native["element_bytes"] == 8
        assert native["wrapped_number"] is True and native["source_allows_number"] is True
        assert native["disjoint_from_prior_live_boxes"] is True
        assert native["literal"] == binding["request"]["literal"]
        registry = products["registry"]
        promotion = registry["tensor_argument"]
        assert promotion["original_binding"] == binding and promotion["native_binding"] == native
        descriptor = {
            "kind": "tensor",
            "dtype": "torch.float32",
            "shape": [row["extent"], row["extent"] + 1],
            "layout": "torch.strided",
            "device": "cpu",
        }
        assert promotion["common_dtype"] == "torch.float32"
        assert promotion["input"] == descriptor
        assert promotion["outputs"] == promotion["boxed_outputs"] == [descriptor]
        assert registry["events"][0]["literal"] == {"kind": "int", "value": 1}
        assert registry["events"][0]["dynamic_overrides"] == []
        assert registry["events"][0]["emitted_operations"] == [
            "arith.constant",
            "arith.sitofp",
            "tensor.splat",
            "tensor.empty",
            "linalg.generic",
        ]
    schema = loads(owner.schema_intake.receipt_json)
    assert schema["tensor_argument_getter"]["unknowns"] == [
        "installed_framework_sdk_build_correspondence",
        "complete_host_linker_loader_dependency_closure",
    ]


@pytest.mark.parametrize("change", ["same_type_integer_coefficient", "reversed_division"])
def test_actual_registered_bodies_refuse_coefficient_or_operand_order_substitution(integer_native_originals, change):
    owner = integer_native_originals
    original_id = "independent-5" if change == "same_type_integer_coefficient" else "independent-1"
    row = next(row for row in receipt(owner)["members"] if row["original_member_id"] == original_id)
    products = source_products(owner, row)
    check_products(products)
    module, _ = B._parse(products["text"], products["limits"])
    block = next(iter(module.body.block.ops)).body.block
    if change == "same_type_integer_coefficient":
        next(iter(block.ops)).properties["value"] = IntegerAttr(2, i64)
    else:
        generic = tuple(block.ops)[-2]
        calculation = next(iter(generic.body.block.ops))
        calculation.operands = list(reversed(calculation.operands))
    products["text"] = str(module)
    with pytest.raises(ValueError):
        check_products(products)


def test_actual_integer_binding_cannot_be_replaced_by_a_float_literal(integer_native_originals):
    owner = integer_native_originals
    row = next(row for row in receipt(owner)["members"] if row["original_member_id"] == "independent-5")
    products = source_products(owner, row)
    check_products(products)
    products["registry"]["events"][0]["literal"] = {"kind": "float", "value_hex": (1.0).hex()}
    with pytest.raises(ValueError):
        check_products(products)


def test_complete_original_slot_and_readout_budgets_refuse_before_another_native_execution(integer_native_originals):
    owner = integer_native_originals
    source_record = loads(owner.source_json)
    budget = loads(owner.selection.read_bytes())["budget"]
    members, totals = V.required_members(source_record, basis=owner.basis, budget=budget)
    assert len(members) == 18 and all(source is not None for _, source in members)
    assert totals["promotion_tensor_elements"] == 63
    smaller = dict(budget, max_members=17)
    with pytest.raises(ValueError):
        V.required_members(source_record, basis=owner.basis, budget=smaller)
    changed = copy.deepcopy(source_record)
    changed["members"][-1]["source_members"].pop()
    with pytest.raises(ValueError):
        V.required_members(changed, basis=owner.basis, budget=budget)
    unavailable, _ = V.required_members(
        source_record, basis=owner.basis, budget=dict(budget, max_promotion_tensor_elements=18)
    )
    assert len(unavailable) == 18
    assert sum(source is None for _, source in unavailable) == 2


def test_copied_live_owner_and_modified_saved_v2_receipt_cannot_issue_conversion(integer_native_originals):
    owner = integer_native_originals
    with pytest.raises(ValueError, match="actual live"):
        replace(owner).record()
    changed = receipt(owner)
    changed["members"].pop()
    with pytest.raises(ValueError, match="actual live"):
        replace(owner, receipt_json=canonical_json(changed)).record()
