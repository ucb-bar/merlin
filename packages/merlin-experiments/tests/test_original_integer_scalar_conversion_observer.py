"""Observer seam controls use an explicit fake SDK, never actual promotion proof."""

import copy
import sys
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import original_scalar_conversion_observer as O
from test_original_integer_scalar_binary_sources import declared_form


def test_exact_literal_profile_keeps_python_integer_kind_without_binary64_conversion():
    value = 2**60 + 2**36 + 1
    assert O._literal(value, version=2) == {"kind": "int", "value": value}
    assert O._literal(-0.0, version=1) == {"kind": "float", "value_hex": (-0.0).hex()}
    with pytest.raises(ValueError):
        O._literal(value, version=1)


@pytest.mark.parametrize("value", [True, 2**63, -(2**63) - 1, None])
def test_new_profile_does_not_license_boolean_unsigned_or_unbound_literals(value):
    with pytest.raises(ValueError):
        O._literal(value, version=2)


def fake_sdk(monkeypatch, binding):
    observed = []

    class Tensor:
        def __init__(self, shape, dtype="torch.float32", value=None):
            self.shape, self.dtype, self.value = shape, dtype, value
            self.layout, self.device = "torch.strided", "cpu"

        def item(self):
            return self.value

        def element_size(self):
            return 8

    def operation(input_, scalar):
        observed.append((input_, scalar))
        return Tensor(list(input_.shape))

    sdk = SimpleNamespace(
        Tensor=Tensor,
        _C=SimpleNamespace(_is_alias_of=lambda lhs, rhs: lhs is rhs),
        ops=SimpleNamespace(aten=SimpleNamespace(mul=SimpleNamespace(Tensor=operation))),
        result_type=lambda input_, boxed: "torch.float32",
    )
    monkeypatch.setitem(sys.modules, "torch", sdk)
    original = copy.deepcopy(binding["native"])
    for name in ("shape", "dtype", "element_bytes", "literal", "disjoint_from_prior_live_boxes"):
        original.pop(name)

    def observe(namespace, overload, index, literal):
        observed.append((namespace, overload, index, type(literal), literal))
        return {**copy.deepcopy(original), "tensor": Tensor([], "torch.int64", literal)}

    return Tensor, SimpleNamespace(observe=observe), observed


def test_fresh_native_wrapper_and_both_direct_and_boxed_dispatch_preserve_original_int_argument(monkeypatch):
    form = declared_form(scalar=2**60 + 2**36 + 1)
    binding = form["tensor_binding"]
    tensor, getter, calls = fake_sdk(monkeypatch, binding)
    input_ = tensor([2, 3])
    live = []
    result = O._promotion({"target": form["target"], "original_tensor_binding": binding}, (input_,), getter, live)
    assert calls[0] == ("aten::mul", "Tensor", 1, int, 2**60 + 2**36 + 1)
    assert type(calls[1][1]) is int and calls[1][1] == 2**60 + 2**36 + 1
    assert calls[2][1] is live[0] and live[0].dtype == "torch.int64"
    assert result["original_binding"] == binding and result["native_binding"] == binding["native"]
    assert result["common_dtype"] == "torch.float32"
    assert result["outputs"] == result["boxed_outputs"] == [result["input"]]


@pytest.mark.parametrize("change", ["dtype", "wrapped", "literal", "schema", "kind", "noncanonical", "alias"])
def test_fresh_getter_drift_cannot_reuse_original_binding_metadata(monkeypatch, change):
    form = declared_form()
    binding = form["tensor_binding"]
    tensor, getter, _ = fake_sdk(monkeypatch, binding)
    input_ = tensor([2, 3])
    original_getter = getter.observe
    box = tensor([], "torch.int64", 1)

    def changed(*args):
        native = original_getter(*args)
        if change == "dtype":
            native["tensor"].dtype = "torch.float32"
        elif change == "wrapped":
            native["wrapped_number"] = False
        elif change == "literal":
            native["tensor"].value = 2
        elif change == "schema":
            native["schema"] = "changed"
        elif change == "alias":
            native["tensor"] = box
        return native

    getter.observe = changed
    if change == "kind":
        binding["request"]["literal"] = {"type": "float", "value_hex": (1.0).hex()}
    elif change == "noncanonical":
        binding["request"]["literal"]["value"] = "01"
    with pytest.raises(ValueError):
        O._promotion(
            {"target": form["target"], "original_tensor_binding": binding},
            (input_,),
            getter,
            [box] if change == "alias" else [],
        )


def test_v2_request_refuses_absent_or_extra_getter_before_any_upstream_import(monkeypatch, tmp_path):
    entered = []
    monkeypatch.setattr(O, "_selected_imports", lambda *args: entered.append(args))
    selected = {
        "schema": "merlin.original_scalar_conversion_request.v2",
        "capture_sources": [],
        "members": [],
        "max_product_bytes": 10000,
    }
    with pytest.raises(ValueError):
        O.observe(selected, capture=tmp_path, destination=tmp_path)
    assert entered == []
    selected.update(
        tensor_argument_getter=None,
        max_tensor_elements=100,
        max_promotion_tensor_elements=100,
        max_total_promotion_tensor_elements=1000,
    )
    with pytest.raises(ValueError):
        O.observe(selected, capture=tmp_path, destination=tmp_path)
    assert len(entered) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_promotion_tensor_elements", 18),
        ("max_total_promotion_tensor_elements", 37),
        ("max_tensor_elements", 11),
        ("promotion_tensor_elements", True),
        ("shape", [2, True]),
        ("shape", [2, 10**20]),
    ],
)
def test_complete_promotion_budgets_refuse_before_imports_or_allocations(field, value):
    member = {
        "index": 0,
        "target": "aten.mul.Tensor",
        "source": "explicit",
        "source_sha256": "a" * 64,
        "original_tensor_binding": declared_form()["tensor_binding"],
        "input": {
            "kind": "tensor",
            "dtype": "torch.float32",
            "shape": [2, 3],
            "layout": "torch.strided",
            "device": "cpu",
        },
        "promotion_tensor_elements": 19,
    }
    request = {
        "members": [member, copy.deepcopy(member)],
        "max_tensor_elements": 100,
        "max_promotion_tensor_elements": 100,
        "max_total_promotion_tensor_elements": 1000,
    }
    O._preflight_promotion(request)
    if field in {"max_promotion_tensor_elements", "max_total_promotion_tensor_elements", "max_tensor_elements"}:
        request[field] = value
    elif field == "shape":
        member["input"]["shape"] = value
    else:
        member[field] = value
    with pytest.raises(ValueError):
        O._preflight_promotion(request)
