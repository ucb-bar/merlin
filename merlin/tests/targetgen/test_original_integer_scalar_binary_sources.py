"""Independent declared native rows test readers, never actual boxed authority."""

import ast
import copy

import pytest
from test_original_scalar_binary_sources import declarations

from merlin.targetgen import original_scalar_binary_sources as S
from merlin.targetgen.frontend_original_call import call_contracts


def tensor_binding(call):
    literal = {"type": "int", "value": str(call["arguments"][1]["value"]["value"])}
    return {
        "request": {
            "node": call["node"],
            "target": call["target"],
            "schema": call["schema"],
            "argument_index": 1,
            "argument_path": "args/1",
            "literal": literal,
        },
        "status": "observed",
        "native": {
            "schema": call["schema"],
            "argument_name": "other",
            "source_allows_number": True,
            "wrapped_number": True,
            "shape": [],
            "dtype": "torch.int64",
            "element_bytes": 8,
            "literal": copy.deepcopy(literal),
            "disjoint_from_prior_live_boxes": True,
        },
    }


def declared_form(target="aten.mul.Tensor", scalar=1, *, bindings=None):
    documents = declarations(target, scalar)
    call = call_contracts(*documents)[0]
    bindings = [tensor_binding(call)] if bindings is None else bindings
    forms = S.scalar_binary_forms(
        *documents, version=2, tensor_bindings=bindings, numerical_semantics={"original_policy_pending": True}
    )
    return forms[0]


@pytest.mark.parametrize("target", sorted(S.TARGETS))
@pytest.mark.parametrize("scalar", [0, 1, -1, 2**53 + 2**29 + 1, -(2**60 + 2**36 + 1), -(2**63), 2**63 - 1])
def test_exact_integer_kind_complete_storage_and_native_binding_survive_bounded_construction(target, scalar):
    form = declared_form(target, scalar)
    assert form["status"] == "supported" and form["form_schema"] == S.INTEGER_FORM_SCHEMA
    source = S.scalar_binary_source(form, extent=2, max_tensor_elements=12)
    metadata = source.metadata()
    assert metadata["inputs"] == [{"name": "X", "dtype": "float32", "shape": [2, 3]}]
    assert metadata["tensor_elements"] == 12 and metadata["logical_payload_bytes"] == 48
    assert metadata["original_tensor_binding"] == form["tensor_binding"]
    assert metadata["parameters"]["other"] == {"kind": "int", "value": scalar}
    model = next(node for node in ast.parse(source.loader).body if isinstance(node, ast.ClassDef))
    literal = model.body[0].body[0].value.args[1]
    value = ast.literal_eval(literal)
    assert type(value) is int and value == scalar
    assert "torch.tensor" not in source.loader
    assert metadata["source_numerical_semantics"] == {"original_policy_pending": True}


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "node",
        "path",
        "index_bool",
        "target",
        "schema",
        "literal",
        "float_kind",
        "native_literal",
        "wrapped",
        "allowed",
        "dtype",
        "bytes_bool",
        "shape",
        "name",
        "disjoint",
        "unknown",
        "extra",
    ],
)
def test_integer_requires_exact_original_native_row_without_kind_storage_or_identity_substitution(change):
    documents = declarations(scalar=1)
    binding = tensor_binding(call_contracts(*documents)[0])
    bindings = [binding]
    if change == "missing":
        bindings = []
    elif change == "duplicate":
        bindings.append(copy.deepcopy(binding))
    elif change in {"node", "path", "target", "schema"}:
        key = {"path": "argument_path"}.get(change, change)
        binding["request"][key] = "changed"
    elif change == "index_bool":
        binding["request"]["argument_index"] = True
    elif change == "literal":
        binding["request"]["literal"]["value"] = "2"
    elif change == "float_kind":
        binding["request"]["literal"] = {"type": "float", "value_hex": (1.0).hex()}
    elif change == "native_literal":
        binding["native"]["literal"]["value"] = "2"
    elif change in {"wrapped", "allowed", "disjoint"}:
        binding["native"][
            {
                "wrapped": "wrapped_number",
                "allowed": "source_allows_number",
                "disjoint": "disjoint_from_prior_live_boxes",
            }[change]
        ] = False
    elif change == "dtype":
        binding["native"]["dtype"] = "torch.float32"
    elif change == "bytes_bool":
        binding["native"]["element_bytes"] = True
    elif change == "shape":
        binding["native"]["shape"] = [1]
    elif change == "name":
        binding["native"]["argument_name"] = "self"
    elif change == "unknown":
        binding.clear()
        binding.update(
            request=tensor_binding(call_contracts(*documents)[0])["request"],
            status="unknown",
            reason="actual wrapper unavailable",
        )
    else:
        binding["grant_numeric"] = True
    assert declared_form(bindings=bindings)["status"] == "unknown"


@pytest.mark.parametrize("scalar", [True, 2**63, -(2**63) - 1])
def test_boolean_and_outside_signed64_forms_stay_required(scalar):
    trace, schemas, defaults = declarations(scalar=scalar)
    assert S.scalar_binary_forms(trace, schemas, defaults, version=2, tensor_bindings=[])[0]["status"] == "unknown"


def test_legacy_integer_remains_unavailable_and_v2_float_preserves_literal_policy():
    documents = declarations(scalar=1)
    row = tensor_binding(call_contracts(*documents)[0])
    assert S.scalar_binary_forms(*documents, tensor_bindings=[row])[0]["status"] == "unknown"
    form = S.scalar_binary_forms(*declarations(scalar=-0.0), version=2)[0]
    source = S.scalar_binary_source(form, extent=1, max_tensor_elements=4)
    assert form["status"] == "supported"
    assert source.metadata()["parameters"]["other"]["value_hex"] == (-0.0).hex()
    assert source.metadata()["original_tensor_binding"] is None


@pytest.mark.parametrize("change", ["binding", "literal", "rank", "storage", "budget"])
def test_reconstruction_reopens_the_exact_binding_before_any_source_allocation(change):
    form = declared_form()
    limit = 12
    if change == "binding":
        form["tensor_binding"]["native"]["dtype"] = "torch.float32"
    elif change == "literal":
        form["parameters"]["other"]["value"] = 2
    elif change == "rank":
        form["rank"] = True
    elif change == "storage":
        form["result_roster"][0]["dtype"] = "float64"
    else:
        limit = 11
    with pytest.raises(ValueError):
        S.scalar_binary_source(form, extent=2, max_tensor_elements=limit)


@pytest.mark.parametrize(
    "dtype,shape", [("float16", (7, 9)), ("bfloat16", (7, 9)), ("float64", (7, 9)), ("float32", ())]
)
def test_unsupported_original_storage_and_scalar_rank_are_not_widened_by_integer_binding(dtype, shape):
    documents = declarations(scalar=1, dtype=dtype, shape=shape)
    call = call_contracts(*documents)[0]
    rows = S.scalar_binary_forms(*documents, version=2, tensor_bindings=[tensor_binding(call)])
    assert len(rows) == 1 and rows[0]["status"] == "unknown"


def test_floating_form_cannot_inherit_integer_boxing_evidence():
    form = S.scalar_binary_forms(*declarations(scalar=1.0), version=2)[0]
    form["tensor_binding"] = declared_form()["tensor_binding"]
    with pytest.raises(ValueError, match="cannot inherit"):
        S.scalar_binary_source(form, extent=2, max_tensor_elements=12)
