"""Original source geometry stays separate from execution-owned placement.

These handwritten protocol fixtures check custody and full SSA reconstruction,
not source packing mathematics or GPU equivalence. Real packed kernels are
qualified independently by the producer integration suite.
"""

from copy import deepcopy

import pytest
from test_execution_model_correspondence import selection as selection
from xdsl.dialects import builtin, func

from merlin.frontends.execution_model_correspondence import checked_execution_model


@pytest.fixture(params=(2, 4))
def source_layout(selection, request):
    parent, bound, trace, interfaces = selection
    trace = deepcopy(trace)
    options = {"target": "source:fixture", "num_warps": request.param, "threads_per_warp": 32, "num_ctas": 1}
    mapping = {key: "captured_" + key for key in options}
    attributes = builtin.DictionaryAttr(
        {
            mapping[key]: builtin.StringAttr(value)
            if isinstance(value, str)
            else builtin.IntegerAttr(value, builtin.i64)
            for key, value in options.items()
        }
    )
    call = next(op for op in parent.walk() if isinstance(op, func.CallOp))
    call.attributes["prov.kernel_args"] = attributes
    record = trace["calls"][0]
    record.update(
        model_call_attributes={key: str(value) for key, value in call.attributes.items()},
        captured_scalar_arguments={mapping[key]: value for key, value in options.items()},
        source_layout_arguments=mapping,
        source_layout_bindings={
            key: {"schema_argument": name, "attribute": str(attributes.data[name])} for key, name in mapping.items()
        },
    )
    trace["source_layout_binding_profile"] = "t2m.captured-source-layout.v1"
    trace["kernels"][next(iter(interfaces))]["source_layout_specialization"] = {
        "schema": "t2m.source-map-layout.v1",
        "source_options": options,
    }
    return parent, bound, trace, interfaces


def test_original_typed_geometry_composes_with_full_execution_correspondence(source_layout):
    assert checked_execution_model(*source_layout)["calls"] == 1


@pytest.mark.parametrize(
    "damage",
    (
        "missing-profile",
        "unknown-profile",
        "missing-bindings",
        "missing-mapping",
        "changed-mapping",
        "changed-attribute",
        "changed-source-selection",
        "duplicate-argument",
        "missing-option",
        "different-parent-call",
        "omitted-parent-ssa",
    ),
)
def test_rehashed_geometry_cannot_change_original_source_semantics(source_layout, damage):
    parent, bound, trace, interfaces = source_layout
    record = trace["calls"][0]
    if damage == "missing-profile":
        trace.pop("source_layout_binding_profile")
    elif damage == "unknown-profile":
        trace["source_layout_binding_profile"] = "unchecked"
    elif damage == "missing-bindings":
        record.pop("source_layout_bindings")
    elif damage == "missing-mapping":
        record["source_layout_arguments"] = {}
    elif damage == "changed-mapping":
        record["source_layout_arguments"]["num_warps"] = "captured_num_ctas"
    elif damage == "changed-attribute":
        record["source_layout_bindings"]["num_warps"]["attribute"] = "8 : i64"
    elif damage == "changed-source-selection":
        trace["kernels"][next(iter(interfaces))]["source_layout_specialization"]["source_options"]["num_warps"] = 8
        record["source_layout_bindings"]["num_warps"]["attribute"] = "8 : i64"
    elif damage == "duplicate-argument":
        record["source_layout_arguments"]["num_ctas"] = "captured_num_warps"
        record["source_layout_bindings"]["num_ctas"]["schema_argument"] = "captured_num_warps"
    elif damage == "missing-option":
        record["source_layout_arguments"].pop("target")
        record["source_layout_bindings"].pop("target")
    elif damage == "different-parent-call":
        record["parent_call_ordinal"] = 1
    else:
        old = next(op for op in bound.walk() if op.name == "arith.trunci")
        old.parent.erase_op(old)
    with pytest.raises(ValueError):
        checked_execution_model(parent, bound, trace, interfaces)


@pytest.mark.parametrize("claim", ("profile", "arguments", "bindings"))
def test_unspecialized_kernels_cannot_invent_source_geometry(selection, claim):
    parent, bound, trace, interfaces = selection
    if claim == "profile":
        trace["source_layout_binding_profile"] = "t2m.captured-source-layout.v1"
    elif claim == "arguments":
        trace["calls"][0]["source_layout_arguments"] = {"num_warps": "captured_num_warps"}
    else:
        trace["calls"][0]["source_layout_bindings"] = {}
    with pytest.raises(ValueError):
        checked_execution_model(parent, bound, trace, interfaces)
