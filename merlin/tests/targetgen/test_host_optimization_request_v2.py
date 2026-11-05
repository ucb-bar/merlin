"""Closed versioned choices select the checked planner under original permissions."""

import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_host_optimization_selection import _LOGICAL, _selection
from test_original_host_optimization import _TARGET, _original, _policy, _scope
from xdsl.dialects.builtin import i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.linalg import LinalgFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.tensor import TensorFunctions
from xdsl.interpreters.utils.ptr import TypedPtr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import toolchain
from merlin.llvmlower.host_optimization_request import (
    REQUEST_SCHEMA,
    REQUEST_SCHEMA_V2,
    HostOptimizationRequest,
    prepare_host_optimization,
    request_from_record,
    write_host_optimization_report,
)
from merlin.targetgen import host_optimization_admission as admission
from merlin.targetgen import target_registry
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract.schemas import ContractViolation, load_schema, validate

_LEGACY = {
    "schema": "merlin.host_optimization_request.v1",
    "source_sha256": hashlib.sha256(_LOGICAL.encode()).hexdigest(),
    "propagate_layout": True,
    "fusion_groups": [],
    "hoist_static_allocs": None,
    "sink_deallocs": None,
    "static_arena": False,
    "minimum_transpose_bytes_saved": 1,
    "minimum_dispatches_saved": 1,
    "maximum_added_operations": 8192,
}
_NEW = {"physical_layout", "maximum_layout_assignments", "maximum_layout_permutations"}


def _v2(source=_LOGICAL, **choices):
    return _selection(
        source,
        choices={
            "schema": REQUEST_SCHEMA_V2,
            "propagate_layout": True,
            "physical_layout": True,
            "maximum_layout_assignments": 64,
            "maximum_layout_permutations": 2,
            **choices,
        },
    )


def _buffer(record):
    return {
        "abi_version": "0.1",
        "target": _TARGET,
        "commands": [],
        "tensors": {"Y": {"shape": [2, 3], "dtype": "i32", "role": "output"}},
        "host_optimization_request": record,
    }


def _execute(source, matrix, bias):
    interpreter = Interpreter(parse_mlir_text(source))
    for implementation in (ArithFunctions(), FuncFunctions(), LinalgFunctions(), TensorFunctions()):
        interpreter.register_implementations(implementation)
    values = tuple(
        ShapedArray(TypedPtr.new(value.reshape(-1).tolist(), xtype=i32), list(value.shape)) for value in (matrix, bias)
    )
    return tuple(
        np.asarray(value.data, dtype=np.int32).reshape(value.shape) for value in interpreter.call_op("kernel", values)
    )


def test_v1_preserves_exact_serialized_fields_content_identity_and_legacy_algorithm():
    request = request_from_record(_LEGACY)
    assert request.schema == REQUEST_SCHEMA
    assert request == HostOptimizationRequest(_LEGACY["source_sha256"], propagate_layout=True)
    assert request.record() == _LEGACY
    assert json.dumps(request.record()) == json.dumps(_LEGACY)
    assert not _NEW & set(request.record())
    selection = replace(_selection(_LOGICAL), request=request)
    identity = hashlib.sha256(
        json.dumps(
            {"context": asdict(selection.context), "request": _LEGACY},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert selection.identity() == identity
    graph = prepare_host_optimization(_LOGICAL, selection).report()["graph"]
    assert graph["contract"]["physical_layout"] is False
    assert [row["name"] for row in graph["layout"]["passes"]] == [
        "layout-source",
        "layout-propagation",
        "layout-conformance",
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_v2_exact_record_roundtrip_preserves_explicit_version_including_disabled_planner(enabled):
    request = _v2(physical_layout=enabled).request
    record = request.record()
    assert set(record) == set(_LEGACY) | _NEW
    assert record["schema"] == REQUEST_SCHEMA_V2
    assert request_from_record(record) == request
    assert request_from_record(record).record() == record
    validate(_buffer(record), "command_buffer")
    graph = prepare_host_optimization(_LOGICAL, _v2(physical_layout=enabled)).report()["graph"]
    assert graph["contract"]["physical_layout"] is enabled
    assert len(graph["layout"]["passes"]) == (4 if enabled else 3)


@pytest.mark.parametrize(
    "name,value",
    [
        ("physical_layout", False),
        ("maximum_layout_assignments", 100_000),
        ("maximum_layout_permutations", 16),
    ],
)
def test_v1_rejects_new_serialized_fields_even_at_their_legacy_defaults(name, value):
    record = {**_LEGACY, name: value}
    with pytest.raises(ValueError, match="unknown or missing"):
        request_from_record(record)
    with pytest.raises(ContractViolation):
        validate(_buffer(record), "command_buffer")


@pytest.mark.parametrize(
    "choices",
    [
        {"physical_layout": True},
        {"maximum_layout_assignments": 64},
        {"maximum_layout_permutations": 2},
    ],
)
def test_v1_constructor_cannot_silently_drop_new_nondefault_choices(choices):
    with pytest.raises(ValueError, match="v1.*v2"):
        HostOptimizationRequest(_LEGACY["source_sha256"], propagate_layout=True, **choices)


def test_out_of_band_v1_choice_mutation_cannot_hide_behind_legacy_serialization():
    selection = _selection(_LOGICAL, choices={"propagate_layout": True})
    object.__setattr__(selection.request, "physical_layout", True)
    with pytest.raises(ValueError, match="v1.*v2"):
        prepare_host_optimization(_LOGICAL, selection)


@pytest.mark.parametrize("missing", sorted(set(_LEGACY) | _NEW))
def test_v2_requires_every_declared_field(missing):
    record = _v2().request.record()
    del record[missing]
    with pytest.raises(ValueError, match="unknown"):
        request_from_record(record)
    with pytest.raises(ContractViolation):
        validate(_buffer(record), "command_buffer")


@pytest.mark.parametrize("schema", [None, False, [], {}, "merlin.host_optimization_request.v3"])
def test_unknown_or_nonstring_schema_is_not_an_implicit_upgrade(schema):
    record = {**_v2().request.record(), "schema": schema}
    with pytest.raises(ValueError, match="unknown schema"):
        request_from_record(record)
    with pytest.raises(ContractViolation):
        validate(_buffer(record), "command_buffer")


@pytest.mark.parametrize("extra", ["index_bits", "sequential_execution", "pipeline", "rtl_facts", "qualification"])
def test_v2_cannot_supply_original_permissions_or_unreviewed_controls(extra):
    record = {**_v2().request.record(), extra: True}
    with pytest.raises(ValueError, match="unknown"):
        request_from_record(record)
    with pytest.raises(ContractViolation):
        validate(_buffer(record), "command_buffer")


@pytest.mark.parametrize("field", ["maximum_layout_assignments", "maximum_layout_permutations"])
@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, None, 1_048_577])
def test_v2_search_bounds_are_exact_positive_bounded_integers(field, bad):
    record = {**_v2().request.record(), field: bad}
    with pytest.raises(ValueError, match="bounded positive"):
        request_from_record(record)
    with pytest.raises(ContractViolation):
        validate(_buffer(record), "command_buffer")


@pytest.mark.parametrize("bound", [1, 1_048_576])
def test_v2_bound_endpoints_are_admitted_without_claiming_search_success(bound):
    request = _v2(maximum_layout_assignments=bound, maximum_layout_permutations=bound).request
    assert request_from_record(request.record()) == request
    validate(_buffer(request.record()), "command_buffer")


def test_v2_physical_choice_needs_exact_bool_and_the_admitted_layout_choice():
    with pytest.raises(ValueError, match="exact booleans"):
        _v2(physical_layout=1)
    record = {**_v2().request.record(), "propagate_layout": False}
    with pytest.raises(ValueError, match="requires propagate_layout"):
        request_from_record(record)
    with pytest.raises(ContractViolation):
        validate(_buffer(record), "command_buffer")


@pytest.mark.parametrize(
    "context,message",
    [
        ({"permitted_optimizations": ()}, "admitted optimization"),
        ({"permitted_stages": ("llvm",)}, "admitted stage"),
        ({"independent_element_order_unobserved": False}, "independent-element order"),
    ],
)
def test_v2_reuses_original_permissions_without_obtaining_them_from_choice(context, message):
    selection = _v2()
    selection = replace(selection, context=replace(selection.context, **context))
    with pytest.raises(ValueError, match=message):
        prepare_host_optimization(_LOGICAL, selection)


def test_v2_preserves_original_float_and_current_source_requirements():
    floating = _LOGICAL.replace("i32", "f32").replace("arith.addi", "arith.addf")
    with pytest.raises(ValueError, match="floating environment"):
        prepare_host_optimization(floating, _v2(floating))
    with pytest.raises(ValueError, match="actual compiler source"):
        prepare_host_optimization(_LOGICAL + "\n", _v2())


def test_v2_forwarded_choices_execute_actual_selected_layout_and_match_independent_reference():
    selection = _v2()
    prepared = prepare_host_optimization(_LOGICAL, selection)
    report = prepared.report()
    graph = report["graph"]
    assert report["selection"]["request"] == selection.request.record()
    assert graph["contract"]["physical_layout"] is True
    assert graph["contract"]["maximum_layout_assignments"] == 64
    assert graph["contract"]["maximum_layout_permutations"] == 2
    assert [row["name"] for row in graph["layout"]["passes"]] == [
        "layout-source",
        "layout-planning",
        "layout-propagation",
        "layout-conformance",
    ]
    assert graph["layout"]["pointwise_work"]["physical_layout"]["new_conversions"] == 0
    assert graph["layout"]["after_transpose_bytes"] == 0
    data = np.array([-(2**31), 2**31 - 1, -1, 0, 1, 7], dtype=np.int32).reshape(2, 3)
    bias = np.array([1, -1, 2**31 - 1], dtype=np.int32)
    reference = np.add(data, bias, dtype=np.int32)
    for source in (_LOGICAL, prepared.selected_mlir):
        (actual,) = _execute(source, data, bias)
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize(
    "field,value",
    [
        ("physical_layout", False),
        ("maximum_layout_assignments", 32),
        ("maximum_layout_permutations", 4),
    ],
)
def test_changed_valid_v2_choice_cannot_inherit_a_prepared_report(field, value):
    selection = _v2()
    report = prepare_host_optimization(_LOGICAL, selection).report()
    changed = replace(selection, request=replace(selection.request, **{field: value}))
    with pytest.raises(ValueError, match="selection changed"):
        write_host_optimization_report(
            report,
            selection=changed,
            source_path=None,
            selected_source_path=None,
            llvm_path=None,
            object_path=None,
            workdir=None,
        )


def test_v2_choice_drift_during_rewrite_cannot_relabel_the_realized_plan(monkeypatch):
    from merlin.llvmlower import graph_optimizations

    selection = _v2()
    original = graph_optimizations.optimize_host_graph

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        object.__setattr__(selection.request, "maximum_layout_assignments", 32)
        return result

    monkeypatch.setattr(graph_optimizations, "optimize_host_graph", changed)
    with pytest.raises(ValueError, match="selection changed while applying"):
        prepare_host_optimization(_LOGICAL, selection)


def test_original_broker_admits_v2_and_refuses_midbuild_search_choice_mutation(tmp_path, monkeypatch):
    policy = _policy(
        permitted_stages=["logical_host"],
        permitted_optimizations=["propagate_layout"],
        independent_element_order_unobserved=True,
        data_layout="e-p:64:64-i64:64-n32:64-S128",
    )
    owner = _original(tmp_path, policy=policy, operator_contract=True)
    monkeypatch.setattr(
        target_registry,
        "resolve",
        lambda target: SimpleNamespace(
            capability_contract_path=owner.contract,
            load_contract=lambda: {"name": _TARGET, "harness_abi": {"entry_symbol": "kernel"}},
        ),
    )
    buffer = _buffer(_v2().request.record())
    validate(buffer, "command_buffer")
    with _scope(owner):
        with admission.selected_buffer_choices(buffer, _LOGICAL, target=_TARGET) as selected:
            assert selected.request.schema == REQUEST_SCHEMA_V2
            assert selected.request.physical_layout
        with pytest.raises(ValueError, match="changed during"):
            with admission.selected_buffer_choices(buffer, _LOGICAL, target=_TARGET):
                buffer["host_optimization_request"]["maximum_layout_assignments"] = 32
    assert admission.active_selection() is None


def test_closed_v1_json_schema_branch_is_unchanged():
    branch = load_schema("command_buffer")["properties"]["host_optimization_request"]["oneOf"][0]
    assert branch["properties"]["schema"]["const"] == REQUEST_SCHEMA
    assert branch["additionalProperties"] is False
    assert set(branch["properties"]) == set(branch["required"]) == set(_LEGACY)
    validate(_buffer(_LEGACY), "command_buffer")


def test_original_v2_broker_runs_actual_logical_lowering_and_native_object(tmp_path, monkeypatch, record_property):
    from merlin.common import invocation_record
    from merlin.runtime.backends import base as backends

    clang = toolchain.clang()
    if not clang.is_file() or not toolchain.mlir_translate().is_file():
        pytest.skip("requires the original selected compiler and MLIR translator")
    policy = _policy(
        permitted_stages=["logical_host"],
        permitted_optimizations=["propagate_layout"],
        independent_element_order_unobserved=True,
        data_layout="e-p:64:64-i64:64-n32:64-S128",
    )
    owner = _original(tmp_path, policy=policy, operator_contract=True, compiler=clang)
    monkeypatch.setattr(
        target_registry,
        "resolve",
        lambda target: SimpleNamespace(
            capability_contract_path=owner.contract,
            load_contract=lambda: {"name": _TARGET, "harness_abi": {"entry_symbol": "kernel"}},
        ),
    )
    # Select only the original neutral recipe; compilation/translation and the
    # ordinary stack preflight remain actual native tool invocations.
    monkeypatch.setattr(backends, "harness_build_recipe", lambda target: owner.service.recipe)
    buffer = _buffer(_v2().request.record())
    validate(buffer, "command_buffer")
    work = tmp_path / "native_v2"
    with _scope(owner), admission.selected_buffer_choices(buffer, _LOGICAL, target=_TARGET):
        obj = compiler.llvm_mlir_to_object(_LOGICAL, work, target=_TARGET)
    assert obj.read_bytes().startswith(b"\x7fELF")
    report_path = work / "kernel.host_optimization.json"
    report = json.loads(report_path.read_text())
    assert report["selection"]["request"] == buffer["host_optimization_request"]
    assert report["graph"]["contract"]["physical_layout"] is True
    assert report["graph"]["layout"]["pointwise_work"]["physical_layout"]["new_conversions"] == 0
    assert report["observed_lowering"]["index_bits"] == 64
    assert report["products"]["object"]["sha256"] == hashlib.sha256(obj.read_bytes()).hexdigest()
    selected = work / "kernel.host.mlir"
    assert report["products"]["selected_ir"]["sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    calls = [invocation_record.verify(path) for path in work.rglob("invocation.json")]
    assert {"llvm_translation", "object"} <= {call["stage"] for call in calls}
    assert json.loads((work / "kernel.stack_frame.json").read_text())["status"] == "passed"
    matrix = np.arange(6, dtype=np.int32).reshape(2, 3)
    bias = np.array([1, -7, 11], dtype=np.int32)
    for source in (_LOGICAL, selected.read_text()):
        (actual,) = _execute(source, matrix, bias)
        np.testing.assert_array_equal(actual, np.add(matrix, bias, dtype=np.int32))
    record_property(
        "native_v2_request",
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "object": report["products"]["object"],
                "selected_ir": report["products"]["selected_ir"],
                "scope": (
                    "actual original V2 broker, logical lowering and native object; "
                    "independently executed selected IR; object execution and hardware support not established"
                ),
            }
        ),
    )
