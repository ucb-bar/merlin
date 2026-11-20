"""Pinned source atomic fields are reconstructed before phase normalization."""

from copy import deepcopy

import pytest
from xdsl.parser import Parser

from merlin.frontends.kernel_atomic_evidence import checked_source_atomic
from merlin.frontends.linalg_mlir import make_context

KINDS = [(1, "and", 2), (2, "or", 14), (3, "xor", 15), (4, "add", 1),
         (6, "max", 6), (7, "min", 10), (8, "umax", 7), (9, "umin", 11), (10, "exch", 3)]


def source(dtype="i64", kind=4, order=4, scope=1, *, cas=False):
    operation = "tt.atomic_cas" if cas else "tt.atomic_rmw"
    fields = "" if cas else f"atomic_rmw_op={kind}:i32,"
    module = Parser(make_context(), f'''module {{func.func @source(%p: !tt.ptr<{dtype}>, %x: {dtype}) {{
        %old = "{operation}"(%p, %x{', %x' if cas else ''})
            <{{{fields}sem={order}:i32,scope={scope}:i32}}>
            : (!tt.ptr<{dtype}>, {dtype}{', ' + dtype if cas else ''})->{dtype}
        func.return
    }} }}''').parse_module()
    return next(op for op in module.walk() if getattr(getattr(op, "op_name", None), "data", "") == operation)


def contract(operation="add", standard=1, dtype="i64", order="acq_rel", scope="gpu", numeric=None):
    return {"operation": operation, "standard_kind": standard,
        "word_bits": 16 if dtype == "bf16" else int(dtype[1:]),
        **({"value_dtype": dtype, "numeric_contract": numeric} if numeric is not None else {}),
        "source_order": order, "source_scope": scope,
        "logical_scope": "one-logical-program" if scope == "cta" else "all-logical-programs-in-private-invocation",
        "standard_order": "acq_rel", "external_participants": "excluded-by-private-buffer-ABI"}


INTEGER = {"profile": "private-invocation-integer-atomics-v1", "f32_atomic_add": None}
TYPED = {"profile": "private-invocation-typed-atomics-v2", "f32_atomic_add": None}


@pytest.mark.parametrize("kind,operation,standard", KINDS)
@pytest.mark.parametrize("dtype", ["i32", "i64"])
@pytest.mark.parametrize("order,order_name", [(1, "relaxed"), (2, "acquire"), (3, "release"), (4, "acq_rel")])
@pytest.mark.parametrize("scope,scope_name", [(1, "gpu"), (2, "cta"), (3, "sys")])
def test_actual_integer_kind_order_scope_and_width(
    kind, operation, standard, dtype, order, order_name, scope, scope_name
):
    checked_source_atomic(source(dtype, kind, order, scope),
        contract(operation, standard, dtype, order_name, scope_name), INTEGER)


@pytest.mark.parametrize("dtype", ["i16", "i32", "i64"])
def test_compare_exchange_has_its_original_old_value_type(dtype):
    checked_source_atomic(source(dtype, cas=True), contract("cas", "compare-exchange", dtype), INTEGER)


@pytest.mark.parametrize("dtype,kind,operation,numeric,policy", [
    ("f32", 10, "exch", "bitwise-exchange", None), ("f64", 10, "exch", "bitwise-exchange", None),
    ("f16", 5, "fadd", "ieee-f16-rne-add-quiet-nans", None),
    ("bf16", 5, "fadd", "ieee-bf16-rne-add-quiet-nans", None),
    ("f64", 5, "fadd", "ieee-f64-rne-add-quiet-nans", None),
    ("f32", 5, "fadd", "ieee-f32-rne-add", "ieee"),
    ("f32", 5, "fadd", "f32-rne-sign-preserving-ftz-add", "ftz"),
])
def test_actual_float_operations_require_their_exact_numeric_policy(dtype, kind, operation, numeric, policy):
    checked_source_atomic(source(dtype, kind), contract(operation, 3 if kind == 10 else 0, dtype, numeric=numeric),
                          {**TYPED, "f32_atomic_add": policy})


@pytest.mark.parametrize("field,value", [("source_order", "relaxed"), ("source_scope", "cta"),
    ("word_bits", 32), ("standard_kind", 0), ("operation", "fadd"), ("external_participants", "allowed")])
def test_changed_primitive_fields_cannot_replace_actual_source(field, value):
    declared = {**contract(), field: value}
    with pytest.raises(ValueError, match="actual typed source"):
        checked_source_atomic(source(), declared, INTEGER)


@pytest.mark.parametrize("op,selection", [
    (lambda: source("f32", 5), TYPED),
    (lambda: source("f32", 10), INTEGER),
    (lambda: source("i16"), INTEGER),
    (lambda: source("i64", 5), INTEGER),
    (lambda: source("f32", 4), TYPED),
    (lambda: source("f32", cas=True), TYPED),
    (lambda: source(order=7), INTEGER),
    (lambda: source(scope=7), INTEGER),
    (lambda: source(kind=15), INTEGER),
])
def test_unqualified_source_types_enums_and_float_policy_refuse(op, selection):
    with pytest.raises(ValueError, match="atomic"):
        checked_source_atomic(op(), deepcopy(contract()), selection)
