"""Memory NaN custody controls: host parsing/projection only, no SDK/native.

NaN metadata is refused until raw-bit capture is exact. Runtime scalar values
are not examined; this does not approve numerical or parser semantics.
"""

import json
import struct
from pathlib import Path

import pytest
from buffer_rank_one_inputs import chain
from test_compiler_memory_origin import active, fake_native_capture, scope
from xdsl.dialects import builtin

from merlin.common.paths import module_source_path
from merlin.llvmlower import leading_cse
from merlin.transforms.certificates import ir_text, parse_ir
from merlin.verify import buffer_reuse_intake as buffer_gate
from merlin.verify import compiler_memory_dce as memory_dce
from merlin.verify import compiler_memory_origin as origin
from merlin.verify import memory_scalar_nan_admission as admission
from merlin.verify.buffer_reuse_bridge import parse
from merlin.verify.compiler_transforms import memory_lowering_pipeline
from merlin.verify.gate_errors import GateUnavailable
from merlin.verify.lean_graph_input_admission import InputAdmissionError
from merlin.verify.memory_scalar_nan_admission import admit_memory_scalar_attributes
from merlin.verify.speculative_gate import GateFailure


def pair(value, *, operation=False):
    metadata = "" if operation else f" attributes {{tag = {value}}}"
    answer = (
        f'%answer = "arith.constant"() <{{value = 0 : i32}}> {{tag = {value}}} : () -> i32'
        if operation
        else "%answer = arith.constant 0 : i32"
    )
    source = (
        "builtin.module { func.func @f() -> i32"
        + metadata
        + " { %dead = arith.constant 7 : index "
        + answer
        + " func.return %answer : i32 } }"
    )
    return source, source.replace("%dead = arith.constant 7 : index ", "")


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("operation", [False, True])
@pytest.mark.parametrize(
    "value,code",
    [
        ("0x7F800001 : f32", "scalar-nan"),
        ("0x7FC00001 : f32", "scalar-nan"),
        ('dense<"0x0100807F0100C07F"> : tensor<2xf32>', "dense-nan"),
        ("dense<0x7F800001> : tensor<1xf32>", "dense-nan"),
        ("dense<[0x7F800001, 0x7FC00001]> : tensor<2xf32>", "dense-nan"),
        ('dense<"0x0100807F00000000"> : tensor<1xcomplex<f32>>', "complex-floating-dense"),
    ],
)
def test_actual_memory_dce_source_or_target_nan_refuses_before_any_proof(side, operation, value, code, monkeypatch):
    before, after = pair("0.0 : f32", operation=operation)
    poisoned = pair(value, operation=operation)
    buffers = [before, after]
    buffers[side] = poisoned[side]
    monkeypatch.setattr(memory_dce.runner, "prove", lambda *a, **k: pytest.fail("NaN input reached SDK"))
    with pytest.raises(InputAdmissionError, match=code):
        memory_dce.prepare(*buffers)


@pytest.mark.parametrize("endpoint", [0, 1, 2, 3])
@pytest.mark.parametrize("dense", [False, True])
def test_actual_buffer_chain_nan_refuses_at_every_real_boundary(endpoint, dense):
    buffers = list(chain(3, tiled=True))
    module = parse_ir(buffers[endpoint])
    value = (
        builtin.DenseIntOrFPElementsAttr(builtin.TensorType(builtin.f32, [1]), builtin.BytesAttr(b"\x01\x00\x80\x7f"))
        if dense
        else builtin.FloatAttr(float("nan"), builtin.f32)
    )
    # Function metadata is retained by the actual producer and complete frame.
    module.body.block.first_op.attributes["tag"] = value
    buffers[endpoint] = ir_text(module)
    with pytest.raises(InputAdmissionError, match="dense-nan" if dense else "scalar-nan"):
        buffer_gate.prepare_chain(tuple(buffers), allocator=buffer_gate.ALLOCATOR)


@pytest.mark.parametrize(
    "raw",
    [
        "builtin.module attributes {tag = [{nested = 0x7FC00001 : f32}]} {}",
        'builtin.module { "test.op"() <{payload = [{nested = 0x7FC00001 : f32}]}> : () -> () }',
        "builtin.module { func.func @f(%x: memref<1xi32, 0x7FC00001 : f32>) { func.return } }",
        'builtin.module { func.func @f() { %x = "test.result"() : () -> memref<1xi32, 0x7FC00001 : f32> '
        '"test.sink"(%x) : (memref<1xi32, 0x7FC00001 : f32>) -> () func.return } }',
    ],
)
@pytest.mark.parametrize("admit", [parse, origin._canonical_source])
def test_nested_metadata_properties_operand_result_and_block_argument_types_refuse(raw, admit):
    with pytest.raises(InputAdmissionError, match="scalar-nan"):
        admit(raw)


def test_fused_location_metadata_is_walked_without_dropping_fields():
    module = parse_ir("builtin.module {}")
    module.location = builtin.FusedLoc(
        builtin.ArrayAttr([builtin.UnknownLoc()]),
        builtin.DictionaryAttr({"nested": builtin.ArrayAttr([builtin.FloatAttr(float("nan"), builtin.f64)])}),
    )
    with pytest.raises(InputAdmissionError, match="scalar-nan.*location"):
        admit_memory_scalar_attributes(module)


@pytest.mark.parametrize(
    "value",
    [
        "0x7F800000 : f32",
        "0xFF800000 : f32",
        "0x00000000 : f32",
        "0x80000000 : f32",
        "1.25 : f32",
        '"nan"',
        'dense<"0x0000807F000080FF00000000000000800000A03F"> : tensor<5xf32>',
        "dense<[0x7F800000, 0xFF800000, 0x00000000, 0x80000000, 0x3FA00000]> : tensor<5xf32>",
    ],
)
def test_infinities_signed_zeros_finite_dense_values_and_nan_string_remain_admitted(value):
    before, after = pair(value)
    candidate = memory_dce.prepare(before, after)
    memory_dce.ready(candidate)
    assert candidate.proposed_sites == 1
    canonical = origin._canonical_source(before)
    admit_memory_scalar_attributes(parse_ir(canonical))
    if value.startswith("dense<"):
        original = parse_ir(before).body.block.first_op.attributes["tag"]
        reopened = parse_ir(canonical).body.block.first_op.attributes["tag"]
        assert (
            original.data.data
            == reopened.data.data
            == struct.pack("<fffff", float("inf"), -float("inf"), 0.0, -0.0, 1.25)
        )


@pytest.mark.parametrize(
    "first,second",
    [
        ("0x7F800000 : f32", "0xFF800000 : f32"),
        ("0x00000000 : f32", "0x80000000 : f32"),
        ('dense<"0x0000807F"> : tensor<1xf32>', 'dense<"0x000080FF"> : tensor<1xf32>'),
        ('dense<"0x00000000"> : tensor<1xf32>', 'dense<"0x00000080"> : tensor<1xf32>'),
    ],
)
def test_infinity_and_zero_sign_changes_remain_distinct_in_actual_frame_and_key(first, second):
    before, after = pair(first)
    unchanged = memory_dce.prepare(before, after)
    mutated = memory_dce.prepare(before, after.replace(first, second))
    assert unchanged.query != mutated.query and unchanged.projection_json != mutated.projection_json
    assert origin._canonical_source(before) != origin._canonical_source(before.replace(first, second))


def test_dense_nan_inspection_uses_raw_stored_payload_without_shape_expansion(monkeypatch):
    attribute = builtin.DenseIntOrFPElementsAttr(
        builtin.TensorType(builtin.f32, [10**12]), builtin.BytesAttr(b"\x01\x00\x80\x7f")
    )
    module = parse_ir("builtin.module {}")
    module.attributes["raw_stored_payload"] = attribute
    monkeypatch.setattr(
        builtin.DenseIntOrFPElementsAttr, "get_values", lambda *a, **k: pytest.fail("expanded shaped payload")
    )
    with pytest.raises(InputAdmissionError, match="dense-nan"):
        admit_memory_scalar_attributes(module)


def test_every_memory_source_roster_binds_helper_and_reused_attribute_walker():
    required = {
        module_source_path("merlin.verify." + name).resolve()
        for name in ("memory_scalar_nan_admission", "lean_graph_input_admission")
    }
    required.update(
        module_source_path(name).resolve()
        for name in (
            "merlin.common.dense_float_ir",
            "xdsl.parser",
            "xdsl.parser.attribute_parser",
            "xdsl.utils.mlir_lexer",
            "merlin.verify.gate_errors",
        )
    )
    buffer = buffer_gate.prepare_chain(tuple(chain(3, tiled=False)), allocator=buffer_gate.ALLOCATOR)
    dce = memory_dce.prepare(*pair("0.0 : f32"))
    for rows in (json.loads(buffer.sources_json), json.loads(dce.sources_json)):
        assert required <= {Path(row["path"]).resolve() for row in rows}
    assert {str(path) for path in required} <= {row[0] for row in origin._sources()}
    names = {function.__name__ for function, _ in origin._functions(canonical=True)}
    assert {"admit_memory_scalar_attributes", "_attributes", "_fields", "_canonical_source"} <= names
    assert {"parse_memory_module", "_parse_bounded"} <= names


@pytest.mark.parametrize("where", ["source", "upstream", "CSE-before-and-after"])
def test_original_memory_scope_cannot_lend_nan_metadata_custody(tmp_path, monkeypatch, where):
    poisoned = pair("0x7F800001 : f32")[0]
    if where == "CSE-before-and-after":
        # The ordinary CSE observation follows unverified canonicalization.
        # Raw identity grants no mathematical/model projection credit.
        with scope(tmp_path, passes=frozenset({"cse"})) as (owner, validation):
            fake_native_capture(tmp_path, owner, monkeypatch, canonicalized=poisoned)
            assert leading_cse.submit(tmp_path) == 0
        report = json.loads((validation.gate.directory / "coverage.json").read_text())
        assert report["verified_boundaries"] == 0
        assert report["unchanged_boundaries"] == 1
        assert report["boundaries"][0]["proof_credit"] is False
        assert not validation.gate.verdicts
        return
    with pytest.raises((InputAdmissionError, GateFailure), match="scalar-nan"):
        with active(tmp_path) as validation:
            with origin.compiler_scope(
                tmp_path,
                original_source=poisoned if where == "source" else pair("0.0 : f32")[0],
                memory_pipeline=memory_lowering_pipeline(),
                validation=validation,
            ) as owner:
                if where == "upstream":
                    owner.bind_upstream(poisoned)


@pytest.mark.parametrize("decoder", [parse, origin._canonical_source, admission.parse_loop_module])
@pytest.mark.parametrize("literal", ['dense<"0x00000000">', "dense<0.0>"])
def test_dense_literal_budget_refuses_before_parent_allocation(decoder, literal, monkeypatch):
    # Small sentinel: 3xf32 would require12 bytes; the test selects8. Never
    # invoke a huge raw splat or construct an unbounded tensor payload.
    monkeypatch.setattr(admission, "MAX_DENSE_FLOAT_BYTES", 8)
    monkeypatch.setattr(admission, "MAX_LOOP_DENSE_BYTES", 8)
    monkeypatch.setattr(
        builtin.DenseIntOrFPElementsAttr,
        "__init__",
        lambda *a, **k: pytest.fail("parent dense allocation reached after budget should refuse"),
    )
    raw = "builtin.module attributes {tag = " + literal + " : tensor<3xf32>} {}"
    with pytest.raises(GateUnavailable, match="decoded-storage resource budget"):
        decoder(raw)


@pytest.mark.parametrize("decoder", [parse, origin._canonical_source, admission.parse_loop_module])
def test_dense_budget_is_aggregate_per_module_and_does_not_cap_symbolic_type_signatures(decoder, monkeypatch):
    monkeypatch.setattr(admission, "MAX_DENSE_FLOAT_BYTES", 8)
    monkeypatch.setattr(admission, "MAX_LOOP_DENSE_BYTES", 8)
    one = 'dense<"0x00000000"> : tensor<1xf32>'
    with pytest.raises(GateUnavailable, match="decoded-storage resource budget"):
        decoder(f"builtin.module attributes {{a = {one}, b = {one}, c = {one}}} {{}}")
    # This is only a symbolic type, not a dense literal or allocation.
    decoder("builtin.module { func.func @f(%x: tensor<1000000xf32>) { func.return } }")


def test_dense_integer_nan_looking_bits_remain_integers():
    raw = 'dense<"0x0100807F"> : tensor<1xi32>'
    before, after = pair(raw)
    candidate = memory_dce.prepare(before, after)
    memory_dce.ready(candidate)
    attribute = parse(before).body.block.first_op.attributes["tag"]
    assert tuple(attribute.iter_values()) == (0x7F800001,)


def test_constructed_dense_float_array_nan_cannot_bypass_memory_attribute_admission():
    module = parse_ir("builtin.module {}")
    module.attributes["array"] = builtin.DenseArrayBase.from_list(builtin.f32, [float("nan")])
    with pytest.raises(InputAdmissionError, match="dense-nan"):
        admit_memory_scalar_attributes(module)
