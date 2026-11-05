"""Original ABI premises and current native wrapper checks are separate gates."""

import hashlib
import json
from dataclasses import asdict, replace

import pytest
from test_owned_buffer_loop_check import COPY, loop_source

from merlin.llvmlower import toolchain
from merlin.llvmlower.borrowed_buffer_calls import (
    SCHEMA,
    BorrowedBufferCall,
    BorrowedBufferCalls,
    BorrowedBufferEffects,
    read_borrowed_calls,
)
from merlin.llvmlower.pipeline import dealloc_placement_violations

EFFECTS = BorrowedBufferEffects("reviewed-neutral-test-ABI-v1", True, True, True)
CALL = BorrowedBufferCall("writer", "borrowed", ("memref<4xi8>", "memref<4xi8>"), 1, EFFECTS)
SUPPORT = """
 func.func private @borrowed(memref<4xi8>,memref<4xi8>) attributes {llvm.emit_c_interface}
 func.func private @writer(%arg0:memref<4xi8>,%unused:memref<4xi8>,%out:memref<4xi8>) {
 func.call @borrowed(%arg0,%out):(memref<4xi8>,memref<4xi8>)->()
 return
 }
"""


def source():
    return loop_source().replace("module {", "module {" + SUPPORT).replace(
        COPY, "func.call @writer(%previous,%fresh,%fresh):(memref<4xi8>,memref<4xi8>,memref<4xi8>)->()"
    )


def selection(tmp_path, text):
    path = tmp_path / "original.mlir"
    path.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return BorrowedBufferCalls(path, digest, digest, (CALL,), (), lambda: None)


@pytest.mark.parametrize("field", ["no_retained_aliases", "no_releases", "complete_before_return"])
@pytest.mark.parametrize("value", [False, None, 1])
def test_each_implementation_effect_is_an_explicit_true_premise(field, value):
    with pytest.raises(ValueError, match="synchronous effects"):
        replace(EFFECTS, **{field: value})


def test_effect_identity_is_not_inferred_from_an_instruction_or_numerical_witness():
    with pytest.raises(ValueError):
        replace(EFFECTS, qualification_identity="")


def test_original_data_round_trip_and_ambiguity_refusal():
    record = dict(schema=SCHEMA, calls=[asdict(CALL)])
    record = json.loads(json.dumps(record))
    assert read_borrowed_calls(record) == (CALL,)
    with pytest.raises(ValueError, match="distinct"):
        read_borrowed_calls(dict(schema=SCHEMA, calls=record["calls"] * 2))
    with pytest.raises(ValueError):
        read_borrowed_calls(dict(schema="unknown", calls=record["calls"]))


def test_source_and_derived_epoch_are_checked_against_original_pins(tmp_path):
    original = source()
    selected = selection(tmp_path, original)
    assert json.loads(json.dumps(selected.record())) == selected.record()
    selected.revalidate(original)
    after = original + "\n// trusted preprocessing product\n"
    current = selected.after_preprocessing(original, after)
    current.revalidate(after)
    with pytest.raises(ValueError, match="stale"):
        current.revalidate(original)
    selected.source_path.write_text(after)
    with pytest.raises(ValueError, match="changed"):
        current.revalidate(after)


def test_transport_without_checked_boundary_refuses_original_effects(tmp_path, monkeypatch):
    from merlin.llvmlower import pipeline, stock_cli

    monkeypatch.setenv("MERLIN_COMPILER_DRIVER", "stock-cli")
    monkeypatch.setattr(stock_cli, "llvm_install", lambda: pytest.fail("native tools reached"))
    original = source()
    selected = selection(tmp_path, original)
    with pytest.raises(pipeline.PipelineError, match="Python-runner options: borrowed_buffer_calls"):
        pipeline.lower_to_llvm_ir(original, workdir=tmp_path, borrowed_buffer_calls=selected)


native = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")


@native
def test_only_original_effects_plus_exact_current_wrapper_admit_the_call():
    text = source()
    assert dealloc_placement_violations(text)
    assert dealloc_placement_violations(text, borrowed_calls=(CALL,)) == []


@native
@pytest.mark.parametrize(
    "before,after",
    [
        ("@borrowed(%arg0,%out)", "@borrowed(%out,%arg0)"),
        ("@borrowed(%arg0,%out)", "@borrowed(%arg0,%unused)"),
        ("return\n }", "memref.dealloc %out:memref<4xi8>\nreturn\n }"),
        ("return\n }", "memref.copy %out,%unused:memref<4xi8> to memref<4xi8>\nreturn\n }"),
    ],
)
def test_changed_forwarding_body_is_not_granted_effects_by_its_name(before, after):
    assert before in source()
    bad = dealloc_placement_violations(source().replace(before, after), borrowed_calls=(CALL,))
    assert any("forwarding body" in message for message in bad)


@native
def test_unselected_callee_with_read_annotations_stays_a_barrier():
    text = source().replace("module {", 'module {func.func private @unknown(memref<4xi8> {bufferization.access="read"},'
                            'memref<4xi8>,memref<4xi8>)')
    text = text.replace("@writer(%previous", "@unknown(%previous")
    bad = dealloc_placement_violations(text, borrowed_calls=(CALL,))
    assert any("unknown" in message for message in bad)


@native
def test_current_borrowed_definition_cannot_replace_original_external_abi():
    text = source().replace(
        "func.func private @borrowed(memref<4xi8>,memref<4xi8>) attributes {llvm.emit_c_interface}",
        "func.func private @borrowed(%x:memref<4xi8>,%y:memref<4xi8>) attributes {llvm.emit_c_interface} {return}",
    )
    bad = dealloc_placement_violations(text, borrowed_calls=(CALL,))
    assert any("declaration/ABI" in message for message in bad)


@native
def test_stale_positional_abi_cannot_be_replaced_by_matching_arity():
    wrong = replace(CALL, argument_types=("memref<2x2xi8>", "memref<2x2xi8>"))
    bad = dealloc_placement_violations(source(), borrowed_calls=(wrong,))
    assert any("declaration/ABI" in message for message in bad)


@native
def test_nested_symbol_table_cannot_shadow_an_original_qualified_wrapper():
    nested = loop_source().replace(
        COPY, "func.call @writer(%previous,%fresh,%fresh):(memref<4xi8>,memref<4xi8>,memref<4xi8>)->()"
    )
    inner = """func.func private @writer(%x:memref<4xi8>,%unused:memref<4xi8>,%out:memref<4xi8>) {
      memref.copy %x,%out:memref<4xi8> to memref<4xi8>
      memref.dealloc %x:memref<4xi8>
      return
    }"""
    text = "module {" + SUPPORT + nested.replace("module {", "module @nested {" + inner) + "}"
    bad = dealloc_placement_violations(text, borrowed_calls=(CALL,))
    assert any("call" in message for message in bad)
