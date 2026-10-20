"""Genuine plural forwarding and generation checks; no target qualification."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_borrowed_buffer_calls import CALL
from test_owned_buffer_loop_check import MULTIPLE

from merlin.llvmlower import toolchain
from merlin.llvmlower.borrowed_buffer_calls import (
    RESULTS_SCHEMA,
    SCHEMA,
    BorrowedBufferCalls,
    BorrowedBufferEffects,
    BorrowedBufferResults,
    bind_native,
    read_borrowed_calls,
    validate_calls,
)
from merlin.llvmlower.pipeline import dealloc_placement_violations

EFFECTS = BorrowedBufferEffects("original-neutral-plural-lifetime-v1", True, True, True)
ROW = BorrowedBufferResults("plural", "borrowed_plural", ("memref<2x19xi8>", "memref<3x7xi16>") * 2, (3, 2), EFFECTS)
COPIES = (
    "memref.copy %previous_a,%fresh_a:memref<2x19xi8> to memref<2x19xi8>\n"
    " memref.copy %previous_b,%fresh_b:memref<3x7xi16> to memref<3x7xi16>"
)
CALLSITE = (
    "func.call @plural(%previous_a,%previous_b,%fresh_a,%fresh_b,%fresh_b,%fresh_a):"
    "(memref<2x19xi8>,memref<3x7xi16>,memref<2x19xi8>,memref<3x7xi16>,"
    "memref<3x7xi16>,memref<2x19xi8>)->()"
)
SUPPORT = """
 func.func private @borrowed_plural(memref<2x19xi8> {bufferization.access="read"},
   memref<3x7xi16> {bufferization.access="read"},
   memref<2x19xi8> {bufferization.access="write"},memref<3x7xi16> {bufferization.access="write"})
   attributes {llvm.emit_c_interface}
 func.func private @plural(%a:memref<2x19xi8>,%b:memref<3x7xi16>,
   %unused_a:memref<2x19xi8>,%unused_b:memref<3x7xi16>,
   %out_b:memref<3x7xi16>,%out_a:memref<2x19xi8>) {
   func.call @borrowed_plural(%a,%b,%out_a,%out_b):
    (memref<2x19xi8>,memref<3x7xi16>,memref<2x19xi8>,memref<3x7xi16>)->()
   return
 }
"""


def source():
    assert COPIES in MULTIPLE
    return MULTIPLE.replace("module {", "module {" + SUPPORT).replace(COPIES, CALLSITE)


def selection(tmp_path, value=ROW):
    path = tmp_path / "source.mlir"
    path.write_text(source())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return BorrowedBufferCalls(path, digest, digest, (value,), (), lambda: None)


def test_v1_record_and_exact_row_are_unchanged():
    record = json.loads(json.dumps(dict(schema=SCHEMA, calls=[asdict(CALL)])))
    assert read_borrowed_calls(record) == (CALL,)


def test_v2_round_trip_mixed_roster_without_duplicate_call_rows(tmp_path):
    chosen = selection(tmp_path)
    record = chosen.record()
    assert record["schema"] == RESULTS_SCHEMA
    assert record["calls"][0]["result_arguments"] == [3, 2]
    assert "result_argument" not in record["calls"][0]
    assert read_borrowed_calls(json.loads(json.dumps(record))) == (ROW,)
    mixed = json.loads(json.dumps(dict(schema=RESULTS_SCHEMA, calls=[asdict(CALL), asdict(ROW)])))
    assert read_borrowed_calls(mixed) == (CALL, ROW)
    with pytest.raises(ValueError, match="distinct"):
        validate_calls((ROW, replace(ROW, wrapper_symbol="other")))


def test_valid_none_observer_has_stable_identity_across_revalidation(tmp_path):
    chosen = selection(tmp_path)
    chosen.revalidate(source())
    chosen.revalidate()
    assert chosen.observe_native is None
    assert chosen.record()["schema"] == RESULTS_SCHEMA


@pytest.mark.parametrize("indices", [(), (2,), (2, 2), (True, 2), (-1, 2), (2, 4), [3, 2], tuple(range(129))])
def test_invalid_or_unbounded_result_positions_refuse(indices):
    with pytest.raises(ValueError, match="result"):
        replace(ROW, result_arguments=indices)


@pytest.mark.parametrize("schema", [SCHEMA, "unknown"])
def test_plural_record_cannot_be_relabelled_as_singleton(schema):
    record = json.loads(json.dumps(dict(schema=schema, calls=[asdict(ROW)])))
    with pytest.raises(ValueError):
        read_borrowed_calls(record)


@pytest.mark.parametrize("field", ["no_retained_aliases", "no_releases", "complete_before_return"])
def test_effect_mutation_refuses_before_native_binding(field):
    effects = replace(EFFECTS)
    row = replace(ROW, effects=effects)
    object.__setattr__(effects, field, False)
    with pytest.raises(ValueError, match="effects"):
        bind_native("_OBL_CALL_ROSTER = ()", (row,))


def test_serialized_roster_ceiling_is_checked_before_native_binding():
    # Distinct symbols prevent ambiguity from masking the byte-budget check.
    rows = tuple(
        replace(ROW, wrapper_symbol=f"w{i}", borrowed_symbol=f"b{i}", argument_types=("m" * 4096,) * 128)
        for i in range(3)
    )
    with pytest.raises(ValueError, match="byte budget"):
        bind_native("_OBL_CALL_ROSTER = ()", rows)


def test_original_rows_and_qualification_are_frozen_before_first_callback(tmp_path):
    effects = replace(EFFECTS)
    row = replace(ROW, effects=effects)
    chosen = selection(tmp_path, row)
    object.__setattr__(effects, "qualification_identity", "another-valid-original-identity")
    with pytest.raises(ValueError, match="selection.*changed"):
        chosen.revalidate()
    with pytest.raises(ValueError, match="selection.*changed"):
        chosen.record()


@pytest.mark.parametrize("field", ["validate_original", "observe_native"])
def test_equal_callable_replacement_does_not_acquire_original_owner_identity(tmp_path, field):
    reached = []

    class EqualCallable:
        def __init__(self, label):
            self.label = label

        def __eq__(self, other):
            return True

        def __call__(self, *args):
            reached.append(self.label)

    chosen = replace(selection(tmp_path), **{field: EqualCallable("original")})
    object.__setattr__(chosen, field, EqualCallable("replacement"))
    with pytest.raises(ValueError, match="selection.*changed"):
        chosen.revalidate()
    assert reached == []


def test_callable_object_implementation_is_frozen_too(tmp_path, monkeypatch):
    reached = []

    class Original:
        def __call__(self):
            reached.append("original")

    chosen = replace(selection(tmp_path), validate_original=Original())
    monkeypatch.setattr(Original, "__call__", lambda self: reached.append("replacement"))
    with pytest.raises(ValueError, match="selection.*changed"):
        chosen.revalidate()
    assert reached == []


@pytest.mark.parametrize("mutation", ["rows", "effects", "validator", "reader", "callback_reader", "class_method"])
def test_original_callback_cannot_change_a_later_row_or_selected_code(tmp_path, monkeypatch, mutation):
    from merlin.llvmlower import borrowed_buffer_calls as B

    path = tmp_path / "source.mlir"
    path.write_text(source())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    first = replace(ROW, effects=replace(EFFECTS))
    second = replace(ROW, wrapper_symbol="other", borrowed_symbol="other_borrowed", effects=replace(EFFECTS))

    def original():
        if mutation == "rows":
            object.__setattr__(chosen, "calls", (second, first))
        elif mutation == "effects":
            object.__setattr__(second.effects, "qualification_identity", "changed-later-row")
        elif mutation == "validator":
            object.__setattr__(chosen, "validate_original", lambda: None)
        elif mutation == "reader":
            monkeypatch.setattr(B, "_plural_state", lambda value: ())
        elif mutation == "callback_reader":
            monkeypatch.setattr(B, "_callback_binding", lambda value: ())
        else:
            monkeypatch.setattr(B.BorrowedBufferResults, "__post_init__", lambda value: None)

    chosen = BorrowedBufferCalls(path, digest, digest, (first, second), (), original)
    with pytest.raises(ValueError, match="changed"):
        chosen.revalidate()


def test_source_is_checked_before_original_callback_and_after_it(tmp_path):
    calls = []
    chosen = selection(tmp_path)
    object.__setattr__(chosen, "validate_original", lambda: calls.append(True))
    # Reissue with the intended original callback before damaging its source.
    chosen = replace(chosen)
    chosen.source_path.write_text(source() + "\n// stale\n")
    with pytest.raises(ValueError, match="source.*changed"):
        chosen.revalidate()
    assert calls == []
    chosen.source_path.write_text(source())

    def mutate():
        chosen.source_path.write_text(source() + "\n// callback drift\n")

    chosen = replace(chosen, validate_original=mutate)
    with pytest.raises(ValueError, match="source.*changed"):
        chosen.revalidate()


def test_actual_private_implementation_sources_are_recorded_and_reopened(tmp_path, monkeypatch):
    from merlin.llvmlower import borrowed_buffer_calls as B

    owner = tmp_path / "owners"
    owner.mkdir()
    for name in ("borrowed_buffer_calls.py", "dealloc_lifetime.py"):
        (owner / name).write_bytes(Path(B.__file__).with_name(name).read_bytes())
    monkeypatch.setattr(B, "__file__", str(owner / "borrowed_buffer_calls.py"))
    chosen = selection(tmp_path)
    assert {item["path"] for item in chosen.record()["implementation_sources"]} == {
        str(owner / "borrowed_buffer_calls.py"),
        str(owner / "dealloc_lifetime.py"),
    }
    with (owner / "dealloc_lifetime.py").open("a") as output:
        output.write("\n# changed\n")
    with pytest.raises(ValueError, match="implementation changed"):
        chosen.revalidate()


def test_from_writers_freezes_all_original_writer_fields_before_callback(tmp_path):
    from test_plural_fresh_tensor_writer import contract, fixture

    from merlin.llvmlower.borrowed_buffer_calls import from_writers
    from merlin.llvmlower.fresh_tensor_writer import rewrite_fresh_tensor_writers
    from merlin.xdsl_dialects._common import text

    writer = contract()
    module = fixture()
    rewrite_fresh_tensor_writers(module, (writer,))
    selected = tmp_path / "prepared.mlir"
    selected.write_text(text(module, generic=True))

    def original():
        object.__setattr__(writer, "result_arguments", (2, 3))

    with pytest.raises(ValueError, match="original plural writer selection changed"):
        from_writers(selected, (writer,), (), original)


native = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")


@native
def test_current_plural_body_and_both_fresh_generations_are_checked():
    assert dealloc_placement_violations(source(), borrowed_calls=(ROW,)) == []
    assert dealloc_placement_violations(source())


@native
@pytest.mark.parametrize(
    "before,after",
    [
        ("@borrowed_plural(%a,%b,%out_a,%out_b)", "@borrowed_plural(%unused_a,%b,%out_a,%out_b)"),
        ("@borrowed_plural(%a,%b,%out_a,%out_b)", "@borrowed_plural(%a,%b,%unused_a,%out_b)"),
        ("@borrowed_plural(%a,%b,%out_a,%out_b)", "@borrowed_plural(%a,%b,%out_a,%unused_b)"),
        ("return\n }", "memref.dealloc %out_b:memref<3x7xi16>\nreturn\n }"),
        ("memref.dealloc %result#1:memref<3x7xi16>", ""),
        ("memref.dealloc %previous_b:memref<3x7xi16>", "memref.dealloc %fresh_b:memref<3x7xi16>"),
        ("scf.yield %fresh_a,%fresh_b", "scf.yield %fresh_a,%previous_b"),
    ],
)
def test_one_changed_output_input_or_release_cannot_inherit_plural_permission(before, after):
    original = source()
    assert before in original
    assert dealloc_placement_violations(original.replace(before, after), borrowed_calls=(ROW,))


@native
def test_same_typed_swapped_outputs_and_aliases_are_not_shape_authority():
    original = source().replace("3x7xi16", "2x19xi8")
    row = replace(ROW, argument_types=("memref<2x19xi8>",) * 4)
    assert dealloc_placement_violations(original, borrowed_calls=(row,)) == []
    swapped = original.replace("@borrowed_plural(%a,%b,%out_a,%out_b)", "@borrowed_plural(%a,%b,%out_b,%out_a)")
    assert dealloc_placement_violations(swapped, borrowed_calls=(row,))
    shared = original.replace("%fresh_b,%fresh_a):", "%fresh_a,%fresh_a):")
    assert any("distinct fresh" in v for v in dealloc_placement_violations(shared, borrowed_calls=(row,)))


@native
def test_inlined_direct_borrowed_call_is_checked_at_its_actual_output_positions():
    direct = (
        "func.call @borrowed_plural(%previous_a,%previous_b,%fresh_a,%fresh_b):"
        "(memref<2x19xi8>,memref<3x7xi16>,memref<2x19xi8>,memref<3x7xi16>)->()"
    )
    assert dealloc_placement_violations(source().replace(CALLSITE, direct), borrowed_calls=(ROW,)) == []


@native
def test_changed_readonly_role_and_unqualified_public_output_owners_refuse():
    changed = source().replace('bufferization.access="read"', 'bufferization.access="write"', 1)
    assert any("argument roles" in value for value in dealloc_placement_violations(changed, borrowed_calls=(ROW,)))
    public = """module {func.func @entry(%a:memref<2x19xi8>,%b:memref<3x7xi16>,
      %out_a:memref<2x19xi8>,%out_b:memref<3x7xi16>) {
      func.call @borrowed_plural(%a,%b,%out_a,%out_b):
        (memref<2x19xi8>,memref<3x7xi16>,memref<2x19xi8>,memref<3x7xi16>)->()
      return
    }}""".replace("module {", "module {" + SUPPORT)
    assert any("distinct fresh" in value for value in dealloc_placement_violations(public, borrowed_calls=(ROW,)))


@native
def test_unknown_observer_or_changed_external_definition_stays_a_barrier():
    text = source().replace("module {", "module {func.func private @unknown(memref<3x7xi16>)")
    text = text.replace(CALLSITE, CALLSITE + "\nfunc.call @unknown(%fresh_b):(memref<3x7xi16>)->()")
    assert dealloc_placement_violations(text, borrowed_calls=(ROW,))
    declaration = SUPPORT[
        SUPPORT.index(" func.func private @borrowed_plural") : SUPPORT.index(" func.func private @plural(")
    ]
    definition = """func.func private @borrowed_plural(%a:memref<2x19xi8>,%b:memref<3x7xi16>,
       %c:memref<2x19xi8>,%d:memref<3x7xi16>) attributes {llvm.emit_c_interface} {return}
"""
    defined = source().replace(declaration, definition)
    assert dealloc_placement_violations(defined, borrowed_calls=(ROW,))
