"""Generation-sensitive native lifetime checks, including unsafe mutations."""

import pytest

from merlin.llvmlower import toolchain
from merlin.llvmlower.pipeline import dealloc_placement_violations

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")

COPY = "memref.copy %previous,%fresh:memref<4xi8> to memref<4xi8>"
GUARD = "scf.if %owned { memref.dealloc %previous:memref<4xi8> }"
FINAL_COPY = "memref.copy %result#0,%out:memref<4xi8> to memref<4xi8>"
FINAL_FREE = "memref.dealloc %result#0:memref<4xi8>"


def loop_source(*, initially_owned=False, trips=3, shape="4", element="i8"):
    source = """module { func.func @replace(%out:memref<4xi8>,%bound:index) {
      %c0=arith.constant 0:index
      %c1=arith.constant 1:index
      %c3=arith.constant TRIPS:index
      %false=arith.constant false
      %true=arith.constant true
      %zero=arith.constant 0:i8
      %initial=memref.alloc():memref<4xi8>
      linalg.fill ins(%zero:i8) outs(%initial:memref<4xi8>)
      %result:2=scf.for %iv=%c0 to %c3 step %c1
      iter_args(%previous=%initial,%owned=INITIAL)->(memref<4xi8>,i1) {
        %fresh=memref.alloc():memref<4xi8>
        memref.copy %previous,%fresh:memref<4xi8> to memref<4xi8>
        scf.if %owned { memref.dealloc %previous:memref<4xi8> }
        scf.yield %fresh,%true:memref<4xi8>,i1
      }
      memref.dealloc %initial:memref<4xi8>
      memref.copy %result#0,%out:memref<4xi8> to memref<4xi8>
      memref.dealloc %result#0:memref<4xi8>
      return
    } }"""
    return (
        source.replace("INITIAL", "%true" if initially_owned else "%false")
        .replace("TRIPS", str(trips))
        .replace("4xi8", shape + "x" + element)
        .replace("0:i8", "0:" + element)
        .replace("%zero:i8", "%zero:" + element)
    )


@pytest.mark.parametrize("trips", [1, 3, 19])
@pytest.mark.parametrize("shape,element", [("4", "i8"), ("2x19", "i16"), ("3x7", "f32")])
def test_fresh_generations_are_distinct_from_initial_and_final_owners(trips, shape, element):
    source = loop_source(trips=trips, shape=shape, element=element)
    if element == "f32":
        source = source.replace("constant 0:f32", "constant 0.0:f32")
    assert dealloc_placement_violations(source) == []


@pytest.mark.parametrize("trips", [0, -2])
def test_zero_or_negative_trip_replacement_cannot_separate_final_owner(trips):
    assert any("positive" in v for v in dealloc_placement_violations(loop_source(trips=trips)))


def test_owned_initial_buffer_is_not_freed_again_after_loop():
    bad = dealloc_placement_violations(loop_source(initially_owned=True))
    assert any("ownership" in v for v in bad)
    assert any("repeating control flow" in v for v in bad)


MUTATIONS = [
    ("dynamic_trip", "to %c3 step", "to %bound step"),
    ("dynamic_step", "step %c1", "step %bound"),
    ("old_backedge", "scf.yield %fresh,%true", "scf.yield %previous,%true"),
    ("initial_backedge", "scf.yield %fresh,%true", "scf.yield %initial,%true"),
    ("false_backedge", "scf.yield %fresh,%true", "scf.yield %fresh,%false"),
    ("early_old_release", COPY + "\n        " + GUARD, GUARD + "\n        " + COPY),
    ("unguarded_release", GUARD, "memref.dealloc %previous:memref<4xi8>"),
    ("wrong_guard", "scf.if %owned", "scf.if %true"),
    ("false_guard", "scf.if %owned", "scf.if %false"),
    ("missing_old_release", GUARD, ""),
    ("duplicate_old_release", GUARD, GUARD + "\n" + GUARD),
    ("fresh_release", "dealloc %previous:", "dealloc %fresh:"),
    ("missing_final_release", FINAL_FREE, ""),
    ("duplicate_final_release", FINAL_FREE, FINAL_FREE + "\n" + FINAL_FREE),
    ("early_final_release", FINAL_COPY + "\n      " + FINAL_FREE, FINAL_FREE + "\n" + FINAL_COPY),
    ("early_initial_release", "%result:2=scf.for", "memref.dealloc %initial:memref<4xi8>\n%result:2=scf.for"),
    ("old_view_after_release", GUARD, GUARD + "\n%v=memref.cast %previous:memref<4xi8> to memref<?xi8>"),
    ("fresh_pointer_escape", COPY,
     COPY + "\n%ptr=memref.extract_aligned_pointer_as_index %fresh:memref<4xi8> -> index"),
    ("old_pointer_escape", COPY,
     COPY + "\n%ptr=memref.extract_aligned_pointer_as_index %previous:memref<4xi8> -> index"),
]


@pytest.mark.parametrize("name,before,after", MUTATIONS, ids=[m[0] for m in MUTATIONS])
def test_unsafe_or_unsupported_replacement_is_refused(name, before, after):
    source = loop_source()
    assert before in source
    assert dealloc_placement_violations(source.replace(before, after)), name


@pytest.mark.parametrize("value", ["%previous", "%fresh", "%result#0"])
def test_unknown_call_is_not_a_borrowed_effect_contract(value):
    source = loop_source().replace("module {", "module { func.func private @escape(memref<4xi8>)")
    anchor = FINAL_COPY if value == "%result#0" else COPY
    call = "func.call @escape(" + value + "):(memref<4xi8>)->()"
    assert dealloc_placement_violations(source.replace(anchor, anchor + "\n" + call))


def test_read_write_annotation_does_not_promise_noescape_or_completion():
    source = loop_source().replace(
        "module {", 'module { func.func private @escape(memref<4xi8> {bufferization.access="read"})'
    )
    assert dealloc_placement_violations(
        source.replace(COPY, COPY + "\nfunc.call @escape(%previous):(memref<4xi8>)->()")
    )


def test_hoisted_replacement_allocation_is_not_fresh_per_iteration():
    allocation = "%fresh=memref.alloc():memref<4xi8>"
    source = loop_source().replace(allocation, "").replace("%result:2=scf.for", allocation + "\n%result:2=scf.for")
    assert dealloc_placement_violations(source)


def test_metadata_base_release_and_subview_reads_preserve_generation():
    source = loop_source()
    metadata = ("%base,%offset,%size,%stride=memref.extract_strided_metadata %previous:memref<4xi8>"
                " -> memref<i8>,index,index,index")
    view = "%v=memref.subview %previous[0] [4] [1]:memref<4xi8> to memref<4xi8,strided<[1]>>"
    source = source.replace(
        COPY, view + "\nmemref.copy %v,%fresh:memref<4xi8,strided<[1]>> to memref<4xi8>\n" + metadata
    )
    source = source.replace("dealloc %previous:memref<4xi8>", "dealloc %base:memref<i8>")
    assert dealloc_placement_violations(source) == []
    assert dealloc_placement_violations(
        source.replace("dealloc %base:memref<i8>", "dealloc %v:memref<4xi8,strided<[1]>>")
    )


def test_borrowed_initial_has_no_allocation_root_but_final_owner_is_checked():
    source = loop_source().replace("%bound:index", "%bound:index,%initial:memref<4xi8>")
    source = source.replace("%initial=memref.alloc():memref<4xi8>", "")
    source = source.replace("memref.dealloc %initial:memref<4xi8>", "")
    assert dealloc_placement_violations(source) == []
    assert dealloc_placement_violations(source.replace(FINAL_COPY + "\n      " + FINAL_FREE,
                                                      FINAL_FREE + "\n" + FINAL_COPY))


MULTIPLE = """module {func.func @multiple(%a:memref<2x19xi8>,%b:memref<3x7xi16>,
 %history:memref<3x2x19xi8>) {
 %c0=arith.constant 0:index
 %c1=arith.constant 1:index
 %c3=arith.constant 3:index
 %false=arith.constant false
 %true=arith.constant true
 %a0=memref.alloc():memref<2x19xi8>
 %b0=memref.alloc():memref<3x7xi16>
 memref.copy %a,%a0:memref<2x19xi8> to memref<2x19xi8>
 memref.copy %b,%b0:memref<3x7xi16> to memref<3x7xi16>
 %result:4=scf.for %iv=%c0 to %c3 step %c1
 iter_args(%previous_a=%a0,%previous_b=%b0,%h=%history,%owned=%false)
 ->(memref<2x19xi8>,memref<3x7xi16>,memref<3x2x19xi8>,i1) {
 %fresh_a=memref.alloc():memref<2x19xi8>
 %fresh_b=memref.alloc():memref<3x7xi16>
 memref.copy %previous_a,%fresh_a:memref<2x19xi8> to memref<2x19xi8>
 memref.copy %previous_b,%fresh_b:memref<3x7xi16> to memref<3x7xi16>
 %view=memref.subview %h[%iv,0,0] [1,2,19] [1,1,1]:
 memref<3x2x19xi8> to memref<2x19xi8,strided<[19,1],offset:?>>
 memref.copy %fresh_a,%view:memref<2x19xi8> to memref<2x19xi8,strided<[19,1],offset:?>>
 scf.if %owned {
 memref.dealloc %previous_a:memref<2x19xi8>
 memref.dealloc %previous_b:memref<3x7xi16>
 }
 scf.yield %fresh_a,%fresh_b,%h,%true:memref<2x19xi8>,memref<3x7xi16>,memref<3x2x19xi8>,i1
 }
 memref.dealloc %a0:memref<2x19xi8>
 memref.dealloc %b0:memref<3x7xi16>
 memref.copy %result#0,%a:memref<2x19xi8> to memref<2x19xi8>
 memref.copy %result#1,%b:memref<3x7xi16> to memref<3x7xi16>
 memref.dealloc %result#0:memref<2x19xi8>
 memref.dealloc %result#1:memref<3x7xi16>
 return
}}"""


def test_two_generations_share_flag_but_not_owners_or_passthrough_history():
    assert dealloc_placement_violations(MULTIPLE) == []
    wrong_release = MULTIPLE.replace("dealloc %previous_b:memref<3x7xi16>",
                                     "dealloc %previous_a:memref<2x19xi8>")
    assert dealloc_placement_violations(wrong_release)
    same_types = MULTIPLE.replace("3x7xi16", "2x19xi8")
    shared = same_types.replace("scf.yield %fresh_a,%fresh_b", "scf.yield %fresh_a,%fresh_a")
    assert dealloc_placement_violations(shared)


@pytest.mark.parametrize("where", ["previous", "fresh", "final"])
def test_nested_memref_store_cannot_hide_a_generation_until_after_free(where):
    source = loop_source().replace(
        "%result:2=scf.for", "%slot=memref.alloc():memref<memref<4xi8>>\n%result:2=scf.for"
    )
    value = "%result#0" if where == "final" else "%" + where
    anchor = FINAL_COPY if where == "final" else COPY
    source = source.replace(anchor, anchor + "\nmemref.store " + value + ",%slot[]:memref<memref<4xi8>>")
    source = source.replace(
        FINAL_FREE,
        FINAL_FREE + "\n%escaped=memref.load %slot[]:memref<memref<4xi8>>\n"
        "memref.copy %escaped,%out:memref<4xi8> to memref<4xi8>\n"
        "memref.dealloc %slot:memref<memref<4xi8>>",
    )
    assert dealloc_placement_violations(source)


def test_copied_buffers_with_nested_handle_elements_are_not_a_closed_numeric_domain():
    source = loop_source().replace("linalg.fill ins(%zero:i8) outs(%initial:memref<4xi8>)", "")
    source = source.replace("4xi8", "4xmemref<i8>")
    assert dealloc_placement_violations(source)


def test_final_consumer_scalar_region_cannot_hide_unknown_effects():
    source = loop_source().replace("module {", "module {func.func private @opaque()")
    consumer = """linalg.generic {
      indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]}
      ins(%result#0:memref<4xi8>) outs(%out:memref<4xi8>) {
      ^bb0(%x:i8,%unused:i8): func.call @opaque():()->() linalg.yield %x:i8
    }"""
    assert dealloc_placement_violations(source.replace(FINAL_COPY, consumer))
