"""Runtime generations require a real zero-trip copy and complete releases."""

import pytest
from test_owned_buffer_loop_check import COPY, FINAL_COPY, FINAL_FREE, GUARD, loop_source

from merlin.llvmlower import toolchain
from merlin.llvmlower.pipeline import dealloc_placement_violations

pytestmark = pytest.mark.skipif(not toolchain.available(), reason="native MLIR unavailable")

PROMOTION = '''%owned_result=scf.if %result#1 -> memref<4xi8> {
 scf.yield %result#0:memref<4xi8>
} else {
 %copied=memref.alloc():memref<4xi8>
 memref.copy %result#0,%copied:memref<4xi8> to memref<4xi8>
 scf.yield %copied:memref<4xi8>
}'''
SEED_FREE = "memref.dealloc %initial:memref<4xi8>"


def promoted(*, shape="4", dtype="i8", bound="%bound"):
    source = loop_source().replace("%bound:index) {", "%bound:index)->memref<4xi8> {")
    source = source.replace("to %c3 step", "to " + bound + " step")
    source = source.replace(SEED_FREE, PROMOTION + "\n" + SEED_FREE)
    source = source.replace(FINAL_COPY, "").replace(FINAL_FREE, "")
    source = source.replace("return\n", "return %owned_result:memref<4xi8>\n")
    source = source.replace("4xi8", shape + "x" + dtype).replace("0:i8", "0:" + dtype)
    source = source.replace("%zero:i8", "%zero:" + dtype)
    if dtype == "f32":
        source = source.replace("constant 0:f32", "constant 0.0:f32")
    return source


@pytest.mark.parametrize("shape,dtype", [("4", "i8"), ("2x19", "i16"), ("3x7", "f32")])
@pytest.mark.parametrize("bound", ["%bound", "%c0", "%c3"])
def test_runtime_zero_and_positive_trips_keep_the_same_returned_owner_proof(shape, dtype, bound):
    assert dealloc_placement_violations(promoted(shape=shape, dtype=dtype, bound=bound)) == []


MUTATIONS = [
    ("borrowed-zero-trip", "%copied=memref.alloc():memref<4xi8>\n"
     " memref.copy %result#0,%copied:memref<4xi8> to memref<4xi8>\n scf.yield %copied",
     "scf.yield %result#0"),
    ("unwritten-zero-trip", "memref.copy %result#0,%copied:memref<4xi8> to memref<4xi8>", ""),
    ("wrong-copy-source", "memref.copy %result#0,%copied", "memref.copy %out,%copied"),
    ("early-seed-free", PROMOTION + "\n" + SEED_FREE, SEED_FREE + "\n" + PROMOTION),
    ("double-seed-free", SEED_FREE, SEED_FREE + "\n" + SEED_FREE),
    ("released-result", "return %owned_result", "memref.dealloc %owned_result:memref<4xi8>\nreturn %owned_result"),
    ("wrong-promotion-flag", "%owned_result=scf.if %result#1", "%owned_result=scf.if %false"),
    ("unguarded-previous", GUARD, "memref.dealloc %previous:memref<4xi8>"),
    ("old-after-release", GUARD, GUARD + "\n" + COPY),
    ("false-backedge", "scf.yield %fresh,%true", "scf.yield %fresh,%false"),
    ("released-fresh", "scf.yield %fresh,%true", "memref.dealloc %fresh:memref<4xi8>\nscf.yield %fresh,%true"),
]


@pytest.mark.parametrize("name,before,after", MUTATIONS, ids=[row[0] for row in MUTATIONS])
def test_unsafe_zero_trip_ownership_or_generation_mutations_refuse(name, before, after):
    source = promoted()
    assert before in source
    assert dealloc_placement_violations(source.replace(before, after)), name


def test_abstract_clone_does_not_supply_freshness_before_materialization():
    source = promoted().replace(
        "%copied=memref.alloc():memref<4xi8>\n memref.copy %result#0,%copied:memref<4xi8> to memref<4xi8>",
        "%copied=bufferization.clone %result#0:memref<4xi8> to memref<4xi8>")
    assert any("materialized" in problem for problem in dealloc_placement_violations(source))


def test_initial_owner_does_not_gain_implicit_noescape_or_address_permission():
    source = promoted().replace("module {", "module {func.func private @escape(memref<4xi8>)")
    assert dealloc_placement_violations(source.replace(
        "%result:2=scf.for", "func.call @escape(%initial):(memref<4xi8>)->()\n%result:2=scf.for"))
    assert dealloc_placement_violations(promoted().replace(
        "%result:2=scf.for", "%ptr=memref.extract_aligned_pointer_as_index %initial:memref<4xi8> -> index\n"
        "%result:2=scf.for"))


@pytest.mark.parametrize("where", ["body", "zero-trip", "returned"])
def test_unknown_effects_cannot_hide_in_returned_generations(where):
    source = promoted().replace("module {", "module {func.func private @opaque()")
    anchor = {"body": COPY, "zero-trip": "%copied=memref.alloc():memref<4xi8>",
              "returned": SEED_FREE}[where]
    source = source.replace(anchor, anchor + "\nfunc.call @opaque():()->()")
    if where == "returned":
        # A void surrounding source effect does not consume/retain the result.
        # The proof concerns memory lifetimes, not arbitrary source purity.
        assert dealloc_placement_violations(source) == []
    else:
        assert dealloc_placement_violations(source)
