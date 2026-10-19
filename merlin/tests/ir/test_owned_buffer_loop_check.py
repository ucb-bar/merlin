"""Keep conservative refusal for unsupported loop-carried ownership provenance."""

import pytest

from merlin.llvmlower import toolchain
from merlin.llvmlower.pipeline import dealloc_placement_violations


@pytest.mark.parametrize("initially_owned", [False, True])
def test_replacing_carried_buffers_does_not_bypass_lifetime_check(initially_owned):
    if not toolchain.available():
        pytest.skip("native MLIR lifetime checker unavailable")
    # With false, this is legal ownership lowering but the verifier cannot yet
    # distinguish first-iteration inputs from fresh backedge allocations. With
    # true, it really frees the initial buffer twice. Both must remain refused
    # until a stronger ownership proof can distinguish them.
    source = """module { func.func @replace(%out:memref<4xi8>) {
      %c0=arith.constant 0:index
      %c1=arith.constant 1:index
      %c3=arith.constant 3:index
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
    } }""".replace("INITIAL", "%true" if initially_owned else "%false")
    violations = dealloc_placement_violations(source)
    assert any("repeating control flow" in message for message in violations)
    assert any("freed at" in message for message in violations)
