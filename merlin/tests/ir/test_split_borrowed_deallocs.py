"""Aggregate splitting uses exact native selected call operations."""

import subprocess

import pytest

from merlin.llvmlower import pipeline as P
from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferEffects, BorrowedBufferResults, bind_native
from merlin.llvmlower.toolchain import m2m_python


SOURCE = '''module {
 func.func private @borrowed(memref<4xi8> {bufferization.access="read"},
 memref<4xi8> {bufferization.access="write"},memref<4xi8> {bufferization.access="write"})
 attributes {llvm.emit_c_interface}
 func.func private @unknown(memref<4xi8>)
 func.func @forward(%a:memref<4xi8>) {
 %x=memref.alloc():memref<4xi8>
 %y=memref.alloc():memref<4xi8>
 func.call @borrowed(%a,%x,%y):(memref<4xi8>,memref<4xi8>,memref<4xi8>)->()
 EXTRA
 %bx,%ox,%sx,%stx=memref.extract_strided_metadata %x:memref<4xi8> -> memref<i8>,index,index,index
 %by,%oy,%sy,%sty=memref.extract_strided_metadata %y:memref<4xi8> -> memref<i8>,index,index,index
 %t=arith.constant true
 "bufferization.dealloc"(%bx,%by,%t,%t) <{operandSegmentSizes=array<i32:2,2,0>}>:
 (memref<i8>,memref<i8>,i1,i1)->()
 return
 }
}'''


@pytest.mark.parametrize("mode", ["selected", "unselected", "unknown", "access", "alias"])
def test_only_selected_current_operations_remove_aggregate_escape_barrier(tmp_path, upstream_host_tools, mode):
    source = SOURCE.replace("EXTRA", "func.call @unknown(%x):(memref<4xi8>)->()" if mode == "unknown" else "")
    if mode == "access":
        source = source.replace('bufferization.access="read"', 'bufferization.access="write"')
    if mode == "alias":
        source = source.replace("@borrowed(%a,%x,%y)", "@borrowed(%a,%x,%x)")
    calls = () if mode == "unselected" else (BorrowedBufferResults("unused_writer", "borrowed",
        ("memref<4xi8>",) * 3, (1, 2), BorrowedBufferEffects("neutral-original-owner", True, True, True)),)
    prelude = bind_native(P.DEALLOC_CHECK_PRELUDE, calls) + P.SPLIT_FRESH_DEALLOCS_PRELUDE
    script = tmp_path / "check.py"
    script.write_text("from torch_mlir import ir\n" + prelude +
        "ctx=ir.Context()\nwith ctx:\n module=ir.Module.parse(" + repr(source) + ")\n"
        " module.operation.verify()\n _split_fresh_local_deallocs(ctx,module)\n print(module)\n")
    result = subprocess.run([str(m2m_python()), str(script)], capture_output=True, text=True, timeout=30)
    if mode in ("access", "alias"):
        assert result.returncode != 0 and "borrowed ABI" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        if mode == "selected":
            assert "OK split_fresh_local_deallocs 2 fresh allocs" in result.stdout
            assert result.stdout.count("memref.dealloc %") == 2
        else:
            assert "OK split_fresh_local_deallocs 0 fresh allocs" in result.stdout
            assert 'bufferization.dealloc' in result.stdout
