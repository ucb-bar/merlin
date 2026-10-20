"""Host preprocessing cannot silently change a callback's declared ABI."""

from xdsl.dialects.func import FuncOp

from merlin.llvmlower.op_profile_structural import _parse_text
from merlin.llvmlower.passes_xdsl import add_c_interface


def test_private_scalar_callbacks_keep_their_abi_and_buffer_routes_are_preserved():
    module = _parse_text('''module {
      func.func private @callback(i32)
      func.func private @helper() { return }
      func.func private @buffer_callback(memref<4xi32>)
      func.func @unwrapped(i32)
      func.func private @wrapped(i32) attributes {llvm.emit_c_interface}
      func.func @forward() { return }
      func.func public @other() { return }
    }''')
    assert add_c_interface(module) == 4
    functions = {op.sym_name.data: op for op in module.walk() if isinstance(op, FuncOp)}
    assert {name for name, op in functions.items() if "llvm.emit_c_interface" in op.attributes} == {
        "forward", "other", "wrapped", "buffer_callback", "unwrapped",
    }
    module.verify()
