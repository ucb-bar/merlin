"""Host preprocessing cannot silently change a callback's declared ABI."""

from xdsl.dialects.func import FuncOp

from merlin.llvmlower.op_profile_structural import _parse_text
from merlin.llvmlower.passes_xdsl import add_c_interface


def test_definitions_get_wrappers_and_declarations_keep_their_selected_abi():
    module = _parse_text('''module {
      func.func private @callback(i32)
      func.func private @helper() { return }
      func.func private @buffer_callback(memref<4xi32>)
      func.func private @kernel(%a: memref<4xi32>) { return }
      func.func @unwrapped(i32)
      func.func private @wrapped(i32) attributes {llvm.emit_c_interface}
      func.func @forward() { return }
      func.func public @other() { return }
    }''')
    assert add_c_interface(module) == 3
    functions = {op.sym_name.data: op for op in module.walk() if isinstance(op, FuncOp)}
    # A definition's wrapper is additive; a declaration's attribute would redirect its calls.
    assert {name for name, op in functions.items() if "llvm.emit_c_interface" in op.attributes} == {
        "forward", "other", "kernel", "wrapped",
    }
    module.verify()
