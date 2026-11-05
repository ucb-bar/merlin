"""The selected completion ABI permits only an unused native void placeholder."""
from io import StringIO
from pathlib import Path
import subprocess

import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr
from xdsl.printer import Printer

from merlin.targetgen.oot_starterkit.llvm_context import parse_llvm_module
from merlin.targetgen.oot_starterkit.plan import _lowered_task_problems, _selected_lowered_entry

ABI = {'entry_symbol': 'alternate_call', 'completion_symbol': 'alternate_finish', 'completion_asm': 'fence'}
TEXT = '''module {
 llvm.func @alternate_call(%p: !llvm.ptr) { llvm.return }
 llvm.func @alternate_finish() {
  %v = "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects}> : () -> !llvm.void
  llvm.return
 }
}'''


def functions(text):
    return [op for op in parse_llvm_module(text).body.block.ops if isinstance(op, llvm.FuncOp)]


@pytest.mark.parametrize('result', ['void', 'none'])
def test_exact_fence_completion_accepts_only_semantically_void_representations(result):
    text = TEXT if result == 'void' else TEXT.replace('%v = ', '').replace('() -> !llvm.void', '() -> ()')
    entry, problem = _selected_lowered_entry(functions(text), ABI)
    assert problem is None
    assert entry.sym_name.data == 'alternate_call'


@pytest.mark.parametrize('result', ['i64', '!llvm.ptr'])
def test_scalar_or_pointer_inline_asm_result_is_not_a_completion_placeholder(result):
    _, problem = _selected_lowered_entry(functions(TEXT.replace('() -> !llvm.void', '() -> ' + result)), ABI)
    assert problem == 'lowered completion differs from the declared fence-only ABI'


def test_void_placeholder_with_a_use_is_not_accepted():
    funcs = functions(TEXT)
    assembly = next(op for op in funcs[1].walk() if isinstance(op, llvm.InlineAsmOp))
    retained_use = llvm.ReturnOp(assembly.res)
    assert assembly.res.uses
    _, problem = _selected_lowered_entry(funcs, ABI)
    assert problem == 'lowered completion differs from the declared fence-only ABI'
    retained_use.drop_all_references()


@pytest.mark.parametrize('old,new', [
    ('asm_string = "fence"', 'asm_string = "nop"'),
    ('constraints = "~{memory}"', 'constraints = ""'),
    (', has_side_effects', ''),
    ('@alternate_finish()', '@unselected_finish()'),
    ('}> : () -> !llvm.void', '}> {merlin.global_task = 0 : i64} : () -> !llvm.void'),
])
def test_void_representation_does_not_relax_completion_identity_or_effects(old, new):
    entry, problem = _selected_lowered_entry(functions(TEXT.replace(old, new)), ABI)
    assert entry is None and problem


def test_actual_native_c_linkage_import_passes_selected_completion_and_owned_task_gate(tmp_path):
    from merlin.targetgen.contract import toolchain
    from merlin.llvmlower import toolchain as host_toolchain
    clang, translate = host_toolchain.clang(), toolchain.mlir_bin('mlir-translate')
    if not all(Path(tool).is_file() for tool in (clang, translate)):
        pytest.skip('selected native C/MLIR toolchain unavailable')
    source, llvm_ir, imported = [tmp_path / name for name in ('control.cpp', 'control.ll', 'control.mlir')]
    # Private protocol-only control, no kernel algorithm or execution. Both
    # externally visible definitions use C linkage, independent of target names.
    source.write_text('extern "C" void alternate_call(void *p) { asm volatile("" ::: "memory"); }\n'
                      'extern "C" void alternate_finish(void) { asm volatile("fence" ::: "memory"); }\n')
    subprocess.run([str(clang), '--target=riscv64-unknown-unknown-elf', '-march=rv64gc', '-mabi=lp64d', '-S', '-emit-llvm', '-O1', '-ffreestanding', '-fno-builtin', str(source), '-o', str(llvm_ir)],
                   capture_output=True, text=True, check=True, timeout=30)
    subprocess.run([str(translate), '--import-llvm', str(llvm_ir), '-o', str(imported)],
                   capture_output=True, text=True, check=True, timeout=30)
    module = parse_llvm_module(imported.read_text())
    funcs = [op for op in module.body.block.ops if isinstance(op, llvm.FuncOp)]
    entry, problem = _selected_lowered_entry(funcs, ABI)
    assert problem is None
    completion = next(op for op in funcs if op.sym_name.data == ABI['completion_symbol'])
    fence = next(op for op in completion.walk() if isinstance(op, llvm.InlineAsmOp))
    assert len(fence.results) == 1 and isinstance(fence.results[0].type, llvm.LLVMVoidType)
    assert not fence.results[0].uses
    # A private structural test assigns the one real entry effect to its task.
    # The completion helper stays untagged and cannot stand in for task work.
    work = next(op for op in entry.walk() if isinstance(op, llvm.InlineAsmOp))
    work.attributes['merlin.global_task'] = IntegerAttr(0, 64)
    work.attributes['merlin.source_op_index'] = IntegerAttr(0, 64)
    output = StringIO(); Printer(stream=output, print_generic_format=True).print_op(module)
    assert _lowered_task_problems(output.getvalue(), {'tasks': [{'task_index': 0, 'source_op_indices': [0]}]},
        {'kernel_abi': {'args': [{'tensor': 'p'}]}}, ABI) == []
