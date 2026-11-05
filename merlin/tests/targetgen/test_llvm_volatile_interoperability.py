"""Native volatile memory operations retain effects through typed inspection."""
import io
import os
from pathlib import Path
import subprocess

import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import UnitAttr
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException

from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context, parse_llvm_module


GENERIC = '''module {
  llvm.func @kernel(%p: !llvm.ptr) {
    %v = "llvm.load"(%p) <{alignment = 4 : i64, ordering = 0 : i64, volatile_}>
      {merlin.global_task = 0 : i64, merlin.source_op_index = 0 : i64}
      : (!llvm.ptr) -> i32
    "llvm.store"(%v, %p) <{alignment = 4 : i64, ordering = 0 : i64, volatile_}>
      {merlin.global_task = 0 : i64, merlin.source_op_index = 0 : i64}
      : (i32, !llvm.ptr) -> ()
    llvm.return
  }
}'''
NATIVE = '''module {
  llvm.func @kernel(%p: !llvm.ptr) {
    %v = llvm.load volatile %p
      {alignment = 4 : i64, merlin.global_task = 0 : i64, merlin.source_op_index = 0 : i64}
      : !llvm.ptr -> i32
    llvm.store volatile %v, %p
      {alignment = 4 : i64, merlin.global_task = 0 : i64, merlin.source_op_index = 0 : i64}
      : i32, !llvm.ptr
    llvm.return
  }
}'''


def _memory(module):
    return [op for op in module.walk() if op.name in {'llvm.load', 'llvm.store'}]


def _assert_preserved(module):
    operations = _memory(module)
    assert [op.name for op in operations] == ['llvm.load', 'llvm.store']
    for op in operations:
        assert isinstance(op.properties['volatile_'], UnitAttr)
        assert op.properties['alignment'].value.data == 4
        assert op.attributes['merlin.global_task'].value.data == 0
        assert op.attributes['merlin.source_op_index'].value.data == 0
    assert operations[1].operands[0] is operations[0].results[0]


def _print(module, *, generic=False):
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=generic).print_op(module)
    return stream.getvalue()


@pytest.fixture
def native_tool(monkeypatch):
    install = os.environ.get('MERLIN_MLIR_INSTALL')
    compiler = os.environ.get('MERLIN_CLANG')
    if install:
        tool = Path(install)/'bin/mlir-opt'
    elif compiler and Path(compiler).is_absolute():
        tool = Path(compiler).parent/'mlir-opt'
    else:
        pytest.skip('no explicitly selected native LLVM installation')
    if not tool.is_file():
        pytest.skip('selected native LLVM tool unavailable')
    monkeypatch.setenv('MERLIN_MLIR_INSTALL', str(tool.parent.parent))
    return tool


def _native(tool, text):
    return subprocess.run([str(tool), '--mlir-print-op-generic',
        '--mlir-print-local-scope', '-o', '-'], input=text,
        capture_output=True, text=True, timeout=30)


def test_direct_generic_volatile_memory_is_typed_without_native_process(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('direct generic typed parsing must stay dependency free')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    original = dict(llvm.LoadOp.get_irdl_definition().properties)
    _assert_preserved(parse_llvm_module(GENERIC))
    assert dict(llvm.LoadOp.get_irdl_definition().properties) == original


@pytest.mark.parametrize('generic', [False, True])
def test_print_reparse_never_discards_volatile_or_task_identity(generic):
    module = parse_llvm_module(GENERIC)
    printed = _print(module, generic=generic)
    assert 'volatile_' in printed
    _assert_preserved(parse_llvm_module(printed))


@pytest.mark.parametrize('operation', ['load', 'store'])
@pytest.mark.parametrize('bad_property', ['volatile_ = "yes"', 'volatile_ = 1 : i64'])
def test_volatile_requires_standard_unit_property(operation, bad_property):
    text = GENERIC
    start = text.index('"llvm.' + operation + '"')
    text = text[:start] + text[start:].replace('volatile_', bad_property, 1)
    with pytest.raises(VerifyException):
        parse_llvm_module(text)


def test_load_still_requires_pointer_operand():
    text = GENERIC.replace('@kernel(%p: !llvm.ptr)', '@kernel(%p: i32)')
    text = text.replace(': (!llvm.ptr) -> i32', ': (i32) -> i32')
    text = text.replace(': (i32, !llvm.ptr) -> ()', ': (i32, i32) -> ()')
    with pytest.raises(VerifyException):
        parse_llvm_module(text)


def test_load_still_requires_typed_alignment():
    with pytest.raises(VerifyException):
        parse_llvm_module(GENERIC.replace('alignment = 4 : i64', 'alignment = "four"', 1))


def test_native_custom_and_generic_forms_both_preserve_memory_effects(native_tool):
    native = _native(native_tool, NATIVE)
    assert native.returncode == 0, native.stderr
    _assert_preserved(parse_llvm_module(NATIVE))
    _assert_preserved(parse_llvm_module(native.stdout))
    module = Parser(make_llvm_context(), native.stdout).parse_module()
    module.verify()
    _assert_preserved(module)
    for generic in [False, True]:
        printed = _print(module, generic=generic)
        reparsed = _native(native_tool, printed)
        assert reparsed.returncode == 0, reparsed.stderr
        _assert_preserved(parse_llvm_module(reparsed.stdout))


def test_native_malformed_type_is_not_repaired(native_tool):
    bad = NATIVE.replace(': !llvm.ptr -> i32', ': !llvm.ptr -> nonexistent')
    assert _native(native_tool, bad).returncode != 0
    with pytest.raises(ValueError, match='native parse/verification failed'):
        parse_llvm_module(bad)


def test_volatile_compatibility_does_not_relax_task_source_ownership():
    from merlin.targetgen.oot_starterkit.plan import _lowered_task_problems
    bad = GENERIC.replace('merlin.source_op_index = 0', 'merlin.source_op_index = 7')
    problems = _lowered_task_problems(bad,
        {'tasks': [{'task_index': 0, 'source_op_indices': [0]}]},
        {'kernel_abi': {'args': [{'tensor': 'p'}]}})
    assert 'lowered source_op_index differs from owning global task' in problems
