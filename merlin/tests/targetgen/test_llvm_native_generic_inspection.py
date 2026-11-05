"""Native custom spellings must not bypass typed or task-ownership checks."""
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from xdsl.dialects import llvm
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError, VerifyException

from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context, parse_llvm_module


NATIVE_CUSTOM = '''module {
  llvm.func @kernel(%p: !llvm.ptr, %i: i64) {
    %c = llvm.mlir.constant(1 : i64) {merlin.global_task = 0 : i64} : i64
    %a = llvm.getelementptr inbounds|nuw %p[%i] {merlin.global_task = 0 : i64}
      : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %f = llvm.inline_asm has_side_effects tail_call_kind = <tail> asm_dialect = att
      {merlin.global_task = 0 : i64, merlin.source_op_index = 0 : i64}
      "fence", "~{memory}" : () -> !llvm.void
    llvm.return
  }
}'''
GENERIC = '''module {
  llvm.func @kernel(%p: !llvm.ptr) {
    %c = "llvm.mlir.constant"() <{value = 1 : i64}>
      {merlin.global_task = 0 : i64} : () -> i64
    llvm.return
  }
}'''


def _mock_native(monkeypatch, *, output=GENERIC, returncode=0, stderr="", failure=None):
    monkeypatch.setenv("MERLIN_MLIR_INSTALL", "/selected-native")
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if failure is not None:
            raise failure
        return SimpleNamespace(stdout=output, stderr=stderr, returncode=returncode)
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_supported_spelling_keeps_original_class_and_requires_no_native_tool(monkeypatch):
    calls = _mock_native(monkeypatch)
    module = parse_llvm_module('module { llvm.func @kernel() { %c = llvm.mlir.constant(1 : i64) : i64\n llvm.return } }')
    assert type(next(op for op in module.walk() if op.name == 'llvm.mlir.constant')) is llvm.ConstantOp
    assert calls == []


def test_fallback_uses_selected_tool_exact_original_input_and_no_passes(monkeypatch):
    calls = _mock_native(monkeypatch)
    module = parse_llvm_module(NATIVE_CUSTOM)
    assert calls == [([
        '/selected-native/bin/mlir-opt', '--mlir-print-op-generic',
        '--mlir-print-local-scope', '-o', '-',
    ], dict(input=NATIVE_CUSTOM, capture_output=True, text=True, timeout=30))]
    constant = next(op for op in module.walk() if op.name == 'llvm.mlir.constant')
    assert constant.attributes['merlin.global_task'].value.data == 0
    assert constant.value.value.data == 1


@pytest.mark.parametrize('failure', [OSError('missing selected tool'), subprocess.TimeoutExpired('mlir-opt', 30)])
def test_unavailable_or_timed_out_selected_tool_fails_closed_with_original_diagnostic(monkeypatch, failure):
    _mock_native(monkeypatch, failure=failure)
    with pytest.raises(ValueError, match='Original xDSL syntax diagnostic') as raised:
        parse_llvm_module(NATIVE_CUSTOM)
    assert isinstance(raised.value.__cause__, ParseError)
    assert 'selected-native/bin/mlir-opt' in str(raised.value)


@pytest.mark.parametrize('output,returncode,stderr', [('', 0, ''), ('', 1, 'native verification rejected source')])
def test_native_failure_never_becomes_an_empty_or_successful_module(monkeypatch, output, returncode, stderr):
    _mock_native(monkeypatch, output=output, returncode=returncode, stderr=stderr)
    with pytest.raises(ValueError, match='native parse/verification failed'):
        parse_llvm_module(NATIVE_CUSTOM)


def test_unreadable_generic_serialization_retains_original_diagnostic(monkeypatch):
    _mock_native(monkeypatch, output='module { malformed')
    with pytest.raises(ValueError, match='cannot read native generic serialization') as raised:
        parse_llvm_module(NATIVE_CUSTOM)
    assert 'Original xDSL syntax diagnostic' in str(raised.value)


def test_direct_typed_verification_failure_never_requests_serialization(monkeypatch):
    calls = _mock_native(monkeypatch)
    bad = GENERIC.replace('value = 1 : i64', 'value = "not an integer"')
    with pytest.raises(VerifyException):
        parse_llvm_module(bad)
    assert calls == []


def test_fallback_still_runs_original_typed_property_verification(monkeypatch):
    _mock_native(monkeypatch, output=GENERIC.replace('value = 1 : i64', 'value = "not an integer"'))
    with pytest.raises(VerifyException):
        parse_llvm_module(NATIVE_CUSTOM)


@pytest.fixture
def native_tool(monkeypatch):
    from merlin.targetgen.contract import toolchain
    tool = toolchain.mlir_bin('mlir-opt')
    if not tool.is_file():
        pytest.skip('selected native MLIR install unavailable')
    # Keep its exact selected installation, never discover another LLVM on PATH.
    monkeypatch.setenv('MERLIN_MLIR_INSTALL', str(tool.parent.parent))
    return tool


def test_real_native_custom_module_retains_all_three_syntax_families(native_tool):
    with pytest.raises(ParseError):
        Parser(make_llvm_context(), NATIVE_CUSTOM).parse_module()
    module = parse_llvm_module(NATIVE_CUSTOM)
    operations = {op.name: op for op in module.walk()}
    assert operations['llvm.mlir.constant'].properties['value'].value.data == 1
    assert operations['llvm.getelementptr'].properties['noWrapFlags'].value.data == 7
    assert operations['llvm.inline_asm'].properties['tail_call_kind'].data == llvm.TailCallKind.TAIL
    for name in ('llvm.mlir.constant', 'llvm.getelementptr', 'llvm.inline_asm'):
        assert operations[name].attributes['merlin.global_task'].value.data == 0
    assert operations['llvm.inline_asm'].attributes['merlin.source_op_index'].value.data == 0
    assert len(operations['llvm.getelementptr'].operands) == 2


def test_real_native_malformed_input_is_rejected_without_spelling_repair(native_tool):
    malformed = NATIVE_CUSTOM.replace(' -> !llvm.ptr, i8', ' -> !llvm.ptr, invalid_type')
    with pytest.raises(ValueError, match='native parse/verification failed'):
        parse_llvm_module(malformed)


@pytest.mark.parametrize('tags,expected', [
    ('all', []),
    ('none', ['planned tasks have no tagged lowered operation: [0]', 'planned task emits no owned computation or control flow']),
    ('constant_only', ['planned task emits no owned computation or control flow']),
])
def test_real_native_serialization_preserves_structural_ownership_failures(native_tool, tags, expected):
    from merlin.targetgen.oot_starterkit.plan import _lowered_task_problems
    text = NATIVE_CUSTOM
    if tags == 'none':
        text = text.replace('merlin.global_task = 0 : i64', 'custom.note = 0 : i64')
    elif tags == 'constant_only':
        first, tail = text.split('llvm.getelementptr', 1)
        text = first + 'llvm.getelementptr' + tail.replace('merlin.global_task = 0 : i64', 'custom.note = 0 : i64')
    findings = _lowered_task_problems(text, {'tasks': [{'task_index': 0, 'source_op_indices': [0]}]},
        {'kernel_abi': {'args': [{'tensor': 'p'}, {'tensor': 'i'}]}})
    if expected:
        assert all(finding in findings for finding in expected)
    else:
        assert findings == []


NATIVE_POINTER_ENTRY = NATIVE_CUSTOM.replace('@kernel(%p: !llvm.ptr, %i: i64)', '@kernel(%p: !llvm.ptr)').replace('%c =', '%i =')


def test_real_native_entry_readers_share_syntax_fallback_without_weakening_pointer_abi(native_tool):
    from merlin.targetgen.contract.compile_only import require_pointer_entry
    from merlin.targetgen.bundle_harness import emitted_entry_arity
    require_pointer_entry(NATIVE_POINTER_ENTRY, entry_symbol='kernel', pointer_arity=1)
    assert emitted_entry_arity(NATIVE_POINTER_ENTRY, entry_symbol='kernel') == 1
    with pytest.raises(ValueError, match='selected ordinary C pointer ABI'):
        require_pointer_entry(NATIVE_POINTER_ENTRY, entry_symbol='kernel', pointer_arity=2)
    assert emitted_entry_arity(NATIVE_POINTER_ENTRY, entry_symbol='missing') is None


def test_real_native_pointer_reader_preserves_no_parameter_attributes_policy(native_tool):
    from merlin.targetgen.contract.compile_only import require_pointer_entry
    from merlin.targetgen.bundle_harness import emitted_entry_arity
    text = NATIVE_POINTER_ENTRY.replace('@kernel(%p: !llvm.ptr)', '@kernel(%p: !llvm.ptr {llvm.noundef})')
    # Valid LLVM annotation, still excluded by the normalized plain-pointer ABI.
    parse_llvm_module(text)
    with pytest.raises(ValueError, match='selected ordinary C pointer ABI'):
        require_pointer_entry(text, entry_symbol='kernel', pointer_arity=1)
    # This owner reports arity only, exactly as before this parser integration.
    assert emitted_entry_arity(text, entry_symbol='kernel') == 1


def test_real_native_entry_readers_fail_closed_on_malformed_native_input(native_tool):
    from merlin.targetgen.contract.compile_only import require_pointer_entry
    from merlin.targetgen.bundle_harness import emitted_entry_arity
    text = NATIVE_POINTER_ENTRY.replace('-> !llvm.ptr, i8', '-> !llvm.ptr, invalid_type')
    with pytest.raises(ValueError, match='malformed LLVM pointer ABI IR'):
        require_pointer_entry(text, entry_symbol='kernel', pointer_arity=1)
    assert emitted_entry_arity(text, entry_symbol='kernel') is None


@pytest.mark.parametrize('install,compiler,expected', [
    ('/selected-install', '/other/bin/clang-23', '/selected-install/bin/mlir-opt'),
    (None, '/supplied/bin/clang-23', '/supplied/bin/mlir-opt'),
])
def test_native_locator_uses_only_explicit_selected_standard_toolchain(monkeypatch, install, compiler, expected):
    from merlin.targetgen.oot_starterkit.llvm_context import _native_mlir_opt
    monkeypatch.delenv('MERLIN_MLIR_INSTALL', raising=False)
    monkeypatch.delenv('MERLIN_CLANG', raising=False)
    if install is not None: monkeypatch.setenv('MERLIN_MLIR_INSTALL', install)
    if compiler is not None: monkeypatch.setenv('MERLIN_CLANG', compiler)
    assert _native_mlir_opt() == Path(expected)


@pytest.mark.parametrize('install,compiler', [
    (None, None), ('relative-install', '/supplied/bin/clang-23'), (None, 'clang-23'),
])
def test_native_locator_never_falls_back_from_missing_or_relative_selection(monkeypatch, install, compiler):
    from merlin.targetgen.oot_starterkit.llvm_context import _native_mlir_opt
    monkeypatch.delenv('MERLIN_MLIR_INSTALL', raising=False)
    monkeypatch.delenv('MERLIN_CLANG', raising=False)
    if install is not None: monkeypatch.setenv('MERLIN_MLIR_INSTALL', install)
    if compiler is not None: monkeypatch.setenv('MERLIN_CLANG', compiler)
    with pytest.raises(ValueError): _native_mlir_opt()


def test_missing_explicit_native_selection_preserves_first_parser_diagnostic(monkeypatch):
    monkeypatch.delenv('MERLIN_MLIR_INSTALL', raising=False)
    monkeypatch.delenv('MERLIN_CLANG', raising=False)
    with pytest.raises(ValueError, match='Original xDSL syntax diagnostic') as raised:
        parse_llvm_module(NATIVE_CUSTOM)
    assert isinstance(raised.value.__cause__, ParseError)
