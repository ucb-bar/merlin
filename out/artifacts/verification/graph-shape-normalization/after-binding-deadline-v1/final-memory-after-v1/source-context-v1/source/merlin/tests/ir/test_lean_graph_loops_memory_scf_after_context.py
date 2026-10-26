"""New wrapper test extension; independent actual raw decoding, SDK-free only."""

import hashlib
import io
import json
import os
from pathlib import Path

import pytest
from xdsl.dialects import builtin, func, memref
from xdsl.printer import Printer

from merlin.verify.lean_graph_loops_memory_scf_after_context import prepare_context
from merlin.verify.lean_graph_loops_shape_projection import Decoder
from test_lean_graph_loops_memory_scf_after_whole import parser

WRAPPER = '''func.func @wrapper(%a: i32, %b: i32, %caller: memref<1xi32>, %bias: i32) -> (i32, index) {
 %value, %extent = func.call @kernel(%a, %b, %caller) : (i32, i32, memref<1xi32>) -> (i32, index)
 %continued = arith.addi %value, %bias : i32
 %zero = arith.constant 0 : index
 %old = memref.load %caller[%zero] : memref<1xi32>
 %observed = arith.addi %old, %continued : i32
 memref.store %observed, %caller[%zero] : memref<1xi32>
 func.return %observed, %extent : i32, index
}'''


def _append(module, text):
    temporary = parser('builtin.module {\n' + text + '\n}')
    for operation in list(temporary.body.block.ops):
        temporary.body.block.detach_op(operation)
        module.body.block.add_op(operation)


def _print(module):
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return stream.getvalue()


def _variant(case):
    if case == "argument_alias":
        return WRAPPER.replace('@kernel(%a, %b, %caller)', '@kernel(%a, %a, %caller)')
    if case == "unknown_caller":
        return ('func.func private @unknown(i32, i32, memref<1xi32>) -> (i32, index)\n'
                + WRAPPER.replace('@kernel(%a, %b, %caller)', '@unknown(%a, %b, %caller)'))
    if case == "repeated":
        return WRAPPER.replace(' %continued =', ' %again, %other = func.call @kernel(%a, %b, %caller) '
                               ': (i32, i32, memref<1xi32>) -> (i32, index)\n %continued =')
    if case == "prefix_allocation":
        return WRAPPER.replace(' %value,', ' %private = memref.alloc() : memref<1xi32>\n %value,', 1)
    if case == "private_escape":
        return WRAPPER.replace('-> (i32, index) {', '-> memref<1xi32> {', 1).replace(
            ' func.return %observed, %extent : i32, index',
            ' %private = memref.alloc() : memref<1xi32>\n func.return %private : memref<1xi32>')
    if case == "transitive":
        inner = WRAPPER.replace('@wrapper(', '@inner(')
        return inner + '\nfunc.func @wrapper(%a: i32, %b: i32, %caller: memref<1xi32>, %bias: i32) '
        # The complete transitive wrapper is assembled below to keep this fixture explicit.
    return WRAPPER


@pytest.mark.parametrize("case", (
    "positive", "argument", "return", "order", "observer", "changed_callee",
    "repeated", "prefix_allocation", "private_escape", "transitive", "argument_alias", "caller_buffer", "unknown_caller", "descriptor_alias",
))
def test_actual_wrapper_context(case, monkeypatch, tmp_path):
    selected = {key: os.environ.get(key) for key in (
        "MERLIN_GRAPH_MEMORY_AFTER_SOURCE", "MERLIN_GRAPH_MEMORY_AFTER_ACTUAL", "MERLIN_GRAPH_MEMORY_AFTER_HINT",
    )}
    if not all(selected.values()):
        pytest.skip("select retained original raw modules and existing untrusted plan")
    base_source = Path(selected["MERLIN_GRAPH_MEMORY_AFTER_SOURCE"]).read_text()
    base_final = Path(selected["MERLIN_GRAPH_MEMORY_AFTER_ACTUAL"]).read_text()
    hint = json.loads(Path(selected["MERLIN_GRAPH_MEMORY_AFTER_HINT"]).read_text())
    source, final = parser(base_source), parser(base_final)
    extension = _variant(case)
    if case == "transitive":
        extension += ('-> (i32, index) { %value, %extent = func.call @inner(%a, %b, %caller, %bias) '
                      ': (i32, i32, memref<1xi32>, i32) -> (i32, index) '
                      'func.return %value, %extent : i32, index }')
    source_extension = extension
    if case == "descriptor_alias":
        source_extension = extension.replace(' %value,', ' %alias = "memref.cast"(%caller) : (memref<1xi32>) -> memref<1xi32>\n %value,', 1)
        source_extension = source_extension.replace('@kernel(%a, %b, %caller)', '@kernel(%a, %b, %alias)')
    _append(source, source_extension)
    _append(final, extension)
    wrapper = next(op for op in final.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "wrapper")
    if case == "argument":
        call = next(op for op in wrapper.body.block.ops if isinstance(op, func.CallOp))
        call.operands = (call.operands[1], call.operands[0], call.operands[2])
    elif case == "caller_buffer":
        call = next(op for op in wrapper.body.block.ops if isinstance(op, func.CallOp))
        allocation = memref.AllocOp.get(builtin.i32, shape=[1])
        wrapper.body.block.insert_op_before(allocation, call)
        call.operands = (*call.operands[:2], allocation.memref)
    elif case == "return":
        returned = wrapper.body.block.last_op
        returned.operands = (wrapper.body.block.args[3], returned.operands[1])
    elif case == "observer":
        store = next(op for op in wrapper.body.block.ops if isinstance(op, memref.StoreOp))
        store.operands = (wrapper.body.block.args[3], *store.operands[1:])
    elif case == "order":
        load = next(op for op in wrapper.body.block.ops if isinstance(op, memref.LoadOp))
        store = memref.StoreOp.get(wrapper.body.block.args[3], wrapper.body.block.args[2], load.indices)
        wrapper.body.block.insert_op_before(store, load)
    elif case == "changed_callee":
        kernel = next(op for op in final.body.block.ops if isinstance(op, func.FuncOp) and op.sym_name.data == "kernel")
        returned = kernel.body.block.last_op
        returned.operands = (kernel.body.block.args[0], returned.operands[1])
    raw_source, raw_final = _print(source), _print(final)
    # Preserve raw inputs before preparation so source-domain refusals retain their complete programs too.
    (tmp_path / 'source.mlir').write_text(raw_source)
    (tmp_path / 'final.mlir').write_text(raw_final)

    def forbidden(*args, **kwargs):
        raise AssertionError('old whole-tree pair matcher cannot provide a context proof')

    for name in ('block', 'project', 'operands'):
        monkeypatch.setattr(Decoder, name, forbidden)
    result = prepare_context(hint['adapter'], raw_source, raw_final, parser=parser, index_bits=64, max_nodes=5000)
    assert result['lean_status'] == 'UNCOMPILED' and not result['kernel_checked'] and not result['qualified']
    assert not result['whole_world_equality_assumed'] and not result['context_proved']
    assert not result['source_extension_is_original_capture'] and not result['repeated_kernel_calls_supported']
    assert result['query_sha256'] == hashlib.sha256(result['query'].encode()).hexdigest()
    assert result['raw_source_sha256'] == hashlib.sha256(raw_source.encode()).hexdigest()
    assert result['raw_final_sha256'] == hashlib.sha256(raw_final.encode()).hexdigest()
    assert result['final_root'] not in result['support_roots']
    assert not any(line.lstrip().startswith('#') for line in result['query'].splitlines())
    unsupported = case in ('repeated', 'prefix_allocation', 'private_escape', 'transitive', 'unknown_caller')
    assert bool(result['context']['refusals']) == unsupported
    if case == 'positive':
        assert len(result['context']['rows']) == 4
        assert result['context']['rows'][-1] == dict(name='wrapper', route='caller', wrapper=0)
        assert len(result['context']['wrappers']) == 1
        assert result['witness']['status'] == 'UNCOMPILED'
        assert result['witness']['wrapper_returns'] == [498, 2]
        assert not result['witness']['source_success_assumed']
        assert 'GraphMemoryAfterContextWitness.final_observation' in result['support_roots']
    if case in ('argument_alias', 'descriptor_alias'):
        assert result['witness']['status'] == 'NOT_EMITTED'
        assert 'GraphMemoryAfterContext.wrapper0_sample_domain' in result['support_roots']
    (tmp_path / 'Context.lean').write_text(result['query'])
    (tmp_path / 'PREPARATION.json').write_text(json.dumps(dict(
        case=case, xdsl_verified=True, kernel_outcome='UNOBSERVED',
        source_level_wrapper_extension=True, original_capture=False,
        expected_finite_binding_acceptance=case in ('positive', 'argument_alias', 'descriptor_alias'), semantic_counterexample_proved=False,
        base_source_sha256=hashlib.sha256(base_source.encode()).hexdigest(),
        base_final_sha256=hashlib.sha256(base_final.encode()).hexdigest(),
        prepared={key:value for key,value in result.items() if key!='query'},
    ), indent=2)+'\n')
