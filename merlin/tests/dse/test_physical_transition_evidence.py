"""Actual typed LLVM copy evidence, independent of candidate packages or targets."""
import hashlib
import json
from io import StringIO

import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i16, i64, f64
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from merlin.perf.physical_transition_evidence import verify_physical_transitions
from merlin.perf.structural_transitions import StaticStridedLayout, strided_copy_transition


def fixture(dtype=i16, stride=2):
    width = dtype.bitwidth // 8
    source = f'''builtin.module {{
      func.func @source() -> tensor<3x{dtype}> {{
        %a = arith.constant dense<0> : tensor<3x{dtype}>
        %b = tensor.cast %a : tensor<3x{dtype}> to tensor<3x{dtype}>
        func.return %b : tensor<3x{dtype}>
      }}
    }}'''
    entry, head, body, end = Block(), Block(arg_types=[i64]), Block(), Block()
    region = Region([entry, head, body, end])
    def constant(n):
        op = llvm.ConstantOp(IntegerAttr(n, i64), i64)
        entry.add_op(op)
        return op.results[0]
    zero, one, limit, step, size = [constant(n) for n in (0, 1, 3, stride, 3*stride)]
    src, dst = llvm.AllocaOp(limit, dtype), llvm.AllocaOp(size, dtype)
    src.attributes['merlin.transition_source'] = StringAttr('copy')
    dst.attributes.update({'merlin.transition_buffer': StringAttr('copy'),
                           'merlin.global_transition': StringAttr('copy')})
    entry.add_ops([src, dst, llvm.BrOp(head, zero)])
    condition = llvm.ICmpOp(head.args[0], limit, IntegerAttr(2, i64))
    head.add_ops([condition, llvm.CondBrOp(condition, body, [], end, [])])
    a = llvm.GEPOp.from_mixed_indices(src, [head.args[0]], dtype)
    address = llvm.MulOp(head.args[0], step)
    b = llvm.GEPOp.from_mixed_indices(dst, [address], dtype)
    load = llvm.LoadOp(a, dtype)
    store = llvm.StoreOp(load.results[0], b)
    update = llvm.AddOp(head.args[0], one)
    body.add_ops([a, address, b, load, store, update, llvm.BrOp(head, update)])
    for block in (head, body):
        for op in block.ops:
            op.attributes['merlin.global_transition'] = StringAttr('copy')
    read = llvm.LoadOp(dst, dtype)
    read.attributes['merlin.structural_region'] = StringAttr('consumer')
    end.add_ops([read, llvm.ReturnOp()])
    module = ModuleOp([llvm.FuncOp('kernel', llvm.LLVMFunctionType([]), body=region)])
    transition = strided_copy_transition(id='copy', buffer='a', producer='producer', consumer='consumer',
        source_layout=StaticStridedLayout((3,), (1,), 3*width),
        destination_layout=StaticStridedLayout((3,), (stride,), 3*stride*width),
        dtype=str(dtype), placement='memory', provenance=('test address functions',)).to_dict()
    transition['source_edge'] = dict(producer_op_index=0, producer_result_index=0,
        consumer_op_index=1, consumer_operand_index=0)
    cb = {'params': {'global_program_plan': {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                                           'physical_transitions': [transition]}}}
    return source, module, cb, dict(src=src, dst=dst, load=load, store=store, read=read, a=a, b=b,
                                   address=address, entry=entry, end=end, head=head)


def verify(data):
    source, module, cb, _ = data
    stream = StringIO()
    Printer(stream=stream).print_op(module)
    return verify_physical_transitions(source_text=source, lowered_text=stream.getvalue(), command_buffer=cb)


@pytest.mark.parametrize('dtype,stride', [(i16, 2), (f64, 3)])
def test_typed_padding_copy_counts_actual_bytes(dtype, stride):
    result = verify(fixture(dtype, stride))
    assert result['status'] == 'verified', result
    row = result['transitions'][0]
    assert row['load_payload_bytes'] == row['store_payload_bytes'] == 3*dtype.bitwidth//8
    assert row['physical_bytes'] == 6*dtype.bitwidth//8
    assert row['materialized'] is True
    assert row['execution_multiplicity_verified'] is True
    assert row['destination_storage_bytes'] == 3*stride*dtype.bitwidth//8
    assert row['bit_preserving_copy']
    assert result['cycles'] is result['dram_bytes'] is None
    assert len(result['command_buffer_sha256']) == 64
    assert result['encoding_activity'] == {
        'status': 'verified',
        'executed_transition_count': 1,
        'materialized_transition_count': 1,
        'physical_read_bytes': 3*dtype.bitwidth//8,
        'physical_write_bytes': 3*dtype.bitwidth//8,
        'physical_bytes': 6*dtype.bitwidth//8,
        'basis': ('exact emitted CFG multiplicity, typed load/store dataflow and proved physical '
                  'address functions'),
    }
    content_address = result.pop('receipt_sha256')
    canonical = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert content_address == hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize('mutation', ['source_edge', 'charge', 'consumer_missing', 'consumer_wrong',
    'same_allocation', 'wrong_stride', 'wrong_gep_type', 'wrong_store', 'extra_writer', 'early_consumer'])
def test_contradictory_address_or_dataflow_refuses(mutation):
    data = fixture()
    _, _, cb, ops = data
    transition = cb['params']['global_program_plan']['physical_transitions'][0]
    if mutation == 'source_edge':
        transition['source_edge']['consumer_op_index'] = 0
    elif mutation == 'charge':
        transition['quantities'][0]['amount'] = 0
    elif mutation == 'consumer_missing':
        del transition['consumer']
        del ops['read'].attributes['merlin.structural_region']
    elif mutation == 'consumer_wrong':
        ops['read'].attributes['merlin.structural_region'] = StringAttr('unrelated')
    elif mutation == 'same_allocation':
        ops['b'].operands = [ops['src'].results[0], ops['address'].results[0]]
    elif mutation == 'wrong_stride':
        ops['b'].operands = [ops['dst'].results[0], ops['head'].args[0]]
    elif mutation == 'wrong_gep_type':
        ops['b'].properties['elem_type'] = i64
    elif mutation == 'wrong_store':
        ops['store'].operands = [ops['read'].results[0], ops['b'].results[0]]
    elif mutation == 'extra_writer':
        ops['end'].insert_op_before(llvm.StoreOp(ops['read'].results[0], ops['dst']), ops['end'].last_op)
    elif mutation == 'early_consumer':
        ops['end'].detach_op(ops['read'])
        ops['entry'].insert_op_before(ops['read'], ops['entry'].last_op)
    result = verify(data)
    assert result['status'] == 'refused', result


def test_unsupported_mechanism_is_unknown_not_verified():
    data = fixture()
    data[2]['params']['global_program_plan']['physical_transitions'][0]['kind'] = 'dma_copy'
    assert verify(data)['status'] == 'UNKNOWN'


@pytest.mark.parametrize('marker', ['merlin.global_transition', 'merlin.transition_source', 'merlin.transition_buffer'])
def test_orphan_marker_refuses_even_when_list_is_empty(marker):
    data = fixture()
    for op in data[1].walk():
        for key in tuple(op.attributes):
            if key.startswith('merlin.'):
                del op.attributes[key]
    data[3]['src'].attributes[marker] = StringAttr('copy')
    data[2]['params']['global_program_plan']['physical_transitions'] = []
    assert verify(data)['status'] == 'refused'


def test_no_markers_and_no_declarations_is_not_declared():
    data = fixture()
    for op in data[1].walk():
        op.attributes.clear()
    data[2]['params']['global_program_plan']['physical_transitions'] = []
    result = verify(data)
    assert result['status'] == 'not_declared'
    assert result['encoding_activity']['status'] == 'UNKNOWN'
    assert result['encoding_activity']['executed_transition_count'] is None
    assert result['encoding_activity']['physical_bytes'] is None
    assert len(result['receipt_sha256']) == 64
