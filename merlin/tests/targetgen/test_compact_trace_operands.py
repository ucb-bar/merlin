"""Compact-loop diagnostics do not fabricate dynamic operands or relax hazards."""

import pytest

from merlin.targetgen import trace_check
from merlin.targetgen.rocc.decode import _parse_module, _resolve
from merlin.targetgen.rocc.static_operands import counted_loop_observations


def _value(body):
    module = _parse_module('module { llvm.func @entry() {\n' + body + '\nllvm.return\n} }')
    assert module is not None
    return list(module.body.block.first_op.body.block.ops)[-2].results[0]


def test_packed_constant_arithmetic_is_fixed_width():
    value = _value('''%a = llvm.mlir.constant(16 : i64) : i64
%b = llvm.mlir.constant(32 : i64) : i64
%c = llvm.shl %a, %b : i64
%d = llvm.mlir.constant(4096 : i64) : i64
%e = llvm.or %c, %d : i64''')
    assert _resolve(value).value == (16 << 32) | 4096
    value = _value('''%a = llvm.mlir.constant(127 : i8) : i8
%b = llvm.mlir.constant(1 : i8) : i8
%c = llvm.add %a, %b : i8''')
    assert _resolve(value).value == -128


@pytest.mark.parametrize('expression', ['llvm.shl %a, %b : i8', 'llvm.add %a, %b overflow<nsw> : i8'])
def test_poison_or_out_of_range_shift_remains_unknown(expression):
    value = _value('%a = llvm.mlir.constant(127 : i8) : i8\n%b = llvm.mlir.constant(8 : i8) : i8\n%c = ' + expression)
    assert _resolve(value).kind == 'unknown'


LOOP = '''module { llvm.func @entry() {
 %zero = llvm.mlir.constant(0 : i64) : i64
 %one = llvm.mlir.constant(1 : i64) : i64
 %limit = llvm.mlir.constant(LIMIT : i64) : i64
 llvm.br ^head(%zero : i64)
^head(%iv : i64):
 %test = llvm.icmp "slt" %iv, %limit : i64
 llvm.cond_br %test, ^body, ^exit
^body:
 %packed = llvm.mul %iv, %limit : i64
 %next = llvm.add %iv, %one : i64
 llvm.br ^head(%next : i64)
^exit:
 llvm.return
} }'''


@pytest.mark.parametrize('trips', [0, 1, 1000000000])
def test_loop_observations_are_compact_and_only_singleton_body_resolves(trips):
    module = _parse_module(LOOP.replace('LIMIT', str(trips)))
    observation, bindings = counted_loop_observations(module)
    blocks = tuple(module.body.block.first_op.body.blocks)
    header, body, exit_block = blocks[1:]
    assert observation['functions'][0]['loops'][0]['trip_count'] == trips
    assert len(observation['functions'][0]['loops']) == 1
    operand = body.first_op.results[0]
    assert _resolve(operand, _bindings=bindings.get(body)).kind == ('const' if trips == 1 else 'unknown')
    assert _resolve(header.args[0], _bindings=bindings.get(header)).kind == 'unknown'
    assert exit_block not in bindings


def test_only_fences_can_intervene_in_preload_pairing():
    def check(middle):
        classes = ['FENCE', 'PRELOAD', *middle, 'COMPUTE_PRELOADED', 'FENCE']
        return trace_check.check({'instructions': [{'class': c, 'decoded': {}} for c in classes]}, {})
    assert check(['FENCE', 'FENCE'])['status'] == 'pass'
    assert check(['MVIN'])['status'] == 'fail'
    assert check(['CONFIG_EX'])['status'] == 'fail'



def test_native_decode_publishes_compact_observation(monkeypatch):
    from types import SimpleNamespace
    from merlin.targetgen.rocc import decode
    monkeypatch.setattr(decode, '_semantics', lambda target: SimpleNamespace(
        word_bits=64, isa_constants=lambda target: {'CUSTOM_OPCODE': 123, 'FUNCT3': 3},
        decode_instruction=lambda funct, rs1, rs2, isa: ('WORK', {'raw_operand': rs1['raw']})))
    source = LOOP.replace('LIMIT', '1').replace('%next =',
        '\n"llvm.inline_asm"(%packed, %packed) <{asm_string = ".insn r 123, 3, 2, x0, $0, $1", '
        'constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()\n%next =')
    result = decode.decode_text(source, target='synthetic')
    assert len(result['instructions']) == 1
    assert result['instructions'][0]['rs1']['raw'] == 0
    assert result['summary']['counted_loops']['instruction_multiplicity'] == 'static_sites_only'
