"""The operator data adapter binds sources and never upgrades raw encodings to semantics."""
import importlib.util
import json
from types import SimpleNamespace

import pytest

from merlin.targetgen.isa_spec import IsaSpecError, resolve_isa
from merlin.targetgen.isa_dialect_gen import encode_instruction
from merlin.common.paths import repo_root

pytestmark = pytest.mark.target('gemmini')

ROOT = repo_root()
SPEC = importlib.util.spec_from_file_location('public_isa_prepare', ROOT / 'examples/gemmini/isa-preparation/prepare.py')
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def test_declaration_preserves_unknown_semantics_and_separates_payload_from_register_width():
    fields = {name: {'offset': offset, 'width': width} for name, offset, width in (
        ('opcode', 0, 7), ('rd', 7, 5), ('xs2', 12, 1), ('xs1', 13, 1), ('xd', 14, 1),
        ('rs1', 15, 5), ('rs2', 20, 5), ('funct', 25, 7))}
    facts = {'facts': {'target': 'gemmini', 'command_encoding': {
        'word_bits': 32, 'payload_bits': 64, 'fields': fields,
        'fixed': {'opcode': 123, 'rd': 0, 'xd': 0, 'xs1': 1, 'xs2': 1},
        'selectors': {'load_cmd': 2}}}}
    declaration = PREP.declaration('a' * 64, ['load_cmd'])
    model = resolve_isa(declaration, facts, target='gemmini', facts_sha256='a' * 64)
    ins = model.instructions[0]
    assert all(arg.type.bits == 64 and arg.unit == 'register_value' for arg in ins.operands)
    assert json.loads(ins.effects_json) == json.loads(ins.semantics_json) == {'status': 'unknown'}
    assert encode_instruction(model, 'load_cmd', {'rs1': 31, 'rs2': 30}) == 0x05EFB07B
    with pytest.raises(IsaSpecError, match='outside range'):
        encode_instruction(model, 'load_cmd', {'rs1': 32, 'rs2': 0})


def test_source_pin_and_existing_product_changes_are_rejected(tmp_path):
    path = tmp_path / 'source'
    path.write_text('public hardware')
    sha = PREP.digest(path)
    assert PREP.pinned(path, sha) == path
    path.write_text('different hardware')
    with pytest.raises(ValueError, match='source bytes differ'):
        PREP.pinned(path, sha)
    product = tmp_path / 'facts.json'
    PREP.publish(product, {'fact': 1})
    PREP.publish(product, {'fact': 1})
    with pytest.raises(ValueError, match='refusing to replace'):
        PREP.publish(product, {'fact': 2})


def test_foreign_operand_protocol_is_rejected_before_any_generation(tmp_path):
    def put(name, value):
        path = tmp_path / name
        path.write_text(value)
        return path
    isa = put('isa.scala', 'public declaration')
    fir = put('source.fir', 'public hardware')
    header = put('rocc.h', 'public encoding')
    original = put('facts.json', json.dumps({'facts': {'target': 'gemmini',
        'source': {'fir_sha256': PREP.digest(fir)}, 'interfaces': [{
            'name': 'funct_decode_table', 'source_declared_names_provenance': {
                'source_sha256': PREP.digest(isa)}}]}}))
    protocol = put('protocol.json', json.dumps({'target': 'gemmini', 'original_facts_sha256': '0' * 64}))
    args = SimpleNamespace(facts=original, facts_sha256=PREP.digest(original), isa_source=isa,
        firrtl=fir, rocc_header=header, rocc_header_sha256=PREP.digest(header),
        operand_protocol=protocol, operand_protocol_sha256=PREP.digest(protocol), out=tmp_path / 'out')
    with pytest.raises(ValueError, match='another original facts'):
        PREP.prepare(args)
    assert not args.out.exists()
