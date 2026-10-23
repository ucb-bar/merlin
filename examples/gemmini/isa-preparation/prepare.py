#!/usr/bin/env python3
"""Prepare public command-encoding data; never import or seed a target backend."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction, generate_package
from merlin.targetgen.isa_spec import resolve_isa
from merlin.targetgen.rtl.circt_introspect import extract_funct_table
from merlin.targetgen.rtl.firrtl_operand_slices import derive_operand_layouts

FIELDS = ('opcode', 'rd', 'xs2', 'xs1', 'xd', 'rs1', 'rs2', 'funct')
# An explicit initial coverage choice, not an inferred complete ISA.
BASIC = (0, 1, 2, 3, 4, 5, 6, 14)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pinned(path, expected):
    path = Path(path).resolve(strict=True)
    if digest(path) != expected:
        raise ValueError(f'source bytes differ from declared pin: {path}')
    return path


def publish(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f'refusing to replace different prepared data: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('xb') as stream:
            stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def ref(*path):
    return {'fact': ['facts', 'command_encoding', *path]}


def declaration(facts_sha, instructions):
    rows = []
    for name in instructions:
        encoding = []
        for field in FIELDS:
            row = {'offset': ref('fields', field, 'offset'),
                   'width': ref('fields', field, 'width')}
            if field in ('rs1', 'rs2'):
                row['argument'] = field
            elif field == 'funct':
                row['fixed'] = ref('selectors', name)
            else:
                row['fixed'] = ref('fixed', field)
            encoding.append(row)
        rows.append({'name': name, 'operands': [
            {'name': arg, 'type': {'kind': 'integer', 'bits': ref('payload_bits'), 'signed': False},
             'unit': 'register_value'} for arg in ('rs1', 'rs2')],
            'results': [], 'immediates': [], 'encoding': encoding,
            'semantics': None, 'effects': None})
    return {'schema': 'merlin.isa_spec.v1', 'namespace': 'gemmini_public_command',
            'facts_sha256': facts_sha, 'word_bits': ref('word_bits'),
            'resources': [], 'instructions': rows}


def check_generated(model, generated, header, out):
    """Independent public C macro comparison, plus typed SSA -> encoded register bindings."""
    from xdsl.dialects.builtin import IntegerType
    from xdsl.ir import Block
    from xdsl.traits import MemoryEffect

    spec = importlib.util.spec_from_file_location('prepared_command_check', generated / 'instruction_dialect.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Register numbers are allocated separately from the two i64 SSA payloads.
    pairs = ((0, 0), (11, 12), (31, 30))
    vectors, expressions = [], []
    for ins in model.instructions:
        selector = next(f.fixed for f in ins.fields if f.offset == 25)
        for lhs, rhs in pairs:
            word = encode_instruction(model, ins.name, {'rs1': lhs, 'rs2': rhs})
            if decode_instruction(model, word) != (ins.name, {'rs1': lhs, 'rs2': rhs}):
                raise ValueError('generated codec roundtrip differs')
            block = Block(arg_types=[IntegerType(64), IntegerType(64)])
            op = module.create(ins.name, operands=dict(zip(('rs1', 'rs2'), block.args)), immediates={})
            registers = {block.args[0]: lhs, block.args[1]: rhs}
            encoded = module.encode_op(op, allocate=lambda value, argument: registers[value])
            if encoded != word or op.get_trait(MemoryEffect).get_effects(op) is not None:
                raise ValueError('typed SSA encoding/effect qualification changed')
            vectors.append({'instruction': ins.name, 'funct': selector, 'rs1': lhs, 'rs2': rhs, 'word': word})
            expressions.append(f'printf("%u\\n", (unsigned) CUSTOMX(3, 0, 1, 1, 0, {lhs}, {rhs}, {selector}));')
    check_dir = out / 'checks'
    check_dir.mkdir(exist_ok=True)
    c_source = '#include <stdio.h>\n#include ' + json.dumps(str(header)) + '\nint main(void) {\n' + '\n'.join(expressions) + '\nreturn 0;\n}\n'
    c_path = check_dir / 'public_macro.c'
    if c_path.exists() and c_path.read_text() != c_source:
        raise ValueError('different independent check already exists')
    c_path.write_text(c_source)
    compiler = shutil.which('cc')
    if compiler is None:
        raise ValueError('host cc is required for independent public macro comparison')
    executable = check_dir / 'public_macro'
    subprocess.run([compiler, '-std=c11', '-O0', str(c_path), '-o', str(executable)], check=True, capture_output=True)
    observed = [int(line) for line in subprocess.check_output([str(executable)], text=True).splitlines()]
    if observed != [row['word'] for row in vectors]:
        raise ValueError('public RoCC header and generated encoding disagree')
    return {'kind': 'public_header_encoding_and_typed_ssa_check', 'vectors': vectors,
            'passed': len(vectors), 'reference_header_sha256': digest(header),
            'reference_source_sha256': digest(c_path), 'host_compiler': compiler,
            'target_execution': False, 'semantic_or_effect_qualification': False}


def prepare(args):
    facts_path = pinned(args.facts, args.facts_sha256)
    original = json.loads(facts_path.read_text())
    facts = original['facts']
    if facts['target'] != 'gemmini':
        raise ValueError('this explicit target example requires gemmini facts')
    table = next(row for row in facts['interfaces'] if row['name'] == 'funct_decode_table')
    provenance = table['source_declared_names_provenance']
    isa_path = pinned(args.isa_source, provenance['source_sha256'])
    fir_path = pinned(args.firrtl, facts['source']['fir_sha256'])
    header = pinned(args.rocc_header, args.rocc_header_sha256)
    protocol_path = pinned(args.operand_protocol, args.operand_protocol_sha256)
    protocol = json.loads(protocol_path.read_text())
    if protocol['original_facts_sha256'] != args.facts_sha256 or protocol['target'] != 'gemmini':
        raise ValueError('operand protocol belongs to another original facts selection')
    if protocol['layouts']['word_bits'] != 64 or table['custom_opcode'] != 123 or table['funct3'] != 3:
        raise ValueError('selected input is outside the reviewed no-result command form')
    # Require the public header's explicit SS form, rather than infer effects from command names.
    if 'CUSTOMX(X, 0, 1, 1, rd, rs1_n, rs2_n, funct)' not in header.read_text():
        raise ValueError('public header no longer declares the selected SS form')
    extracted = extract_funct_table(isa_path.read_text(), start_comment=provenance['start_comment'],
                                    stop_declaration=provenance['stop_declaration'])
    selectors = {}
    for code in BASIC:
        name = extracted['names'].get(str(code))
        if code not in table['legal_funct'] or name != table['names'].get(str(code)):
            raise ValueError(f'public source and selected observed decoder disagree on {code}')
        selectors[name.lower()] = code
    selections = [{'bundle': 'RoCCInstruction', 'name': field, 'module': 'Rocket',
                   'input': '_io_rocc_cmd_bits_inst_WIRE_1',
                   'field': '_io_rocc_cmd_bits_inst_WIRE.' + field} for field in FIELDS]
    layout = derive_operand_layouts(fir_path, expected_sha256=facts['source']['fir_sha256'],
                                    selections=selections, word_bits=32)
    fields = {row['name']: {'offset': row['offset'], 'width': row['width']} for row in layout['fields']}
    source_rows = [
        {'role': role, 'path': str(path), 'sha256': digest(path)} for role, path in
        [('original_facts', facts_path), ('original_firrtl', fir_path), ('public_isa_source', isa_path),
         ('public_rocc_header', header), ('operand_protocol', protocol_path)]]
    normalized = {'schema': 'merlin.public_command_isa_facts.v1',
        'facts': {'target': 'gemmini', 'command_encoding': {
            'word_bits': layout['word_bits'], 'payload_bits': protocol['layouts']['word_bits'],
            'fields': fields, 'fixed': {'opcode': table['custom_opcode'], 'rd': 0, 'xd': 0, 'xs1': 1, 'xs2': 1},
            'selectors': selectors}},
        'provenance': {'sources': source_rows, 'outer_layout': layout, 'selector_extraction': extracted,
            'selector_source_lines': [number for number, line in enumerate(isa_path.read_text().splitlines(), 1)
                if any(line.strip().startswith('val ' + name + ' =') for name in extracted['names'].values())],
            'rocc_form_source_lines': [number for number, line in enumerate(header.read_text().splitlines(), 1)
                if 'CUSTOMX(X, 0, 1, 1, rd, rs1_n, rs2_n, funct)' in line
                or '#define ROCC_INSTRUCTION_SS(' in line
                or 'ROCC_INSTRUCTION_I_R_R(X, 0, rs1, rs2, funct, 11, 12)' in line],
            'fixed_form': 'Public ROCC_INSTRUCTION_I_R_R: xd=0, xs1=1, xs2=1; rd=x0 as ROCC_INSTRUCTION_SS.',
            'payload_width': 'Selected operand-protocol layouts.word_bits; no payload packing is generated.'},
        'assurance': {'encoding_only': True, 'complete_isa': False, 'command_acceptance': 'unproven',
            'semantics': 'unknown', 'effects': 'unknown', 'packing': 'not_generated',
            'observed_but_undeclared': sorted(set(table['legal_funct']) - set(extracted['legal_funct'])),
            'declared_but_unobserved': sorted(set(extracted['legal_funct']) - set(table['legal_funct'])),
            'excluded': [{'selector': code, 'reason': 'outside the explicit basic no-result command subset'}
                         for code in sorted(set(table['legal_funct']) | set(extracted['legal_funct'])) if code not in BASIC]}}
    out = Path(args.out).resolve()
    sha = publish(out / 'normalized-facts.json', normalized)
    spec = declaration(sha, selectors)
    publish(out / 'isa-declaration.json', spec)
    model = resolve_isa(spec, normalized, target='gemmini', facts_sha256=sha)
    generated = out / 'generation-check-only'
    generate_package(model, generated)
    report = check_generated(model, generated, header, out)
    report.update({'facts_sha256': sha, 'model_sha256': model.identity(), 'command_count': len(selectors),
                   'generated_scaffold_is_author_seed': False, 'historical_facts_sha256': args.facts_sha256,
                   'preparation_script_sha256': digest(__file__)})
    publish(out / 'preparation-check.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('facts', 'facts-sha256', 'firrtl', 'isa-source', 'rocc-header', 'rocc-header-sha256',
                 'operand-protocol', 'operand-protocol-sha256', 'out'):
        parser.add_argument('--' + name, required=True)
    report = prepare(parser.parse_args())
    print(json.dumps({key: report[key] for key in ('command_count', 'passed', 'facts_sha256', 'model_sha256')}))


if __name__ == '__main__':
    main()
