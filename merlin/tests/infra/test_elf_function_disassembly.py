"""Whole original function coverage, including local labels and padding."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.compile.analysis.artifact_reader import PinnedELF, function_listing_coverage, open_ordinary
from merlin.compile.analysis.disassembly import RawInsn, objdump_bin, tokenize_text


def test_function_coverage_rejects_prefix_missing_padding_and_changed_bytes():
    extent = dict(address=0, size_bytes=6, section='.text.selected')
    rows = [RawInsn(0, 'addi', [], '0001', section_name='.text.selected'),
            RawInsn(2, 'ret', [], '8082', section_name='.text.selected'),
            RawInsn(4, 'unimp', [], '0000', section_name='.text.selected')]
    original = bytes.fromhex('010082800000')
    assert function_listing_coverage(rows, extent, original)['covered_bytes'] == 6
    for prefix in (rows[:1], rows[:2]):
        with pytest.raises(ValueError, match='listing_incomplete'):
            function_listing_coverage(prefix, extent, original)
    with pytest.raises(ValueError, match='listing_bytes_differ'):
        function_listing_coverage(rows, extent, bytes.fromhex('010082800100'))
    with pytest.raises(ValueError, match='listing_incomplete'):
        function_listing_coverage(rows + rows, extent, original)


def test_function_disassembly_crosses_local_relocation_labels(tmp_path, monkeypatch):
    clang = os.environ.get('MERLIN_CLANG') or shutil.which('clang')
    if not clang or Path(objdump_bin()).name == 'objdump':
        pytest.skip('RISC-V clang and llvm-objdump required')
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    source, output = tmp_path/'selected.S', tmp_path/'selected.o'
    source.write_text('''
.section .text.selected,"ax",@progbits
.globl selected
.type selected,@function
selected:
  .insn 4, 0x0200307b
.Lpcrel_hi0:
  auipc a0, %pcrel_hi(object)
  addi a0, a0, %pcrel_lo(.Lpcrel_hi0)
  .insn 4, 0x0400307b
  ret
  .2byte 0
.size selected, .-selected
.section .text.other,"ax",@progbits
.globl decoy
.type decoy,@function
decoy:
  .insn 4, 0x0600307b
  ret
.size decoy, .-decoy
.section .data,"aw",@progbits
object: .word 0
''')
    subprocess.run([clang, '--target=riscv64-unknown-elf', '-march=rv64gc', '-c',
                    str(source), '-o', str(output)], check=True, capture_output=True, timeout=30)
    image = PinnedELF(open_ordinary(output), 'selected.o')
    try:
        result = image.function_disassembly('selected')
        assert '<.Lpcrel_hi0>:' in result['text']
        assert '<decoy>:' not in result['text']
        assert result['coverage']['covered_bytes'] == result['extent']['size_bytes']
        assert result['coverage']['instructions'] == 6  # includes the zero halfword after ret
        prefix = subprocess.check_output([objdump_bin(), '-d', '--disassemble-symbols=selected',
                                          str(output)], text=True, timeout=30)
        assert len(tokenize_text(prefix)) < result['coverage']['instructions']
    finally:
        image.close()


def _float_object(tmp_path, *, keep_attributes):
    """A linked-style object whose F/D words need the extension selection to decode."""
    clang = os.environ.get('MERLIN_CLANG') or shutil.which('clang')
    objcopy = shutil.which('llvm-objcopy')
    if not clang or not objcopy or Path(objdump_bin()).name == 'objdump':
        pytest.skip('RISC-V clang, llvm-objcopy and llvm-objdump required')
    source, output = tmp_path/'float.S', tmp_path/'float.o'
    source.write_text('''
.section .text.selected,"ax",@progbits
.globl selected
.type selected,@function
selected:
  fadd.d fa0, fa0, fa1
  fmul.s fa2, fa2, fa3
  mul a0, a0, a1
  ret
.size selected, .-selected
''')
    subprocess.run([clang, '--target=riscv64-unknown-elf', '-march=rv64gc', '-c',
                    str(source), '-o', str(output)], check=True, capture_output=True, timeout=30)
    if not keep_attributes:
        # Drop the arch attribute and the ISA-carrying mapping symbols, as in the whole-model ELFs.
        subprocess.run([objcopy, '--remove-section=.riscv.attributes', '--strip-unneeded', str(output)],
                       check=True, capture_output=True, timeout=30)
    return output


def _mnemonics(image):
    return [row.mnemonic for row in tokenize_text(image.function_disassembly('selected')['text'])]


def test_contract_isa_decodes_float_words_of_an_elf_without_arch_attribute(tmp_path, monkeypatch):
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    output = _float_object(tmp_path, keep_attributes=False)
    declared = dict(value='rv64gc', source='selected_contract:runner.family.toolchain.cflags:-march')
    image = PinnedELF(open_ordinary(output), 'float.o', declared_isa=declared)
    try:
        assert image.arch_attribute is None
        record = image.decoder_isa
        assert record['status'] == 'selected' and record['source'] == declared['source']
        assert record['value'] == 'rv64gc' and {'f', 'd', 'm', 'c'} <= set(record['features'])
        assert record['corroboration']['consistent'] is True
        mnemonics = _mnemonics(image)
        assert mnemonics[:3] == ['fadd.d', 'fmul.s', 'mul'] and '<unknown>' not in mnemonics
        assert image.function_disassembly('selected')['decoder_isa']['source'] == declared['source']
    finally:
        image.close()


def test_no_isa_source_leaves_the_decoder_isa_unknown(tmp_path, monkeypatch):
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    output = _float_object(tmp_path, keep_attributes=False)
    image = PinnedELF(open_ordinary(output), 'float.o')
    try:
        record = image.decoder_isa
        assert record['status'] == 'unknown' and record['value'] is None and record['options'] == []
        # The header flags corroborate only; they never select an extension.
        assert record['corroboration']['float_abi'] in ('double', 'soft') and 'features' not in record
        assert '<unknown>' in _mnemonics(image)
    finally:
        image.close()


def test_elf_arch_attribute_wins_over_the_contract_declaration(tmp_path, monkeypatch):
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    output = _float_object(tmp_path, keep_attributes=True)
    image = PinnedELF(open_ordinary(output), 'float.o', declared_isa=dict(value='rv64i', source='contract'))
    try:
        assert image.decoder_isa['source'] == 'elf:.riscv.attributes:Tag_RISCV_arch'
        assert image.decoder_isa['value'].startswith('rv64i') and 'd' in image.decoder_isa['features']
        assert _mnemonics(image)[:2] == ['fadd.d', 'fmul.s']
    finally:
        image.close()


def test_isa_strings_translate_structurally():
    from merlin.compile.analysis.host_isa import isa_features, select_decoder_isa

    assert isa_features('rv64gc') == (64, ('i', 'm', 'a', 'f', 'd', 'zicsr', 'zifencei', 'c'))
    assert isa_features('rv64imafdcb_zicsr_zfh_zba') == (
        64, ('i', 'm', 'a', 'f', 'd', 'c', 'b', 'zicsr', 'zfh', 'zba'))
    assert isa_features('rv64i2p1_m2p0_f2p2_zicsr2p0_zvl128b1p0') == (
        64, ('i', 'm', 'f', 'zicsr', 'zvl128b'))
    for bad in ('', 'x86_64', 'rv_gc', 'rv64'):
        with pytest.raises(ValueError):
            isa_features(bad)
    narrow = select_decoder_isa(elf_attribute=None, declared=dict(value='rv32gc', source='c'),
                                e_flags=0, elf_xlen=64)
    assert narrow['status'] == 'unknown' and narrow['options'] == []
    invalid = select_decoder_isa(elf_attribute=dict(value=None, source='elf', error='elf_arch_attribute_invalid'),
                                 declared=dict(value='rv64gc', source='c'), e_flags=0, elf_xlen=64)
    assert invalid['status'] == 'unknown' and invalid['options'] == []


def test_contract_host_isa_reads_the_selected_runner_family():
    from merlin.compile.analysis.host_isa import contract_host_isa

    family = dict(engines=dict(functional=dict(argv=['--isa=rv64imafdc_zicsr', '{elf}']),
                               rtl=dict(argv=['{elf}'])),
                  toolchain=dict(cflags=['-march=rv64gc', '-O2']))
    contract = dict(runner=dict(backend='selected_family', selected_family=family))
    assert contract_host_isa(contract) == dict(
        value='rv64imafdc_zicsr', source='selected_contract:runner.selected_family.engines.functional.argv:--isa')
    family['engines']['functional']['argv'] = ['{elf}']
    assert contract_host_isa(contract)['source'] == 'selected_contract:runner.selected_family.toolchain.cflags:-march'
    family['engines']['rtl']['argv'] = ['--isa=rv64gc', '{elf}']
    family['engines']['functional']['argv'] = ['--isa=rv64imac', '{elf}']
    assert contract_host_isa(contract) is None  # conflicting declarations select nothing
    assert contract_host_isa(dict(runner=dict(backend='absent'))) is None
    assert contract_host_isa(None) is None


@pytest.mark.skipif(not os.environ.get('MERLIN_TEST_EXPERT_OBJECT'), reason='operator-owned original expert')
def test_original_expert_qk_setup_prefix_is_not_the_whole_function(tmp_path, monkeypatch):
    """No private reference bytes or operator path are committed to the test."""
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    source = Path(os.environ['MERLIN_TEST_EXPERT_OBJECT'])
    image = PinnedELF(open_ordinary(source), 'operator-owned-expert.o')
    try:
        result = image.function_disassembly('att_qk')
        prefix = subprocess.check_output([objdump_bin(), '-d', '--disassemble-symbols=att_qk',
                                          str(source)], text=True, timeout=30)
        assert result['coverage']['covered_bytes'] == result['extent']['size_bytes'] == 0x1b6
        assert len(tokenize_text(prefix)) == 37
        assert result['coverage']['instructions'] > len(tokenize_text(prefix))
        assert tokenize_text(result['text'])[-1].addr == 0x1b4
    finally:
        image.close()
