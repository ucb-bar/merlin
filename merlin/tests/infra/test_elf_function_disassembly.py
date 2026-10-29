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
