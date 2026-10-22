"""Exact output export never changes the executed ELF or exports arbitrary memory."""
import struct
import shutil
import subprocess

import pytest
from merlin_experiments.phase2 import retained_memory_observation as M


def test_symbol_supplement_has_only_absolute_bounds_and_no_loads():
    raw = M.signature_symbols(0x80001000, 32)
    header = struct.unpack_from('<HHIQQQIHHHHHH', raw, 16)
    assert header[0:3] == (2, 243, 1)
    assert header[3] == header[4] == header[9] == 0  # entry, phoff, phnum
    sections = [struct.unpack_from('<IIQQQQIIQQ', raw, header[5] + i*64) for i in range(4)]
    assert all(section[2] == 0 for section in sections)  # no ALLOC/WRITE/EXEC
    sym = sections[2]
    rows = [struct.unpack_from('<IBBHQQ', raw, sym[4]+i*24) for i in range(3)]
    assert [row[4] for row in rows[1:]] == [0x80001000, 0x80001020]
    assert all(row[3] == 0xfff1 for row in rows[1:])
    assert raw.count(b'begin_signature\0') == raw.count(b'end_signature\0') == 1


@pytest.mark.parametrize('base,size', [(0,0),(0,65537),(-1,1),(2**64-1,2),(True,1)])
def test_bounds_are_strict(base,size):
    with pytest.raises(ValueError):M.signature_symbols(base,size)


def test_signature_requires_every_byte_and_no_extras():
    assert M.signature_bytes(b'ff\n01\n',2) == b'\xff\x01'
    for raw in (b'ff\n', b'ff\n01\n00\n', b'zz\n01\n', b'ff 01\n'):
        with pytest.raises(ValueError):M.signature_bytes(raw,2)


@pytest.fixture
def original(tmp_path):
    cc = shutil.which('cc')
    if cc is None:pytest.skip('host C compiler unavailable')
    source=tmp_path/'fixture.c';elf=tmp_path/'fixture'
    source.write_text('static int tensor_0[2] __attribute__((used)); int main(void){return 0;}\n')
    subprocess.run([cc,'-O0','-no-pie',str(source),'-o',str(elf)],check=True,capture_output=True)
    cb={'target':'fixture','commands':[],'params':{},'tensors':{'Y':{'role':'output','dtype':'i32','shape':[1,2]}},
        '_oracle_expected_outputs':{'Y':[[-2,3]]}}
    return elf.read_bytes(), cb


def test_output_address_comes_from_typed_writable_elf_not_caller_metadata(original):
    raw,cb=original
    cb['output_regions']=[{'base':0,'bytes':100000000}]
    region,expected=M.output_region(raw,cb)
    assert region['base']>0 and region['bytes']==8
    assert expected==(-2).to_bytes(4,'little',signed=True)+(3).to_bytes(4,'little',signed=True)
    assert raw==original[0]


def test_rejects_wrong_size_old_signature_and_oracle(original):
    raw,cb=original
    for changed in (raw+b'begin_signature\0', raw[:16]+b'\x03\x00'+raw[18:]):
        with pytest.raises(ValueError):M.output_region(changed,cb)
    cb['_oracle_expected_outputs']['Y']=[[2**40,3]]
    with pytest.raises(ValueError):M.output_region(raw,cb)
    cb['_oracle_expected_outputs']['Y']=[[-2,3]];cb['tensors']['Y']['shape']=[1,3]
    with pytest.raises(ValueError):M.output_region(raw,cb)
