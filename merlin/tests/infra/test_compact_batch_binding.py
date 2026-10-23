"""Compact pass lines need both result-table and executable identity binding."""
import hashlib
import importlib.util
from pathlib import Path
from merlin.common.paths import repo_root


def checker():
    s=importlib.util.spec_from_file_location('batch_checker',repo_root()/'build_tools/scripts/compact_batch_check.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_compact_bindings_and_three_negative_controls(tmp_path):
    m=checker();elf=tmp_path/'batch.elf';elf.write_bytes(b'original executable')
    manifest=dict(elf=str(elf),elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),mode='compact',cases=[dict(case_id='one',lane='device',expected_fnv=127)])
    batch,table=m.expected_ids(manifest['cases'])
    console=f'P0\nBATCH_HASH 0x{batch:x}\nBATCH_ELF_ID 0x{table:x}\nPERF n=1 cyc=2 max=2 k=0\nBATCH_DONE fails=0\n'
    assert m.check(manifest,console)['pass']
    manifest['cases'][0]['expected_fnv']^=1
    assert not m.check(manifest,console)['pass']
    manifest['cases'][0]['expected_fnv']^=1
    elf.write_bytes(b'different executable')
    assert not m.check(manifest,console)['pass']
    elf.write_bytes(b'original executable')
    assert not m.check(manifest,console.replace(f'0x{batch:x}',f'0x{batch^1:x}'))['pass']
    assert not m.check(manifest,console.replace('P0','F0'))['pass']
    assert not m.check(manifest,console.replace(f'BATCH_HASH 0x{batch:x}\n',''))['pass']
