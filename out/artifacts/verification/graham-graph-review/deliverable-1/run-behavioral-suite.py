"""Run the selected pinned-producer behavioral tests and retain the native row-batching pair.

Behavioral observations only: no Lean qualification, replay or review approval.
Uses the existing pinned LLVM23 build and compiler interpreter; downloads nothing.
"""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).resolve().parent
INDEX = json.loads((OUT / 'source-index.json').read_text())
PRODUCER = Path(INDEX['producer'])
VERIFY = Path(INDEX['verification'])
LLVM = Path('/home/grammy/blast/merlin/third_party/llvm-build')
LLVM_REVISION = 'a47bddccec30255619bb8c37fa59700e661d4e66'
TESTS = OUT / 'behavioral-tests'
FILES = [f'merlin/tests/infra/test_{name}.py' for name in (
    'schedule_pipeline', 'configuration_elimination', 'configuration_loops', 'private_copy_elimination',
    'copy_coalescing', 'copy_pipeline', 'temporary_storage', 'device_spills', 'physical_fusion', 'program_limits',
    'command_compaction', 'staged_dense_tile_order', 'effect_cleanup', 'dead_payload_writes')] + [
    'merlin/tests/ir/test_recomputation.py', 'merlin/tests/ir/test_dense_row_batching.py']
NATIVE_CASE = 'test_ordinary_native_row_batch0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc)


settings = {
    'MERLIN_COMPILER_DRIVER': 'python',
    'MERLIN_COMPILER_PYTHON': str(VERIFY / 'out/build/lean/mlir-venv/bin/python'),
    'MERLIN_MLIR_INSTALL': str(LLVM),
    'MERLIN_MLIR_TRANSLATE': str(LLVM / 'bin/mlir-translate'),
    'MERLIN_CLANG': str(LLVM / 'bin/clang-23'),
}
tools = [LLVM / 'bin' / name for name in ('clang-23', 'mlir-opt', 'mlir-translate')] + [Path(settings['MERLIN_COMPILER_PYTHON'])]
versions = {t.name: subprocess.run([str(t), '--version'], capture_output=True, text=True, check=True).stdout
            for t in tools[:3]}
assert LLVM_REVISION in versions['clang-23'], versions['clang-23']

TESTS.mkdir(exist_ok=False)
started = now()
basetemp = VERIFY / 'out/build/verification-setup/graham-d1-suite' / started.strftime('%Y%m%dT%H%M%SZ')
basetemp.parent.mkdir(parents=True, exist_ok=True)
(TESTS / 'toolchain.json').write_text(json.dumps({
    'llvm_revision': LLVM_REVISION, 'settings': settings, 'versions': versions,
    'files': [{'path': str(t), 'sha256': sha(t), 'bytes': t.stat().st_size} for t in tools],
    'downloads': False}, indent=2) + '\n')
command = [str(VERIFY / '.venv/bin/python'), '-B', '-m', 'pytest', *FILES, '-q', '--basetemp', str(basetemp),
           '-o', 'tmp_path_retention_policy=all', '--junitxml', str(TESTS / 'suite.xml')]
environment = dict(os.environ, **settings, PYTHONPATH=str(PRODUCER / 'src'))
with (TESTS / 'suite.log').open('w') as log:
    completed = subprocess.run(command, cwd=PRODUCER, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
suites = ET.parse(TESTS / 'suite.xml').getroot().findall('testsuite')
stats = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
(TESTS / 'suite.json').write_text(json.dumps({
    'purpose': 'Selected pinned-producer behavioral tests in the configured native environment; not Lean qualification',
    'producer_revision': INDEX['producer_revision'], 'verification_revision': INDEX['verification_revision'],
    'command': command, 'cwd': str(PRODUCER), 'environment': {**settings, 'PYTHONPATH': environment['PYTHONPATH']},
    'start_utc': started.isoformat(), 'end_utc': now().isoformat(), 'returncode': completed.returncode,
    'results': stats}, indent=2) + '\n')

# Retain the native row-batching pair: every file the test kept, plus the owners/products it names.
source = basetemp / NATIVE_CASE
copy = TESTS / 'row-batching-native'
shutil.copytree(source, copy)
files = [{'original': str(p), 'copy': str((copy / p.relative_to(source)).relative_to(TESTS)), 'bytes': p.stat().st_size,
          'sha256': sha(p)} for p in sorted(source.rglob('*')) if p.is_file()]
pair = json.loads((source / 'pair.json').read_text())
owners = pair['owners'] + [f for arm in pair['arms'].values() for field in ('calls', 'artifacts') for f in arm[field]]
(TESTS / 'native-evidence.json').write_text(json.dumps({
    'scope': 'ordinary native CPU source algebra; no device retention/timing qualification',
    'original_pair': str(source / 'pair.json'), 'copy_pair': 'row-batching-native/pair.json',
    'owner_and_product_hashes_checked': len(owners), 'all_hashes_match': all(sha(f['path']) == f['sha256'] for f in owners),
    'baseline_and_candidate_differ': (source / 'baseline.mlir').read_bytes() != (source / 'candidate.mlir').read_bytes(),
    'calls_per_arm': {name: len(arm['calls']) for name, arm in pair['arms'].items()},
    'files': files, 'new_lean_qualification': False}, indent=2) + '\n')
print(json.dumps({'returncode': completed.returncode, **stats, 'native_files': len(files), 'owners': len(owners)}))
sys.exit(completed.returncode)
