"""Installed-shaped phase-1 input preparation without a checkout or native imports."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import sysconfig

from merlin_experiments.phase1 import run_inputs as RI

from merlin.common.paths import module_source_path


def test_installed_run_input_records_seed_and_private_drift(tmp_path):
    source = module_source_path("merlin_experiments").parent
    installed = tmp_path / "installed"
    shutil.copytree(source, installed / "merlin_experiments", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    program = r"""
import hashlib, importlib.abc, json, os, pathlib, stat, sys
sys.path[:0] = [sys.argv[1], *json.loads(sys.argv[2])]
class NoCheckout(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in {'merlin', '_common', 'run_baseline_qa_loop'}, fullname
sys.meta_path.insert(0, NoCheckout())
before = dict(os.environ)
from merlin_experiments.phase1 import run_inputs as RI
assert dict(os.environ) == before
assert pathlib.Path(RI.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
root = pathlib.Path.cwd()
source, ws, run, bundle, hidden = [root / name for name in ('seed','workspace','run','bundle','private')]
for path in (source, ws, run, bundle, hidden): path.mkdir()
payloads = {'manifest.yaml': b'commands: {}\n', 'compiler.py': b'VALUE = 7\n'}
for name, payload in payloads.items(): (source / name).write_bytes(payload)
(source / 'build').mkdir(); (source / 'build' / 'stale').write_text('not copied')
(source / 'CMakeCache.txt').write_text('not copied')
for path in source.rglob('*'):
    if path.is_file(): path.chmod(0o444)
source.chmod(0o555)
record = RI.seed_submission(ws, source, run)
identity = hashlib.sha256()
for name in sorted(payloads):
    data = payloads[name]
    identity.update(name.encode() + b'\0' + str(len(data)).encode() + b'\0' + data + b'\0')
expected = {'version':1, 'source':str(source.resolve()), 'content_sha256':identity.hexdigest(),
            'n_files':2, 'n_bytes':sum(map(len,payloads.values()))}
assert record == expected
assert (run/'seed_submission.json').read_text() == json.dumps(expected,indent=2,sort_keys=True)+'\n'
assert (source/'compiler.py').stat().st_mode & stat.S_IWUSR == 0
assert (ws/'submission'/'compiler.py').stat().st_mode & stat.S_IWUSR
assert not (ws/'submission'/'CMakeCache.txt').exists()
assert not (ws/'submission'/'build').exists()
for path in (ws/'TASK.md',run/'TASK.md'): path.write_text('public task\n')
for path in (bundle/'input_bundle_manifest.yaml',run/'input_bundle_manifest.yaml'):
    path.write_text('allowed: []\n')
treatment = RI.treatment_snapshot_record(ws,run,bundle,['one','two'])
assert list(treatment) == ['version','content_sha256','n_files_present','files','resolved_tool_ids']
assert [row['name'] for row in treatment['files']] == [
    'served/TASK.md','archived/TASK.md','archived_bundle/input_bundle_manifest.yaml',
    'served/TASK_ADDENDUM.md','served/ALLOWED_MERLIN_TOOLS.md','served/MERLIN_PROVENANCE_TEMPLATE.md',
    'source_bundle/input_bundle_manifest.yaml','source_bundle/allowed_files.txt','source_bundle/tools.txt']
(hidden/'capsule.yaml').write_text('private-synthetic-answer-sentinel\n')
private_record = RI.subtree_snapshot_record(hidden)
assert list(private_record) == ['version','path','content_sha256','n_files','n_bytes','n_capsules']
assert 'private-synthetic-answer-sentinel' not in json.dumps(treatment)
errata = root/'correction.md'; errata.write_text('reviewed public correction\n')
errata_record = RI.stage_operator_errata(run,errata)
environment = {'target':'fixture','task_scope':{'kind':'fixture'},'treatment_snapshot':treatment,
               'hidden_capsule_snapshot':private_record,'operator_errata':errata_record}
kwargs = dict(identity={'target':'fixture'},task_scope={'kind':'fixture'},ws=ws,run_dir=run,
              bundle_dir=bundle,resolved_tools=['one','two'],expected_hidden_dir=hidden)
assert RI.verify_persisted_run_inputs(environment,**kwargs) == hidden
def refuses(fragment):
    try: RI.verify_persisted_run_inputs(environment,**kwargs)
    except RuntimeError as exc: assert fragment in str(exc), str(exc)
    else: raise AssertionError('drift admitted')
(bundle/'tools.txt').write_text('added after setup'); refuses('source_bundle/tools.txt')
(bundle/'tools.txt').unlink()
(run/'ERRATA.md').write_text('changed'); refuses('operator errata drifted')
(run/'ERRATA.md').write_bytes(errata.read_bytes())
(hidden/'capsule.yaml').write_text('changed private bytes'); refuses('hidden capsule snapshot drifted')
assert not any(name == 'merlin' or name.startswith('merlin.') for name in sys.modules)
source.chmod(0o755)
print('installed run-input verification passed')
"""
    dependencies = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program, str(installed), json.dumps(dependencies)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "installed run-input verification passed"
    assert "private-synthetic-answer-sentinel" not in result.stdout + result.stderr


def test_run_input_module_has_no_native_controller_facade():
    assert RI.seed_submission.__module__ == "merlin_experiments.phase1.run_inputs"
    assert RI.verify_persisted_run_inputs.__module__ == "merlin_experiments.phase1.run_inputs"
    assert "_common" not in RI.__dict__
