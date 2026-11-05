"""Installed-shaped phase-1 input preparation without a checkout or native imports."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import sysconfig

import pytest
import yaml
from merlin_experiments.phase1 import oot_history as OH
from merlin_experiments.phase1 import run_inputs as RI

from merlin.common import oot_repo as O
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
RI.verify_seed_submission(ws,run,record)
graded = RI.snapshot_seed_for_grade(ws,run,record)
assert (graded/'compiler.py').read_bytes() == payloads['compiler.py']
(graded/'compiler.py').write_text('changed only graded copy')
try: RI.snapshot_seed_for_grade(ws,run,record)
except RuntimeError as exc: assert 'already exists' in str(exc), str(exc)
else: raise AssertionError('existing graded submission was overwritten')
(ws/'submission'/'compiler.py').write_text('changed staged candidate')
try: RI.verify_seed_submission(ws,run,record)
except RuntimeError as exc: assert 'changed after selection' in str(exc), str(exc)
else: raise AssertionError('staged candidate drift admitted')
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


def test_authoring_seed_requires_same_lineage_final_oot_tree(tmp_path):
    previous = tmp_path / "previous-run"
    source = previous / "submission"
    source.mkdir(parents=True)
    (source / "manifest.yaml").write_text("commands: {}\n")
    (source / "compiler.py").write_text("VALUE = 1\n")
    (previous / "environment.yaml").write_text(yaml.safe_dump({
        "run_id": "r53", "lineage_id": "r53",
    }))
    O.init(previous / "oot")
    commit = O.commit_candidate(previous / "oot", source, label="freeze", when=1, run_id="r53")
    O.tag(previous / "oot", O.FROZEN_TAG, commit.commit)
    ws, run = tmp_path / "workspace", tmp_path / "new-run"
    ws.mkdir()
    run.mkdir()
    record = RI.seed_submission(ws, source, run, require_lineage=True)
    assert record["lineage"] == {
        "lineage_id": "r53", "source_run_id": "r53", "source_commit": commit.commit,
        "source_tree_sha256": O.tree_digest(previous / "oot", commit.commit),
    }
    RI.verify_seed_submission(ws, run, record)
    (source / "compiler.py").write_text("operator edit\n")
    with pytest.raises(RuntimeError, match="final oot commit tree"):
        RI.verify_seed_submission(ws, run, record)


def test_r40_style_arbitrary_seed_is_rejected_before_copy(tmp_path):
    source = tmp_path / "r40-tap-final-submission"
    source.mkdir()
    (source / "manifest.yaml").write_text("commands: {}\n")
    ws, run = tmp_path / "workspace", tmp_path / "run"
    ws.mkdir()
    run.mkdir()
    with pytest.raises(RuntimeError, match="final submission"):
        RI.seed_submission(ws, source, run, require_lineage=True)
    assert not (ws / "submission").exists()


def test_turn_end_refuses_bytes_outside_final_oot_commit(tmp_path):
    run, submission = tmp_path / "run", tmp_path / "submission"
    run.mkdir()
    submission.mkdir()
    (submission / "manifest.yaml").write_text("commands: {}\n")
    OH.start(run)
    committed = OH.commit_graded(run, submission, label="round", key="1")
    assert committed is not None
    assert OH.verify_turn_end(run, submission, committed.commit)["matched"] is True
    (submission / "manifest.yaml").write_text("commands: {changed: true}\n")
    with pytest.raises(O.OotRepoError, match="submission changed"):
        OH.verify_turn_end(run, submission, committed.commit)
    assert json.loads((run / "turn_end_provenance.json").read_text())["matched"] is False
