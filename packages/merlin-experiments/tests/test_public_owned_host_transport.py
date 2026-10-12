"""Operator selfcheck children receive exact frozen host inputs before formal grading."""
import hashlib
import json
import os
import subprocess
import sys

import pytest
import yaml

from merlin_experiments.phase1.feedback import public_owned_host_selection as selection


def test_pinned_host_leaf_build_selection_reaches_operator_child(tmp_path, monkeypatch):
    monkeypatch.setenv(selection.ENV, 'untrusted inherited value')
    row = {'board': 'selected_board', 'arena_mb': 32}
    for field in ('board_catalog', 'host_dts', 'rtl_facts'):
        path = tmp_path / field
        path.write_text(field)
        row[field] = str(path)
        row[field + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    declaration = tmp_path / 'instruction.json'
    declaration.write_text('{}')
    spec = tmp_path / 'spec.yaml'
    spec.write_text(yaml.safe_dump({'target': 'selected_target', 'models': [row]}))
    record = selection.install_for_session(spec, declaration, target='selected_target', run_dir=tmp_path / 'run')
    script = '''import os,json,pathlib,hashlib
r=json.loads(os.environ['MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION'])
p=pathlib.Path(r['path']); assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
s=json.loads(p.read_text()); assert s['arena_mb']==32
for k in ('board_catalog','dts','rtl_facts','instruction_declaration'):
 p=pathlib.Path(s[k]['path']); assert hashlib.sha256(p.read_bytes()).hexdigest()==s[k]['sha256']
print(s['board'])
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], env=os.environ.copy(),
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == row['board']
    assert json.loads(os.environ[selection.ENV]) == record
    (tmp_path / 'rtl_facts').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        selection.install_for_session(spec, declaration, target='selected_target', run_dir=tmp_path / 'run')
    assert selection.ENV not in os.environ


def test_missing_launch_inputs_clear_inherited_selection(tmp_path, monkeypatch):
    monkeypatch.setenv(selection.ENV, 'untrusted inherited value')
    assert selection.install_for_session(None, None, target='selected_target', run_dir=tmp_path) is None
    assert selection.ENV not in os.environ
