"""Raw-only author/candidate masks follow descriptor adapters, including aliases."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import shutil
import subprocess
import sys

import pytest

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.mount_visibility import _MountVisibility
from merlin_experiments.phase1 import author_profiles as A


@pytest.fixture
def mask_fixture(monkeypatch, tmp_path):
    assets = tmp_path / 'arbitrary-adapter-assets'
    assets.mkdir()
    summary = assets / 'generated-summary.json'
    summary.write_text('{"derived":true}\n')
    harness = tmp_path / 'declared-harness'
    harness.mkdir()
    (harness / 'private_driver.c').write_text('int private_driver;\n')
    public = tmp_path / 'public'
    public.mkdir()
    raw = public / 'hardware.fir'
    raw.write_text('circuit public_raw :\n')
    compiler = public / 'ordinary-compiler'
    compiler.write_text('ordinary compiler fixture\n')
    target = SimpleNamespace(sim_via='unrelated-test-adapter', curated_harness='fixture')
    monkeypatch.setitem(TC.SIM_TOOLCHAINS, target.sim_via,
                        TC.SimToolchain(bind_paths=(str(assets),)))
    monkeypatch.setattr(TC, 'curated_harness_dir', lambda selected: str(harness))
    # Full original bundle verification is covered separately. This fixture tests
    # mask composition with no staged corpus and keeps package masks out of scope.
    calls = []
    monkeypatch.setattr(A, 'verify_bundle', lambda profile, bundle: calls.append((profile.name, bundle)))
    monkeypatch.setattr(A, '_mounted_python_packages', lambda argv, names: ())
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
    guest = tmp_path / 'guest-aliases'
    guest.mkdir()
    for name in ('renamed-adapter', 'symbolic-adapter', 'renamed-harness'):
        (guest / name).mkdir()
    for name in ('late-generated-leaf.json', 'raw-hardware.fir', 'ordinary-compiler'):
        (guest / name).touch()
    return SimpleNamespace(guest=guest, assets=assets, summary=summary, harness=harness, public=public,
                           raw=raw, compiler=compiler, target=target, calls=calls)


def bundle(name):
    return {'author_profile': {'name': name}}


@pytest.mark.parametrize('level', (1, 2))
def test_raw_only_roots_are_declared_by_adapter_and_harness(mask_fixture, level):
    f = mask_fixture
    roots = A.withheld_simulator_roots(f.target, bundle(f'xs-el-raw-rtl-v2/el{level}'))
    assert roots == tuple(sorted((f.assets, f.harness)))
    assert f.calls and f.calls[-1][0] == f'xs-el-raw-rtl-v2/el{level}'


@pytest.mark.parametrize('name', ('xs-el-raw-rtl-v2/el3', 'xs-el-v1/el1',
                                  'xs-el-no-expert-v1/el2', 'xs-f6-v1/inherited'))
def test_historical_and_fact_permitted_profiles_do_not_withhold_simulator_assets(mask_fixture, name):
    assert A.withheld_simulator_roots(None, bundle(name)) == ()
    args = ['--ro-bind', str(mask_fixture.assets), '/legacy-assets']
    assert A.runtime_masks(args, bundle(name)) == args


def test_no_profile_preserves_existing_policy(mask_fixture):
    assert A.withheld_simulator_roots(None, {}) == ()
    assert A.runtime_masks(['--ro-bind', '/usr', '/usr'], {}) == ['--ro-bind', '/usr', '/usr']


def test_restricted_profile_without_declared_target_refuses(mask_fixture):
    with pytest.raises(ValueError, match='requires its declared target'):
        A.runtime_masks([], bundle('xs-el-raw-rtl-v2/el1'))


def test_original_profile_verification_cannot_be_bypassed(mask_fixture, monkeypatch):
    def refuse(profile, selected):
        raise ValueError('original bundle identity changed')
    monkeypatch.setattr(A, 'verify_bundle', refuse)
    with pytest.raises(ValueError, match='original bundle identity changed'):
        A.withheld_simulator_roots(mask_fixture.target, bundle('xs-el-raw-rtl-v2/el1'))


@pytest.mark.parametrize('unsafe', ('/', 'relative/assets', '/declared/../root', '/bad\0root'))
def test_unsafe_adapter_asset_root_refuses(mask_fixture, monkeypatch, unsafe):
    monkeypatch.setitem(TC.SIM_TOOLCHAINS, mask_fixture.target.sim_via,
                        TC.SimToolchain(bind_paths=(unsafe,)))
    with pytest.raises(ValueError, match='unsafe asset root'):
        A.withheld_simulator_roots(mask_fixture.target, bundle('xs-el-raw-rtl-v2/el2'))


def _alias_mounts(f, tmp_path):
    symbolic = tmp_path / 'adapter-alias'
    symbolic.symlink_to(f.assets, target_is_directory=True)
    args = ['--ro-bind', str(tmp_path), str(tmp_path),
            '--ro-bind', str(f.assets), str(f.guest / 'renamed-adapter'),
            '--ro-bind', str(symbolic), str(f.guest / 'symbolic-adapter'),
            '--ro-bind', str(f.summary), str(f.guest / 'late-generated-leaf.json'),
            '--ro-bind', str(f.harness), str(f.guest / 'renamed-harness'),
            '--ro-bind', str(f.raw), str(f.guest / 'raw-hardware.fir'),
            '--ro-bind', str(f.compiler), str(f.guest / 'ordinary-compiler')]
    return args


@pytest.mark.parametrize('level', (1, 2))
def test_final_masks_close_parent_leaf_symlink_and_late_bind_aliases(mask_fixture, tmp_path, level):
    f = mask_fixture
    args = _alias_mounts(f, tmp_path)
    closed = A.runtime_masks(args, bundle(f'xs-el-raw-rtl-v2/el{level}'), target=f.target)
    visible = _MountVisibility(closed)
    for path in (f.summary, f.harness / 'private_driver.c',
                 Path(str(f.guest / 'renamed-adapter/generated-summary.json')),
                 Path(str(f.guest / 'symbolic-adapter/generated-summary.json')),
                 Path(str(f.guest / 'late-generated-leaf.json')), Path(str(f.guest / 'renamed-harness/private_driver.c'))):
        assert not visible.is_exposed(path), str(path)
    assert visible.is_exposed(f.raw)
    assert visible.is_exposed(Path(str(f.guest / 'raw-hardware.fir')))
    assert visible.is_exposed(Path(str(f.guest / 'ordinary-compiler')))


@pytest.mark.parametrize('level', (1, 2))
def test_real_bwrap_author_and_helper_reads_deny_assets_keep_raw_runtime(mask_fixture, tmp_path, level):
    f = mask_fixture
    binary = shutil.which('bwrap')
    if binary is None:
        pytest.skip('bubblewrap unavailable; real namespace isolation not tested')
    # Controlled process reads only fixture paths. This broad read-only prefix
    # deliberately exercises a denial that must survive an enclosing mount.
    args = ['--die-with-parent', '--unshare-pid', '--ro-bind', '/', '/', '--proc', '/proc', '--dev', '/dev',
            *_alias_mounts(f, tmp_path)]
    closed = A.runtime_masks(args, bundle(f'xs-el-raw-rtl-v2/el{level}'), target=f.target)
    payload = {'deny': [str(f.summary), str(f.guest / 'renamed-adapter/generated-summary.json'),
                        str(f.guest / 'symbolic-adapter/generated-summary.json'), str(f.guest / 'late-generated-leaf.json'),
                        str(f.guest / 'renamed-harness/private_driver.c')],
               'allow': [str(f.guest / 'raw-hardware.fir'), str(f.guest / 'ordinary-compiler')], 'helper': True}
    script = '''from pathlib import Path
import json,subprocess,sys
p=json.loads(sys.argv[1])
for name in p['deny']:
    try: raw=Path(name).read_bytes()
    except (FileNotFoundError,PermissionError,IsADirectoryError): pass
    else: assert raw == b'', name
for name in p['allow']: assert Path(name).read_bytes(), name
if p['helper']:
    p['helper']=False
    r=subprocess.run([sys.executable,'-I','-B','-c',sys.argv[2],json.dumps(p),''],capture_output=True,text=True)
    assert r.returncode==0, r.stderr
print('fixture author/helper reads passed')
'''
    result = subprocess.run([binary, *closed, sys.executable, '-I', '-B', '-c',
                             script, json.dumps(payload), script],
                            capture_output=True, text=True, timeout=30)
    if result.returncode and ('Operation not permitted' in result.stderr or 'No permissions' in result.stderr):
        pytest.skip('bubblewrap namespaces unavailable; actual isolation not tested')
    assert result.returncode == 0, result.stderr
    assert 'fixture author/helper reads passed' in result.stdout
