"""Component profiles prove their copied API boundary rather than legacy SDK mounts."""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import full_compiler_workspace as F
from merlin_experiments.phase1 import tooling_readiness as R

from merlin.common.paths import repo_root


@pytest.fixture(scope='module')
def component_exports(tmp_path_factory):
    source = Path(os.environ.get('MERLIN_COMPONENT_READINESS_TEST_REPO', str(repo_root())))
    root = tmp_path_factory.mktemp('component-sources')
    return {level: A.prepare_sources(source, root / str(level), profile=A.SCRATCH_NAMES[level - 1],
                                    review_id='component readiness test') for level in (1, 2, 3)}


def bundle_for(level, selection):
    profile = A.resolve(A.SCRATCH_NAMES[level - 1])
    return dict(author_profile=profile.record(), author_profile_sources=None if selection is None else selection.record(),
                allowed=[], tools=list(profile.tools))


@pytest.mark.parametrize('level', [1, 2, 3])
def test_exact_component_selection_replaces_only_legacy_registry_grants(component_exports, level):
    bundle = bundle_for(level, component_exports[level])
    checks = R._grant_checks(SimpleNamespace(), bundle, tuple(bundle['tools']))
    assert checks and all(row['ok'] for row in checks)
    assert checks[0]['name'] == 'selected_component_sources'
    if level > 1:
        assert checks[-1]['name'] == 'grant:cpp_oot_generators:component_closure'
    # No ordinary arm/profile is permitted to silently waive legacy source grants.
    ordinary = R._grant_checks(SimpleNamespace(), dict(allowed=[]), ('cpp_oot_generators',))
    assert ordinary and not all(row['ok'] for row in ordinary)


def test_missing_wrong_or_changed_component_selection_refuses(component_exports, tmp_path):
    with pytest.raises(ValueError, match='original source selection'):
        R._component_selection(bundle_for(2, None))
    with pytest.raises(ValueError, match='another author profile'):
        R._component_selection(bundle_for(2, component_exports[3]))
    original = component_exports[2]
    copied = tmp_path / 'sources'
    shutil.copytree(original.root, copied)
    changed = copied / original.members[0][0]
    changed.chmod(0o644)
    changed.write_bytes(changed.read_bytes() + b'\n# changed\n')
    changed_bundle = bundle_for(2, original)
    changed_bundle['author_profile_sources']['root'] = str(copied)
    with pytest.raises(ValueError, match='baseline changed'):
        R._component_selection(changed_bundle)
    with pytest.raises(ValueError, match='no Merlin'):
        R._component_selection(bundle_for(1, original))


@pytest.mark.parametrize('level', [1, 2, 3])
def test_real_component_namespace_and_generator_probe(component_exports, tmp_path, level):
    """Actual Python/C++/MLIR subprocess in the same final profile mask owner."""
    binary = shutil.which('bwrap') or '/work/merlin/hw/bubblewrap-0.11.0/bin/bwrap'
    if not Path(binary).is_file():
        pytest.skip('bubblewrap unavailable')
    selection = component_exports[level]
    workspace = tmp_path / 'probe'
    workspace.mkdir()
    if selection is not None:
        F.stage(selection, workspace)
    bundle = bundle_for(level, selection)
    schema_dir = tmp_path / 'public-schemas'
    schema_dir.mkdir()
    source = Path(os.environ.get('MERLIN_COMPONENT_READINESS_TEST_REPO', str(repo_root())))
    schema = schema_dir / 'command_buffer.schema.json'
    shutil.copyfile(source / 'merlin/contract/schemas/command_buffer.schema.json', schema)
    bundle['allowed'].append(dict(path=str(schema), mode='ro', note='common public ABI schema'))
    argv = [binary, '--die-with-parent', '--unshare-pid', '--proc', '/proc', '--dev', '/dev']
    for path in ('/usr', '/bin', '/lib', '/lib64', '/etc'):
        if Path(path).exists():
            argv += ['--ro-bind', path, path]
    runtime_roots = {Path(sys.executable).absolute().parent.parent,
                     Path(sys._base_executable).absolute().parent.parent}
    for root in sorted(runtime_roots):
        argv += ['--ro-bind', str(root), str(root)]
    clang = Path(os.environ.get('MERLIN_CLANG', '/work/merlin/toolchains/llvm-a47bddcc/install/bin/clang-23'))
    llvm = clang.parent.parent.resolve()
    if not (llvm / 'bin/mlir-opt').is_file():
        pytest.skip('selected LLVM/MLIR toolchain unavailable')
    argv += ['--ro-bind', str(llvm), str(llvm), '--bind', str(workspace), str(workspace),
             '--ro-bind', str(schema_dir), str(schema_dir)]
    argv = A.runtime_masks(argv, bundle)
    argv += ['--setenv', 'PATH', str(llvm / 'bin') + ':/usr/bin:/bin',
             '--setenv', 'PYTHONDONTWRITEBYTECODE', '1', '--setenv', 'PYTHONNOUSERSITE', '1',
             '--setenv', 'PYTHONPATH', str(workspace / 'compiler_library'), '--chdir', str(workspace)]
    result = subprocess.run([*argv, sys.executable, '-B', '-c', R._component_probe(bundle)],
                            capture_output=True, text=True, timeout=60)
    if result.returncode and any(s in result.stderr for s in ('Operation not permitted', 'No permissions')):
        pytest.skip('kernel disallows bubblewrap: ' + result.stderr[:150])
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'COMPONENT_PROFILE_BOUNDARY_AND_APIS_OK' in result.stdout


def test_relative_public_schema_grant_uses_explicit_repository(component_exports, tmp_path):
    bundle = bundle_for(3, component_exports[3])
    schema_dir = tmp_path / 'selected-repo/merlin/contract/schemas'
    schema_dir.mkdir(parents=True)
    schema = schema_dir / 'command_buffer.schema.json'
    schema.write_text('{"type":"object"}')
    bundle['allowed'] = [dict(path='merlin/contract/schemas/', mode='ro', note='public ABI')]
    with pytest.raises(ValueError, match='selected repository owner'):
        R._component_probe(bundle)
    probe = R._component_probe(bundle, repo=tmp_path / 'selected-repo')
    assert str(schema) in probe
    assert "_schema = Path('merlin/contract/schemas" not in probe


def test_live_probe_copies_schema_from_original_frozen_owner(component_exports, tmp_path, monkeypatch):
    from contextlib import contextmanager

    bundle = bundle_for(3, component_exports[3])
    source = tmp_path / 'selected/merlin/contract/schemas/command_buffer.schema.json'
    source.parent.mkdir(parents=True)
    source.write_text('changed live data must not be read')
    frozen = tmp_path / 'frozen-command-buffer.schema.json'
    frozen.write_text('{"type":"object","frozen":"original public schema"}')
    bundle['allowed'] = [dict(path='merlin/contract/schemas/', mode='ro')]
    actual_ws = tmp_path / 'actual-workspace'
    actual_ws.mkdir()
    context = SimpleNamespace(repo=tmp_path / 'selected')
    observed = {}

    def snapshot_paths(workspace, selected_bundle, paths, *, repo):
        assert workspace == actual_ws and selected_bundle is bundle
        assert paths == [source] and repo == context.repo
        observed['frozen_owner'] = True
        return [frozen]

    @contextmanager
    def probe_session(ctx, probe_ws, selected_bundle, tools, *, facts_workspace, component_schema):
        assert ctx is context and selected_bundle is bundle and facts_workspace == actual_ws
        assert component_schema.parent == probe_ws
        assert component_schema.read_bytes() == frozen.read_bytes()
        assert component_schema.stat().st_mode & 0o222 == 0
        assert sorted(p.name for p in probe_ws.iterdir() if p.is_file()) == ['public-command-buffer.schema.json']
        observed['same_public_data'] = True
        yield "print('AUTHORING_AND_BROKER_ROUNDTRIPS_OK')"

    monkeypatch.setattr(R.BW, 'snapshot_input_paths', snapshot_paths)
    monkeypatch.setattr(R, 'public_probe_session', probe_session)
    monkeypatch.setattr(R.BW, 'full_argv', lambda *args: [])
    from merlin.targetgen.sandbox import toolchain
    monkeypatch.setattr(toolchain, 'sandbox_env', lambda *args, **kwargs: '')
    result = R._live_probe(context, object(), actual_ws, bundle, ('cpp_oot_generators',))
    assert result['ok'] and observed == dict(frozen_owner=True, same_public_data=True)
