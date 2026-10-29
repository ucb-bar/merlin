"""New execution authority must survive broker startup without legacy proof grants."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import full_compiler_workspace as F
from merlin_experiments.phase1.brokers import model_check as M, selfcheck as B


def test_external_broker_never_constructs_legacy_common_proof_service(tmp_path, monkeypatch):
    from merlin_experiments.phase1.brokers import common_transforms

    calls = []
    monkeypatch.setattr(common_transforms, 'CommonTransformService', lambda **kw: calls.append(kw) or kw)
    monkeypatch.setenv('MERLIN_COMPILER_EXECUTION_MODE', 'snapshot_external_v1')
    assert B._legacy_common_service(tmp_path, tmp_path / 'manifest', tmp_path) is None
    assert not calls
    monkeypatch.setenv('MERLIN_COMPILER_EXECUTION_MODE', 'profile_strict_v1')
    assert B._legacy_common_service(tmp_path, tmp_path / 'manifest', tmp_path)['repo'] == tmp_path
    assert len(calls) == 1
    monkeypatch.setenv('MERLIN_COMPILER_EXECUTION_MODE', 'invalid')
    with pytest.raises(ValueError, match='unknown compiler execution mode'):
        B._legacy_common_service(tmp_path, tmp_path / 'manifest', tmp_path)


@pytest.fixture
def selected(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    for name in ('merlin/__init__.py', 'merlin/compile/__init__.py', 'merlin/compile/example.py'):
        path = repo / 'src' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('value=1\n')
    selection = F.prepare(repo, tmp_path / 'public', review_id='broker admission fixture')
    package = tmp_path / 'workspace/submission'
    F.stage(selection, package)
    (package / 'manifest.yaml').write_text('commands: {}\n')
    monkeypatch.setenv('MERLIN_COMPILER_EXECUTION_MODE', 'snapshot_external_v1')
    monkeypatch.setenv(F.ENV, json.dumps(selection.record()))
    return selection, package


def test_full_compiler_model_admission_requires_context_and_valid_tree(selected, tmp_path, monkeypatch):
    selection, package = selected
    context = SimpleNamespace(repo=tmp_path)
    M._require_compiler_selection(package, tmp_path / 'manifest', context, legacy_library=None)
    for manifest, ctx in [(None, context), (tmp_path / 'manifest', None)]:
        with pytest.raises(ValueError, match='original compiler profile context'):
            M._require_compiler_selection(package, manifest, ctx, legacy_library=None)
    monkeypatch.setenv('MERLIN_COMPILER_EXECUTION_MODE', 'profile_strict_v1')
    with pytest.raises(ValueError, match='selected compiler library'):
        M._require_compiler_selection(package, tmp_path / 'manifest', context, legacy_library=None)
    monkeypatch.setenv('MERLIN_COMPILER_EXECUTION_MODE', 'snapshot_external_v1')
    denied = package / 'compiler_library/merlin/runtime/reference.py'
    denied.parent.mkdir(parents=True, exist_ok=True)
    denied.write_text('private_replacement=True\n')
    with pytest.raises(ValueError, match='outside public namespace'):
        M._require_compiler_selection(package, tmp_path / 'manifest', context, legacy_library=None)


def test_full_model_broker_launch_preserves_explicit_selection(selected, tmp_path, monkeypatch):
    selection, package = selected
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setenv('MERLIN_TARGET_CONTRACT', str(Path(__file__).resolve().parents[3] /
                         'examples/gemmini/target/contracts/target_contract.yaml'))
    monkeypatch.setenv('MERLIN_MODEL_CHECK_BUNDLE', str(tmp_path / 'model-bundle'))
    seen = []

    class Child:
        returncode = 0
        def __init__(self, argv, **kwargs):
            seen.append(argv)
            assert '--compiler-bundle-manifest' in argv
            assert argv[argv.index('--compiler-bundle-manifest') + 1] == str(tmp_path / 'manifest')
            assert '--compiler-library-root' not in argv  # no fabricated legacy authority
            assert F.selected().record() == selection.record()
            Path(argv[argv.index('--out') + 1]).write_text(json.dumps({
                'schema': M.REPORT_SCHEMA, 'command': 'check', 'model': 'resnet50',
                'status': 'pass', 'exit_code': 0, 'agent_view': True, 'programs': [], 'split': 'visible'}))
        def poll(self):
            return 0

    monkeypatch.setattr(M.subprocess, 'Popen', Child)
    result = M.execute_visible(package.parent, {'action': 'check', 'model': 'resnet50', 'split': 'visible'},
                               compiler_bundle_manifest=tmp_path / 'manifest',
                               context=SimpleNamespace(repo=tmp_path, descriptor=tmp_path / 'descriptor', readback_policy=None))
    assert len(seen) == 1 and result['status'] == 'pass'
