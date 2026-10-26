"""Public implementation freedom does not expose private evaluator source or data."""
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import full_compiler_workspace as F
from merlin_experiments.phase1 import fresh_backend as FB


@pytest.fixture
def selection(tmp_path):
    repo = tmp_path / 'repo'
    sources = {
        'merlin/__init__.py': '',
        'merlin/compile/__init__.py': '',
        'merlin/compile/example.py': 'from merlin.common.helper import value\n',
        'merlin/common/__init__.py': '',
        'merlin/common/helper.py': 'value=3\n',
        'merlin/compile_cli.py': '', 'merlin/compile_core.py': '',
        'merlin/runtime/reference.py': 'private_oracle=123\n',
        'merlin/compile/needs_private.py': 'import merlin.runtime.reference\n',
        'merlin/compile/capture_client.py': (
            'import merlin.targetgen._dynamic_integer_reference\n'
            'import merlin.targetgen.materialized_reference\n'),
        'merlin/targetgen/_dynamic_integer_reference.py': 'private_oracle=123\n',
        'merlin/targetgen/materialized_reference.py': 'private_reference_reader=123\n',
        'merlin/kernels/known_answer.py': 'do_not_copy=1\n',
    }
    for name, text in sources.items():
        path = repo / 'src' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    private = repo / 'capsules' / 'case'
    private.mkdir(parents=True)
    (private / 'golden.npy').write_bytes(b'private output')
    return F.prepare(repo, tmp_path / 'public', review_id='fixture public compiler')


def test_complete_public_sources_exclude_answers_and_report_unavailable_dependency(selection):
    names = {row[0] for row in selection.members}
    assert 'merlin/compile/example.py' in names
    assert 'merlin/common/helper.py' in names
    assert not any('reference.py' in p or 'known_answer' in p or 'golden' in p for p in names)
    assert 'merlin.runtime.reference' in selection.diagnostics
    assert selection.permits('merlin.compile.new_agent_pass')
    assert selection.permits('merlin.runtime.new_host_schedule')
    assert not selection.permits('merlin.runtime.reference')
    assert not selection.permits('merlin_experiments.phase1.feedback')
    for module in ('merlin.targetgen._dynamic_integer_reference', 'merlin.targetgen.materialized_reference'):
        assert module in selection.diagnostics
        assert not selection.permits(module)
        assert module not in {row[2] for row in selection.members}


def test_staging_preserves_continuation_edits_without_links(selection, tmp_path):
    submission = tmp_path / 'submission'
    first = F.stage(selection, submission)
    source = submission / F.DIRECTORY / 'merlin/compile/example.py'
    assert source.stat().st_ino != (selection.root / 'merlin/compile/example.py').stat().st_ino
    source.write_text('edited=9\n')
    new = source.with_name('new_agent_pass.py')
    new.write_text('added=True\n')
    second = F.stage(selection, submission)
    assert not second['copied_missing_members']
    assert 'merlin/compile/new_agent_pass.py' in second['preserved_existing_members']
    assert source.read_text() == 'edited=9\n' and new.read_text() == 'added=True\n'
    versions = second['source_versions']
    assert versions['preserved_differences'] == [{
        'path': 'merlin/compile/example.py',
        'selected_baseline_sha256': next(digest for path, digest, _ in selection.members
                                         if path == 'merlin/compile/example.py'),
        'preserved_sha256': second['preserved_existing_members']['merlin/compile/example.py'],
    }]
    assert versions['additional_authored_members'] == ['merlin/compile/new_agent_pass.py']
    assert versions['qualification_issued'] is False
    assert first['selection']['manifest_sha256'] == selection.manifest_sha256
    assert 'merlin/compile/new_agent_pass.py' in F.candidate_sources(selection, submission)
    (source.parent / 'alias.py').symlink_to(source)
    with pytest.raises(ValueError, match='links'):
        F.candidate_sources(selection, submission)


def test_changed_baseline_and_manifest_refuse(selection):
    path = selection.root / selection.members[0][0]
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='baseline changed'):
        selection.verify()
    with pytest.raises(ValueError, match='identity mismatch'):
        F.select(selection.root, selection.manifest_path, '0' * 64)


def test_generic_baseline_is_not_a_target_backend_seed(selection, tmp_path):
    ws = tmp_path / 'workspace'
    F.stage(selection, ws / 'submission')
    options = SimpleNamespace(fresh_backend=True, compiler_access='full_compiler',
                              seed_submission='', qualify_submission='', sandbox='bwrap',
                              allow_unsandboxed=False, joint_epoch='')
    with F.scoped_full_compiler(selection):
        assert FB.origin(options, ws) == {'schema': 'merlin.empty_backend_origin.v1', 'initial_files': 0}
        (ws / 'submission' / 'target.py').write_text('authored=True\n')
        with pytest.raises(ValueError, match='zero target backend'):
            FB.origin(options, ws)


def test_policy_uses_only_already_bound_candidate_tree(selection, tmp_path):
    @dataclass(frozen=True)
    class Policy:
        package: Path
        argv: tuple = ('sandbox',)
        env_prefix: str = ''
    submission = tmp_path / 'submission'
    F.stage(selection, submission)
    policy = Policy(submission)
    with F.scoped_full_compiler(selection):
        updated = F.apply_policy(policy)
    assert updated.argv == policy.argv
    assert str(submission / F.DIRECTORY) in updated.env_prefix
    assert str(selection.root) not in updated.env_prefix
    assert "unset " + F.ENV in updated.env_prefix


def test_integrity_allows_new_compiler_modules_but_not_private_imports(selection, tmp_path):
    from merlin.targetgen.package_runtime import integrity_scan, CertFailure

    submission = tmp_path / 'submission'
    F.stage(selection, submission)
    backend = submission / 'backend.py'
    backend.write_text('import merlin.compile.new_agent_pass\n')
    helper = submission / F.DIRECTORY / 'merlin/compile/new_agent_pass.py'
    helper.write_text('value=42\n')
    package = SimpleNamespace(directory=submission, integrity_exempt=False)
    with F.scoped_full_compiler(selection):
        # Untouched reviewed source's dormant private import is not exposed code.
        integrity_scan(package)
        backend.write_text('import merlin.runtime.reference\n')
        with pytest.raises(CertFailure, match='reference'):
            integrity_scan(package)
        backend.write_text('import merlin.compile.new_agent_pass\n')
        helper.write_text('import merlin.runtime.reference\n')
        with pytest.raises(CertFailure, match='reference'):
            integrity_scan(package)


def test_new_private_module_cannot_be_smuggled_into_compiler_tree(selection, tmp_path):
    submission = tmp_path / 'submission'
    F.stage(selection, submission)
    private = submission / F.DIRECTORY / 'merlin/runtime/reference.py'
    private.parent.mkdir(parents=True)
    private.write_text('private=True\n')
    with pytest.raises(ValueError, match='outside public namespace'):
        F.candidate_sources(selection, submission)


def test_explicit_none_scope_does_not_read_environment(monkeypatch):
    monkeypatch.setenv(F.ENV, 'not valid JSON')
    with F.scoped_full_compiler(None):
        assert F.selected() is None


def test_full_guidance_does_not_invoke_sparse_fork_stage(selection, tmp_path, monkeypatch):
    from merlin_experiments.phase1 import compiler_library as CL
    from merlin.targetgen import compiler_library_fork
    workspace, run = tmp_path / 'workspace', tmp_path / 'run'
    workspace.mkdir(); run.mkdir()
    (workspace / 'TASK.md').write_text('Author compiler.\n')
    monkeypatch.setattr(compiler_library_fork, 'stage', lambda *a: pytest.fail('sparse stage invoked'))
    with F.scoped_full_compiler(selection):
        CL.append_guidance(workspace, run, None, None)
    assert 'complete reviewed public compiler' in (workspace / 'TASK.md').read_text()
    assert (workspace / 'FULL_COMPILER_SOURCE_SELECTION.json').is_file()


def test_initial_source_differences_reach_author_without_overwriting_edits(selection, tmp_path):
    from merlin_experiments.phase1 import compiler_library as CL

    workspace, run = tmp_path / 'workspace', tmp_path / 'run'
    workspace.mkdir(); run.mkdir()
    (workspace / 'TASK.md').write_text('Author compiler.\n')
    F.stage(selection, workspace / 'submission')
    edited = workspace / 'submission/compiler_library/merlin/compile/example.py'
    edited.write_text('retained_old_api=True\n')
    staging = F.stage(selection, workspace / 'submission')
    with F.scoped_full_compiler(selection):
        CL.append_guidance(workspace, run, None, None, staging_record=staging)
    report = json.loads((workspace / 'COMPILER_SOURCE_VERSIONS.json').read_text())
    assert report['preserved_differences'][0]['path'] == 'merlin/compile/example.py'
    assert 'retained 1 member(s)' in (workspace / 'TASK.md').read_text()
    assert edited.read_text() == 'retained_old_api=True\n'
    assert (run / 'COMPILER_SOURCE_VERSIONS.json').read_bytes() == (workspace / 'COMPILER_SOURCE_VERSIONS.json').read_bytes()
    foreign = dict(staging, selection={'manifest_sha256': '0' * 64})
    with F.scoped_full_compiler(selection), pytest.raises(ValueError, match='another source selection'):
        CL.append_guidance(workspace, run, None, None, staging_record=foreign)


def test_author_overlay_uses_editable_tree_without_legacy_profile(selection, tmp_path, monkeypatch):
    from merlin_experiments.phase1 import compiler_library as L
    from merlin.targetgen import compiler_profile_admission as admission
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    F.stage(selection, workspace / 'submission')
    def reject(*args, **kwargs):
        pytest.fail('full author overlay must not install legacy proof imports')
    monkeypatch.setattr(admission, 'original_compiler_imports', reject)
    with F.scoped_full_compiler(selection):
        argv, environment = L.sandbox_overlay(['bwrap'], workspace=workspace,
            bundle={'compiler_tool_profile': {'old': 'read-only input declaration'}},
            context=None, private_run_dir=None)
    assert argv == ['bwrap']
    assert str(workspace / 'submission' / F.DIRECTORY) in environment
    assert 'unset ' + F.ENV in environment


def test_compiler_support_assets_preserve_namespace_and_source_boundaries(selection, tmp_path):
    submission = tmp_path / 'submission'
    F.stage(selection, submission)
    root = submission / F.DIRECTORY
    for suffix in F.SUPPORT_SUFFIXES:
        path = root / ('merlin/compile/new-policy' + suffix)
        path.write_text('ordinary authored compiler configuration or source\n')
    sources = F.candidate_sources(selection, submission)
    assert 'merlin/compile/new-policy.json' in sources
    assert 'merlin/compile/new-policy.mlir' in sources
    private = root / 'merlin/runtime/reference.json'
    private.parent.mkdir(exist_ok=True)
    private.write_text('{}')
    with pytest.raises(ValueError, match='outside public namespace'):
        F.candidate_sources(selection, submission)
    private.unlink()
    binary = root / 'merlin/compile/generated.o'
    binary.write_bytes(b'ELF')
    with pytest.raises(ValueError, match='non-source member'):
        F.candidate_sources(selection, submission)
