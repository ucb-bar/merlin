"""Versioned author packages are real copy and sandbox boundaries, not prompt-only arms."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import full_compiler_workspace as F

from merlin.common.paths import repo_root


@pytest.fixture(scope='module')
def code_repo():
    # A normal checkout uses its own sources. The patch validation process selects
    # a disposable complete source copy before these tests are imported.
    return Path(os.environ.get('MERLIN_AUTHOR_PROFILE_TEST_REPO', str(repo_root())))


@pytest.fixture(scope='module')
def exports(code_repo, tmp_path_factory):
    root = tmp_path_factory.mktemp('profile-exports')
    return {name: A.prepare_sources(code_repo, root / name.rsplit('/', 1)[1],
                                   profile=name, review_id='test source boundaries') for name in A.SCRATCH_NAMES}


def options(level, **changes):
    fields = dict(author_profile=A.NAMES[level - 1], sandbox='bwrap',
                  compiler_execution='snapshot_external_v1', fresh_backend=True,
                  seed_submission='', qualify_submission='', joint_epoch='', with_tool=[], without_tool=[],
                  compiler_access='selected_library' if level == 1 else 'full_compiler', full_compiler_root='',
                  reference_experiments='packet.json' if level == 4 else '', reference_experiments_tasks='off')
    return SimpleNamespace(**(fields | changes))


def test_profiles_are_additive_and_keep_historical_level_mapping():
    from merlin_experiments.phase1.levels import LEVELS
    assert A.resolve('') is None
    assert [x['id'] for x in LEVELS] == ['EL1', 'EL2', 'EL3', 'EL4', 'EL3-E', 'EL4-V', 'EL5', 'EL6']
    for index in range(1, 5):
        assert A.validate_options(options(index)).level == index
    with pytest.raises(ValueError, match='unknown'):
        A.resolve('EL3')


@pytest.mark.parametrize('level,changed', [
    (1, dict(compiler_access='full_compiler')),
    (2, dict(reference_experiments='packet.json')),
    (3, dict(with_tool=['cca_tools'])),
    (4, dict(reference_experiments='')),
    (4, dict(reference_experiments_tasks='on')),
    (2, dict(seed_submission='/seed')),
    (3, dict(sandbox='copy')),
])
def test_inconsistent_grants_and_seeded_profiles_refuse(level, changed):
    with pytest.raises(ValueError):
        A.validate_options(options(level, **changed))


def test_exact_component_exports_exclude_siblings_algorithms_and_portable_kernels(exports):
    assert exports[A.NAMES[0]] is None
    cpp = exports[A.NAMES[1]]
    light = exports[A.NAMES[2]]
    modern = exports[A.NAMES[3]]
    assert not cpp.diagnostics and not light.diagnostics
    cpp_modules = {module for _, _, module in cpp.members}
    light_modules = {module for _, _, module in light.members}
    modern_modules = {module for _, _, module in modern.members}
    assert 'merlin.targetgen.generate.mlir_scaffold' in cpp_modules
    assert 'merlin.targetgen.generate.xdsl' not in cpp_modules
    assert set(A.LIGHT_ENTRIES) <= light_modules
    for name in ('merlin.targetgen.oot_starterkit.transforms', 'merlin.sched.ir',
                 'merlin.transforms.dense_planning', 'merlin.compile.scheduling.schedule_ir'):
        assert name not in light_modules
    assert 'merlin.compile.scheduling.schedule_ir' in modern_modules
    assert all(module is not None for _, _, module in light.members)
    with pytest.raises(ValueError, match='another author profile'):
        A.verify_sources(A.NAMES[2], modern)


def test_future_component_dependency_widening_fails_before_authoring(exports, tmp_path):
    original = exports[A.NAMES[1]]
    root = tmp_path / 'widened'
    shutil.copytree(original.root, root)
    extra = root / 'merlin/transforms/unreviewed.py'
    extra.parent.mkdir(parents=True)
    extra.write_text('value = 1\n')
    document = json.loads(original.manifest_path.read_text())
    import hashlib
    document['members'].append(dict(path='merlin/transforms/unreviewed.py', module='merlin.transforms.unreviewed',
                                    sha256=hashlib.sha256(extra.read_bytes()).hexdigest()))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(document))
    widened = F.select(root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='exceeds'):
        A.verify_sources(A.NAMES[1], widened)


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_host_broker_selection_cannot_be_replaced_by_author_request(level, monkeypatch):
    monkeypatch.setenv(A.ENV, A.NAMES[level - 1])
    A.check_broker_action('selfcheck')
    A.check_broker_action('mechanism-question')  # common measurement admission, not CCA assistance
    for action in ('reference-analysis', 'candidate-diagnostics', 'common-transform'):
        if level == 4:
            A.check_broker_action(action)
        else:
            with pytest.raises(FileNotFoundError):
                A.check_broker_action(action)


def test_lower_profiles_stage_no_algorithm_skills(monkeypatch):
    from merlin_experiments.phase1 import skill_staging
    monkeypatch.setattr(skill_staging, 'collect', lambda: pytest.fail('must not discover modern skills'))
    selected = skill_staging.selection(SimpleNamespace(driver='codex', author_skills='on', author_profile=A.NAMES[2]))
    assert selected['enabled'] is False


def test_modern_feedback_stays_out_of_lower_profiles(monkeypatch):
    from merlin_experiments.phase1.feedback import family_rollup, next_steps, performance_brief
    monkeypatch.setenv(A.ENV, A.NAMES[0])
    assert performance_brief.build([{'all_pass': True}]) is None
    assert family_rollup.build(rows=[{'all_pass': True}]) is None
    result = {'cases': [{'cycles': 23, 'all_pass': True}]}
    next_steps.attach(result, [], None)
    assert result == {'cases': [{'cycles': 23, 'all_pass': True}]}


def test_new_task_does_not_inherit_historical_whole_model_or_tool_grants():
    body = A.task(A.resolve(A.NAMES[0]))
    assert 'xs capsule' in body
    assert '20 CPUs' not in body and '16 helpers' not in body
    assert 'merlin.transforms' not in body and 'merlin.compile.scheduling' not in body
    assert 'complete models' not in body


def test_lazy_packages_keep_historical_helpers_and_do_not_import_siblings():
    script = '''import sys
from merlin.targetgen.generate import mlir_scaffold
assert "merlin.targetgen.generate.xdsl" not in sys.modules
from merlin.targetgen.oot_starterkit import CommandBufferBuilder, verify_module
assert "merlin.targetgen.oot_starterkit.transforms" not in sys.modules
from merlin.targetgen.oot_starterkit import transforms
assert transforms.tile_to_dim(2, 2, 2, 2)
assert callable(verify_module)
'''
    subprocess.run([sys.executable, '-B', '-c', script], check=True, timeout=20)


def _bwrap():
    binary = shutil.which('bwrap')
    if binary is None:
        explicit = Path('/work/merlin/hw/bubblewrap-0.11.0/bin/bwrap')
        binary = str(explicit) if explicit.is_file() else None
    if binary is None:
        pytest.skip('bubblewrap is unavailable; OS isolation not tested')
    return binary


@pytest.mark.parametrize('level', [1, 2, 3, 4])
def test_real_bubblewrap_author_and_helper_module_boundaries(exports, tmp_path, level):
    # Same final mount mask function is used by the author/helper and candidate
    # compiler paths. Exercise it against actual Python installations and aliases.
    binary = _bwrap()
    args = [binary, '--die-with-parent', '--unshare-pid', '--proc', '/proc', '--dev', '/dev']
    for root in ['/usr', '/bin', '/lib', '/lib64', '/etc']:
        if Path(root).exists():
            args += ['--ro-bind', root, root]
    runtime_roots = {Path(sys.executable).absolute().parent.parent,
                     Path(sys._base_executable).absolute().parent.parent}
    for root in sorted(runtime_roots):
        args += ['--ro-bind', str(root), str(root)]
    selection = exports[A.NAMES[level - 1]]
    sdk = selection.root if selection is not None else tmp_path / 'empty-sdk'
    sdk.mkdir(exist_ok=True)
    args += ['--ro-bind', str(sdk), '/author-sdk']
    bundle = dict(author_profile=A.resolve(A.NAMES[level - 1]).record(),
                  tools=list(A.resolve(A.NAMES[level - 1]).tools), allowed=[])
    args = A.runtime_masks(args, bundle)
    args += ['--setenv', 'PYTHONPATH', '/author-sdk', '--setenv', 'PYTHONDONTWRITEBYTECODE', '1',
             '--setenv', 'PYTHONNOUSERSITE', '1', '--chdir', '/']
    positive = [] if level == 1 else ['merlin.targetgen.generate.mlir_scaffold']
    if level >= 3:
        positive += ['xdsl.ir', 'merlin.targetgen.oot_starterkit.llvm_context']
    if level == 4:
        positive += ['merlin.transforms.dense_planning', 'merlin.compile.scheduling.schedule_ir']
    negative = ['merlin_experiments.phase1.session', 'merlin.runtime.reference']
    if level == 1:
        negative += ['merlin.transforms', 'xdsl.ir']
    if level == 2:
        negative += ['xdsl.ir', 'merlin.targetgen.generate.xdsl']
    if level < 4:
        negative += ['merlin.sched.ir', 'merlin.transforms.dense_planning', 'merlin.targetgen.oot_starterkit.transforms']
    script = 'import importlib\n'
    script += '\n'.join('importlib.import_module(' + repr(name) + ')' for name in positive) + '\n'
    for name in negative:
        script += ('try:\n importlib.import_module(' + repr(name) + ')\n'
                   'except ModuleNotFoundError:\n pass\nelse:\n raise AssertionError(' + repr(name) + ')\n')
    run = subprocess.run([*args, sys.executable, '-B', '-c', script], text=True, capture_output=True, timeout=30)
    if run.returncode and any(message in run.stderr for message in ('Operation not permitted', 'No permissions')):
        pytest.skip('kernel disallows bubblewrap namespaces: ' + run.stderr[:150])
    assert run.returncode == 0, run.stderr


def _base_bundle(code_repo, tmp_path):
    from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V1, attach_common_substrate
    common = tmp_path / 'raw-isa.h'
    common.write_text('/* raw ISA */\n')
    bundle = dict(arm='merlin_assisted', bundle_id='fixture_public_v0', variant='public_v0',
                  allowed=[dict(path=str(common), mode='ro', note='raw ISA')], denied=[], host_inputs=[],
                  tools=['isa_tools', 'cca_tools'])
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V1, python_source_root=code_repo / 'src')
    return bundle, dict(path=str(common), mode='ro', note='raw ISA')


def test_profile_replaces_all_sdk_grants_but_keeps_same_private_judge(code_repo, exports, tmp_path):
    from merlin.targetgen.tool_profile import verify_tool_profile
    base, common = _base_bundle(code_repo, tmp_path)
    selected = A.materialize_bundle(base, profile=A.NAMES[1], common_allowed=[common],
                                    source_selection=exports[A.NAMES[1]])
    assert selected['allowed'] == [common]
    assert selected['tools'] == ['cpp_oot_generators']
    assert selected['compiler_tool_profile']['proof_policy'] == base['compiler_tool_profile']['proof_policy']
    assert selected['compiler_tool_profile']['common_sources'] == base['compiler_tool_profile']['common_sources']
    withheld = {row['path'] for row in base['allowed']} - {common['path']}
    assert withheld <= {row['path'] for row in selected['host_inputs']}
    assert selected['author_profile_sources'] == exports[A.NAMES[1]].record()
    verify_tool_profile(selected, tools=('cpp_oot_generators',))
    A.verify_bundle(A.resolve(A.NAMES[1]), selected)
    with pytest.raises(ValueError, match='SDK packages'):
        A.materialize_bundle(base, profile=A.NAMES[1], common_allowed=[dict(path=str(code_repo / 'src'), mode='ro')],
                             source_selection=exports[A.NAMES[1]])


def test_canonical_common_uses_relocated_base_rows_and_excludes_task_sdk(code_repo, tmp_path, monkeypatch):
    from merlin.targetgen import generate_bundles
    raw = tmp_path / 'public.h'; raw.write_text('raw')
    facts = tmp_path / 'facts'; facts.mkdir(); (facts / 'facts.json').write_text('{}')
    llvm = tmp_path / 'llvm'; llvm.mkdir()
    te = SimpleNamespace(experiment_resource=lambda name: '/released/' + name)
    monkeypatch.setattr(generate_bundles, '_shared_allow', lambda te, variant: [
        dict(path=str(raw), mode='ro'), dict(path='/released/task/', mode='ro'),
        dict(path='third_party/llvm-install/', mode='ro', note='LLVM/MLIR 23 toolchain')])
    rows = [dict(path=str(raw), mode='ro'),
            dict(path=str(llvm)+'/', mode='ro', note='LLVM/MLIR 23 toolchain'),
            dict(path=str(facts)+'/', mode='ro'), dict(path='/released/task/', mode='ro'),
            dict(path=str(code_repo/'src'), mode='ro', note='common compiler infrastructure')]
    selected = A.canonical_common(dict(allowed=rows, selected_rtl_facts_file=str(facts/'facts.json')), te, 'public_v0')
    assert selected == rows[:3]


def test_seeded_profile_binds_explicit_sources_and_never_prepares_latest(code_repo, exports, tmp_path):
    original = exports[A.NAMES[2]]
    selected = A.prepare_sources(code_repo, tmp_path/'unused', profile=A.SEEDED_NAMES[0],
                                 review_id='seeded fixture', source_selection=original.record())
    assert selected.record() == original.record()
    assert not (tmp_path/'unused').exists()
    base, common = _base_bundle(code_repo, tmp_path)
    bundle = A.materialize_bundle(base, profile=A.SEEDED_NAMES[0], common_allowed=[common], source_selection=selected)
    A.verify_sources(A.SEEDED_NAMES[0], selected, bundle=bundle)
    with pytest.raises(ValueError, match='sealed author profile'):
        A.verify_sources(A.SEEDED_NAMES[0], exports[A.NAMES[1]], bundle=bundle)
    spec = options(4, author_profile=A.SEEDED_NAMES[0], seed_submission='/original-frozen',
                   reference_experiments='', fresh_backend=False)
    assert A.validate_options(spec).seeded
    assert A.resolve(A.SEEDED_NAMES[0]).tools == ('isa_tools',)


def test_cpp_profile_is_checked_at_actual_candidate_entrypoint():
    from merlin_experiments.phase1.feedback.snapshot_execution import SnapshotExecutor
    executor = object.__new__(SnapshotExecutor)
    executor.live = True; executor.pid = os.getpid(); executor.candidate = Path('/candidate')
    executor.owner = SimpleNamespace(bundle=dict(author_profile=A.resolve(A.NAMES[0]).record()))
    pkg = SimpleNamespace(directory=Path('/candidate'), integrity_exempt=False, language='python', manifest={})
    with pytest.raises(ValueError, match='C..'):
        executor._package(pkg)
    pkg.language = 'cpp'; pkg.manifest = dict(build={'argv': ['cmake', '--build']})
    executor._package(pkg)


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("mount_project", [False, True])
def test_real_alternate_python_environment_cannot_restore_ambient_packages(exports, tmp_path, level, mount_project):
    """A different real interpreter/package tree and its alias obey the same policy."""
    import sysconfig
    import venv

    import xdsl

    project = tmp_path / "alternate-project"
    environment = project / ".venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    site = environment / "lib" / ("python" + sysconfig.get_python_version()) / "site-packages"
    shutil.copytree(Path(xdsl.__file__).parent, site / "xdsl",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("merlin", "merlin_experiments"):
        package = site / name
        package.mkdir()
        (package / "__init__.py").write_text("AMBIENT_MARKER = True\n")
    source, destination = (project, Path("/alternate-project")) if mount_project else (environment, Path("/alternate-runtime"))
    guest_env = destination / ".venv" if mount_project else destination
    args = [_bwrap(), "--die-with-parent", "--unshare-pid", "--proc", "/proc", "--dev", "/dev"]
    for root in ["/usr", "/bin", "/lib", "/lib64", "/etc"]:
        if Path(root).exists():
            args += ["--ro-bind", root, root]
    for root in sorted({Path(sys.executable).absolute().parent.parent,
                        Path(sys._base_executable).absolute().parent.parent}):
        args += ["--ro-bind", str(root), str(root)]
    args += ["--ro-bind", str(source), str(destination), "--chdir", "/",
             "--setenv", "PYTHONNOUSERSITE", "1", "--setenv", "PYTHONDONTWRITEBYTECODE", "1"]
    binary = str(guest_env / "bin/python")
    before = subprocess.run([*args, "--setenv", "PYTHONPATH", "", binary, "-B", "-c",
        "import merlin, merlin_experiments, xdsl; assert merlin.AMBIENT_MARKER and merlin_experiments.AMBIENT_MARKER"],
        text=True, capture_output=True, timeout=30)
    if before.returncode and any(message in before.stderr for message in ("Operation not permitted", "No permissions")):
        pytest.skip("kernel disallows bubblewrap namespaces: " + before.stderr[:150])
    assert before.returncode == 0, before.stderr
    selection = exports[A.NAMES[level - 1]]
    sdk = selection.root if selection is not None else tmp_path / "empty-sdk"
    sdk.mkdir(exist_ok=True)
    args += ["--ro-bind", str(sdk), "/author-sdk"]
    bundle = dict(author_profile=A.resolve(A.NAMES[level - 1]).record(),
                  tools=list(A.resolve(A.NAMES[level - 1]).tools), allowed=[])
    args = A.runtime_masks(args, bundle)
    checks = "import importlib\n"
    for name in ("merlin_experiments", *(("merlin",) if level == 1 else ()), *(("xdsl",) if level < 3 else ())):
        # A masked namespace may remain as an empty directory; no module code or
        # stored implementation may remain accessible through the mount alias.
        checks += ("try:\n m=importlib.import_module(" + repr(name) + ")\n"
                   "except ModuleNotFoundError:\n pass\nelse:\n assert getattr(m, '__file__', None) is None\n")
    if level > 1:
        checks += "from merlin.common.paths import repo_root; assert callable(repo_root)\n"
    if level == 3:
        checks += "import xdsl; assert xdsl.__file__.startswith(" + repr(str(guest_env)) + ")\n"
    guest_site = guest_env / "lib" / ("python" + sysconfig.get_python_version()) / "site-packages"
    for name in ("merlin", "merlin_experiments", *(("xdsl",) if level < 3 else ())):
        checks += ("try:\n data=open(" + repr(str(guest_site / name / "__init__.py")) + ", 'rb').read(1)\n"
                   "except OSError:\n pass\nelse:\n assert not data\n")
    after = subprocess.run([*args, "--setenv", "PYTHONPATH", "/author-sdk", binary, "-B", "-c", checks],
                           text=True, capture_output=True, timeout=30)
    assert after.returncode == 0, after.stderr


def test_canonical_common_preserves_pinned_ordinary_os_runtime(tmp_path, monkeypatch):
    from merlin.targetgen import generate_bundles

    raw = tmp_path / "raw.h"; raw.write_text("public ISA")
    runtime = tmp_path / "libc.so.6"; runtime.write_bytes(b"pinned ordinary runtime")
    sdk = tmp_path / "private-sdk.py"; sdk.write_text("private_compiler = True\n")
    te = SimpleNamespace(experiment_resource=lambda name: "/released/" + name)
    monkeypatch.setattr(generate_bundles, "_shared_allow", lambda te, variant: [dict(path=str(raw), mode="ro")])
    rows = [dict(path=str(raw), mode="ro"),
            dict(path=str(runtime), mode="ro", note="selected ordinary OS runtime; not formal authoring capability"),
            dict(path=str(sdk), mode="ro", note="common compiler infrastructure")]
    assert A.canonical_common(dict(allowed=rows), te, "public_v0") == rows[:2]


def test_canonical_common_preserves_exact_admitted_software_data(tmp_path, monkeypatch):
    from merlin.targetgen import generate_bundles

    raw = tmp_path / "raw.h"; raw.write_text("public ISA")
    semantics = tmp_path / "software-spec.yaml"; semantics.write_text("public: semantics\n")
    te = SimpleNamespace(experiment_resource=lambda name: "/released/" + name)
    monkeypatch.setattr(generate_bundles, "_shared_allow", lambda te, variant: [dict(path=str(raw), mode="ro")])
    rows = [dict(path=str(raw), mode="ro"),
            dict(path=str(semantics), mode="ro", note="admitted software semantics")]
    bundle = dict(allowed=rows, compiler_tool_profile=dict(semantic_inputs=dict(software_spec=dict(
        path=str(semantics), sha256=hashlib.sha256(semantics.read_bytes()).hexdigest()))))
    assert A.canonical_common(bundle, te, "public_v0") == rows
    semantics.write_text("changed: true\n")
    with pytest.raises(ValueError, match="exact admitted declaration"):
        A.canonical_common(bundle, te, "public_v0")
