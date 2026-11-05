"""Version mapping only: original source, permissions and scope checks stay active.

No SDK grant, Lean proof, compilation or simulation is issued by these fixtures.
The external archive boundary is supplied explicitly to exercise the actual
version-dependent routing and its independent identity/mutation refusals.
"""
from hashlib import sha256
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import compiler_profile_admission as admission_api
from merlin.targetgen import generate_bundles as bundles
from merlin.targetgen import host_optimization_admission as host
from merlin.targetgen import tool_profile
from merlin.targetgen.common_transforms import CommonTransformService
from merlin.targetgen.target_experiment import load_target_experiment
from merlin.verify import lean_emission_context as emission


def test_v13_bundle_selects_original_semantics_once_and_withdraws_legacy_task(monkeypatch):
    descriptor = load_target_experiment(
        repo_root() / 'merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml')
    selected = []
    def attach(bundle, **options):
        selected.append((bundle, options))
    monkeypatch.setattr(tool_profile, 'attach_common_substrate', attach)
    result = bundles.generate_bundles(descriptor, arms=('raw_baseline',),
        python_source_root=repo_root()/'src', common_substrate=tool_profile.COMMON_SUBSTRATE_V13)
    assert len(selected) == 1
    bundle, options = selected[0]
    assert bundle is next(iter(result.values()))
    assert options['target'] == descriptor.target
    assert options['version'] == tool_profile.COMMON_SUBSTRATE_V13
    assert 'software_spec_path' in options and 'facts_path' in options
    legacy = descriptor.experiment_resource('task')+'/'
    assert legacy not in {row['path'] for row in bundle['allowed']}
    assert any(row['path']==legacy and 'never a candidate grant' in row['note'] for row in bundle['host_inputs'])


def test_cli_accepts_exact_v13_and_refuses_unregistered_future(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(bundles,'materialize_bundles',lambda *args,**kwargs: observed.append(kwargs) or [])
    args = ['--descriptor',str(repo_root()/'merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml'),
            '--dest',str(tmp_path/'unused'), '--common-substrate',tool_profile.COMMON_SUBSTRATE_V13]
    assert bundles._main(args) == 0
    assert observed[0]['common_substrate'] == tool_profile.COMMON_SUBSTRATE_V13
    with pytest.raises(SystemExit) as error:
        bundles._main([*args[:-1],'merlin.common_compiler_substrate.v14'])
    assert error.value.code == 2
    assert len(observed) == 1


def test_v13_host_choices_retain_original_reviewed_bytes_and_permissions(tmp_path,monkeypatch):
    from merlin.common import bundle_snapshot

    source = tmp_path/'software.yaml'
    source.write_text(yaml.safe_dump({'schema':'merlin.software_spec.v1','status':'reviewed','target':'generic_fixture',
        'host_optimization':{'schema':host.POLICY_SCHEMA,'permitted_stages':['llvm'],
                            'permitted_optimizations':['static_arena'],'sequential_execution':True}}))
    digest = sha256(source.read_bytes()).hexdigest()
    verified = []
    selected = SimpleNamespace(workspace=tmp_path,bundle={},repo=repo_root(),
        profile={'common_substrate':tool_profile.COMMON_SUBSTRATE_V13,'host_index_bits':64,
                 'semantic_inputs':{'target':'generic_fixture','software_spec':{'path':str(source),'sha256':digest}}},
        verify=lambda:verified.append(True))
    def snapshot(workspace,bundle,paths,*,repo):
        assert workspace==tmp_path and bundle is selected.bundle and paths==[source] and repo==repo_root()
        return [source]
    monkeypatch.setattr(bundle_snapshot,'snapshot_input_paths',snapshot)
    context = host._original_context(selected,'original_entry')
    assert context.entry=='original_entry' and context.semantic_sha256==digest
    assert context.sequential_execution is True and context.successful_allocations is False
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError,match='original software bytes'):
        host._original_context(selected,'original_entry')
    selected.profile['common_substrate']='merlin.common_compiler_substrate.v14'
    with pytest.raises(ValueError,match='original substrate'):
        host._original_context(selected,'original_entry')
    assert len(verified)==3


@pytest.mark.parametrize('version',[tool_profile.COMMON_SUBSTRATE_V12,tool_profile.COMMON_SUBSTRATE_V13])
def test_actual_original_emission_scope_exists_and_detects_snapshot_change(tmp_path,monkeypatch,version):
    # Only the preceding archive validation is outside this focused seam. The
    # actual issued scope, identity verification and cleanup remain unchanged.
    service = object.__new__(CommonTransformService)
    service.workspace=tmp_path
    service.bundle_manifest=tmp_path/'archive.yaml'
    service.repo=repo_root()
    service.profile={'common_substrate':version}
    service.snapshot_digest='original-snapshot'
    selected = SimpleNamespace(workspace=service.workspace,bundle_manifest=service.bundle_manifest,
        repo=service.repo,profile=service.profile,snapshot_sha256=service.snapshot_digest)
    monkeypatch.setattr(admission_api,'active_compiler_imports',lambda:selected)
    monkeypatch.setattr(emission.E,'_service_selection',lambda s: ('already-validated-original-archive',))
    with emission.original_emission_checks(service):
        scope = emission.active_emission_checks()
        assert scope is not None and scope.service is service and scope.admission is selected
        selected.snapshot_sha256='foreign-snapshot'
        with pytest.raises(ValueError,match='original archive/profile/snapshot'):
            emission.active_emission_checks()
        selected.snapshot_sha256=service.snapshot_digest
    assert not scope.live and emission.active_emission_checks() is None
    selected.repo=tmp_path/'foreign-owner'
    with pytest.raises(ValueError,match='different original owner'):
        with emission.original_emission_checks(service):
            pytest.fail('foreign original owner admitted')
