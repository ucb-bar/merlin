"""Role-based compiler-only author tools; no native simulator or model run."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.targetgen.sandbox import toolchain as TC
from merlin_experiments.phase1 import author_profiles as AP
from merlin_experiments.phase1 import author_toolchain as A
from merlin_experiments.phase1.providers import execution as E


@pytest.fixture
def resources(tmp_path, monkeypatch):
    def directory(name):
        path = tmp_path/name
        path.mkdir()
        return str(path)

    paths = TC.ToolchainPaths(repo=tmp_path, venv=directory('language'), llvm=directory('compiler'),
        compat_lib=directory('compat'), clang_install=directory('clang'), uv_python=directory('runner'),
        python_import_roots=())
    simulator, harness = directory('opaque-device-resources'), directory('declared-runtime-inputs')
    sim = TC.SimToolchain(bind_paths=(simulator,), path_dirs=(simulator+'/commands',),
        ld_dirs=(simulator+'/libraries',), env_extra={'DECLARED_DEVICE_TOOLS':simulator},
        probes=(TC.ToolProbe('device executor', 'must-not-be-run', simulator),))
    monkeypatch.setattr(TC.ToolchainPaths, 'from_checkout', lambda:paths)
    monkeypatch.setattr(TC, '_sim', lambda target:sim)
    monkeypatch.setattr(TC, 'curated_harness_dir', lambda target:harness)
    monkeypatch.setattr(TC, 'paths_for_bundle', lambda ws,bundle,paths=None,_paths=paths:paths or _paths)
    import importlib
    answer_surfaces=importlib.import_module('merlin.targetgen.sandbox.answer_surfaces')
    monkeypatch.setattr(answer_surfaces, 'experimenter_memory_dir', lambda:None)
    target = SimpleNamespace(target='alternate_device')
    profile = SimpleNamespace(name='selected-information-policy', disclosure={'rtl_facts':False})
    bundle = {'author_profile':{'name':profile.name, 'disclosure':{'rtl_facts':False}}}
    monkeypatch.setattr(AP, 'resolve', lambda name:profile if name==profile.name else None)

    def verify(selected, actual):
        if actual['author_profile'] != {'name':profile.name,'disclosure':profile.disclosure}:
            raise ValueError('original information selection changed')

    monkeypatch.setattr(AP, 'verify_bundle', verify)
    monkeypatch.setattr(AP, 'withheld_simulator_roots',
        lambda target,bundle:tuple(map(Path,(simulator,harness))), raising=False)
    return SimpleNamespace(target=target,bundle=bundle,profile=profile,paths=paths,sim=sim,
        simulator=simulator,harness=harness,workspace=tmp_path/'workspace')


def test_restricted_selection_removes_simulator_harness_and_probes_but_keeps_compilers(resources):
    r = resources
    selected = A.select(r.target,r.bundle)
    assert selected.record()['roles']==['compiler']
    assert selected.withheld_roots==tuple(map(Path,(r.simulator,r.harness)))
    binds = A.toolchain_binds(r.target,r.bundle)
    assert r.paths.llvm in binds and r.paths.venv in binds
    assert r.simulator not in binds and r.harness not in binds
    assert ('--unsetenv','DECLARED_DEVICE_TOOLS') in list(zip(binds,binds[1:]))
    environment = A.sandbox_env(r.target,r.workspace,bundle=r.bundle)
    assert r.simulator not in environment and r.harness not in environment
    assert 'unset DECLARED_DEVICE_TOOLS;' in environment
    assert 'unset MERLIN_ALTERNATE_DEVICE_HARNESS_DIR;' in environment
    assert [p.label for p in A.required_tool_probes(r.target,bundle=r.bundle)]==['python3','mlir-opt','clang-23']


def test_host_judge_retains_declared_simulator_resources_unchanged(resources):
    r=resources
    # The restricted selection must not edit the registered role or global defaults.
    A.toolchain_binds(r.target,r.bundle)
    assert r.simulator in TC.toolchain_binds(r.target)
    assert r.harness in TC.toolchain_binds(r.target)
    assert TC.required_tool_probes(r.target)[-1].label=='device executor'
    assert r.simulator in TC.sandbox_env(r.target,r.workspace)


@pytest.mark.parametrize('mode',['unspecified','legacy','facts-allowed'])
def test_unrestricted_paths_environment_and_probes_are_byte_preserved(resources,mode):
    r=resources
    if mode=='unspecified':bundle={}
    else:
        r.profile.disclosure=None if mode=='legacy' else {'rtl_facts':True}
        bundle={'author_profile':{'name':r.profile.name,'disclosure':r.profile.disclosure}}
    assert A.toolchain_binds(r.target,bundle)==TC.toolchain_binds(r.target)
    assert A.sandbox_env(r.target,r.workspace,bundle=bundle)==TC.sandbox_env(r.target,r.workspace,bundle=bundle)
    assert A.required_tool_probes(r.target,bundle=bundle)==TC.required_tool_probes(r.target)


def test_forged_disclosure_refuses_before_any_resource_selection(resources,monkeypatch):
    r=resources
    changed={'author_profile':{'name':r.profile.name,'disclosure':{'rtl_facts':True}}}
    monkeypatch.setattr(TC,'_sim',lambda *_:pytest.fail('resource selection before exact-profile check'))
    for call in [lambda:A.toolchain_binds(r.target,changed),
                 lambda:A.sandbox_env(r.target,r.workspace,bundle=changed),
                 lambda:A.required_tool_probes(r.target,bundle=changed)]:
        with pytest.raises(ValueError,match='selection changed'):call()


def test_invalid_adapter_environment_name_refuses(resources):
    resources.sim.env_extra['BAD;COMMAND']='unused'
    with pytest.raises(ValueError,match='shell identifiers'):
        A.select(resources.target,resources.bundle)


def test_runtime_verifier_consumes_same_restricted_selection_and_refuses_broader_record(resources,monkeypatch):
    r=resources
    from merlin.targetgen import target_experiment
    monkeypatch.setattr(target_experiment,'load_target_experiment',lambda _:r.target)
    runtime=SimpleNamespace(verify=lambda:None,
        toolchain_mounts=tuple(A.toolchain_binds(r.target,r.bundle)),
        tool_environment=A.sandbox_env(r.target,r.workspace,bundle=r.bundle))
    context=SimpleNamespace(descriptor=Path('selected-descriptor'))
    E._verify_codex_runtime(runtime,context,r.workspace,r.bundle)
    runtime.toolchain_mounts=tuple(TC.toolchain_binds(r.target))
    with pytest.raises(ValueError,match='mount selection changed'):
        E._verify_codex_runtime(runtime,context,r.workspace,r.bundle)


def test_codex_permission_selection_never_advertises_withheld_resource_roots(resources,monkeypatch,tmp_path):
    r=resources
    from merlin_experiments.phase1.providers import codex_runtime as R
    from merlin.targetgen import target_experiment
    from merlin.targetgen.sandbox import bwrap as BW

    monkeypatch.setattr(R,'selection_record',lambda _:dict(binary=str(tmp_path/'client'),
        auth_source=str(tmp_path/'credentials'),home_root=str(tmp_path/'homes'),
        helper_path=str(tmp_path/'helper')))
    monkeypatch.setattr(target_experiment,'load_target_experiment',lambda _:r.target)
    public=tmp_path/'public-input'
    public.write_text('mathematical input')
    monkeypatch.setattr(BW,'_snapshot_grants',lambda *args:({},[('public-input',public,public)]))
    runtime=R.select(object(),SimpleNamespace(descriptor=tmp_path/'descriptor',repo=tmp_path),
        r.bundle,r.workspace,tmp_path/'private-run')
    assert str(public) in runtime.read_paths
    assert r.paths.llvm in runtime.read_paths
    assert not any(Path(path).is_relative_to(Path(root)) for path in runtime.read_paths
                   for root in (r.simulator,r.harness))
    assert runtime.toolchain_mounts==tuple(A.toolchain_binds(r.target,r.bundle))
    assert runtime.tool_environment==A.sandbox_env(r.target,r.workspace,bundle=r.bundle)


def test_candidate_sandbox_factory_requires_only_permitted_author_tools(resources,monkeypatch):
    r=resources
    from merlin.targetgen import sandbox as SB

    monkeypatch.setattr(SB,'load_target_experiment',lambda _:r.target)
    monkeypatch.setattr(SB,'resolve_kind',lambda _:None)
    monkeypatch.setattr(SB,'answer_surfaces',lambda _:[])
    candidate=SB.build_sandbox(Path('selected-descriptor'),r.workspace,r.bundle)
    assert [p.label for p in candidate.required_tools]==['python3','mlir-opt','clang-23']
    assert candidate.env()==A.sandbox_env(r.target,r.workspace,bundle=r.bundle)


def test_acceptance_child_consumes_actual_manifest_profile_for_tools_and_final_masks(resources,monkeypatch,tmp_path):
    import json
    r=resources
    from merlin_experiments.phase1 import author_acceptance as acceptance
    from merlin_experiments.phase2 import compiler_libraries
    from merlin.targetgen import target_experiment
    from merlin.targetgen.sandbox import bwrap as BW

    manifest=tmp_path/'bundle.json'
    manifest.write_text(json.dumps(r.bundle))
    monkeypatch.setattr(target_experiment,'load_target_experiment',lambda _:r.target)
    monkeypatch.setattr(BW,'base_argv',lambda *args,**kwargs:[])
    observed=[]
    def masks(argv,target,workspace,bundle,**kwargs):
        assert bundle==r.bundle
        observed.append(bundle)
        return argv
    monkeypatch.setattr(BW,'apply_final_answer_masks',masks)
    monkeypatch.setattr(compiler_libraries,'_environment_prefix',lambda *args:'')
    result=[]
    exec(compile(acceptance._SANDBOX_ARGV,'acceptance-probe','exec'),
         {'sys':SimpleNamespace(argv=['probe',str(r.workspace),'descriptor',str(tmp_path),str(manifest)]),
          '_emit':result.append})
    assert observed==[r.bundle]
    assert r.simulator not in result[0]['argv'] and r.harness not in result[0]['argv']
    assert r.simulator not in result[0]['env'] and r.harness not in result[0]['env']
