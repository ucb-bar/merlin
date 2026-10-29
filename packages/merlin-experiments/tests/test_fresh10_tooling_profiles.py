"""Capability removal is a source/service boundary over the deployed fresh10 union."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import fresh10_profiles as P
from merlin_experiments.phase1 import fresh10_sources as S
from merlin_experiments.phase1 import fresh10_disclosure as D
from merlin_experiments.phase1 import full_compiler_workspace as F

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope='module')
def selections(tmp_path_factory):
    root = tmp_path_factory.mktemp('fresh10-projections')
    return {name: A.prepare_sources(REPO, root / name.rsplit('/', 1)[-1], profile=name,
                                   review_id='test exact reviewed fresh10 source projection') for name in P.NAMES}


def test_original_profile_records_and_levels_stay_fixed():
    for level, name in enumerate((*A.SCRATCH_NAMES, *A.SEEDED_NAMES), 1):
        profile = A.resolve(name)
        assert profile.level == level
        assert not profile.fresh10
        assert profile.raw_kernel_references
    assert [A.resolve(name).level for name in P.NAMES] == [4, 4, 4]
    assert all(not A.resolve(name).modern for name in P.NAMES)


@pytest.mark.parametrize('name', P.NAMES)
def test_services_and_reference_scope(name, monkeypatch):
    profile = A.resolve(name)
    monkeypatch.setenv(A.ENV, name)
    assert profile.tools == P.TOOLS
    assert not {'cca_spine', 'cca_tools'} & set(profile.tools)
    for action in ('reference-analysis', 'kernel-learning'):
        with pytest.raises(FileNotFoundError): A.check_broker_action(action)
    for action in ('selfcheck', 'compile-only', 'caller-layout', 'candidate-diagnostics', 'mechanism-question'):
        A.check_broker_action(action)
    for action in ('common-transform', 'common-check'):
        if profile.allows('transform_library'): A.check_broker_action(action)
        else:
            with pytest.raises(FileNotFoundError): A.check_broker_action(action)
    assert profile.raw_kernel_references == (name != P.NAMES[2])
    from merlin_experiments.phase1.feedback import reference_baselines, seed_timing
    assert reference_baselines.load(Path('/does-not-exist')) is None
    assert seed_timing.baseline(Path('/does-not-exist')) == {}
    assert seed_timing.carried(Path('/does-not-exist'), capsule='x', key={}) is None


def test_exact_inventory_and_nested_semantic_delta(selections):
    modules = [{m for _, _, m in selections[name].members if m} for name in P.NAMES]
    assert list(map(len, modules)) == [304, 324, 1083]
    assert modules[0] < modules[1] < modules[2]
    assert len(modules[1] - modules[0]) == 20
    for selected in modules:
        assert 'merlin.targetgen.oot_starterkit.verify' in selected
        assert 'merlin.xdsl_dialects.cca' not in selected
    assert 'merlin.sched.ir.kernel' not in modules[0]
    assert 'merlin.sched.ir.kernel' in modules[1]
    assert 'merlin.compile.analysis.schedule_structure' in modules[1]
    for absent in ('merlin.sched.primitives', 'merlin.transforms.tensor_views',
                   'merlin.compile.scheduling.dense_panel_epochs'):
        assert absent not in modules[1]
        assert absent in modules[2]
    # Every common A/B source byte, including four reviewed existing namespace
    # facades and common schema repair, is identical rather than reimplemented.
    a = {p: h for p, h, _ in selections[P.NAMES[0]].members}
    b = {p: h for p, h, _ in selections[P.NAMES[1]].members}
    assert all(b[p] == h for p, h in a.items())


@pytest.mark.parametrize('name', P.NAMES)
def test_real_source_and_readonly_sdk_imports_are_masked_and_usable(name, selections, tmp_path):
    from merlin.targetgen.compiler_library import CompilerLibraryContract, CompilerLibrarySelection, LibraryMember
    selected = selections[name]
    rows = S._rows(name, 'sdk')
    sdk = tmp_path / 'sdk'
    for row in rows:
        path = sdk / row['path']; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(S._payload(REPO, row))
    data = S._data()['profiles'][S._arm(name)]
    contract = CompilerLibraryContract('test reviewed retained SDK', tuple(data['sdk_public_modules']),
                tuple(LibraryMember(r['path'], r['sha256'], r['module']) for r in rows))
    S.verify_library(name, CompilerLibrarySelection(contract, sdk))
    # -I discards ambient PYTHONPATH; merlin is loaded only from the exact
    # selected copy, so installed Merlin cannot satisfy a missing dependency.
    script = '''import hashlib,importlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); inventory=json.loads(Path(sys.argv[2]).read_text())
sys.path[:]=[str(root),*[p for p in sys.path if 'site-packages' in p or ('merlin' not in p and p)]]
import merlin
assert list(map(Path,merlin.__path__))==[root/'merlin']
for name in inventory['entry_modules']:
 module=importlib.import_module(name); path=Path(module.__file__).resolve()
 assert path.is_relative_to(root),(name,str(path))
for name in ('merlin.kernels','merlin.mining','merlin.xdsl_dialects.cca','merlin.runtime.reference'):
 try: importlib.import_module(name)
 except ModuleNotFoundError: pass
 else: raise AssertionError('withheld '+name)
'''
    if name != P.NAMES[2]:
        script += '''
for name in ('merlin.compile.scheduling.dense_panel_epochs','merlin.sched.primitives','merlin.transforms.tensor_views'):
 try: importlib.import_module(name)
 except ModuleNotFoundError: pass
 else: raise AssertionError('withheld '+name)
'''
    if name != P.NAMES[0]:
        from merlin_experiments.phase1.fresh10_semantics_probe import command
        script += command()
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(selected.root),
                             str(selected.manifest_path)], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_extra_source_or_changed_bytes_fail_before_admission(selections, tmp_path):
    import shutil
    selected = selections[P.NAMES[0]]
    copy = tmp_path / 'copy'; shutil.copytree(selected.root, copy)
    extra = copy / 'merlin/transforms/hidden.py'; extra.parent.mkdir(exist_ok=True);extra.write_text('x=1\n')
    with pytest.raises(ValueError, match='membership'):
        S._exact_files(copy, S._rows(P.NAMES[0], 'full'))
    extra.unlink()
    (copy / 'merlin/__init__.py').chmod(0o644)
    (copy / 'merlin/__init__.py').write_text('x=1\n')
    with pytest.raises(ValueError, match='membership'):
        S._exact_files(copy, S._rows(P.NAMES[0], 'full'))


def test_all_actual_skill_files_parse_and_keep_fenced_blocks_whole():
    from merlin_experiments.phase1 import skill_staging as K
    original = K.collect()
    outputs=[]
    for name in P.NAMES:
        skills = D.skill_sources(original, name);outputs.append(skills)
        assert len(skills) == (20 if name == P.NAMES[2] else 11)
        for skill, files in skills.items():
            for relative, data in files.items():
                if relative.endswith('.py'): ast.parse(data)
                else:
                    D.filter_markdown(data.decode(), library=True, references=True)
                    if relative == 'SKILL.md': K.frontmatter(data.decode())
                assert b'--reference-analysis' not in data
    assert outputs[0] == outputs[1]
    text='Keep.\n\n```python\nimport merlin.transforms.tensor_views\n\nprint(1)\n```\n\nKeep again.\n'
    assert D.filter_markdown(text, library=False, references=False) == 'Keep.\n\nKeep again.\n'


def test_common_task_retains_models_numerics_placement_and_isolates_refs(tmp_path):
    original='''# Fresh compiler\nAll six programs must compile and link. ResNet50 model; SmolVLA prefix_encode flow_denoise action_decode; TinyLlama prefill decode.\n\n## Deliverable\nPreserve bit-exact integer reference, approximations, four entrypoints and manifest.\n\n## Model deployment and numerical boundaries\nPreserve all live outputs, source-bound placement, numerical rounding, saturation and offload census.\n\n## Joint editable compiler development\nProhibited instruction role `loop_descriptor`. Complete-call cycles include host packing and synchronization.\n\n## Reference map\nRead REFERENCE_MAP.json in reference-kernels/.\n\n## Scheduling integration and shared-pass development\nRead merlin.transforms.tensor_views.\n'''
    records=[]
    for index,name in enumerate(P.NAMES):
        ws=tmp_path/str(index);ws.mkdir();run=ws/'run';run.mkdir();(ws/'TASK.md').write_text(original)
        record=D.stage_task(ws,run,name,SimpleNamespace());records.append(record)
        text=(ws/'TASK.md').read_text()
        for required in ('All six programs','prefix_encode','prefill','bit-exact','offload census','loop_descriptor','Complete-call cycles'):
            assert required in text
        assert ('Read REFERENCE_MAP.json' in text) == A.resolve(name).raw_kernel_references
        assert 'Read merlin.transforms.tensor_views' not in text
    assert len({r['common_core_sha256'] for r in records}) == 1


@pytest.mark.parametrize('name', P.NAMES[:2])
def test_actual_feedback_retains_diagnostics_without_withheld_apis(name, tmp_path, monkeypatch):
    import composition_fixtures as fixtures
    from test_idle_tuning import _performance
    from merlin_experiments.phase1.feedback import performance_brief as brief
    monkeypatch.setenv(A.ENV, name)
    fixtures.declare(tmp_path, 'layers', fixtures.conv_pool('chain'))
    (tmp_path/'submission').mkdir();(tmp_path/'submission/backend.py').write_text('import json\n')
    (tmp_path/'PUBLIC_CAPABILITIES.json').write_text(json.dumps({'schema':'merlin.public_capability_inventory.v1',
        'members':{'chain':{'workload_role':'connected'}}}))
    rows=[{'capsule':'chain','pass':True,'performance':_performance(400)}]
    result=brief.attach({'per_capsule':rows},rows,workspace=tmp_path)
    encoded=json.dumps(result)
    assert 'performance_brief' in encoded
    for token in ('merlin.compile.scheduling','heavy-pass-recipes','apply_library_pass','ab-library-pass','pass-library'):
        assert token not in encoded


@pytest.mark.parametrize('name', P.NAMES)
def test_fresh10_prompt_uses_its_own_contract_before_historical_profile_fallback(name, monkeypatch):
    from merlin_experiments.phase1.providers import authoring_prompt as owner

    monkeypatch.setenv(A.ENV, name)
    assert owner.profile_authoring_prompt() is None
    assert owner.profile_authoring_prompt(continuation=True) is None
    text = owner.authoring_prompt()
    assert text.startswith('Construct one general compiler under the selected fresh10 contract')


def test_c_receives_real_library_guidance_and_a_does_not(selections, tmp_path, monkeypatch):
    from merlin.targetgen.compiler_library import CompilerLibraryContract, CompilerLibrarySelection, LibraryMember
    from merlin_experiments.phase1 import compiler_library as library
    calls=[]
    monkeypatch.setattr(library,'_performance_guidance',lambda *args,**kwargs: calls.append(args) or
        '## Library guidance\n\nUse merlin.transforms.tensor_views from the selected library.\n')
    for index,name in enumerate((P.NAMES[0],P.NAMES[2])):
        monkeypatch.setenv(A.ENV,name)
        ws=tmp_path/str(index);ws.mkdir();run=ws/'run';run.mkdir();sdk=ws/'sdk'
        data=S._data()['profiles'][S._arm(name)];rows=data['sdk_members']
        for row in rows:
            path=sdk/row['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(S._payload(REPO,row))
        selected=CompilerLibrarySelection(CompilerLibraryContract('test retained library',tuple(data['sdk_public_modules']),
            tuple(LibraryMember(r['path'],r['sha256'],r['module']) for r in rows)),sdk)
        with F.scoped_full_compiler(selections[name]):
            library.append_guidance(ws,run,None,selected,repo=REPO)
        assert (ws/'FULL_COMPILER_SOURCE_SELECTION.json').is_file()
        assert (ws/'SELECTED_LIBRARY_GUIDANCE.md').exists() == (name==P.NAMES[2])
    assert len(calls)==1


@pytest.mark.parametrize('name', P.NAMES)
def test_common_public_data_under_reference_named_run_is_not_reference_material(name, tmp_path):
    # Reproduce the actual failing generation's run-ID ancestor. None of these
    # eleven public ABI/toolchain leaves is raw reference material.
    run = tmp_path / 'fresh10-tooling-20261028-r2-no-reference-kernels'
    common = ('contracts/target_contract.yaml', 'interface_grammar.md',
              'mlir_oot_backend_contract.yaml', 'interface_dialect_contract.yaml',
              'target_dialect_contract.yaml', 'command_buffer_abi.yaml', 'integrity_policy.md',
              'schemas/command_buffer.schema.json', 'schemas/manifest.schema.json',
              'isa/README.md', 'install/lib/cmake/mlir/MLIRConfig.cmake')
    reference = ('reference-kernels/origin/kernel.c', 'public-references/origin/model.elf',
                 'public-references/origin/container.zip', 'REFERENCE_MAP.json')
    timing = ('performance_baselines/baselines.json', 'performance-baselines/baselines.json',
              'expert_targets/targets.json', 'expert-targets/targets.json',
              'SEED_PERFORMANCE/baseline.json')
    rows = [dict(path=str(run / path), mode='ro') for path in (*common, *reference, *timing)]
    selected = S.common_allowed({'allowed': rows}, name)
    paths = {row['path'] for row in selected}
    assert {str(run / path) for path in common} <= paths
    assert len([path for path in common if str(run / path) in paths]) == 11
    expected_references = {str(run / path) for path in reference} if A.resolve(name).raw_kernel_references else set()
    assert paths == {str(run / path) for path in common} | expected_references


def test_timing_words_in_run_id_do_not_withhold_its_common_data(tmp_path):
    run = tmp_path / 'performance-baselines-expert-targets-review'
    common = run / 'contracts/target_contract.yaml'
    target = run / 'expert-targets/targets.json'
    rows = [dict(path=str(path), mode='ro') for path in (common, target)]
    assert S.common_allowed({'allowed': rows}, P.NAMES[2]) == [rows[0]]
