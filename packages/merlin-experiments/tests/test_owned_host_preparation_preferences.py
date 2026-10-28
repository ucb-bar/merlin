"""Ordinary owner preferences consume frozen software, without granting premises."""

import json
from types import SimpleNamespace

import pytest
import yaml

from merlin.compile.model_execution_inputs import file_sha256
from merlin.targetgen import owned_host_model_execution as owned
from merlin.targetgen.source_composition_preferences import host_preparation


def context(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_runner

    software = tmp_path/'software.yaml'
    software.write_text(yaml.safe_dump(
        {'schema': 'merlin.software_spec.v1', 'status': 'reviewed', 'target': 'neutral'}))
    recipe = tmp_path/'numeric.yaml'
    recipe.write_text(yaml.safe_dump({'software_spec': 'software.yaml'}))
    lane = (SimpleNamespace(numeric_profile='numeric.yaml', _source_root=lambda: tmp_path), tmp_path/'host', {})
    monkeypatch.setattr(capsule_runner, '_resolve_model_host_lane', lambda *args: lane)
    return software, recipe, lane


@pytest.mark.parametrize('mutate', [False, True])
def test_ordinary_owner_forwards_requests_and_rechecks_real_unchanged_products(tmp_path, monkeypatch, mutate):
    from merlin_experiments.phase1.feedback import private_instruction_coordinator, private_instruction_declaration
    from merlin_experiments.phase1.feedback.composition_brief import build

    from merlin.common import artifacts
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.graph_optimizations import recheck_host_graph_artifacts
    from merlin.llvmlower.host_graph_policy import prepare_group_layout
    from merlin.llvmlower.host_storage_policy import prepare as prepare_storage
    from merlin.llvmlower.host_storage_policy import recheck as recheck_storage

    software, _, lane = context(tmp_path, monkeypatch)
    candidate, bundle, product = (tmp_path/name for name in ('candidate', 'bundle', 'product'))
    for directory in (candidate, bundle, product):
        directory.mkdir()
    source = 'builtin.module { func.func @forward() { return } }'
    (bundle/'model.mlir').write_text(source)
    (candidate/'source_composition.json').write_text(json.dumps({
        'schema': 'merlin.source-composition-preferences.v1', 'shared_input_siblings': False,
        'prepack_weights': False, 'propagate_layouts': True, 'reuse_intermediate_storage': True}))
    selected = dict(board='original-board', arena_mb=1)
    for role in ('board_catalog', 'dts', 'instruction_declaration', 'rtl_facts'):
        path = tmp_path/role
        path.write_text('original selected test metadata')
        selected[role] = {'path': str(path), 'sha256': file_sha256(path)}
    selected_path = tmp_path/'selected.json'
    selected_path.write_text(json.dumps(selected))
    selection_record = {'path': str(selected_path), 'sha256': file_sha256(selected_path)}
    monkeypatch.setattr(artifacts, 'new_product', lambda *a, **k: SimpleNamespace(
        path=product, add_artifact=lambda *a: None, write_manifest=lambda: None))
    authority = SimpleNamespace(service=object(), verify=lambda: None, record=lambda: {})
    monkeypatch.setattr(private_instruction_declaration, 'read', lambda *a, **k: object())
    monkeypatch.setattr(private_instruction_coordinator, 'prepare', lambda *a, **k: authority)
    materialized = SimpleNamespace(record=lambda: {}, verify=lambda *a: None)
    monkeypatch.setattr(owned, 'MaterializedModelInputs', SimpleNamespace(bind=lambda *a, **k: materialized))
    monkeypatch.setattr(owned, 'form_exact_groups', lambda *a: ([], None))
    monkeypatch.setattr(owned, '_source_placement', lambda *a, **k: {
        'frozen_placement': {}, 'eligible_source_regions': []})
    monkeypatch.setattr(owned, '_independently_eligible_group_ids', lambda *a: set())
    monkeypatch.setattr(owned, '_audit_owned_host_device_build', lambda *a, **k: {})
    monkeypatch.setattr(owned, '_execute_required_tiers', lambda *a, **k: {})
    forwarded = []

    def compile_model(**kwargs):
        forwarded.append(kwargs)
        work = kwargs['output']/'build'
        work.mkdir(parents=True)
        preparation = {}
        compilation = SimpleNamespace(bind_preparation=lambda role, path: preparation.update({
            role: {'path': str(path), 'sha256': file_sha256(path)}}))
        layout = kwargs['pre_group_layout']
        storage = kwargs['intermediate_storage']
        prepare_group_layout(parse_mlir_text(source), layout, work/'pre_group_layout')
        layout_pins = recheck_host_graph_artifacts(work/'pre_group_layout', layout)
        for role, pin in layout_pins.items():
            preparation['pre_group_layout_'+role] = pin
        llvm = work/'model.ll'
        llvm.write_text('define void @forward() {\n  ret void\n}\n')
        prepare_storage(llvm, storage, work/'intermediate_storage', compilation)
        recipe = work/'compilation_recipe.json'
        recipe.write_text(json.dumps({'preparation': preparation}))
        result = {'inputs': {'device': {'pre_group_layout': layout.record(), 'intermediate_storage': storage.record()}},
                  'output': {'pre_group_layout': layout_pins,
                             'intermediate_storage': recheck_storage(work/'intermediate_storage', storage),
                             'compilation_recipe': {'path': str(recipe), 'sha256': file_sha256(recipe)}}}
        if mutate:
            software.write_text(software.read_text()+'\n')
        return result

    monkeypatch.setattr(owned, 'OwnedHostModelCompiler', SimpleNamespace(select=lambda **kw: SimpleNamespace(
        compile=compile_model)))
    arguments = dict(capsule={'operation': {'attributes': {'dtype': 'i8'}}}, candidate_package=candidate,
                     target='neutral', bundle=bundle,
                     provenance={'bundle': {'artifacts': {}}, 'tool': {}, 'source': {}},
                     verify=lambda: None, output=tmp_path/'requested-output', selection_record=selection_record,
                     simulator='spike', timeout=1)
    if mutate:
        with pytest.raises(ValueError, match='software specification changed'):
            owned.execute_owned_host_model(**arguments)
        return
    record = owned.execute_owned_host_model(**arguments)
    assert len(forwarded) == 1 and forwarded[0]['require_all_accelerator_groups'] is True
    assert forwarded[0]['pre_group_layout'].software_path == software
    assert forwarded[0]['intermediate_storage'].software_path == software
    features = build(record)['features']
    for name in ('propagate_layouts', 'reuse_intermediate_storage'):
        assert features[name]['requested'] and features[name]['selected']
        assert features[name]['status'] == 'unchanged'
        assert features[name]['reason'] == 'permission_not_admitted'
        assert features[name]['permissions_admitted'] is False
    assert not build(record)['performance_ranking_eligible']


def test_composed_layout_defers_while_storage_policy_keeps_original_ownership(tmp_path, monkeypatch):
    context(tmp_path, monkeypatch)
    options, selection = host_preparation({'propagate_layouts': True, 'reuse_intermediate_storage': True},
                                         target='neutral', composed=True)
    assert 'pre_group_layout' not in options and 'intermediate_storage' in options
    assert selection['propagate_layouts']['status'] == 'deferred'
    assert selection['reuse_intermediate_storage']['permissions_admitted'] is False
