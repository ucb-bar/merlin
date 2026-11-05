"""The supplied callable name must equal the selected native caller ABI."""
from pathlib import Path
import copy
import hashlib

import pytest
import yaml

from merlin_experiments.phase1 import public_runtime_abi as P


def protocol(symbol='{target}_kernel'):
    return {'logical_kernel_abi': {'version': 2, 'symbol': symbol,
        'signature': f'void {symbol}(void *arg_0, ..., void *arg_N)',
        'argument_order': {'default': ['inputs', 'outputs']},
        'pointee': {'layout': 'dense row-major', 'alignment_bytes': 64}},
        'unchanged_numerical_rules': {'poison': 165, 'completion': 'return results'}}


def contract(symbol='arbitrary_callable', completion='selected_completion'):
    return {'name': 'unrelated_target', 'harness_abi': {'entry_symbol': symbol, 'fence_symbol': completion},
        'runner': {'backend': 'control_adapter', 'control_adapter': {
            'toolchain': {'cflags': ['-march=rv32imc'],
                          'kernel_stack_frame': {'entry_symbol': symbol, 'max_static_bytes': 512}}}},
        'private_hardware_detail': 'DO_NOT_PUBLISH'}


@pytest.mark.parametrize('symbol', ['alternate_entry', '_selected_dispatch42'])
def test_projection_follows_selected_caller_not_target_name(symbol):
    original = protocol()
    result = yaml.safe_load(P.rewrite_protocol(yaml.safe_dump(original).encode(), contract=contract(symbol)))
    expected = copy.deepcopy(original)
    expected['logical_kernel_abi']['symbol'] = symbol
    expected['logical_kernel_abi']['signature'] = original['logical_kernel_abi']['signature'].replace('{target}_kernel', symbol)
    expected['logical_kernel_abi']['required_callables'] = P.selected_callables(contract(symbol))
    expected['logical_kernel_abi']['software_limits'] = P.selected_software_limits(contract(symbol))
    expected['logical_kernel_abi']['completion'] = result['logical_kernel_abi']['completion']
    expected['logical_kernel_abi']['normalized_lowered_representation'] = P.selected_lowered_representation(contract(symbol))
    expected['whole_program_attribution'] = result['whole_program_attribution']
    assert result == expected
    assert 'DO_NOT_PUBLISH' not in yaml.safe_dump(result)
    assert original['logical_kernel_abi']['symbol'] == '{target}_kernel'


@pytest.mark.parametrize('name', ['finish_selected_work', '_different_callback42', None])
def test_complete_software_abi_projects_only_selected_runtime_completion_sequence(name):
    selected = contract(completion=name)
    selected['logical_harness'] = {'completion_asm': 'selected_fixed_barrier',
                                  'unrelated_instruction': 'DO_NOT_PUBLISH_ASSEMBLY'}
    original = protocol()
    original['logical_kernel_abi']['completion'] = 'harness executes logical_harness.completion_asm'
    result = yaml.safe_load(P.rewrite_protocol(yaml.safe_dump(original).encode(), contract=selected))
    block = result['logical_kernel_abi']
    assert len(block['required_callables']) == (1 if name is None else 2)
    assert 'DO_NOT_PUBLISH' not in yaml.safe_dump(result)
    assert block['normalized_lowered_representation']['selected_abi']['completion_asm'] == 'selected_fixed_barrier'
    if name is None:
        assert 'no separate candidate completion function' in block['completion']
    else:
        assert block['required_callables'][1]['signature'] == f'void {name}(void)'
        assert f'void {name}(void)' in block['completion']
        assert 'runner does not supply this function' in block['completion']


@pytest.mark.parametrize('change', ['same_name', 'invalid_name', 'extra_extern', 'wrong_completion_prototype'])
def test_undisclosed_or_conflicting_runtime_callbacks_are_refused(change):
    selected = contract()
    if change == 'same_name': selected['harness_abi']['fence_symbol'] = selected['harness_abi']['entry_symbol']
    elif change == 'invalid_name': selected['harness_abi']['fence_symbol'] = 'invalid.symbol'
    elif change == 'extra_extern': selected['harness_abi']['extern_decls'] = ['extern void undisclosed_callback(void);']
    else: selected['harness_abi']['extern_decls'] = ['extern void selected_completion(void *value);']
    with pytest.raises(ValueError, match='completion|external declaration'):
        P.selected_callables(selected)


def test_selected_caller_and_declared_stack_recipe_must_match():
    selected = contract()
    selected['runner']['control_adapter']['toolchain']['kernel_stack_frame']['entry_symbol'] = 'different_entry'
    with pytest.raises(ValueError, match='caller and stack recipe entry symbols differ'):
        P.rewrite_protocol(yaml.safe_dump(protocol()).encode(), contract=selected)


@pytest.mark.parametrize('symbol', ['bad.name', 'not plain', 'unicode_\u03b1'])
def test_non_c_callable_is_not_guessed_or_sanitized(symbol):
    with pytest.raises(ValueError, match='plain C ABI identifier'):
        P.selected_symbol(contract(symbol))


def test_conflicting_public_signature_refuses_instead_of_guessing():
    document = protocol()
    document['logical_kernel_abi']['signature'] = 'void another_name(void *arg_0)'
    with pytest.raises(ValueError, match='symbol and signature are inconsistent'):
        P.rewrite_protocol(yaml.safe_dump(document).encode(), contract=contract())


def prepared(tmp_path):
    repo = tmp_path / 'repo'
    source = repo / P.DECLARED
    source.parent.mkdir(parents=True)
    source.write_text(yaml.safe_dump(protocol(), sort_keys=False))
    selected = repo / 'private/selected-runtime.yaml'
    selected.parent.mkdir()
    selected.write_text(yaml.safe_dump(contract(), sort_keys=False))
    common = [dict(path=P.DECLARED, mode='ro'),
              dict(path=str(selected), mode='ro', note='selected capability contract (ALL arms)')]
    (repo / 'public').mkdir()
    common, record = P.prepare(common, destination=repo / 'public/runtime-protocol', repo=repo)
    return repo, dict(allowed=common, public_runtime_abi=record), selected


def test_preparation_pins_one_public_protocol_and_only_runtime_call_data(tmp_path):
    repo, bundle, selected = prepared(tmp_path)
    verified = P.verify(bundle)
    assert verified['entry_symbol'] == 'arbitrary_callable'
    assert verified['callables'] == P.selected_callables(contract())
    assert verified['target_contract']['path'] == str(selected)
    assert not any(row['path'] == P.DECLARED for row in bundle['allowed'])
    public = Path(verified['protocol']['path'])
    assert b'DO_NOT_PUBLISH' not in public.read_bytes()
    assert public.stat().st_mode & 0o222 == 0


@pytest.mark.parametrize('tamper', ['entry', 'digest', 'public_symbol', 'old_canonical_grant', 'duplicate_alias', 'different_alias_path'])
def test_binding_and_content_contradictions_are_rejected(tmp_path, tamper):
    repo, bundle, selected = prepared(tmp_path)
    pin = bundle[P.KEY]['protocol']
    if tamper == 'entry':
        bundle[P.KEY]['entry_symbol'] = 'other_entry'
    elif tamper == 'digest':
        pin['sha256'] = 'f' * 64
    elif tamper == 'public_symbol':
        path = Path(pin['path'])
        doc = yaml.safe_load(path.read_bytes())
        doc['logical_kernel_abi']['symbol'] = 'wrong_entry'
        path.chmod(0o600)
        path.write_text(yaml.safe_dump(doc))
        pin['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif tamper == 'old_canonical_grant':
        bundle['allowed'].append(dict(path=P.DECLARED, mode='ro'))
    elif tamper == 'different_alias_path':
        bundle['allowed'].append(dict(path=str(repo / 'another-protocol.yaml'), mode='ro', **{'as': P.ALIAS}))
    else:
        bundle['allowed'].append(copy.deepcopy(bundle['allowed'][0]))
    with pytest.raises(ValueError, match='runtime ABI|public callable ABI'):
        P.verify(bundle)


@pytest.mark.parametrize('change', ['missing_completion', 'wrong_prototype', 'wrong_owner', 'old_completion_text', 'wrong_stack_limit', 'wrong_fixed_sequence', 'wrong_completion_constraints', 'wrong_source_numbering', 'missing_plan_trigger'])
def test_digest_repinning_cannot_hide_incomplete_public_runtime_abi(tmp_path, change):
    repo, bundle, selected = prepared(tmp_path)
    pin = bundle[P.KEY]['protocol']; path = Path(pin['path'])
    doc = yaml.safe_load(path.read_text()); block = doc['logical_kernel_abi']
    if change == 'missing_completion': block['required_callables'] = block['required_callables'][:1]
    elif change == 'wrong_prototype': block['required_callables'][1]['parameters'] = ['void *']
    elif change == 'wrong_owner': block['required_callables'][1]['defined_by'] = 'runner'
    elif change == 'wrong_stack_limit': block['software_limits']['entry_max_static_stack_bytes'] = 513
    elif change == 'wrong_fixed_sequence': block['normalized_lowered_representation']['selected_abi']['completion_asm'] = 'substituted_sequence'
    elif change == 'wrong_completion_constraints': block['normalized_lowered_representation']['completion_shim']['inline_asm_constraints'] = ''
    elif change == 'wrong_source_numbering': doc['whole_program_attribution']['selected_source_inventory']['selected_representation'] = 'Count tensor declarations as operations.'
    elif change == 'missing_plan_trigger': doc['whole_program_attribution']['required_plan'] = 'The source plan is optional.'
    else: block['completion'] = 'harness executes completion_asm'
    path.chmod(0o600);path.write_text(yaml.safe_dump(doc));pin['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='public callable ABI'):
        P.verify(bundle)


def test_fixed_runtime_sequence_follows_owning_abi_without_target_names():
    from merlin.targetgen.contract.harness_abi import lowered_plan_abi
    selected = contract('other_entry', 'other_finish')
    assert P.selected_lowered_representation(selected)['selected_abi'] == lowered_plan_abi(selected, target=selected['name'])
    selected['logical_harness'] = {'completion_asm': 'alternate_fixed_completion'}
    assert P.selected_lowered_representation(selected)['selected_abi']['completion_asm'] == 'alternate_fixed_completion'
    selected['logical_harness'] = {}
    selected['runner']['control_adapter']['toolchain']['cflags'] = []
    with pytest.raises(ValueError, match='fixed runtime sequence'):
        P.selected_lowered_representation(selected)


def test_selected_format_and_numbering_are_part_of_frozen_binding(tmp_path):
    from merlin_experiments.phase1 import author_profiles as A
    _, bundle, _ = prepared(tmp_path)
    bundle['author_profile'] = A.resolve('xs-el-raw-rtl-v2/el1').record()
    with pytest.raises(ValueError, match='source protocol differs'):
        P.verify(bundle)


def test_native_verification_uses_frozen_private_and_public_bytes(tmp_path, monkeypatch):
    from merlin.targetgen.sandbox import bwrap as B
    monkeypatch.setenv('MERLIN_BUNDLE_CAS', '')
    repo, bundle, selected = prepared(tmp_path)
    bundle['host_inputs'] = []  # In this fixture selected caller is already an explicit RO grant.
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    B.materialize_bundle_inputs(workspace, bundle, repo=repo)
    # Changing live sources after freezing cannot change the selected native bytes.
    selected.write_text(yaml.safe_dump(contract('mutated_live_entry')))
    verified = P.verify(bundle, workspace=workspace, repo=repo)
    assert verified['entry_symbol'] == 'arbitrary_callable'
    with pytest.raises(ValueError, match='content differs'):
        P.verify(bundle)


@pytest.mark.parametrize('symbol', ['different_selected_entry', '_alternate42'])
@pytest.mark.parametrize('report_kind', ['valid', 'wrong', 'duplicate', 'dynamic_helper', 'over_budget'])
def test_selected_callable_preserves_exact_one_static_row_and_budget(tmp_path, symbol, report_kind):
    from merlin.targetgen.contract.stack_usage import measure_entrypoint, StackFramePreflightError
    entry = P.selected_symbol(contract(symbol))
    llvm, report = tmp_path / 'kernel.ll', tmp_path / 'kernel.su'
    llvm.write_text(f'define void @{entry}() {{ ret void }}\n')
    row = f'kernel.ll:{entry}\t128\tstatic\n'
    if report_kind == 'wrong': row = 'kernel.ll:other_entry\t128\tstatic\n'
    elif report_kind == 'duplicate': row += row
    elif report_kind == 'dynamic_helper': row += 'kernel.ll:helper\t64\tdynamic\n'
    elif report_kind == 'over_budget': row = f'kernel.ll:{entry}\t513\tstatic\n'
    report.write_text(row)
    if report_kind == 'valid':
        result = measure_entrypoint(report, llvm_path=llvm, entry_symbol=entry, max_static_bytes=512)
        assert result.entry_symbol == entry and result.frame_bytes == 128 and result.max_static_bytes == 512
    else:
        with pytest.raises(StackFramePreflightError):
            measure_entrypoint(report, llvm_path=llvm, entry_symbol=entry, max_static_bytes=512)


@pytest.mark.parametrize('level', [1, 2, 3])
def test_all_raw_levels_receive_same_selected_callable_through_native_snapshot(tmp_path, monkeypatch, level):
    import json
    from merlin.common.paths import repo_root
    from merlin.targetgen.sandbox import bwrap as B
    from merlin_experiments.phase1 import author_profiles as A
    from merlin_experiments.phase1 import raw_rtl_inputs as R
    from merlin_experiments.phase1 import public_contract_preflight as F
    from merlin_experiments.phase1.public_contract_staging import PUBLIC_FILES

    monkeypatch.setenv('MERLIN_BUNDLE_CAS', '')
    repo = repo_root()
    original = tmp_path / 'original'
    original.mkdir()
    raw = original / 'hardware.fir'
    raw.write_text('FIRRTL version 3.3.0\n')
    facts = original / 'rtl_facts/facts.json'
    facts.parent.mkdir()
    facts.write_text(json.dumps({'source_consistency': {'status': 'verified'}, 'inputs': {
        'firrtl_inputs': [{'path': str(raw), 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()}]}}))
    tests = original / 'tests'
    tests.mkdir()
    (tests / 'source.mlir').write_text('module { func.func @example() { %c = arith.constant 0 : i32 return } }\n')
    selected = original / 'selected-runtime.yaml'
    selected.write_text(yaml.safe_dump(contract('selected_public_callable')))
    before = selected.read_bytes()
    common = [dict(path='merlin/contract/' + name, mode='ro') for name in sorted(PUBLIC_FILES)]
    common += [dict(path='merlin/contract/schemas/', mode='ro'),
               dict(path=str(facts.parent) + '/', mode='ro'),
               dict(path=str(tests), mode='ro', note='capsule corpus'),
               dict(path=str(selected), mode='ro', note='selected capability contract (ALL arms)')]
    experiment = tmp_path / 'experiment'
    experiment.mkdir()
    profile = A.resolve(f'xs-el-raw-rtl-v2/el{level}')
    result, public = R.prepare(dict(allowed=common, host_inputs=[], selected_rtl_facts_file=str(facts)),
        common, profile=profile, experiment=experiment, repo=repo, corpus_paths=[str(tests)])
    result.update(allowed=public, author_profile=profile.record(), tools=list(profile.tools))
    public_paths = {row['path'] for row in public}
    result['host_inputs'] += [dict(path=row['path']) for row in common if row['path'] not in public_paths]
    A.verify_bundle(profile, result)
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    B.materialize_bundle_inputs(workspace, result, repo=repo)
    reads = F.expected_reads(workspace, result, repo=repo)
    assert reads['expected_files'][P.ALIAS] == result[P.KEY]['protocol']['sha256']
    verified = P.verify(result, workspace=workspace, repo=repo)
    frozen_protocol = B.snapshot_input_paths(workspace, result,
        [Path(result[P.KEY]['protocol']['path'])], repo=repo)[0]
    public_doc = yaml.safe_load(frozen_protocol.read_bytes())
    assert public_doc['logical_kernel_abi']['symbol'] == verified['entry_symbol'] == 'selected_public_callable'
    assert public_doc['logical_kernel_abi']['required_callables'] == P.selected_callables(contract('selected_public_callable'))
    assert public_doc['logical_kernel_abi']['required_callables'][1]['signature'] == 'void selected_completion(void)'
    assert 'DO_NOT_PUBLISH' not in yaml.safe_dump(public_doc)
    assert selected.read_bytes() == before
    assert not any(row['path'] == P.DECLARED for row in public)
    if level == 1:
        assert b'merlin_iface' not in frozen_protocol.read_bytes()
    source_rules = public_doc['whole_program_attribution']['selected_source_inventory']
    assert source_rules['input_format'] == profile.disclosure['input_format']
    rules = source_rules['selected_representation']
    if level == 1:
        assert 'expressions array positions' in rules and 'result_index is 0' in rules
        assert 'inputs array order' in rules and 'outputs array order' in rules
        assert 'resident_pack' not in rules and 'merlin_iface' not in rules
        neutral = next(row for row in public if row.get('as') == 'neutral_input_contract.md')
        assert b'source_op_indices' in Path(neutral['path']).read_bytes()
    else:
        assert 'tensor declarations' in rules
        assert 'excluded from source_op_count' in rules and 'including resident_pack, matmul and evict' in rules
    normalized = public_doc['logical_kernel_abi']['normalized_lowered_representation']
    assert normalized['selected_abi']['completion_asm'] == 'fence'
    assert normalized['completion_shim']['inline_asm_constraints'] == '~{memory}'
    assert 'attribute dictionaries must be empty' in normalized['entry']
    assert 'Inline all ordinary helper definitions' in normalized['functions']
    requirement = public_doc['whole_program_attribution']['required_plan']
    assert 'single operation' in requirement and 'readwrite access grants no seed' in requirement
    assert 'software_spec' not in yaml.safe_dump(public_doc)


@pytest.mark.parametrize('level', [1, 2, 3])
def test_native_raw_preflight_requires_selected_callable_binding(tmp_path, level):
    from merlin_experiments.phase1 import public_contract_preflight as F
    with pytest.raises(ValueError, match='runtime ABI binding is missing'):
        F.expected_reads(tmp_path, {'author_profile': {'name': f'xs-el-raw-rtl-v2/el{level}'}}, repo=tmp_path)
