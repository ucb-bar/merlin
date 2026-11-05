"""The seeded CCA-coverage profile: fresh10 tooling, current sources, one pinned raw reference grant."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import fresh10_disclosure as D
from merlin_experiments.phase1 import fresh10_profiles as P

RAW = P.COVERAGE_NAMES[0]


def options(**changes):
    fields = dict(author_profile=RAW, sandbox='bwrap', compiler_execution='snapshot_external_v1',
                  fresh_backend=False, seed_submission='/seed', qualify_submission='', joint_epoch='',
                  preflight_only=False, with_tool=[], without_tool=[], compiler_access='full_compiler',
                  full_compiler_root='', reference_experiments='', reference_experiments_tasks='off')
    return SimpleNamespace(**(fields | changes))


def test_raw_profile_keeps_fresh10_tooling_with_raw_references_and_no_cca():
    profile = A.resolve(RAW)
    assert (profile.level, profile.fresh10, profile.coverage, profile.seeded) == (4, True, True, True)
    assert profile.tools == P.TOOLS and profile.entries is None and profile.raw_kernel_references
    for service in ('scheduling_semantics', 'transform_library', 'catalog_advice', 'raw_kernel_references'):
        assert profile.allows(service)
    for service in ('reference_analysis', 'automatic_kernel_learning'):
        assert not profile.allows(service)
    record = profile.record()
    assert record['live_cca'] is False and record['seeded'] is True and record['entry_modules'] is None
    assert record['source_scope'] == 'current_reviewed_full_public' and record['baseline_source_revision'] is None
    assert record['raw_reference_sha256'] == sorted(P.COVERAGE_RAW_REFERENCE_SHA256)
    # The whole v2 originals/ folder: 22 original files plus its manifest.
    assert len(P.COVERAGE_RAW_REFERENCE_SHA256) == 23
    # The original fresh10 profiles are unchanged.
    for name in P.NAMES:
        assert 'seeded' not in A.resolve(name).record() and not A.resolve(name).seeded


def test_raw_profile_requires_a_seed_and_refuses_reference_analysis():
    assert A.validate_options(options()).name == RAW
    assert A.validate_options(options(seed_submission='', preflight_only=True)).name == RAW
    for changed in (dict(seed_submission=''), dict(reference_experiments='packet.json'),
                    dict(without_tool=['isa_tools']), dict(sandbox='copy')):
        with pytest.raises(ValueError):
            A.validate_options(options(**changed))


def test_broker_withholds_cca_and_kernel_learning(monkeypatch):
    monkeypatch.setenv(A.ENV, RAW)
    for action in ('reference-analysis', 'kernel-learning'):
        with pytest.raises(FileNotFoundError):
            A.check_broker_action(action)
    for action in ('selfcheck', 'common-transform', 'common-check', 'candidate-diagnostics'):
        A.check_broker_action(action)


def test_disclosure_states_continuation_raw_references_and_disabled_cca():
    text = A.task(A.resolve(RAW))
    assert text.startswith('## Selected fresh10 compiler capabilities')
    assert 'Continue the supplied seed compiler' in text and 'empty target backend' not in text
    assert 'reference-kernels/' in text and 'CCA helpers, reference-analysis services' in text
    assert 'No CCA products, disassemblies, kernel indexes' in text
    assert 'upstream Exo Gemmini apps' in text and 'supplied Exo LLM source' in text
    assert D.author_prompt('Begin now.', continuation=True).startswith(
        'Construct one general compiler under the selected fresh10 contract, continuing the supplied seed')
    assert 'empty target backend' in D.author_prompt('Begin now.')


@pytest.fixture
def grant(tmp_path, monkeypatch):
    payloads = {name: (name * 64).encode() for name in ('a', 'b', 'c')}
    digests = tuple(hashlib.sha256(data).hexdigest() for data in payloads.values())
    monkeypatch.setattr(P, 'COVERAGE_RAW_REFERENCE_SHA256', digests)
    root = tmp_path / 'payload/experiment/reference-kernels'
    origins = []
    for name, data in payloads.items():
        (root / ('origin_' + name)).mkdir(parents=True)
        (root / ('origin_' + name) / (name + '.elf')).write_bytes(data)
        origins.append(dict(name='origin_' + name, files=[dict(destination=name + '.elf',
                                                                sha256=hashlib.sha256(data).hexdigest())]))
    (root / 'reference-manifest.json').write_text(json.dumps(dict(origins=origins)))
    (root / 'reference-selection.json').write_text('{}')
    bundle = dict(allowed=[dict(path=str(root) + '/', **{'as': 'reference-kernels'}, mode='ro'),
                           dict(path=str(tmp_path / 'payload/experiment/rtl_facts'), mode='ro')])
    return bundle, root


def test_reference_grant_is_exactly_the_pinned_raw_executables(grant):
    bundle, root = grant
    A._verify_coverage_references(bundle)
    (root / 'origin_a' / 'a.elf.dis').write_text('derived disassembly')
    with pytest.raises(ValueError, match='pinned raw executables'):
        A._verify_coverage_references(bundle)


@pytest.mark.parametrize('change', ['second_grant', 'writable', 'manifest'])
def test_other_reference_routes_refuse(grant, change):
    bundle, root = grant
    if change == 'second_grant':
        bundle['allowed'].append(dict(path=str(root.parent / 'REFERENCE_MAP.json'), mode='ro'))
    elif change == 'writable':
        bundle['allowed'][0]['mode'] = 'rw'
    else:
        (root / 'reference-manifest.json').write_text(json.dumps(dict(origins=[])))
    with pytest.raises(ValueError):
        A._verify_coverage_references(bundle)


def test_sealed_bundle_reopened_elsewhere_still_checks_the_grant_shape(grant, tmp_path):
    bundle, _root = grant
    moved = dict(allowed=[dict(bundle['allowed'][0], path='relative/reference-kernels/')])
    A._verify_coverage_references(moved, content=False)
    with pytest.raises(ValueError):
        A._verify_coverage_references(moved)


# Bank arms: the Raw arm's grant plus one pinned precomputed CCA bank (data-driven profile table).

BANK = P.COVERAGE_BANK_PREFIX + 'fixture'
RAW_TASK_SHA256 = '41b59944a2e5010a38acf6d7d08e79a2df4922e094bcbfe244ec2486ff7942f9'
RAW_RECORD_SHA256 = '2752ed21a5b9e594cc08fa9e6f287cebd4c663a0325f0e234c1769d08d0aba07'


def test_raw_arm_record_and_disclosure_are_byte_identical_with_bank_arms_defined(monkeypatch):
    monkeypatch.setattr(P, 'COVERAGE_BANKS', {BANK: 'a' * 64})
    profile = A.resolve(RAW)
    assert hashlib.sha256(A.task(profile).encode()).hexdigest() == RAW_TASK_SHA256
    assert hashlib.sha256(json.dumps(profile.record(), sort_keys=True).encode()).hexdigest() == RAW_RECORD_SHA256
    assert profile.bank_sha256 is None and not profile.allows('reference_analysis')


def test_bank_profiles_are_data_driven_and_pinned(monkeypatch):
    with pytest.raises(ValueError):
        A.resolve(BANK)
    monkeypatch.setattr(P, 'COVERAGE_BANKS', {BANK: 'b' * 64})
    profile = A.resolve(BANK)
    assert (profile.level, profile.coverage, profile.seeded, profile.fresh10) == (4, True, True, True)
    assert profile.tools == A.resolve(RAW).tools and profile.raw_kernel_references
    assert profile.allows('reference_analysis') and not profile.allows('automatic_kernel_learning')
    record = profile.record()
    assert record['cca_bank_sha256'] == 'b' * 64 and record['live_cca'] is False
    # The raw grant is the pinned bank's own raw list, never the Raw arm's full set.
    assert 'raw_reference_sha256' not in record and record['raw_reference_grant'] == 'cca_bank_raw_list_only'
    assert record['seeded'] is True
    assert P.coverage_names() == (RAW, BANK)
    for invalid in ({'cca-coverage-v1/other': 'b' * 64}, {BANK: 'B' * 64}, {BANK: 'short'}):
        monkeypatch.setattr(P, 'COVERAGE_BANKS', invalid)
        with pytest.raises(ValueError):
            P.resolve(next(iter(invalid)))


def test_bank_arm_keeps_reference_experiments_guidance_and_map_off(monkeypatch):
    monkeypatch.setattr(P, 'COVERAGE_BANKS', {BANK: 'b' * 64})
    assert A.validate_options(options(author_profile=BANK)).name == BANK
    with pytest.raises(ValueError, match='reference experiments stay off'):
        A.validate_options(options(author_profile=BANK, reference_experiments='packet.json'))
    monkeypatch.setenv(A.ENV, BANK)
    A.check_broker_action('reference-analysis')
    with pytest.raises(FileNotFoundError):
        A.check_broker_action('kernel-learning')


def test_bank_disclosure_names_the_service_and_its_boundary(monkeypatch):
    monkeypatch.setattr(P, 'COVERAGE_BANKS', {BANK: 'b' * 64})
    text = A.task(A.resolve(BANK))
    assert 'precomputed bottom-up CCA for a listed set of reference units only' in text
    assert 'not optimization recipes' in text and 'cca-evidence/' in text and 'reference-kernels/' in text
    for command in ('--reference-analysis inventory', '--reference-analysis lift --reference-entry UNIT_ID',
                    '--reference-analysis compare --reference-entry UNIT_ID'):
        assert command in text
    assert 'outside_selected_bank' in text and 'CCA helpers, reference-analysis services' not in text
    # Only the bank's own originals: other models' originals and other banks are named as absent.
    assert 'Only the raw expert reference originals this arm\'s CCA bank was prepared from' in text
    assert "outside that list (other models' originals)" in text and 'every other CCA bank are not supplied' in text
    assert 'the three original whole-model executables' not in text and 'supplied Exo LLM source' not in text
    raw = A.task(A.resolve(RAW))
    assert text.replace(D._BANK_SERVICES, D._RAW_SERVICES).replace(D._BANK_DERIVED, D._RAW_DERIVED).replace(
        D._BANK_USAGE, '').replace(D._bank_originals(None), D._raw_originals(raw)) == raw
    assert D._raw_originals(raw).endswith('the supplied Exo LLM source, and their licenses. ')
    listed = D.capability_text(BANK, bank_raw=['originals/upstream-exo/b.py', 'originals/upstream-exo/a.py'])
    assert '(corpus paths: originals/upstream-exo/a.py, originals/upstream-exo/b.py)' in listed


def test_staged_bank_raw_reads_only_the_pinned_bank(monkeypatch, tmp_path):
    document = dict(raw=[dict(id='a', path='originals/upstream-exo/a.py', sha256='0' * 64)])
    (tmp_path / 'cca-evidence').mkdir()
    (tmp_path / 'cca-evidence/bank.json').write_text(json.dumps(document))
    pin = hashlib.sha256((tmp_path / 'cca-evidence/bank.json').read_bytes()).hexdigest()
    monkeypatch.setattr(P, 'COVERAGE_BANKS', {BANK: pin})
    assert D._staged_bank_raw(tmp_path, BANK) == ['originals/upstream-exo/a.py']
    assert D._staged_bank_raw(tmp_path, RAW) is None
    monkeypatch.setattr(P, 'COVERAGE_BANKS', {BANK: 'e' * 64})
    with pytest.raises(ValueError):
        D._staged_bank_raw(tmp_path, BANK)


def _bank_conditions():
    from merlin.common.paths import artifacts_dir

    pytest.importorskip('merlin.mining.reference_subsets')
    selected = os.environ.get('MERLIN_CCA_BANK_CONDITIONS')
    root = Path(selected) if selected else artifacts_dir() / 'kernel-mining/gemmini/expert-kernel-subsets-v3/conditions'
    sized = []
    for bank in sorted(root.glob('*/bank.json')) if root.is_dir() else ():
        units = len(json.loads(bank.read_bytes()).get('units', ()))
        if units:
            sized.append((units, bank.parent))
    if len(sized) < 2:
        pytest.skip('published CCA bank conditions are unavailable in this checkout')
    sized.sort()
    # The two smallest CCA-enabled conditions, copied under the fixture's local names.
    return {'cca-4': sized[0][1], 'cca-8': sized[1][1]}


def _set_bank_raw(bank, names):
    """Point a copied bank's raw list at the fixture originals (bank.json is outside its own file pins)."""
    document = json.loads((bank / 'bank.json').read_bytes())
    document['raw'] = [dict(id='origin_' + name, path='originals/' + name + '.elf',
                            sha256=hashlib.sha256((name * 64).encode()).hexdigest()) for name in names]
    (bank / 'bank.json').write_text(json.dumps(document))
    return hashlib.sha256((bank / 'bank.json').read_bytes()).hexdigest()


def _drop_original(root, name):
    shutil.rmtree(root / ('origin_' + name))
    manifest = json.loads((root / 'reference-manifest.json').read_bytes())
    manifest['origins'] = [origin for origin in manifest['origins'] if origin['name'] != 'origin_' + name]
    (root / 'reference-manifest.json').write_text(json.dumps(manifest))


def _restore_original(root, name):
    data = (name * 64).encode()
    (root / ('origin_' + name)).mkdir()
    (root / ('origin_' + name) / (name + '.elf')).write_bytes(data)
    manifest = json.loads((root / 'reference-manifest.json').read_bytes())
    manifest['origins'].append(dict(name='origin_' + name, files=[dict(
        destination=name + '.elf', sha256=hashlib.sha256(data).hexdigest())]))
    (root / 'reference-manifest.json').write_text(json.dumps(manifest))


@pytest.fixture
def bank_grant(grant, tmp_path, monkeypatch):
    """cca-4 was prepared from originals a and b only; cca-8 from a, b and c (the Raw arm's whole set)."""
    bundle, root = grant
    conditions = _bank_conditions()
    banks = tmp_path / 'payload/experiment/cca-banks'
    for label in ('cca-4', 'cca-8'):
        shutil.copytree(conditions[label], banks / label)
    pin = _set_bank_raw(banks / 'cca-4', 'ab')
    _set_bank_raw(banks / 'cca-8', 'abc')
    _drop_original(root, 'c')
    monkeypatch.setattr(P, 'COVERAGE_BANKS', {BANK: pin})
    bundle['allowed'].append(dict(path=str(banks / 'cca-4') + '/', mode='ro', **{'as': 'cca-evidence'}))
    return bundle, banks, pin


@pytest.mark.parametrize('route', ['alias', 'schema', 'historical_bytes', 'inside_bank'])
@pytest.mark.parametrize('arm', ['raw', 'bank'])
def test_no_reference_map_reaches_a_coverage_arm(bank_grant, tmp_path, route, arm):
    from merlin.common.paths import artifacts_dir

    bundle, banks, pin = bank_grant
    if arm == 'raw':
        bundle['allowed'].pop()
        pin = None
        _restore_original(Path(bundle['allowed'][0]['path']), 'c')
    A._verify_coverage_references(bundle, bank_sha256=pin)  # Admitted until a map is added.
    hidden = tmp_path / 'public/data.json'
    hidden.parent.mkdir()
    if route == 'alias':
        hidden.write_text('{}')
        bundle['allowed'].append(dict(path=str(hidden), mode='ro', **{'as': 'REFERENCE_MAP.json'}))
    elif route == 'schema':
        hidden.write_text(json.dumps(dict(schema='merlin.reference_map.v1', entries=[])))
        bundle['allowed'].append(dict(path=str(hidden), mode='ro'))
    elif route == 'historical_bytes':
        root = Path(os.environ.get('MERLIN_REFERENCE_PREPARATION') or artifacts_dir() / 'preparation')
        historical = [path for path in root.glob('fresh*-reference-map/REFERENCE_MAP.json')
                      if hashlib.sha256(path.read_bytes()).hexdigest() in P.WITHHELD_REFERENCE_MAP_SHA256]
        if not historical:
            pytest.skip('historical reference maps are unavailable in this checkout')
        shutil.copyfile(historical[0], hidden)
        bundle['allowed'].append(dict(path=str(hidden), mode='ro'))
    else:
        if arm == 'raw':
            pytest.skip('the raw arm has no bank')
        (banks / 'cca-4/REFERENCE_MAP.json').write_text(json.dumps(dict(schema='merlin.reference_map.v1')))
    with pytest.raises(ValueError):
        A._verify_coverage_references(bundle, bank_sha256=pin)


def test_bank_arm_admits_exactly_its_pinned_bank(bank_grant):
    bundle, _banks, pin = bank_grant
    A._verify_coverage_references(bundle, bank_sha256=pin)
    A._verify_coverage_references(bundle, content=False, bank_sha256=pin)
    # The Raw arm refuses the same bundle: a bank grant is a second reference grant.
    with pytest.raises(ValueError):
        A._verify_coverage_references(bundle)


@pytest.mark.parametrize('change', ['extra_original', 'missing_original', 'no_raw_grant', 'second_raw_grant',
                                    'writable_raw', 'manifest_lists_other', 'unpinned_raw_list',
                                    'empty_raw_list'])
def test_bank_arm_raw_grant_is_exactly_its_banks_raw_list(bank_grant, change, monkeypatch):
    """No other model's original: the raw grant equals the pinned bank's own raw list by bytes."""
    bundle, banks, pin = bank_grant
    root = Path(bundle['allowed'][0]['path'])
    if change == 'extra_original':
        _restore_original(root, 'c')  # e.g. another model's ELF beside the bank's own originals
    elif change == 'missing_original':
        _drop_original(root, 'b')
    elif change == 'no_raw_grant':
        bundle['allowed'].pop(0)
    elif change == 'second_raw_grant':
        bundle['allowed'].append(dict(bundle['allowed'][0], path=str(root / 'origin_a') + '/'))
    elif change == 'writable_raw':
        bundle['allowed'][0]['mode'] = 'rw'
    elif change == 'manifest_lists_other':
        manifest = json.loads((root / 'reference-manifest.json').read_bytes())
        manifest['origins'][0]['files'][0]['sha256'] = hashlib.sha256(b'c' * 64).hexdigest()
        (root / 'reference-manifest.json').write_text(json.dumps(manifest))
    else:
        # A bank whose raw list names a file outside the study's originals, or none at all.
        document = json.loads((banks / 'cca-4/bank.json').read_bytes())
        document['raw'] = ([dict(id='x', path='originals/x.elf', sha256='d' * 64)]
                           if change == 'unpinned_raw_list' else [])
        (banks / 'cca-4/bank.json').write_text(json.dumps(document))
        pin = hashlib.sha256((banks / 'cca-4/bank.json').read_bytes()).hexdigest()
        monkeypatch.setattr(P, 'COVERAGE_BANKS', {BANK: pin})
    with pytest.raises(ValueError):
        A._verify_coverage_references(bundle, bank_sha256=pin)


def test_a_bank_of_every_original_gets_the_raw_arms_exact_grant(bank_grant, monkeypatch):
    """With c as the originals manifest, cca-4 (a, b) is the all-originals bank: its grant is a, b, c."""
    bundle, banks, pin = bank_grant
    monkeypatch.setattr(P, 'COVERAGE_ORIGINALS_MANIFEST_SHA256', hashlib.sha256(b'c' * 64).hexdigest())
    assert A.bank_raw_sha256(banks / 'cca-4') == sorted(P.COVERAGE_RAW_REFERENCE_SHA256)
    with pytest.raises(ValueError, match='pinned raw executables'):
        A._verify_coverage_references(bundle, bank_sha256=pin)  # The manifest is missing.
    _restore_original(Path(bundle['allowed'][0]['path']), 'c')
    A._verify_coverage_references(bundle, bank_sha256=pin)
    # A bank of fewer originals gets exactly its raw list: never the manifest naming the others.
    document = json.loads((banks / 'cca-4/bank.json').read_bytes())
    document['raw'] = document['raw'][:1]
    (banks / 'cca-4/bank.json').write_text(json.dumps(document))
    assert A.bank_raw_sha256(banks / 'cca-4') == [hashlib.sha256(b'a' * 64).hexdigest()]


def test_each_bank_arm_admits_only_its_own_raw_list(bank_grant, monkeypatch):
    bundle, banks, _pin = bank_grant
    root = Path(bundle['allowed'][0]['path'])
    _restore_original(root, 'c')
    pin8 = hashlib.sha256((banks / 'cca-8/bank.json').read_bytes()).hexdigest()
    monkeypatch.setattr(P, 'COVERAGE_BANKS', {BANK: pin8})
    bundle['allowed'][-1]['path'] = str(banks / 'cca-8') + '/'
    A._verify_coverage_references(bundle, bank_sha256=pin8)
    assert A.bank_raw_sha256(banks / 'cca-8') == sorted(P.COVERAGE_RAW_REFERENCE_SHA256)
    assert len(A.bank_raw_sha256(banks / 'cca-4')) == 2


@pytest.mark.parametrize('change', ['other_condition', 'second_bank', 'writable', 'ancestor', 'corpus',
                                    'extra_file', 'mutated_card', 'edited_roster', 'symlink', 'missing'])
def test_bank_arm_refuses_every_other_bank_route(bank_grant, change, tmp_path):
    bundle, banks, pin = bank_grant
    row = bundle['allowed'][-1]
    if change == 'other_condition':
        row['path'] = str(banks / 'cca-8') + '/'
    elif change == 'second_bank':
        bundle['allowed'].append(dict(path=str(banks / 'cca-8') + '/', mode='ro', **{'as': 'cca-evidence'}))
    elif change == 'writable':
        row['mode'] = 'rw'
    elif change == 'ancestor':
        bundle['allowed'].append(dict(path=str(banks) + '/', mode='ro'))
    elif change == 'corpus':
        corpus = tmp_path / 'corpus'
        corpus.mkdir()
        (corpus / 'database.sqlite').write_bytes(b'SQLite format 3\0')
        row['path'] = str(corpus) + '/'
    elif change == 'extra_file':
        (banks / 'cca-4/REFERENCE_MAP.json').write_text('{}')
    elif change == 'mutated_card':
        card = next((banks / 'cca-4/units').glob('*/card.json'))
        card.write_text(card.read_text().replace('"accelerator_sites": ', '"accelerator_sites": 1', 1))
    elif change == 'edited_roster':
        document = json.loads((banks / 'cca-4/bank.json').read_bytes())
        document['units'] = document['units'][:1]
        (banks / 'cca-4/bank.json').write_text(json.dumps(document))
    elif change == 'symlink':
        (banks / 'cca-4/units/alias').symlink_to(banks / 'cca-8/units')
    else:
        bundle['allowed'].pop()
    with pytest.raises(ValueError):
        A._verify_coverage_references(bundle, bank_sha256=pin)


def test_readiness_accepts_the_staged_full_compiler_instead_of_legacy_source_grants(grant):
    from merlin_experiments.phase1 import tooling_readiness as R
    bundle, _root = grant
    profiled = dict(bundle, author_profile=A.resolve(RAW).record(), tools=list(P.TOOLS),
                    author_profile_sources={'manifest_sha256': 'f' * 64})
    assert R._coverage_sources(profiled) == 'f' * 64
    assert R._coverage_sources(dict(bundle)) is None
    profiled['allowed'] = profiled['allowed'] + [dict(path='/x/REFERENCE_MAP.json', mode='ro')]
    with pytest.raises(ValueError):
        R._coverage_sources(profiled)


def test_coverage_full_compiler_keeps_the_seed_runs_projected_tools():
    from merlin.common.paths import repo_root
    from merlin_experiments.phase1 import fresh10_sources as S
    extra = S.coverage_extra_modules(repo_root())
    for module in ('merlin.targetgen.generate.mlir_scaffold', 'merlin.targetgen.synthesize.dialect_plan',
                   'merlin.targetgen.evidence.store', 'merlin.targetgen.oot_starterkit.verify'):
        assert module in extra
    assert not any(name.startswith(('merlin.mining', 'merlin.kernels')) for name in extra)


def test_readiness_imports_the_coverage_compiler_from_the_staged_tree(grant, monkeypatch):
    from merlin_experiments.phase1 import tooling_readiness as R
    bundle, _root = grant
    profiled = dict(bundle, author_profile=A.resolve(RAW).record(), tools=list(P.TOOLS))
    monkeypatch.setattr(R, '_component_selection', lambda _bundle: (A.resolve(RAW), object()))
    probe = R._component_probe(profiled)
    assert "compiler_library" in probe and "projection" not in probe
    compile(probe, 'probe', 'exec')
