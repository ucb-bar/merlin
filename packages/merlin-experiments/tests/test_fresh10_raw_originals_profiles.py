"""From-scratch fresh10 arms over exactly the pinned raw originals (and one all-originals bank)."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import fresh10_disclosure as D
from merlin_experiments.phase1 import fresh10_profiles as P
from merlin_experiments.phase1 import fresh10_sources as S

FULL, BANK, INFRA = P.RAW_ORIGINALS_NAMES
CCA_ALL = "8c5ce71524744c4e0c6236a21462f7b196acf82de362d2087d7367ca15d1265a"


def options(name=FULL, **changes):
    fields = dict(author_profile=name, sandbox='bwrap', compiler_execution='snapshot_external_v1',
                  fresh_backend=True, seed_submission='', qualify_submission='', joint_epoch='',
                  preflight_only=False, with_tool=[], without_tool=[], compiler_access='full_compiler',
                  full_compiler_root='', reference_experiments='', reference_experiments_tasks='off')
    return SimpleNamespace(**(fields | changes))


def test_names_keep_the_original_fresh10_positions():
    assert P.NAMES == ("fresh10-tooling-v1/infra", "fresh10-tooling-v1/scheduling",
                       "fresh10-tooling-v1/no-reference-kernels")
    assert P.RAW_ORIGINALS_NAMES == ("fresh10-tooling-v1/full-ref", "fresh10-tooling-v1/full-ref-bank-all",
                                     "fresh10-tooling-v1/infra-originals")
    assert P.RAW_ORIGINALS_BANKS == {BANK: CCA_ALL} and P.COVERAGE_BANKS["cca-coverage-v1/bank-cca-all"] == CCA_ALL
    # The baseline projection belongs to the original three arms only.
    assert {name: S._arm(name) for name in P.NAMES} == {P.NAMES[0]: 'A', P.NAMES[1]: 'B', P.NAMES[2]: 'C'}
    for name in (FULL, BANK):
        with pytest.raises(ValueError, match='current reviewed library'):
            S._arm(name)


def test_infra_originals_refuses_preparation_until_current_sources_can_be_withheld(tmp_path):
    profile = A.resolve(INFRA)
    assert not profile.allows('scheduling_semantics') and not profile.allows('transform_library')
    for call in (lambda: P.require_preparable(INFRA), lambda: A.validate_options(options(INFRA)),
                 lambda: A.prepare_sources(tmp_path, tmp_path / 'out', profile=INFRA, review_id='r')):
        with pytest.raises(ValueError, match='not preparable'):
            call()
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('name, scheduling, transform', [(FULL, True, True), (BANK, True, True),
                                                         (INFRA, False, False)])
def test_profiles_are_unseeded_level4_fresh10_over_the_pinned_originals(name, scheduling, transform):
    profile = A.resolve(name)
    assert (profile.level, profile.fresh10, profile.coverage, profile.seeded, profile.raw_originals,
            profile.current_library, profile.entries) == (4, True, False, False, True, True, None)
    assert profile.tools == P.TOOLS and 'cca_tools' not in profile.tools and not profile.modern
    assert profile.raw_kernel_references
    assert (profile.allows('scheduling_semantics'), profile.allows('transform_library'),
            profile.allows('catalog_advice')) == (scheduling, transform, transform)
    assert not profile.allows('automatic_kernel_learning')
    assert profile.allows('reference_analysis') == (name == BANK)
    record = profile.record()
    assert record['live_cca'] is False and 'seeded' not in record and record['reference_map'] is False
    # The current reviewed full compiler and library, as the coverage arms stage them.
    assert record['source_scope'] == 'current_reviewed_full_public'
    assert record['baseline_source_revision'] is None and record['entry_modules'] is None
    # Every granted file by bytes: the 23 pinned originals.
    assert record['raw_reference_sha256'] == sorted(P.COVERAGE_RAW_REFERENCE_SHA256)
    assert len(record['raw_reference_sha256']) == 23
    if name == BANK:
        assert profile.bank_sha256 == CCA_ALL and record['cca_bank_sha256'] == CCA_ALL
        assert record['raw_reference_grant'] == 'cca_bank_raw_list_only'
        assert record['cca_bank_service'] == 'precomputed_selected_units_only'
    else:
        assert profile.bank_sha256 is None and 'cca_bank_sha256' not in record
        assert record['raw_reference_grant'] == 'pinned_originals_only'


def test_existing_profile_records_are_unchanged():
    for name in P.NAMES:
        record = A.resolve(name).record()
        assert not {'seeded', 'raw_reference_sha256', 'reference_map', 'cca_bank_sha256'} & set(record)
        assert A.resolve(name).bank_sha256 is None and not A.resolve(name).raw_originals
    raw = A.resolve(P.COVERAGE_NAMES[0])
    assert not raw.raw_originals and 'reference_map' not in raw.record()


def test_profiles_are_selectable_everywhere():
    from merlin_experiments.phase1 import public_contract_staging
    from merlin_experiments.phase1.options import author_profile_names

    for name in P.RAW_ORIGINALS_NAMES:
        assert name in A.NAMES and name in author_profile_names()
        assert name in public_contract_staging.PROFILES


@pytest.mark.parametrize('name', (FULL, BANK))
def test_from_scratch_options(name):
    assert A.validate_options(options(name)).name == name
    for changed in (dict(seed_submission='/seed'), dict(fresh_backend=False), dict(reference_experiments='x'),
                    dict(with_tool=['cca_tools'])):
        with pytest.raises(ValueError):
            A.validate_options(options(name, **changed))


@pytest.mark.parametrize('name', P.RAW_ORIGINALS_NAMES)
def test_broker_serves_reference_analysis_only_from_the_bank(name, monkeypatch):
    monkeypatch.setenv(A.ENV, name)
    with pytest.raises(FileNotFoundError):
        A.check_broker_action('kernel-learning')
    if name == BANK:
        A.check_broker_action('reference-analysis')
    else:
        with pytest.raises(FileNotFoundError):
            A.check_broker_action('reference-analysis')


@pytest.mark.parametrize('name', P.RAW_ORIGINALS_NAMES)
def test_disclosure_uses_the_originals_sentence_never_map_wording(name):
    text = A.task(A.resolve(name))
    assert 'Start with an empty target backend' in text and 'Continue the supplied seed' not in text
    for stale in ('reference bank is supplied', 'operator-built', 'Any operator-built reference map'):
        assert stale not in text
    if name == BANK:
        assert D._BANK_SERVICES in text and D._BANK_USAGE in text and D._BANK_DERIVED in text
        assert "Only the raw expert reference originals this arm's CCA bank was prepared from" in text
        assert D._RAW_SERVICES not in text
    else:
        assert D._ORIGINALS_TEXT in text and D._RAW_SERVICES in text
        assert 'upstream Exo Gemmini apps' in text and 'supplied Exo LLM source' in text
    assert ('omits Merlin scheduling abstractions' in text) == (name == INFRA)
    # The current full compiler is named as such, never as the baseline's reviewed subset.
    assert ('complete current reviewed public compiler sources' in text) == (name != INFRA)
    if name != INFRA:
        assert 'explicitly reviewed subset' not in text
        raw = A.task(A.resolve(P.COVERAGE_NAMES[0]))
        assert text.replace(D._current_scratch_origin(), D._coverage_origin()) == (
            raw if name == FULL else text.replace(D._current_scratch_origin(), D._coverage_origin()))


# The grant: exactly the 23 pinned originals, the same check as the Raw arm, but unseeded.

@pytest.fixture
def originals(tmp_path, monkeypatch):
    payloads = {'o%02d' % index: ('original %d' % index).encode() * 8 for index in range(23)}
    digests = tuple(hashlib.sha256(data).hexdigest() for data in payloads.values())
    monkeypatch.setattr(P, 'COVERAGE_RAW_REFERENCE_SHA256', digests)
    monkeypatch.setattr(P, 'COVERAGE_ORIGINALS_MANIFEST_SHA256', digests[-1])
    root = tmp_path / 'payload/experiment/reference-kernels'
    files = []
    for name, data in payloads.items():
        (root / 'originals').mkdir(parents=True, exist_ok=True)
        (root / 'originals' / (name + '.bin')).write_bytes(data)
        files.append(dict(member=name + '.bin', destination=name + '.bin', sha256=hashlib.sha256(data).hexdigest()))
    (root / 'reference-manifest.json').write_text(json.dumps(dict(
        schema='operator.allowed_reference_sources.v2', origins=[dict(name='originals', files=files)])))
    (root / 'reference-selection.json').write_text('{}')
    bundle = dict(allowed=[dict(path=str(root) + '/', mode='ro', **{'as': 'reference-kernels'})])
    return bundle, root, payloads


def synthetic_bank(directory, raw):
    """A minimal bank that merlin.mining.reference_subsets.verify_bank accepts."""
    card = directory / 'units/u1/card.json'
    card.parent.mkdir(parents=True)
    card.write_text(json.dumps(dict(kernel_id='u1', occurrences=[])))
    files = {'units/u1/card.json': hashlib.sha256(card.read_bytes()).hexdigest()}
    (directory / 'bank.json').write_text(json.dumps(dict(
        schema='merlin.reference_subset_bank.v1', label='cca-all', cca_enabled=True, raw=raw, files=files,
        units=[dict(kernel_id='u1', card='units/u1/card.json', reports=[])], coverage={})))
    return hashlib.sha256((directory / 'bank.json').read_bytes()).hexdigest()


def test_grant_accepts_exactly_the_23_pinned_originals(originals):
    bundle, root, _payloads = originals
    A._verify_coverage_references(bundle, all_originals=True)
    A._verify_coverage_references(bundle, content=False, all_originals=True)


@pytest.mark.parametrize('change', ['extra_file', 'missing_hash', 'reference_map_inside', 'reference_map_grant',
                                    'map_schema_grant'])
def test_grant_refuses_anything_else(originals, change, tmp_path):
    bundle, root, _payloads = originals
    if change == 'extra_file':
        (root / 'originals/notes.md').write_text('weight reuse: derived note')
    elif change == 'missing_hash':
        (root / 'originals/o00.bin').unlink()
    elif change == 'reference_map_inside':
        (root / 'REFERENCE_MAP.json').write_text(json.dumps(dict(schema='merlin.reference_map.v1')))
    else:
        hidden = tmp_path / 'public/data.json'
        hidden.parent.mkdir()
        if change == 'reference_map_grant':
            hidden.write_text('{}')
            bundle['allowed'].append(dict(path=str(hidden), mode='ro', **{'as': 'REFERENCE_MAP.json'}))
        else:
            hidden.write_text(json.dumps(dict(schema='merlin.reference_map.v1', entries=[])))
            bundle['allowed'].append(dict(path=str(hidden), mode='ro'))
    with pytest.raises(ValueError):
        A._verify_coverage_references(bundle, all_originals=True)


def test_bank_arm_grant_is_its_all_originals_bank(originals, tmp_path, monkeypatch):
    pytest.importorskip('merlin.mining.reference_subsets')
    bundle, root, payloads = originals
    raw = [dict(id=name, path='originals/' + name + '.bin', sha256=hashlib.sha256(data).hexdigest())
           for name, data in list(payloads.items())[:-1]]  # Every original but the originals manifest.
    banks = tmp_path / 'payload/experiment/banks'
    pin = synthetic_bank(banks / 'all', raw)
    partial = synthetic_bank(banks / 'part', raw[:5])
    bundle['allowed'].append(dict(path=str(banks / 'all') + '/', mode='ro', **{'as': 'cca-evidence'}))
    assert A.bank_raw_sha256(banks / 'all') == sorted(P.COVERAGE_RAW_REFERENCE_SHA256)
    A._verify_coverage_references(bundle, bank_sha256=pin, all_originals=True)
    # A bank of fewer originals is no raw-originals bank, even with a matching raw grant.
    bundle['allowed'][-1]['path'] = str(banks / 'part') + '/'
    with pytest.raises(ValueError, match='every pinned original'):
        A._verify_coverage_references(bundle, bank_sha256=partial, all_originals=True)


def test_materialize_and_verify_bundle_check_the_grant_without_a_seed(originals, monkeypatch):
    bundle, root, _payloads = originals
    calls = []
    monkeypatch.setattr(A, '_verify_coverage_references',
                        lambda selected, **kw: calls.append((selected is not None, kw)))
    profile = A.resolve(FULL)
    selected = dict(bundle, author_profile=profile.record(), tools=list(profile.tools))
    A.verify_bundle(profile, selected)
    assert calls == [(True, dict(content=False, bank_sha256=None, all_originals=True))]
    calls.clear()
    A.verify_bundle(A.resolve(P.NAMES[0]), dict(bundle, author_profile=A.resolve(P.NAMES[0]).record(),
                                                tools=list(P.TOOLS)))
    assert calls == []


@pytest.mark.parametrize('name', [FULL, INFRA, P.NAMES[0]])
def test_staged_task_keeps_the_native_map_section_only_for_map_arms(name, tmp_path):
    workspace, run = tmp_path / 'ws', tmp_path / 'run'
    workspace.mkdir()
    run.mkdir()
    (workspace / 'TASK.md').write_text('# Task\n\n## Reference map\n\nREFERENCE_MAP.json lists entries.\n')
    D.stage_task(workspace, run, name, None)
    text = (workspace / 'TASK.md').read_text()
    assert ('REFERENCE_MAP.json lists entries.' in text) == (name == P.NAMES[0])


@pytest.fixture(scope='module')
def export(tmp_path_factory):
    """A fresh git-archive export of HEAD's sources, as fresh_campaign.py export makes it."""
    import subprocess

    from merlin.common.paths import repo_root

    root = tmp_path_factory.mktemp('export')
    archive = subprocess.run(['git', '-C', str(repo_root()), 'archive', 'HEAD', '--', 'src'],
                             capture_output=True)
    if archive.returncode:
        pytest.skip('no git checkout to export from')
    subprocess.run(['tar', '-x', '-C', str(root)], input=archive.stdout, check=True)
    return root


@pytest.mark.parametrize('name', (FULL, BANK))
def test_current_full_compiler_is_staged_like_the_coverage_arms(export, tmp_path, name):
    raw = A.prepare_sources(export, tmp_path / 'raw', profile=P.COVERAGE_NAMES[0], review_id='test')
    selection = A.prepare_sources(export, tmp_path / 'arm', profile=name, review_id='test')
    assert selection.members == raw.members and selection.diagnostics == raw.diagnostics
    document = json.loads(selection.manifest_path.read_bytes())
    assert document['author_profile'] == A.resolve(name).record()
    assert 'fresh10_projection' not in document
    A.verify_sources(name, selection)
    with pytest.raises(ValueError, match='another author profile'):
        A.verify_sources(name, raw)
