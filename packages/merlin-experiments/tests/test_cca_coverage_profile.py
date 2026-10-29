"""The seeded CCA-coverage profile: fresh10 tooling, current sources, one pinned raw reference grant."""
import hashlib
import json
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
    assert 'No CCA products, disassemblies' in text
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
