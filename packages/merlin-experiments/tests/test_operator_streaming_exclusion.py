"""Original sealed aliases and complete typed projection never become waivers."""

import json
from pathlib import Path

import pytest
import test_operator_exclusion_issuer as H
import test_operator_mixed_exclusion as MIX
from merlin_experiments.corpus import release
from merlin_experiments.phase0 import operator_exclusion as E
from merlin_experiments.phase0 import operator_exclusion_intake as I
from merlin_experiments.phase0 import operator_exclusion_issuer as U
from merlin_experiments.phase0 import operator_mixed_exclusion as M
from merlin_experiments.phase0 import operator_streaming_exclusion_issuer as S
from merlin_experiments.phase0 import operator_typed_exclusion as T
from merlin_experiments.phase1.source_inputs import fingerprint

LIMIT = 1 << 20


def fixture(tmp_path):
    section, root = H.validation_fixture(tmp_path)
    source = root/'synthetic/programs/unit/dev/model.prepared.mlir'
    target = tmp_path/'original-prepared.mlir'
    target.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(target)
    catalog = source.parent.parent/'catalog.json'
    catalog.write_text(json.dumps(dict(schema='merlin.model_check.catalog.v1', dev_prepared=str(source))))
    names = list(I._sums((root/'SHA256SUMS').read_bytes()))
    (root/'SHA256SUMS').write_text(''.join(f'{H.pin(root/name)["sha256"]}  {name}\n' for name in names))
    manifest = json.loads((root/'MANIFEST.json').read_bytes())
    manifest['seal']['sums_sha256'] = H.pin(root/'SHA256SUMS')['sha256']
    (root/'MANIFEST.json').write_text(json.dumps(manifest))
    producer = json.loads(Path(section['producer_receipt']['path']).read_bytes())
    producer['manifest_sha256'] = H.pin(root/'MANIFEST.json')['sha256']
    Path(section['producer_receipt']['path']).write_text(json.dumps(producer))
    section = dict(manifest=H.pin(root/'MANIFEST.json'), sums=H.pin(root/'SHA256SUMS'),
                   producer_receipt=H.pin(Path(section['producer_receipt']['path'])))
    return section, root, source, target


def test_explicit_alias_route_binds_exact_catalog_namespace_complete_seal_and_current_target(tmp_path, monkeypatch):
    section, root, source, target = fixture(tmp_path)
    original_read = Path.read_bytes

    def no_answers(path):
        if path.suffix == '.npy':
            pytest.fail('protected answer values are not intake inputs')
        return original_read(path)

    monkeypatch.setattr(Path, 'read_bytes', no_answers)
    with pytest.raises(ValueError, match='ordinary'):
        I.validation_sources(section)
    selected = I.validation_sources_v2(section)
    assert set(selected) == {str(source)}
    row = selected[str(source)]
    assert row['sha256'] == H.pin(target)['sha256'] and row['read_path'] == str(target)
    assert row['alias']['role'] == 'dev' and row['alias']['literal_target'] == str(target)
    assert row['alias']['catalog'] == H.pin(source.parent.parent/'catalog.json')
    assert I.validation_sources_v2(section) == selected
    (root/'extra').write_text('namespace drift')
    with pytest.raises(ValueError, match='namespace'):
        I.validation_sources_v2(section)


@pytest.mark.parametrize('mutation', ['catalog_role', 'catalog_namespace', 'target_alias', 'missing_target'])
def test_source_alias_never_borrows_another_catalog_or_nonordinary_target(tmp_path, mutation):
    section, root, source, target = fixture(tmp_path)
    if mutation.startswith('catalog'):
        path = source.parent.parent/'catalog.json'
        doc = json.loads(path.read_bytes())
        if mutation == 'catalog_role':
            doc['ref_prepared'] = doc.pop('dev_prepared')
        else:
            doc['dev_prepared'] = str(target)
        path.write_text(json.dumps(doc))
        # Old sealed ownership stays pinned: either changed bytes or wrong
        # original role must refuse, never discover a replacement catalog.
    elif mutation == 'target_alias':
        other = tmp_path/'other-source'
        other.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(other)
    else:
        target.unlink()
    with pytest.raises((ValueError, OSError)):
        I.validation_sources_v2(section)


def limits():
    return dict(max_ir_bytes=LIMIT, max_total_ir_bytes=LIMIT,
                max_source_bytes=4*LIMIT, max_total_source_bytes=4*LIMIT)


def capsule_fixture(tmp_path, monkeypatch, *, last_label='hidden'):
    root = tmp_path/'generation'
    root.mkdir()
    rows = []
    for ordinal, label in enumerate(['public', 'dev', last_label]):
        directory = root/str(ordinal)
        directory.mkdir()
        (directory/'capsule.yaml').write_text('label: '+label+'\n')
        (directory/'capsule.interface.mlir').write_text('builtin.module {}')
        rows.append(dict(member=str(ordinal), sha256=fingerprint(directory)))
    manifest = root/'MANIFEST.yaml'
    evidence = tmp_path/'evidence.json'
    evidence.write_text('{}')
    generation = tmp_path/'generation.json'
    H.write_json(manifest, dict(phase0_evidence=dict(manifest=str(evidence), generation_receipt=str(generation))))
    generation_pin = H.write_json(generation, dict(schema='merlin.phase0_generation.v1', corpus_manifest=str(manifest),
        capsule_commitments=rows, capsules_written=len(rows)))
    producer = H.write_json(tmp_path/'producer.json', dict(state='execution_succeeded'))
    prep = H.write_json(tmp_path/'prep.json', dict(generation_lineage=dict(
        generated_manifest_sha256=H.pin(manifest)['sha256'], generation_receipt_sha256=generation_pin['sha256'],
        evidence_manifest_sha256=H.pin(evidence)['sha256'], capsules=len(rows),
        capsule_roster_sha256=I.digest(sorted((r['member'], r['sha256']) for r in rows)))))
    seal = H.write_json(tmp_path/'seal.json', {})
    calls = []
    monkeypatch.setattr(release, 'verify', lambda *a: calls.append(a))
    pointer = dict(protected_capsule_inventory=dict(manifest=H.pin(manifest), generation_receipt=generation_pin,
        producer_receipt=producer, source_evidence_manifest=H.pin(evidence)),
        selected_release=dict(preparation=prep, seal=seal))
    return pointer, root, calls


def test_streaming_protection_preserves_original_public_roles_and_still_checks_entire_generation(tmp_path, monkeypatch):
    pointer, root, calls = capsule_fixture(tmp_path, monkeypatch)
    assert len(I.capsule_sources(pointer)) == 3  # Historical ordinary behavior unchanged.
    selected = I.capsule_sources(pointer, protected_only=True)
    assert set(selected) == {str(root/'2/capsule.interface.mlir')} and len(calls) == 2
    (root/'0/capsule.interface.mlir').write_text('changed public bytes')
    with pytest.raises(ValueError, match='committed capsule bytes'):
        I.capsule_sources(pointer, protected_only=True)


def test_unknown_original_roles_cannot_shrink_the_protected_inventory(tmp_path, monkeypatch):
    pointer, _, _ = capsule_fixture(tmp_path, monkeypatch, last_label='unknown')
    with pytest.raises(ValueError, match='recognized protected/public role'):
        I.capsule_sources(pointer, protected_only=True)


def test_streamed_issuer_matches_original_signature_hashes_and_retains_all_source_ids(tmp_path, monkeypatch):
    section, _, source, target = fixture(tmp_path)
    selected = I.validation_sources_v2(section)
    calls = []
    monkeypatch.setattr(I, 'resolve_streaming_v2', lambda *a: calls.append(a) or selected)
    private = tmp_path/'private'
    private.mkdir(mode=0o700)
    dest = private/'selection.json'
    live, receipt = S.issue(tmp_path/'pointer.json', 'a'*64, dest,
                            forbidden_roots=[tmp_path/'public'], **limits())
    assert len(calls) == 2 and dest.stat().st_mode & 0o777 == 0o600
    saved = json.loads(dest.read_bytes())
    assert saved == dict(schema=E.SCHEMA_V2, excluded_source_sha256=[H.pin(target)['sha256']],
                         excluded_contraction_sha256=[],
                         excluded_typed_ir_sha256=list(T.signatures(target.read_bytes(), max_bytes=LIMIT)))
    assert receipt['catalog_bound_source_aliases'] == 1
    assert receipt['signature_schema'] == T.SCHEMA and receipt['protected_source_count'] == 1
    assert receipt['parse_limits'] == dict(max_ir_bytes=LIMIT, max_total_ir_bytes=LIMIT)
    assert str(source) not in dest.read_text() and 'tensor' not in dest.read_text()
    assert repr(live) == '<OperatorPrivateExclusion>'


@pytest.mark.parametrize('mutation', ['source_hash', 'source_drift', 'target_drift', 'parse', 'stream', 'semantic'])
def test_complete_streamed_issuance_has_no_partial_selection_on_any_failure(tmp_path, monkeypatch, mutation):
    section, _, _, target = fixture(tmp_path)
    selected = I.validation_sources_v2(section)
    if mutation == 'source_hash':
        next(iter(selected.values()))['sha256'] = 'f'*64
    if mutation == 'semantic':
        target.write_text('malformed original IR')
        next(iter(selected.values()))['sha256'] = H.pin(target)['sha256']
    calls = []

    def resolve(*_):
        calls.append(None)
        if len(calls) > 1:
            if mutation == 'source_drift':
                target.write_bytes(target.read_bytes()+b'\n')
            if mutation == 'target_drift':
                return {}
        return selected

    monkeypatch.setattr(I, 'resolve_streaming_v2', resolve)
    private = tmp_path/'private'
    private.mkdir(mode=0o700)
    dest = private/'selection.json'
    bounds = limits()
    if mutation == 'parse':
        bounds['max_ir_bytes'] = 1
    if mutation == 'stream':
        bounds['max_source_bytes'] = 1
    with pytest.raises(U.IssuanceRefusal) as error:
        S.issue(tmp_path/'pointer.json', 'a'*64, dest, forbidden_roots=[tmp_path/'public'], **bounds)
    assert not dest.exists() and 'synthetic' not in str(error.value) and str(target) not in str(error.value)


def test_changed_alias_binding_is_not_hidden_by_identical_source_contents(tmp_path, monkeypatch):
    section, _, source, target = fixture(tmp_path)
    selected = I.validation_sources_v2(section)
    other = tmp_path/'same-bytes-different-target.mlir'
    other.write_bytes(target.read_bytes())
    calls = []

    def resolve(*_):
        calls.append(None)
        if len(calls) > 1:
            source.unlink()
            source.symlink_to(other)
        return I.validation_sources_v2(section)

    monkeypatch.setattr(I, 'resolve_streaming_v2', resolve)
    private = tmp_path/'private'
    private.mkdir(mode=0o700)
    with pytest.raises(U.IssuanceRefusal):
        S.issue(tmp_path/'pointer.json', 'a'*64, private/'selection.json',
                forbidden_roots=[tmp_path/'public'], **limits())
    assert not (private/'selection.json').exists() and selected[str(source)]['read_path'] == str(target)


def test_mixed_source_mode_is_explicit_v3_and_legacy_streaming_keeps_its_refusal(tmp_path, monkeypatch):
    source = tmp_path/'original-interface.mlir'
    source.write_bytes(MIX.interface())
    selected = {str(source): dict(sha256=H.pin(source)['sha256'], read_path=str(source), alias=None)}
    monkeypatch.setattr(I, 'resolve_streaming_v2', lambda *_: selected)
    private = tmp_path/'private'
    private.mkdir(mode=0o700)
    with pytest.raises(U.IssuanceRefusal):
        S.issue(tmp_path/'pointer.json', 'a'*64, private/'old-selection.json',
                forbidden_roots=[tmp_path/'public'], **limits())
    assert not (private/'old-selection.json').exists()
    destination = private/'v3-selection.json'
    _, receipt = S.issue(tmp_path/'pointer.json', 'a'*64, destination,
                         forbidden_roots=[tmp_path/'public'], source_mode=M.SCHEMA, **limits())
    saved = json.loads(destination.read_bytes())
    assert saved['schema'] == E.SCHEMA_V3
    assert saved['excluded_typed_ir_sha256'] == list(M.signatures(source.read_bytes(), max_bytes=LIMIT))
    assert receipt['signature_schema'] == M.SCHEMA and receipt['typed_signature_schema'] == T.SCHEMA
