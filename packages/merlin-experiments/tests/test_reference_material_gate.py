"""The read-only reference-material gate over rendered author workspaces."""
import hashlib
import json
import os
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import fresh10_profiles as P
from merlin_experiments.phase1 import reference_material_gate as G

FULL, BANK, INFRA = P.RAW_ORIGINALS_NAMES


@pytest.fixture
def pins(monkeypatch):
    payloads = {'o%02d' % index: ('original %d\n' % index).encode() * 8 for index in range(23)}
    # One original legitimately carries the annotation words; originals are exempt.
    payloads['o03'] = b'/* weight reuse: computes are COMPUTE_AND_STAY in this header */\n'
    digests = tuple(hashlib.sha256(data).hexdigest() for data in payloads.values())
    monkeypatch.setattr(P, 'COVERAGE_RAW_REFERENCE_SHA256', digests)
    monkeypatch.setattr(P, 'COVERAGE_ORIGINALS_MANIFEST_SHA256', digests[-1])
    return payloads


def _manifest(payloads):
    return dict(schema='operator.allowed_reference_sources.v2', origins=[dict(name='originals', files=[
        dict(member=name + '.bin', destination=name + '.bin', sha256=hashlib.sha256(data).hexdigest())
        for name, data in payloads.items()])])


def workspace(tmp_path, payloads, name=FULL):
    ws = tmp_path / 'arm' / 'workspace'
    root = ws / 'reference-kernels'
    (root / 'originals').mkdir(parents=True)
    for member, data in payloads.items():
        (root / 'originals' / (member + '.bin')).write_bytes(data)
    (root / 'reference-manifest.json').write_text(json.dumps(_manifest(payloads)))
    (root / 'reference-selection.json').write_text('{}')
    (ws / A.RECORD).write_text(json.dumps(A.resolve(name).record(), sort_keys=True))
    (ws / 'TASK.md').write_text('# Task\n\nBuild one general compiler.\n')
    (ws / 'layers').mkdir()
    (ws / 'layers/conv.json').write_text(json.dumps(dict(op='conv2d')))
    for owned in ('submission', 'artifacts', '.qa_channel'):
        (ws / owned).mkdir()
    (ws / 'submission/notes.md').write_text('weight reuse: my own analysis')  # Author work is not input.
    return ws


def reasons(report):
    return {(row['check'], row['reason']) for row in report['failures']}


def test_clean_raw_originals_workspace_passes(tmp_path, pins):
    ws = workspace(tmp_path, pins)
    manifest = tmp_path / 'pinned-manifest.json'
    manifest.write_text(json.dumps(_manifest(pins)))
    for selected in (None, manifest):
        report = G.check_workspace(ws, manifest=selected)
        assert report['status'] == 'pass', report['failures']
        assert report['granted_reference_files'] == 23 and report['author_profile'] == FULL
    assert 'submission' not in ''.join(report['rendered_inputs'])


@pytest.mark.parametrize('plant, expected', [
    ('map_top_level', {('allow_list', 'reference_derived_input'), ('tripwire', 'reference_map_schema')}),
    ('map_neutral_name', {('tripwire', 'reference_map_schema'), ('provenance', 'hand_authored_derived_artifact')}),
    ('map_inside_references', {('allow_list', 'unlisted_reference_file'), ('tripwire', 'reference_map_schema')}),
    ('techniques', {('tripwire', 'techniques_field_in_reference_derived_json'),
                    ('provenance', 'hand_authored_derived_artifact')}),
    ('extra_reference_file', {('allow_list', 'unlisted_reference_file'),
                              ('profile_record', 'record_omits_granted_reference_files')}),
    ('missing_original', {('allow_list', 'pinned_reference_file_missing')}),
    ('annotation', {('tripwire', 'reference_map_annotation')}),
    ('map_hash_mention', {('tripwire', 'mentions_withheld_reference_map')}),
    ('reference_notes', {('allow_list', 'reference_derived_input')}),
    ('unpinned_bank', {('allow_list', 'bank_rendered_without_a_pinned_bank_arm')}),
])
def test_planted_reference_material_fails(tmp_path, pins, plant, expected):
    ws = workspace(tmp_path, pins)
    map_document = json.dumps(dict(schema='merlin.reference_map.v1', entries=[]))
    if plant == 'map_top_level':
        (ws / 'REFERENCE_MAP.json').write_text(map_document)
    elif plant == 'map_neutral_name':
        (ws / 'index.json').write_text(map_document)
    elif plant == 'map_inside_references':
        (ws / 'reference-kernels/originals/REFERENCE_MAP.json').write_text(map_document)
    elif plant == 'techniques':
        (ws / 'kernels.json').write_text(json.dumps(dict(
            derived_from=dict(map_sha256='0' * 64), entries=[dict(techniques=['weight stationary'])])))
    elif plant == 'extra_reference_file':
        (ws / 'reference-kernels/originals/conv.dis').write_text('derived disassembly')
    elif plant == 'missing_original':
        (ws / 'reference-kernels/originals/o00.bin').unlink()
    elif plant == 'annotation':
        (ws / 'TASK.md').write_text('# Task\n\nweight reuse: 378 of 496 computes are COMPUTE_AND_STAY\n')
    elif plant == 'map_hash_mention':
        (ws / 'TASK.md').write_text('# Task\n\nSee map ' + sorted(P.WITHHELD_REFERENCE_MAP_SHA256)[0] + '.\n')
    elif plant == 'reference_notes':
        (ws / 'REFERENCE_NOTES.md').write_text('locators')
    else:
        (ws / 'cca-evidence').mkdir()
        (ws / 'cca-evidence/bank.json').write_text('{}')
    report = G.check_workspace(ws)
    assert report['status'] == 'fail'
    assert expected <= reasons(report), report['failures']


def test_withheld_map_bytes_trip_and_include_the_named_hashes(tmp_path, pins):
    for named in ('9d7d56c59072493f2f1aa4b3a2471811e21a5da62e1d548e900201e275bfcfd3',
                  '51568c1de0842ab143d2fad503a146a9a41056f08c21c3ae73fdf7dbbbf65720',
                  '5134c637c5d154008e7c877090a72868009f82655d4b2d23e58c8efa4c173ef7'):
        assert named in P.WITHHELD_REFERENCE_MAP_SHA256
    ws = workspace(tmp_path, pins)
    data = b'{"entries": []}\n'
    pins_with_map = P.WITHHELD_REFERENCE_MAP_SHA256 | {hashlib.sha256(data).hexdigest()}
    (ws / 'opaque.bin').write_bytes(data)
    import unittest.mock as mock

    with mock.patch.object(P, 'WITHHELD_REFERENCE_MAP_SHA256', pins_with_map):
        assert ('tripwire', 'withheld_reference_map') in reasons(G.check_workspace(ws))


def test_rendered_directories_are_read_only_when_deep(tmp_path, pins):
    ws = workspace(tmp_path, pins)
    (ws / 'layers/index.json').write_text(json.dumps(dict(schema='merlin.reference_map.v1')))
    shallow = G.check_workspace(ws)
    assert shallow['status'] == 'pass' and 'layers' in shallow['directories_not_content_scanned']
    assert ('tripwire', 'reference_map_schema') in reasons(G.check_workspace(ws, deep=True))


def test_hardware_instruction_names_are_not_map_annotations(tmp_path, pins):
    ws = workspace(tmp_path, pins)
    (ws / 'target_contract.yaml').write_text('instructions:\n  - COMPUTE_AND_STAY\n')
    assert G.check_workspace(ws)['status'] == 'pass'


def test_profile_record_must_hash_every_granted_file(tmp_path, pins):
    ws = workspace(tmp_path, pins, name=P.NAMES[0])  # infra records no reference hashes
    report = G.check_workspace(ws)
    assert ('profile_record', 'record_omits_granted_reference_files') in reasons(report)
    no_record = workspace(tmp_path / 'x', pins)
    (no_record / A.RECORD).unlink()
    assert ('profile_record', 'no_profile_record_for_granted_references') in reasons(G.check_workspace(no_record))


def test_no_reference_arm_must_render_no_references(tmp_path, pins):
    ws = workspace(tmp_path, pins, name=P.NAMES[2])
    found = reasons(G.check_workspace(ws))
    assert ('allow_list', 'unlisted_reference_file') in found
    assert ('profile_record', 'no_reference_profile_with_granted_references') in found


def test_bank_arm_uses_the_banks_own_provenance_record(tmp_path, pins, monkeypatch):
    pytest.importorskip('merlin.mining.reference_subsets')
    from test_fresh10_raw_originals_profiles import synthetic_bank

    ws = workspace(tmp_path, pins, name=BANK)
    raw = [dict(id=name, path='originals/' + name + '.bin', sha256=hashlib.sha256(data).hexdigest())
           for name, data in list(pins.items())[:-1]]
    source = tmp_path / 'conditions/cca-all'
    pin = synthetic_bank(source, raw)
    monkeypatch.setitem(P.RAW_ORIGINALS_BANKS, BANK, pin)
    (ws / A.RECORD).write_text(json.dumps(A.resolve(BANK).record(), sort_keys=True))
    import shutil

    shutil.copytree(source, ws / 'cca-evidence')
    report = G.check_workspace(ws, bank=source)
    assert report['status'] == 'pass', report['failures']
    assert report['allowed_reference_source'] == 'bank raw list'
    [row] = report['provenance']
    assert row['equivalent']['tool'] == 'merlin.reference_subset_bank.v1' and row['equivalent']['inputs'] == 22
    # A file outside the bank's pins no longer verifies.
    os.chmod(ws / 'cca-evidence', 0o755)
    (ws / 'cca-evidence/notes.json').write_text('{}')
    assert ('provenance', 'bank_pins_do_not_verify') in reasons(G.check_workspace(ws, bank=source))
    (ws / 'cca-evidence/notes.json').unlink()
    # A bank arm without its bank rendered fails.
    shutil.rmtree(ws / 'cca-evidence')
    assert ('allow_list', 'pinned_bank_not_rendered') in reasons(G.check_workspace(ws))


def test_cross_arm_diff_explains_capability_inputs_and_flags_the_rest(tmp_path, pins):
    full = workspace(tmp_path / 'full', pins)
    infra = workspace(tmp_path / 'infra', pins, name=INFRA)
    for name in ('OPTIMIZATION_CATALOG.json', 'SELECTED_LIBRARY_GUIDANCE.md'):
        (full / name).write_text('{}')
    (infra / 'REFERENCE_MAP.json').write_text('{}')
    diff = G.diff_arms([G.check_workspace(full), G.check_workspace(infra)])
    assert {row['input'] for row in diff['explained']} == {'OPTIMIZATION_CATALOG.json', 'SELECTED_LIBRARY_GUIDANCE.md'}
    assert [row['input'] for row in diff['unexplained']] == ['REFERENCE_MAP.json']
    assert diff['status'] == 'fail'


def test_author_work_after_launch_is_not_rendered_input(tmp_path, pins):
    ws = workspace(tmp_path, pins)
    record = ws.parent / '.workspace.bwrap-args.0'
    record.write_text('')
    os.utime(record, (1_000_000, 1_000_000))
    for path in ws.rglob('*'):
        os.utime(path, (999_000, 999_000), follow_symlinks=False)
    (ws / 'my_notes.json').write_text(json.dumps(dict(schema='merlin.reference_map.v1')))
    report = G.check_workspace(ws)
    assert report['status'] == 'pass' and report['author_written_top_level'] == ['my_notes.json']
    assert G.check_workspace(ws, launched=False)['status'] == 'fail'


def test_preflight_continuation_writes_its_report_and_refuses(tmp_path, pins):
    ws = workspace(tmp_path, pins)
    run = tmp_path / 'run'
    run.mkdir()
    prepared = SimpleNamespace(bundle=dict(author_profile=A.resolve(FULL).record()), workspace=ws, run_dir=run)
    assert G.check_prepared(prepared)['status'] == 'pass'
    assert json.loads((run / 'reference_material_gate.json').read_text())['status'] == 'pass'
    (ws / 'REFERENCE_MAP.json').write_text('{}')
    with pytest.raises(ValueError, match='reference material gate failed'):
        G.check_prepared(prepared)
    assert G.check_prepared(SimpleNamespace(bundle={}, workspace=ws, run_dir=run)) is None


def test_cli_writes_a_report(tmp_path, pins, capsys):
    ws = workspace(tmp_path, pins)
    manifest, report = tmp_path / 'm.json', tmp_path / 'report.json'
    manifest.write_text(json.dumps(_manifest(pins)))
    assert G.main([str(ws), '--manifest', str(manifest), '--report', str(report)]) == 0
    assert json.loads(report.read_text())['status'] == 'pass'
    assert G.main(['--diff', str(report), str(report), '--report', str(tmp_path / 'diff.json')]) == 0
    (ws / 'REFERENCE_MAP.json').write_text('{}')
    assert G.main([str(ws), '--manifest', str(manifest)]) == 1
