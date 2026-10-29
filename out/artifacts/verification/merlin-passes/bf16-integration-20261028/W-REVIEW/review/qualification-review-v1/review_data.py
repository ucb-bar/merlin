"""Read F's actual MACC source/report/process records without executing native tools."""
from collections import Counter
import hashlib
import json
from pathlib import Path

B = Path('out/artifacts/verification/merlin-passes/bf16-integration-20261028')
F = B / 'F-QUALIFICATION'
OUT = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text())


def reference(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def bound(ref):
    path = Path(ref['path'])
    assert path.suffix in ('.json', '.mlir', '.txt', '.stdout'), path
    data = path.read_bytes()
    assert len(data) == ref['bytes'] and hashlib.sha256(data).hexdigest() == ref['sha256'], path
    return data


def main():
    matrix = load(F / 'fixtures-v3/MATRIX.json')
    results = load(F / 'compiled-checks-v1/RESULT.json')
    by_name = {case['name']: case for case in matrix['cases']}
    rows = []
    for case in results['cases']:
        fixture = by_name[case['name']]
        case_dir = F / 'compiled-checks-v1' / case['name']
        report = load(case_dir / 'report.json')
        process = load(case_dir / 'PROCESS.json')
        intent = load(case_dir / 'INTENT.json')
        assert process == {'returncode': 1, 'timeout': False}, case['name']
        assert intent['command'][1] == 'check-bf16-f32acc' and intent['command'][-1] == '64'
        for side in ('before', 'after'):
            assert fixture[side]['process'] == {'returncode': 0, 'timeout': False}, case['name']
            bound(fixture[side]['source'])
            exported = json.loads(bound(fixture[side]['export']))
            for key in ('sha256', 'bytes'):
                assert exported['source'][key] == fixture[side]['source'][key], case['name']
        local = report['pass_boundary']
        if case['outcome'] == 'RAW_POLICY_MATCH':
            assert all(local[key] is True for key in ('checked', 'semantic_match', 'site_qualification_passed'))
            assert local['covered_sites'] == local['expected_sites'] == report['candidates'] == fixture['sites'] > 0
        elif case['outcome'] == 'LEAN_POLICY_REFUSAL':
            assert local['checked'] is False
        elif case['outcome'] == 'DOMAIN_REFUSAL':
            assert report['candidates'] == local['covered_sites'] == 0
        else:
            raise ValueError(case)
        rows.append({'name': case['name'], 'outcome': case['outcome'], 'sites': local['covered_sites'],
                     'report': reference(case_dir / 'report.json'), 'process': reference(case_dir / 'PROCESS.json'),
                     'intent': reference(case_dir / 'INTENT.json'),
                     'authoritative_inputs': {side: {k: fixture[side][k] for k in ('source', 'export')}
                                              for side in ('before', 'after')}})

    # The selected gate has15 roots/statements. Two chain roots have extra axiom
    # audits for the segment; their elaborated statements are not in this JSON.
    audit = load(F / 'captured-receipts-v3/AUDIT.json')
    gate_roots = load(B / 'Q-INTEGRATION/source-cut-v2/ROOTS-AND-PINS.json')['roots']
    broader_roots = load(B / 'W-REVIEW/review/THEOREM_ROOTS.json')['roots']
    assert set(audit['statements']) == set(gate_roots)
    assert set(audit['axioms']) == set(broader_roots)
    assert audit['closure']['violations'] == [] and audit['closure']['theorems_audited'] == 9749
    assert all(set(axioms) <= {'propext', 'Classical.choice', 'Quot.sound'}
               for axioms in audit['axioms'].values())

    receipt_path = F / 'captured-receipts-v3/SY_model_smolvla/receipt.json'
    receipt = load(receipt_path)
    report = json.loads(bound(receipt['cert']))
    local = report['pass_boundary']
    replay_path = F / 'captured-receipts-v3/SY_model_smolvla-replay.json'
    replay = load(replay_path)
    assert all(local[k] is True for k in ('checked', 'semantic_match', 'site_qualification_passed'))
    assert local['covered_sites'] == local['expected_sites'] == report['candidates'] == 24
    assert report['qualification_passed'] is False and report['policy_reviewed'] is False
    assert receipt['verdict'] == 'abstracted'
    assert replay['fresh'] is True and replay['matches_recorded'] is True
    assert replay['stale'] == [] and replay['differences'] == []
    assert replay['rechecked']['pass_boundary'] == local
    for side in ('before', 'after'):
        bound(receipt[side])
        bound(receipt['exports'][side])
    q_sources = load(B / 'Q-INTEGRATION/source-cut-v2/SOURCES.json')['sources']
    f_sources = load(F / 'captured-receipts-v3/SOURCE-CUT.json')['sources']
    overlap = []
    for source in q_sources:
        if source['path'] in f_sources:
            assert f_sources[source['path']] == source['sha256'], source['path']
            overlap.append(source['path'])
    result = {
        'kind': 'independent DATA review of actual F results; no new native execution',
        'counts': dict(Counter(case['outcome'] for case in results['cases'])),
        'matrix': reference(F / 'fixtures-v3/MATRIX.json'), 'cases': rows,
        'audit': {'ref': reference(F / 'captured-receipts-v3/AUDIT.json'), 'theorems_audited': 9749,
                  'statement_roots': len(gate_roots), 'axiom_roots': len(broader_roots), 'std3_only': True},
        'smolvla': {'receipt': reference(receipt_path), 'report': receipt['cert'], 'replay': reference(replay_path),
                    'fresh': True, 'matches_recorded': True, 'covered_sites': 24,
                    'whole_capture_qualified': False, 'verdict': 'abstracted', 'human_reviewed': False},
        'Qv2_source_overlap_exact': overlap,
        'pending_at_observation': ['SmolVLA segment replay', 'TinyLlama replay', 'independent human policy approval'],
        'first_data_review_failure_retained': 'FIRST-DATA-REVIEW-FAILURE.txt',
    }
    (OUT / 'DATA_REVIEW.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'cases': len(rows), 'counts': result['counts'], 'smolvla_sites': 24,
                      'statement_roots': len(gate_roots), 'axiom_roots': len(broader_roots)}))


if __name__ == '__main__':
    main()
