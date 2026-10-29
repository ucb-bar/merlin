"""Read-only source/log/type binding of completed F2 case reports.

Never launch/control foreign jobs, read SDK/object contents, or issue authority.
An incomplete running campaign remains incomplete regardless of case successes.
"""
from pathlib import Path
import hashlib
import json
import sys

BASE = Path(__file__).resolve().parent
BUFFER = BASE.parents[2]
REPO = Path.cwd()


def exact(ref):
    path = Path(ref['path'])
    assert path.is_relative_to(REPO)
    assert path.suffix not in ('.olean', '.so', '.a', '.o', '.pyc')
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref['sha256'] and len(data) == ref['bytes']
    return data


positives = json.loads((BUFFER / 'F/qualification-v2/POSITIVES.json').read_text())
rows = []
for case in positives:
    if case['status'] != 'checked':
        rows.append(dict(case=case['name'], observed_status=case['status'], reviewed=False))
        continue
    report = json.loads(exact(case['report']))
    root = Path(case['report']['path']).parent
    expected = BUFFER / 'Q/source-cut-v5/case-requests' / case['name']
    roots = json.loads((expected / 'ROOTS.json').read_text())
    assert report['status'] == 'checked' and report['audit']['violations'] == []
    assert set(report['audit']['axioms_used']) <= {'propext', 'Classical.choice', 'Quot.sound'}
    assert set(report['statements']) == set(roots)
    assert report['audit_modules'] == [r['module'] for r in report['generated_queries']]
    assert len(report['generated_queries']) == 12
    for i, ref in enumerate(report['actual4']):
        assert exact(ref) == (expected / f'stage{i}.mlir').read_bytes()
    for row in report['generated_queries']:
        assert exact(row['source']) == (expected / (row['module'] + '.lean')).read_bytes()
    for ref in report['sources']:
        assert Path(ref['path']).suffix in ('.lean', '.py')
        # Some source refs use size rather than bytes.
        exact(dict(ref, bytes=ref.get('bytes', ref.get('size'))))
    processes = []
    for ref in report['process_evidence']:
        data = exact(ref)
        if ref['path'].endswith('.process.json'):
            process = json.loads(data)
            assert process['returncode'] == 0
            processes.append(process)
    assert json.loads((root / 'audit.stdout').read_text()) == report['audit']
    assert json.loads((root / 'types.stdout').read_text())['statements'] == report['statements']
    project_imports = [row for row in report['audit']['imports']
                       if row['module'].startswith('Merlin.') or row['module'] in report['audit_modules']]
    for row in project_imports:
        expected_object = root / 'lib' / (row['module'].replace('.', '/') + '.olean')
        assert Path(row['olean']).resolve() == expected_object.resolve()
    flags = {key: report[key] for key in ('independent_review_approved', 'independently_qualified',
                                        'native_allocator_conformance', 'production_enforcement',
                                        'full_error_world_equivalence')}
    assert not any(flags.values())
    rows.append(dict(case=case['name'], observed_status='checked', report=case['report'],
                     raw4_exact=True, generated_sources_exact=12, bound_sources_exact=len(report['sources']),
                     project_import_paths=len(project_imports), process_successes=len(processes),
                     roots=len(roots), audit_theorems=report['audit']['theorems_audited'], std3=True,
                     consumer_called=case['consumer_called'], final_bytes_identical=case['final_bytes_identical'],
                     flags=flags, object_or_sdk_contents_read=False))
output = BASE / sys.argv[1]
with output.open('x') as stream:
    json.dump(dict(cases=rows, required_positive_count=5, full_campaign_qualification=False,
                   limitation='Read-only case review; F owns full campaign/seal, Q5 imported-source gap remains explicit.'),
              stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(observed_cases=len(rows), checked=sum(row['observed_status'] == 'checked' for row in rows))))
