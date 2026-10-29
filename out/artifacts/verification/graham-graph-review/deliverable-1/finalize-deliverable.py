"""Check report links and gap ids, then write the artifact manifest.

Run last, after audit-deliverable.py. Read-only outside this directory.
"""
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
INDEX = json.loads((OUT / 'source-index.json').read_text())
rows = list(csv.DictReader((OUT / 'coverage.csv').open()))
open_gaps = sorted({g for r in rows for g in r['gap_ids'].split(';')})
validation = json.loads((OUT / 'validation.json').read_text())

links, broken, documented = 0, [], set()
for page in sorted(OUT.glob('*.md')):
    text = page.read_text()
    documented |= {word.strip('|* ') for word in text.split() if word.strip('|* ').startswith('D1-G')}
    for part in text.split('](')[1:]:
        target = part.split(')')[0].split('#')[0]
        if not target or target.startswith(('http:', 'https:')):
            continue
        links += 1
        if target.startswith('/') or not (page.parent / target).resolve().exists():
            broken.append({'page': page.name, 'target': target})
gaps_table = {line.split('|')[1].strip() for line in (OUT / 'gaps.md').read_text().splitlines() if line.startswith('| D1-G')}
review = {
    'local_markdown_links_checked': links,
    'broken_or_absolute_links': broken,
    'matrix_gap_ids_missing_from_gaps_md': sorted(set(open_gaps) - gaps_table),
    'gaps_md_ids_unused_by_matrix': sorted(gaps_table - set(open_gaps)),
    'audit_checks': len(validation['checks']),
    'audit_passed': validation['passed'],
    'open_gaps': open_gaps,
    'no_new_Lean_qualification': True,
}
(OUT / 'final-review.json').write_text(json.dumps(review, indent=2) + '\n')

files = [{'path': str(p.relative_to(OUT)), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
         for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'audit-manifest.json']
(OUT / 'audit-manifest.json').write_text(json.dumps({
    'schema': 'graham.deliverable-1.artifacts.v3', 'producer_revision': INDEX['producer_revision'],
    'verification_revision': INDEX['verification_revision'], 'new_qualification': False, 'files': files}, indent=2) + '\n')
print(json.dumps({k: review[k] for k in ('local_markdown_links_checked', 'broken_or_absolute_links',
                  'matrix_gap_ids_missing_from_gaps_md', 'gaps_md_ids_unused_by_matrix', 'audit_passed')}, indent=2))
ok = review['audit_passed'] and not broken and not review['matrix_gap_ids_missing_from_gaps_md'] and not review['gaps_md_ids_unused_by_matrix']
raise SystemExit(not ok)
