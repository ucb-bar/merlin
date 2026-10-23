"""Bounded joint-run memory from public inventory and already redacted feedback."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


def _read(path):
    try:
        data = Path(path).read_text()
        return json.loads(data)
    except (OSError, ValueError):
        return None


def _text(value, limit=100):
    return ''.join(c for c in str(value) if c.isprintable()).replace('`', '')[:limit]


def _rows(verdict, members):
    return {row['capsule']: row for row in (verdict.get('per_capsule') or [])
            if isinstance(row, dict) and row.get('capsule') in members}


def _tier(row, name):
    tier = (row.get('tiers') or {}).get(name)
    return isinstance(tier, dict) and tier.get('status') == 'pass'


def _failure_key(row, member):
    return (member.get('family', 'unknown'),
            row.get('failure_plane') or row.get('first_failure_plane') or row.get('plane') or 'unknown',
            row.get('failure_code') or row.get('status') or 'unknown')


def render(inventory, latest, previous=None):
    """Summarize at most three failure families and three scoped performance groups.

    Uses only explicit public membership, never raw stdout, numerical payloads,
    private comparison values or inferred correctness. A reported pass and an
    observed L2/L3 pass remain separate counters.
    """
    members = inventory['members']
    rows = _rows(latest, members)
    prior = _rows(previous or {}, members)
    source = latest.get('graded_candidate_sha256') or latest.get('graded_source_sha256')
    lines = ['# Joint compiler progress', '',
             'Build one compiler by semantic capability; preserve correct behavior while improving performance.',
             'The counts below belong to the latest evaluated snapshot, not automatically to current edits.',
             f"Compiler snapshot: `{_text(source or 'unavailable', 64)}`.", '',
             '| Public class | Observed / required | Reported pass | L2 pass | L3 pass |',
             '|---|---:|---:|---:|---:|']
    cohorts = sorted({row['cohort'] for row in members.values()})
    for cohort in cohorts:
        names = {name for name, member in members.items() if member['cohort'] == cohort}
        seen = [rows[name] for name in names if name in rows]
        lines.append(f"| {_text(cohort)} | {len(seen)} / {len(names)} | "
                     f"{sum(row.get('status') == 'pass' for row in seen)} | "
                     f"{sum(_tier(row, 'L2') for row in seen)} | {sum(_tier(row, 'L3') for row in seen)} |")
    lines += ['', 'Missing tier records remain unobserved; reported passes do not manufacture L3 evidence.', '']
    groups = defaultdict(list)
    old_keys = {_failure_key(row, members[name]) for name, row in prior.items() if row.get('status') != 'pass'}
    regressions = []
    for name, row in rows.items():
        if row.get('status') != 'pass':
            groups[_failure_key(row, members[name])].append(name)
            if prior.get(name, {}).get('status') == 'pass':
                regressions.append(name)
    lines += [f"Reported snapshot regressions: **{len(regressions)}** (repair affected shared paths).",
              '## Shared failure families', '']
    for key in sorted(groups, key=lambda key: (key in old_keys, -len(groups[key]), str(key)))[:3]:
        examples = ', '.join(f'`{_text(name)}`' for name in sorted(groups[key])[:3])
        lines.append(f"- {'new: ' if key not in old_keys else ''}{' / '.join(_text(v) for v in key)}: "
                     f"{len(groups[key])} cases; examples {examples}.")
    if not groups:
        lines.append('No failures recorded in the observed public subset; unobserved cases still need execution.')
    lines += [f'{len(groups)} failure families total. Detailed rows remain in `qa/verdict.json`.', '',
              '## Performance work', '',
              'Prioritize the scalability/performance cohort. Original correctness cases have low performance priority; '
              'calibration cases diagnose unresolved model mechanisms. Work on these concurrently.',
              'Passing code may be improved through reusable passes and shape/resource policies. '
              'Retain a working checkpoint and check affected semantic boundaries before merging.',
              'Use qualified tuned timing within its supported domain; otherwise label estimates provisional. '
              'Compare whole-call timing, preparation, host work, traffic and storage as well as the roofline.']
    from .joint_performance_priority import summarize

    cohorts = {lane: [name for name, member in members.items() if member['cohort'] == cohort]
               for lane, cohort in [('primary', 'scalability_performance'),
                                    ('correctness_regression', 'original_correctness'),
                                    ('calibration_diagnostics', 'calibration')]}
    ranked = summarize(list(rows.values()), cohorts=cohorts)
    for group in ranked['comparison_groups'][:3]:
        lane = 'qualified' if group['qualified'] else 'provisional_diagnostics'
        if not group[lane]:
            continue
        row = group[lane][0]
        model = group['comparison_group'].get('model_sha256')
        lines.append(f"- {_text(row['capsule'])}: {_text(row.get('gap_ratio'))}× structural compute floor; "
                     f"{lane}; model `{_text(model, 12)}`. Rank only within the recorded comparison scope.")
    lines += ['', 'Submit ready capabilities immediately and consume individual results while other work continues. '
              'Required original gSIM checks remain mandatory; additional calibration witnesses are selective.',
              'Read `PUBLIC_CAPABILITIES.md` first; query relevant entries in `PUBLIC_CAPABILITIES.json` on demand. '
              'Give helpers bounded capability + compiler-boundary briefs (graph, numeric/layout, schedule/storage, '
              'ISA lowering, host/runtime). Merge their completed changes into this compiler. '
              'A blocked test or capability must not stop independent construction.', '']
    return '\n'.join(lines)


def build_if_selected(run_dir, workspace, *, errata=(), notes_stale=False):
    """Use fresh inventory-bearing joint runs; preserve legacy brief behavior otherwise."""
    inventory = _read(Path(workspace) / 'PUBLIC_CAPABILITIES.json')
    if not isinstance(inventory, dict) or inventory.get('schema') != 'merlin.public_capability_inventory.v1':
        return None
    members = inventory.get('members')
    if not isinstance(members, dict) or not members:
        return None
    records = []
    paths = sorted((Path(run_dir) / 'qa_history').glob('verdict_round_*.json'))[-8:]
    for path in paths:
        value = _read(path)
        if isinstance(value, dict):
            records.append(value)
    # The live redacted verdict is the original publisher's current projection.
    live = _read(Path(workspace) / 'qa/verdict.json')
    if isinstance(live, dict) and (not records or live != records[-1]):
        records.append(live)
    latest = records[-1] if records else {}
    previous = records[-2] if len(records) >= 2 else None
    result = '\n'.join(errata) + '\n' + render(inventory, latest, previous)
    notes = Path(workspace) / 'submission/docs/iteration_notes.md'
    body = notes.read_text(errors='replace') if notes.is_file() else ''
    result += '\n## Compiler approach ledger\n\n'
    result += ('Record hypothesis, changed passes/options, exact snapshot, affected capabilities, '
               'observed outcomes and next decision in `submission/docs/iteration_notes.md`. '
               'Keep complete artifacts on disk; include their paths in bounded helper briefs.\n')
    if body.strip():
        result += '\n```markdown\n' + body[-4000:] + '\n```\n'
    if notes_stale:
        result += '\nThe approach ledger has not changed; update it with completed work and evidence.\n'
    return result
