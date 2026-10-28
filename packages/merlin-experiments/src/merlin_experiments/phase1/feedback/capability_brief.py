"""Bounded joint-run memory from public inventory and already redacted feedback."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..brokers.selfcheck_tuning import DECLARATION as TUNING_DECLARATION
from . import reconcile
from .completed_grade import FILENAME as COMPLETED_FILENAME
from .completed_grade import POINTER, SUMMARY_FILENAME


def _read(path):
    try:
        data = Path(path).read_text()
        from .result_details import hydrate

        return hydrate(json.loads(data), document_path=path)
    except (OSError, ValueError):
        return None


def _text(value, limit=100):
    return "".join(c for c in str(value) if c.isprintable()).replace("`", "")[:limit]


def _rows(verdict, members):
    return {
        row["capsule"]: row
        for row in (verdict.get("per_capsule") or [])
        if isinstance(row, dict) and row.get("capsule") in members
    }


def _tier(row, name):
    tier = (row.get("tiers") or {}).get(name)
    return isinstance(tier, dict) and tier.get("status") == "pass"


def _failure_key(row, member):
    return (
        member.get("family", "unknown"),
        row.get("failure_plane") or row.get("first_failure_plane") or row.get("plane") or "unknown",
        row.get("failure_code") or row.get("status") or "unknown",
    )


def render(inventory, latest, previous=None):
    """Summarize at most three failure families and three scoped performance groups.

    Uses only explicit public membership, never raw stdout, numerical payloads,
    private comparison values or inferred correctness. A reported pass and an
    observed L2/L3 pass remain separate counters.
    """
    from .idle_tuning import LEDGER as TUNING_LEDGER

    members = inventory["members"]
    rows = _rows(latest, members)
    prior = _rows(previous or {}, members)
    source = latest.get('graded_candidate_sha256') or latest.get('graded_source_sha256')
    lines = ['# Joint compiler progress', '',
             'Build one compiler by semantic capability; preserve correct behavior while improving performance.',
             'These are per-case recorded observations, not automatically current-source qualification.',
             "Merged rows can describe different sources; each row's `graded` records its identity.",
             f"Compiler snapshot: `{_text(source or 'unavailable', 64)}`.", '',
             "Current pass counts only rows reconciled as `current` (the bytes the case reaches are unchanged "
             "since its grade); an `older_version` row is evidence for the older source it names.", '',
             '| Public class | Observed / required | Reported pass | Current pass | L2 pass | L3 pass |',
             '|---|---:|---:|---:|---:|---:|']
    cohorts = sorted({row['cohort'] for row in members.values()})
    for cohort in cohorts:
        names = {name for name, member in members.items() if member["cohort"] == cohort}
        seen = [rows[name] for name in names if name in rows]
        lines.append(f"| {_text(cohort)} | {len(seen)} / {len(names)} | "
                     f"{sum(row.get('status') == 'pass' for row in seen)} | "
                     f"{sum(row.get('status') == 'pass' and not reconcile.is_older(row) for row in seen)} | "
                     f"{sum(_tier(row, 'L2') for row in seen)} | {sum(_tier(row, 'L3') for row in seen)} |")
    if any('workload_role' in member for member in members.values()):
        lines += ['', '| Search role | Cases | Reported pass |', '|---|---:|---:|']
        for role in sorted({m.get('workload_role', 'unclassified') for m in members.values()}):
            names = [n for n, m in members.items() if m.get('workload_role', 'unclassified') == role]
            lines.append(f"| {role} | {len(names)} | {sum(rows.get(n, {}).get('status') == 'pass' for n in names)} |")
        lines += ['', 'Small controls provide correctness and mechanism diagnostics; no routine performance tuning.']
    else:
        lines += ['', 'Workload classification needed: no performance ranking or routine tuning '
                  'without selected roles.']
    lines += ['', 'Missing tier records remain unobserved; reported passes do not manufacture L3 evidence.', '']
    groups = defaultdict(list)
    old_keys = {_failure_key(row, members[name]) for name, row in prior.items() if row.get("status") != "pass"}
    regressions = []
    for name, row in rows.items():
        if row.get("status") != "pass":
            groups[_failure_key(row, members[name])].append(name)
            if prior.get(name, {}).get("status") == "pass":
                regressions.append(name)
    lines += [
        f"Reported snapshot regressions: **{len(regressions)}** (repair affected shared paths).",
        "## Shared failure families",
        "",
    ]
    for key in sorted(groups, key=lambda key: (key in old_keys, -len(groups[key]), str(key)))[:3]:
        examples = ", ".join(f"`{_text(name)}`" for name in sorted(groups[key])[:3])
        lines.append(
            f"- {'new: ' if key not in old_keys else ''}{' / '.join(_text(v) for v in key)}: "
            f"{len(groups[key])} cases; examples {examples}."
        )
    if not groups:
        lines.append('No failures recorded in the observed public subset; unobserved cases still need execution.')
    lines += [f'{len(groups)} failure families total. Completed rows remain in `qa/latest_completed_grade.json`.',
              '`failure_clusters` in both verdict files groups failing cases by one shared redacted failure, '
              'with member names, one detail and a reproduce command: each cluster is a unit a helper can take.',
              'If you edit while a grade runs, `qa/verdict.json` waits for a grade of current bytes; the newest '
              'completed result for every test case (rows, ranked headroom, gate reasons) is in '
              '`qa/latest_completed_grade.json`: a case a newer grade did not run keeps its last result, and '
              "each row's `graded` names the source digest it describes.",
              f'`qa/{SUMMARY_FILENAME}` lists those rows one line per test case, with the failure clusters; '
              'read it before querying the full file.', '',
              '## Performance work', '',
              'Performance is part of every change, not a final phase. `performance_priority.development_headroom` '
              'in `qa/verdict.json` ranks only primary-role passing test cases by cycles above their roofline bounds; '
              'work the '
              'largest headroom alongside correctness. Calibration cases diagnose unresolved model mechanisms.',
              'Passing code may be improved through reusable passes and shape/resource policies. '
              'Retain a working checkpoint and check affected semantic boundaries before merging.',
              'Use qualified tuned timing within its supported domain; otherwise label estimates provisional. '
              'Compare whole-call timing, preparation, host work, traffic and storage as well as the roofline.',
              'Idle CPUs time each `variants/<name>/` that differs from `submission/` and each value of a knob '
              f'declared in `submission/{TUNING_DECLARATION}` on the cases with the most headroom; results '
              f'appear in `{TUNING_LEDGER}` and the self-check `performance_brief`.']
    from .joint_performance_priority import summarize

    cohorts = {lane: [name for name, member in members.items() if member['cohort'] == cohort]
               for lane, cohort in [('primary', 'scalability_performance'),
                                    ('correctness_regression', 'original_correctness'),
                                    ('calibration_diagnostics', 'calibration')]}
    ranked = summarize(list(rows.values()), cohorts=cohorts,
                       roles={name: member.get("workload_role", "unclassified") for name, member in members.items()})
    for group in ranked['comparison_groups'][:3]:
        lane = 'qualified' if group['qualified'] else 'provisional_diagnostics'
        if not group[lane]:
            continue
        row = group[lane][0]
        model = group["comparison_group"].get("model_sha256")
        lines.append(
            f"- {_text(row['capsule'])}: {_text(row.get('gap_ratio'))}× structural compute floor; "
            f"{lane}; model `{_text(model, 12)}`. Rank only within the recorded comparison scope."
        )
    lines += [
        "",
        "Submit ready capabilities immediately and consume individual results while other work continues. "
        "Required original gSIM checks remain mandatory; additional calibration witnesses are selective.",
        "Read `PUBLIC_CAPABILITIES.md` first; query relevant entries in `PUBLIC_CAPABILITIES.json` on demand. "
        "Give helpers bounded capability + compiler-boundary briefs (graph, numeric/layout, schedule/storage, "
        "ISA lowering, host/runtime). Merge their completed changes into this compiler. "
        "A blocked test or capability must not stop independent construction.",
        "",
    ]
    return "\n".join(lines)


def build_if_selected(run_dir, workspace, *, errata=(), notes_stale=False):
    """Use fresh inventory-bearing joint runs; preserve legacy brief behavior otherwise."""
    inventory = _read(Path(workspace) / "PUBLIC_CAPABILITIES.json")
    if not isinstance(inventory, dict) or inventory.get("schema") != "merlin.public_capability_inventory.v1":
        return None
    members = inventory.get("members")
    if not isinstance(members, dict) or not members:
        return None
    records = []
    paths = sorted((Path(run_dir) / "qa_history").glob("verdict_round_*.json"))[-8:]
    for path in paths:
        value = _read(path)
        if isinstance(value, dict):
            records.append(value)
    # The live redacted verdict is the original publisher's current projection.
    live = _read(Path(workspace) / "qa/verdict.json")
    if not (isinstance(live, dict) and live.get("per_capsule")):
        # An author who keeps editing leaves no snapshot current, so the live verdict can hold only the
        # model gate for hours; the newest completed result per test case is the readable progress then.
        from .completed_grade import FILENAME

        completed = _read(Path(workspace) / "qa" / FILENAME)
        live = completed if isinstance(completed, dict) and completed.get("per_capsule") else live
    if isinstance(live, dict) and (not records or live != records[-1]):
        records.append(live)
    completed = _read(Path(workspace) / 'qa' / COMPLETED_FILENAME)
    if isinstance(completed, dict) and isinstance(completed.get(POINTER), dict):
        records.append(completed)
    latest = records[-1] if records else {}
    previous = records[-2] if len(records) >= 2 else None
    result = "\n".join(errata) + "\n" + render(inventory, latest, previous)
    from ..iteration_notes import NOTES_NAME, read_notes

    body = read_notes(workspace)
    result += "\n## Compiler approach ledger\n\n"
    result += (
        "Record hypothesis, changed passes/options, exact snapshot, affected capabilities, "
        f"observed outcomes and next decision in `{NOTES_NAME}` (workspace root, outside submission/). "
        "Keep complete artifacts on disk; include their paths in bounded helper briefs.\n"
    )
    if body.strip():
        result += "\n```markdown\n" + body[-4000:] + "\n```\n"
    if notes_stale:
        result += "\nThe approach ledger has not changed; update it with completed work and evidence.\n"
    return result


def refresh_completed(workspace, document):
    """Refresh existing joint-session memory from the durable merged grade, without grading authority."""
    from .brief import _leading_resume_note
    from .reference_experiments import brief_lines

    workspace = Path(workspace)
    inventory = _read(workspace / 'PUBLIC_CAPABILITIES.json')
    if (not isinstance(inventory, dict)
            or inventory.get('schema') != 'merlin.public_capability_inventory.v1'
            or not isinstance(inventory.get('members'), dict) or not inventory['members']):
        return False
    path = workspace / 'qa/round_brief.md'
    existing = path.read_text() if path.is_file() else ''
    pointer = document.get(POINTER) or {}
    prefix, marker, prior = existing.partition('# Joint compiler progress')
    prefix = prefix if marker else _leading_resume_note(existing)
    ledger_marker = '\n## Compiler approach ledger'
    _, has_ledger, ledger = prior.partition(ledger_marker)
    reference = "\n".join(brief_lines(workspace))
    text = (prefix + (reference + "\n" if reference else "") + render(inventory, document)
            + f"\nCompleted-grade token: `{_text(pointer.get('token', 'unavailable'))}`. "
            + f"Read `qa/{SUMMARY_FILENAME}` for per-case source/tier identities.\n")
    if has_ledger:
        text += ledger_marker + ledger
    temporary = path.with_suffix('.next.md')
    temporary.write_text(text)
    temporary.replace(path)
    return True
