"""Compact, scope-labelled operator progress alerts for the unified console."""
from __future__ import annotations

import math


def _count(value):
    return str(value) if type(value) is int and value >= 0 else 'unknown'


def _seconds(value):
    return f'{value:.1f}s' if type(value) in (int, float) and math.isfinite(value) and value >= 0 else 'unknown'


def summary(health):
    """Never present retained snapshot totals as current-compiler coverage."""
    if not health:
        return []
    numeric = health.get('numeric') or {}
    lines = [f" Numerical executions {_count(numeric.get('completed'))}; passed {_count(numeric.get('passed'))}; "
             f"failed {_count(numeric.get('failed'))} [observed snapshot/capsule pairs]"]
    aggregate = numeric.get('aggregate_pass_evidence')
    if type(aggregate) is int and aggregate > 0 and aggregate > (numeric.get('passed') or 0):
        lines[0] = (f" Numerical row evidence {_count(numeric.get('completed'))}; "
                    f"aggregate execution passes {aggregate} [separate, not additive; retained snapshots]")
    requests = health.get('requests') or []
    if requests:
        latest = requests[0]
        source = latest.get('source_sha256')
        label = str(source)[:12] if source else 'not recorded'
        lines.append(f" Latest check snapshot {label} | queue {_seconds(latest.get('queue_wall_s'))} | "
                     f"start-to-finish {_seconds(latest.get('started_to_completion_wall_s'))} | "
                     f"stage {latest.get('stage') or 'unknown'}")
    return lines


def alert(health):
    """Most urgent recorded diagnostic; no grading authority or missing-data pass."""
    alerts = (health or {}).get('alerts') or []
    priority = {'error': 0, 'warning': 1, 'info': 2}
    if not alerts:
        return None
    item = min(alerts, key=lambda row: priority.get(row.get('severity'), 3))
    names = {
        'NO_OBSERVED_NUMERICAL_EXECUTION': 'No completed numerical execution in observed records',
        'GRADE_EXCEPTIONS': 'Grader exceptions reported',
        'REPEATED_PRE_EXECUTION_FAILURE': 'Repeated failure before numerical execution',
        'REQUEST_TIMEOUTS': 'Self-check timeout recorded; inspect stage timings',
        'PROGRESS_BEHIND_SELFCHECK': 'Progress record trails newer self-check results',
    }
    return names.get(item.get('code'), str(item.get('code') or 'Progress diagnostic'))
