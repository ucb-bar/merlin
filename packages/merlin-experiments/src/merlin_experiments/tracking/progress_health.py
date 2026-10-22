"""Bounded read-only progress alerts; returned verdicts are not execution evidence.

This operator view never grades, dispatches, or writes. Findings describe only
observed retained records; an incomplete inventory cannot prove campaign-wide
absence. Private snapshot identities and failure details are hidden by default.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def _epoch(value):
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except ValueError:
            return None
    return None


def _snapshot(doc):
    return (doc.get('submission_sha256') or doc.get('graded_source_sha256')
            or doc.get('candidate_sha256'))


def _rows(doc):
    rows = doc.get('per_capsule')
    if isinstance(rows, dict):
        return [v for v in rows.values() if isinstance(v, dict)]
    return [v for v in rows if isinstance(v, dict)] if isinstance(rows, list) else []


def _numeric(row):
    value = row.get('numeric')
    status = value.get('status') if isinstance(value, dict) else row.get('numeric_status')
    if row.get('numerical_execution_passed') is True or status in ('pass', 'passed'):
        return 'passed'
    if status in ('fail', 'failed', 'mismatch', 'mismatched'):
        return 'failed'
    return None  # skipped, unknown, gated and absent do not establish execution.


class ProgressHealth:
    """``sample(core_snapshot)`` reads paths from campaign.run_dir/workspace.

    Incremental response discovery is bounded by max_files; max_bytes bounds all
    new document reads per refresh. Cache entries are display reuse only, never
    trusted grading reuse. No locks or filesystem writes are performed.
    """

    def __init__(self, task: Path, operator_private=False, *, max_files=32,
                 max_bytes=2 << 20, max_records=128, stale_s=120):
        if min(max_files, max_bytes, max_records) <= 0 or stale_s < 0:
            raise ValueError('positive read bounds and nonnegative stale interval required')
        self.task = Path(task)
        self.operator_private = operator_private
        self.max_files, self.max_bytes, self.max_records = max_files, max_bytes, max_records
        self.stale_s = stale_s
        self._run = None
        self._cache = {}
        self._iterator = None
        self._archive_iterator = None
        self._archive_owners = {}

    def _read(self, path):
        if self._files >= self.max_files:
            self._notices.add('file read bound reached')
            return None
        self._files += 1
        try:
            if path.is_symlink():
                raise ValueError('indirect record refused')
            st = path.stat()
            sig = st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
            cached = self._cache.get(path)
            if cached and cached[0] == sig:
                return cached[1]
            if st.st_size > min(self._remaining, 512 << 10):
                self._notices.add('record exceeds remaining byte bound')
                return None
            with path.open('rb') as f:
                raw = f.read(min(self._remaining, 512 << 10) + 1)
            self._remaining -= len(raw)
            now = path.stat()
            if sig != (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns):
                raise ValueError('record changed while read')
            doc = json.loads(raw)
            if not isinstance(doc, dict):
                raise ValueError('record is not an object')
            self._cache[path] = sig, doc
            while len(self._cache) > self.max_records:
                self._cache.pop(next(iter(self._cache)))
            return doc
        except (OSError, ValueError, UnicodeError):
            self._cache.pop(path, None)
            return None

    def sample(self, core_snapshot):
        campaign = core_snapshot.get('campaign') or {}
        run, workspace = campaign.get('run_dir'), campaign.get('workspace')
        out = {'schema': 'merlin.progress_health.v1', 'timestamp': time.time(),
               'operator_private': self.operator_private, 'read_only': True,
               'promotion_authority': False, 'alerts': [], 'scope': 'bounded retained record inventory',
               'numeric': {'completed': None, 'passed': None, 'failed': None},
               'records_observed': 0, 'requests': [], 'snapshot_count': 0, 'snapshots': [],
               'freshness': {}, 'completed_case_records': None, 'snapshot_observations': [], 'notices': []}
        if not isinstance(run, str) or not isinstance(workspace, str):
            out['notices'] = ['current run/workspace unavailable']
            return out
        if self._run != run:
            if self._iterator is not None:
                self._iterator.close()
            self._iterator = None
            if self._archive_iterator is not None:
                self._archive_iterator.close()
            self._archive_iterator = None
            self._archive_owners.clear()
            self._cache.clear()
            self._run = run
        self._files, self._remaining, self._notices = 0, self.max_bytes, set()
        qa = Path(workspace) / 'qa'
        progress = self._read(qa / 'progress.json') or {}
        verdict = self._read(qa / 'verdict.json') or {}
        channel = Path(workspace) / '.qa_channel'
        if self._iterator is None:
            try:
                self._iterator = os.scandir(channel)
            except OSError:
                self._notices.add('selfcheck channel unavailable')
        if self._iterator is not None:
            try:
                for _ in range(self.max_files):
                    entry = next(self._iterator)
                    if entry.name.startswith('resp_') and entry.name.endswith('.json'):
                        self._read(Path(entry.path))
            except (StopIteration, OSError):
                self._iterator.close()
                self._iterator = None
        work = Path(run) / '_qa_work'
        if self._archive_iterator is None:
            try:
                self._archive_iterator = os.scandir(work)
            except OSError:
                self._notices.add('snapshot archives unavailable')
        if self._archive_iterator is not None:
            try:
                for _ in range(self.max_files):
                    entry = next(self._archive_iterator)
                    token = entry.name.removeprefix('fruns_')
                    if (not entry.name.startswith('fruns_') or not token.isascii()
                            or not token.isdecimal() or not entry.is_dir(follow_symlinks=False)):
                        continue
                    path = Path(entry.path) / 'snapshot_progress.json'
                    owner_path = work / 'background_snapshot_owners' / (token + '.json')
                    owner = self._read(owner_path)
                    doc = self._read(path)
                    if (isinstance(owner, dict) and owner.get('schema') == 'merlin.background_snapshot_owner.v1'
                            and owner.get('token') == token and owner.get('runs_dir') == str(path.parent)
                            and isinstance(doc, dict) and doc.get('schema') == 'merlin.phase1.grade_progress.v1'):
                        self._archive_owners[path] = (owner_path, owner)
                    else:
                        self._archive_owners.pop(path, None)
            except (StopIteration, OSError):
                self._archive_iterator.close()
                self._archive_iterator = None
        archives = [(p, self._cache[p][1]) for p in self._archive_owners if p in self._cache]
        # Bound provenance retention together with the document cache.
        self._archive_owners = {p: v for p, v in self._archive_owners.items() if p in self._cache}
        responses = [(p, v[1]) for p, v in self._cache.items() if p.parent == channel]
        records = [(qa / 'progress.json', progress), (qa / 'verdict.json', verdict), *responses, *archives]
        records = [(p, d) for p, d in records if d]
        out['records_observed'] = len(records)
        # Deduplicate the same snapshot/capsule represented by progress and a response.
        executed, failures, failure_details, exceptions, timeouts = {}, {}, {}, [], []
        completed_pairs = set()
        by_source = {}
        newest_response = None
        for path, doc in records:
            source = _snapshot(doc)
            if isinstance(source, str):
                by_source.setdefault(source, {'completed': set(), 'owners': []})
            if path in self._archive_owners and isinstance(source, str):
                owner_path, owner = self._archive_owners[path]
                by_source[source]['owners'].append({
                    'owner_id': owner['token'], 'owner_state': owner.get('state'),
                    'recorded_n_selected': doc.get('n_selected'),
                    'recorded_n_completed': doc.get('n_completed'),
                    'snapshot_state': doc.get('state'),
                    'source_sha256': source if self.operator_private else None,
                    'owner_admission_source_sha256': owner.get('source') if self.operator_private else None,
                    'source_authority': 'graded_source_sha256; owner admission source is separate bookkeeping',
                    'source': str(path) if self.operator_private else None,
                    'owner_source': str(owner_path) if self.operator_private else None})
            for row in _rows(doc):
                capsule = row.get('capsule') or row.get('name')
                if not isinstance(source, str) or not isinstance(capsule, str):
                    continue
                if row.get('status') in ('pass', 'passed', 'fail', 'failed', 'declined', 'error'):
                    completed_pairs.add((source, capsule))
                    by_source[source]['completed'].add(capsule)
                numeric = _numeric(row)
                if numeric:
                    executed[(source, capsule)] = numeric
                failure = row.get('failure')
                if not isinstance(failure, dict) and row.get('failure_category'):
                    failure = {'plane': row.get('failure_plane'), 'category': row['failure_category'],
                               'stage': row.get('failure_stage'), 'detail': row.get('failure_detail')}
                if numeric is None and isinstance(failure, dict) and failure.get('category'):
                    signature = (failure.get('plane'), failure.get('category'), failure.get('stage'))
                    detail = failure.get('detail')
                    detail_pin = hashlib.sha256(detail.encode()).hexdigest() if isinstance(detail, str) else None
                    key = (source, *signature)
                    failures.setdefault(key, set()).add(capsule)
                    failure_details.setdefault(key, set()).add(detail_pin)
            if doc.get('error'):
                exceptions.append(path)
                # Exact native broker outcome, not a heuristic over arbitrary stderr.
                if (doc.get('timed_out') is True
                        or doc['error'] == 'broker: self-check aborted (timed out)'):
                    timeouts.append({'stage': doc.get('timeout_stage'),
                                     'stage_status': 'recorded' if doc.get('timeout_stage') else 'UNKNOWN',
                                     'source_sha256': source if self.operator_private else None})
            if path.parent == channel:
                requested, started = doc.get('requested_at_unix_ns'), doc.get('grade_started_at_unix_ns')
                completed = doc.get('graded_at_unix_ns')
                def elapsed(a, b):
                    return (b - a) / 1e9 if type(a) is int and type(b) is int and b >= a else None
                out['requests'].append({
                    'request_id': doc.get('selfcheck_request_id') if self.operator_private else None,
                    'source_sha256': source if self.operator_private else None,
                    'queue_wall_s': elapsed(requested, started),
                    'started_to_completion_wall_s': elapsed(started, completed),
                    'total_wall_s': elapsed(requested, completed),
                    'completed_at_epoch': completed / 1e9 if type(completed) is int else None,
                    'stage': doc.get('timeout_stage'),
                    'stage_status': 'recorded' if doc.get('timeout_stage') else 'UNKNOWN',
                    'scope': 'producer request timestamps, not CPU or simulator service time'})
                ns = doc.get('graded_at_unix_ns')
                if isinstance(ns, int) and (newest_response is None or ns > newest_response[0]):
                    newest_response = ns, source
        out['completed_case_records'] = len(completed_pairs) if records else None
        out['snapshot_observations'] = [
            {'source_sha256': source if self.operator_private else None,
             'completed_case_records': len(group['completed']),
             'matches_live_progress_source': source == _snapshot(progress),
             'live_progress_claims_current': progress.get('snapshot_current') if source == _snapshot(progress) else None,
             'owners': group['owners'],
             'scope': 'unique capsules within this exact graded source; owner denominators are never summed'}
            for source, group in by_source.items()]
        passed = sum(value == 'passed' for value in executed.values())
        failed = sum(value == 'failed' for value in executed.values())
        # A native aggregate proves successful execution even if compact rows are absent.
        aggregate = max((d.get('n_execution_passed', 0) for _, d in records
                         if type(d.get('n_execution_passed', 0)) is int), default=0)
        out['snapshot_count'] = len({_snapshot(d) for _, d in records if _snapshot(d)})
        if self.operator_private:
            out['snapshots'] = sorted({_snapshot(d) for _, d in records if isinstance(_snapshot(d), str)})
        out['requests'] = sorted(out['requests'], key=lambda r: r['completed_at_epoch'] or 0, reverse=True)[:8]
        out['numeric'] = {'completed': len(executed) if records else None, 'passed': passed if records else None, 'failed': failed if records else None,
                          'aggregate_pass_evidence': aggregate if records else None,
                          'scope': 'unique observed snapshot/capsule rows; aggregate not added to avoid double counting'}
        if records and not executed and aggregate == 0:
            out['alerts'].append({'code': 'NO_OBSERVED_NUMERICAL_EXECUTION', 'severity': 'warning',
                                  'message': 'No completed numerical execution in the inspected records; returned/gated verdicts do not establish progress.'})
        if exceptions:
            out['alerts'].append({'code': 'GRADE_EXCEPTIONS', 'severity': 'warning', 'count': len(exceptions),
                                  'message': 'Grading returned exceptions; these are distinct from numerical failures.'})
        for key, capsules in failures.items():
            if len(capsules) < 2:
                continue
            source, plane, category, stage = key
            out['alerts'].append({'code': 'REPEATED_PRE_EXECUTION_FAILURE', 'severity': 'warning',
                                  'source_sha256': source if self.operator_private else None,
                                  'plane': plane, 'category': category, 'stage': stage,
                                  'detail_signature_count': len(failure_details[key]),
                                  'detail_sha256s': sorted(x for x in failure_details[key] if x) if self.operator_private else None,
                                  'capsules': sorted(capsules) if self.operator_private else None,
                                  'count': len(capsules), 'scope': 'same exact snapshot and structured failure; no unrun cases inferred'})
        if timeouts:
            out['alerts'].append({'code': 'REQUEST_TIMEOUTS', 'severity': 'warning',
                                  'count': len(timeouts), 'observations': timeouts,
                                  'message': 'Timeout attribution is UNKNOWN unless the producer recorded a stage.'})
        updated = _epoch(progress.get('updated_at'))
        out['freshness'] = {'progress_updated_at_epoch': updated,
                            'latest_response_at_epoch': newest_response[0] / 1e9 if newest_response else None,
                            'same_snapshot': bool(_snapshot(progress) and newest_response
                                                  and _snapshot(progress) == newest_response[1]),
                            'scope': 'producer timestamps from bounded observed inventory'}
        if newest_response and updated is not None and newest_response[0] / 1e9 - updated > self.stale_s:
            same = bool(_snapshot(progress) and _snapshot(progress) == newest_response[1])
            out['alerts'].append({'code': 'PROGRESS_BEHIND_SELFCHECK', 'severity': 'warning' if same else 'info',
                                  'same_snapshot': same,
                                  'lag_s': newest_response[0] / 1e9 - updated,
                                  'message': 'A completed selfcheck is newer than live progress; different snapshots are not contradictory verdicts.'})
        out['notices'] = sorted(self._notices | {'Inventory is bounded; absence is not campaign-wide proof.'})
        return out
