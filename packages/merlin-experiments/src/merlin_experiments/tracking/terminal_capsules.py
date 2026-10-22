"""Bounded, read-only capsule counts from native records; never promotion authority."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path


def obj(value):
    return value if isinstance(value, dict) else {}


def full_source(value):
    return value if isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) else None


def project(rosters, progress, ledger, physical, responses):
    """Count exact-source native evidence; never combine engines across sources."""
    from .terminal_metadata import capsule_metric
    from ..phase1.feedback.joint_curriculum import _passed

    sets, aliases, perf, failures, gsim_instruments = {}, {}, set(), {}, set()
    def bucket(source):
        return sets.setdefault(source, {k: set() for k in ('all', 'L2', 'L3', 'refusal')})
    def alias(short, full):
        if isinstance(short, str) and full_source(full):
            aliases.setdefault(short, set()).add(full)
    # Native boundary records explicitly connect the distinct digest schemes.
    for name, state in obj(ledger).items():
        for tier in ('L2', 'L3'):
            records = [obj(state).get(tier), *obj(obj(obj(state).get('<certs>')).get(tier)).values()]
            for record in records:
                record = obj(record)
                boundary = obj(record.get('tier_boundary'))
                common = obj(boundary.get('common'))
                source = full_source(common.get('compiler_sha256'))
                if (not source or common.get('capsule') != name or boundary.get('tier') != tier
                        or record.get('digest') != common.get('submission_digest')
                        or boundary.get('engine') != {'L2': 'spike', 'L3': 'gsim'}[tier]):
                    continue
                alias(record.get('digest'), source)
                if tier == 'L3' and record.get('instrument'):
                    gsim_instruments.add(record['instrument'])
                if record.get('status') == 'pass':
                    bucket(source)[tier].add(name)
    def resolve(source):
        if full_source(source):
            return source
        found = aliases.get(source, set())
        return next(iter(found)) if len(found) == 1 else None

    documents = [(progress, 'progress'), *responses]
    for document, identity in documents:
        document = obj(document)
        declared = document.get('graded_source_sha256') or document.get('submission_sha256')
        for row in document.get('per_capsule', document.get('capsules', [])) or []:
            if not isinstance(row, dict):
                continue
            source = full_source(obj(row.get('compiler_source')).get('candidate_sha256')) or resolve(declared)
            if not source or not isinstance(row.get('capsule'), str):
                continue
            # A conflicting explicit full source cannot be silently relabelled.
            if full_source(declared) and source != declared:
                continue
            name, counts = row['capsule'], bucket(source)
            tiers, numeric = obj(row.get('tiers')), obj(row.get('numeric')).get('status')
            if row.get('status') == 'fail' or numeric == 'fail':
                failures.setdefault(source, set()).add(name)
            # Compact historical responses can contain tier strings. They are
            # display summaries, not complete native qualification records.
            native_row = {**row, 'tiers': {k: obj(v) for k, v in tiers.items()},
                          'numeric': obj(row.get('numeric')),
                          'expected_refusal': obj(row.get('expected_refusal'))}
            native_prerequisites = _passed(native_row)
            refusal = native_prerequisites == 'expected_refusal'
            if refusal:
                counts['refusal'].add(name)
                counts['all'].add(name)
                continue
            if numeric == 'pass':
                for tier in ('L2', 'L3'):
                    observed = obj(tiers.get(tier))
                    engine = observed.get('engine') or obj(observed.get('sim_provenance')).get('engine')
                    if observed.get('status') == 'pass' and engine == {'L2': 'spike', 'L3': 'gsim'}[tier]:
                        counts[tier].add(name)
                # Native prerequisite semantics admit exactly paired, explained
                # whole-program L0/L1 nonapplicability with all-output evidence.
                required = [v for t, v in tiers.items() if t not in {'L0', 'L1'}
                            and obj(v).get('mandatory') is True]
                extra = ('L2', 'L3') if name in (rosters.get('functional') or ()) else ('L2',)
                if (row.get('status') == 'pass' and native_prerequisites == 'spike_correctness' and required
                        and all(obj(v).get('status') == 'pass' for v in required)
                        and all(obj(tiers.get(t)).get('status') == 'pass' for t in extra)):
                    counts['all'].add(name)
            metric = capsule_metric(row, source_sha256=source)
            if metric.get('cycles') is not None:
                perf.add((source, identity))
    # This is the native already-joined projection, not a join invented by UI.
    if obj(physical).get('schema') == 'merlin.joint_physical_correctness.v1':
        source = resolve(physical.get('source_sha256'))
        if source:
            counts = bucket(source)
            for name, row in obj(physical.get('obligations')).items():
                if obj(row).get('status') == 'expected_refusal_satisfied':
                    counts['refusal'].add(name); counts['all'].add(name)
                elif obj(row).get('status') == 'pass' and physical.get('instrument_sha256') in gsim_instruments:
                    counts['L3'].add(name)
                    # This report certifies the physical obligation. It is not
                    # a complete native row for every other declared gate, so
                    # the UI must not manufacture all-gates credit by joining it.
    # A contradictory retained execution of this exact source cannot become
    # all-required coverage merely because another response reported a pass.
    for source, names in failures.items():
        bucket(source)['all'].difference_update(names)
    current = resolve(obj(progress).get('graded_source_sha256'))
    observed = bool(progress or ledger or responses)
    current_counts = sets.get(current, {k: set() for k in ('all', 'L2', 'L3', 'refusal')})
    historical = {k: set().union(*(s[k] for s in sets.values())) if sets else set() for k in current_counts}
    known = set().union(*(s for s in rosters.values() if s is not None))
    overlapping = set()
    for name in known:
        if sum(name in s for s in rosters.values() if s is not None) > 1:
            overlapping.add(name)
    def count(names, values):
        return {k: len(v & names) if names is not None and observed else None for k, v in values.items()}
    rows = []
    for name, members in rosters.items():
        names = members - overlapping if members is not None else None
        rows.append({'cohort': name, 'total': len(names) if names is not None else None,
                     **(count(names, current_counts) if current else {k: None for k in current_counts})})
    total_names = known - overlapping if all(v is not None for v in rosters.values()) else None
    return {'schema': 'merlin.terminal_capsules.v1', 'source_sha256': current,
            # A stale progress file's boolean is not a fresh working-tree observation.
            'source_current': None, 'evaluated_at': obj(progress).get('updated_at'),
            'cohorts': rows, 'total': {'total': len(total_names) if total_names is not None else None,
                **(count(total_names, current_counts) if current else {k: None for k in current_counts})},
            'historical_union': count(total_names, historical),
            'unclassified_observed': len(set().union(*historical.values()) - (known - overlapping)),
            'completed_perf_evaluations': (sum(s == current for s, _ in perf) or None) if current else None,
            'scope': 'observed native evidence; historical union is not one compiler or completion certification',
            'promotion_authority': False}


class Capsules:
    """Five-second polling, stat cache, bounded documents; no tree or ELF hashing."""
    def __init__(self, *, interval=5, max_bytes=16 << 20, max_documents=128):
        self.interval, self.max_bytes, self.max_documents = interval, max_bytes, max_documents
        self.cache, self.last, self.run, self.value = {}, 0, None, None

    def read(self, path, *, yaml=False, sha256=None):
        try:
            path = Path(path)
            if path.is_symlink():
                return {}
            st = path.stat()
            signature = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            cached = self.cache.get(path)
            if cached and cached[0] == signature:
                return cached[1] if sha256 in (None, cached[2]) else {}
            if st.st_size > min(self.remaining, 8 << 20):
                self.partial = True
                return {}
            data = path.read_bytes()
            self.remaining -= len(data)
            after = path.stat()
            if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                return {}
            digest = hashlib.sha256(data).hexdigest()
            if sha256 not in (None, digest):
                return {}
            if yaml:
                import yaml as Y
                try:
                    value = Y.load(data, Loader=getattr(Y, 'CSafeLoader', Y.SafeLoader))
                except Y.YAMLError:
                    return {}
            else:
                value = json.loads(data)
            value = obj(value)
            if len(self.cache) >= self.max_documents:
                self.cache.pop(next(iter(self.cache)))
            self.cache[path] = (signature, value, digest)
            return value
        except (OSError, ValueError, UnicodeError):
            return {}

    def sample(self, run, workspace):
        run, workspace = Path(run), Path(workspace)
        now = time.monotonic()
        if self.run != run:
            self.cache.clear(); self.run, self.last, self.value = run, 0, None
        if self.value is not None and now - self.last < self.interval:
            return self.value
        self.remaining, self.partial = self.max_bytes, False
        env = self.read(run / 'environment.yaml', yaml=True)
        policy = obj(self.read(run / 'joint_campaign.json').get('policy'))
        original = obj(policy.get('correctness_curriculum')).get('original_capsules')
        if original is None:
            original = obj(policy.get('gsim_correctness')).get('capsules')
        original = set(original) if isinstance(original, list) and all(isinstance(x, str) for x in original) else None
        calibration = obj(policy.get('native_calibration'))
        cal = self.read(calibration['path'], sha256=calibration.get('sha256')) if calibration.get('path') else {}
        cal = set(cal['members']) if isinstance(cal.get('members'), dict) else None
        scale = obj(obj(env.get('public_corpus_input')).get('public_scalability')).get('path')
        scale = self.read(Path(scale) / 'selection.json') if scale else {}
        scale = set(scale['cases']) if scale.get('schema') == 'merlin.corpus.public_scalability.staged.v1' and isinstance(scale.get('cases'), dict) else None
        progress = self.read(workspace / 'qa/progress.json')
        ledger = self.read(workspace / 'qa/tier_state.json')
        physical = self.read(workspace / 'qa/joint_physical_correctness.json')
        paths = []
        for pattern in ('resp_*.json', 'simresp_*.json'):
            for i, path in enumerate((workspace / '.qa_channel').glob(pattern)):
                if i >= 2048:
                    self.partial = True; break
                try:
                    paths.append((path.stat().st_mtime_ns, path))
                except OSError:
                    pass
        limit = max(0, self.max_documents - 8)
        if len(paths) > limit:
            self.partial = True
        responses = [(self.read(path), str(path)) for _, path in sorted(paths, reverse=True)[:limit]]
        self.value = project({'functional': original, 'calibration': cal, 'scalability': scale},
                             progress, ledger, physical, responses)
        admitted = obj(policy.get('gsim')).get('witness_capsules')
        eligible = len(cal & set(admitted)) if cal is not None and isinstance(admitted, list) else None
        for row in self.value['cohorts']:
            row['L3_obligation'] = {'functional': 'required; expected refusals are non-executing',
                                    'calibration': 'selected witnesses/diagnostics',
                                    'scalability': 'not required'}[row['cohort']]
            if row['cohort'] == 'calibration':
                row['L3_eligible_witnesses'] = eligible
                row['L3_selected_required'] = None  # Admission permission is not an obligation.
        self.value.update(partial=self.partial, refreshed_monotonic=now,
                          bytes_read=self.max_bytes - self.remaining)
        self.last = now
        return self.value
