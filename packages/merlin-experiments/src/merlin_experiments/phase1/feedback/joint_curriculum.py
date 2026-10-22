"""Bounded correctness progression using private, exact native evidence.

Scheduling observations never certify coverage or authorize physical measurements.
A changed compiler is recompiled; the native ELF/instrument cache still decides
whether its unchanged executable can reuse numerical or physical evidence.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from merlin.common.read_lease_digest import digest as leased_digest


def digest(path):
    return leased_digest(path) or sha256(Path(path).read_bytes()).hexdigest()


def _key(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def _pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('curriculum evidence must be a direct regular file')
    return {'path': str(path.resolve()), 'sha256': digest(path)}


def _source(path):
    leaves = sorted(Path(path).rglob('*'))
    if any(p.is_symlink() for p in leaves):
        raise ValueError('curriculum source reuse refuses symlink leaves')
    return _key({str(p.relative_to(path)): digest(p) for p in leaves if p.is_file()})


def _passed(row):
    tiers = row.get('tiers') or {}
    numeric = row.get('numeric') or {}
    # Negative conformance/refusal cases are not numerical calibration witnesses.
    refusal = row.get('expected_refusal') or {}
    if (row.get('status') == 'pass' and refusal.get('status') == 'met'
            and row.get('declined') and not row.get('failure')):
        return 'expected_refusal'
    structural = all(tiers.get(t, {}).get('status') == 'pass' for t in ('L0', 'L1'))
    whole_program_reasons = {
        'L0': 'command-buffer reference cannot interpret scalar regions of a whole-program ELF',
        'L1': 'command-buffer simulate cannot execute scalar regions of a whole-program ELF'}
    whole_program = all(tiers.get(t, {}).get('status') == 'skipped'
        and tiers[t].get('not_applicable') is True and tiers[t].get('reason') == reason
        for t, reason in whole_program_reasons.items())
    outputs = numeric.get('per_output')
    whole_program = (whole_program and isinstance(outputs, dict) and bool(outputs)
        and numeric.get('missing_outputs') == [] and numeric.get('mismatch_count') == 0
        and isinstance(numeric.get('golden_source'), str) and bool(numeric['golden_source'])
        and all(isinstance(v, dict) and v.get('status') == 'pass'
                and type(v.get('n_elements')) is int and v['n_elements'] > 0 for v in outputs.values()))
    if (numeric.get('status') == 'pass' and (structural or whole_program)
            and tiers.get('L2', {}).get('status') == 'pass' and tiers['L2'].get('engine') == 'spike'
            and not row.get('failure')
            and (row.get('compiler_source') or row.get('candidate_native_execution'))):
        return 'spike_correctness'
    return None


def plan(records, *, policy, attempts, passed, hint=(), limit=None):
    """Rotate unresolved original prerequisites without dropping any obligation."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('curriculum preview limit must be positive')
    if len(records) > 2048:
        raise ValueError('curriculum exceeds bounded admitted roster')
    names = {r[2] for r in records}
    configured = policy.get('correctness_curriculum', {}).get('original_capsules')
    from ..joint_control import held_capsules
    held = held_capsules({'policy': policy})
    if configured is not None and (not isinstance(configured, list)
            or any(type(n) is not str for n in configured)
            or len(configured) != len(set(configured)) or not set(configured) <= names | held):
        raise ValueError('original correctness roster must name selected core cases only')
    original = (set(configured) if configured is not None else names) - held
    witnesses = set(policy.get('gsim', {}).get('witness_capsules', ())) & names
    questions = policy.get('operator_measurement_policy', {}).get('witness_questions', {})
    passed_names = set(passed) & names
    unresolved = names - passed_names
    preferred = set(hint) & unresolved
    # Prior attempts precede witness/hint priority: a fixed hint cannot starve
    # middle regimes. Missing source/engine evidence never becomes a pass here.
    ranked = sorted((r for r in records if r[2] in unresolved),
        key=lambda r: (attempts.get(r[2], 0), r[2] not in original,
                       r[2] not in witnesses, r[2] not in preferred, r[1], r[0], r[2]))
    chosen = ranked[:limit]
    witness_rows = [dict(capsule=n, correctness=('matched_spike_pass' if passed.get(n) == 'spike_correctness'
                         else 'pending_matching_spike_pass'), question=questions.get(n, 'selected witness; unresolved question required'),
                         physical_status='unknown_until_exact_ELF_input_config_scope_and_model_join',
                         dispatch_authorized=False)
                    for n in sorted(witnesses)] if isinstance(passed, dict) else []
    # Keep statuses distinct from set membership (including expected refusals).
    return chosen, dict(schema='merlin.joint_correctness_curriculum.v1',
        original_development_required=len(original), original_unresolved=len(original - passed_names),
        original_final_only_retained=len(set(configured or ()) & held),
        exact_correctness_reused=sorted(passed_names), pending=sorted(unresolved),
        next_batch=[r[2] for r in chosen[:16]],
        next_batch_after_this=[r[2] for r in ranked[(limit or 16):(limit or 16)+16]],
        queued=[r[2] for r in chosen], queue_is_execution=False,
        original_roster_basis='explicit_operator_roster' if configured is not None else 'all_selected_core_conservatively',
        witness_prerequisites=witness_rows,
        physical_rule='cheap source/structural/numerical Spike first; required original gSIM correctness is separate from selective calibration questions and fit qualification',
        parallel_large_spike='independent source cases remain available concurrently; no global correctness sweep blocks them',
        correctness_scope='L0/L1/L2 Spike prerequisites only; original required L3/gSIM obligations remain unchanged and unsatisfied by this report',
        full_coverage_required=True, certified_complete=False)


class Progress:
    def __init__(self, root, *, identity, runs_root, original_capsules=None, verify_reuse=None,
                 ownership_identity=None):
        self.root, self.identity, self.runs_root = Path(root), identity, Path(runs_root)
        self.original_capsules = original_capsules
        self.owner = uuid4().hex
        # Ownership only prevents duplicate work; this weaker fallback must never
        # enable receipt reuse when full simulator identity is unavailable.
        self.ownership_identity = (identity if identity is not None else ownership_identity)
        if self.ownership_identity is None:
            self.ownership_identity = {'conservative_unqualified_run': str(self.root.resolve())}
        self.ownership_keys = {}
        self.verify_reuse = verify_reuse
        self.paths = {}
        self.admitted = set()
        self.priorities = {}
        self.keys = {}
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _ledger(self):
        path = self.root / 'progress.json'
        with (self.root / '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = json.loads(path.read_text()) if path.is_file() else {'version': 1, 'members': {}}
            if (value.get('version') != 1 or len(value.get('members', {})) > 2048
                    or len(value.get('in_flight', {})) > 4096):
                raise ValueError('invalid bounded curriculum ledger')
            yield value
            temporary = path.with_suffix('.next.json')
            temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
            temporary.replace(path)

    def _identity(self, name):
        return _key({'context': self.identity, 'source': _source(self.paths[name])})

    def select(self, records, *, policy, hint):
        if self.original_capsules is not None:
            policy = dict(policy, correctness_curriculum={'original_capsules': self.original_capsules})
        self.records, self.policy, self.hint = records, policy, hint
        self.paths = {r[2]: r[3] for r in records}
        # Both key spaces describe this same selection observation. Reuse the
        # observed digest within this call; completion and later selections
        # still reopen source identity through their original custody checks.
        source_digests = {name: _source(path) for name, path in self.paths.items()}
        self.keys = ({name: _key({'context': self.identity, 'source': source_digests[name]})
                      for name in self.paths}
                     if self.identity is not None else {})
        self.ownership_keys = {name: _key({'context': self.ownership_identity,
            'capsule': name, 'source': source_digests[name]}) for name in self.paths}
        passed, attempts, active = {}, {}, set()
        reuse_checked = False
        reuse_allowed = True
        with self._ledger() as ledger:
            for name in self.paths:
                entry = ledger['members'].get(name, {})
                if self.ownership_keys[name] in ledger.get('in_flight', {}):
                    active.add(name)
                attempts[name] = entry.get('admissions', entry.get('attempts', 0))
                if self.identity is None or not entry.get('passes'):
                    continue
                saved = entry['passes'].get(self.keys[name])
                if saved is None:
                    continue
                try:
                    if not reuse_checked and self.verify_reuse is not None:
                        reuse_checked = True
                        try:
                            self.verify_reuse()
                        except (OSError, ValueError, RuntimeError):
                            reuse_allowed = False
                    if not reuse_allowed:
                        continue
                    pin = saved['result']
                    if _pin(pin['path']) == pin:
                        kind = _passed(json.loads(Path(pin['path']).read_text()))
                        if kind:
                            passed[name] = kind
                except (OSError, ValueError, KeyError, TypeError):
                    pass  # Missing/changed evidence schedules ordinary native grading.
        chosen, report = plan(records, policy=policy, attempts=attempts, passed=passed, hint=hint)
        chosen = [r for r in chosen if r[2] not in active]
        report.update(queued=[r[2] for r in chosen], next_batch=[r[2] for r in chosen[:16]],
                      next_batch_after_this=[r[2] for r in chosen[16:32]],
                      in_flight_pending=sorted(active & set(report['pending'])),
                      ownership_scope='execution exclusion only; never pass reuse')
        # Selection alone is never an attempt. Even a package-wide preflight
        # failure leaves every unsubmitted member's fairness counter unchanged.
        original = set(policy.get('correctness_curriculum', {}).get('original_capsules', self.paths))
        witnesses = set(policy.get('gsim', {}).get('witness_capsules', ()))
        preferred = set(hint)
        self.priorities = {r[2]: (attempts.get(r[2], 0), r[2] not in original,
                                  r[2] not in witnesses, r[2] not in preferred) for r in chosen}
        report['fairness_basis'] = 'actual scheduler admissions; queued and preflight-blocked members are not attempts'
        report['admitted_this_snapshot'] = sorted(self.admitted)
        report['unadmitted_pending'] = sorted(set(report['pending']) - self.admitted - active)
        report['admission_window_s'] = getattr(self, 'admission_window_s', None)
        report['snapshot_identity_sha256'] = _key(self.identity) if self.identity is not None else None
        questions = policy.get('operator_measurement_policy', {}).get('witness_questions', {})
        report['witness_prerequisites'] = [dict(capsule=n,
            correctness='matched_spike_pass' if passed.get(n) == 'spike_correctness' else 'pending_matching_spike_pass',
            question=questions.get(n, 'selected witness; unresolved question required'),
            physical_status='unknown_until_exact_ELF_input_config_scope_and_model_join', dispatch_authorized=False)
            for n in sorted(set(policy.get('gsim', {}).get('witness_capsules', ())) & self.paths.keys())]
        return chosen, report

    def report(self):
        # Reopen completed native receipts after the batch; no new execution.
        return self.select(self.records, policy=self.policy, hint=self.hint)[1]

    def started(self, capsule):
        """Called only by the original scheduler when it submits this member."""
        name = capsule['name']
        if name not in self.paths:
            raise ValueError('scheduler admitted a capsule outside its frozen curriculum')
        if name in self.admitted:
            return
        with self._ledger() as ledger:
            active = ledger.setdefault('in_flight', {})
            key = self.ownership_keys[name]
            if key in active:
                raise RuntimeError('capsule already owned by a live or undrained snapshot')
            active[key] = {'owner': self.owner, 'pid': os.getpid(), 'capsule': name}
            entry = ledger['members'].setdefault(name, {'attempts': 0, 'passes': {}})
            entry['admissions'] = entry.get('admissions', entry.get('attempts', 0)) + 1
        self.admitted.add(name)

    def priority(self, capsule):
        return self.priorities[capsule['name']]

    def release(self, name=None):
        """Release only this owner's completed jobs or its fully drained scope.

        No time expiry or PID guessing: a stale owner remains conservatively held
        until a trusted drain/recovery explicitly retires it.
        """
        with self._ledger() as ledger:
            active = ledger.setdefault('in_flight', {})
            for key, value in list(active.items()):
                if value['owner'] == self.owner and (name is None or value['capsule'] == name):
                    del active[key]

    def completed(self, callback):
        try:
            self._completed(callback)
        finally:
            if callback.get('capsule') in self.paths:
                self.release(callback['capsule'])

    def _completed(self, callback):
        name = callback.get('capsule')
        if name not in self.paths or callback.get('background_deferred') or callback.get('status') == 'gated':
            return
        matches = list((self.runs_root / 'runs').glob(f'*/{name}/capsule_result.json'))
        if len(matches) != 1:
            return
        pin = _pin(matches[0])
        row = json.loads(matches[0].read_text())
        if row.get('capsule', name) != name or row.get('background_deferred'):
            return
        with self._ledger() as ledger:
            entry = ledger['members'].setdefault(name, {'attempts': 0, 'passes': {}})
            entry['attempts'] += 1
            if (self.identity is not None and _passed(row)
                    and self.keys[name] == self._identity(name)):
                entry['passes'][self.keys[name]] = {'result': pin, 'recorded_ns': time.time_ns()}
                entry['passes'] = dict(sorted(entry['passes'].items(),
                    key=lambda item: item[1]['recorded_ns'])[-8:])


# This is an evaluator residency bound, not a new physical budget or a CPU
# reservation. Started cases drain under their existing finite per-case timeout.
SNAPSHOT_ADMISSION_S = 300


@contextmanager
def admission(progress, *, timeout, remaining_s, history):
    from merlin.targetgen.capsule_scheduling import measured_costs, rolling_admission

    window = min(SNAPSHOT_ADMISSION_S, max(0, remaining_s))
    deadline = time.monotonic() + max(0, remaining_s)
    costs, _ = measured_costs(history, requested_tiers={'L2'}, engine='spike')
    progress.admission_window_s = window
    try:
        with rolling_admission(on_start=progress.started,
                admission_window_s=window, admission_deadline_monotonic=deadline, admission_costs=costs,
                admission_priority=progress.priority, admission_allow_oversized_first=True):
            yield timeout
    finally:
        # Native grade returns only after its submitted workers have drained.
        progress.release()


def native(run_dir, *, inputs, compiler_sha256, target, runs_root, runtime_config):
    """Exact frozen context enables scheduling reuse; missing identity only disables reuse."""
    from .. import joint_control
    from merlin.targetgen.tier_cache import instrument_digest
    from merlin.runtime.backends.base import get_backend
    state = joint_control.selected()
    if state is None:
        return None
    roster_pin = json.loads(os.environ['MERLIN_JOINT_CORRECTNESS_ROSTER']) if os.environ.get('MERLIN_JOINT_CORRECTNESS_ROSTER') else None
    original = None
    if roster_pin is not None:
        if _pin(roster_pin['path']) != roster_pin:
            raise ValueError('original correctness roster pin changed')
        roster = json.loads(Path(roster_pin['path']).read_text())
        if roster.get('schema') != 'merlin.original_correctness_roster.v1':
            raise ValueError('original correctness roster schema differs')
        original = roster['original_public_capsules']
    identity = None
    try:
        profile = inputs.compiler_profile_service
        if profile is not None and inputs.verify_grade_inputs is not None:
            # The original frozen-input check is needed before an actual reuse
            # hit, not for every failed/cold batch. Native grading keeps its own
            # original custody checks. Existing read leases amortize this check.
            instrument = instrument_digest(target, 'L2', rtl_tier=False)
            backend = get_backend(target)
            # Reuse only when the existing runtime owner can identify its actual
            # selected binary/extension/config; target name alone is insufficient.
            engine_owner = getattr(backend, '_engine_identity', None)
            engine = engine_owner('spike') if callable(engine_owner) else None
            if engine is None:
                identify = getattr(backend, 'simulator_identity', None)
                engine = identify('spike') if callable(identify) else None
                if not engine or not engine.get('digest'):
                    engine = None
            if instrument and engine:
                identity = dict(compiler_sha256=compiler_sha256, descriptor=_pin(inputs.context.descriptor),
                    compiler_profile=_pin(profile.bundle_manifest), instrument_sha256=instrument,
                    native_engine=engine,
                    original_roster=roster_pin,
                    joint_policy=state['policy'], performance_selection=state.get('performance_selection'),
                    runtime_config={k: v.record() if hasattr(v, 'record') else v for k, v in runtime_config.items()},
                    additional_forbidden=list(inputs.additional_forbidden))
    except (OSError, ValueError, TypeError, AttributeError):
        identity = None
    return Progress(Path(run_dir) / '_qa_work' / 'joint_curriculum', identity=identity,
                    runs_root=runs_root, original_capsules=original,
                    verify_reuse=inputs.verify_grade_inputs,
                    ownership_identity={'compiler_sha256': compiler_sha256,
                        'descriptor': _pin(inputs.context.descriptor), 'target': target,
                        'runtime_config': {k: v.record() if hasattr(v, 'record') else v
                                           for k, v in runtime_config.items()},
                        'joint_policy': state['policy'],
                        'performance_selection': state.get('performance_selection')})
