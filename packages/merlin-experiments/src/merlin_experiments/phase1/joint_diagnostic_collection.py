"""Operator-only sparse model diagnostics, isolated from fit and promotion labels.

The explicitly pinned request narrows the existing campaign witness roster and
budget. An independent owner may poll completed native products while the author
continues; no candidate command, physical engine or runtime is taken from it.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from merlin.runtime import _native_files as F
from . import joint_calibration as C, joint_control
from .feedback import joint_performance as PERF
from .feedback.joint_timing_selection import choose
from ..phase2.contracts import document_sha256

SCHEMA = 'merlin.joint_diagnostic_collection_request.v1'
UNKNOWN_COST = 'unresolved_cost'
VALIDATION = 'structural_validation'


def purpose(value):
    selected = value.get('purpose', UNKNOWN_COST)
    if selected not in {UNKNOWN_COST, VALIDATION}:
        raise ValueError('unknown diagnostic observation purpose')
    return selected


def request(pin, state):
    value = PERF._read(pin)
    policy = state['policy']
    selected_purpose = purpose(value)
    questions = value.get('questions')
    if (value.get('schema') != SCHEMA or value.get('policy_sha256') != state['policy_sha256']
            or value.get('profile') != policy.get('native_calibration')
            or value.get('fit_eligible') is not False or not value.get('review_id')
            or type(value.get('max_observations')) is not int or not 1 <= value['max_observations'] <= 4
            or not isinstance(questions, dict) or not 1 <= len(questions) <= 4
            or any(name not in policy['gsim']['witness_capsules'] or not isinstance(question, str)
                   or not question.strip() or len(question) > 2048 for name, question in questions.items())):
        raise ValueError('diagnostic request lacks reviewed questions within original physical policy')
    if selected_purpose == VALIDATION:
        # An explicitly reviewed comparison is about this exact evaluator. A
        # later model epoch needs a new review, not an implicit watch rebase.
        profile = PERF._read(policy['native_calibration'])
        current = state.get('performance_selection') or profile['performance_selection']
        if value.get('performance_selection') != current:
            raise ValueError('validation request must pin the current performance selection')
        PERF._read(current)
    return value


def eligible(entry, *, profile, selection, questions, observation_purpose=UNKNOWN_COST):
    """Cheap original-grade screen; the execution path reopens full ELF custody."""
    capsule = C._capsule(entry['capsule'])
    name = capsule.get('name')
    if (name not in questions or name not in profile['members'] or capsule.get('kind') == 'model'
            or capsule.get('label') not in {'public', 'dev'}):
        return None
    source = PERF.selected_capsule(selection, capsule)
    member = selection.get('members', {}).get(name)
    if (not member or PERF.capsule_identity(source) != member['capsule_sha256']
            or entry.get('source_group_sha256') != profile['members'][name]['source_group_sha256']
            or (observation_purpose == UNKNOWN_COST and entry.get('estimate_status') != 'UNKNOWN')
            or (observation_purpose == VALIDATION and entry.get('estimate_status') not in
                {'UNKNOWN', 'PROVISIONAL', 'QUALIFIED', 'REFUTED'})
            or observation_purpose not in {UNKNOWN_COST, VALIDATION}):
        return None
    result = PERF._read(entry['result'])
    tier = result.get('tiers', {}).get('L2', {})
    if (result.get('numeric', {}).get('status') != 'pass' or tier.get('status') != 'pass'
            or tier.get('engine') != 'spike' or result.get('failure')
            or result.get('trace_check', {}).get('status') != 'pass'
            or not (result.get('compiler_source') or result.get('candidate_native_execution'))):
        return None
    return name, source


def step(*, request_pin, state_path, context, output):
    """At most one newly reviewed exact-ELF diagnostic; failed attempts never retry."""
    state_path = Path(state_path).resolve()
    if Path(os.environ.get(joint_control.ENV, '')).resolve() != state_path:
        raise ValueError('diagnostic owner does not select the supplied campaign state')
    output = Path(output).resolve()
    # The collector never writes round-*/physical or any fitting registry.
    if any(part in {'calibration_registry', 'calibration_candidate_index'} or part.startswith('round-')
           for part in output.parts):
        raise ValueError('diagnostics require a separate operator output tree')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'owner.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(status='another_diagnostic_owner_active', fit_eligible=False)
        return _step(request_pin, state_path, context, output)


def _step(request_pin, state_path, context, output):
    state = joint_control.selected()
    if state is None:
        raise ValueError('diagnostic owner requires the existing campaign state')
    selected_request = request(request_pin, state)
    identity = output / 'request.json'
    if identity.exists():
        if json.loads(identity.read_text()) != request_pin:
            raise ValueError('diagnostic owner cannot change its pinned request')
    else:
        C._write(identity, request_pin)
    if joint_control.stop_reason():
        return dict(status='campaign_stopped', fit_eligible=False)
    if state.get('started_at_epoch') is None:
        return dict(status='waiting_for_author_clock', fit_eligible=False)
    profile = PERF._read(selected_request['profile'])
    current_pin = state.get('performance_selection') or profile['performance_selection']
    current = PERF._read(current_pin)
    from .joint_runtime_revision import validate
    validate(PERF._read(profile['performance_selection']), current)
    attempts = list(output.glob('attempt-*/attempt.json'))
    spent = list(output.glob('attempt-*/physical-intent.json'))
    if len(spent) >= selected_request['max_observations']:
        return dict(status='reviewed_diagnostic_limit_reached', fit_eligible=False)
    if len(attempts) >= 64:
        return dict(status='diagnostic_attempt_review_required', fit_eligible=False)
    registry = state_path.parent / 'calibration_registry'
    paths = sorted(registry.glob('*/entry.json'))[:2049]
    if len(paths) > 2048:
        raise ValueError('diagnostic registry exceeds bounded review roster')
    for path in paths:
        entry_pin = F.pin(path, max_bytes=16 << 20)
        entry = PERF._read(entry_pin)
        match = eligible(entry, profile=profile, selection=current, questions=selected_request['questions'],
                         observation_purpose=purpose(selected_request))
        if match is None:
            continue
        name, capsule = match
        epoch = choose(current, PERF.capsule_identity(capsule))
        elf = PERF.original_executable(entry['generated_root'])
        elf_pin = F.pin(elf, max_bytes=64 << 20)
        attempt_identity = [elf_pin['sha256'], epoch['runtime'], selected_request['questions'][name]]
        if purpose(selected_request) == VALIDATION:
            attempt_identity.append(VALIDATION)
        key = document_sha256(attempt_identity)
        directory = output / ('attempt-' + key)
        if directory.exists():
            continue
        directory.mkdir()
        C._write(directory / 'attempt.json', dict(request=request_pin, entry=entry_pin,
            performance_selection=current_pin, runtime=epoch['runtime'], elf=elf_pin,
            capsule_sha256=PERF.capsule_identity(capsule), question=selected_request['questions'][name],
            purpose=purpose(selected_request), fit_eligible=False, promotion_authority=False))
        try:
            runtime = PERF._runtime(epoch['runtime'], facts_pin=current['facts'])
            observed_elf, observation, report = C._observe(entry, runtime=runtime, context=context,
                directory=directory / 'observation', require_complete=False)
            if observed_elf != elf or F.pin(observed_elf, max_bytes=64 << 20) != elf_pin:
                raise ValueError('diagnostic changed original ELF')
            if (purpose(selected_request) == UNKNOWN_COST
                    and report.get('timing_complete') is True and report.get('unresolved') == []):
                value = dict(status='model_now_complete_no_diagnostic', fit_eligible=False)
            else:
                value = _collect(entry, selected_request, name, current_pin, runtime, observation,
                                 report, elf, profile, context, directory, state)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            value = dict(status='diagnostic_refused', reason=str(error), fit_eligible=False,
                         retry_authorized=False)
        pin = C._write(directory / 'outcome.json', value)
        return dict(value, outcome=pin)
    return dict(status=('waiting_for_matching_validation_product'
                        if purpose(selected_request) == VALIDATION
                        else 'waiting_for_matching_unknown_model_product'), fit_eligible=False)


def _collect(entry, selected_request, name, current_pin, runtime, observation,
             report, elf, profile, context, directory, original_state):
    from .joint_calibration_launcher import physical_runtime

    current_state = joint_control.selected()
    PERF._read(current_pin)  # Reopen the exact selected pin before any admission.
    if current_state != original_state or joint_control.stop_reason():
        raise ValueError('diagnostic campaign selection changed before physical admission')
    runtime.verify()
    model = json.loads(runtime.model_json)
    with physical_runtime(profile, context=context, directory=directory) as backend:
        # Intent precedes permission/execution: interruption consumes this local
        # review slot conservatively; the original ledger remains cost authority.
        C._write(directory / 'physical-intent.json', dict(question=selected_request['questions'][name],
            ledger=str(joint_control.gsim_ledger()), fit_eligible=False))
        C._authorize_if_needed(name, dict(question=selected_request['questions'][name], stage='refinement'))
        measurement = C._measure(entry, backend=backend, elf=elf, observation=observation,
                                 model=model, output=directory / 'diagnostic-physical', diagnostic=True)
    runtime.verify()
    if joint_control.selected() != original_state:
        raise ValueError('diagnostic campaign selection changed during observation')
    return dict(status='diagnostic_observed', purpose=purpose(selected_request), measurement=measurement,
        observation_report=observation['report'], unresolved=report.get('unresolved'),
        model_timing_complete=report.get('timing_complete'), fit_eligible=False,
        promotion_authority=False, formal_certificate=False,
        label_role='structural diagnostic only; separate reviewed scope admission required for any later fit')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'request-sha256', 'state', 'descriptor', 'repo', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args(argv)
    if not 5 <= args.poll_seconds <= 60:
        parser.error('poll interval must be 5..60 seconds')
    os.environ[joint_control.ENV] = str(Path(args.state).resolve())
    from .context import load_context
    context = load_context(args.descriptor, repo=args.repo)
    pin = dict(path=str(Path(args.request).resolve()), sha256=args.request_sha256)
    output = Path(args.output)
    while True:
        try:
            value = step(request_pin=pin, state_path=args.state, context=context, output=output)
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
            value = dict(status='blocked', reason=str(error), fit_eligible=False)
        output.mkdir(parents=True, exist_ok=True)
        pending = output / 'status.next.json'
        pending.write_text(json.dumps(value, indent=2) + '\n')
        pending.replace(output / 'status.json')
        if args.once or value['status'] in {'campaign_stopped', 'reviewed_diagnostic_limit_reached',
                                           'diagnostic_attempt_review_required', 'blocked'}:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
