"""One caller-owned bounded evaluator serves exact variant and shape domains."""
import hashlib
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from multiprocessing import get_context

import pytest
from test_evaluated_exploration import CONTEXT, identity, observe, run
from test_pass_exploration import emission, facts, families, source

from merlin.transforms.evaluated_exploration import (
    EvaluationRequest,
    evaluate_emitted,
)
from merlin.transforms.pass_exploration import explore_candidates, generate_candidates


def requests():
    space = generate_candidates(families(), output='program')
    return [EvaluationRequest(c.emission, c.evaluation.result.receipt, identity())
            for c in explore_candidates(source(), space, facts, output='program',
                                        emission=emission, context=CONTEXT)
            if c.emission is not None and c.duplicate_of is None]


def process_evaluate(program, receipt, domain):
    # A real external process receives serialized bytes/receipt/domain only.
    assert type(receipt) is dict and 'after' in receipt
    return observe(program, domain)


def test_parallel_completion_unblocks_emission_and_preserves_exact_cycle_ranking():
    release_slow, fast_started = threading.Event(), threading.Event()
    published, active, maximum = [], set(), []

    @contextmanager
    def hold(request):
        active.add(request.key)
        maximum.append(len(active))
        try:
            yield
        finally:
            active.remove(request.key)

    def evaluate(program, receipt, domain):
        if b'arith.addi' in program.data:
            assert release_slow.wait(5), 'inline evaluation prevented second unique emission'
        else:
            fast_started.set()
        return observe(program, domain)

    def publish(request, result):
        assert request.key in active
        published.append(result.record())
        if result.observation.cycles == 100:
            assert fast_started.is_set()
            release_slow.set()

    with ThreadPoolExecutor(max_workers=2) as owner:
        result = run(evaluate, executor=owner, max_in_flight=2,
                     hold_inputs=hold, on_completion=publish)
    assert [r['cycles'] for r in published] == [100, 200]
    assert len(result.candidates) == 4 and len(result.observations) == 2
    assert max(maximum) == 2 and not active
    assert result.winner.emission.data == requests()[1].program.data
    assert result.ranking['ranked'][0][1] == (100,)
    assert all(c.feedback is not None for c in result.candidates if c.duplicate_of is None)


def test_cross_shape_domains_share_one_executor_but_never_dedup_evidence():
    original, _ = requests()
    other = EvaluationRequest(original.program, original.receipt,
                              replace(original.identity, inputs_sha256='3'*64))
    calls, completions = [], []

    def evaluate(program, receipt, domain):
        calls.append(domain.digest())
        return observe(program, domain)

    with ThreadPoolExecutor(max_workers=2) as owner:
        completed = list(evaluate_emitted((original, original, other), evaluate,
            executor=owner, max_in_flight=2, on_completion=lambda r, c: completions.append(c.record())))
    assert len(calls) == len(completed) == len(completions) == 2
    assert len(set(calls)) == 2
    assert len({c.program_sha256 for c in completed}) == 1
    assert len({c.identity.digest() for c in completed}) == 2


def test_receipt_is_snapshotted_and_serial_inputs_close_before_yield():
    initial = requests()[0]
    original = initial.receipt
    request = EvaluationRequest(initial.program, original, initial.identity)
    original['after']['context']['hardware'] = 'mutated-caller-object'
    assert request.receipt['after']['context']['hardware'] == CONTEXT.hardware
    active = []

    @contextmanager
    def hold(req):
        active.append(req.key)
        try:
            yield
        finally:
            active.clear()

    def evaluate(program, receipt, domain):
        assert active
        receipt.clear()  # It is an isolated plain callback payload.
        return observe(program, domain)

    stream = evaluate_emitted((request,), evaluate, hold_inputs=hold,
        on_completion=lambda req, done: active or pytest.fail('input pin released before publication'))
    completed = next(stream)
    assert not active and completed.observation.eligible
    assert request.receipt['after']['context']['hardware'] == CONTEXT.hardware
    assert list(stream) == []


def test_existing_exact_incumbent_reuse_and_unknown_feedback_stay_unranked():
    first, second = requests()
    incumbent = observe(first.program, first.identity)
    calls = []

    def evaluate(program, receipt, domain):
        calls.append(program)
        return observe(program, domain, timing_status='unknown', cycles=None)

    with ThreadPoolExecutor(max_workers=2) as owner:
        result = run(evaluate, executor=owner, max_in_flight=2, incumbent=incumbent)
    assert len(calls) == 1 and result.winner is None
    assert result.ranking['selection'] == 'incumbent_retained'
    assert hashlib.sha256(second.program.data).hexdigest() in result.ranking['unranked_programs']
    assert sum(bool(c.feedback) and c.feedback['observation_origin'] == 'explicit_incumbent'
               for c in result.candidates) == 1


def test_in_flight_bound_limits_pins_even_with_a_larger_caller_executor():
    base = requests()[0]
    inputs = tuple(EvaluationRequest(base.program, base.receipt,
        replace(base.identity, inputs_sha256=f'{i:064x}')) for i in range(5))
    barrier, lock = threading.Barrier(2), threading.Lock()
    active, maximum, calls = set(), [], []

    @contextmanager
    def hold(request):
        active.add(request.key)
        maximum.append(len(active))
        assert len(active) <= 2
        try:
            yield
        finally:
            active.remove(request.key)

    def evaluate(program, receipt, domain):
        with lock:
            calls.append(domain)
            ordinal = len(calls)
        if ordinal <= 2:
            barrier.wait(5)  # Actual callbacks overlap; emission does not wait inline.
        return observe(program, domain)

    with ThreadPoolExecutor(max_workers=8) as owner:
        completed = list(evaluate_emitted(inputs, evaluate, executor=owner,
                                         max_in_flight=2, hold_inputs=hold))
    assert len(completed) == 5 and max(maximum) == 2 and not active


def test_process_executor_receives_no_mutable_ir_or_local_session():
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as owner:
        result = list(evaluate_emitted(requests(), process_evaluate,
                                      executor=owner, max_in_flight=2))
    assert sorted(c.observation.cycles for c in result) == [100, 200]


def test_parallel_transport_error_publishes_then_propagates_after_running_input_pins_close():
    first, second = requests()
    second_started, release = threading.Event(), threading.Event()
    active, published = set(), []

    @contextmanager
    def hold(request):
        active.add(request.key)
        try:
            yield
        finally:
            active.remove(request.key)

    def evaluate(program, receipt, domain):
        if program.data == first.program.data:
            assert second_started.wait(5)
            raise RuntimeError('external transport failed')
        second_started.set()
        assert release.wait(5)
        assert second.key in active
        return observe(program, domain)

    def publish(request, completed):
        assert request.key in active
        published.append(completed.record())
        release.set()

    with ThreadPoolExecutor(max_workers=2) as owner:
        with pytest.raises(RuntimeError, match='external transport failed'):
            list(evaluate_emitted((first, second), evaluate, executor=owner,
                max_in_flight=2, hold_inputs=hold, on_completion=publish))
    assert not active and published[0]['status'] == 'evaluation_error'
    assert published[0]['cycles'] is None


@pytest.mark.parametrize('options', [{'max_in_flight': 0}, {'max_in_flight': True},
                                     {'max_in_flight': 2}, {'max_requests': 0}])
def test_explicit_budget_required_before_evaluation(options):
    with pytest.raises(ValueError, match='bounded'):
        list(evaluate_emitted((), lambda *args: pytest.fail('bad budget launched evaluation'), **options))


def test_cross_domain_reply_still_refuses_in_parallel():
    with ThreadPoolExecutor(max_workers=2) as owner:
        with pytest.raises(ValueError, match='exact source/program/provider/ROI'):
            list(evaluate_emitted(requests(),
                lambda p, r, i: observe(p, replace(i, provider_sha256='4'*64)),
                executor=owner, max_in_flight=2))
