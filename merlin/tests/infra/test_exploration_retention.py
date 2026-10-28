"""Small completion results survive optional cleanup; live jobs never lose inputs."""

import json

import pytest

from merlin.common import storage_lifecycle
from merlin.transforms.exploration_retention import cleanup_retired_candidate, hold_candidate_artifacts


@pytest.fixture
def output(tmp_path, monkeypatch):
    from merlin.common import storage_cli

    root = tmp_path/'out'
    monkeypatch.setenv('MERLIN_OUT_ROOT', str(root))
    monkeypatch.delenv('MERLIN_BUNDLE_CAS', raising=False)
    monkeypatch.setenv('MERLIN_CITED_ROOTS', str(root))
    monkeypatch.setattr(storage_cli, 'contract', lambda: {'scan_roots': []})
    return root


def candidate(output, *, decision='rejected', status='rankable'):
    root = output/'artifacts/candidates/one'
    products = root/'build'
    products.mkdir(parents=True)
    (products/'rebuildable.o').write_bytes(b'x'*4096)
    record = dict(schema='merlin.emitted_evaluation_completion.v1', evaluation_identity='1'*64,
                  program_sha256='2'*64, feedback_sha256='3'*64, status=status,
                  numerical_status='pass', timing_status='complete' if status == 'rankable' else 'unknown',
                  complete_roi=status == 'rankable', cycles=123 if status == 'rankable' else None,
                  retention=dict(decision=decision, reason='explicit original evaluated decision',
                                 source_commit='4'*40, duplicate_of='2'*64 if decision == 'duplicate' else None))
    receipt = root/'result.json'
    receipt.write_text(json.dumps(record))
    return root, products, receipt


def test_one_job_lease_spans_callback_and_terminal_cleanup_preserves_existing_result(output):
    root, products, receipt = candidate(output)
    original = receipt.read_bytes()
    with hold_candidate_artifacts(root):
        blocked = cleanup_retired_candidate(products, receipt=receipt, apply=True)
        assert not blocked['applied'] and any('running' in reason for reason in blocked['blockers'])
        assert (products/'rebuildable.o').exists()
        # The ordinary on_completion callback still holds this same job lease.
        assert receipt.read_bytes() == original
    dry = cleanup_retired_candidate(products, receipt=receipt)
    assert dry['reclaimable_bytes'] == 4096 and products.exists() and not dry['applied']
    assert cleanup_retired_candidate(products, receipt=receipt, apply=True)['applied']
    assert not products.exists() and receipt.read_bytes() == original
    assert json.loads(receipt.read_text())['cycles'] == 123


def test_later_reader_pin_blocks_previously_reclaimable_products(output):
    root, products, receipt = candidate(output, decision='duplicate')
    with hold_candidate_artifacts(root):
        pass
    assert cleanup_retired_candidate(products, receipt=receipt)['reclaimable_bytes'] == 4096
    pin = storage_lifecycle.pin(products, reason='current incumbent rollback product')
    assert not cleanup_retired_candidate(products, receipt=receipt, apply=True)['applied']
    storage_lifecycle.unpin(products, pin)
    with hold_candidate_artifacts(products):
        assert not cleanup_retired_candidate(products, receipt=receipt, apply=True)['applied']
    assert cleanup_retired_candidate(products, receipt=receipt, apply=True)['applied']


@pytest.mark.parametrize('status', ['pending', 'unranked', 'evaluation_error'])
def test_unknown_pending_and_failed_timing_stay_deferred(output, status):
    root, products, receipt = candidate(output, status=status)
    with hold_candidate_artifacts(root):
        pass
    with pytest.raises(ValueError, match='pending/UNKNOWN'):
        cleanup_retired_candidate(products, receipt=receipt, apply=True)
    assert products.exists()


def test_outside_output_symlinks_and_whole_result_tree_refused(output, tmp_path):
    root, products, receipt = candidate(output)
    outside = tmp_path/'outside'
    outside.mkdir()
    with pytest.raises(ValueError, match='within current output'):
        cleanup_retired_candidate(outside, receipt=receipt, apply=True)
    link = root/'alias'
    link.symlink_to(products, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        cleanup_retired_candidate(link, receipt=receipt, apply=True)
    with pytest.raises(ValueError, match='outside rebuildable products'):
        cleanup_retired_candidate(root, receipt=receipt, apply=True)
    assert receipt.is_file() and outside.is_dir()


def test_historical_unclassified_and_winner_results_not_cleanup_candidates(output):
    root, products, receipt = candidate(output)
    assert not cleanup_retired_candidate(products, receipt=receipt, apply=True)['applied']
    record = json.loads(receipt.read_text())
    record['retention']['decision'] = 'winner'
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='terminal rejected/duplicate'):
        cleanup_retired_candidate(products, receipt=receipt, apply=True)
    assert products.exists()


def test_real_evaluate_emitted_hook_releases_only_after_completion_receipt(output):
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import asdict

    from merlin.transforms.evaluated_exploration import (
        CompletedObservation,
        EvaluationRequest,
        ExplorationIdentity,
        evaluate_emitted,
        exploration_context_identity,
    )
    from merlin.transforms.pass_exploration import EmittedProgram
    from merlin.transforms.pipeline_state import PipelineContext

    root, products, receipt = candidate(output)
    context = PipelineContext()
    identity = ExplorationIdentity(*('1'*64 for _ in range(9)), exploration_context_identity(context),
                                   'xdsl', 'modeled_roi')
    request = EvaluationRequest(EmittedProgram(b'module {}', 'xdsl', '2'*64),
                                {'after': {'context': asdict(context)}}, identity)

    def evaluate(program, pipeline_receipt, domain):
        return CompletedObservation(domain, hashlib.sha256(program.data).hexdigest(), '3'*64,
                                    'pass', 'complete', True, 150)

    def publish(original, completion):
        record = completion.record()
        record['retention'] = dict(decision='rejected', reason='explicit evaluated alternative rejected',
                                   source_commit='4'*40)
        receipt.write_text(json.dumps(record))
        assert original.key == request.key
        assert not cleanup_retired_candidate(products, receipt=receipt, apply=True)['applied']

    with ThreadPoolExecutor(max_workers=2) as executor:
        for completion in evaluate_emitted((request,), evaluate, executor=executor, max_in_flight=1,
                hold_inputs=lambda original: hold_candidate_artifacts(root), on_completion=publish):
            # Cleanup is submitted AFTER collection/yield, off the evaluation
            # critical path. The evaluator has now released the one job lease.
            assert completion.record()['cycles'] == 150
            assert cleanup_retired_candidate(products, receipt=receipt, apply=True)['applied']
    assert receipt.exists() and json.loads(receipt.read_text())['cycles'] == 150
