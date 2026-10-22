import hashlib
import json

import pytest

from merlin_experiments.phase1 import joint_handoff as H, oot_history as O


def test_operator_handoff_keeps_source_and_prior_evidence_without_grade(tmp_path):
    workspace = tmp_path / 'workspace'
    source = workspace / 'submission'
    source.mkdir(parents=True)
    (source / 'manifest.yaml').write_text('name: independently-authored\n')
    (source / 'compiler.py').write_text('# preserved generic compiler\n')
    (workspace / 'qa').mkdir()
    prior = {'n_required_development': 167, 'n_passed': 3, 'n_capsules': 10,
             'graded_source_sha256': 'earlier-source', 'all_pass': False}
    (workspace / 'qa/verdict.json').write_text(json.dumps(prior))
    run = tmp_path / 'run'
    run.mkdir()
    O.start(run, sandbox_roots=(workspace,))
    with pytest.raises(ValueError, match='explicit operator'):
        H.checkpoint(workspace, run, 0)
    (run / 'joint_epoch_stop.json').write_text(json.dumps({
        'schema': 'merlin.joint_epoch_stop.v1', 'reason': 'reviewed library repair'}))
    result = H.checkpoint(workspace, run, 0)
    assert result['all_pass'] is None and result['n_capsules'] == 0
    assert result['n_pending_development'] == 167
    assert result['n_passed'] == 0 and result['current_source_eligible'] is False
    artifact = run / 'qa_history/epoch_handoff_00_prior.json'
    assert json.loads(artifact.read_text()) == prior
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == result['prior_evidence']['sha256']
    assert O.verify_turn_end(run, source, result['oot_commit'])['matched']
    (source / 'compiler.py').write_text('# changed after custody\n')
    with pytest.raises(H.oot_repo.OotRepoError, match='changed outside'):
        O.verify_turn_end(run, source, result['oot_commit'])
