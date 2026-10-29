import json
from merlin_experiments.phase1.tools.await_verdict import _summary


def test_wait_summary_preserves_package_failure_and_cooperative_checkpoint(tmp_path):
    path = tmp_path / 'verdict.json'
    path.write_text(json.dumps({'n_capsules': 0, 'all_pass': False,
                               'package_failure': {'plane': 'integrity', 'detail': 'unexported input parser'},
                               'operator_epoch_notice': {'schema': 'merlin.operator_epoch_notice.v1',
                                                         'action': 'checkpoint_and_return',
                                                         'instruction': 'Save compiler and return.'}}))
    report = _summary(path)
    assert report['n_capsules'] == 0 and report['all_pass'] is False
    assert report['package_failure']['detail'] == 'unexported input parser'
    assert report['operator_epoch_notice']['instruction'] == 'Save compiler and return.'
