"""Compact display never replaces the full owned model-check artifact."""
import hashlib
import json

from merlin_experiments.phase1.tools import model_check as M


def fixture():
    error = "cannot import name 'selected_pass' from compiler module"
    return {'status': 'fail', 'model': 'example', 'action': 'check', 'source_sha256': 'a'*64,
            'returncode': 1, 'report': {
                'schema': 'merlin.model_check.report.v1', 'command': 'check', 'split': 'visible',
                'gates': {'build': {'status': 'fail', 'detail': '0/100 kernels built'}},
                'programs': [{'program': 'decode', 'build': {'built': 0, 'refused': 100},
                              'refused': {f'k{i}': error for i in range(100)},
                              'kernels': [{'unnecessary_detail': list(range(1000))}],
                              'census': {'denominator': 100, 'device': 0, 'refused': 100}, 'images': []}]}}


def test_summary_retains_actionable_failure_counts_and_not_kernel_payload():
    result = fixture()
    value = M.compact_result(result, artifact={'path': 'full.json', 'sha256': 'b'*64})
    assert value['source_sha256'] == 'a'*64
    assert value['failure_count'] == 100 and value['distinct_failure_count'] == 1
    assert 'selected_pass' in value['first_distinct_failure']
    assert value['programs'][0]['census']['refused'] == 100
    assert 'unnecessary_detail' not in json.dumps(value)
    assert len(json.dumps(value)) < 2500
    assert len(result['report']['programs'][0]['refused']) == 100


def test_cli_keeps_full_artifact_and_explicit_full_stdout(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    channel = tmp_path / '.model_check_channel'; channel.mkdir()
    result = fixture()
    (channel / 'resp_test.json').write_text(json.dumps(result))
    assert M.main(['wait', '--job-id', 'test', '--out', 'detail/full.json', '--wait-budget', '0']) == 1
    displayed = json.loads(capsys.readouterr().out)
    saved = tmp_path / 'detail/full.json'
    assert json.loads(saved.read_text()) == result
    assert displayed['full_artifact']['sha256'] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert M.main(['wait', '--job-id', 'test', '--full-json', '--wait-budget', '0']) == 1
    assert json.loads(capsys.readouterr().out) == result


def test_pass_and_unavailable_stay_separate_without_inferred_wholechain():
    result = {'status': 'pass', 'report': {'gates': {'kernel_exact': {'status': 'pass', 'partial': False}},
              'programs': [{'program': 'prefill', 'images': [{'cases': 22, 'engine': 'spike'}]}]}}
    value = M.compact_result(result, artifact={})
    assert value['status'] == 'pass' and value['first_distinct_failure'] is None
    assert value['programs'][0]['replay_cases'] == 22
    assert value['scope']['mode'] == 'unspecified'
    error = M.compact_result({'status': 'unavailable', 'error': 'selected runtime missing'}, artifact={})
    assert error['status'] == 'unavailable'
    assert error['first_distinct_failure'] == 'selected runtime missing'
