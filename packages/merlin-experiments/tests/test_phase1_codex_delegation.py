"""Explicit native delegation is opt-in and belongs to the recorded run identity."""
import dataclasses
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import session
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.providers import codex_agent, execution


@pytest.mark.parametrize('count', [None, 1])
def test_delegation_limit_reaches_actual_provider_call(tmp_path, monkeypatch, count):
    from merlin_experiments.phase1.context import InvocationContext

    from merlin.targetgen import target_experiment
    context = InvocationContext(tmp_path, tmp_path / 'descriptor', tmp_path, 'fixture',
                                tmp_path, tmp_path, tmp_path, ())
    (tmp_path / 'TASK.md').write_text('sealed')
    config = execution.ExecutionConfig(context, execution.ProviderConfig('codex', codex_max_subagents=count),
                                       lambda: (), tmp_path / 'timing')
    monkeypatch.setattr(execution.FL, 'start_brokers', lambda *a: [])
    monkeypatch.setattr(execution.FL, 'stop_brokers', lambda *a, **k: None)
    monkeypatch.setattr(target_experiment, 'load_target_experiment', lambda *a: object())
    from merlin_experiments.phase1.providers import agent_bridge
    monkeypatch.setattr(agent_bridge, 'bridged_name', lambda *a: None)
    received = []
    def run(*args, **kwargs):
        received.append(kwargs)
        return 0, tmp_path / 'transcript'
    monkeypatch.setattr(codex_agent, 'run_round', run)
    execution.launch(tmp_path, tmp_path, 'gpt-6.1-sol', 'high', 'bwrap', {}, 0, 10, config=config)
    if count is None:
        assert 'max_concurrent_subagents' not in received[0]
    else:
        assert received[0]['max_concurrent_subagents'] == count


def test_delegation_cli_metadata_and_resume_identity(tmp_path):
    options = parse_options(['--run-id', 'test', '--codex-max-subagents', '1'])
    request = SimpleNamespace(options=options, launcher_argv=(), context=SimpleNamespace(readback_policy=None))
    record = session.RunRequest.run_config.fget(request)
    assert record['codex_max_subagents'] == 1
    session._verify_client_selection(options, {'run_config': record})
    with pytest.raises(RuntimeError, match='delegation limit changed'):
        session._verify_client_selection(dataclasses.replace(options, codex_max_subagents=2), {'run_config': record})
    assert parse_options(['--run-id', 'test']).codex_max_subagents is None
    with pytest.raises(SystemExit):
        parse_options(['--run-id', 'test', '--codex-max-subagents', '0'])


def test_explicit_delegation_user_instruction_preserves_unset_prompt_and_recorded_cap():
    initial, continuation = "Original initial instruction", "Original continuation"
    for prompt in (initial, continuation):
        assert codex_agent.delegation_user_prompt(prompt, None) == prompt
        selected = codex_agent.delegation_user_prompt(prompt, 1)
        assert selected.startswith(prompt)
        assert "I explicitly request native sub-agent delegation" in selected
        assert "at most 1 helper" in selected
        assert "disjoint source files" in selected
        assert "before freezing a candidate or requesting a self-check" in selected
    with pytest.raises(ValueError, match="positive concurrency"):
        codex_agent.delegation_user_prompt(initial, 0)
