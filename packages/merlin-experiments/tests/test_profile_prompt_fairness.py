"""Versioned EL1/2/3 change available abstractions, not tasks or workflow mandates."""
import json
import subprocess
import sys

import pytest
from merlin_experiments.phase1 import authoring_conformance as workflow
from merlin_experiments.phase1 import conformance, full_compiler_workspace, joint_control
from merlin_experiments.phase1.feedback import snapshot_execution
from merlin_experiments.phase1.providers import authoring_prompt as prompts
from merlin_experiments.phase1.providers import codex_agent as driver

PROFILES = ('xs-el-v1/el1', 'xs-el-v1/el2', 'xs-el-v1/el3')


def test_same_neutral_task_before_both_legacy_prompt_selectors(monkeypatch):
    def unavailable():
        raise AssertionError('restricted profile must not select legacy modern guidance')
    monkeypatch.setattr(full_compiler_workspace, 'selected', unavailable)
    monkeypatch.setattr(joint_control, 'selected', unavailable)
    first, continued = [], []
    for profile in PROFILES:
        monkeypatch.setenv('MERLIN_AUTHOR_PROFILE', profile)
        first.append(prompts.authoring_prompt())
        continued.append(prompts.profile_authoring_prompt(continuation=True))
    assert len(set(first)) == len(set(continued)) == 1
    for text in (*first, *continued):
        for stale in ('250', '153', '77', '16 helpers', '20 CPUs', 'heavy-pass-recipes',
                      'scheduling-api', 'shape_policy', 'parallel_schedule_recipe', 'supplied pass library'):
            assert stale not in text
        assert 'AUTHOR_PROFILE.json' in text and 'TASK.md' in text
        assert 'unavailable optional tooling is not a required discovery step' in text


def test_historical_and_modern_profiles_retain_original_prompt_route(monkeypatch):
    monkeypatch.delenv('MERLIN_AUTHOR_PROFILE', raising=False)
    monkeypatch.setattr(full_compiler_workspace, 'selected', lambda: object())
    historical = prompts.authoring_prompt()
    assert 'heavy-pass-recipes' in historical
    assert prompts.profile_authoring_prompt() is None
    monkeypatch.setenv('MERLIN_AUTHOR_PROFILE', 'xs-el-v1/el4')
    assert prompts.authoring_prompt() == historical
    assert prompts.profile_authoring_prompt(continuation=True) is None


@pytest.mark.parametrize('profile', PROFILES)
def test_real_provider_writes_neutral_first_and_resume_prompts(profile, tmp_path, monkeypatch):
    monkeypatch.setenv('MERLIN_AUTHOR_PROFILE', profile)
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'TASK.md').write_text('sealed neutral test task')
    fake = tmp_path / 'offline-author'
    fake.write_text('#!' + sys.executable + '\n'
                    'import json,sys\n'
                    'sys.stdin.read()\n'
                    'for e in [{"type":"thread.started","thread_id":"offline"},'
                    '{"type":"turn.started"},{"type":"turn.completed",'
                    '"usage":{"input_tokens":1,"output_tokens":1}}]:\n'
                    ' print(json.dumps(e),flush=True)\n')
    fake.chmod(0o700)
    monkeypatch.setattr(driver, 'cli_version', lambda *a: 'offline')
    monkeypatch.setattr(driver, '_CONTINUE_MIN_S', 0)
    monkeypatch.setattr(driver, '_CONTINUE_MAX_TURNS', 2)
    original = subprocess.Popen

    def local_only(argv, **kwargs):
        if '--pid' in argv:
            raise OSError('offline sampler disabled')
        assert str(fake) in argv
        return original(argv, **kwargs)

    monkeypatch.setattr(driver.subprocess, 'Popen', local_only)
    rc, transcript = driver.run_round(workspace, tmp_path / 'run', 'offline', {}, None, 'none', 0, 600,
                                      effective_model='offline', continue_session=True, codex_binary=fake)
    assert rc == 0
    first = (transcript.parent / 'round_00.prompt.txt').read_text()
    continued = (transcript.parent / 'round_00.prompt.cont01.txt').read_text()
    assert first == prompts.authoring_prompt()
    assert continued == prompts.profile_authoring_prompt(continuation=True)
    summary = json.loads((transcript.parent / 'round_00.codex_summary.json').read_text())
    assert summary['turns_started'] == 2


@pytest.mark.parametrize('profile', PROFILES)
def test_workflow_policy_is_equal_without_mandatory_selfcheck_or_tool_use(profile, tmp_path, monkeypatch):
    monkeypatch.setenv('MERLIN_AUTHOR_PROFILE', profile)
    access = 'selected_library' if profile.endswith('/el1') else 'full_compiler'
    monkeypatch.setattr(full_compiler_workspace, 'selected', lambda: None if access == 'selected_library' else object())
    monkeypatch.setattr(snapshot_execution, 'selected', lambda: True)
    transcript = tmp_path / 'empty.jsonl'
    transcript.write_text('')
    submission = tmp_path / 'submission'
    submission.mkdir()
    conf, conformant = workflow.workflow_conformance(transcript, submission, 'merlin_assisted', 'external_backend', ())
    assert conformant is True and conf['checks'] == {}
    assert conf['workflow_policy'] == 'author_profile_guidance_v1'
    assert conf['author_profile'] == profile
    assert conf['compiler_access'] == access
    assert conf['compiler_execution'] == 'snapshot_external_v1'
    assert conf['qualification_authority'] is False
    from merlin_experiments.phase1.feedback.certification import _authoring_completion
    assert _authoring_completion(False, conformant) is False


def test_unprofiled_selected_library_keeps_legacy_full_selfcheck_gate(tmp_path, monkeypatch):
    monkeypatch.delenv('MERLIN_AUTHOR_PROFILE', raising=False)
    conf = conformance.compute(tmp_path / 'absent.jsonl', tmp_path, 'merlin_assisted', '', resolved_tools=(),
                               compiler_access='selected_library', compiler_execution='snapshot_external_v1')
    assert conf['conformant'] is False and conf['checks']['full_selfcheck'] is False


@pytest.mark.parametrize('profile', ['invalid/el1', *PROFILES])
def test_invalid_profile_or_nonisolated_profile_workflow_fails_closed(profile, tmp_path):
    with pytest.raises(ValueError):
        conformance.compute(tmp_path / 'absent.jsonl', tmp_path, 'merlin_assisted', '',
                            resolved_tools=(), author_profile=profile)
