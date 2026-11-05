"""Actual initial and resumed provider prompts respect the selected catalog boundary."""

import json
from types import SimpleNamespace

import pytest
import yaml

from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import fresh10_profiles as P
from merlin_experiments.phase1.providers import codex_agent as CA
from merlin_experiments.phase1.providers import codex_context as C
from test_author_context import _fake_cli


@pytest.mark.parametrize('name', P.NAMES)
def test_recorded_catalog_advice_survives_resume_without_ambient_profile(name, tmp_path, monkeypatch):
    record = C.selection(SimpleNamespace(driver='codex', sandbox='bwrap', author_profile=name))
    expected = A.resolve(name).allows('catalog_advice')
    assert record['library_inventory_advice'] is expected
    (tmp_path / 'environment.yaml').write_text(yaml.safe_dump({'run_config': {C.KEY: record}}))
    monkeypatch.delenv(A.ENV, raising=False)
    resumed = C.launch_kwargs(tmp_path, 'bwrap')[C.KEY]
    assert resumed == record
    assert (C.LIBRARY_INVENTORY_NOTE in C.round_note(resumed, 8)) is expected
    assert 'task-*' in C.round_note(resumed, 8)
    assert '175,000 tokens' in C.round_note(resumed, 8)
    assert 'fork_turns "none"' in C.round_note(resumed, 8)


@pytest.mark.parametrize('name', P.NAMES)
def test_actual_provider_first_and_continuation_prompts_do_not_reintroduce_withheld_catalog(
        name, tmp_path, monkeypatch):
    ws, run = tmp_path / 'workspace', tmp_path / 'run'
    ws.mkdir(); run.mkdir()
    (ws / 'TASK.md').write_text('Common compiler authoring contract.\n')
    record = C.selection(SimpleNamespace(driver='codex', sandbox='bwrap', author_profile=name))
    monkeypatch.setenv(A.ENV, name)
    # A stand-in transport returns an ordinary completed turn twice; the real
    # provider performs both first and resume composition without any inference.
    cli = _fake_cli(tmp_path)
    count = tmp_path / 'turn-count'
    script = cli.read_text().replace(
        'sys.stdin.read()',
        "sys.stdin.read(); from pathlib import Path; "
        f"_count = Path({str(count)!r}); "
        "_count.write_text(str(int(_count.read_text()) + 1) if _count.exists() else '1')")
    cli.write_text(script)
    monkeypatch.setattr(CA, '_CONTINUE_MIN_S', 0)
    _status, transcript = CA.run_round(ws, run, 'gpt-6.1-sol', {}, None, 'none', 0, 60,
                 codex_binary=cli, author_context=record, continue_session=True,
                 continuation_stop=lambda: 'two diagnostic turns complete'
                 if count.exists() and int(count.read_text()) >= 2 else None)
    assert int(count.read_text()) == 2
    first = (run / 'rounds/round_00.prompt.txt').read_text()
    continuation = (run / 'rounds/round_00.prompt.cont01.txt').read_text()
    expected = A.resolve(name).allows('catalog_advice')
    assert (C.LIBRARY_INVENTORY_NOTE in first) is expected
    assert 'task-*' in first and 'task-*' in continuation
    if not expected:
        for prompt in (first, continuation):
            assert 'library_inventory.py' not in prompt
            assert 'composition_routes.py' not in prompt
            assert 'optimization_next.library_pass' not in prompt
    events = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert len([event for event in events if event.get('type') == 'codex_session_turn']) == 2


def test_historical_context_retains_its_original_catalog_advice():
    record = C.selection(SimpleNamespace(driver='codex', sandbox='bwrap'))
    assert 'library_inventory_advice' not in record
    assert C.LIBRARY_INVENTORY_NOTE in C.round_note(record, 8)


def test_malformed_recorded_catalog_selection_is_rejected(tmp_path):
    record = C.selection(SimpleNamespace(driver='codex', sandbox='bwrap', author_profile=P.NAMES[0]))
    record['library_inventory_advice'] = 'false'
    (tmp_path / 'environment.yaml').write_text(yaml.safe_dump({'run_config': {C.KEY: record}}))
    with pytest.raises(ValueError, match='library inventory advice'):
        C.launch_kwargs(tmp_path, 'bwrap')
