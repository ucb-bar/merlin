"""The ordinary session persists the selected soft budget and rejects resume drift."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import session as S
from merlin_experiments.phase1.options import parse_options


def request(tmp_path, *, joint=False, extra=()):
    args = ['--run-id', 'budget-test', *extra]
    if joint:
        path = tmp_path / 'campaign.json'
        path.write_text('{}')
        args += ['--joint-campaign', str(path)]
    options = parse_options(args)
    return S.RunRequest(context=SimpleNamespace(readback_policy=None), options=options,
        treatment=None, bundle_manifest=tmp_path/'bundle.yaml', launcher_argv=(),
        source_entrypoint=tmp_path/'entry.py', require_native_source=False,
        account={}, resolved_tools=lambda: ())


def test_session_records_joint_default_and_preserves_legacy_absence(tmp_path):
    assert 'author_token_budget' not in request(tmp_path).run_config
    selected = request(tmp_path, joint=True)
    recorded = selected.run_config
    assert recorded['author_token_budget']['target'] == 40_000_000
    assert recorded['author_token_budget']['reserve'] == 8_000_000
    S._verify_client_selection(selected.options, {'run_config': recorded})
    # Missing historical budget is not silently replaced on resume.
    old = dict(recorded)
    del old['author_token_budget']
    S._verify_client_selection(selected.options, {'run_config': old})


def test_session_rejects_explicit_changed_budget_on_resume(tmp_path):
    selected = request(tmp_path, extra=('--author-token-target', '20000000', '--author-token-reserve', '4000000'))
    recorded = selected.run_config
    S._verify_client_selection(selected.options, {'run_config': recorded})
    changed = replace(selected.options, author_token_target=30_000_000)
    with pytest.raises(ValueError, match='changed on resume'):
        S._verify_client_selection(changed, {'run_config': recorded})
    with pytest.raises(ValueError, match='changed on resume'):
        S._verify_client_selection(selected.options, {'run_config': {}})
