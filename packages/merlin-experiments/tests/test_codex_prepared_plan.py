"""The adjacent permission probe and paid command share a single mount plan."""
import shlex
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1.providers import execution as E, codex_agent as C
from merlin.targetgen.sandbox import bwrap


def test_plan_retains_masks_and_quotes_each_distinct_payload(tmp_path, monkeypatch):
    composed = []
    def compose(parts, suffix, workspace):
        composed.append((tuple(parts), shlex.split(suffix), workspace))
        return 'true'
    monkeypatch.setattr(bwrap, 'compose_command', compose)
    plan = E.SandboxPlan(('--ro-bind', '/public', '/public', '--tmpfs', '/private'),
                         'export SAFE=1; ', tmp_path)
    plan.command("printf '%s' 'x; $(false)'")
    plan.command('exec client --json')
    assert composed[0][0] == composed[1][0] == plan.parts
    assert composed[0][1] == ['bash', '-c', "export SAFE=1; printf '%s' 'x; $(false)'"]
    assert composed[1][1][-1] == 'export SAFE=1; exec client --json'
    with pytest.raises(Exception):
        plan.parts = ()


def test_prepared_probe_does_not_reassemble_or_select_runtime(tmp_path, monkeypatch):
    builds, composed = [], []
    def prepare(inner, ws, bundle, binds, **kw):
        builds.append((ws, bundle.copy(), tuple(binds)))
        return E.SandboxPlan(tuple(binds), 'env ', ws)
    monkeypatch.setattr(E, 'sandbox_plan', prepare)
    monkeypatch.setattr(bwrap, 'compose_command', lambda parts, suffix, ws: composed.append(tuple(parts)) or 'true')
    builder = E.CodexSandboxCommand(SimpleNamespace(), tmp_path)
    plan = builder.prepare(tmp_path, {'selection':'original'}, ['--tmpfs','/private'])
    monkeypatch.setattr(C.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stderr=''))
    def never(*a, **kw):
        pytest.fail('duplicate mount/runtime assembly')
    C._preflight_candidate_sandbox(tmp_path, tmp_path/'home', 'client', {}, never,
                                  tmp_path/'rounds', 0, runtime_binds=never, prepared_plan=plan)
    plan.command('client exec')
    assert len(builds) == 1 and composed == [plan.parts, plan.parts]
    # A later launch with changed grants prepares a new plan; there is no cache.
    builder('client resume', tmp_path, {'selection':'new'}, ['--tmpfs','/other'])
    assert len(builds) == 2 and composed[-1] == ('--tmpfs','/other')


def test_failed_permission_probe_still_refuses_paid_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(bwrap, 'compose_command', lambda *a: 'false')
    plan = E.SandboxPlan((), '', tmp_path)
    monkeypatch.setattr(C.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1, stderr='denied'))
    with pytest.raises(RuntimeError, match='refusing untrusted launch'):
        C._preflight_candidate_sandbox(tmp_path, tmp_path/'home', 'client', {}, None,
                                      tmp_path/'rounds', 0, prepared_plan=plan)


def test_private_launch_script_replacement_preserves_sealed_predecessor(tmp_path):
    import os
    import subprocess
    first = C._sandbox_script(tmp_path, 0, 0, 'printf first')
    path = tmp_path/'round_00.sandbox.turn00.sh'
    with path.open() as retained:
        assert path.stat().st_mode & 0o777 == 0o500
        second = C._sandbox_script(tmp_path, 0, 0, 'printf second')
        assert first == second
        assert retained.read() == 'printf first'
    assert subprocess.check_output(second, text=True) == 'second'
    assert path.stat().st_mode & 0o777 == 0o500
