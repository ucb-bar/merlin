"""Successful source custody, semantic SDK usage and strict first-mutation ordering."""

from __future__ import annotations

import json
import shlex
from dataclasses import replace

import pytest
from merlin_experiments.phase1 import conformance as C
from merlin_experiments.phase1 import conformance_sources as S

SCRIPT = """from pathlib import Path
from merlin.targetgen.rtl import facts as F
from rtl_backend import target_profile as profile, derived_levers as levers
from generate.xdsl import generate as scaffold
f = F.load_facts('fixture')
p = profile('fixture')
l = levers(p)
plan = {'target': 'fixture', 'dialect_name': 'fixture', 'ops': []}
a = scaffold(plan)
print('startup complete')
"""


def context(tmp_path):
    facts = tmp_path / 'facts.json'
    facts.write_text('{}')
    return S.execution_context(tmp_path, 'fixture', facts, [facts])


def call(command, i=0, *, success=True, end=None, ctx=None):
    return C.ToolCall('bash', {'command': command}, str(i), True, success, 'ok',
                      i * 2, i * 2 + 1 if end is None else end, ctx)


def recover(calls):
    return S.attribute(calls, unwrap=C._unwrap_shell, tokenize=C._shell_tokens)


def write(path, source=SCRIPT):
    return f"cat > {shlex.quote(str(path))} <<'PY'\n{source}PY"


def transcript(path, commands, ctx=None):
    events = [{'type': 'system', 'subtype': 'init', 'driver': 'codex', 'conformance_context': ctx}]
    for i, command in enumerate(commands):
        events += [{'type':'assistant','message':{'content':[{'type':'tool_use','id':str(i),'name':'bash',
                    'input':{'command':command}}]}},
                   {'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':str(i),
                    'is_error':False,'content':'ok'}]}}]
    path.write_text(''.join(json.dumps(event)+'\n' for event in events))
    return path


def test_successful_saved_script_has_all_discovery_before_actual_first_edit(tmp_path):
    ctx = context(tmp_path)
    script = tmp_path/'startup.py'
    t = transcript(tmp_path/'t.jsonl', [
        'python cca_contract.py check-bijection fixture',
        'python action_catalog.py escalation-ladder fixture',
        write(script), f'python3 {script}', 'echo implementation > submission/backend.py',
    ], ctx)
    sub = tmp_path/'submission'
    sub.mkdir()
    verdict = C.compute(t, sub, 'merlin_assisted', '', resolved_tools={'rtl_facts','rtl_generators'})
    for key in ('rtl_facts_used','rtl_derived_levers_used','scaffold_generators_used',
                'arm4_discovery_before_submission_mutation'):
        assert verdict['checks'][key] is True
    calls = C._tool_calls(t)
    assert calls[2].discovery_calls == frozenset()  # Writing never earns execution credit.
    assert calls[3].discovery_calls == {'facts','derived','scaffold'}


@pytest.mark.parametrize('prefix', ['if False:\n    ', 'def unused():\n    '])
def test_dormant_source_does_not_prove_execution(prefix):
    source = 'from merlin.targetgen.rtl.facts import load_facts\n'+prefix+"load_facts('fixture')\n"
    assert 'facts' not in S.discovery(source)


@pytest.mark.parametrize('middle', [
    "from evil import load_facts; load_facts('fixture')",
    "def load_facts(*a): pass\nload_facts('fixture')",
    "lf = lambda *a: None\nlf('fixture')",
    "setattr(F, 'load_facts', lambda *a: None)\nF.load_facts('fixture')",
    "class Patch:\n    F.load_facts=lambda *a:None\nF.load_facts('fixture')",
    "def fake(x=setattr(F,'load_facts',lambda *a:None)):pass\nF.load_facts('fixture')",
    "if bool(1):\n    F.load_facts=lambda *a:None\n"
    "from merlin.targetgen.rtl import facts as F\nF.load_facts('fixture')",
    "x={setattr(F,'load_facts',lambda *a:None):None}\nF.load_facts('fixture')",
    "True and setattr(F,'load_facts',lambda *a:None)\nF.load_facts('fixture')",
    "[setattr(F,'load_facts',lambda *a:None) for x in [1]]\nF.load_facts('fixture')",
    "x=setattr(F,'load_facts',lambda *a:None) if bool(1) else None\nF.load_facts('fixture')",
    "exec('anything')\nfrom merlin.targetgen.rtl import facts as F\nF.load_facts('fixture')",
])
def test_opaque_or_rebound_apis_cannot_mint_evidence(middle):
    assert 'facts' not in S.discovery('from merlin.targetgen.rtl import facts as F\n'+middle)


def test_binding_aliases_and_summary_stdout_do_not_change_semantics(tmp_path):
    ctx = context(tmp_path)
    assert S.discovery(SCRIPT, ctx) == {'facts','derived','scaffold'}
    named = SCRIPT.replace("load_facts('fixture')", "load_facts(target='fixture', explicit=None)").replace(
        "profile('fixture')", "profile(target='fixture')").replace("levers(p)", "levers(profile=p)").replace(
        "scaffold(plan)", "scaffold(dialect_plan=plan)")
    assert S.discovery(named, ctx) == {'facts','derived','scaffold'}
    assert 'facts' not in S.discovery(SCRIPT.replace("load_facts('fixture')", "load_facts('other')"), ctx)
    assert 'facts' not in S.discovery(SCRIPT.replace("load_facts('fixture')",
              "load_facts('fixture', explicit=Path('/unselected/facts.json'))"), ctx)
    assert 'facts' not in S.discovery(SCRIPT, {**ctx,'selected_facts':None})
    assert not S.discovery(SCRIPT, {'invalid':True})
    assert 'facts' not in S.discovery(SCRIPT.replace("load_facts('fixture')",
              "load_facts('fixture', **{'explicit':'/wrong/facts.json'})"), ctx)


@pytest.mark.parametrize('intervening', [
    'touch /untracked', 'find . -delete', 'find . -exec false ;', 'rg --pre unknown anything',
    'echo `unknown`', 'echo "$(unknown)"', 'echo changed >> {path}',
    "sed -i 's/True/False/' {path}", "sed -i 's/a.path/a.relpath/g' {path}",
])
def test_unknown_writers_and_unsupported_edits_invalidate_custody(tmp_path, intervening):
    path = tmp_path/'startup.py'
    calls = [call(write(path),0), call(intervening.format(path=path),1), call(f'python {path}',2)]
    assert recover(calls)[-1].sources == ()


@pytest.mark.parametrize('suffix', [' || true', '; true', ' | tail -5'])
def test_failed_inner_python_cannot_be_masked_by_outer_shell(tmp_path, suffix):
    command = shlex.join(['bash','-lc', 'python -c '+shlex.quote(SCRIPT)])+suffix
    assert recover([call(command)])[0].sources == ()


def test_failed_missing_and_overlapping_writes_cannot_seed_source(tmp_path):
    path = tmp_path/'startup.py'
    for writer in (call(write(path),0,success=False), replace(call(write(path),0),result_present=False,succeeded=False),
                   call(write(path),0,end=5)):
        assert recover([writer,call(f'python {path}',1)])[-1].sources == ()
    pending = replace(call('touch untracked',2),end_event=None,succeeded=False)
    assert recover([call(write(path),0),call(f'python {path}',1,end=7),pending])[1].sources == ()


def test_safe_literal_global_edit_preserves_exact_source_and_no_backdating(tmp_path):
    path=tmp_path/'startup.py'
    original=SCRIPT.replace('startup complete','OLD OLD')
    rows=recover([call(write(path,original),0),call(f"sed -i 's/OLD/NEW/g' {path}",1),call(f'python {path}',2)])
    assert rows[-1].sources == (original.replace('OLD','NEW'),)
    rows=recover([call(write(path,'print(1)\n'),0),call(f'python {path}',1),call(write(path),2)])
    assert S.discovery(rows[1].sources[0]) == set()
    assert rows[2].sources == ()
    assert not path.exists()  # The observer never opens current script files.


def test_relative_paths_need_explicit_per_command_cwd(tmp_path):
    assert recover([call(write('startup.py'),0),call('python startup.py',1)])[-1].sources == ()
    prefix=f'cd {tmp_path} && '
    rows=recover([call(prefix+write('startup.py'),0),call(prefix+'python startup.py',1)])
    assert rows[-1].sources == (SCRIPT,)
    assert rows[-1].cwd == str(tmp_path)
    assert recover([call(prefix+write('startup.py'),0),call('cd /elsewhere && python startup.py',1)])[-1].sources == ()


def test_saved_source_writing_submission_is_a_real_first_mutation(tmp_path):
    ctx=context(tmp_path)
    script=tmp_path/'startup.py'
    source="from pathlib import Path\nPath('submission/x.py').write_text('x')\n"+SCRIPT
    t=transcript(tmp_path/'t.jsonl',[write(script, source),f'python {script}'],ctx)
    calls=C._tool_calls(t)
    assert C._submission_mutation(calls[1]) is True
