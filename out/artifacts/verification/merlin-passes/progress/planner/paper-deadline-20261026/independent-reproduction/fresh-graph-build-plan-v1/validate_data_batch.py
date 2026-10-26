"""One actual-plan data batch + tiny controls + unrelated-CWD same-file relocation.

Only own stdlib utility code executes. Copied Python/Lean/provider/controller
source is DATA; the audit denies execution from payload and every process/socket/
ctypes path. All control inputs are new tiny text fixtures, not semantic tests.
"""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import runpy
import sys

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from build_plan import PlanError, Reader, build, canonical, decode, header, pin
from verify_plan import verify

denied = []

def audit(event, args):
    if event.startswith(('subprocess.', 'socket.', 'ctypes.')) or event in ('os.system', 'os.exec', 'os.posix_spawn', 'os.fork'):
        denied.append({'event': event})
        raise PermissionError('data batch forbids process/native action')
    if event == 'exec':
        filename = args[0].co_filename
        if '/payload/' in filename:
            raise PermissionError('copied payload code must never execute')
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0])
        path = Path(name).absolute()
        mode, flags = args[1] or '', args[2]
        write = any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        allowed = path.is_relative_to(OUT) or name.startswith('/usr/lib/python3.14/')
        if write:
            allowed = path.is_relative_to(OUT)
        if not allowed or any(t in name for t in ('/.elan/', '/toolchains/', 'liblean', 'original-runtime')):
            denied.append({'event': event, 'path': name})
            raise PermissionError('outside source-plan data allowlist')

sys.addaudithook(audit)
actual = verify(OUT)
assert not actual['unavailable_profiles'] and not actual['unavailable_cases']
(OUT / 'ACTUAL_PLAN_VALIDATION.json').write_text(json.dumps(actual, indent=2) + '\n')

controls = OUT / 'controls'
controls.mkdir()

def fixture(name):
    directory = controls / name
    directory.mkdir()
    payload = directory / 'payload'
    payload.mkdir()
    def ref(filename, raw):
        (payload / filename).write_bytes(raw)
        return {'file': 'payload/' + filename, **pin(raw)}
    base = ref('Base.lean', b'theorem base : True := by trivial\n')
    use = ref('Use.lean', b'import Merlin.Base\ntheorem use : True := by trivial\n')
    query = ref('Query.lean', b'import Merlin.Use\ntheorem toy_final : True := by trivial\n')
    meta = ref('request.json', json.dumps({'query': (directory / query['file']).read_text(),
                'final_root': 'toy_final', 'support_roots': ['Merlin.Base.base']}).encode())
    case = {'id': 'tiny_case', 'module': 'Query', 'query': query, 'request_metadata': meta,
            'prepared_metadata': meta, 'bindings': {'query': {'path': ['query']},
                'final_root': {'path': ['final_root']}, 'support_roots': {'path': ['support_roots']}},
            'final_root': 'toy_final', 'support_roots': ['Merlin.Base.base'], 'buffers': [],
            'domain': None, 'relation': None, 'assumptions': None}
    profile = {'id': 'tiny', 'project_prefixes': ['Merlin.'],
               'catalog': [{'module': 'Merlin.Base', 'source': base, 'provenance': 'new tiny DATA'},
                           {'module': 'Merlin.Use', 'source': use, 'provenance': 'new tiny DATA'}], 'cases': [case]}
    inputs = {'schema': 'merlin.fresh-graph-plan-inputs.v1', 'profiles': [profile],
              'primary_baseline': 'tiny_DATA_only', 'future_root': 'fresh-build'}
    return directory, inputs, ref

results = []
for name in ('valid', 'missing_source', 'missing_file', 'conflicting_module', 'cycle', 'source_mutation',
             'query_mutation', 'support_root_swap', 'plan_root_swap', 'unknown_header'):
    directory, inputs, ref = fixture(name)
    profile = inputs['profiles'][0]
    if name == 'missing_source':
        profile['catalog'] = profile['catalog'][1:]
    elif name == 'missing_file':
        profile['catalog'][0]['source']['file'] = 'payload/not_present.lean'
    elif name == 'conflicting_module':
        profile['catalog'].append({'module': 'Merlin.Base',
            'source': ref('conflict.lean', b'theorem different : True := by trivial\n'), 'provenance': 'conflict DATA'})
    elif name == 'cycle':
        profile['catalog'][0]['source'] = ref('Cycle.lean', b'import Merlin.Use\ntheorem base : True := by trivial\n')
    elif name == 'source_mutation':
        (directory / profile['catalog'][0]['source']['file']).write_bytes(b'--changed source DATA\n')
    elif name == 'query_mutation':
        (directory / profile['cases'][0]['query']['file']).write_bytes(b'--changed query DATA\n')
    elif name == 'support_root_swap':
        profile['cases'][0]['support_roots'] = ['Merlin.Base.wrong_root']
    elif name == 'unknown_header':
        profile['catalog'][0]['source'] = ref('Unknown.lean', b'unsupported_header Merlin.Base\n')
    plan = build(inputs, Reader(directory))
    (directory / 'INPUTS.json').write_text(json.dumps(inputs, indent=2) + '\n')
    if name == 'plan_root_swap':
        plan['profiles'][0]['cases'][0]['final_root'] = 'wrong_root'
    (directory / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    if name in ('valid', 'plan_root_swap'):
        try:
            observed = verify(directory)
            assert name == 'valid'
            assert not observed['unavailable_profiles'] and not observed['unavailable_cases']
        except PlanError as error:
            assert name == 'plan_root_swap'
            observed = {'status': 'EXPECTED_PLAN_REFUSAL', 'reason': str(error)}
    else:
        p = plan['profiles'][0]
        errors = [p.get('reason')] if p['status'] == 'UNAVAILABLE' else [c['reason'] for c in p['cases'] if c['status'] == 'UNAVAILABLE']
        assert errors, name
        observed = {'status': 'EXPECTED_SOURCE_DATA_UNAVAILABLE', 'reasons': errors}
    results.append({'control': name, 'observed': observed, 'semantic_or_native_case': False})
(OUT / 'BOUNDED_CONTROLS.json').write_text(json.dumps(results, indent=2) + '\n')

# Same inode input/plan/code copies under a new root; no archive, extraction,
# installation or process. The relocated CLI imports its own copied utility.
relocated = OUT / 'relocated'
relocated.mkdir()
inputs = decode((OUT / 'INPUTS.json').read_bytes())
names = ['INPUTS.json', 'PLAN.json', 'build_plan.py', 'verify_plan.py'] + [r['file'] for r in inputs['packet_data_refs']]
for name in names:
    target = relocated / name
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(OUT / name, target)
assert all(os.path.samefile(OUT / name, relocated / name) for name in names)
unrelated = OUT / 'unrelated-cwd'
unrelated.mkdir()
previous_cwd, previous_argv, previous_path = Path.cwd(), sys.argv[:], sys.path[:]
previous_builder = sys.modules.pop('build_plan')
try:
    os.chdir(unrelated)
    sys.argv = ['verify_plan.py', '--packet', str(relocated)]
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        try:
            runpy.run_path(str(relocated / 'verify_plan.py'), run_name='__main__')
        except SystemExit as result:
            assert result.code == 0
    relocation = decode(stdout.getvalue())
    assert relocation == actual
finally:
    os.chdir(previous_cwd)
    sys.argv, sys.path = previous_argv, previous_path
    sys.modules['build_plan'] = previous_builder
(OUT / 'RELOCATION_CHECK.json').write_text(json.dumps({'status': 'PASSED_SAME_FILE_UNRELATED_CWD',
    'same_file_associations': len(names), 'result': relocation,
    'copied_provider_or_Lean_contents_executed': False, 'native_launches': 0}, indent=2) + '\n')
assert not any(name.startswith(('merlin', 'xdsl', 'z3', 'lean_graph_')) for name in sys.modules)
summary = {'scope': 'one bounded source/data validation batch', 'actual_profiles': actual['profiles'],
           'actual_source_data_files': actual['source_data_files_checked'], 'tiny_controls': len(results),
           'all_tiny_controls_expected': True, 'same_file_relocation': 'PASSED',
           'guard_denials': denied, 'SDK_or_native_reads_or_process_launches': 0,
           'provider_controller_or_Lean_execution': False, 'proof_or_qualification': False}
(OUT / 'VALIDATION_BATCH.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
