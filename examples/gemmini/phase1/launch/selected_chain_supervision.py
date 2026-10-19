"""Operator coordination of independently admitted native builds and full chains."""
from __future__ import annotations

from pathlib import Path

import launch_phase1 as P
from merlin.common.digest import sha256_file


def selected(launch, run_dir, deadline):
    from merlin_experiments.phase1.feedback import private_numerical_selection as selection
    spec = launch.path(launch.document['inputs']['private_full_model_spec'])
    value = selection.read(spec)
    if value is None:
        return None
    if Path(value['config']) != launch.config.path or value['deadline_unix'] != deadline:
        raise ValueError('selected native full chain config/deadline differs')
    root = Path(value['static_root'])
    if root.exists():
        raise ValueError('selected native static workspace must be fresh')
    root.mkdir(parents=True, mode=0o700)
    return {'selection': value, 'root': root, 'spec': spec, 'run_dir': run_dir}


def worker_argv(context, launch, name, submission, pin, owner_pin):
    python = launch.expand(launch.document['launcher']['python'])
    return [python, '-u', str(Path(__file__).with_name('native_full_chain.py')),
            '--job', str(pin), '--owner-pin', str(owner_pin), '--run-dir', str(context['run_dir']),
            '--descriptor', str(launch.path(launch.document['inputs']['descriptor'])),
            '--submission', str(submission), '--out', str(context['root'] / name),
            '--model', name, '--deadline-unix', str(context['selection']['deadline_unix'])]


def publish_static(context, names):
    path = context['root'] / 'static-index.json'
    if path.exists():
        return
    records = {name: context['root'] / name / 'native-static.json' for name in names}
    if not all((p.parent / 'partition-ready.json').is_file() for p in records.values()):
        return
    # Each ready marker is observational. The native consumer independently
    # reopens each original completed invocation, static report and linked image.
    P.save(path, {'schema': 'merlin.phase1.native_static_chain_index.v1',
                  'run_dir': str(context['run_dir']),
                  'records': {name: str(p) for name, p in records.items()}})


def publish_numerical(context, launch, *, children, names, state, identity, package, output_context, output):
    path = context['root'] / 'full-chain-qualification.json'
    if path.exists() or not all(state['models'].get(n, {}).get('full_chain_complete') is True for n in names):
        return
    final = output.finalize(output_context, {n: children[n] for n in names})
    if final['status'] != 'pass':
        raise ValueError('full-chain all-output finalization failed')
    def pin(p):
        return {'path': str(p), 'sha256': sha256_file(p)}
    P.save(path, {
        'schema': 'merlin.phase1.selected_quantized_full_chain.v1', 'passed': True,
        'run_dir': str(context['run_dir']), 'private_spec_sha256': sha256_file(context['spec']),
        'candidate_tree_sha256': package['hash_tree_sha256'], 'submission': package,
        'config': pin(launch.config.path),
        'resolution': {key.lower(): launch.variables[key] for key in ('TASK_ROOT', 'PREP', 'RUN_ID')},
        'selected_identity': identity,
        'models': {n: {'exit_code': children[n]['process'].poll(), 'report': pin(children[n]['report']),
                       'job': pin(children[n]['pin'])} for n in names},
        'all_outputs_finalization': final,
        'host_execution': {'selected': False, 'status': 'not_selected'},
    })
    state['selected_numerical_receipt'] = pin(path)
