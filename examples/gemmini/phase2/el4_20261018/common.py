"""Small, task-local record keeping for the public EL4 experiment campaign."""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

MERLIN_ROOT = next(p for p in Path(__file__).resolve().parents if (p/'.git').exists())
TASK = MERLIN_ROOT.parent
HERE = Path(__file__).resolve().parent
os.environ.setdefault('MERLIN_REPO_ROOT',str(MERLIN_ROOT))
os.environ.setdefault('MERLIN_OUT_ROOT',str(TASK/'out'))
sys.path.insert(0,str(MERLIN_ROOT/'src'))
from merlin.common.paths import artifacts_dir, build_dir
EVIDENCE = artifacts_dir()/'perf-studies/gemmini/el4-20261018'
OUT = build_dir()/'phase2-campaign'
P1 = TASK.parent/'phase1-harness-v2'
CORPUS = P1 / '.tmp/handoff/functional-phase1-baseline-v1/phase0-v11/phase0'
BASE = TASK / 'gemmini-mlir/phase1_generated/el4_fire_aligned_v23_20261018/package'
CANDIDATE = TASK / 'gemmini-mlir/phase2_candidates/rhs_residency/package'
PYTHON = P1 / 'merlin/.venv/bin/python'
START = dt.datetime.fromisoformat('2026-10-18T19:57:49.333+00:00')
DEADLINE = START + dt.timedelta(hours=4)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def logical_command_buffer(value):
    """Keep semantics/source/placement/ABI; omit only final ISA position annotations."""
    value = copy.deepcopy(value)
    plan = value.get('params', {}).get('global_program_plan')
    if plan is not None:
        for field in ('schedule_instruction_count','prologue_instruction_range','epilogue_instruction_range'):
            plan.pop(field,None)
        for task in plan.get('tasks',[]):
            task.pop('instruction_start',None)
            task.pop('instruction_end',None)
    return value


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tree_hash(path):
    h = hashlib.sha256()
    files = []
    for item in sorted(Path(path).rglob('*')):
        if not item.is_file() or any(p in {'build', '__pycache__', '.git'} for p in item.relative_to(path).parts):
            continue
        name = item.relative_to(path).as_posix()
        h.update(name.encode() + b'\0' + item.read_bytes() + b'\0')
        files.append(name)
    return h.hexdigest(), files


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=TASK/'.tmp',prefix='phase2-record-',delete=False) as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        temporary=Path(stream.name)
    temporary.replace(path)


def environment():
    env = dict(os.environ)
    env.update({
        'TASK_DIR': str(TASK), 'TMPDIR': str(TASK / '.tmp'),
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONPATH': ':'.join(str(TASK / p) for p in (
            'merlin/src', 'merlin/packages/merlin-experiments/src', 'merlin/packages/merlin-timing/src')),
        'MERLIN_REPO_ROOT': str(TASK / 'merlin'), 'MERLIN_OUT_ROOT': str(TASK / 'out'),
        'MERLIN_MLIR_INSTALL': str(TASK.parent.parent/'toolchains/llvm-a47bddcc/install'),
        'MERLIN_CLANG': str(TASK.parent.parent/'toolchains/llvm-a47bddcc/install/bin/clang-23'),
        'MERLIN_RTL_FACTS': str(P1 / 'merlin/out/artifacts/probes/gemmini/fire-aligned-30mhz/input/v6/facts.json'),
        'MERLIN_TARGET_CONTRACT': str(P1 / 'merlin/out/artifacts/probes/gemmini/v1/probes_gemmini_v1_20261018T090833Z_580ba4a/target_contract.json'),
        'MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME': str(P1 / 'out/fire-aligned-spike-30mhz/runtime'),
        'MERLIN_GEMMINI_GSIM_EMU': str(P1 / '.tmp/gsim-fire-aligned-provisional-v3/native/emulator'),
        'MERLIN_GSIM_EMU_GEMMINI': str(P1 / '.tmp/gsim-fire-aligned-provisional-v3/native/emulator'),
        'MERLIN_COMPUTE_CPUS': '6,7',
        'MERLIN_COMPUTE_CPU_SLOTS': str(P1 / '.tmp/phase1-compute-slots'),
        'MERLIN_GSIM_SLOT_ROOT': str(P1 / '.tmp/phase1-gsim-slots'),
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    })
    return env


def configure():
    os.environ.update(environment())
    import sys
    for path in os.environ['PYTHONPATH'].split(':'):
        if path not in sys.path:
            sys.path.insert(0, path)


def initialize():
    if (EVIDENCE / 'campaign.json').exists():
        return json.loads((EVIDENCE / 'campaign.json').read_text())
    baseline, files = tree_hash(BASE)
    assert baseline == 'b847c973da82f408a32c7622d0c49fb6902f0a58d0f716f131d8501e34175338' and len(files) == 21
    value = {
        'schema': 'phase2.exploratory.campaign.v1',
        'start_utc': START.isoformat(), 'deadline_utc': DEADLINE.isoformat(),
        'start_basis': 'original Codex session_meta timestamp; includes Plan Mode audit',
        'session_id': '01a15097-8394-7b52-9648-6baf3a45c962',
        'final_reserve_seconds': 2700, 'minimum_improvement': 0.01, 'plateau_batches': 3,
        'baseline_sha256': baseline, 'baseline_files': files,
        'merlin_revision': '4acf64171c21a9a15117e1dc05f655c00f43ae09',
        'gemmini_revision': '50f34dbcbf23d1b489b988c02fd659cba0f262ee',
        'formal_admission': False, 'route': 'exploratory_public_kernels',
        'resources': {'cpus': [6, 7], 'max_campaign_simulations': 1,
                      'shared_cpu_slots': environment()['MERLIN_COMPUTE_CPU_SLOTS'],
                      'shared_gsim_slots': environment()['MERLIN_GSIM_SLOT_ROOT']},
        'dependencies': [
            'Phase1: TinyLlama22-layer batch1 32-prefill/32-decode same-contract reference comparison',
            'Phase1: locate first SmolVLA divergence; accepted 0.1335856828 approximation is not equivalence',
            'Phase1: selected ResNet54-contraction scalar or grouped16 readout and recipe qualification',
            'Phase1: eight column-readout admissions and one public micro-model capture; seal complete corpus',
            'Phase1/operator: live formal baseline, runtime and protocol/timing certificates',
        ],
        'usage': {'author': 'session-local token events; aggregate only latest cumulative total',
                  'operator': None, 'helper': None, 'preliminary_interrupted_session': 'usage unavailable'},
    }
    save(EVIDENCE / 'campaign.json', value)
    return value


def remaining(*, final=False):
    reserve = 0 if final else initialize()['final_reserve_seconds']
    return DEADLINE.timestamp() - time.time() - reserve


@contextlib.contextmanager
def stage(name, **details):
    initialize()
    started = time.monotonic()
    done = threading.Event()
    job_path = OUT / 'jobs' / f'{os.getpid()}.json'

    def refresh():
        while True:
            now = dt.datetime.now(dt.timezone.utc)
            value = {'at': now.isoformat(), 'active': name, 'details': details,
                     'elapsed_seconds': (now - START).total_seconds(),
                     'remaining_seconds': (DEADLINE - now).total_seconds(),
                     'optimization_remaining_seconds': remaining(),
                     'formal_admission': False, 'spike_qualified': False,
                     'best': 'immutable baseline; candidates unpromoted',
                     'dependencies': initialize()['dependencies']}
            save(job_path, dict(value, pid=os.getpid(), running=True))
            value['active_jobs'] = []
            for job in (OUT/'jobs').glob('*.json'):
                try:
                    record = json.loads(job.read_text())
                    os.kill(record['pid'], 0)
                    if record.get('running'):
                        value['active_jobs'].append({k:record[k] for k in ('pid','active','details','at')})
                except (OSError, ValueError, KeyError):
                    pass
            save(OUT / 'status.json', value)
            (OUT / 'LIVE.log').parent.mkdir(parents=True, exist_ok=True)
            with (OUT / 'LIVE.log').open('a') as f:
                f.write(f"{value['at']} {name} elapsed={value['elapsed_seconds']:.0f}s remaining={value['remaining_seconds']:.0f}s {json.dumps(details)}\n")
            if done.wait(30):
                return

    thread = threading.Thread(target=refresh, daemon=True)
    thread.start()
    try:
        yield details
    finally:
        done.set()
        thread.join()
        if job_path.exists():
            value = json.loads(job_path.read_text())
            save(job_path, dict(value, running=False))
        with (OUT / 'costs.jsonl').open('a') as f:
            f.write(json.dumps({'stage': name, 'elapsed_s': time.monotonic() - started,
                                'details': details, 'at': dt.datetime.now(dt.timezone.utc).isoformat()}) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['init', 'status'])
    args = parser.parse_args()
    print(json.dumps(initialize() if args.action == 'init' else json.loads((OUT / 'status.json').read_text()), indent=2))
