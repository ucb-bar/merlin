"""Reopen a missing historical ELF through its original recorded replay input.

Only the exact pinned replay receipt's own workspace/program.elf is admissible.
This preserves the missing original path, adds no execution/numerical credit and
issues no runtime view or calibration permission.
"""
from __future__ import annotations

import json
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.runtime import _native_files as F


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _pin(path, *, limit=64 << 20):
    path=F.canonical(Path(path))
    _require(path.stat().st_nlink==1,'recorded replay member has aliased storage')
    return F.pin(path,max_bytes=limit)


def _reopen(pin, *, limit=64 << 20):
    path=Path(pin['path'])
    _require(_pin(path,limit=limit)==pin,'original recorded replay member changed')
    return path


def verify(receipt_pin, receipt, *, runtime_sha256, model_pin):
    """Return exact provenance, never an arbitrary matching-file fallback."""
    original=receipt['original']['elf']
    original_path=F.canonical(Path(original['path']),exists=False)
    if original_path.exists():
        _reopen(original)
        return dict(kind='original_elf_present',original=original)
    owner=_reopen(receipt_pin,limit=16 << 20).parent
    _require(json.loads(Path(receipt_pin['path']).read_bytes())==receipt,
             'replay equivalence differs from its pinned original receipt')
    _require(receipt.get('schema')=='operator.retained_same_elf_observation.v1'
             and receipt.get('transport')=='existing_htif_symbol_elf_signature'
             and receipt.get('physical_runs')==0 and receipt.get('calibration_eligible') is False
             and receipt.get('runtime_sha256')==runtime_sha256 and receipt.get('model')==model_pin
             and receipt.get('functional_console_equal') is True,
             'recorded input equivalence requires the exact original diagnostic replay owner')
    model=json.loads(_reopen(model_pin,limit=16 << 20).read_bytes())
    report_path=_reopen(receipt['report'],limit=16 << 20)
    report=json.loads(report_path.read_bytes());binding=receipt['scope_binding']
    _require(binding.get('elf_sha256')==original['sha256'] and binding.get('input_sha256')==original['sha256']
             and binding.get('scope_sha256')==model['domain']['scope_sha256']
             and report.get('elf_sha256')==original['sha256'] and report.get('model_sha256')==model_pin['sha256']
             and report.get('scope_binding')==binding and report.get('timing_complete') is True
             and report.get('execution_completed') is True and report.get('unresolved')==[],
             'recorded replay input lost its exact model/counter scope')
    copied=owner/'workspace/program.elf';scope=owner/'workspace/scope.json'
    copied_pin,scope_pin=_pin(copied),_pin(scope,limit=16 << 20)
    _require(copied_pin['sha256']==original['sha256'] and json.loads(scope.read_bytes())==binding,
             'original replay-owned input copy or scope changed')
    runs=receipt.get('runs')
    _require(isinstance(runs,list) and len(runs)==2,'recorded replay needs both original native invocations')
    consoles=[]
    for index,run in enumerate(runs):
        invocation=_reopen(run['invocation'],limit=1 << 20)
        _require(invocation.is_relative_to(owner/f'run-{index}'/'invocations'),
                 'replay input equivalence selected another invocation owner')
        row=I.verify(invocation)
        _require(row.get('kind')=='subprocess' and row.get('stage')=='archive_screen_count'
                 and copied_pin in row['inputs'] and scope_pin in row['inputs']
                 and model_pin in row['dependencies'] and row['argv'][-1]==str(copied),
                 'original native invocation does not pin this ELF/scope/model input')
        if index==1:
            _require(receipt['report'] in row['outputs'] and all(row['argv'].count(option)==1 for option in (
                '--timing-model='+model_pin['path'], '--timing-report='+str(report_path),
                '--timing-scope-binding='+str(scope), '--timing-input-sha256='+original['sha256'])),
                'annotated native producer does not bind this exact model/report/scope')
        else:
            _require(not any(arg.startswith('--timing-')for arg in row['argv']),
                     'original functional replay was replaced by an annotated command')
        consoles.append(_reopen(row['stdout']).read_bytes())
    _require(consoles[0]==consoles[1],'original replay consoles differ')
    return dict(kind='original_recorded_replay_input_equivalence',original=original,
        original_path_present=False,receipt=receipt_pin,input=copied_pin,scope_input=scope_pin,
        model=model_pin,scope_binding=binding,invocations=[r['invocation']for r in runs],
        qualification='byte/input equivalence only; no new observation, runtime view or fit permission')
