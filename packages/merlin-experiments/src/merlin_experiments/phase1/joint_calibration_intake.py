"""Public completed-product evidence intake; never a fit or dispatch roster.

The native completed-row owner supplies results. Selected public membership is
reopened before recording, and canonical ELF link custody is checked separately.
An observation remains pending until an operator reviews its protocol, independent
source group, missing modelling question and physical cost. No consumer of this
index can expand the calibration profile or physical whitelist implicitly.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

from merlin.runtime import _native_files as F

from ..phase2.contracts import document_sha256

MAX_ENTRIES = 2048


def project(*, capsule, result, performance, source_sha256, compiler_sha256,
            elf, selection_pin, facts_pin, report_pin=None, report=None,
            reviewed_group=None):
    """Compact projection of already checked inputs; no authority is issued."""
    report = report or {}
    binding = report.get('scope_binding') or {}
    matches = binding.get('elf_sha256') == elf['sha256']
    complete = bool(matches and binding.get('input_sha256') and binding.get('scope_sha256')
                    and report.get('feature_observation', {}).get('complete') is True)
    return dict(
        schema='merlin.joint_calibration_candidate.v1',
        capsule_name=capsule['name'], capsule_sha256=source_sha256,
        compiler_sha256=compiler_sha256, executable=elf,
        performance_selection=selection_pin, facts=facts_pin, timing_report=report_pin,
        numerical_status=result.get('numeric', {}).get('status'),
        structural_qualification=performance.get('structural_qualification'),
        model_status=performance.get('status'),
        protocol_status='complete_observation' if complete else 'unknown_or_incomplete',
        scope_binding=binding if matches else None,
        model_sha256=report.get('model_sha256'),
        source_group_sha256=reviewed_group,
        source_group_basis='reviewed_calibration_profile' if reviewed_group else 'unreviewed',
        source_provenance={k: capsule[k] for k in
                           ('source_role', 'source_reference', 'group_identity_sha256') if k in capsule},
        eligible_candidate=False, status='pending_review',
        required_reviews=['supported_protocol', 'independent_source_group', 'missing_model_question',
                          'physical_cost_and_whitelist', 'exact_elf_config_scope_join'],
        gsim_observation_reuse='pending_exact_identity_review',
        dispatch_authorized=False, fit_authorized=False,
    )


def register(row, root, *, result, compiler, capsule, state, profile):
    """Index a selected public numerical result without running any observer."""
    from merlin.benchharness import hash_tree
    from . import joint_control
    from .feedback import joint_performance as PERF
    from ..phase2.census_launch import _producer

    if (capsule.get('kind') == 'model'
            or capsule.get('name') in state['policy'].get('final_only_capsules', ())
            or result.get('numeric', {}).get('status') != 'pass'
            or not (result.get('compiler_source') or result.get('candidate_native_execution'))):
        return None
    from .joint_calibration_eligibility import eligible
    if not eligible(result, capsule):
        return None
    # This selection is the public, source-bound roster, not hidden capsule discovery.
    selection_pin = state.get('performance_selection') or profile['performance_selection']
    selection = PERF._read(selection_pin)
    if capsule.get('name') not in selection.get('members', {}):
        return None
    selected = PERF.selected_capsule(selection, capsule)
    source = PERF.capsule_identity(selected)
    root = Path(root).resolve()
    elf = _producer(PERF.original_executable(root),
                    SimpleNamespace(max_elf_bytes=64 << 20, max_bins=4096))
    compiler_sha = hash_tree(Path(compiler).resolve())['sha256']
    report_pin = report = None
    report_path = root / 'joint_timing' / 'report.json'
    if report_path.is_file():
        report_pin = F.pin(report_path, max_bytes=16 << 20)
        report = PERF._read(report_pin)
    member = profile['members'].get(capsule['name'], {})
    value = project(capsule=selected, result=result, performance=row.get('performance') or {},
                    source_sha256=source, compiler_sha256=compiler_sha, elf=elf,
                    selection_pin=selection_pin, facts_pin=selection['facts'],
                    report_pin=report_pin, report=report,
                    reviewed_group=member.get('source_group_sha256'))
    # Retain native correctness receipts (including any gSIM observations) without
    # relabelling their scope or claiming that physical/model identities agree.
    value['native_result'] = result
    value['generated_root'] = str(root)
    key = document_sha256(value)
    registry = Path(os.environ[joint_control.ENV]).parent / 'calibration_candidate_index'
    registry.mkdir(parents=True, exist_ok=True)
    path = registry / (key + '.json')
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(encoded) > 2 << 20:
        raise ValueError('public calibration evidence entry exceeds metadata budget')
    # Serialize the bound and publication across native row owners. Dispatch
    # never watches this descriptive directory.
    with (registry / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return F.pin(path, max_bytes=2 << 20)
        if sum(1 for p in registry.iterdir() if p.suffix == '.json') >= MAX_ENTRIES:
            raise ValueError('public calibration evidence index reached its bounded capacity')
        with path.open('xb') as stream:
            stream.write(encoded)
        return F.pin(path, max_bytes=2 << 20)
