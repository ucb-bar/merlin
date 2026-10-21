"""Bounded automatic projection of existing selfcheck observations; no target work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.common.tree_hash import hash_tree

from . import joint_diagnostics, joint_runtime_observations


def _receipt(root):
    path = Path(root) / 'joint_compiler_observations.json'
    if not path.exists():
        return {}
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or path.stat().st_size > 1 << 20:
        raise ValueError('compiler observation receipt must be a bounded direct file')
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get('schema') != 'merlin.compiler_source_observations.v1':
        raise ValueError('unsupported compiler observation receipt')
    artifact = {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}
    return {name: {'pins': document.get('pins', {}), 'artifact': artifact,
                   'planned': section.get('planned', {}), 'emitted': section.get('emitted', {})}
            for name, section in document.get('sections', {}).items()
            if name in {'packing_reuse', 'crosslayer_lifetimes', 'connected_fusion'} and isinstance(section, dict)}


def attach(row, root, *, report, observation, result, compiler):
    """Attach six bounded sections after the existing performance projection.

    Only original observer products and this candidate's optional compile receipt
    are read. A receipt is descriptive, never shared-pass realization proof.
    Physical/reference archives are not inputs. Failure cannot change grading.
    """
    performance = row['performance']
    compiler_sha256 = hash_tree(Path(compiler))['sha256']
    context = dict(compiler_sha256=compiler_sha256,
                   capsule_sha256=performance.get('capsule_sha256'),
                   scope_sha256=performance.get('scope_sha256'),
                   model_sha256=performance.get('model_sha256'),
                   elf_sha256=(report or {}).get('elf_sha256'))
    observations = {}
    receipt_status = 'absent'
    try:
        observations.update(_receipt(root))
        if observations:
            receipt_status = 'present; context validated per section'
    except (OSError, ValueError, TypeError, KeyError):
        receipt_status = 'unavailable or malformed'
    runtime = None
    if report is not None and observation.get('status') == 'observed':
        runtime = joint_runtime_observations.produce(
            report, context=context, artifact=observation['report'])
        observations.update(runtime['observations'])
    diagnostics = joint_diagnostics.project(
        performance, compiler_sha256=compiler_sha256,
        capsule_sha256=context['capsule_sha256'],
        correctness=(result.get('numeric') or {}).get('status', 'unknown'),
        observations=observations)
    diagnostics['compiler_receipt_status'] = receipt_status
    if runtime is not None:
        diagnostics['runtime'] = {key: runtime[key] for key in
                                  ('selected_window', 'feature_counts_complete', 'pressure', 'wholechain', 'unavailable')}
    row['compiler_diagnostics'] = diagnostics
