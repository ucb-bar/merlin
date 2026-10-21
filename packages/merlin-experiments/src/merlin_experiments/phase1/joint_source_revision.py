"""Reviewed software-screen rebinding; numerical/source files stay byte-identical."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

from ..phase2.contracts import document_sha256
from .feedback import joint_performance as PERF


def _capsule(pin):
    path = Path(pin['path'])
    if (not path.is_absolute() or path != path.resolve() or path.name != 'capsule.yaml'
            or path.is_symlink() or not path.is_file() or path.stat().st_size > 1 << 20):
        raise ValueError('source revision requires a bounded direct capsule.yaml')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin['sha256']:
        raise ValueError('source revision capsule changed')
    document = yaml.safe_load(raw)
    if not isinstance(document, dict):
        raise ValueError('source revision capsule must be a mapping')
    files = {}
    total = 0
    for leaf in sorted(path.parent.rglob('*')):
        if leaf.is_symlink() or not (leaf.is_file() or leaf.is_dir()):
            raise ValueError('source revision has indirect source members')
        if leaf.is_file() and leaf != path:
            total += leaf.stat().st_size
            if len(files) >= 128 or total > 256 << 20:
                raise ValueError('source revision source tree exceeds review budget')
            files[str(leaf.relative_to(path.parent))] = hashlib.sha256(leaf.read_bytes()).hexdigest()
    return document, files


def normalized(initial, current):
    """Verify explicit added reviews, then restore initial hashes for policy comparison."""
    before = initial.get('source_binding_revisions', [])
    after = current.get('source_binding_revisions', [])
    if (not isinstance(before, list) or not isinstance(after, list) or len(after) > 16
            or after[:len(before)] != before):
        raise ValueError('source revision ancestry changed')
    value = copy.deepcopy(current)
    members = copy.deepcopy(initial['members'])
    for revision in after[len(before):]:
        if (set(revision) != {'schema', 'review_id', 'evidence'}
                or revision['schema'] != 'merlin.source_screen_revision.v1'
                or not isinstance(revision['review_id'], str) or not revision['review_id'].strip()):
            raise ValueError('source revision needs an explicit bounded operator review')
        evidence = PERF._read(revision['evidence'])
        if set(evidence) != {'schema', 'name', 'before', 'after'} or evidence['schema'] != 'merlin.source_screen_equivalence.v1':
            raise ValueError('source revision evidence has an unsupported scope')
        old, old_files = _capsule(evidence['before'])
        new, new_files = _capsule(evidence['after'])
        name = evidence['name']
        if (old.get('name') != name or new.get('name') != name or name not in members
                or {k:v for k,v in old.items() if k != 'software_screen'} != {k:v for k,v in new.items() if k != 'software_screen'}
                or old_files != new_files or not old_files):
            raise ValueError('source revision changes source, inputs, outputs, or numerical contract')
        screen = new.get('software_screen', {})
        source = new.get('linalg_mlir') or new.get('interface_mlir')
        if (screen.get('status') != 'admitted' or not isinstance(source, str)
                or screen.get('program_sha256') != new_files.get(source)):
            raise ValueError('source revision lacks admitted source-bound software screen')
        if members[name].get('capsule_sha256') != document_sha256(old):
            raise ValueError('source revision previous member binding differs')
        members[name]['capsule_sha256'] = document_sha256(new)
    if members != current['members']:
        raise ValueError('source revision changes unreviewed member facts')
    value['members'] = copy.deepcopy(initial['members'])
    value.pop('source_binding_revisions', None)
    return value
