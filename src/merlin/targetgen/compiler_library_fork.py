"""Candidate compiler source overlays; no candidate code is imported by this module.

The operator's fixed public inventory bounds names. Candidate bytes stay untrusted
and execute only through the ordinary package sandbox, with independent grading.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .compiler_library import CompilerLibraryError

REVIEW_PREFIX = 'editable-generic-compiler-v1:'
DIRECTORY = 'compiler_library'
MAX_BYTES = 64 << 20


def enabled(selection):
    return selection is not None and selection.contract.review_id.startswith(REVIEW_PREFIX)


def editable(member):
    # Compiler-side copies never replace independent host checks. Keep grading,
    # provenance, target facts and runtime endpoints on the fixed guest release.
    roots = ('merlin.compile', 'merlin.transforms', 'merlin.sched',
             'merlin.semantic_compiler', 'merlin.llvmlower', 'merlin.frontends',
             'merlin.xdsl_dialects', 'merlin.common', 'merlin.capture', 'merlin.integrations')
    return member.module != 'merlin.common.access' and member.module is not None and any(
        member.module == root or member.module.startswith(root + '.') for root in roots)



def permits_namespace(contract, module):
    """Exact already-listed namespace identity, never a recursive module grant."""
    return contract.review_id.startswith(REVIEW_PREFIX) and any(
        member.module == module and member.path.endswith('/__init__.py') and editable(member)
        for member in contract.members)


def candidate_sources(selection, package):
    """Read and hash the exact bounded fork without executing or trusting it."""
    if not enabled(selection) or package is None:
        return {}, None
    root = Path(package) / DIRECTORY
    if not root.exists() and not root.is_symlink():
        return {}, None
    if root.is_symlink() or not root.is_dir():
        raise CompilerLibraryError('candidate compiler fork must be an unlinked directory')
    allowed = {m.path for m in selection.contract.members if editable(m)}
    result, rows, size = {}, {}, 0
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise CompilerLibraryError('candidate compiler fork cannot contain links')
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or relative not in allowed:
            raise CompilerLibraryError('candidate compiler fork contains an ungranted module: ' + relative)
        if path.stat().st_size > MAX_BYTES - size:
            raise CompilerLibraryError('candidate compiler fork exceeds source byte budget')
        payload = path.read_bytes()
        size += len(payload)
        if size > MAX_BYTES:
            raise CompilerLibraryError('candidate compiler fork exceeds source byte budget')
        payload.decode('utf-8')
        result[relative] = path
        rows[relative] = hashlib.sha256(payload).hexdigest()
    record = {'schema': 'merlin.candidate_compiler_fork.v1',
              'baseline_contract_sha256': selection.contract.sha256, 'files': rows}
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result, digest


def stage(selection, package):
    """Seed public generic source only once; never overwrite an author's fork."""
    if not enabled(selection):
        return
    root = Path(package) / DIRECTORY
    if root.exists() or root.is_symlink():
        candidate_sources(selection, package)
        return
    selection.record()
    root.mkdir(parents=True)
    for member in selection.contract.members:
        if not editable(member):
            continue
        path = root / member.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((selection.root / member.path).read_bytes())
    candidate_sources(selection, package)
