"""Review host package routing metadata without changing captured typed math."""
from __future__ import annotations

import copy
from pathlib import Path

from merlin.common import mlir_query
from xdsl.dialects.builtin import TensorType

from ..phase2.contracts import document_sha256


def verify(evidence, *, members):
    from .joint_source_revision import _capsule

    if (set(evidence) != {'schema', 'name', 'before', 'after'}
            or evidence['schema'] != 'merlin.captured_host_lane_review.v1'):
        raise ValueError('host lane correction requires an exact operator source review')
    old, old_files = _capsule(evidence['before'])
    new, new_files = _capsule(evidence['after'])
    name = evidence['name']
    attrs = (old.get('operation') or {}).get('attributes') or {}
    if (name not in members or old.get('name') != name or new.get('name') != name
            or old.get('kind') != 'model_slice'
            or attrs.get('dtype') != 'int8' or attrs.get('compile_dtype') != 'fp32'
            or old_files != new_files or not old_files
            or members[name]['capsule_sha256'] != document_sha256(old)):
        raise ValueError('host lane correction changes source or has stale metadata')
    expected = copy.deepcopy(old)
    expected['operation']['attributes']['compile_dtype'] = 'int8'
    if expected != new:
        raise ValueError('host lane correction changes another source or numerical obligation')
    source = old.get('linalg_mlir')
    screen = old.get('software_screen', {})
    if (source not in old_files or old.get('interface_mlir') != source
            or screen.get('status') not in {'admitted', 'unsupported'}
            or screen.get('program_sha256') != old_files[source]):
        raise ValueError('host lane correction requires unchanged source-bound placement screen')
    module = mlir_query.parse((Path(evidence['before']['path']).parent / source).read_text())
    contractions = []
    for op in module.walk():
        if op.name != 'linalg.generic':
            continue
        inputs = [str(v.type.get_element_type()) for v in op.inputs if isinstance(v.type, TensorType)]
        outputs = [str(v.type.get_element_type()) for v in op.outputs if isinstance(v.type, TensorType)]
        names = {nested.name for nested in op.walk()}
        if inputs[:2] == ['i8', 'i8'] and outputs == ['i32'] and {'arith.muli', 'arith.addi'} <= names:
            contractions.append(op)
    if not contractions:
        raise ValueError('host lane correction requires original i8 to i32 contraction math')
    # This changes only package routing. Native owned-host execution independently
    # requires respect_captured_quantization_scope=True in its build receipt;
    # no dtype rewrite, numerical pass, or placement authority is issued here.
    return name, document_sha256(new)
