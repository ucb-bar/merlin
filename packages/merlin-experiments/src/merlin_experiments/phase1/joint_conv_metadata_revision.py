"""Bind a corrected duplicate channel count to an unchanged typed convolution."""
from __future__ import annotations

import copy
import math
from pathlib import Path

from merlin.targetgen.contract.interface_emit import parse_interface_mlir

from ..phase2.contracts import document_sha256


def verify(evidence, *, members):
    from .joint_source_revision import _capsule

    if (set(evidence) != {'schema', 'name', 'before', 'after'}
            or evidence['schema'] != 'merlin.typed_convolution_channel_metadata_review.v1'):
        raise ValueError('channel metadata needs an exact typed-source review')
    old, old_files = _capsule(evidence['before'])
    new, new_files = _capsule(evidence['after'])
    name = evidence['name']
    source = old.get('interface_mlir')
    if (name not in members or old.get('name') != name or new.get('name') != name
            or old.get('kind') != 'layer' or old.get('operation', {}).get('op') != 'conv2d'
            or not isinstance(source, str) or source not in old_files
            or old_files != new_files or not old_files
            or members[name].get('capsule_sha256') != document_sha256(old)):
        raise ValueError('channel metadata changes source members or has a stale binding')
    before = old['operation'].get('attributes', {}).get('ci')
    after = new.get('operation', {}).get('attributes', {}).get('ci')
    if type(before) is not int or type(after) is not int or min(before, after) <= 0 or before == after:
        raise ValueError('channel metadata requires one changed positive integer count')
    restored = copy.deepcopy(new)
    restored['operation']['attributes']['ci'] = before
    if restored != old:
        raise ValueError('channel metadata changes more than the duplicate channel count')
    program = parse_interface_mlir((Path(evidence['after']['path']).parent / source).read_text())
    commands = [command for command in program['commands'] if command['opcode'] == 'CONV2D']
    if (len(commands) != 1
            or any(command['opcode'] not in {'CONV2D', 'RES_PACK', 'EVICT'} for command in program['commands'])):
        raise ValueError('channel metadata requires one unchanged typed convolution')
    command = commands[0]
    attributes, tensors = command['attributes'], program['tensors']
    kernel = attributes.get('kernel')
    if (not isinstance(kernel, list) or len(kernel) != 4
            or any(type(extent) is not int or extent <= 0 for extent in kernel)
            or attributes.get('layout') != 'nhwc'):
        raise ValueError('channel metadata requires explicit typed NHWC geometry')
    input_shape = tensors[command['operands']['ifm']]['shape']
    output_shape = tensors[command['operands']['dst']]['shape']
    if (len(input_shape) != 4 or not output_shape
            or after != kernel[2] or after != input_shape[-1]
            or output_shape[-1] != kernel[-1]):
        raise ValueError('corrected channel count differs from unchanged typed operands')
    contraction = [math.prod(output_shape[:-1]), math.prod(kernel[:3]), kernel[-1]]
    if members[name].get('contractions') != [contraction]:
        raise ValueError('channel metadata changes the selected typed contraction geometry')
    # Only the declaration identity changes. Traffic, facts, prices, numerics,
    # required tiers and every companion byte remain the original selection.
    return name, document_sha256(new)
