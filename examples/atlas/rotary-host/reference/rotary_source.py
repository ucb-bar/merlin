"""Replay the rotary dependency slice of captured source FX in reference v3.

The frontend trace supplies original ATen schemas, scalar arguments and tensor
metadata only. It supplies no numerical outputs, lowered semantics or code.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import rotary_exact
import source_trace


def program_from_trace(trace_path, positions_end=None):
    import torch
    trace_path = rotary_exact.checked(trace_path)
    trace = json.loads(trace_path.read_text())
    source = trace['graphs']['quantized']
    nodes = source['nodes']
    indexed = {n['id']: n for n in nodes}
    def dependencies(value):
        if isinstance(value, dict):
            if 'node_id' in value:
                yield value['node_id']
            else:
                for child in value.values():
                    yield from dependencies(child)
        elif isinstance(value, list):
            for child in value:
                yield from dependencies(child)
    outputs = {}
    for n in nodes:
        if n['target'] == 'aten.to.dtype':
            deps = list(dependencies(n['args']))
            if len(deps) != 1:
                continue
            parent = indexed[deps[0]]
            if parent['target'] != 'aten.mul.Tensor' or parent['args'][1:] != [1.0]:
                continue
            original = indexed[list(dependencies(parent['args'][0]))[0]]
            if original['target'] in ('aten.sin.default', 'aten.cos.default'):
                if n['results'][0]['dtype'] != 'bfloat16':
                    raise ValueError('captured rotary output cast is not BF16')
                name = original['target'].split('.')[1]
                if name in outputs:
                    raise ValueError('source trace has multiple rotary output casts')
                outputs[name] = n['id']
    if set(outputs) != {'sin', 'cos'}:
        raise ValueError('source trace lacks unique scaled rotary sin/cos BF16 outputs')
    closure = set()
    def visit(key):
        if key in closure:
            return
        closure.add(key)
        for dependency in dependencies([indexed[key]['args'], indexed[key]['kwargs']]):
            visit(dependency)
    for key in outputs.values():
        visit(key)
    for record in nodes:
        if record['id'] in closure and record['op'] == 'placeholder':
            if (record.get('input_kind') != 'BUFFER'
                    or record.get('input_target') != 'model.model.rotary_emb.inv_freq'):
                raise ValueError('rotary source dependency is not captured inverse frequency')

    def load_constant(record):
        if record.get('input_target') != 'model.model.rotary_emb.inv_freq':
            raise ValueError('rotary source dependency is not captured inverse frequency')
        from safetensors import safe_open
        weights = rotary_exact.checked(trace_path.parent / 'weights.safetensors')
        with safe_open(str(weights), framework='np') as archive:
            frequency = archive.get_tensor(record['input_target'])
        if (frequency.dtype != np.float32 or frequency.shape != (32,)
                or sha256(frequency.astype('<f4').tobytes()).hexdigest() != rotary_exact.INV_FREQ_SHA256):
            raise ValueError('source lifted inverse-frequency bytes differ')
        return torch.from_numpy(frequency.copy())

    def transform_args(record, args):
        if positions_end is not None and record['target'] == 'aten.arange.default':
            # Qualification expands only source arange extent.
            return [positions_end]
        return args

    concatenated = False

    def transform_metadata(record, results):
        nonlocal concatenated
        if positions_end is None:
            return results
        result = results[0]
        shape = [positions_end if extent == 64 else extent for extent in result['shape']]
        # The final head dimension is 64, not a sequence dimension.
        if record['target'] == 'aten.cat.default':
            concatenated = True
        if concatenated and len(shape) == 3 and result['shape'][-1] == 64:
            shape[-1] = 64
        return [{**result, 'shape': shape}]

    program, _ = source_trace.rebuild_program(nodes, load_constant, selected_ids=closure,
        output_ids=[outputs[name] for name in ('cos', 'sin')],
        transform_args=transform_args, transform_metadata=transform_metadata)
    return program, {'trace_sha256': sha256(trace_path.read_bytes()).hexdigest(),
                     'trace_rebuild_sha256': sha256(Path(source_trace.__file__).read_bytes()).hexdigest(),
                     'source_graph_stage': 'quantized', 'source_graph_sha256': source.get('sha256'),
                     'source_output_ids': outputs, 'source_node_count': len(closure),
                     'source_inv_freq_sha256': rotary_exact.INV_FREQ_SHA256,
                     'qualification_arange_extent': positions_end}


def replay(trace_path, policy, positions_end=None, retain=()):
    import whole_reference
    program, provenance = program_from_trace(trace_path, positions_end)
    interpreter = whole_reference.Interpreter(program, policy, retain=retain)
    outputs = interpreter.run([])
    return {'cos': outputs[0], 'sin': outputs[1]}, provenance, interpreter
