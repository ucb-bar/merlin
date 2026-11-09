"""Replay the rotary dependency slice of captured source FX in reference v3.

The frontend trace supplies original ATen schemas, scalar arguments and tensor
metadata only. It supplies no numerical outputs, lowered semantics or code.
"""
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rotary_exact


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
    graph = torch.fx.Graph()
    built, specs, state = {}, [], {}
    concatenated = False
    dtypes = {name: getattr(torch, name) for name in ('float32', 'bfloat16', 'int64', 'int32', 'bool', 'uint8')}
    def resolve(value):
        if isinstance(value, dict):
            if 'node_id' in value:
                return built[value['node_id']]
            if value.get('kind') == 'dtype':
                return getattr(torch, value['value'].removeprefix('torch.'))
            if value.get('kind') == 'device':
                return torch.device(value['value'])
            if value.get('kind') == 'layout':
                return getattr(torch, value['value'].removeprefix('torch.'))
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value
    for record in nodes:
        if record['id'] not in closure:
            continue
        if record['op'] == 'placeholder':
            if record.get('input_target') != 'model.model.rotary_emb.inv_freq':
                raise ValueError('rotary source dependency is not captured inverse frequency')
            node = graph.placeholder(record['target'])
            specs.append(SimpleNamespace(arg=SimpleNamespace(name=record['target']),
                                         kind=SimpleNamespace(name='BUFFER'), target=record['input_target']))
            from safetensors import safe_open
            weights = rotary_exact.checked(trace_path.parent / 'weights.safetensors')
            with safe_open(str(weights), framework='np') as archive:
                frequency = archive.get_tensor(record['input_target'])
            if (frequency.dtype != np.float32 or frequency.shape != (32,)
                    or sha256(frequency.astype('<f4').tobytes()).hexdigest() != rotary_exact.INV_FREQ_SHA256):
                raise ValueError('source lifted inverse-frequency bytes differ')
            state[record['input_target']] = torch.from_numpy(frequency.copy())
        elif record['op'] == 'call_function':
            target = torch.ops
            for part in record['target'].split('.'):
                target = getattr(target, part)
            args = resolve(record['args'])
            if positions_end is not None and record['target'] == 'aten.arange.default':
                # Qualification expands only source arange extent. No source
                # arithmetic, inverse frequencies or output words are changed.
                args = [positions_end]
            node = graph.call_function(target, tuple(args), resolve(record['kwargs']))
        else:
            raise ValueError('rotary source slice has unsupported node role')
        result = record['results'][0]
        shape = result['shape']
        if positions_end is not None:
            shape = [positions_end if extent == 64 else extent for extent in shape]
            # The final head dimension is 64, not a sequence dimension.
            if record['target'] == 'aten.cat.default':
                concatenated = True
            if concatenated and len(shape) == 3 and record['results'][0]['shape'][-1] == 64:
                shape[-1] = 64
        node.meta['val'] = torch.empty(shape, dtype=dtypes[result['dtype']], device='meta')
        built[record['id']] = node
    graph.output(tuple(built[outputs[name]] for name in ('cos', 'sin')))
    program = SimpleNamespace(graph=graph, graph_signature=SimpleNamespace(input_specs=specs), state_dict=state, constants={})
    return program, {'trace_sha256': sha256(trace_path.read_bytes()).hexdigest(),
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
