"""Replay unchanged FX with Atlas contractions and explicitly labeled host context."""
from __future__ import annotations

import json
import re
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from . import MODEL_PIN, SOURCE_PIN
from .activation import quantize_act
from .arithmetic import bf16_bits, bf16_values
from .contractions import contraction_nt, qlinear
from .ops import register
from .io import digest, checked_path

BUNDLE_FILES = {'model.pt2', 'fixtures.safetensors', 'parity.json',
                'buffer_inventory.json', 'graph_inspection.json', 'policy.json',
                'graph.txt', 'graph_signature.txt'}


def raw_numpy(tensor):
    t = tensor.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.uint16).numpy()
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8).numpy()
    if t.dtype == torch.float8_e4m3fnuz:
        raise ValueError('FNUZ storage is outside the E4M3FN graph contract')
    return t.numpy()


def tensor_record(tensor):
    data = raw_numpy(tensor)
    return {'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
            'sha256': digest(data.tobytes())}


def array_record(data, filename):
    data = np.asarray(data)
    return {'file': str(filename), 'shape': list(data.shape), 'dtype': data.dtype.str,
            'sha256': digest(data.tobytes())}


def write_manifest(output, record):
    # Replace the whole manifest, so readers never observe a partial JSON file.
    with tempfile.NamedTemporaryFile(mode='w', dir=output, prefix='.reference-',
                                     suffix='.json', delete=False) as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(output / 'reference.json')


def source_module(node):
    stack = node.meta.get('nn_module_stack', {})
    return next(reversed(stack.values()))[0] if stack else ''


def layer_name(node):
    stack = node.meta.get('nn_module_stack', {})
    for path, typename in reversed(list(stack.values())):
        if re.search(r'(?:^|\.)layers\.\d+$', path):
            return path
    return None


class Oracle(torch.fx.Interpreter):
    def __init__(self, module, out, unit, case, *, save_operands=False,
                 engine='numpy', threads=1):
        super().__init__(module)
        self.directory = Path(out)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.unit, self.case = unit, case
        self.save_operands = save_operands
        self.engine, self.threads = engine, threads
        self.entries, self.host_nodes, self.layers = [], [], []
        self.last_layer_nodes = {}
        for node in module.graph.nodes:
            if layer_name(node) and node.op == 'call_function':
                self.last_layer_nodes[layer_name(node)] = node.name

    def save(self, name, data):
        if not re.fullmatch(r'[A-Za-z0-9_.]+', name):
            raise ValueError('unsafe boundary name')
        data = np.asarray(data)
        filename = name + '.npy'
        np.save(checked_path(self.directory / filename), data, allow_pickle=False)
        return array_record(data, filename)

    def run_node(self, node):
        start = time.monotonic()
        target = str(node.target)
        if node.op == 'call_function' and target.startswith('atlas.'):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            value, entry = self.quantized(node, args, kwargs)
            entry.update(node=node.name, op=target, module=source_module(node),
                         layer=layer_name(node), case=self.case, unit=self.unit,
                         elapsed_seconds=round(time.monotonic()-start, 6))
            self.entries.append(entry)
            print(json.dumps({'node': node.name, 'op': target, 'case': self.case,
                              'unit': self.unit, 'elapsed_seconds': entry['elapsed_seconds']}), flush=True)
        else:
            value = super().run_node(node)
            if node.op == 'call_function':
                self.host_nodes.append({'node': node.name, 'op': target,
                                        'module': source_module(node),
                                        'arithmetic': 'unchanged PyTorch graph context'})
        layer = layer_name(node)
        if layer and self.last_layer_nodes.get(layer) == node.name and isinstance(value, torch.Tensor):
            self.layers.append({'layer': layer, 'node': node.name,
                                'scope': 'Atlas quantized ops, unchanged PyTorch ordinary nodes',
                                'output': self.save('layer.' + layer, raw_numpy(value)),
                                'logical_dtype': str(value.dtype)})
        return value

    def quantized(self, node, args, kwargs):
        kind = str(node.target).split('.')[1]
        args = list(args)
        if kwargs:
            schema = node.target._schema
            args = [args[i] if i < len(args) else kwargs.get(a.name, a.default_value)
                    for i, a in enumerate(schema.arguments)]
        if kind == 'quantize_act':
            x, mode, reduce, rule, static_exp, code_dtype, *tail = args
            if x.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError('activation source must be BF16 or float32')
            if code_dtype not in ('uint8', 'float8_e4m3fn'):
                raise ValueError('unsupported activation storage dtype')
            tag = tail[0] if tail else ''
            rows = tail[1] if len(tail) > 1 else 32
            codes, exp = quantize_act(raw_numpy(x) if x.dtype == torch.bfloat16 else x.float().numpy(),
                                      mode, reduce, rule, static_exp, rows)
            code_tensor = torch.from_numpy(codes.copy())
            if code_dtype == 'float8_e4m3fn':
                code_tensor = code_tensor.view(torch.float8_e4m3fn)
            entry = {'codes': self.save(node.name + '.codes', codes),
                     'exponents': self.save(node.name + '.exponents', exp),
                     'policy': dict(mode=mode, reduce=reduce, rule=rule,
                                    static_exp=static_exp, rows=rows, tag=tag),
                     'input': tensor_record(x)}
            return (code_tensor, torch.from_numpy(exp.copy())), entry
        if kind == 'qconv2d':
            raise ValueError('qconv2d is explicitly outside the TinyLlama reference scope')
        if kind not in ('qlinear', 'qk_matmul', 'pv_matmul'):
            raise ValueError('unsupported Atlas custom operator: ' + kind)
        a, ax, b, bx = (raw_numpy(v) for v in args[:4])
        operands = dict(a_codes=a, a_exp=ax, b_codes=b, b_exp=bx)
        bias = None
        if kind == 'qlinear':
            bias, out_dtype, fmt = args[4:7]
            if fmt != 'atlas_e4m3' or out_dtype not in ('bfloat16', 'float32'):
                raise ValueError('unsupported qlinear format or output dtype')
            if bias is not None and bias.dtype != torch.bfloat16:
                raise ValueError('only unchanged BF16 source bias is admitted')
            bias = None if bias is None else raw_numpy(bias)
            result = qlinear(a, ax, b, bx, unit=self.unit, bias_bf16=bias,
                             engine=self.engine, threads=self.threads)
            logical_dtype = getattr(torch, out_dtype)
        else:
            if len(args) > 4 and args[4] != 0:
                raise ValueError('folded PV pop_shift is forbidden by the unchanged graph contract')
            result = contraction_nt(a, ax, b, bx, unit=self.unit,
                                    engine=self.engine, threads=self.threads)
            logical_dtype = torch.float32
        if bias is not None:
            operands['bias_bf16'] = bias
        entry = {'operands': {k: {'shape': list(v.shape), 'dtype': v.dtype.str,
                                  'sha256': digest(v.tobytes())} for k, v in operands.items()},
                 'accumulator_bf16': self.save(node.name + '.accumulator_bf16', result.accumulator_bf16),
                 'readout_bf16': self.save(node.name + '.readout_bf16', result.readout_bf16),
                 'output_bf16': self.save(node.name + '.output_bf16', result.output_bf16),
                 'logical_dtype': str(logical_dtype)}
        if self.save_operands:
            filename = node.name + '.operands.npz'
            np.savez(self.directory / filename, **operands)
            entry['operand_file'] = filename
        value = torch.from_numpy(result.output_bf16.copy()).view(torch.bfloat16)
        value = value.to(logical_dtype)
        entry['output'] = self.save(node.name + '.output', raw_numpy(value))
        return value, entry


def validate_bundle(directory):
    directory = checked_path(directory)
    manifest = json.loads((directory / 'fx_manifest.json').read_text())
    if manifest.get('schema') != 'atlas_torchao.fx_bundle.v1' or not manifest.get('native_fp8'):
        raise ValueError('expected native FP8 FX bundle')
    if set(manifest.get('files', {})) != BUNDLE_FILES:
        raise ValueError('incomplete or unexpected source file inventory')
    for name, record in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('unsafe bundle filename')
        content = checked_path(directory / name).read_bytes()
        if digest(content) != record['sha256'] or len(content) != record['bytes']:
            raise ValueError('bundle file integrity failure: ' + name)
    policy = json.loads((directory / 'policy.json').read_text())
    if 'pack_fold' in json.dumps(policy) and _contains_enabled_pack_fold(policy):
        raise ValueError('folded source artifact is outside the selected contract')
    return manifest


def validate_graph_bindings(program, directory, held):
    inspected = json.loads((directory / 'graph_inspection.json').read_text())
    bindings = inspected.get('weight_site_bindings', [])
    if not bindings or inspected.get('weight_sites_verified') != len(bindings):
        raise ValueError('missing or inconsistent source weight bindings')
    nodes = {n.name: n for n in program.module().graph.nodes}
    counts = {kind: sum(str(n.target) == f'atlas.{kind}.default' for n in nodes.values())
              for kind in ('qlinear', 'qconv2d', 'qk_matmul', 'pv_matmul')}
    pairs = sum(row['pairs'] for row in inspected.get('native_fp8_attention_modules', []))
    if (counts['qlinear'] != inspected.get('native_fp8_linear_sites')
            or counts['qconv2d'] != inspected.get('native_fp8_conv2d_sites')
            or counts['qk_matmul'] != pairs or counts['pv_matmul'] != pairs):
        raise ValueError('graph operator coverage differs from source inspection')
    for binding in bindings:
        if not binding.get('native') or binding.get('kind') != 'linear':
            raise ValueError('non-native or non-linear weight binding is outside scope')
        node = nodes.get(binding['qlinear'])
        if node is None or str(node.target) != 'atlas.qlinear.default':
            raise ValueError('missing graph qlinear weight binding')
        for index, key in ((2, 'code_buffer'), (3, 'scale_buffer')):
            source = node.args[index]
            if (not isinstance(source, torch.fx.Node) or source.op != 'get_attr'
                    or source.target != binding[key] or source.target not in held):
                raise ValueError('graph weight binding differs from the source buffer inventory')
    return counts


def _contains_enabled_pack_fold(value):
    if isinstance(value, dict):
        return any((k in ('pack_fold', 'pack_folded') and v not in (False, None, {}, []))
                   or _contains_enabled_pack_fold(v) for k, v in value.items())
    return isinstance(value, list) and any(_contains_enabled_pack_fold(v) for v in value)


def run_bundle(bundle, output, *, units=('mxu0',), cases=None, save_operands=False,
               engine='numpy', threads=1):
    from safetensors.torch import load_file
    directory, output = checked_path(bundle), checked_path(output)
    if not units or len(set(units)) != len(units) or any(u not in ('mxu0', 'mxu1') for u in units):
        raise ValueError('units must be unique MXU0 or MXU1 selections')
    if engine not in ('numpy', 'native') or not 1 <= threads <= 8:
        raise ValueError('invalid arithmetic engine or thread count')
    manifest = validate_bundle(directory)
    register()
    program = torch.export.load(directory / 'model.pt2')
    expected_inventory = json.loads((directory / 'buffer_inventory.json').read_text())
    held = dict(program.named_buffers())
    quantized = {k: tensor_record(v) for k, v in held.items()
                 if k.rsplit('.', 1)[-1] in ('weight_codes', 'weight_scale_exp',
                                            'weight_lo_codes', 'weight_lo_scale_exp')}
    if not quantized or quantized != expected_inventory:
        raise ValueError('source quantized buffer inventory is empty or incomplete')
    for name, expected in expected_inventory.items():
        if name not in held or tensor_record(held[name]) != expected:
            raise ValueError('graph buffer identity failure: ' + name)
    graph_counts = validate_graph_bindings(program, directory, held)
    before = {k: tensor_record(v) for k, v in held.items()}
    parameters_before = {k: tensor_record(v) for k, v in program.named_parameters() if not v.is_meta}
    fixtures = load_file(str(directory / 'fixtures.safetensors'))
    fixtures_before = {k: tensor_record(v) for k, v in fixtures.items()}
    parity = json.loads((directory / 'parity.json').read_text())
    cases = list(range(len(parity['cases']))) if cases is None else list(cases)
    if not cases or len(set(cases)) != len(cases):
        raise ValueError('fixture case selection must be nonempty and unique')
    result = {'schema': 'atlas_graph_reference.v1', 'model_pin': MODEL_PIN,
              'graph_source_pin_contract': SOURCE_PIN,
              'source_manifest_sha256': digest((directory / 'fx_manifest.json').read_bytes()),
              'source_graph_sha256': manifest['files']['model.pt2']['sha256'],
              'source_provenance': manifest.get('provenance', {}),
              'source_native_fp8': manifest['native_fp8'],
              'source_quantized_op_counts': graph_counts,
              'reference_sources': {str(p.relative_to(Path(__file__).parent.parent)):
                                    digest(p.read_bytes()) for p in
                                    (Path(__file__).parent / n for n in
                                     ('__init__.py', 'arithmetic.py', 'activation.py',
                                      'contractions.py', 'bundle.py', 'ops.py', 'io.py',
                                      *(['native.py', 'native.c'] if engine == 'native' else [])))},
              'torch_context_version': torch.__version__, 'k_order': 'ascending',
              'engine': engine, 'threads': threads,
              'k_instruction_width': 32,
              'scope': 'device quantized op outputs in unchanged PyTorch ordinary-node context',
              'whole_layer_device_exactness': 'PENDING ordinary-node device arithmetic policy',
              'requested_units': list(units), 'requested_cases': cases, 'complete': False,
              'runs': []}
    regeneration = directory / 'REGENERATED.json'
    if regeneration.is_file():
        regeneration_bytes = checked_path(regeneration).read_bytes()
        regeneration_record = json.loads(regeneration_bytes)
        if regeneration_record.get('pack_fold'):
            raise ValueError('regenerated source identifies a forbidden pack fold')
        result['regeneration'] = regeneration_record
        result['regeneration_record_sha256'] = digest(regeneration_bytes)
    if engine == 'native':
        from .native import backend_metadata
        result['native_backend'] = backend_metadata()
    validation = Path(__file__).parent.parent / 'out' / 'model-validation.json'
    if validation.is_file():
        qualification = json.loads(validation.read_text())
        if (qualification.get('pass') and qualification.get('model', {}).get('revision') == MODEL_PIN
                and qualification.get('arithmetic_source_sha256') == result['reference_sources']['atlas_ref/arithmetic.py']):
            result['handmade_model_validation'] = {'sha256': digest(validation.read_bytes()),
                                                    'totals': qualification['totals']}
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, result)
    for unit in units:
        for case in cases:
            if case < 0 or case >= len(parity['cases']):
                raise ValueError('case index outside fixtures')
            record = parity['cases'][case]
            interpreter = Oracle(program.module(), output / unit / f'case_{case}', unit, case,
                                 save_operands=save_operands, engine=engine, threads=threads)
            args = [fixtures[name] for name in record['inputs']]
            with torch.no_grad():
                value = interpreter.run(*args)
            leaves = torch.utils._pytree.tree_leaves(value)
            final = [interpreter.save(f'final.{i}', raw_numpy(v)) for i, v in enumerate(leaves)
                     if isinstance(v, torch.Tensor)]
            run = {'case': case, 'unit': unit, 'directory': f'{unit}/case_{case}',
                   'input_fixtures': record['inputs'], 'entries': interpreter.entries,
                   'layers': interpreter.layers, 'final': final,
                   'ordinary_nodes': interpreter.host_nodes}
            result['runs'].append(run)
            after = {k: tensor_record(v) for k, v in program.named_buffers()}
            parameters_after = {k: tensor_record(v) for k, v in program.named_parameters() if not v.is_meta}
            fixtures_after = {k: tensor_record(v) for k, v in fixtures.items()}
            if after != before or parameters_after != parameters_before or fixtures_after != fixtures_before:
                raise ValueError('source buffer, parameter, or fixture mutated during independent replay')
            result['all_graph_buffers_immutable'] = True
            result['all_materialized_parameters_and_fixtures_immutable'] = True
            result['logical_codes_and_exponents_identical'] = True
            write_manifest(output, result)
    result['complete'] = True
    write_manifest(output, result)
    return result
