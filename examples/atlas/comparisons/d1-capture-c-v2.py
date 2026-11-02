"""Attention V2 Tiny capture with isolated native projections and original47 attention."""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
MERLIN_ROOT = HERE.parents[2]
ATLAS_ROOT = Path(os.environ['ATLAS_ROOT']).absolute()
ROOT = Path(os.environ['TINY_WORK_ROOT']).absolute()
ATLAS_HELPERS = ATLAS_ROOT/'tools/tiny_full'
MERLIN_HELPERS = MERLIN_ROOT/'examples/atlas/comparisons'
MODEL_ROOT = Path(os.environ['ATLAS_NPU_MODEL_ROOT']).absolute()
TOOLS = ATLAS_ROOT / 'tools'
COMPARISONS = MERLIN_ROOT / 'examples/atlas/comparisons'
sys.path[:0] = [str(COMPARISONS), str(MERLIN_ROOT / 'src'), str(TOOLS)]
import numpy as np
import torch
from atlas_path_guard import checked, lexical_absolute_path
from capture_tinyllama_fused import (FusedProjectionCapture, MODEL_PIN, ProjectionWorkers, SCALE_CONTRACT_SHA,
                                    assemble_fp8, down_scales, load_layout)
from capture_tinyllama_native import array_identity, identity
from capture_atlas_attention import fixed_exponent
from tiny_fp8_pack import encode
from tiny_workload_inventory import column_shards
from capture_tinyllama_full import padded_words
from compare_tinyllama_attention import raw_array
import tiny_workload_native as native
from tiny_full.capture_boundaries import Parameters, State, Phase, Projection, PROVIDER, HEADERS, bind_library

Attention = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_void_p, ct.c_void_p, ct.c_void_p,
                        ct.c_size_t, ct.c_size_t, ct.c_void_p)
FusedProjection = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p,
                              ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_void_p)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pointer_words(pointer, rows, width):
    return np.ctypeslib.as_array(ct.cast(pointer, ct.POINTER(ct.c_uint16)), shape=(rows*width,)).reshape(rows, width)


def python_sources(inputs):
    result = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).absolute()
            if path.suffix == '.py' and (path.is_relative_to(COMPARISONS) or path.is_relative_to(TOOLS)
                    or path.is_relative_to(MERLIN_ROOT / 'src')):
                inputs.pin(path)
                result[str(path)] = inputs.pins[str(path)]
    return result


def host_libraries(inputs):
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[-1].startswith('/'):
            path = Path(parts[-1])
            if path.name.startswith(('libm-', 'libm.so', 'libc-', 'libc.so', 'ld-', 'ld-linux', 'libtorch_cpu.so')):
                paths.add(checked(path))
    if not any(path.name.startswith(('libm-', 'libm.so')) for path in paths):
        raise ValueError('loaded native math library is not bound')
    for path in sorted(paths):
        inputs.pin(path)
    return [str(path) for path in sorted(paths)]


def worker_interpreter(path, inputs):
    """Bind a venv alias without losing its package prefix when launching it."""
    original = lexical_absolute_path(Path(path).absolute())
    candidate = original
    for _ in range(40):
        current, linked = Path(candidate.anchor), False
        for index, part in enumerate(candidate.parts[1:], 1):
            current /= part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                target = Path(os.readlink(current))
                if any(word in part.lower() for part in target.parts for word in ('hammer', 'vlsi')):
                    raise ValueError('restricted interpreter symlink target')
                target = target if target.is_absolute() else current.parent/target
                target = Path(os.path.normpath(str(target)))
                candidate = lexical_absolute_path(target.joinpath(*candidate.parts[index+1:]))
                linked = True
                break
        if not linked:
            if not stat.S_ISREG(candidate.stat().st_mode):
                raise ValueError('worker interpreter must be a regular executable')
            break
    else:
        raise ValueError('worker interpreter symlink chain is too long')
    inputs.pin(candidate)
    config = checked(original.parent.parent/'pyvenv.cfg')
    inputs.pin(config)
    return dict(alias=str(original), executable=str(candidate), executable_sha256=inputs.pins[str(candidate)],
                venv_config=str(config), venv_config_sha256=inputs.pins[str(config)])


class Inputs:
    def __init__(self, capture, contract):
        self.pins = {}
        contract_raw = self.pin(contract, SCALE_CONTRACT_SHA)
        self.contract = json.loads(contract_raw)
        self.path = checked(capture)
        self.data = json.loads(self.pin(self.path, self.contract['capture']['sha256']))
        if (self.data.get('status') != 'PASS_FULL_TINY_ARITHMETIC_CAPTURE_REPLAY'
                or len(self.data.get('parameters', {})) != 155 or len(self.data.get('host_parameters', {})) != 49):
            raise ValueError('complete frozen A parameter capture required')

    def pin(self, path, expected=None):
        path = checked(path)
        raw = path.read_bytes()
        value = digest(raw)
        if expected is not None and value != expected:
            raise ValueError('source digest differs: '+str(path))
        if self.pins.get(str(path), value) != value:
            raise ValueError('source changed during C capture: '+str(path))
        self.pins[str(path)] = value
        return raw

    def read(self, desc):
        member = Path(desc['file'])
        if member.is_absolute() or '..' in member.parts:
            raise ValueError('input parameter member escapes capture')
        raw = self.pin(self.path.parent / member, desc['sha256'])
        dtype = np.dtype(desc['dtype'])
        if dtype not in (np.dtype('uint16'), np.dtype('int64')) or len(raw) != desc['bytes']:
            raise ValueError('input parameter type or byte extent differs')
        if np.prod(desc['shape'])*dtype.itemsize != len(raw):
            raise ValueError('input parameter shape differs')
        return np.frombuffer(raw, dtype=dtype).reshape(desc['shape']).copy()

    def bookends(self):
        for path, value in list(self.pins.items()):
            self.pin(path, value)


def compile_host(inputs, out, compiler):
    compiler = checked(compiler)
    model = TOOLS / 'tiny_full/tiny_full_model.c'
    attention = ATLAS_ROOT / 'runtime/tiny_attention.c'
    runtime = attention.parent
    units = (model, attention, *(COMPARISONS/name for name in PROVIDER))
    for path in (compiler, Path(__file__).absolute(), model.with_suffix('.h'), runtime/'tiny_attention.h',
                 runtime/'atlas_numeric.h', TOOLS/'tiny_full/capture_boundaries.py',
                 TOOLS/'atlas_path_guard.py', *(COMPARISONS/name for name in PROVIDER+HEADERS)):
        inputs.pin(path)
    commands = []
    for unit in units:
        command = [str(compiler), '-std=c11', '-I', str(COMPARISONS), '-I', str(runtime), '-M', str(unit)]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        commands.append(command)
        for path in result.stdout.replace('\\\n', ' ').split(':', 1)[1].split():
            inputs.pin(path)
    command = [str(compiler), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
               '-ffp-contract=off', '-fno-fast-math', '-shared', '-fPIC', '-I', str(COMPARISONS), '-I', str(runtime),
               *(str(unit) for unit in units), '-lm', '-o', str(out/'native.so')]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (out/'compile.log').write_text(result.stdout+result.stderr)
    if result.returncode:
        raise ValueError('real native C host graph compilation failed')
    inputs.pin(out/'native.so')
    return bind_library(out/'native.so'), dict(compile_command=command, dependency_commands=commands)


class AttentionV2Workers(ProjectionWorkers):
    """One original47 process, with an explicit model-root argument."""
    def __init__(self, python, script, model_root, records):
        records.append(identity(script))
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env.update(PYTHONDONTWRITEBYTECODE='1', NPU_MODEL_RTL_NATIVE_MXU0='0')
        self.workers = []
        try:
            self.workers.append(subprocess.Popen([str(python), '-B', str(script), '--model-root', str(model_root),
                '--jsonl'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env))
        except BaseException:
            self.close()
            raise


ATTENTION_SCALE_CONTRACT_SHA = 'c6f5d6407ea784f6cc3c871d6cb93eb8cbf680cf276f8f37005f584f0858b8eb'


def attention_scales(inputs, path):
    contract = json.loads(inputs.pin(path, ATTENTION_SCALE_CONTRACT_SHA))
    result = {}
    for panel in contract['projection_kernels']:
        if panel.get('logical') != 3:
            continue
        key = panel['step'], panel['layer']
        if (key in result or panel['columns'] != [0, 2048] or panel['width'] != 2048
                or panel['physical_mkn'][1:] != [2048, 2048]
                or type(panel['A_scale']) is not int or not -60 <= panel['A_scale'] <= 60):
            raise ValueError('attention V2 scale policy is not one complete o projection per site')
        result[key] = panel['A_scale']
    if set(result) != {(step, layer) for step in range(3) for layer in range(22)}:
        raise ValueError('attention V2 fixed scale policy requires all 66 sites')
    return result


def fp8_placeholder(layout, raw, rows, width, scale):
    m = 128 if rows == 128 else 32
    if len(raw) != m*width:
        raise ValueError('attention V2 FP8 producer extent differs')
    physical = layout.unpack_fp8_tiles(raw, m, width)
    decoded = torch.from_numpy(physical.copy()).view(torch.float8_e4m3fn).float()
    return (decoded*2.0**scale).to(torch.bfloat16).view(torch.uint16).numpy()[:rows].copy()


def validate_attention_reply(record, config):
    rows, keys = config['query_rows'], config['keys']
    qbytes, kvbytes = (524288 if rows == 128 else 16384), ((keys+31)//32)*32*256*2
    loads = qbytes+2*kvbytes+(128+32+2048 if rows == 128 else 64+32+4096)
    dma = dict(load_bytes=loads, store_bytes=rows*2048)
    if (record.get('schema') != 'atlas.tiny.attention_launch.v2' or record.get('kind') != 'attention'
            or record.get('model_commit') != MODEL_PIN or record.get('numerics') != 'rtl'
            or record.get('accelerated') is not False or record.get('native_dispatch_enabled') is not False
            or record.get('config') != config or record.get('mode') != 'functional'
            or record.get('output_dtype') != 'fp8' or record.get('output_layout') != 'fp8_tiled'
            or record.get('output_scale') != config['o_exponent']
            or record.get('output_extent_bytes') != (262144 if rows == 128 else 65536)
            or record.get('requested_dma') != dma or record.get('execution', {}).get('observed') != dma
            or any(record.get(key) is not True for key in
                   ('source_bookends_verified', 'halt_exact', 'outside_written_spans_preserved'))):
        raise ValueError('attention V2 reply is not bound to this original47 functional request')


class CProjection(FusedProjectionCapture):
    """Share immutable producer staging with the Torch capture, using C boundaries."""
    def __init__(self, args, inputs, save, records):
        self.out, self.save, self.torch, self.inputs = args.out, save, torch, inputs
        self.records, self.calls, self.parameters = records, [], {}
        self.sites = tuple(f'model.layers.{i}.{site}' for i in range(args.layer_count) for site in native.PROJECTIONS)
        if not args.block_only:
            self.sites += ('lm_head',)
        self.layout = load_layout(args.atlas_layout, args.atlas_layout_sha256, records)
        self.scales = down_scales(args.scale_contract, records)
        self.o_scales = attention_scales(inputs, args.attention_scale_contract)
        self.pending_attention = None
        self.weights, self.saved, self.methods, self.context, self.modules = {}, {}, [], None, {}
        self.native_checker, self.native_proofs = None, []
        if getattr(args, 'native_proof_script', None) is not None:
            proof_id = dict(path=str(args.native_proof_script), sha256=args.native_proof_sha256)
            inputs.pin(proof_id['path'], proof_id['sha256'])
            records.append(dict(proof_id, bytes=Path(proof_id['path']).stat().st_size))
            spec = importlib.util.spec_from_file_location('d1_capture_native_proof', proof_id['path'])
            provider = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(provider)
            self.native_checker = provider.ProofClient(args.worker_python)
        try:
            self.workers = ProjectionWorkers(args.worker_python, args.worker_script, records)
        except BaseException:
            if self.native_checker is not None:
                self.native_checker.close()
            raise
        self.attention_workers = self.workers
        if getattr(args, 'attention_worker_python', None) is not None:
            try:
                self.attention_workers = AttentionV2Workers(args.attention_worker_python,
                    args.attention_worker_script, args.attention_model_root, records)
            except BaseException:
                self.workers.close()
                if self.native_checker is not None:
                    self.native_checker.close()
                raise
        self.attention_calls, self.attention_saved = [], {}
        self.counter, self.mode, self.step, self.errors = 0, 'capture', 0, []
        self.project_callback = Projection(self.project)
        self.fused_callback = FusedProjection(self.fused)
        self.attention_callback = Attention(self.attention)

    def linear(self, site, module, a, step, index, mode):
        rows, k = a.shape
        m = 128 if rows == 128 else 32
        role = 7 if site == 'lm_head' else native.PROJECTIONS.index('.'.join(site.split('.')[-2:]))
        context = self.context
        if role != 7 and (context is None or context['layer'] != int(site.split('.')[2])):
            raise ValueError('projection lost its decoder residual context')
        operand = None
        if role == 3:
            operand = context['entry']
        elif role == 5:
            operand = context['gate']
            if array_identity(a) != context['gate_input']:
                raise ValueError('gate and up normalized predecessors differ')
        elif role == 6:
            operand = context['residual']
        producer = context.get('attention_v2') if role == 3 else None
        if role == 3 and (producer is None or producer['step'] != step
                          or producer['layer'] != context['layer'] or producer['consumed']):
            raise ValueError('o projection requires its unconsumed typed attention V2 producer')
        physical = None
        signature = (array_identity(a), None if operand is None else array_identity(operand),
                     None if producer is None else (digest(producer['raw']), producer['scale'], producer['index']))
        key = (step, site)
        if mode == 'replay':
            old, words, emitted = self.saved[key]
            if signature != old:
                raise ValueError('replay fused predecessor or residual changed')
            if role in (3, 6) and (producer['raw'] if role == 3 else context['down_fp8']) != emitted:
                raise ValueError('replay emitted down FP8 predecessor changed')
        else:
            if site not in self.weights:
                raw_b = raw_array(module.weight.T, self.torch)
                self.parameters[site] = dict(B_bf16=self.save(f'parameters/{index:03d}-B.bin', raw_b))
                pieces = []
                for panel, shard in enumerate(column_shards(k, module.out_features)):
                    lo, hi = shard['columns']
                    encoded, scale = encode(np.ascontiguousarray(raw_b[:, lo:hi]))
                    desc = self.save_bytes(f'parameters/{index:03d}-{panel:02d}-B-fp8.bin',
                                           self.layout.pack_fp8_weights(encoded))
                    pieces.append((lo, hi, desc, scale))
                self.weights[site] = module.weight._version, module.weight.data_ptr(), pieces
            version, address, pieces = self.weights[site]
            if (version, address) != (module.weight._version, module.weight.data_ptr()):
                raise ValueError('cached parameter changed')
            stem = f'calls/{step:02d}-{index:03d}'
            down_scale = self.scales[(step, site.replace('up_proj', 'down_proj'))] if role == 5 else 0
            if role == 3:
                a_scale, packed_a = producer['scale'], producer['raw']
            elif role == 6:
                a_scale, packed_a = context['down_scale'], context['down_fp8']
            else:
                encoded, a_scale = encode(a if physical is None else physical)
                if physical is not None:
                    encoded[rows:] = 0
                elif rows == 1:
                    encoded = np.concatenate((encoded, np.zeros((31, k), np.uint8)))
                packed_a = self.layout.pack_fp8_tiles(encoded)
            a_desc = self.save_bytes(stem + '-A-fp8.bin', packed_a)
            call = dict(step=step, site=site, site_index=index, A_bf16=self.save(stem + '-A.bin', a),
                        A_fp8=a_desc, A_scale=a_scale, panels=[])
            if producer is not None:
                call.update(input_layout='fp8_tiled', attention_output_layout='attnfp8_tiled',
                            producer_index=producer['index'], input_attention=producer['index'])
            requests = []
            for panel, (lo, hi, b_desc, b_scale) in enumerate(pieces):
                panel_stem = stem + f'-{panel:02d}'
                request = dict(command='projection', root=str(self.out), out=str(self.out / panel_stem),
                               step=step, site=site, panel=panel, layer=-1 if role == 7 else context['layer'],
                               m=m, k=k, n=hi-lo, logical_rows=rows, role=role, column=lo, width=module.out_features,
                               rescale=a_scale+b_scale, a_scale=a_scale, b_scale=b_scale, output_scale=down_scale,
                               A_fp8=a_desc, B_fp8=b_desc)
                if operand is not None:
                    compact = np.ascontiguousarray(operand[:, lo:hi])
                    if compact.shape[0] == rows:
                        compact = padded_words(compact, rows)
                    elif compact.shape[0] != m:
                        raise ValueError('epilogue operand physical rows differ')
                    request['operand_tiles'] = self.save_bytes(panel_stem + '-operand.bin',
                                                              self.layout.pack_bf16_tiles(compact))
                requests.append(request)
            print(json.dumps(dict(milestone='projection_start', step=step, site=site)), flush=True)
            results = self.workers.run(requests)
            if len(results) != len(pieces):
                raise ValueError('incomplete projection panel replies')
            logical = []
            for panel, ((lo, hi, _, b_scale), launch) in enumerate(zip(pieces, results)):
                self.validate_launch(launch)
                for source in launch['implementation_sources']:
                    if source not in self.records:
                        self.records.append(source)
                output = np.frombuffer(self.read(launch['output_bf16']), dtype='<u2').reshape(rows, hi-lo).copy()
                if role == 5:
                    # FP8 producer BF16 is a Torch placeholder decoded by the worker, never a native product.
                    raw = self.layout.unpack_fp8_tiles(self.read(launch['output_fp8']), m, hi-lo)
                    decoded = self.torch.from_numpy(raw.copy()).view(self.torch.float8_e4m3fn).float()
                    expected = (decoded * 2.0**down_scale).to(self.torch.bfloat16).view(self.torch.uint16).numpy()
                    if not np.array_equal(expected[:rows], output):
                        raise ValueError('FP8 worker placeholder differs from exact emitted bytes')
                else:
                    expected = self.layout.unpack_bf16_tiles(self.read(launch['output_tiled']), m, hi-lo)
                logical.append(output)
                call['panels'].append(dict(columns=[lo, hi], logical_mkn=[rows, k, hi-lo],
                    physical_mkn=[m, k, hi-lo], A_scale=a_scale, B_scale=b_scale, rescale=a_scale+b_scale,
                    expected_bf16=self.save(stem + f'-{panel:02d}-expected.bin', expected), launch=launch))
                print(json.dumps(dict(milestone='projection_panel', step=step, site=site, panel=panel)), flush=True)
            words = np.concatenate(logical, axis=1)
            emitted = packed_a if role in (3, 6) else None
            if role == 5:
                emitted = assemble_fp8(call['panels'], self.read, m, module.out_features)
                call['output_fp8'] = self.save_bytes(stem + '-output-fp8.bin', emitted)
                call['output_scale'] = down_scale
            self.saved[key] = signature, words.copy(), emitted
            self.calls.append(call)
        if role == 3:
            producer['consumed'] = True
        elif role == 4:
            context['gate'], context['gate_input'] = words.copy(), array_identity(a)
        elif role == 5:
            context['down_fp8'] = emitted
            context['down_scale'] = self.scales[(step, site.replace('up_proj', 'down_proj'))]
        return self.tensor(words)

    def tensor(self, words):
        return words

    def validate_launch(self, launch):
        if self.native_checker is None:
            if (launch.get('model_commit') != MODEL_PIN or launch.get('accelerated') is not False
                    or launch.get('exact') is not True or launch.get('full_arena_exact') is not True):
                raise ValueError('original47 exact projection required')
            return
        if (launch.get('model_commit') != '70e2e7ceb81dac86210c8b3cca54024909c57d14'
                or launch.get('base_model_commit') != MODEL_PIN or launch.get('qualification_candidate') is not False
                or any(launch.get(key) is not True for key in
                       ('accelerated', 'native_dispatch_enabled', 'requires_original_backfill'))):
            raise ValueError('only admitted native70/base47 normal projections are accepted')
        proof = self.native_checker.verify(self.out, launch)
        canonical = hashlib.sha256(json.dumps(launch, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if (proof.get('status') != 'PASS_NATIVE_RECORD_ADMITTED_REQUIRES_ORIGINAL47_BACKFILL'
                or proof.get('record_sha256') != canonical or proof.get('program_sha256') != launch['program_sha256']
                or proof.get('requires_original_backfill') is not True
                or proof.get('original_final_graph_numerics_qualified') is not False):
            raise ValueError('native provenance proof is not bound to this unqualified launch')
        self.native_proofs.append(proof)
        for source in proof['source_records']:
            if source not in self.records:
                self.records.append(source)

    def parameter(self, site, index, k, width):
        if site not in self.modules:
            original = self.inputs.data['parameters'][site]['B_bf16']
            raw_b = self.inputs.read(original)
            if raw_b.shape != (k, width):
                raise ValueError('captured B shape differs from C callback')
            self.parameters[site] = dict(B_bf16=self.save(f'parameters/{index:03d}-B.bin', raw_b))
            pieces = []
            for panel, shard in enumerate(column_shards(k, width)):
                lo, hi = shard['columns']
                encoded, scale = encode(np.ascontiguousarray(raw_b[:, lo:hi]))
                desc = self.save_bytes(f'parameters/{index:03d}-{panel:02d}-B-fp8.bin',
                                       self.layout.pack_fp8_weights(encoded))
                pieces.append((lo, hi, desc, scale))
            self.weights[site] = 0, 0, pieces
            self.modules[site] = SimpleNamespace(out_features=width,
                weight=SimpleNamespace(_version=0, data_ptr=lambda: 0))
        return self.modules[site]

    def invoke(self, layer, role, a, k, width):
        index = 154 if role == 7 else layer*7+role
        site = 'lm_head' if role == 7 else f'model.layers.{layer}.{native.PROJECTIONS[role]}'
        if self.counter >= len(self.sites)*(self.decode_count+1) or self.sites[self.counter % len(self.sites)] != site:
            raise ValueError('C projection order or invocation count differs')
        if role == 0:
            self.context = dict(layer=layer)
        module = self.parameter(site, index, k, width)
        words = self.linear(site, module, a, self.step, index, self.mode)
        self.counter += 1
        return words

    def physical_output(self, step, site, rows):
        call = self.calls[step*len(self.sites)+self.sites.index(site)]
        pieces = [np.frombuffer(self.read(panel['expected_bf16']), dtype='<u2').reshape(panel['expected_bf16']['shape'])
                  for panel in call['panels']]
        result = np.concatenate(pieces, axis=1)
        if result.shape[0] != (128 if rows == 128 else 32):
            raise ValueError('producer physical reply rows differ')
        return result

    def project(self, _, layer, role, incoming, rows, k, width, output):
        try:
            if role not in (0, 1, 2, 7):
                raise ValueError('unfused projection callback at fused site')
            words = self.invoke(layer, role, pointer_words(incoming, rows, k).copy(), k, width)
            if role != 7:
                site = f'model.layers.{layer}.{native.PROJECTIONS[role]}'
                raw = self.layout.pack_bf16_tiles(self.physical_output(self.step, site, rows))
            else:
                raw = words.astype('<u2').tobytes()
            ct.memmove(output, raw, len(raw))
            return 0
        except BaseException as error:
            self.errors.append(repr(error))
            return -99

    def fused(self, _, layer, role, incoming, rows, k, width, operand, output):
        try:
            m = 128 if rows == 128 else 32
            if role == 3:
                producer = self.pending_attention
                if (producer is None or producer['address'] != incoming or producer['step'] != self.step
                        or producer['layer'] != layer or producer['consumed']
                        or ct.string_at(incoming, m*k) != producer['raw']):
                    raise ValueError('C o input differs from exact typed attention V2 producer')
                self.context['attention_v2'] = producer
                a = fp8_placeholder(self.layout, producer['raw'], rows, k, producer['scale'])
                if operand:
                    raise ValueError('D2 o callback must not stage a real residual')
                self.context['entry'] = np.zeros((rows,width),np.uint16)
            elif role in (4, 5):
                a = pointer_words(incoming, rows, k).copy()
                if role == 5:
                    actual_gate = self.layout.unpack_bf16_tiles(ct.string_at(operand, m*width*2), m, width)
                    expected = self.physical_output(self.step, f'model.layers.{layer}.mlp.gate_proj', rows)
                    if not np.array_equal(actual_gate, expected):
                        raise ValueError('C up gate operand differs from actual device producer')
                    self.context['gate'] = actual_gate
            elif role == 6:
                packed = ct.string_at(incoming, m*k)
                if packed != self.context['down_fp8']:
                    raise ValueError('C down FP8 input differs from assembled device producer')
                raw = self.layout.unpack_fp8_tiles(packed, m, k)
                decoded = torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn).float()
                words = (decoded*2.0**self.context['down_scale']).to(torch.bfloat16).view(torch.uint16).numpy()
                a = words[:rows].copy()
                if operand:
                    raise ValueError('D2 down callback must not stage a real residual')
                self.context['residual'] = np.zeros((rows,width),np.uint16)
            else:
                raise ValueError('unknown fused C callback role')
            words = self.invoke(layer, role, a, k, width)
            if role == 4:
                raw = self.layout.pack_bf16_tiles(self.physical_output(self.step,
                    f'model.layers.{layer}.mlp.gate_proj', rows))
            elif role == 5:
                raw = self.context['down_fp8']
            else:
                raw = words.astype('<u2').tobytes()
                if role == 3:
                    self.context['residual'] = words.copy()
            ct.memmove(output, raw, len(raw))
            return 0
        except BaseException as error:
            self.errors.append(repr(error))
            return -99

    def attention(self, _, layer, query, keys, values, rows, key_rows, output):
        try:
            if self.counter % len(self.sites) != layer*7+3 or key_rows != 128+self.step:
                raise ValueError('C attention order or cache length differs')
            m = 128 if rows == 128 else 32
            q = self.layout.unpack_bf16_tiles(ct.string_at(query, m*2048*2), m, 2048)[:rows]
            q = q.reshape(rows, 32, 64).transpose(1, 0, 2).copy()
            kv = [self.layout.unpack_bf16_tiles(ct.string_at(pointer, 160*256*2), 160, 256)[:key_rows]
                  .reshape(key_rows, 4, 64).transpose(1, 0, 2).copy() for pointer in (keys, values)]
            arrays = (q, *kv)
            signature = [array_identity(value) for value in arrays]
            key = self.step, layer
            if self.mode == 'replay':
                previous, raw = self.attention_saved[key]
                if signature != previous:
                    raise ValueError('C replay post-RoPE attention predecessors differ')
            else:
                stem = f'calls/attention-{self.step:02d}-{layer:02d}'
                descs = {name+'_bf16': self.save(stem+'-'+name+'.bin', array)
                         for name, array in zip(('q', 'k', 'v'), arrays)}
                config = dict(query_rows=rows, keys=key_rows, query_heads=32, kv_heads=4, head_dim=64,
                    mask='causal', query_start=key_rows-rows, scale=0.125,
                    q_exponent=fixed_exponent(q), k_exponent=fixed_exponent(kv[0]),
                    v_exponent=fixed_exponent(kv[1]), p_exponent=-8,
                    output_fp8=True, o_exponent=self.o_scales[key], packed_decode=True)
                normal = {name+'_tiled': self.save_bytes(stem+'-'+name+'-normal-tiled.bin',
                    ct.string_at(pointer, extent)) for name, pointer, extent in
                    (('q', query, m*2048*2), ('k', keys, ((key_rows+31)//32)*32*256*2),
                     ('v', values, ((key_rows+31)//32)*32*256*2))}
                record = self.attention_workers.run([dict(schema='atlas.tiny.attention_launch.v2',
                    command='attention_v2', root=str(self.out), out=str(self.out/stem), config=config,
                    step=self.step, layer=layer, site=f'model.layers.{layer}.self_attn.attention',
                    mode='functional', **descs, **normal)])[0]
                validate_attention_reply(record, config)
                for source in record['implementation_sources']:
                    if source not in self.records:
                        self.records.append(source)
                words = np.frombuffer(self.read(record['output_bf16']), dtype='<u2').reshape(32, rows, 64).copy()
                raw = self.read(record['output_tiled'])
                logical = fp8_placeholder(self.layout, raw, rows, 2048, self.o_scales[key])\
                    .reshape(rows, 32, 64).transpose(1, 0, 2)
                if not np.array_equal(logical, words):
                    raise ValueError('attention logical reply differs from exact physical output')
                record.update(step=self.step, layer=layer, index=len(self.attention_calls),
                    name=f'model.layers.{layer}.self_attn.attention', site=f'model.layers.{layer}.self_attn.attention')
                self.attention_calls.append(record)
                self.attention_saved[key] = signature, raw
            self.pending_attention = dict(address=output, raw=raw, scale=self.o_scales[key], step=self.step,
                layer=layer, index=self.step*(len(self.sites)//7)+layer, consumed=False)
            ct.memmove(output, raw, len(raw))
            return 0
        except BaseException as error:
            self.errors.append(repr(error))
            return -99

    def close(self):
        try:
            self.workers.close()
        finally:
            try:
                if self.attention_workers is not self.workers:
                    self.attention_workers.close()
            finally:
                if getattr(self, 'native_checker', None) is not None:
                    self.native_checker.close()


class CSession:
    def __init__(self, host, lib, layout, capture):
        self.lib, self.layout, self.capture, self.host = lib, layout, capture, host
        self.parameters = Parameters(host['model.embed_tokens.weight'].ctypes.data,
            (ct.c_void_p*22)(*[host[f'model.layers.{i}.input_layernorm.weight'].ctypes.data for i in range(22)]),
            (ct.c_void_p*22)(*[host[f'model.layers.{i}.post_attention_layernorm.weight'].ctypes.data
                              for i in range(22)]),
            host['model.norm.weight'].ctypes.data, host['rotary_cosine'].ctypes.data, host['rotary_sine'].ctypes.data,
            1e-5, ct.cast(capture.attention_callback, ct.c_void_p).value,
            ct.cast(capture.fused_callback, ct.c_void_p).value)
        self.keys = [np.zeros((22, 160*256), np.uint16) for _ in range(2)]
        self.values = [np.zeros_like(value) for value in self.keys]
        self.state = State((ct.c_void_p*2)(*[v.ctypes.data for v in self.keys]),
                           (ct.c_void_p*2)(*[v.ctypes.data for v in self.values]), 0, 0, 0)
        self.phase = Phase()
        self.workspace = np.zeros((layout.workspace_bytes+7)//8, np.uint64)
        self.logits = np.zeros((128, 32000), np.uint16)

    def require(self, status):
        if status or self.capture.errors:
            raise ValueError(f'C graph failed: status={status}, errors={self.capture.errors}')

    def run(self, label, layer_count, block_only, decode_count):
        trace, previous = [], {}
        capture = self.capture
        for step in range(decode_count+1):
            capture.step = step
            tokens = (self.host['prefill_tokens'].astype(np.uint32) if step == 0
                      else np.array([self.state.next_token], np.uint32))
            rows = len(tokens)
            capture.begin_session(label)
            common = (ct.byref(self.parameters), ct.byref(self.state))
            tail = self.workspace.ctypes.data, self.workspace.nbytes
            self.require(self.lib.tiny_full_model_begin(*common, ct.byref(self.phase), tokens.ctypes.data, rows, *tail))
            for layer in range(layer_count):
                self.require(self.lib.tiny_full_model_layer(*common, ct.byref(self.phase), layer,
                    capture.project_callback, None, *tail))
            row = dict(step=step, input_tokens=[tokens.tolist()], cache=[])
            if block_only:
                offset = self.layout.hidden_offset[layer_count%2]
                words = pointer_words(self.workspace.ctypes.data+offset, rows, 2048).copy()
                row['hidden'] = array_identity(words)
                if label == 'target':
                    row['hidden_payload'] = capture.save(f'calls/hidden-{step:02d}.bin', words)
                bank, position = 1-self.state.bank, rows
            else:
                self.require(self.lib.tiny_full_model_finish(*common, ct.byref(self.phase),
                    capture.project_callback, None, *tail, self.logits.ctypes.data, self.logits.size))
                logits = (self.logits[rows-1].astype(np.uint32)<<16).view(np.float32)
                row.update(next_token=int(self.state.next_token), logits=array_identity(logits))
                if label == 'target':
                    row['logits_payload'] = capture.save(f'calls/logits-{step:02d}.bin', logits)
                    if step == 0:
                        all_logits = (self.logits[:rows].astype(np.uint32)<<16).view(np.float32)
                        row['all_logits_payload'] = capture.save('calls/logits-prefill-all.bin', all_logits)
                bank, position = self.state.bank, self.state.position
            for layer in range(layer_count):
                for field, banks in (('keys', self.keys), ('values', self.values)):
                    words = capture.layout.unpack_bf16_tiles(banks[bank][layer].tobytes(), 160, 256)[:position]
                    words = words.reshape(position, 4, 64).transpose(1, 0, 2)[None].copy()
                    key = layer, field
                    if step and not np.array_equal(words[:, :, :-1], previous[key]):
                        raise ValueError('C graph KV prefix changed during decode')
                    previous[key] = words
                    desc = (capture.save(f'calls/cache-{step:02d}-{layer:02d}-{field}.bin', words)
                            if label == 'target' else array_identity(words))
                    row['cache'].append(dict(layer=layer, field=field, **desc))
            trace.append(row)
            print(json.dumps(dict(milestone=label, step=step, next_token=row.get('next_token'))), flush=True)
        return trace


def run(args):
    torch.set_num_threads(1)
    if (not 1 <= args.layer_count <= 22 or not 0 <= args.decode_count <= 2
            or args.block_only and args.decode_count or args.layer_count != 22 and not args.block_only):
        raise ValueError('partial C capture requires block-only prefill; full supports zero through two decode steps')
    args.out = checked(args.out)
    if not args.out.is_relative_to(ROOT) or args.out.exists():
        raise ValueError('fresh output beneath D required')
    args.out.mkdir(parents=True)
    inputs, projection = None, None
    if (args.attention_worker_script is None
            or checked(args.worker_script) == checked(args.attention_worker_script)):
        raise ValueError('V2 requires isolated admitted native projections and a separate original47 attention worker')
    report = dict(status='FAILED', schema='atlas.tiny_fused_c_capture.v1', fusions=True, attention_version=2,
                  fixed_attention_scale_contract_sha256=ATTENTION_SCALE_CONTRACT_SHA, residual_adds=False,
                  capture_residual_surrogate='residual kernels with zero operands; final plain programs require original47',
                  host_reference='compiled_runtime_c', quality_qualified=False, timing_qualified=False)
    started = time.monotonic()
    records = []
    def save(relative, value):
        value = np.ascontiguousarray(value)
        path = args.out/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.tobytes())
        return dict(file=relative, **array_identity(value), bytes=value.nbytes)
    try:
        inputs = Inputs(args.source_capture, args.scale_contract)
        python_before = python_sources(inputs)
        report['worker_interpreter'] = worker_interpreter(args.worker_python, inputs)
        if getattr(args, 'attention_worker_python', None) is not None:
            report['attention_worker_interpreter'] = worker_interpreter(args.attention_worker_python, inputs)
        (lib, layout), build = compile_host(inputs, args.out, args.cc)
        report.update(build)
        report['host_runtime_libraries'] = host_libraries(inputs)
        host, host_parameters = {}, {}
        for index, (name, desc) in enumerate(inputs.data['host_parameters'].items()):
            host[name] = inputs.read(desc)
            host_parameters[name] = save(f'parameters/host-{index:03d}.bin', host[name])
        host_before = {name: array_identity(value) for name, value in host.items()}
        projection = CProjection(args, inputs, save, records)
        projection.decode_count = args.decode_count
        target = CSession(host, lib, layout, projection).run('target', args.layer_count,
                                                           args.block_only, args.decode_count)
        if projection.counter != len(projection.sites)*(args.decode_count+1):
            raise ValueError('incomplete target C graph invocation ledger')
        projection.counter, projection.mode = 0, 'replay'
        replay = CSession(host, lib, layout, projection).run('replay', args.layer_count,
                                                           args.block_only, args.decode_count)
        normalized = json.loads(json.dumps(target))
        for row in normalized:
            row.pop('logits_payload', None)
            row.pop('all_logits_payload', None)
            row.pop('hidden_payload', None)
            for entry in row['cache']:
                entry.pop('file', None)
                entry.pop('bytes', None)
        if replay != normalized or projection.counter != len(projection.sites)*(args.decode_count+1):
            raise ValueError('fresh-state C saved-reply replay changed')
        if {name: array_identity(value) for name, value in host.items()} != host_before:
            raise ValueError('native C changed host parameters or tokens')
        if worker_interpreter(args.worker_python, inputs) != report['worker_interpreter']:
            raise ValueError('worker interpreter alias binding changed')
        if (getattr(args, 'attention_worker_python', None) is not None
                and worker_interpreter(args.attention_worker_python, inputs) != report['attention_worker_interpreter']):
            raise ValueError('attention interpreter alias binding changed')
        if python_sources(inputs) != python_before:
            raise ValueError('capture Python source closure changed')
        for record in records:
            inputs.pin(record['path'], record['sha256'])
        inputs.bookends()
        sources = {record['path']: record for record in records}
        for path, value in inputs.pins.items():
            sources[path] = dict(path=path, sha256=value, bytes=Path(path).stat().st_size)
        launches = [panel['launch'] for call in projection.calls for panel in call['panels']]
        launches.extend(projection.attention_calls)
        recovered = [item['original_reply_recovery'] for item in launches
                     if item.get('original_reply_recovery', {}).get('fresh_instruction_execution') is False]
        if recovered:
            report['original47_recovery'] = dict(reused_launches=len(recovered),
                fresh_instruction_launches=len(launches)-len(recovered), witnesses=recovered,
                exact_request_matching_required=True, interrupted_run_preserved=True)
        native_count = sum(item['model_commit'] != MODEL_PIN for item in launches)
        expected_native = (12*args.layer_count + (0 if args.block_only else 16))*(args.decode_count+1)
        expected_attention = args.layer_count*(args.decode_count+1)
        if (native_count not in (0, expected_native) or len(projection.attention_calls) != expected_attention
                or len(launches) != expected_native+expected_attention):
            raise ValueError('V2 graph did not route every projection to native70 and every attention to original47')
        report.update(status='PASS_TINY_FUSED_C_BLOCK_CAPTURE_REPLAY' if args.block_only else
            'PASS_FULL_TINY_FUSED_C_CAPTURE_REPLAY', host_c_sources_verified=True, source_bookends_verified=True,
            target_replay_exact=True, cache_prefixes_exact=True, atlas_instruction_execution=True,
            layer_count=args.layer_count, block_only=args.block_only, include_lm_head=not args.block_only,
            decode_count=args.decode_count, greedy_feedback_actual=not args.block_only,
            all22_cache_prefixes_exact=args.layer_count == 22, parameters=projection.parameters,
            host_parameters=host_parameters, calls=projection.calls, attention_calls=projection.attention_calls,
            projection_calls=len(projection.calls), target_trace=target, source_records=list(sources.values()),
            host_source_sha256=inputs.pins, projection_reference_model=MODEL_PIN,
            limitations=['Native host C semantics are bound; original Torch restoration is not claimed.',
                         'RV64 captured-reply replay remains a separate gate.'])
        report['capture_execution_models'] = dict(original47_launches=len(launches)-native_count,
                                                  native70_launches=native_count)
        if native_count:
            report.update(status='PASS_TINY_FUSED_C_NATIVE_BLOCK_CAPTURE_REPLAY_REQUIRES_ORIGINAL47_BACKFILL'
                if args.block_only else 'PASS_FULL_TINY_FUSED_C_NATIVE_CAPTURE_REPLAY_REQUIRES_ORIGINAL47_BACKFILL',
                requires_original_backfill=True, final_numerics_qualified=False,
                projection_execution_model='70e2e7ceb81dac86210c8b3cca54024909c57d14',
                attention_execution_model=MODEL_PIN, native_proofs=projection.native_proofs)
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        try:
            if projection is not None:
                projection.close()
        finally:
            report['wall_seconds'] = time.monotonic()-started
            (args.out/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-capture', 'scale-contract', 'attention-scale-contract', 'attention-model-root', 'atlas-layout', 'worker-script', 'cc', 'out'):
        parser.add_argument('--'+name, type=checked, required=True)
    parser.add_argument('--worker-python', type=Path, required=True)
    parser.add_argument('--atlas-layout-sha256', required=True)
    parser.add_argument('--attention-worker-python', type=Path)
    parser.add_argument('--attention-worker-script', type=checked)
    parser.add_argument('--native-proof-script', type=checked)
    parser.add_argument('--native-proof-sha256')
    parser.add_argument('--layer-count', type=int, default=22)
    parser.add_argument('--decode-count', type=int, default=2)
    parser.add_argument('--block-only', action='store_true')
    args = parser.parse_args()
    if bool(args.attention_worker_python) != bool(args.attention_worker_script):
        parser.error('both separate attention worker arguments are required together')
    if bool(args.native_proof_script) != bool(args.native_proof_sha256):
        parser.error('native provenance script and explicit SHA256 are required together')
    run(args)


if __name__ == '__main__':
    main()
