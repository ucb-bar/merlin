"""Run C block arithmetic with explicitly injected archived model outputs."""

from merlin.common.jsonio import write_pretty_json
from merlin.common.digest import sha256_bytes
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from source_relocation import physical_path as checked
import numpy as np

ROOT = Path(__file__).absolute().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--capture-receipt', type=Path, required=True)
parser.add_argument('--capture-receipt-sha256', required=True)
parser.add_argument('--rope-tables-receipt', type=Path, required=True)
parser.add_argument('--rope-tables-receipt-sha256', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
for value in (args.capture_receipt_sha256, args.rope_tables_receipt_sha256):
    if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Require an explicit lowercase SHA256')
OUT = checked(args.output)
SOURCE = ROOT
sys.path.insert(0, str(ROOT))
from workload_replay_admission import admit_executable
ROLES =('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj',
    'mlp.down_proj')
FIELDS =('normalized', 'query', 'key', 'value', 'rotary_query', 'rotary_key', 'attention', 'projected', 'residual', 'postnorm',
    'gate', 'up',
    'mlp_input', 'down', 'output', 'scores', 'attention_workspace', 'probabilities', 'mask', 'groups', 'positions')
class Parameters(ctypes.Structure):
    _fields_ =[('input_norm_weight', ctypes.c_void_p), ('post_norm_weight', ctypes.c_void_p), ('input_epsilon', ctypes.c_float), (
        'post_epsilon', ctypes.c_float), ('cosine', ctypes.c_void_p), ('sine', ctypes.c_void_p), ('first_table_position',
        ctypes.c_int64), ('table_rows', ctypes.c_size_t)]
class Frame(ctypes.Structure):
    _fields_ =[(name, ctypes.c_void_p) for name in FIELDS]
class State(ctypes.Structure):
    _fields_ =[(name, ctypes.c_uint) for name in ('started', 'completed_linears', 'last_role', 'failed', 'complete',
        'cache_valid')] +[(
        'callback_code', ctypes.c_int), ('result', ctypes.c_int)]
Callback = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_void_p, ctypes.c_size_t)
pins =[]
def pin(path, expected=None):
    path = checked(path)
    item = {'path': str(path), 'sha256': sha256_bytes(path.read_bytes())}
    if expected is not None and item['sha256'] != expected:
        raise ValueError('Changed pinned input')
    pins.append(item)
    return item
report = {'status': 'failed', 'command': [sys.executable, *sys.argv], 'cwd': os.getcwd(),
        'scope': (
            'C selected-layer block composition with seven archived model outputs injected by '
            'Python callbacks; no device dispatch, additional model/RTL ticks, full C policy, '
            'raw runtime, ELF or ISA execution'
        ),
        'callback_source': 'CPU injection of authenticated archived model outputs',
        'checks': [], 'reference_or_tolerance_changed': False}
outputs = {}
OUT.mkdir(exist_ok=False)
try:
    pin(__file__)
    (OUT / 'executed-source.py').write_bytes(Path(__file__).read_bytes())
    report['executed_source_snapshot'] = pin(OUT / 'executed-source.py')
    pin(ROOT / 'workload_replay_admission.py')
    for name in ('smol_vlm_block_prototype.c', 'smol_vlm_block_prototype.h'):
        pin(ROOT / name)
        (OUT / name).write_bytes((ROOT / name).read_bytes())
    for stem in ('smol_host_ops', 'smol_host_rope', 'smol_host_softmax', 'smol_host_attention', 'smol_host_controls'):
        for suffix in ('.c', '.h'):
            pin(SOURCE /(stem + suffix))
            (OUT /(stem + suffix)).write_bytes((SOURCE /(stem + suffix)).read_bytes())
    for name in ('smol_host_bits.h', 'smol_host_sleef_LICENSE.txt'):
        pin(SOURCE / name)
        (OUT / name).write_bytes((SOURCE / name).read_bytes())
    capture_path = checked(args.capture_receipt)
    pin(capture_path, args.capture_receipt_sha256)
    capture = json.loads(capture_path.read_text())
    if capture['status'] != 'passed_source_capture_with_explicit_archived_output_injection_only':
        raise ValueError('Unadmitted source capture')
    pin(capture['output']['path'], capture['output']['sha256'])
    with np.load(capture['output']['path'], allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    expected_shapes = {'input': (177, 960), 'input_norm_weight': (960,), 'post_norm_weight': (960,),
                     'expected_output': (177, 960), 'expected_key_cache': (177, 5, 64), 'expected_value_cache': (177, 5, 64)}
    for role in ROLES:
        expected_shapes[role + '_expected_input'] =(177, 2560 if role == 'mlp.down_proj' else 960)
        expected_shapes[role + '_injected_output'] =(177, 320 if role in ('self_attn.k_proj',
            'self_attn.v_proj') else 2560 if role in ('mlp.gate_proj', 'mlp.up_proj') else 960)
    for name, shape in expected_shapes.items():
        if data[name].dtype != np.uint16 or data[name].shape != shape:
            raise ValueError('Invalid fixed composition BF16 input ABI: ' + name)
    for name in ('pad', 'attention_boundaries'):
        if data[name].dtype != np.uint8 or data[name].shape !=(177,):
            raise ValueError('Invalid fixed composition Boolean input ABI: ' + name)
    table_path = checked(args.rope_tables_receipt)
    pin(table_path, args.rope_tables_receipt_sha256)
    table = json.loads(table_path.read_text())
    pin(table_path.parent / 'tables.npz', table['outputs']['tables.npz'])
    with np.load(table_path.parent / 'tables.npz', allow_pickle=False) as archive:
        cosine, sine = archive['cosine'].copy(), archive['sine'].copy()
    if cosine.dtype != np.float32 or sine.dtype != np.float32 or cosine.shape != sine.shape or cosine.ndim != 2 or cosine.shape[
        1] != 32 or cosine.shape[0] == 0:
        raise ValueError('Invalid source-native rotary table ABI')
    if type(table['first_position']) is not int or not -(2 ** 63) <= table['first_position'] < 2 ** 63:
        raise ValueError('Invalid rotary table lower position')
    report['compiler'] = admit_executable(shutil.which('cc'))
    report['interpreter'] = admit_executable(sys.executable)
    for binding in (report['compiler'], report['interpreter']):
        item = binding['physical_binary']
        pin(item['path'], item['sha256'])
    command =['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-ffp-contract=off', '-fno-fast-math', '-fPIC',
        '-shared',
        str(OUT / 'smol_vlm_block_prototype.c'), *[str(OUT /(stem + '.c')) for stem in ('smol_host_ops', 'smol_host_rope',
        'smol_host_softmax', 'smol_host_attention', 'smol_host_controls')], '-lm', '-o', str(OUT / 'block.so')]
    report['build_command'] = command
    build = subprocess.run(command, capture_output=True, text=True)
    (OUT / 'build.log').write_text(build.stdout + build.stderr)
    if build.returncode:
        raise ValueError('C block composition compilation failed')
    report['compiled_library'] = pin(OUT / 'block.so')
    report['dependencies'] = subprocess.check_output(['ldd', str(OUT / 'block.so')], text=True)
    report['native_library_pins'] =[]
    for line in report['dependencies'].splitlines():
        if '=>' in line and line.split('=>', 1)[0].strip() in ('libm.so.6', 'libc.so.6'):
            binding = admit_executable(line.split('=>', 1)[1].strip().split()[0])
            report['native_library_pins'].append(binding)
            item = binding['physical_binary']
            pin(item['path'], item['sha256'])
    owner = ctypes.CDLL(str(OUT / 'block.so'))
    function = owner.smol_vlm_block_prototype
    function.argtypes =[ctypes.c_void_p] * 3 +[ctypes.POINTER(Parameters), ctypes.POINTER(Frame), ctypes.POINTER(State), Callback,
        ctypes.c_void_p]
    pointer = lambda a: a.ctypes.data
    parameters = Parameters(pointer(data['input_norm_weight']), pointer(data['post_norm_weight']), capture['input_epsilon'],
        capture[
        'post_epsilon'], pointer(cosine), pointer(sine), table['first_position'], cosine.shape[0])
    def allocate():
        buffers = {}
        for name in FIELDS:
            shape, dtype =(177, 960), np.uint16
            if name in ('key', 'value', 'rotary_key'):
                shape =(177, 320)
            elif name in ('gate', 'up', 'mlp_input'):
                shape =(177, 2560)
            elif name == 'scores':
                shape, dtype =(15, 177, 177), np.float32
            elif name == 'probabilities':
                shape =(15, 177, 177)
            elif name == 'attention_workspace':
                shape, dtype = (177,), np.float32
            elif name == 'mask':
                shape, dtype =(177, 177), np.uint8
            elif name in ('groups', 'positions'):
                shape, dtype = (177,), np.int64
            buffers[name] = np.full(shape, 0x1234 if dtype == np.uint16 else 7, dtype)
        return buffers, Frame(*[pointer(buffers[name]) for name in FIELDS])
    def callback(events, abort=None):
        @ Callback
        def invoke(context, index, address, rows, width, destination, columns):
            try:
                if index >= len(ROLES) or index != len(events):
                    return 90
                role = ROLES[index]
                expected = data[role + '_expected_input']
                actual = np.ctypeslib.as_array(ctypes.cast(address, ctypes.POINTER(ctypes.c_uint16)), shape=(rows * width,
                    )).reshape(
                    rows, width)
                mismatch = int(np.count_nonzero(actual != expected)) if actual.shape == expected.shape else - 1
                event = {'role': role, 'shape': [rows, width, columns], 'input_mismatches': mismatch,
                    'callback_source': 'archived_model_output_injection'}
                events.append(event)
                if mismatch:
                    return 91
                if index == abort:
                    return 42
                target = data[role + '_injected_output']
                if target.shape !=(rows, columns):
                    return 92
                ctypes.memmove(destination, target.ctypes.data, target.nbytes)
                return 0
            except Exception as error:
                events.append({'error': str(error)})
                return 99
        return invoke
    input_before = {name: data[name].copy() for name in ('input', 'pad', 'attention_boundaries', 'input_norm_weight',
        'post_norm_weight')}
    buffers, frame = allocate()
    state = State()
    events =[]
    inject = callback(events)
    code = function(pointer(data['input']), pointer(data['pad']), pointer(data['attention_boundaries']), ctypes.byref(parameters),
        ctypes.byref(frame), ctypes.byref(state), inject, None)
    output_mismatch = int(np.count_nonzero(buffers['output'] != data['expected_output']))
    key_mismatch = int(np.count_nonzero(buffers['rotary_key'].reshape(177, 5, 64) != data['expected_key_cache']))
    value_mismatch = int(np.count_nonzero(buffers['value'].reshape(177, 5, 64) != data['expected_value_cache']))
    unchanged = all(np.array_equal(data[name], before) for name, before in input_before.items())
    report['checks'].append({'case': 'actual_layer1_composition_with_injected_callbacks', 'returncode': code, 'state': {
        name: getattr(
        state, name) for name, _ in State._fields_}, 'callbacks': events, 'output_words': buffers['output'].size,
        'output_mismatches': output_mismatch, 'resident_key_mismatches': key_mismatch,
        'resident_value_mismatches': value_mismatch,
        'inputs_unchanged': unchanged})
    outputs.update(buffers)
    if (code or output_mismatch or key_mismatch or value_mismatch or not unchanged
            or state.completed_linears != 7 or not state.complete or not state.cache_valid):
        raise ValueError('C composition does not match unchanged source-native block with identical injected outputs')
    for abort in (0, 3, 6):
        buffers, frame = allocate()
        state = State()
        events =[]
        inject = callback(events, abort)
        code = function(pointer(data['input']), pointer(data['pad']), pointer(data['attention_boundaries']), ctypes.byref(
            parameters),
            ctypes.byref(frame), ctypes.byref(state), inject, None)
        calls = len(events)
        retry = function(pointer(data['input']), pointer(data['pad']), pointer(data['attention_boundaries']), ctypes.byref(
            parameters),
            ctypes.byref(frame), ctypes.byref(state), inject, None)
        passed =(code == 4 and retry == 2 and len(
            events) == calls == abort + 1 and state.failed == 1 and state.completed_linears == abort
            and state.callback_code == 42 and not state.complete and not state.cache_valid and np.all(
            buffers['output'] == 0x1234))
        report['checks'].append({'case': 'failure_latches_at_role_' + str(abort), 'passed': bool(passed), 'returncode': code,
            'retry_returncode': retry, 'callbacks_before_retry': calls, 'callbacks_after_retry': len(events)})
        if not passed:
            raise ValueError('C callback failure ownership/order refusal failed')
    for alias in ('output_to_input', 'query_to_normalized'):
        buffers, frame = allocate()
        state = State()
        events =[]
        inject = callback(events)
        if alias == 'output_to_input':
            frame.output = pointer(data['input'])
        else:
            frame.query = frame.normalized
        code = function(pointer(data['input']), pointer(data['pad']), pointer(data['attention_boundaries']), ctypes.byref(
            parameters),
            ctypes.byref(frame), ctypes.byref(state), inject, None)
        passed = code == 1 and not events and not state.started
        report['checks'].append({'case': alias, 'returncode': code, 'passed': passed})
        if not passed:
            raise ValueError('C composition alias refusal failed')
    report['status'] = 'passed_C_block_composition_with_archived_output_injection_only'
except Exception as error:
    report['error'] = {'type': type(error).__name__, 'message': str(error)}
    report[
        'next_action'] = (
            'Stop before runtime/ELF; repair the first C arithmetic/callback/ownership mismatch'
            ' without changing source reference or tolerance'
        )
finally:
    np.savez(OUT / 'outputs.npz', **outputs)
    report['output'] = pin(OUT / 'outputs.npz')
    for item in tuple(pins):
        pin(item['path'], item['sha256'])
    report['source_input_toolchain_pins'] = pins
    write_pretty_json(OUT / 'receipt.json', report, sort_keys=False)
print(json.dumps({'status': report['status'], 'checks': report['checks'], 'error': report.get('error')}), flush=True)
raise SystemExit(0 if report['status'] == 'passed_C_block_composition_with_archived_output_injection_only' else 1)
