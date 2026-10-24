"""Qualify shared native C primitives on source-bound TinyLlama host values.

This gate compiles an explicit immutable shared-source snapshot. It does not
implement C attention, a complete C host graph, a device runtime, or an ELF.
"""

from merlin.common.digest import sha256_bytes
import argparse
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from source_relocation import physical_path as safe
import zipfile
import numpy as np


def identity(value, expected=None):
    path = safe(value)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    digest = h.hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('identity mismatch ' + str(path))
    return {'path': str(path), 'sha256': digest}


def arrays(path, keys):
    path = safe(path)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = {x.filename for x in members}
        if len(members) != len(keys) or names != {k + '.npy' for k in keys}:
            raise ValueError('archive members differ')
        if any(x.flag_bits & 1 or (x.external_attr >> 16) & 0o170000 == 0o120000 or x.file_size <= 0 for x in members):
            raise ValueError('archive metadata refused')
        if sum(x.file_size for x in members) > 64 * 1024 * 1024:
            raise ValueError('bounded archive size exceeded')
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in keys}


def bits(value, torch):
    return value.detach().contiguous().view(torch.uint16).numpy().copy()
def tensor(words, torch):
    return torch.from_numpy(np.ascontiguousarray(words)).view(torch.bfloat16)
def pointer(array):
    return array.ctypes.data_as(ctypes.c_void_p)
def words(value):
    if value.dtype != np.float32 or np.any(value.view(np.uint32) & 65535):
        raise ValueError('not exact BF16 views')
    return (value.view(np.uint32) >> 16).astype(np.uint16)


def gate(a, r):
    import torch
    from transformers.models.llama import modeling_llama
    from safetensors import safe_open
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    if a.cpu not in os.sched_getaffinity(0):
        raise ValueError('CPU outside allowed affinity')
    os.sched_setaffinity(0, {a.cpu})
    r['cpu_affinity'] = sorted(os.sched_getaffinity(0))
    records =[]
    r['source_records'] = records
    def pin(p, h=None):
        item = identity(p, h)
        records.append(item)
        return item
    pin(a.native_gate, a.native_gate_sha256)
    native = json.loads(a.native_gate.read_text())
    pin(a.block_gate, a.block_gate_sha256)
    block = json.loads(a.block_gate.read_text())
    pin(a.decode_prepared, a.decode_prepared_sha256)
    decode = json.loads(a.decode_prepared.read_text())
    if native['status'] != 'passed_frozen_native_BF16_session' or not native['source_bookends_verified']:
        raise ValueError('native gate refused')
    if block['status'] != 'passed_exact_target_model_block_with_informational_native_quality' or not block[
        'source_bookends_verified']:
        raise ValueError('prefill model gate refused')
    if decode['status'] != 'prepared_live_selected_block_firstdecode_arithmetic_only' or not decode[
        'no_new_model_or_RTL_execution']:
        raise ValueError('decode preparation refused')
    for name in ('workload_replay_admission.py', 'capture_tinyllama_native.py'):
        pin(a.support_dir / name)
    sys.path.insert(0, str(a.support_dir))
    from workload_replay_admission import admit_executable
    compiler = admit_executable(shutil.which('cc'))
    interpreter = admit_executable(sys.executable)
    pin(a.shared_source / 'receipt.json', a.shared_source_sha256)
    source_snapshot = json.loads((a.shared_source / 'receipt.json').read_text())
    for item in source_snapshot['files']:
        pin(item['path'], item['sha256'])
        p = a.out / Path(item['path']).name
        p.write_bytes(safe(item['path']).read_bytes())
        pin(p, item['sha256'])
    pin(__file__)
    command =[compiler['entry'], '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic', '-ffp-contract=off',
        '-fno-fast-math',
        '-fPIC', '-shared', str(a.out / 'smol_host_ops.c'), '-lm', '-o', str(a.out / 'host_ops.so')]
    r.update(build_command=command, compiler_identity=compiler, interpreter_identity=interpreter,
        compiler_version=subprocess.check_output([compiler['entry'], '--version'], text=True))
    build = subprocess.run(command, cwd=a.out, capture_output=True, text=True, timeout=120)
    (a.out / 'build.log').write_text(build.stdout + build.stderr)
    r['build_returncode'] = build.returncode
    if build.returncode:
        raise RuntimeError('strict native C build failed')
    deps = subprocess.check_output(['ldd', str(a.out / 'host_ops.so')], text=True)
    r['linked_dependencies'] = deps
    dep_ids =[]
    for line in deps.splitlines():
        if '=>' in line:
            name, tail = line.split('=>', 1)
            if name.strip() in ('libc.so.6', 'libm.so.6'):
                dep_ids.append({'name': name.strip(), 'binding': admit_executable(tail.split()[0])})
    if {d['name'] for d in dep_ids} != {'libc.so.6', 'libm.so.6'}:
        raise ValueError('native library identities absent')
    r['native_dependencies'] = dep_ids
    library = ctypes.CDLL(str(a.out / 'host_ops.so'))
    residual = library.smol_host_residual_bf16
    silu = library.smol_host_silu_mul_bf16
    norm = library.smol_host_rmsnorm_bf16
    for function in (residual, silu):
        function.argtypes =[ctypes.c_void_p] * 3 +[ctypes.c_size_t]
        function.restype = ctypes.c_int
    norm.argtypes =[ctypes.c_void_p] * 2 +[ctypes.c_float, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    norm.restype = ctypes.c_int
    source_by_path = {x['path']: x['sha256'] for x in native['source_records']}
    for p in (Path(modeling_llama.__file__), Path(inspect.getfile(torch.nn.Linear)), Path(
        torch.__file__).parent / 'lib/libtorch_cpu.so'):
        pin(p, source_by_path[str(p)])
    weight_record = next(x for x in native['source_records'] if Path(x['path']).name == 'model.safetensors')
    pin(weight_record['path'], weight_record['sha256'])
    config_record = next(x for x in native['source_records'] if Path(x['path']).name == 'config.json')
    pin(config_record['path'], config_record['sha256'])
    config = json.loads(safe(config_record['path']).read_text())
    epsilon = config['rms_norm_eps']
    for item in [native['session']]:
        pin(item['path'], item['sha256'])
    session = arrays(native['session']['path'], {'logits_bf16', 'tokens', 'entry_bf16'})
    prefill_entry = np.ascontiguousarray(session['entry_bf16'][0])
    tokens = session['tokens']
    with safe_open(str(safe(weight_record['path'])), framework='pt', device='cpu') as checkpoint:
        input_weight = checkpoint.get_tensor('model.layers.0.input_layernorm.weight')
        post_weight = checkpoint.get_tensor('model.layers.0.post_attention_layernorm.weight')
        decode_entry = bits(checkpoint.get_slice('model.embed_tokens.weight')[int(tokens[0, 128]):int(tokens[0, 128]) + 1], torch)
    for name, value in [('model.layers.0.input_layernorm.weight', input_weight), (
        'model.layers.0.post_attention_layernorm.weight',
        post_weight)]:
        expected = native['full_parameters'][name]
        if value.dtype != torch.bfloat16 or list(value.shape) != expected['shape'] or sha256_bytes(bits(value, torch).tobytes(
            )) != expected['sha256']:
            raise ValueError('norm weight differs')
    iw = bits(input_weight, torch)
    pw = bits(post_weight, torch)
    calls = {c['tag']: c for c in block['target_programs']}
    def child(tag):
        item = calls[tag]['child']['receipt']
        pin(item['path'], item['sha256'])
        folder = Path(item['path']).parent
        c = json.loads(safe(item['path']).read_text())
        if c['status'] != 'passed_model_only' or not c['runs'] or not c['source_bookends']['unchanged']:
            raise ValueError('model source refused')
        if not all(all(x[k] is True for k in ('all_output_bytes_exact', 'input_and_guards_preserved',
            'drained_before_halt_observation')) for x in c['runs']):
            raise ValueError('model numerical gate refused')
        for name in ('source.npz', 'result.npz'):
            pin(folder / name, c['artifacts'][name])
        source = arrays(folder / 'source.npz', {'A', 'B', 'native_C'})
        result = arrays(folder / 'result.npz', {'actual_bits', 'expected_bits', 'target_C'})
        if not np.array_equal(result['actual_bits'], result['expected_bits']):
            raise ValueError('model result disagrees')
        return source, result['actual_bits']
    prefill_input = words(child('0-self_attn.q_proj-0')[0]['A'])
    _, prefill_o = child('3-self_attn.o_proj-0')
    prefill_gate = np.concatenate([child('4-mlp.gate_proj-' + str(i))[1] for i in range(3)], axis=1)
    prefill_up = np.concatenate([child('5-mlp.up_proj-' + str(i))[1] for i in range(3)], axis=1)
    prefill_post = words(child('4-mlp.gate_proj-0')[0]['A'])
    prefill_mlp = words(child('6-mlp.down_proj-0')[0]['A'])
    prefill_down = np.concatenate([child('6-mlp.down_proj-' + str(i))[1] for i in range(2)], axis=1)
    prepared = {x['tag']: x for x in decode['prepared_decode_panels']}
    def prepared_panel(tag):
        panel = prepared[tag]
        for item in panel['artifacts'].values():
            pin(item['path'], item['sha256'])
        src = arrays(panel['artifacts']['operands.npz']['path'], {'A', 'B', 'native_C'})
        target = np.load(safe(panel['artifacts']['expected-logical-u16.npy']['path']), allow_pickle=False)
        return src, target
    decode_input = words(prepared_panel('0-self_attn.q_proj-0')[0]['A'][:1])
    _, decode_o = prepared_panel('3-self_attn.o_proj-0')
    decode_gate = np.concatenate([prepared_panel('4-mlp.gate_proj-' + str(i))[1] for i in range(3)], axis=1)
    decode_up = np.concatenate([prepared_panel('5-mlp.up_proj-' + str(i))[1] for i in range(3)], axis=1)
    decode_post = words(prepared_panel('4-mlp.gate_proj-0')[0]['A'][:1])
    decode_mlp = words(prepared_panel('6-mlp.down_proj-0')[0]['A'][:1])
    decode_down = np.concatenate([prepared_panel('6-mlp.down_proj-' + str(i))[1] for i in range(2)], axis=1)
    references = {}
    r['checks'] = {}
    def check(name, actual, expected):
        mismatches = int(np.count_nonzero(actual != expected))
        r['checks'][name] = {'words': int(actual.size), 'mismatches': mismatches}
        references[name + '_actual'] = actual
        references[name + '_expected'] = expected
        if mismatches:
            r['first_mismatches'] = {'check': name, 'positions': np.argwhere(actual != expected)[:16].tolist()}
            np.savez(a.out / 'failure-arrays.npz', actual=actual, expected=expected)
            r['failure_arrays'] = identity(a.out / 'failure-arrays.npz')
            raise ValueError('C/native parity failed ' + name)
    def c_binary(function, left, right):
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        output = np.empty_like(left)
        if function(pointer(left), pointer(right), pointer(output), output.size):
            raise ValueError('C primitive refused input')
        return output
    def c_norm(inputs, weight):
        output = np.empty_like(inputs)
        if norm(pointer(inputs), pointer(weight), ctypes.c_float(epsilon), pointer(output), *inputs.shape):
            raise ValueError('C RMSNorm refused input')
        return output
    def t_norm(inputs, weight):
        module = modeling_llama.LlamaRMSNorm(inputs.shape[1], eps=epsilon)
        module.weight = torch.nn.Parameter(tensor(weight, torch), requires_grad=False)
        return bits(module(tensor(inputs, torch)), torch)
    for label, entry, normalized, o, g, u, post, mlp, down in [('prefill_model_reply', prefill_entry, prefill_input, prefill_o,
        prefill_gate,
        prefill_up, prefill_post, prefill_mlp, prefill_down), ('firstdecode_arithmetic_prepared_reply', decode_entry,
        decode_input,
        decode_o, decode_gate, decode_up, decode_post, decode_mlp, decode_down)]:
        native_input = t_norm(entry, iw)
        check(label + '_input_norm_reference_binding', native_input, normalized)
        check(label + '_input_norm', c_norm(entry, iw), native_input)
        native_residual = bits(tensor(entry, torch) + tensor(o, torch), torch)
        actual_residual = c_binary(residual, entry, o)
        check(label + '_residual1', actual_residual, native_residual)
        native_post = t_norm(native_residual, pw)
        check(label + '_post_norm_reference_binding', native_post, post)
        check(label + '_post_norm', c_norm(actual_residual, pw), native_post)
        native_mlp = bits(torch.nn.functional.silu(tensor(g, torch)) * tensor(u, torch), torch)
        check(label + '_MLP_reference_binding', native_mlp, mlp)
        check(label + '_SiLU_mul', c_binary(silu, g, u), native_mlp)
        check(label + '_residual2', c_binary(residual, actual_residual, down), bits(tensor(native_residual, torch) + tensor(down,
            torch),
            torch))
    rng = np.random.default_rng(20482026)
    for rows in (1, 17, 128):
        x = bits(torch.from_numpy(rng.standard_normal((rows, 2048), dtype=np.float32)).to(torch.bfloat16), torch)
        check('directed_norm_rows' + str(rows), c_norm(x, iw), t_norm(x, iw))
    finite = np.arange(65536, dtype=np.uint16)
    finite = finite[(finite & 0x7f80) != 0x7f80]
    ones = np.full_like(finite, 0x3f80)
    check('all_finite_BF16_SiLU_mul_one', c_binary(silu, finite, ones), bits(torch.nn.functional.silu(tensor(finite,
        torch)) * tensor(
        ones, torch), torch))
    if residual(None, None, None, 1) != - 1 or silu(None, None, None, 1) != - 1 or norm(None, None, ctypes.c_float(epsilon), None,
        1, 2048) != - 1:
        raise ValueError('null refusal differs')
    r['checks']['null_refusal'] = True
    for item in records:
        identity(item['path'], item['sha256'])
    for item in [compiler['physical_binary'], interpreter['physical_binary']] +[d['binding']['physical_binary'] for d in dep_ids]:
        identity(item['path'], item['sha256'])
    np.savez(a.out / 'parity-arrays.npz', **references)
    r.update(status='passed_Tiny_bounded_shared_C_primitives_only', source_input_bookends_unchanged=True, source_records=records,
        library=identity(a.out / 'host_ops.so'), arrays=identity(a.out / 'parity-arrays.npz'), torch_version=torch.__version__,
        torch_cpu_capability=torch.backends.cpu.get_cpu_capability(),
        native_dtype='BF16; RMSNorm FP32 reduction with cast before BF16 weight multiply', not_implemented=['RoPE', 'GQA/QK/PV',
        'full-axis masked softmax', 'KV-cache C implementation', 'complete C host graph', 'raw runtime integration', 'ELF'])


def main():
    from workload_cli import run_report
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native-gate', 'block-gate', 'decode-prepared', 'shared-source', 'support-dir', 'out'):
        p.add_argument('--' + name, type=safe, required=True)
    for name in ('native-gate', 'block-gate', 'decode-prepared', 'shared-source'):
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--cpu', type=int, required=True)
    a = p.parse_args()
    report = {'status': 'failed', 'command': [sys.executable, *sys.argv],
        'scope': 'Tiny-specific bounded native C parity from exact shared-source snapshot; not a C graph/device/runtime/ELF gate'}
    raise SystemExit(run_report(a, gate,
        filename='receipt.json', output=a.out, report=report,
        summary=lambda report: {'status': report['status'], 'checks': report.get('checks'), 'error': report.get('error')},
        allow_nan=True,
        failure_fields={"next_action": (
            'Stop this failed primitive; inspect source-native dtype/reduction/library policy before '
            'changing its implementation'
        )}))
if __name__ == '__main__':
    main()
