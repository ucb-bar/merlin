"""Bounded Tiny host layout parity, using supplied native rotary tables.

Actual selected prefill model replies and first-decode arithmetic replies are
the inputs. This is not attention, C rotary-table generation, or a C graph.
"""
import argparse
import ctypes
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from tiny_host_ops_parity import arrays, bits, identity, pointer, safe, tensor


def gate(a, receipt):
    import torch
    from transformers.models.llama import modeling_llama
    from transformers import LlamaConfig
    from transformers.cache_utils import DynamicCache
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    if a.cpu not in os.sched_getaffinity(0):
        raise ValueError('CPU outside allowed affinity')
    os.sched_setaffinity(0, {a.cpu})
    pins =[]
    receipt.update(source_records=pins, cpu_affinity=sorted(os.sched_getaffinity(0)))

    def pin(path, expected=None):
        record = identity(path, expected)
        pins.append(record)
        return record

    pin(a.primitives, a.primitives_sha256)
    prior = json.loads(a.primitives.read_text())
    if prior['status'] != 'passed_Tiny_bounded_shared_C_primitives_only' or not prior['source_input_bookends_unchanged']:
        raise ValueError('primitive gate refused')
    for record in prior['source_records']:
        pin(record['path'], record['sha256'])
    known = {record['path']: record['sha256'] for record in pins}
    block_path = next(Path(p) for p in known if p.endswith('/tiny/block-v1/receipt.json'))
    decode_path = next(Path(p) for p in known if p.endswith('/tiny/dynamic-decode-block-v1/receipt.json'))
    native_path = next(Path(p) for p in known if p.endswith('/tiny/native-v1/receipt.json'))
    block = json.loads(block_path.read_text())
    decode = json.loads(decode_path.read_text())
    native = json.loads(native_path.read_text())
    native_sources = {record['path']: record['sha256'] for record in native['source_records']}
    for path in (Path(modeling_llama.__file__), Path(inspect.getfile(DynamicCache))):
        pin(path, native_sources[str(path)])
    config_path = next(Path(p) for p in known if p.endswith('/config.json'))
    pin(Path(inspect.getfile(LlamaConfig)))
    config = LlamaConfig.from_json_file(str(config_path))
    if (config.hidden_size, config.num_attention_heads, config.num_key_value_heads) !=(2048, 32, 4):
        raise ValueError('Tiny head geometry differs')
    # Native tables deliberately remain a supplied host contract in this gate.
    rotary = modeling_llama.LlamaRotaryEmbedding(config=config)
    pin(__file__)
    pin(Path(__file__).with_name('tiny_host_ops_parity.py'))
    sources =[Path(__file__).with_name('llama_host_layout' + suffix) for suffix in ('.c', '.h')]
    for source in sources:
        record = pin(source)
        destination = a.out / source.name
        destination.write_bytes(source.read_bytes())
        pin(destination, record['sha256'])
    support = next(Path(p).parent for p in known if p.endswith('/workload_replay_admission.py'))
    sys.path.insert(0, str(support))
    from workload_replay_admission import admit_executable
    compiler = admit_executable(shutil.which('cc'))
    interpreter = admit_executable(sys.executable)
    command =[compiler['entry'], '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
               '-ffp-contract=off', '-fno-fast-math', '-fPIC', '-shared',
               str(a.out / 'llama_host_layout.c'), '-o', str(a.out / 'layout.so')]
    build = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (a.out / 'build.log').write_text(build.stdout + build.stderr)
    receipt.update(build_command=command, build_returncode=build.returncode,
                   compiler_identity=compiler, interpreter_identity=interpreter)
    if build.returncode:
        raise RuntimeError('strict C build failed')
    dependencies = subprocess.check_output(['ldd', str(a.out / 'layout.so')], text=True)
    (a.out / 'linked-dependencies.txt').write_text(dependencies)
    linked =[]
    for line in dependencies.splitlines():
        if '=>' in line and line.split('=>', 1)[0].strip() == 'libc.so.6':
            linked.append(admit_executable(line.split('=>', 1)[1].split()[0]))
    if len(linked) != 1:
        raise ValueError('native C library identity absent')
    receipt['linked_dependencies'] = linked
    library = ctypes.CDLL(str(a.out / 'layout.so'))
    split = library.llama_host_split_heads_bf16
    rotate = library.llama_host_rotary_bf16
    append = library.llama_host_cache_append_bf16
    repeat = library.llama_host_repeat_kv_bf16
    split.argtypes =[ctypes.c_void_p] * 2 +[ctypes.c_size_t] * 3
    rotate.argtypes =[ctypes.c_void_p] * 4 +[ctypes.c_size_t] * 3
    append.argtypes =[ctypes.c_void_p] * 3 +[ctypes.c_size_t] * 4
    repeat.argtypes =[ctypes.c_void_p] * 2 +[ctypes.c_size_t] * 4
    for function in (split, rotate, append, repeat):
        function.restype = ctypes.c_int
    prefill = {p['tag']: p for p in block['target_programs']}
    prepared = {p['tag']: p for p in decode['prepared_decode_panels']}

    def replies(step, index, name):
        tag = str(index) + '-self_attn.' + name + '_proj-0'
        if step == 0:
            record = prefill[tag]['child']['receipt']
            pin(record['path'], record['sha256'])
            child = json.loads(safe(record['path']).read_text())
            if child['status'] != 'passed_model_only' or not child['runs'] or not child['source_bookends']['unchanged']:
                raise ValueError('prefill child refused')
            for run in child['runs']:
                if not all(run[k] is True for k in ('all_output_bytes_exact', 'input_and_guards_preserved',
                    'drained_before_halt_observation')):
                    raise ValueError('prefill numerical qualification absent')
            path = Path(record['path']).parent / 'result.npz'
            pin(path, child['artifacts']['result.npz'])
            result = arrays(path, {'actual_bits', 'expected_bits', 'target_C'})
            if not np.array_equal(result['actual_bits'], result['expected_bits']):
                raise ValueError('prefill result disagrees')
            return result['actual_bits']
        record = prepared[tag]['artifacts']['expected-logical-u16.npy']
        pin(record['path'], record['sha256'])
        return np.load(safe(record['path']), allow_pickle=False)

    saved = {}
    receipt['checks'] = {}

    def check(name, actual, expected):
        if actual.shape != expected.shape or actual.dtype != np.uint16 or expected.dtype != np.uint16:
            raise ValueError('layout or datatype differs ' + name)
        count = int(np.count_nonzero(actual != expected))
        receipt['checks'][name] = dict(words=int(actual.size), mismatches=count)
        saved[name + '_actual'] = actual
        saved[name + '_expected'] = expected
        if count:
            receipt['first_mismatch'] = dict(name=name, positions=np.argwhere(actual != expected)[:16].tolist())
            np.savez(a.out / 'failure-arrays.npz', actual=actual, expected=expected)
            receipt['failure_arrays'] = identity(a.out / 'failure-arrays.npz')
            raise ValueError('C/native parity failed ' + name)

    cache = DynamicCache(config=config)
    c_keys = c_values = None
    for step, rows in enumerate((128, 1)):
        linear =[replies(step, i, name) for i, name in enumerate(('q', 'k', 'v'))]
        if any(x.dtype != np.uint16 or x.shape !=(rows, n) for x, n in zip(linear, (2048, 256, 256))):
            raise ValueError('actual selected reply geometry differs')
        native_heads =[tensor(x, torch).reshape(1, rows, -1, 64).transpose(1, 2) for x in linear]
        c_heads =[]
        for name, x, reference in zip(('q', 'k', 'v'), linear, native_heads):
            output = np.empty((reference.shape[1], rows, 64), np.uint16)
            if split(pointer(x), pointer(output), rows, output.shape[0], 64):
                raise ValueError('split-head primitive refused')
            check(f'step{step}_{name}_heads', output, bits(reference[0], torch))
            c_heads.append(output)
        positions = torch.arange(0, 128).unsqueeze(0) if step == 0 else torch.tensor([[128]])
        cosine, sine = rotary(native_heads[0], positions)
        cos_words = bits(cosine[0], torch)
        sin_words = bits(sine[0], torch)
        saved[f'step{step}_supplied_cosine'] = cos_words
        saved[f'step{step}_supplied_sine'] = sin_words
        nq, nk = modeling_llama.apply_rotary_pos_emb(native_heads[0], native_heads[1], cosine, sine)
        for name, index, reference in (('q', 0, nq), ('k', 1, nk)):
            output = np.empty_like(c_heads[index])
            if rotate(pointer(c_heads[index]), pointer(cos_words), pointer(sin_words), pointer(output), output.shape[0], rows,
                64):
                raise ValueError('rotary primitive refused')
            check(f'step{step}_{name}_rotary_supplied_tables', output, bits(reference[0], torch))
            c_heads[index] = output
        native_keys, native_values = cache.update(nk, native_heads[2], 0)
        for name, tail, prefix, reference in (('keys', c_heads[1], c_keys, native_keys), ('values', c_heads[2], c_values,
            native_values)):
            past = 0 if prefix is None else prefix.shape[1]
            output = np.empty((4, past + rows, 64), np.uint16)
            if append(None if prefix is None else pointer(prefix), pointer(tail), pointer(output), 4, past, rows, 64):
                raise ValueError('cache primitive refused')
            check(f'step{step}_{name}_cache', output, bits(reference[0], torch))
            if name == 'keys':
                c_keys = output
            else:
                c_values = output
            expanded = np.empty((32, past + rows, 64), np.uint16)
            if repeat(pointer(output), pointer(expanded), 4, 8, past + rows, 64):
                raise ValueError('repeat-kv primitive refused')
            expected = bits(modeling_llama.repeat_kv(reference, 8)[0], torch)
            check(f'step{step}_{name}_GQA_repeat', expanded, expected)
    witness = np.array([[[0x0000, 0x8000, 0x3f80, 0xbf80]],
                        [[0x8000, 0x0000, 0xbf80, 0x3f80]]], np.uint16)
    cosine = np.array([[0x3f80, 0xbf80, 0x0000, 0x8000]], np.uint16)
    sine = np.array([[0x8000, 0x0000, 0x3f80, 0xbf80]], np.uint16)
    output = np.empty_like(witness)
    if rotate(pointer(witness), pointer(cosine), pointer(sine), pointer(output), 2, 1, 4):
        raise ValueError('directed rotary witness refused')
    reference, _ = modeling_llama.apply_rotary_pos_emb(tensor(witness, torch).unsqueeze(0),
                                                      tensor(witness, torch).unsqueeze(0),
                                                      tensor(cosine, torch).unsqueeze(0),
                                                      tensor(sine, torch).unsqueeze(0))
    check('directed_width4_signed_zero_rotary', output, bits(reference[0], torch))
    maximum = ctypes.c_size_t(-1).value
    source = np.ones(16, np.uint16)
    destination = np.zeros_like(source)
    p, q = pointer(source), pointer(destination)
    refusal = {
        'split_null': split(None, q, 1, 1, 4),
        'split_zero_rows': split(p, q, 0, 1, 4),
        'split_overflow': split(p, q, maximum, 2, 4),
        'split_exact_alias': split(p, p, 1, 1, 4),
        'rotary_null': rotate(None, p, p, q, 1, 1, 4),
        'rotary_zero_heads': rotate(p, p, p, q, 0, 1, 4),
        'rotary_odd_width': rotate(p, p, p, q, 1, 1, 3),
        'rotary_overflow': rotate(p, p, p, q, 2, maximum, 4),
        'rotary_input_alias': rotate(p, q, q, p, 1, 1, 4),
        'rotary_cosine_alias': rotate(q, p, q, p, 1, 1, 4),
        'rotary_sine_alias': rotate(q, q, p, p, 1, 1, 4),
        'cache_null_tail': append(p, None, q, 1, 1, 1, 4),
        'cache_null_prefix': append(None, p, q, 1, 1, 1, 4),
        'cache_zero_rows': append(p, p, q, 1, 1, 0, 4),
        'cache_row_overflow': append(p, p, q, 1, maximum, 1, 4),
        'cache_extent_overflow': append(p, p, q, maximum, 1, 1, 4),
        'cache_prefix_alias': append(p, q, p, 1, 1, 1, 4),
        'cache_tail_alias': append(q, p, p, 1, 1, 1, 4),
        'repeat_null': repeat(None, q, 1, 1, 1, 4),
        'repeat_zero_groups': repeat(p, q, 1, 0, 1, 4),
        'repeat_group_overflow': repeat(p, q, 2, maximum, 1, 4),
        'repeat_exact_alias': repeat(p, p, 1, 1, 1, 4),
    }
    receipt['refusal_controls'] = refusal
    if any(code != - 1 for code in refusal.values()):
        raise ValueError('directed invalid input accepted')
    for record in pins:
        identity(record['path'], record['sha256'])
    for binding in (compiler, interpreter, *linked):
        identity(binding['physical_binary']['path'], binding['physical_binary']['sha256'])
    np.savez(a.out / 'parity-arrays.npz', **saved)
    receipt.update(status='passed_Tiny_C_layout_supplied_rotary_tables_cache_GQA_only',
                   source_input_bookends_unchanged=True, library=identity(a.out / 'layout.so'),
                   arrays=identity(a.out / 'parity-arrays.npz'),
                   excluded=['C rotary-table generation', 'QK', 'softmax', 'PV', 'complete C graph', 'device runtime', 'ELF'])


def main():
    from workload_cli import run_report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primitives', type=safe, required=True)
    parser.add_argument('--primitives-sha256', required=True)
    parser.add_argument('--out', type=safe, required=True)
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    report = dict(status='failed', command=[sys.executable, *sys.argv], scope=__doc__)
    raise SystemExit(run_report(args, gate,
        filename='receipt.json', output=args.out, report=report,
        summary=lambda report: dict(status=report['status'], error=report.get('error')), allow_nan=True,
        failure_fields={"next_action": (
            'Stop the failing operator; preserve exact mismatch and inspect native layout/dtype '
            'contract'
        )}))


if __name__ == '__main__':
    main()
