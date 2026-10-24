"""Prepare bounded model/config rotary parameters and qualify C position lookup.

Only cosine/sine parameters enter the runtime table bundle. Activations, cache
values and expected outputs remain separate parity evidence. Native Torch
produces tables offline; runtime C rejects positions outside 0 through 159.
"""

from merlin.common.jsonio import write_pretty_json
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

import numpy as np

from tiny_host_ops_parity import arrays, bits, identity, pointer, safe, tensor


def gate(args, report):
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama import modeling_llama
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    assert args.cpu in os.sched_getaffinity(0)
    os.sched_setaffinity(0, {args.cpu})
    records =[]
    report.update(source_records=records, cpu_affinity=sorted(os.sched_getaffinity(0)))

    def pin(path, expected=None):
        record = identity(path, expected)
        records.append(record)
        return record

    pin(args.layout, args.layout_sha256)
    layout = json.loads(args.layout.read_text())
    assert layout['status'] == 'passed_Tiny_C_layout_supplied_rotary_tables_cache_GQA_only'
    assert layout['source_input_bookends_unchanged']
    for record in layout['source_records']:
        pin(record['path'], record['sha256'])
    known = {record['path']: record['sha256'] for record in records}
    config_path = next(Path(p) for p in known if p.endswith('/config.json'))
    for path in (Path(inspect.getfile(LlamaConfig)), Path(modeling_llama.__file__)):
        pin(path, known[str(path)])
    config = LlamaConfig.from_json_file(str(config_path))
    assert (config.hidden_size, config.num_attention_heads, config.num_key_value_heads) ==(2048, 32, 4)
    config_record = identity(config_path)
    native_path = next(Path(p) for p in known if p.endswith('/tiny/native-v1/receipt.json'))
    native = json.loads(native_path.read_text())
    position_ids = torch.arange(160).unsqueeze(0)
    rotary = modeling_llama.LlamaRotaryEmbedding(config=config)
    assert rotary.rope_type == 'default', 'bounded default rotary producer required'
    # Default frequency construction is a method in modeling_llama; the
    # forward's dynamic-update decorator comes from modeling_rope_utils.
    parameter_dependencies =[pin(Path(inspect.getfile(rotary.compute_default_rope_parameters))),
                              pin(Path(inspect.getfile(modeling_llama.dynamic_rope_update)))]
    frequency = rotary.inv_freq.detach().contiguous().numpy()
    frequency_id = dict(shape=list(frequency.shape), dtype=str(frequency.dtype),
                        sha256=sha256_bytes(frequency.tobytes()))
    assert frequency_id == {k: v for k, v in native['buffers']['model.rotary_emb.inv_freq'].items() if k != 'native_dtype'}
    cosine, sine = rotary(torch.empty((1, 160, 64), dtype=torch.bfloat16), position_ids)
    cosine, sine = bits(cosine[0], torch), bits(sine[0], torch)
    assert cosine.shape == sine.shape ==(160, 64) and cosine.dtype == sine.dtype == np.uint16
    context_bindings =[]
    for step, context in enumerate(native['original_summary']['contexts']):
        lo, hi =(0, 128) if step == 0 else (127 + step, 128 + step)
        for name, table in (('rope_cos', cosine), ('rope_sin', sine)):
            selected = table[None, lo:hi]
            found = dict(shape=list(selected.shape), dtype=str(selected.dtype),
                         sha256=sha256_bytes(selected.tobytes()))
            assert found == {k: v for k, v in context[name].items(
                ) if k != 'native_dtype'}, 'original native rotary context differs'
        context_bindings.append(dict(step=step, positions=[lo, hi]))
    assert len(context_bindings) == 33
    report['all_33_original_native_rotary_contexts_exact'] = context_bindings
    pin(__file__)
    # Runtime bundle contains parameters only, never observed activations.
    np.savez(args.out / 'rotary-parameters.npz', cosine_bf16=cosine, sine_bf16=sine)
    header =['#ifndef TINY_ROTARY_PARAMETERS_H', '#define TINY_ROTARY_PARAMETERS_H', '#include <stdint.h>']
    for name, table in (('cosine', cosine), ('sine', sine)):
        header.append(f'static const uint16_t tiny_rotary_{name}[160][64] = {{')
        header.extend('  {' + ','.join(f'0x{int(x):04x}' for x in row) + '},' for row in table)
        header.append('};')
    header.append('#endif')
    (args.out / 'rotary-parameters.h').write_text('\n'.join(header) + '\n')
    table_manifest = dict(schema_version=1, kind='immutable_model_config_parameters_only',
                          admitted_position_range=[0, 159], table_shape=[160, 64], dtype='BF16_words_uint16',
                          native_producer=identity(Path(modeling_llama.__file__)), config=config_record,
                          parameter_dependencies=parameter_dependencies, original_inv_freq=frequency_id,
                          original_native_context_gate=identity(native_path), all_33_native_contexts_exact=True,
                          producer=identity(__file__), artifacts={n: identity(args.out / n) for n in ('rotary-parameters.npz',
        'rotary-parameters.h')},
                          excluded=['activations', 'KV cache', 'expected outputs'])
    write_pretty_json(args.out / 'rotary-parameters.json', table_manifest, sort_keys=False)
    for name in ('rotary-parameters.npz', 'rotary-parameters.h', 'rotary-parameters.json'):
        pin(args.out / name)
    bundle = arrays(args.out / 'rotary-parameters.npz', {'cosine_bf16', 'sine_bf16'})
    assert np.array_equal(bundle['cosine_bf16'], cosine) and np.array_equal(bundle['sine_bf16'], sine)
    header_rows =[line.strip()[1:-2].split(',') for line in (args.out / 'rotary-parameters.h').read_text().splitlines()
                   if line.startswith('  {')]
    header_words = np.array([[int(word, 16) for word in row] for row in header_rows], dtype=np.uint16)
    assert np.array_equal(header_words, np.concatenate([cosine, sine], axis=0))
    # Drive C from the actual emitted runtime bundle after independent readback.
    cosine, sine = bundle['cosine_bf16'], bundle['sine_bf16']
    support = next(Path(p).parent for p in known if p.endswith('/workload_replay_admission.py'))
    sys.path.insert(0, str(support))
    from workload_replay_admission import admit_executable
    compiler, interpreter = admit_executable(shutil.which('cc')), admit_executable(sys.executable)
    for suffix in ('.c', '.h'):
        source = Path(__file__).with_name('llama_rotary_table' + suffix)
        record = pin(source)
        target = args.out / source.name
        target.write_bytes(source.read_bytes())
        pin(target, record['sha256'])
    command =[compiler['entry'], '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
               '-ffp-contract=off', '-fno-fast-math', '-fPIC', '-shared',
               str(args.out / 'llama_rotary_table.c'), '-o', str(args.out / 'lookup.so')]
    build = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (args.out / 'build.log').write_text(build.stdout + build.stderr)
    report.update(build_command=command, build_returncode=build.returncode,
                  compiler_identity=compiler, interpreter_identity=interpreter,
                  native_dependencies=layout['linked_dependencies'])
    assert build.returncode == 0, 'strict lookup build failed'
    pin(layout['library']['path'], layout['library']['sha256'])
    application_library = ctypes.CDLL(layout['library']['path'])
    apply = application_library.llama_host_rotary_bf16
    apply.argtypes =[ctypes.c_void_p] * 4 +[ctypes.c_size_t] * 3
    apply.restype = ctypes.c_int
    library = ctypes.CDLL(str(args.out / 'lookup.so'))
    lookup = library.llama_host_rotary_lookup_bf16
    lookup.argtypes =[ctypes.c_void_p] * 2 +[ctypes.c_size_t] * 2 +[ctypes.c_void_p, ctypes.c_size_t] +[ctypes.c_void_p] * 2
    lookup.restype = ctypes.c_int
    parity = {}
    report['checks'] = {}

    def check(name, actual, expected):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype == np.uint16
        mismatches = int(np.count_nonzero(actual != expected))
        report['checks'][name] = dict(words=int(actual.size), mismatches=mismatches)
        parity[name + '_actual'], parity[name + '_expected'] = actual, expected
        if mismatches:
            np.savez(args.out / 'failure-arrays.npz', actual=actual, expected=expected)
            report['failure_arrays'] = identity(args.out / 'failure-arrays.npz')
            raise ValueError('rotary lookup parity failed ' + name)

    def fetch(positions):
        positions = np.ascontiguousarray(positions, dtype=np.int64)
        cos_out, sin_out = np.empty((positions.size, 64), np.uint16), np.empty((positions.size, 64), np.uint16)
        assert lookup(pointer(cosine), pointer(sine), 160, 64, pointer(positions), positions.size, pointer(cos_out), pointer(
            sin_out)) == 0
        return cos_out, sin_out

    positions = np.concatenate([np.arange(159, -1, -1), np.array([0, 128, 159, 0])]).astype(np.int64)
    actual = fetch(positions)
    for name, found, table in zip(('cosine', 'sine'), actual, (cosine, sine)):
        check('all_positions_permuted_repeated_' + name, found, table[positions])
    pin(layout['arrays']['path'], layout['arrays']['sha256'])
    names = {f'step{step}_{name}_{suffix}' for step in (0, 1)
             for name in ('q_heads', 'k_heads', 'v_heads', 'q_rotary_supplied_tables',
                          'k_rotary_supplied_tables', 'keys_cache', 'keys_GQA_repeat',
                          'values_cache', 'values_GQA_repeat') for suffix in ('actual', 'expected')}
    names |= {f'step{step}_supplied_{name}' for step in (0, 1) for name in ('cosine', 'sine')}
    names |= {'directed_width4_signed_zero_rotary_actual', 'directed_width4_signed_zero_rotary_expected'}
    archive = arrays(layout['arrays']['path'], names)
    for step, positions in ((0, np.arange(128)), (1, np.array([128]))):
        cos_out, sin_out = fetch(positions)
        for name, found in (('cosine', cos_out), ('sine', sin_out)):
            check(f'step{step}_actual_position_{name}', found, archive[f'step{step}_supplied_{name}'])
        for name in ('q', 'k'):
            heads = archive[f'step{step}_{name}_heads_actual'].copy()
            output = np.empty_like(heads)
            assert apply(pointer(heads), pointer(cos_out), pointer(sin_out), pointer(output), heads.shape[0], positions.size,
                64) == 0
            check(f'step{step}_{name}_rotary_runtime_lookup', output, archive[f'step{step}_{
                name}_rotary_supplied_tables_expected'])
    sentinel = np.full((2, 64), 0x5555, np.uint16)
    other = np.full_like(sentinel, 0xaaaa)
    refusals = {}
    for value in (-1, 160, np.iinfo(np.int64).min, np.iinfo(np.int64).max):
        positions = np.array([0, value], np.int64)
        refusals[str(value)] = lookup(pointer(cosine), pointer(sine), 160, 64, pointer(positions), 2, pointer(sentinel), pointer(
            other))
        assert np.all(sentinel == 0x5555) and np.all(other == 0xaaaa), 'invalid positions partially wrote outputs'
    position = np.array([0], np.int64)
    c, s, p, a, b = pointer(cosine), pointer(sine), pointer(position), pointer(sentinel), pointer(other)
    maximum = ctypes.c_size_t(-1).value
    refusals.update(null_table=lookup(None, s, 160, 64, p, 1, a, b),
                    null_positions=lookup(c, s, 160, 64, None, 1, a, b),
                    zero_rows=lookup(c, s, 160, 64, p, 0, a, b),
                    zero_tables=lookup(c, s, 0, 64, p, 1, a, b),
                    extent_overflow=lookup(c, s, maximum, 64, p, 1, a, b),
                    output_alias=lookup(c, s, 160, 64, p, 1, a, a),
                    table_alias=lookup(c, s, 160, 64, p, 1, c, b),
                    position_alias=lookup(c, s, 160, 64, p, 1, p, b))
    report['refusal_controls'] = refusals
    assert all(code == - 1 for code in refusals.values())
    for record in records:
        identity(record['path'], record['sha256'])
    for binding in (compiler, interpreter, *layout['linked_dependencies']):
        identity(binding['physical_binary']['path'], binding['physical_binary']['sha256'])
    np.savez(args.out / 'parity-arrays.npz', **parity)
    report.update(status='passed_bounded_rotary_parameters_and_C_runtime_lookup', source_bookends_verified=True,
                  runtime_parameter_manifest=identity(args.out / 'rotary-parameters.json'),
                  library=identity(args.out / 'lookup.so'), parity=identity(args.out / 'parity-arrays.npz'),
                  excluded=['C table generation', 'attention', 'complete C graph', 'device runtime', 'ELF'])


def main():
    from workload_cli import run_report
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', type=safe, required=True)
    parser.add_argument('--layout-sha256', required=True)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--out', type=safe, required=True)
    args = parser.parse_args()
    report = dict(status='failed', command=[sys.executable, *sys.argv], scope=__doc__)
    raise SystemExit(run_report(args, gate,
        filename='receipt.json', output=args.out, report=report,
        summary=lambda report: dict(status=report['status'], error=report.get('error')), allow_nan=True,
        failure_fields={'next_action': 'Stop failed parameter identity or lookup/refusal gate and preserve evidence'}))


if __name__ == '__main__':
    main()
