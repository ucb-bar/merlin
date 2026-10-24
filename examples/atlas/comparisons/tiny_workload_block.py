"""Dynamic first TinyLlama block with 12 exact device programs and native suffix.

Seven logical projections use actual target predecessors; output column shards
preserve K order and explicitly report whether B scales equal the global scale.
"""
from merlin.common.jsonio import write_pretty_json
import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import random
import numpy as np
import tiny_workload_native as native
from tiny_workload_inventory import column_shards
from capture_tinyllama_native import safe_path, identity, array_identity
from compare_tinyllama_attention import raw_array, tensor_identity, error
from workload_replay_admission import admit_executable, producer_identities, admit_child_result


def tensor_scale_from_exact_BF16_views(array):
    if array.dtype != np.float32 or not np.isfinite(array).all() or np.any(array.view(np.uint32) & 65535):
        raise ValueError('expected exact finite BF16 views')
    top = int(((array.view(np.uint32) >> 16) & 32767).max())
    if not top:
        return 0
    exp =(top >> 7) & 255
    if exp:
        significand = 128 +(top & 127)
        exp -= 127
    else:
        fraction = top & 127
        bit = fraction.bit_length() - 1
        significand = fraction <<(7 - bit)
        exp = bit - 133
    shift = exp - 8 + int(significand > 224)
    if not -126 <= shift <= 127:
        raise ValueError('global tensor scale outside admitted domain')
    return shift


def guarded_session(model, tokens, torch, cache_utils, modeling_llama, adapter=None):
    """Independently observe actual layer 0 K/V tails on all 33 native forwards."""
    attn = model.model.layers[0].self_attn
    seen = {}
    handles =[]
    tails =[]
    def pre(module, inputs, kwargs):
        seen.clear()
        seen['cache'] = kwargs['past_key_values']
        seen['cos'], seen['sin'] = kwargs['position_embeddings']
    handles.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
    for key in ('q_proj', 'k_proj', 'v_proj'):
        def observe(module, inputs, output, key=key):
            seen[key] = output.detach().clone()
        handles.append(getattr(attn, key).register_forward_hook(observe))
    def wrapped(name, a, c, b):
        result = adapter(name, a, c, b)
        if name.startswith('self_attn.') and name != 'self_attn.o_proj':
            seen[name.rsplit('.', 1)[1]] = result.detach().clone()
        return result
    def post(module, inputs, output):
        q = seen['q_proj']
        k = seen['k_proj']
        v = seen['v_proj']
        rows = q.shape[1]
        q = q.view(1, rows, 32, 64).transpose(1, 2)
        k = k.view(1, rows, 4, 64).transpose(1, 2)
        v = v.view(1, rows, 4, 64).transpose(1, 2)
        _, expected = modeling_llama.apply_rotary_pos_emb(q, k, seen['cos'], seen['sin'])
        cache = seen['cache'].layers[0]
        native.require(np.array_equal(raw_array(cache.keys[..., -rows:,:], torch), raw_array(expected, torch)) and
                       np.array_equal(raw_array(cache.values[..., -rows:,:], torch), raw_array(v, torch)),
            'actual native RoPE K/V cache tail differs')
        tails.append({'rows': rows, 'K': tensor_identity(expected, torch), 'V': tensor_identity(v, torch)})
    handles.append(attn.register_forward_hook(post))
    try:
        session = native.run_session(model, tokens, torch, cache_utils, modeling_llama, wrapped if adapter else None)
    finally:
        for handle in handles:
            handle.remove()
    native.require(len(tails) == 33, 'missing actual K/V tail checks')
    return session, tails


def evaluate(args, out, report):
    import torch
    rng = torch.get_rng_state().clone()
    nrng = np.random.get_state()
    prng = random.getstate()
    records =[]
    try:
        native_gate_id = identity(args.native_gate, args.native_gate_sha256)
        records.append(native_gate_id)
        gate = json.loads(args.native_gate.read_text())
        native.require(gate.get('status') == 'passed_frozen_native_BF16_session' and gate.get('native_repeat_exact') is True
                       and gate.get('source_bookends_verified') is True, 'require complete fresh native gate')
        for r in gate['source_records']:
            identity(r['path'], r['sha256'])
        producer_ids = producer_identities(args.producer, args.producer_sha256)
        python_id = admit_executable(args.model_python)
        python = Path(python_id["entry"])
        model, tokens, old, torch, cache_utils, modeling_llama, q = native.prepare(args.historical, records)
        parameters, buffers = native.state(model, torch)
        native.require(parameters == gate['full_parameters'] and buffers == gate['buffers'],
            'native state differs from fresh gate')
        caller_tokens = tokens.clone()
        reference, tails = guarded_session(model, tokens, torch, cache_utils, modeling_llama)
        native.require(native.summary(reference, torch) == gate['original_summary'], 'observed original native session differs')
        records +=[identity(__file__), identity(native.__file__), identity(Path(__file__).with_name(
            'tiny_workload_inventory.py')),
                    identity(Path(__file__).with_name('workload_replay_admission.py'))]
        child_env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
        child_env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        calls =[]
        logical =[]
        mode = 'execute'
        counter = 0
        first_model_binding = None
        def adapter(name, a, c, b):
            nonlocal counter, first_model_binding
            index = native.PROJECTIONS.index(name)
            if index != counter:
                raise ValueError('projection dataflow/order changed')
            counter += 1
            if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16 or c.dtype != torch.bfloat16 or a.shape[0] != 1 or a.shape[
                1] != 128:
                raise ValueError('native first block BF16 128-token ABI changed')
            af = a[0].detach().float().numpy().copy()
            bf = b.T.detach().float().numpy().copy()
            cf = c[0].detach().float().numpy().copy()
            k, n = bf.shape
            shards = column_shards(k, n)
            global_b = tensor_scale_from_exact_BF16_views(bf)
            pieces =[]
            shard_records =[]
            for shard_index, shard in enumerate(shards):
                start, end = shard['columns']
                arrays = {'A': af, 'B': bf[:, start:end].copy(), 'native_C': cf[:, start:end].copy()}
                runtime_ids = {key: array_identity(value) for key, value in arrays.items()}
                shape =(128, k, end - start)
                tag = str(index) + '-' + name + '-' + str(shard_index)
                directory = out / tag
                source = directory / 'source'
                child = directory / 'model'
                if mode == 'execute':
                    directory.mkdir()
                    source.mkdir()
                    np.savez(source / 'operands.npz', **arrays)
                    manifest = {'schema_version': 1, 'source_dtype': 'float32', 'npz_sha256': identity(source / 'operands.npz')[
                        'sha256'],
                              'source': {'kind': 'live_native_BF16_views_from_actual_target_predecessors',
                        'native_gate': native_gate_id,
                                        'native_precision': 'torch.bfloat16; no FP32-native reference claim', 'driver': identity(
                        __file__),
                                        'logical_B_global_scale': global_b, 'column_shard': shard['columns'],
                        'runtime_arrays': runtime_ids},
                              'site': {'module': 'model.layers.0.' + name, 'positions': [0, 128], 'K': k, 'columns': shard[
                        'columns']},
                        'arrays': runtime_ids}
                    write_pretty_json(source / 'manifest.json', manifest, sort_keys=False, allow_nan=False)
                    if index == 0 and shard_index == 0 and args.reuse_q is not None:
                        child = args.reuse_q
                    else:
                        command =[str(python), str(args.producer), '--model-root', str(args.model_root), '--input', str(
                            source / 'operands.npz'),
                                 '--manifest', str(source / 'manifest.json'), '--out', str(child), '--max-cycles', str(
                            args.max_cycles),
                            '--repeats', '1']
                        write_pretty_json(directory / 'command.json', {'command': command, 'env': {k: child_env[k] for k in (
                            'PYTHONDONTWRITEBYTECODE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}},
                            sort_keys=False, allow_nan=False)
                        with (directory / 'model.log').open('w') as log:
                            result = subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT, timeout=5400)
                        if result.returncode:
                            raise RuntimeError('fresh exact model failed: ' + tag)
                    write_pretty_json(directory / 'provider.json', {'path': str(child)}, sort_keys=False, indent=None)
                else:
                    child = safe_path(json.loads((directory / 'provider.json').read_text())['path'])
                words, receipt, binding = admit_child_result(child, producer_ids, runtime_ids, shape, 128)
                if first_model_binding is None:
                    first_model_binding = receipt['model_binding_before']
                elif first_model_binding != receipt['model_binding_before']:
                    raise ValueError('target model source epoch differs across kernels')
                pieces.append(words)
                scale = receipt['conversion']['B']['scale_exponent']
                record = {'tag': tag, 'columns': [start, end], 'shape': list(shape), 'child': binding, 'A_scale': receipt[
                    'conversion']['A'][
                    'scale_exponent'],
                        'B_scale': scale, 'rescale': receipt['output_rescale_exponent'], 'B_global_scale': global_b,
                    'B_scale_equals_global': scale == global_b,
                        'cycles': [r['cycles'] for r in receipt['runs']], 'runtime_arrays': runtime_ids}
                shard_records.append(record)
                if mode == 'execute':
                    calls.append(record)
                    write_pretty_json(out / 'progress.json', {'status': 'running', 'completed_programs': len(calls),
                        'last': record}, sort_keys=False, allow_nan=False)
            output = np.concatenate(pieces, axis=1)
            native.require(output.shape ==(128, n), 'column coverage/order differs')
            if mode == 'execute':
                logical.append({'projection': name, 'programs': shard_records, 'all_B_scales_global_equal': all(r[
                    'B_scale_equals_global'] for r in shard_records),
                                              'quality_vs_native_same_live_input': error((output.astype(np.uint32) << 16).view(
                    np.float32), cf)})
            return torch.from_numpy(output.copy()).view(torch.bfloat16).unsqueeze(0)
        candidate, candidate_tails = guarded_session(model, tokens, torch, cache_utils, modeling_llama, adapter)
        native.require(counter == 7 and len(calls) == 12, 'firstblock program inventory incomplete')
        mode = 'replay'
        counter = 0
        repeated, repeated_tails = guarded_session(model, tokens, torch, cache_utils, modeling_llama, adapter)
        native.require(native.summary(repeated, torch) == native.summary(candidate, torch) and repeated_tails == candidate_tails,
            'authenticated hybrid replay changed native outputs/cache')
        restored, restored_tails = guarded_session(model, tokens, torch, cache_utils, modeling_llama)
        native.require(native.summary(restored, torch) == native.summary(reference, torch) and restored_tails == tails,
            'removing adapters did not restore original native session')
        native.require(native.state(model, torch) ==(parameters, buffers) and torch.equal(tokens, caller_tokens),
            'caller parameters/buffers/tokens changed')
        native.require(candidate['contexts'] == reference['contexts'], 'native masks/positions/RoPE changed')
        errors = {'scored_logits': error(candidate['logits'][:32].float().numpy(), reference['logits'][:32].float().numpy()),
                'unscored_final_logits': error(candidate['logits'][32:].float().numpy(), reference['logits'][32:].float().numpy(
            ))}
        native_quality = all(r['allclose_rtol1e4_atol1e4'] for r in errors.values())
        np.savez(out / 'session-logits.npz', original_bf16=raw_array(reference['logits'], torch), candidate_bf16=raw_array(
            candidate[
            'logits'], torch), tokens=tokens.numpy())
        for record in records:
            identity(record['path'], record['sha256'])
        for record in gate['source_records']:
            identity(record['path'], record['sha256'])
        for key, record in producer_ids.items():
            identity(record['path'], record['sha256'])
        report.update(status='passed_exact_target_model_block_with_informational_native_quality', target_programs=calls,
            logical_projections=logical,
                      target_model_binding=first_model_binding, source_records=records, producer_ids=producer_ids,
            model_python=python_id,
                      native_reference_precision='original frozen BF16 model, not FP32', original_metrics=native.score(reference[
            'logits'], tokens, torch),
                      candidate_metrics=native.score(candidate['logits'], tokens, torch), errors=errors,
            native_gate_passed=native_quality,
                      native_gate_policy='original rtol=atol=1e-4 unchanged; quality is an explicit informational relaxation',
                      source_bookends_verified=True, native_hybrid_repeats_exact=True, native_restoration_exact=True,
                      parameters_buffers_tokens_preserved=True, all_22_cache_prefixes_lengths_exact=True,
            actual_layer0_KV_tails_verified_all_33_steps=True,
                      masks_positions_RoPE_unchanged=True, prefill_device_logical_ops=7, prefill_device_programs=12,
            decode_device_substitutions=0,
                      full_model_native_continuation_executed=True, full_model_Atlas_execution=False, session_logits=identity(
            out / 'session-logits.npz'))
    finally:
        torch.set_rng_state(rng)
        np.random.set_state(nrng)
        random.setstate(prng)
        report['caller_torch_rng_restored'] = torch.equal(rng, torch.get_rng_state())


def main():
    from workload_cli import run_report
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('historical', 'native-gate', 'producer', 'model-python', 'model-root', 'out'):
        p.add_argument('--' + name, type=safe_path if name != 'model-python' else Path, required=True)
    p.add_argument('--native-gate-sha256', required=True)
    p.add_argument('--producer-sha256', required=True)
    p.add_argument('--reuse-q', type=safe_path)
    p.add_argument('--max-cycles', type=int, default=200000000)
    args = p.parse_args()
    report = {'status': 'failed', 'command': sys.argv,
        'scope': 'dynamic firstblock actual target model outputs with original BF16 fullnative continuation'}
    raise SystemExit(run_report(args, lambda args, report: evaluate(args, args.out, report),
        filename='receipt.json', output=args.out, report=report,
        summary=('status', 'wall_seconds', 'peak_rss_kib')))
if __name__ == '__main__':
    main()
