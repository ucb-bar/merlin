"""Whole planned Tiny partition arithmetic-only quality; no instruction execution.

All 154 decoder projections/MLPs and the LM head use exact ordered target MAC
arithmetic. Native BF16 host operators, actual caches and frozen token inputs
remain unchanged. Firstblock device-word coupling is a separately checked gate.
"""
from merlin.common.jsonio import write_pretty_json
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
import numpy as np
import tiny_workload_native as native
from tiny_workload_block import guarded_session
from tiny_workload_inventory import column_shards
from tiny_workload_quality import read_session, logits, comparison, score
from tiny_fp8_pack import encode, tensor_scale_bits
from capture_tinyllama_native import safe_path, identity, array_identity
from compare_tinyllama_attention import raw_array, tensor_identity


def load_arithmetic(path, expected_sha256, records):
    """Import the caller-selected target oracle after binding its exact source."""
    record = identity(path, expected_sha256)
    spec = importlib.util.spec_from_file_location("selected_atlas_vector_arithmetic", record['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not all(callable(getattr(module, name, None)) for name in ('ordered_matmul', 'rescale_bf16_array')):
        raise ValueError('Selected Atlas arithmetic helper lacks the required API')
    identity(record['path'], record['sha256'])
    records.append(record)
    return module


def evaluate(args, out, report):
    import torch
    rng = torch.get_rng_state().clone()
    nrng = np.random.get_state()
    prng = random.getstate()
    records =[]
    handles =[]
    start = time.monotonic()
    try:
        oracle = load_arithmetic(args.atlas_arithmetic, args.atlas_arithmetic_sha256, records)
        ordered_matmul, rescale_bf16_array = oracle.ordered_matmul, oracle.rescale_bf16_array
        affinity = os.sched_getaffinity(0)
        cpu = args.cpu if args.cpu is not None else max(affinity)
        if cpu not in affinity:
            raise ValueError('requested CPU outside allowed affinity')
        os.sched_setaffinity(0, {cpu})
        report['cpu_affinity'] =[cpu]
        gate_id = identity(args.native_gate, args.native_gate_sha256)
        fp_id = identity(args.fp32_gate, args.fp32_gate_sha256)
        records +=[gate_id, fp_id]
        gate = json.loads(args.native_gate.read_text())
        fp, fp_arrays, more = read_session(args.fp32_gate, 'passed_separate_FP32_reference', {'logits', 'tokens'})
        records += more
        if gate.get('status') != 'passed_frozen_native_BF16_session' or gate.get('native_repeat_exact') is not True or gate.get(
            'source_bookends_verified') is not True:
            raise ValueError('require fresh complete native BF16 gate')
        for r in gate['source_records']:
            identity(r['path'], r['sha256'])
        model, tokens, old, torch, cache_utils, modeling_llama, q = native.prepare(args.historical, records)
        params, buffers = native.state(model, torch)
        native.require(params == gate['full_parameters'] and buffers == gate['buffers'], 'native state differs')
        caller_tokens = tokens.clone()
        reference, reference_tails = guarded_session(model, tokens, torch, cache_utils, modeling_llama)
        native.require(native.summary(reference, torch) == gate['original_summary'],
            'complete BF16 reference differs from frozen gate')
        sites=[f'model.layers.{i}.{name}'for i in range(22)for name in native.PROJECTIONS]+['lm_head']
        modules = {name: m for name, m in model.named_modules() if type(m) is torch.nn.Linear}
        native.require(set(modules) == set(sites) and len(sites) == 155, 'whole planned linear operator inventory differs')
        records +=[identity(Path(__file__).with_name(name)) for name in ('tiny_workload_native.py', 'tiny_workload_block.py',
            'tiny_workload_inventory.py',
                   'tiny_workload_quality.py', 'tiny_fp8_pack.py', 'capture_tinyllama_native.py',
            'compare_tinyllama_attention.py',
            'workload_replay_admission.py')]
        records.append(identity(__file__))
        cache = {}
        outputs = {}
        calls =[]
        couplings =[]
        counter = 0
        mode = 'compute'
        useful_MACs = 0
        def hook(module, inputs, output, site):
            nonlocal counter, useful_MACs
            step, index = divmod(counter, 155)
            if step > 32 or site != sites[index]:
                raise ValueError('full linear dataflow/order differs')
            rows = 128 if not step else 1
            a = inputs[0]
            if (a.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
                    or module.weight.dtype != torch.bfloat16 or module.bias is not None
                    or tuple(a.shape) !=(1, rows, module.in_features) or tuple(output.shape) !=(1, rows, module.out_features)):
                raise ValueError('native BF16 wholelinear ABI differs')
            if time.monotonic() - start > args.wall_seconds:
                raise TimeoutError('full arithmetic wall limit')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 24 * 1024 ** 3:
                raise MemoryError('24 GiB RSS cap')
            source_a = raw_array(a[0], torch)
            a_id = array_identity(source_a)
            key =(step, site)
            if mode == 'replay':
                saved, expected_id = outputs[key]
                if a_id != expected_id:
                    raise ValueError('arithmetic replay used a different actual predecessor')
                counter += 1
                return torch.from_numpy(saved.copy()).view(torch.bfloat16).unsqueeze(0)
            if site not in cache:
                raw_b = raw_array(module.weight.T, torch)
                global_scale = tensor_scale_bits(raw_b)
                panels =[]
                for shard in column_shards(module.in_features, module.out_features):
                    lo, hi = shard['columns']
                    bits = raw_b[:, lo:hi].copy()
                    encoded, scale = encode(bits)
                    encoded.flags.writeable = False
                    panels.append({'columns': [lo, hi], 'encoded': encoded, 'scale': scale, 'raw_id': array_identity(bits)})
                cache[site] = {'panels': panels, 'global_scale': global_scale, 'raw_B': array_identity(raw_b),
                    'parameter_version': module.weight._version,
                             'data_ptr': module.weight.data_ptr()}
            weights = cache[site]
            if module.weight._version != weights['parameter_version'] or module.weight.data_ptr() != weights['data_ptr']:
                raise ValueError('cached weight changed')
            encoded_a, ascale = encode(source_a)
            pieces =[]
            for panel_index, panel in enumerate(weights['panels']):
                lo, hi = panel['columns']
                exponent = ascale + panel['scale']
                result = rescale_bf16_array(ordered_matmul(encoded_a, panel['encoded']), exponent)
                pieces.append(result)
                record = {'step': step, 'site': site, 'columns': [lo, hi], 'logical_mkn': [rows, module.in_features, hi - lo],
                        'planned_physical_M': 128 if not step else 32, 'A_words': a_id, 'B_words': panel['raw_id'],
                        'A_scale': ascale, 'B_scale': panel['scale'], 'B_global_scale': weights['global_scale'],
                        'B_scale_equals_global': panel['scale'] == weights['global_scale'], 'rescale': exponent,
                    'output_words': array_identity(result)}
                calls.append(record)
                if not step and index < 7:
                    # Retain enough to authenticate the later actual npu result;
                    # no captured activation substitution occurs during this run.
                    af = a[0].detach().float().numpy().copy()
                    bf = module.weight.T[:, lo:hi].detach().float().numpy().copy()
                    cf = output[0,:, lo:hi].detach().float().numpy().copy()
                    payload=out/f'firstblock-{index}-{panel_index}.u16.npy'
                    np.save(payload, result)
                    couplings.append({'projection': native.PROJECTIONS[index], 'panel': panel_index, 'columns': [lo, hi],
                                      'logical_shape': [rows, module.in_features, hi - lo],
                                      'runtime_arrays': {name: array_identity(value) for name, value in [('A', af), ('B', bf), (
                        'native_C', cf)]},
                                      'arithmetic_words': identity(payload), 'B_scale': panel['scale'], 'A_scale': ascale})
            words = np.concatenate(pieces, axis=1)
            if words.shape !=(rows, module.out_features):
                raise ValueError('column concat coverage differs')
            words.flags.writeable = False
            outputs[key] =(words, a_id)
            useful_MACs += rows * module.in_features * module.out_features
            counter += 1
            if index == 154:
                write_pretty_json(out / 'progress.json', {'status': 'running_arithmetic_only',
                    'completed_native_forwards': step + 1,
                             'logical_linear_calls': counter, 'panel_calls': len(calls), 'useful_MACs': useful_MACs,
                    'wall_seconds': time.monotonic() - start}, sort_keys=False, allow_nan=False)
            return torch.from_numpy(words.copy()).view(torch.bfloat16).unsqueeze(0)
        for site, module in modules.items():
            handles.append(module.register_forward_hook(lambda module, inputs, output, site=site: hook(module, inputs, output,
                site)))
        try:
            candidate, candidate_tails = guarded_session(model, tokens, torch, cache_utils, modeling_llama)
            native.require(counter == 155 * 33 and len(calls) == 280 * 33 and useful_MACs == 160 * sum(m.weight.numel(
                ) for m in modules.values()), 'full arithmetic invocation inventory incomplete')
            mode = 'replay'
            counter = 0
            repeated, repeated_tails = guarded_session(model, tokens, torch, cache_utils, modeling_llama)
            # Internal observer fields named native_* see effective target outputs
            # here, so compare only accurately named logits/cache/context fields.
            def actual_summary(session):
                return {k: native.summary(session, torch)[k] for k in ('logits', 'contexts', 'cache_steps')}
            native.require(actual_summary(candidate) == actual_summary(repeated) and candidate_tails == repeated_tails,
                'actual predecessor replay/native host repeat differs')
        finally:
            for handle in handles:
                handle.remove()
            handles.clear()
        restored, restored_tails = guarded_session(model, tokens, torch, cache_utils, modeling_llama)
        native.require(native.summary(restored, torch) == native.summary(reference, torch) and reference_tails == restored_tails,
            'restoring original whole native model failed')
        native.require(native.state(model, torch) ==(params, buffers) and torch.equal(tokens, caller_tokens),
            'caller state changed')
        native.require(candidate['contexts'] == reference['contexts'], 'native masks/positions/RoPE changed')
        native.require(np.array_equal(tokens.numpy(), fp_arrays['tokens']), 'FP32 reference token stream differs')
        actual = candidate['logits'].float().numpy()
        original = reference['logits'].float().numpy()
        FP32 = logits(fp_arrays['logits'], False)
        np.savez(out / 'session-logits.npz', original_bf16=raw_array(reference['logits'], torch), candidate_bf16=raw_array(
            candidate[
            'logits'], torch), tokens=tokens.numpy())
        for rec in records:
            identity(rec['path'], rec['sha256'])
        for rec in gate['source_records']:
            identity(rec['path'], rec['sha256'])
        weight_info = {site: {k: value for k, value in data.items() if k not in ('panels', 'data_ptr', 'parameter_version')}
                     | {'panels': [{k: v for k, v in panel.items() if k != 'encoded'} for panel in data['panels']]} for site,
            data in cache.items()}
        report.update(status='completed_full_planned_partition_arithmetic_only', source_records=records,
            source_bookends_verified=True,
                      original_native_precision='BF16 unchanged', FP32_reference_policy=fp['reference_policy'],
                      task_quality={'actual_vs_original_BF16': comparison(actual, original),
            'actual_vs_separate_FP32': comparison(
            actual, FP32),
                                    'original_BF16_scoring': score(original, tokens.numpy()), 'actual_scoring': score(actual,
            tokens.numpy()), 'FP32_scoring': score(FP32, tokens.numpy())},
                      original_Torch_metrics=native.score(reference['logits'], tokens, torch), actual_Torch_metrics=native.score(
            candidate['logits'], tokens, torch),
                      full_linear_sites=155, full_linear_calls=counter, panel_calls=len(calls), useful_linear_MACs=useful_MACs,
                      weights=weight_info, calls=calls, firstblock_coupling_records=couplings,
                      original_restored_exact=True, arithmetic_replay_actual_predecessors_exact=True,
            all_22_cache_prefixes_lengths_verified=True,
                      all_33_actual_layer0_KV_tails_verified=True, parameters_buffers_tokens_preserved=True,
            masks_positions_RoPE_unchanged=True,
                      session_logits=identity(out / 'session-logits.npz'), quality_is_explicit_relaxation=True,
                      not_claimed=['full-model instruction-model execution', 'RTL execution', 'whole-model compilation',
            'host ELF execution', 'fullSoC', 'FireSim', 'FPGA', 'existing native gate passing'])
    finally:
        for handle in handles:
            handle.remove()
        torch.set_rng_state(rng)
        np.random.set_state(nrng)
        random.setstate(prng)
        report['caller_torch_rng_restored'] = torch.equal(rng, torch.get_rng_state())


def main():
    from workload_cli import run_report
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('historical', 'native-gate', 'fp32-gate', 'out'):
        p.add_argument('--' + name, type=safe_path, required=True)
    p.add_argument('--atlas-arithmetic', type=safe_path, required=True)
    p.add_argument('--atlas-arithmetic-sha256', required=True)
    p.add_argument('--native-gate-sha256', required=True)
    p.add_argument('--fp32-gate-sha256', required=True)
    p.add_argument('--cpu', type=int)
    p.add_argument('--wall-seconds', type=int, default=14400)
    args = p.parse_args()
    report = {'status': 'failed', 'command': sys.argv,
        'scope': (
            'complete 155-linear planned partition arithmetic-only; exact vector oracle; no '
            'instruction/device/RTL execution'
        )}
    raise SystemExit(run_report(args, lambda args, report: evaluate(args, args.out, report),
        filename='receipt.json', output=args.out, report=report,
        summary=('status', 'wall_seconds', 'peak_rss_kib')))
if __name__ == '__main__':
    main()
