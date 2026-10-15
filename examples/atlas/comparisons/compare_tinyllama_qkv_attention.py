"""Admit actual Q/K/V, run native attention, and capture its conditional O site.

The original native attention producer remains unchanged. Observation retains
its model and original token input; subsequent calls use that same native
forward with fresh caches. Native RoPE/GQA/masks/attention own all such math.
"""

import argparse
import inspect
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

import compare_tinyllama_attention as attention
from capture_tinyllama_native import (array_identity, check_memory, identity,
                                     safe_path, verify_declared_sources, load_provider_validator)
from capture_tinyllama_native_kv import (projection_dimensions, source_controls,
                                        validate_projection_arrays)


def admit_projection(inputs, output, source, torch):
    if (inputs.dtype != torch.bfloat16 or output.dtype != torch.bfloat16 or
            inputs.shape != (1,) + source['A_bf16'].shape or
            output.shape != (1,) + source['native_C_bf16'].shape or
            not np.array_equal(attention.raw_array(inputs[0], torch), source['A_bf16']) or
            not np.array_equal(attention.raw_array(output[0], torch), source['native_C_bf16'])):
        raise ValueError('Native projection invocation differs from original source')


def require_raw_words(actual, expected, label):
    if (actual.dtype != np.uint16 or expected.dtype != np.uint16 or
            actual.shape != expected.shape or not np.array_equal(actual, expected)):
        raise ValueError(label + ' differs from native transform of admitted actual words')


def evaluate(args, out, report):
    import torch
    from transformers import cache_utils
    from transformers.models.llama import modeling_llama

    started = time.monotonic()
    baseline_id = identity(args.attention_bundle / 'receipt.json', args.attention_receipt_sha256)
    baseline = json.loads(Path(baseline_id['path']).read_text())
    if baseline['status'] != 'passed_native_attention_baseline' or baseline['source_bookends_verified'] is not True:
        raise ValueError('Require original authenticated native attention baseline')
    validator, validator_id = load_provider_validator(
        args.producer_root, args.validator_sha256, args.q_provider, args.q_provider_sha256)
    source, manifests, replacement, provider_records, providers = {}, {}, {}, [], {}
    records = [baseline_id, validator_id, identity(__file__), identity(attention.__file__),
               identity(inspect.getfile(source_controls)), identity(inspect.getfile(modeling_llama.apply_rotary_pos_emb))]
    for name, bundle, manifest_sha, receipt, receipt_sha in (
            ('q_proj', args.query_bundle, args.query_manifest_sha256, args.q_provider, args.q_provider_sha256),
            ('k_proj', args.key_bundle, args.key_manifest_sha256, args.k_provider, args.k_provider_sha256),
            ('v_proj', args.value_bundle, args.value_manifest_sha256, args.v_provider, args.v_provider_sha256)):
        native, arrays, actual, bindings, details = validator.admit_contraction(
            receipt_path=receipt, receipt_sha256=receipt_sha, native_bundle=bundle,
            native_manifest_sha256=manifest_sha, producer_root=args.producer_root, model_root=args.model_root)
        width = arrays['native_C'].shape[1]
        site = native['site']
        if (native['status'] != 'native_selective_capture_passed' or native['source_dtype'] != 'bfloat16' or
                site['module'] != 'model.layers.0.self_attn.' + name or site['batch'] != 0 or
                site['positions'] != [0, 128] or site['output_channels'] != [0, width] or
                site['k_range'] != [0, 2048] or site['full_k_reduction'] is not True or
                not all(native['negative_controls'].values())):
            raise ValueError('Require complete original native projection source: ' + name)
        validate_projection_arrays(arrays, 128, 2048, width)
        verify_declared_sources(native['source'])
        source[name], manifests[name] = arrays, native
        words = (actual.view(np.uint32) >> 16).astype(np.uint16)
        replacement[name] = torch.from_numpy(words.copy()).view(torch.bfloat16).unsqueeze(0)
        require_raw_words(attention.raw_array(replacement[name][0], torch), words, 'Actual projection adapter')
        provider_records.extend(bindings)
        providers[name] = {'receipt': identity(receipt, receipt_sha), 'native_manifest': identity(bundle / 'manifest.json', manifest_sha),
                           'details': details, 'actual_words': array_identity(words), 'adapter_raw_exact': True}
        records.append(providers[name]['native_manifest'])
    query_id = providers['q_proj']['native_manifest']
    if baseline['source_manifest'] != query_id:
        raise ValueError('Original attention and full Q source disagree')
    for name in ('k_proj', 'v_proj'):
        if (manifests[name]['source']['original_query_manifest'] != query_id or
                manifests[name]['source']['original_attention_baseline'] != baseline_id or
                not np.array_equal(source[name]['A_bf16'], source['q_proj']['A_bf16'])):
            raise ValueError('K/V source chain differs from original Q/attention')
    if len({provider['details']['recipe'] for provider in providers.values()}) != 1:
        raise ValueError('This bounded experiment requires one declared target recipe for Q/K/V')

    holders, token_inputs, hooks = [], [], []

    def observe_model(module, _inputs):
        if type(module) is not modeling_llama.LlamaModel:
            return
        if holders:
            if holders[0] is not module:
                raise AssertionError('More than one native model observed')
            return
        holders.append(module)

        def embedding_input(_module, inputs):
            token_inputs.append(inputs[0].detach().clone())

        hooks.append(module.embed_tokens.register_forward_pre_hook(embedding_input))

    observer = torch.nn.modules.module.register_module_forward_pre_hook(observe_model)
    original_out = out / 'producer-native-baseline'
    original_out.mkdir()
    original_report = {'status': 'failed'}
    try:
        attention.evaluate(SimpleNamespace(native_bundle=args.query_bundle,
                           native_manifest_sha256=args.query_manifest_sha256, provider_receipt=None),
                           original_out, original_report)
    except Exception as exc:
        original_report.update(status='failed', error=repr(exc))
        raise
    finally:
        observer.remove()
        for hook in hooks:
            hook.remove()
        (original_out / 'receipt.json').write_text(json.dumps(original_report, indent=2, allow_nan=False) + '\n')
    if (original_report['native_arrays'] != baseline['native_arrays'] or len(holders) != 1 or len(token_inputs) != 2 or
            not torch.equal(token_inputs[0], token_inputs[1]) or token_inputs[0].shape != (1, 128)):
        raise AssertionError('Retained model/token/native baseline descriptor differs')
    model, tokens = holders[0], token_inputs[0]
    tokens_before = tokens.clone()
    layer, config = model.layers[0], model.config
    hidden, width, head_dim = projection_dimensions(config, layer.self_attn.k_proj)
    if hidden != 2048 or width != source['k_proj']['native_C'].shape[1]:
        raise ValueError('Native config differs from projection source')
    for name in source:
        projection = getattr(layer.self_attn, name)
        if not np.array_equal(attention.raw_array(projection.weight.T, torch), source[name]['B_bf16']):
            raise ValueError('Original checkpoint weights differ: ' + name)
    parameters = lambda: {'model.' + name: attention.tensor_identity(value, torch)
                         for name, value in model.named_parameters() if not value.is_meta}
    buffers = lambda: {name: attention.tensor_identity(value, torch) for name, value in model.rotary_emb.named_buffers()}
    parameter_before, buffers_before = parameters(), buffers()
    if parameter_before != original_report['materialized_parameters']:
        raise ValueError('Retained model parameters differ from original native producer')

    class AttentionCaptured(Exception):
        pass

    def forward(candidate=False):
        observed, counts = {}, {name: 0 for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'attention', 'stop')}
        cache = cache_utils.DynamicCache(config=config)
        if cache.get_seq_length():
            raise AssertionError('Fresh cache is not empty')

        def projection_hook(name):
            def hook(_module, inputs, output):
                admit_projection(inputs[0], output, source[name], torch)
                counts[name] += 1
                label = {'q_proj': 'query', 'k_proj': 'key', 'v_proj': 'value'}[name]
                observed['native_' + label] = output.detach().clone()
                effective = replacement[name].clone() if candidate else output
                observed['effective_' + label] = effective.detach().clone()
                if name == 'q_proj':
                    observed['normalized_input'] = inputs[0].detach().clone()
                return effective
            return hook

        def attention_pre(_module, _inputs, kwargs):
            observed['position_ids'] = kwargs['position_ids'].detach().clone()
            observed['attention_mask'] = kwargs['attention_mask'].detach().clone()
            observed['rope_cos'], observed['rope_sin'] = (value.detach().clone() for value in kwargs['position_embeddings'])

        def o_pre(_module, inputs):
            observed['o_input'] = inputs[0].detach().clone()

        def o_post(_module, _inputs, output):
            observed['o_output'] = output.detach().clone()
            counts['o_proj'] += 1

        def attention_post(_module, _inputs, output):
            observed['attention_output'] = output[0].detach().clone()
            counts['attention'] += 1

        def stop(_module, inputs):
            observed['residual_output'] = inputs[0].detach().clone()
            counts['stop'] += 1
            raise AttentionCaptured()

        local_hooks = [getattr(layer.self_attn, name).register_forward_hook(projection_hook(name)) for name in source]
        local_hooks += [layer.self_attn.register_forward_pre_hook(attention_pre, with_kwargs=True),
                        layer.self_attn.o_proj.register_forward_pre_hook(o_pre),
                        layer.self_attn.o_proj.register_forward_hook(o_post),
                        layer.self_attn.register_forward_hook(attention_post),
                        layer.post_attention_layernorm.register_forward_pre_hook(stop)]
        try:
            with torch.inference_mode():
                try:
                    model(input_ids=tokens, past_key_values=cache, use_cache=True)
                except AttentionCaptured:
                    pass
                else:
                    raise AssertionError('Native model did not stop at attention residual')
        finally:
            for hook in local_hooks:
                hook.remove()
        if any(count != 1 for count in counts.values()) or cache.get_seq_length() != 128 or any(cache.get_seq_length(i) for i in range(1, 22)):
            raise AssertionError('Wrong native invocation/cache/layer bounds')
        require_raw_words(attention.raw_array(observed['o_output'], torch),
                          attention.raw_array(observed['attention_output'], torch), 'Native O/attention output boundary')
        observed['keys'], observed['values'] = cache.layers[0].keys.detach().clone(), cache.layers[0].values.detach().clone()
        if candidate:
            q = replacement['q_proj'].view(1, 128, config.num_attention_heads, head_dim).transpose(1, 2)
            k = replacement['k_proj'].view(1, 128, config.num_key_value_heads, head_dim).transpose(1, 2)
            v = replacement['v_proj'].view(1, 128, config.num_key_value_heads, head_dim).transpose(1, 2)
            with torch.inference_mode():
                _, native_rope_keys = modeling_llama.apply_rotary_pos_emb(q, k, observed['rope_cos'], observed['rope_sin'])
            require_raw_words(attention.raw_array(observed['keys'], torch), attention.raw_array(native_rope_keys, torch), 'Candidate K cache')
            require_raw_words(attention.raw_array(observed['values'], torch), attention.raw_array(v, torch), 'Candidate V cache')
        return observed

    descriptor = lambda values: {key: attention.tensor_identity(value, torch) for key, value in values.items()}
    reference, repeated = forward(), forward()
    if descriptor(reference) != descriptor(repeated) or {key: descriptor(reference)[key] for key in baseline['native_arrays']} != baseline['native_arrays']:
        raise AssertionError('Complete original native attention descriptor differs')
    candidate, repeated_candidate = forward(True), forward(True)
    if descriptor(candidate) != descriptor(repeated_candidate):
        raise AssertionError('Candidate repeat changed raw bytes')
    for name, label in (('q_proj', 'query'), ('k_proj', 'key'), ('v_proj', 'value')):
        require_raw_words(attention.raw_array(candidate['effective_' + label], torch),
                          attention.raw_array(replacement[name], torch), 'Effective actual ' + label)
    unchanged = ('normalized_input', 'native_query', 'native_key', 'native_value', 'position_ids', 'attention_mask', 'rope_cos', 'rope_sin')
    if any(attention.tensor_identity(reference[key], torch) != attention.tensor_identity(candidate[key], torch) for key in unchanged):
        raise AssertionError('Candidate changed original source input/projections/masks/positions/RoPE')
    if descriptor(forward()) != descriptor(reference):
        raise AssertionError('Removing candidate hooks did not restore original native attention')
    if not torch.equal(tokens, tokens_before) or parameters() != parameter_before or buffers() != buffers_before:
        raise AssertionError('Original native tokens/parameters/rotary buffers changed')
    errors = {key: attention.error(candidate[key].float().numpy(), reference[key].float().numpy())
              for key in ('attention_output', 'residual_output', 'o_input', 'o_output')}
    projection_errors = {name: attention.error(replacement[name][0].float().numpy(), source[name]['native_C']) for name in source}
    gate = all(errors[key]['allclose_rtol1e4_atol1e4'] for key in ('attention_output', 'residual_output'))
    for label, values in (('original-native-attention', reference), ('candidate-native-attention', candidate)):
        np.savez(out / (label + '.npz'), **{key: attention.raw_array(value, torch) for key, value in values.items()})
    arrays = {}
    for name, tensor in (('A', candidate['o_input'][0]), ('B', layer.self_attn.o_proj.weight.T), ('native_C', candidate['o_output'][0])):
        arrays[name], arrays[name + '_bf16'] = tensor.detach().contiguous().float().numpy().copy(), attention.raw_array(tensor, torch)
    expected = {name: arrays[name + '_bf16'].copy() for name in ('A', 'B', 'native_C')}
    validate_projection_arrays(arrays, 128, hidden, hidden, expected)
    controls = source_controls(arrays, 128, hidden, hidden, expected)
    o_out = out / 'candidate-o'
    o_out.mkdir()
    np.savez(o_out / 'operands.npz', **arrays)
    validator.recheck(provider_records)
    for native in manifests.values():
        verify_declared_sources(native['source'])
    for record in records + original_report['source_records']:
        identity(record['path'], record['sha256'])
    peak = check_memory()
    o_source = dict(manifests['q_proj']['source'])
    o_source.update(original_query_capture_script=o_source['capture_script'], capture_script=identity(__file__),
                    original_native_attention=baseline_id, actual_qkv_providers=providers,
                    native_attention_producer=identity(attention.__file__),
                    native_rope_source=identity(inspect.getfile(modeling_llama.apply_rotary_pos_emb)))
    manifest = {'schema_version': 1, 'status': 'native_selective_capture_passed', 'source_dtype': 'bfloat16',
                'npz_sha256': identity(o_out / 'operands.npz')['sha256'], 'source': o_source,
                'site': {'module': 'model.layers.0.self_attn.o_proj', 'batch': 0, 'positions': [0, 128],
                         'output_channels': [0, hidden], 'k_range': [0, hidden], 'native_execution_shape': [1, 128, hidden],
                         'full_k_reduction': True, 'candidate_context': 'Authenticated actual Q/K/V before original native RoPE/GQA/attention'},
                'arrays': {key: array_identity(value) for key, value in arrays.items()}, 'negative_controls': controls,
                'capture': {'source_bookends_verified': True, 'native_and_candidate_repeats_raw_exact': True,
                            'native_restored_after_substitution_raw_exact': True, 'candidate_cache_matches_native_rope_actual_k_and_actual_v': True,
                            'native_attention_gate_passed': gate, 'native_o_boundary': 'Original checkpoint Linear on actual-QKV candidate attention context'},
                'claim_limits': ['conditional_native_O_source_only', 'no_target_O_execution', 'no_full_model_or_language_quality_claim']}
    (o_out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    report.update(status='passed_native_qkv_attention_gate' if gate else 'rejected_native_qkv_attention_gate',
                  native_attention_gate_passed=gate, errors=errors, actual_qkv_providers=providers,
                  projection_diagnostics=projection_errors,
                  original_native_attention=identity(out / 'original-native-attention.npz'),
                  candidate_native_attention=identity(out / 'candidate-native-attention.npz'),
                  original_native_descriptor_verified=True, candidate_repeat_raw_exact=True,
                  candidate_cache_matches_native_rope_actual_k_and_actual_v=True, unchanged_native_inputs_masks_positions_rope=True,
                  native_restored_after_substitution_raw_exact=True, source_bookends_verified=True,
                  original_token_input=attention.tensor_identity(tokens, torch), caller_tokens_preserved=True,
                  native_parameters_and_rotary_buffers_preserved=True, materialized_parameters=parameter_before,
                  native_configuration=original_report['native_model_configuration'],
                  source_records=records + original_report['source_records'], provider_bindings=provider_records,
                  candidate_o_manifest=identity(o_out / 'manifest.json'), wall_seconds=time.monotonic() - started,
                  peak_rss_bytes=peak, cpu_threads=1, full_model_executed=False, mlp_executed=False, language_quality_claim=False)


def main():
    from merlin.common.paths import artifacts_dir

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('query', 'key', 'value'):
        parser.add_argument('--' + name + '-bundle', type=safe_path, required=True)
        parser.add_argument('--' + name + '-manifest-sha256', required=True)
    parser.add_argument('--attention-bundle', type=safe_path, required=True)
    parser.add_argument('--attention-receipt-sha256', required=True)
    for name in ('q', 'k', 'v'):
        parser.add_argument('--' + name + '-provider', type=safe_path, required=True)
        parser.add_argument('--' + name + '-provider-sha256', required=True)
    parser.add_argument('--producer-root', type=safe_path, required=True)
    parser.add_argument('--validator-sha256', required=True)
    parser.add_argument('--model-root', type=safe_path, required=True)
    parser.add_argument('--out', type=safe_path, required=True)
    args = parser.parse_args()
    if not args.out.is_relative_to(safe_path(artifacts_dir())):
        parser.error('--out must remain inside this Merlin artifacts_dir')
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'argv': sys.argv, 'python': sys.version}
    try:
        evaluate(args, args.out, report)
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        candidate_manifest = args.out / 'candidate-o/manifest.json'
        if candidate_manifest.is_file():
            failed_capture = json.loads(candidate_manifest.read_text())
            failed_capture.update(status='failed', error=repr(exc))
            candidate_manifest.write_text(json.dumps(failed_capture, indent=2, allow_nan=False) + '\n')
        raise
    finally:
        (args.out / 'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'native_attention_gate_passed', 'wall_seconds', 'peak_rss_bytes')}))
    if report['native_attention_gate_passed'] is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
