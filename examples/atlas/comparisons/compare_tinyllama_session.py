"""Original native TinyLlama session with four optional prefill substitutions.

The full pinned BF16 LlamaForCausalLM owns every forward, LM head, mask,
position, RoPE, cache update and decode step. Actual Atlas model words replace
only layer-zero Q/K/V/O during the 128-token prefill. This development session
is a partial-Atlas hybrid, with no accelerator application timing or canonical
language-quality claim.
"""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import pickle
import random
import re
import resource
import sys
import time
import zipfile

import numpy as np

import compare_tinyllama_attention as attention
from compare_tinyllama_qkv_attention import admit_projection, require_raw_words
from capture_tinyllama_native import (CACHE_SHA, CORPUS_SHA, LLAMA_SHA, TOKEN_SHA,
    WEIGHT_SHA, array_identity, identity, safe_path, verify_declared_sources, load_provider_validator)
from capture_tinyllama_native_kv import validate_projection_arrays


PROJECTIONS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')


def load_descriptor(record, names):
    """Authenticate and admit flat, bounded numerical members before loading."""
    path = Path(identity(record['path'], record['sha256'])['path'])
    if not names or any(not re.fullmatch(r'[a-z_]+', name) for name in names):
        raise ValueError('Unexpected native descriptor keys')
    required = {name + '.npy' for name in names}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (len(members) != len(required) or {x.filename for x in members} != required or
                sum(x.file_size for x in members) > 32 * 1024**2):
            raise ValueError('Unexpected native descriptor archive members')
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in names}


def scoring_rows(logits, tokens):
    # Row zero predicts token128 from prefill; rows1..31 predict129..159.
    # Row32 follows the final teacher-forced token159 and is deliberately unscored.
    if logits.shape != (33, 32000) or tokens.shape != (1, 160):
        raise ValueError('Require the exact 128+32 teacher-forced session')
    return logits[:32], tokens[0, 128:160]


def require_native_array(actual, expected, label):
    if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise ValueError(label + ' differs in native dtype/shape/raw bytes')


def require_cache_prefix(current, previous, raw):
    if current.shape[:-2] != previous.shape[:-2] or current.shape[-1] != previous.shape[-1]:
        raise ValueError('Native cache geometry changed')
    if current.shape[-2] <= previous.shape[-2] or not np.array_equal(raw(current[..., :previous.shape[-2], :]), raw(previous)):
        raise ValueError('Native cache append changed its existing prefix')


def rng_snapshot(torch):
    return {'torch_cpu': torch.get_rng_state().numpy().copy(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def rng_fingerprints(states):
    return {key: hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()
            for key, value in states.items()}


def evaluate(args, out, report):
    import torch

    before = rng_snapshot(torch)
    try:
        _evaluate(args, out, report)
    finally:
        try:
            observed = rng_snapshot(torch)
        finally:
            # Restore the caller on success and failure, including constructor
            # or library code that may consume a global generator.
            torch.set_rng_state(torch.from_numpy(before['torch_cpu'].copy()))
            np.random.set_state(before['numpy'])
            random.setstate(before['python'])
        restored = rng_snapshot(torch)
        report['rng_bookends'] = {'before': rng_fingerprints(before),
                                  'observed_after': rng_fingerprints(observed),
                                  'restored': rng_fingerprints(restored),
                                  'evaluation_unchanged': rng_fingerprints(before) == rng_fingerprints(observed),
                                  'caller_restored': rng_fingerprints(before) == rng_fingerprints(restored)}
        if report['rng_bookends']['caller_restored'] is not True:
            report['status'] = 'failed'
            raise AssertionError('Caller CPU Torch/NumPy/Python RNG restoration failed')


def _evaluate(args, out, report):
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoTokenizer, cache_utils
    from transformers.models.llama import modeling_llama

    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    if transformers.__version__ != '5.4.0':
        raise ValueError('Require pinned native Transformers5.4.0')
    spec_id = identity(args.spec, args.spec_sha256)
    selected = json.loads(Path(spec_id['path']).read_text())
    if set(selected) != {'projections', 'attention', 'qkv_attention', 'producer_root', 'validator_sha256', 'model_root'} or set(selected['projections']) != set(PROJECTIONS):
        raise ValueError('Unexpected explicit session binding fields')
    producer_root, model_root = safe_path(selected['producer_root']), safe_path(selected['model_root'])
    q = selected['projections']['q_proj']
    validator, validator_id = load_provider_validator(
        producer_root, selected['validator_sha256'], q['provider'], q['provider_sha256'])
    records = [spec_id, validator_id, identity(__file__), identity(attention.__file__),
               identity(inspect.getfile(admit_projection)), identity(inspect.getfile(validate_projection_arrays)),
               identity(modeling_llama.__file__, LLAMA_SHA), identity(cache_utils.__file__, CACHE_SHA),
               identity(inspect.getfile(modeling_llama.create_causal_mask)),
               identity(inspect.getfile(modeling_llama.LlamaConfig)),
               identity(inspect.getfile(torch.nn.Linear)), identity(inspect.getfile(torch.nn.functional.cross_entropy)),
               identity(Path(torch.__file__).parent / 'lib/libtorch_cpu.so')]
    source, manifests, actual, providers, provider_bindings = {}, {}, {}, {}, []
    for name in PROJECTIONS:
        binding = selected['projections'][name]
        if set(binding) != {'bundle', 'manifest_sha256', 'provider', 'provider_sha256'}:
            raise ValueError('Unexpected projection binding fields')
        bundle = safe_path(binding['bundle'])
        native, arrays, target, bound, details = validator.admit_contraction(
            receipt_path=safe_path(binding['provider']), receipt_sha256=binding['provider_sha256'],
            native_bundle=bundle, native_manifest_sha256=binding['manifest_sha256'],
            producer_root=producer_root, model_root=model_root)
        width = 256 if name in ('k_proj', 'v_proj') else 2048
        site = native['site']
        if (native['status'] != 'native_selective_capture_passed' or native['source_dtype'] != 'bfloat16' or
                site['module'] != 'model.layers.0.self_attn.' + name or site['batch'] != 0 or
                site['positions'] != [0, 128] or site['output_channels'] != [0, width] or
                site['k_range'] != [0, 2048] or site['full_k_reduction'] is not True or
                not all(native['negative_controls'].values())):
            raise ValueError('Require complete source-bound native projection: ' + name)
        validate_projection_arrays(arrays, 128, 2048, width)
        verify_declared_sources(native['source'])
        words = (target.view(np.uint32) >> 16).astype(np.uint16)
        actual[name] = torch.from_numpy(words.copy()).view(torch.bfloat16).unsqueeze(0)
        require_raw_words(attention.raw_array(actual[name][0], torch), words, 'Actual projection adapter')
        source[name], manifests[name] = arrays, native
        providers[name] = {'receipt': identity(binding['provider'], binding['provider_sha256']),
                           'native_manifest': identity(bundle / 'manifest.json', binding['manifest_sha256']),
                           'details': details, 'actual_words': array_identity(words)}
        provider_bindings.extend(bound)
        records.append(providers[name]['native_manifest'])
    if len({x['details']['recipe'] for x in providers.values()}) != 1:
        raise ValueError('Require one explicit recipe for the four projections')

    def read_receipt(binding):
        if set(binding) != {'path', 'sha256'}:
            raise ValueError('Unexpected receipt binding')
        record = identity(binding['path'], binding['sha256'])
        records.append(record)
        return json.loads(Path(record['path']).read_text()), record

    baseline, baseline_id = read_receipt(selected['attention'])
    qkv, qkv_id = read_receipt(selected['qkv_attention'])
    if (baseline['status'] != 'passed_native_attention_baseline' or baseline['source_bookends_verified'] is not True or
            baseline['source_manifest'] != providers['q_proj']['native_manifest'] or
            qkv['status'] not in ('passed_native_qkv_attention_gate', 'rejected_native_qkv_attention_gate') or
            qkv['source_bookends_verified'] is not True or qkv['original_native_descriptor_verified'] is not True or
            qkv['candidate_cache_matches_native_rope_actual_k_and_actual_v'] is not True or
            qkv['candidate_o_manifest'] != providers['o_proj']['native_manifest']):
        raise ValueError('Require authenticated original attention and conditional O capture')
    verify_declared_sources(qkv['source_records'])
    verify_declared_sources(qkv['provider_bindings'])
    for name in ('q_proj', 'k_proj', 'v_proj'):
        if qkv['actual_qkv_providers'][name]['receipt'] != providers[name]['receipt']:
            raise ValueError('Conditional O capture uses different actual Q/K/V')
    for name in ('k_proj', 'v_proj'):
        if (manifests[name]['source']['original_query_manifest'] != providers['q_proj']['native_manifest'] or
                manifests[name]['source']['original_attention_baseline'] != baseline_id or
                not np.array_equal(source[name]['A_bf16'], source['q_proj']['A_bf16'])):
            raise ValueError('Original projection source chain differs')
    if manifests['o_proj']['source']['actual_qkv_providers'] != qkv['actual_qkv_providers']:
        raise ValueError('Conditional O manifest differs from actual QKV context')
    descriptor_keys = {'normalized_input', 'native_query', 'effective_query', 'native_key', 'effective_key',
                       'native_value', 'effective_value', 'position_ids', 'attention_mask', 'rope_cos', 'rope_sin',
                       'o_input', 'o_output', 'attention_output', 'residual_output', 'keys', 'values'}
    original = load_descriptor(qkv['original_native_attention'], descriptor_keys)
    candidate_context = load_descriptor(qkv['candidate_native_attention'], descriptor_keys)
    records += [qkv['original_native_attention'], qkv['candidate_native_attention']]
    for name, tensor in (('A', candidate_context['o_input'][0]), ('native_C', candidate_context['o_output'][0])):
        require_raw_words(tensor, source['o_proj'][name + '_bf16'], 'Conditional O ' + name)

    native_source = manifests['q_proj']['source']
    if native_source['python'] != sys.version or native_source['torch'] != torch.__version__ or native_source['transformers'] != transformers.__version__:
        raise ValueError('Native interpreter/framework differs from authenticated source')
    checkpoint = safe_path(native_source['weights']['path']).parent
    identity(checkpoint / 'model.safetensors', WEIGHT_SHA)
    corpus = safe_path(native_source['corpus']['path'])
    identity(corpus, CORPUS_SHA)
    if identity(inspect.getfile(torch.nn.Linear)) != native_source['native_linear'] or identity(Path(torch.__file__).parent / 'lib/libtorch_cpu.so') != native_source['torch_cpu_library']:
        raise ValueError('Native CPU implementation differs')
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False, use_fast=True)
    tokens = tokenizer(corpus.read_text(encoding='utf-8'), return_tensors='pt', add_special_tokens=True)['input_ids'][:, :160].contiguous()
    if tokens.shape != (1, 160) or array_identity(tokens.numpy())['sha256'] != TOKEN_SHA:
        raise ValueError('Original raw-text token sequence differs')
    tokens_before = tokens.clone()
    config = modeling_llama.LlamaConfig.from_pretrained(str(checkpoint), local_files_only=True)
    if (config.hidden_size, config.num_hidden_layers, config.intermediate_size, config.num_attention_heads,
            config.num_key_value_heads, config.vocab_size, config.tie_word_embeddings) != (2048, 22, 5632, 32, 4, 32000, False):
        raise ValueError('Original full model structure differs')
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = modeling_llama.LlamaForCausalLM(config)
    model.model.rotary_emb = modeling_llama.LlamaRotaryEmbedding(config=config, device='cpu')
    expected_parameters = dict(model.named_parameters())
    with safe_open(str(checkpoint / 'model.safetensors'), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(expected_parameters) or len(expected_parameters) != 201:
            raise ValueError('Complete checkpoint/native parameter keyset differs')
        for name, expected in expected_parameters.items():
            value = handle.get_tensor(name)
            if value.dtype != torch.bfloat16 or value.shape != expected.shape:
                raise ValueError('Original BF16 checkpoint parameter differs: ' + name)
            module, _, attribute = name.rpartition('.')
            setattr(model.get_submodule(module), attribute, torch.nn.Parameter(value, requires_grad=False))
    del expected_parameters, value
    model.eval()
    if any(value.is_meta or value.device.type != 'cpu' or value.dtype != torch.bfloat16 for value in model.parameters()):
        raise ValueError('Complete native BF16 parameters were not materialized')
    raw = lambda value: attention.raw_array(value, torch)
    descriptor = lambda values: {key: attention.tensor_identity(value, torch) for key, value in values.items()}
    parameters = lambda: {name: attention.tensor_identity(value, torch) for name, value in model.named_parameters()}
    buffers = lambda: {name: attention.tensor_identity(value, torch) for name, value in model.named_buffers()}
    parameter_before, buffer_before = parameters(), buffers()
    if sum(value.numel() for value in model.parameters()) != 1100048384:
        raise ValueError('Complete checkpoint parameter count differs')
    layer = model.model.layers[0]
    for name in PROJECTIONS:
        require_raw_words(raw(getattr(layer.self_attn, name).weight.T), source[name]['B_bf16'], 'Original checkpoint ' + name)

    def run_session(candidate=False):
        cache = cache_utils.DynamicCache(config=config)
        if any(cache.get_seq_length(i) for i in range(22)):
            raise AssertionError('Fresh full native cache is not empty')
        prefill, last_logits, contexts, cache_steps = None, [], [], []
        for step in range(33):
            observed, counts = {}, {name: 0 for name in PROJECTIONS + ('attention', 'residual')}
            token_slice = tokens[:, :128] if step == 0 else tokens[:, 127 + step:128 + step]
            old = [(x.keys.detach().clone(), x.values.detach().clone()) for x in cache.layers] if step else []

            def projection_hook(name):
                def hook(_module, inputs, output):
                    counts[name] += 1
                    if step == 0:
                        if candidate or name != 'o_proj':
                            admit_projection(inputs[0], output, source[name], torch)
                        else:
                            require_raw_words(raw(inputs[0]), original['o_input'], 'Original O input')
                            require_raw_words(raw(output), original['o_output'], 'Original O output')
                    effective = actual[name].clone() if candidate and step == 0 else output
                    label = {'q_proj': 'query', 'k_proj': 'key', 'v_proj': 'value', 'o_proj': 'o'}[name]
                    if name == 'o_proj':
                        observed['o_input'], observed['native_o_output'] = inputs[0].detach().clone(), output.detach().clone()
                        observed['o_output'] = effective.detach().clone()
                    else:
                        observed['native_' + label] = output.detach().clone()
                        observed['effective_' + label] = effective.detach().clone()
                        if name == 'q_proj':
                            observed['normalized_input'] = inputs[0].detach().clone()
                    if candidate and step == 0:
                        require_raw_words(raw(effective), raw(actual[name]), 'Effective actual ' + name)
                    return effective
                return hook

            def attention_pre(_module, _inputs, kwargs):
                for name in ('position_ids', 'attention_mask'):
                    observed[name] = kwargs[name].detach().clone()
                observed['rope_cos'], observed['rope_sin'] = (value.detach().clone() for value in kwargs['position_embeddings'])

            def attention_post(_module, _inputs, output):
                counts['attention'] += 1
                observed['attention_output'] = output[0].detach().clone()

            def residual_pre(_module, inputs):
                counts['residual'] += 1
                observed['residual_output'] = inputs[0].detach().clone()

            hooks = [getattr(layer.self_attn, name).register_forward_hook(projection_hook(name)) for name in PROJECTIONS]
            hooks += [layer.self_attn.register_forward_pre_hook(attention_pre, with_kwargs=True),
                      layer.self_attn.register_forward_hook(attention_post),
                      layer.post_attention_layernorm.register_forward_pre_hook(residual_pre)]
            try:
                with torch.inference_mode():
                    # Keep native logits_to_keep=0, including all128 prefill rows.
                    output = model(input_ids=token_slice, past_key_values=cache, use_cache=True)
            finally:
                for hook in hooks:
                    hook.remove()
            if any(count != 1 for count in counts.values()) or output.past_key_values is not cache:
                raise AssertionError('Native projection/attention/cache invocation changed')
            if output.logits.dtype != torch.bfloat16 or output.logits.shape != (1, token_slice.shape[1], 32000):
                raise AssertionError('Original native LM head shape/dtype differs')
            if not torch.isfinite(output.logits).all():
                raise ValueError('Nonfinite native full-model logits')
            last_logits.append(output.logits[0, -1].detach().clone())
            expected_length = 128 + step
            for index, item in enumerate(cache.layers):
                if cache.get_seq_length(index) != expected_length or item.keys.shape != (1, 4, expected_length, 64) or item.values.shape != item.keys.shape:
                    raise AssertionError('All22 native cache lengths/geometry must match caller positions')
                if step:
                    require_cache_prefix(item.keys, old[index][0], raw)
                    require_cache_prefix(item.values, old[index][1], raw)
            if len(cache.layers) != 22:
                raise AssertionError('Complete native model did not execute all22 layers')
            q = observed['effective_query'].view(1, token_slice.shape[1], 32, 64).transpose(1, 2)
            k = observed['effective_key'].view(1, token_slice.shape[1], 4, 64).transpose(1, 2)
            v = observed['effective_value'].view(1, token_slice.shape[1], 4, 64).transpose(1, 2)
            with torch.inference_mode():
                _, expected_keys = modeling_llama.apply_rotary_pos_emb(q, k, observed['rope_cos'], observed['rope_sin'])
            require_raw_words(raw(cache.layers[0].keys[..., -token_slice.shape[1]:, :]), raw(expected_keys), 'Layer0 native cache K tail')
            require_raw_words(raw(cache.layers[0].values[..., -token_slice.shape[1]:, :]), raw(v), 'Layer0 native cache V tail')
            expected_positions = torch.arange(expected_length - token_slice.shape[1], expected_length).unsqueeze(0)
            if not torch.equal(observed['position_ids'], expected_positions):
                raise AssertionError('Original native absolute positions differ')
            contexts.append(descriptor({key: observed[key] for key in ('position_ids', 'attention_mask', 'rope_cos', 'rope_sin')}))
            cache_steps.append([descriptor({'keys': item.keys, 'values': item.values}) for item in cache.layers])
            if step == 0:
                observed['keys'], observed['values'] = cache.layers[0].keys.detach().clone(), cache.layers[0].values.detach().clone()
                if candidate:
                    for key in ('normalized_input', 'native_query', 'native_key', 'native_value', 'effective_query', 'effective_key', 'effective_value',
                                'position_ids', 'attention_mask', 'rope_cos', 'rope_sin', 'o_input', 'keys', 'values'):
                        require_native_array(raw(observed[key]), candidate_context[key], 'Actual QKV native prefill ' + key)
                else:
                    if {key: descriptor(observed)[key] for key in baseline['native_arrays']} != baseline['native_arrays']:
                        raise AssertionError('Full native layer0 prefill differs from authenticated original attention descriptor')
                    for key in descriptor_keys:
                        require_native_array(raw(observed[key]), original[key], 'Original native prefill ' + key)
                prefill = {key: value for key, value in observed.items()}
            del output, old
        return {'logits': torch.stack(last_logits), 'prefill': prefill,
                'contexts': contexts, 'cache_steps': cache_steps}

    def summary(session):
        return {'logits': attention.tensor_identity(session['logits'], torch),
                'prefill': descriptor(session['prefill']), 'contexts': session['contexts'], 'cache_steps': session['cache_steps']}

    reference = run_session()
    if summary(run_session()) != summary(reference):
        raise AssertionError('Original full native session repeat changed raw outputs/cache')
    candidate = run_session(True)
    if summary(run_session(True)) != summary(candidate):
        raise AssertionError('Actual-prefill hybrid repeat changed raw outputs/cache')
    if candidate['contexts'] != reference['contexts']:
        raise AssertionError('Hybrid changed original native masks/positions/RoPE')
    if summary(run_session()) != summary(reference):
        raise AssertionError('Removing actual adapters did not restore full native session')
    if not torch.equal(tokens, tokens_before) or parameters() != parameter_before or buffers() != buffer_before:
        raise AssertionError('Full native caller tokens/parameters/buffers changed')

    def score(session):
        logits, labels = scoring_rows(session['logits'], tokens)
        losses = torch.nn.functional.cross_entropy(logits.float(), labels, reduction='none')
        if not torch.isfinite(losses).all():
            raise ValueError('Nonfinite32-token NLL')
        mean = float(losses.mean())
        predicted = logits.argmax(-1)
        return {'scored_target_positions': [128, 160], 'target_count': 32,
                'nll': losses.tolist(), 'mean_nll': mean,
                'perplexity': math.exp(mean) if mean < math.log(sys.float_info.max) else None,
                'perplexity_overflow': mean >= math.log(sys.float_info.max),
                'argmax_ids': predicted.tolist(), 'argmax_correct': int((predicted == labels).sum())}

    errors = {'scored_logits': attention.error(candidate['logits'][:32].float().numpy(), reference['logits'][:32].float().numpy()),
              'unscored_after_token159_logits': attention.error(candidate['logits'][32:].float().numpy(), reference['logits'][32:].float().numpy()),
              'prefill_attention': attention.error(candidate['prefill']['attention_output'].float().numpy(), reference['prefill']['attention_output'].float().numpy()),
              'prefill_residual': attention.error(candidate['prefill']['residual_output'].float().numpy(), reference['prefill']['residual_output'].float().numpy())}
    np.savez(out / 'session-logits.npz', original_bf16=raw(reference['logits']), candidate_bf16=raw(candidate['logits']),
             original_logits=reference['logits'].float().numpy(), candidate_logits=candidate['logits'].float().numpy(),
             tokens=tokens.numpy(), targets=tokens[0, 128:160].numpy())
    for label, session in (('original', reference), ('candidate', candidate)):
        np.savez(out / (label + '-prefill.npz'), **{key: raw(value) for key, value in session['prefill'].items()})
    validator.recheck(provider_bindings)
    for native in manifests.values():
        verify_declared_sources(native['source'])
    verify_declared_sources(qkv['source_records'])
    verify_declared_sources(qkv['provider_bindings'])
    for record in records:
        identity(record['path'], record['sha256'])
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak > 6 * 1024**3:
        raise MemoryError('Complete native session exceeded its6GiB RSS lease')
    gate = all(value['allclose_rtol1e4_atol1e4'] for value in errors.values())
    report.update(status='passed_native_session_gate' if gate else 'rejected_native_session_gate',
                  native_session_gate_passed=gate, original_metrics=score(reference), candidate_metrics=score(candidate),
                  errors=errors, actual_providers=providers, original_attention=baseline_id, qkv_context=qkv_id,
                  original_summary=summary(reference), candidate_summary=summary(candidate),
                  session_logits=identity(out / 'session-logits.npz'), original_prefill=identity(out / 'original-prefill.npz'),
                  candidate_prefill=identity(out / 'candidate-prefill.npz'),
                  full_native_parameters=parameter_before, original_buffers=buffer_before,
                  source_records=records, provider_bindings=provider_bindings, source_bookends_verified=True,
                  native_and_hybrid_repeats_raw_exact=True, native_restored_after_adapters_raw_exact=True,
                  all22_cache_lengths_and_prefixes_verified=True, native_layer0_cache_tails_verified=True,
                  actual_prefill_adapters_raw_exact=True, decode_substitutions=0,
                  caller_tokens_parameters_buffers_preserved=True, full_model_executed=True,
                  prefill_tokens=128, teacher_forced_decode_inputs=32, final_cache_tokens=160,
                  scored_tokens=32, unscored_final_next_token_distribution=True,
                  cpu_threads=1, peak_rss_bytes=peak, wall_seconds=time.monotonic() - started,
                  canonical_language_quality_claim=False, complete_atlas_execution=False,
                  application_accelerator_timing_claim=False,
                  claim='One fixed development session; actual layer0 prefill Q/K/V/O, native remainder and decode')


def main():
    from merlin.common.paths import artifacts_dir

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=safe_path, required=True)
    parser.add_argument('--spec-sha256', required=True)
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
        raise
    finally:
        (args.out / 'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'native_session_gate_passed', 'wall_seconds', 'peak_rss_bytes')}))
    if report['native_session_gate_passed'] is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
