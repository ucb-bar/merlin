"""Frozen TinyLlama BF16 native session, graph inventory and projection sources.

The original 128+32-token native precision is BF16. FP32 views in source bundles
retain exactly those BF16 values; they are not a FP32 native model reference.
"""
from merlin.common.jsonio import write_pretty_json
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import sys
import numpy as np
from capture_tinyllama_native import (WEIGHT_SHA, CORPUS_SHA, TOKEN_SHA, LLAMA_SHA, CACHE_SHA,
                                     safe_path, identity, array_identity)
from compare_tinyllama_attention import raw_array, tensor_identity

PROJECTIONS =('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
             'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
HISTORICAL_SHA = 'e4e6381223871e71db8ee787525c04b7abcbfc7c98dd4e09f3430dd78bf18063'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(historical, records, precision='bfloat16'):
    import torch
    import transformers
    from transformers import AutoTokenizer, cache_utils
    from transformers.models.llama import modeling_llama
    from safetensors import safe_open
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    require(transformers.__version__ == '5.4.0', 'require original Transformers 5.4.0')
    old_id = identity(historical, HISTORICAL_SHA)
    records.append(old_id)
    old = json.loads(Path(old_id['path']).read_text())
    q_record = next(r for r in old['source_records'] if Path(r['path']).name == 'manifest.json' and 'native-q-full-v1' in r[
        'path'])
    q = json.loads(Path(identity(q_record['path'], q_record['sha256'])['path']).read_text())
    records.append(q_record)
    source = q['source']
    checkpoint = safe_path(source['weights']['path']).parent
    records +=[identity(checkpoint / 'model.safetensors', WEIGHT_SHA), identity(source['corpus']['path'], CORPUS_SHA),
                identity(modeling_llama.__file__, LLAMA_SHA), identity(cache_utils.__file__, CACHE_SHA),
                identity(inspect.getfile(torch.nn.Linear), source['native_linear']['sha256']),
                identity(Path(torch.__file__).parent / 'lib/libtorch_cpu.so', source['torch_cpu_library']['sha256']),
                identity(__file__), identity(inspect.getfile(raw_array)), identity(inspect.getfile(safe_path))]
    for r in source['checkpoint_files']:
        records.append(identity(r['path'], r['sha256']))
    require(sys.version == source['python'] and torch.__version__ == source['torch'], 'original interpreter/Torch differs')
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False, use_fast=True)
    tokens = tokenizer(safe_path(source['corpus']['path']).read_text(), return_tensors='pt', add_special_tokens=True)[
        'input_ids'][:,
        :160].contiguous()
    require(tuple(tokens.shape) ==(1, 160) and array_identity(tokens.numpy())['sha256'] == TOKEN_SHA,
        'frozen raw text tokens differ')
    config = modeling_llama.LlamaConfig.from_pretrained(str(checkpoint), local_files_only=True)
    require((config.hidden_size, config.intermediate_size, config.num_hidden_layers, config.num_attention_heads,
             config.num_key_value_heads, config.vocab_size, config.tie_word_embeddings) ==(2048, 5632, 22, 32, 4, 32000, False),
        'model structure differs')
    config._attn_implementation = 'eager'
    with torch.device('meta'):
        model = modeling_llama.LlamaForCausalLM(config)
    model.model.rotary_emb = modeling_llama.LlamaRotaryEmbedding(config=config, device='cpu')
    parameters = dict(model.named_parameters())
    with safe_open(str(checkpoint / 'model.safetensors'), framework='pt', device='cpu') as handle:
        require(set(handle.keys()) == set(parameters) and len(parameters) == 201, 'complete parameter keyset differs')
        for name, expected in parameters.items():
            value = handle.get_tensor(name)
            require(value.dtype == torch.bfloat16 and value.shape == expected.shape, 'checkpoint parameter differs: ' + name)
            parent, _, attribute = name.rpartition('.')
            setattr(model.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))
    del parameters, value
    require(sum(p.numel() for p in model.parameters()) == 1100048384, 'complete parameter count differs')
    model.eval()
    if precision == 'float32':
        model.float()
    return model, tokens, old, torch, cache_utils, modeling_llama, q


def state(model, torch):
    return {name: tensor_identity(value, torch) for name, value in model.named_parameters()},\
           {name: tensor_identity(value, torch) for name, value in model.named_buffers()}


def run_session(model, tokens, torch, cache_utils, modeling_llama, adapter=None):
    raw = lambda t: raw_array(t, torch)
    desc = lambda d: {name: tensor_identity(value, torch) for name, value in d.items()}
    layer = model.model.layers[0]
    cache = cache_utils.DynamicCache(config=model.config)
    logits =[]
    contexts =[]
    cache_steps =[]
    captured = {}
    prefill = None
    for step in range(33):
        observed = {}
        counts = {name: 0 for name in PROJECTIONS}
        handles =[]
        previous =[(raw(item.keys), raw(item.values)) for item in cache.layers] if step else []
        token_slice = tokens[:,:128] if not step else tokens[:, 127 + step:128 + step]
        for name in PROJECTIONS:
            module = layer.get_submodule(name)
            def hook(module, inputs, output, name=name):
                counts[name] += 1
                effective = adapter(name, inputs[0], output, module.weight) if adapter is not None and step == 0 else output
                if not step:
                    captured[name] = {'A': inputs[0][0].detach().float().numpy().copy(),
                                    'B': module.weight.T.detach().float().numpy().copy(),
                                    'native_C': output[0].detach().float().numpy().copy()}
                    label = {'self_attn.q_proj': 'query', 'self_attn.k_proj': 'key', 'self_attn.v_proj': 'value',
                        'self_attn.o_proj': 'o'}.get(name)
                    if label == 'o':
                        observed.update(o_input=inputs[0].detach().clone(), native_o_output=output.detach().clone(),
                            o_output=effective.detach().clone())
                    elif label:
                        observed['native_' + label] = output.detach().clone()
                        observed['effective_' + label] = effective.detach().clone()
                        if label == 'query':
                            observed['normalized_input'] = inputs[0].detach().clone()
                return effective if effective is not output else None
            handles.append(module.register_forward_hook(hook))
        def attention_pre(module, inputs, kwargs):
            observed.update(position_ids=kwargs['position_ids'].detach().clone(), attention_mask=kwargs['attention_mask'].detach(
                ).clone())
            observed['rope_cos'], observed['rope_sin'] =(v.detach().clone() for v in kwargs['position_embeddings'])
        def attention_post(module, inputs, output):
            observed['attention_output'] = output[0].detach().clone()
        def residual_pre(module, inputs):
            observed['residual_output'] = inputs[0].detach().clone()
        def entry_pre(module, inputs):
            if not step:
                observed['entry'] = inputs[0].detach().clone()
        handles +=[layer.self_attn.register_forward_pre_hook(attention_pre, with_kwargs=True),
            layer.self_attn.register_forward_hook(attention_post),
                    layer.post_attention_layernorm.register_forward_pre_hook(residual_pre),
            layer.input_layernorm.register_forward_pre_hook(entry_pre)]
        try:
            with torch.inference_mode():
                output = model(input_ids=token_slice, past_key_values=cache, use_cache=True)
        finally:
            for handle in handles:
                handle.remove()
        require(all(n == 1 for n in counts.values()) and output.past_key_values is cache, 'native invocation/cache changed')
        require(tuple(output.logits.shape) ==(1, token_slice.shape[1], 32000) and torch.isfinite(output.logits).all(),
            'native logits geometry/finite differs')
        logits.append(output.logits[0, -1].detach().clone())
        length = 128 + step
        require(len(cache.layers) == 22, 'full 22-layer execution missing')
        for i, item in enumerate(cache.layers):
            require(cache.get_seq_length(i) == length and tuple(item.keys.shape) ==(1, 4, length,
                64) and item.values.shape == item.keys.shape, 'cache geometry/length differs')
            if step:
                require(np.array_equal(raw(item.keys[...,:length - 1,:]), previous[i][0]) and np.array_equal(raw(item.values[...,
                    :length - 1,:]), previous[i][1]), 'cache prefix changed')
        require(torch.equal(observed['position_ids'], torch.arange(length - token_slice.shape[1], length).unsqueeze(0)),
            'absolute positions differ')
        contexts.append(desc({k: observed[k] for k in ('position_ids', 'attention_mask', 'rope_cos', 'rope_sin')}))
        cache_steps.append([desc({'keys': item.keys, 'values': item.values}) for item in cache.layers])
        if not step:
            observed.update(keys=cache.layers[0].keys.detach().clone(), values=cache.layers[0].values.detach().clone())
            prefill = observed
        del output, previous
    return {'logits': torch.stack(logits), 'contexts': contexts, 'cache_steps': cache_steps, 'prefill': prefill,
        'projections': captured}


def summary(session, torch):
    return {'logits': tensor_identity(session['logits'], torch), 'contexts': session['contexts'], 'cache_steps': session[
        'cache_steps'],
            'prefill': {key: tensor_identity(value, torch) for key, value in session['prefill'].items() if key != 'entry'}}


def score(logits, tokens, torch):
    losses = torch.nn.functional.cross_entropy(logits[:32].float(), tokens[0, 128:160], reduction='none')
    mean = float(losses.mean())
    pred = logits[:32].argmax(-1)
    return {'scored_tokens': 32, 'mean_nll': mean, 'perplexity': math.exp(mean), 'argmax_ids': pred.tolist(),
        'argmax_correct': int((
        pred == tokens[0, 128:160]).sum()), 'nll': losses.tolist()}


def native(args, out, report):
    import torch
    rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    python_rng = random.getstate()
    records =[]
    try:
        model, tokens, old, torch, cache_utils, modeling_llama, q = prepare(args.historical, records)
        params, buffers = state(model, torch)
        require(params == old['full_native_parameters'], 'all 201 materialized BF16 parameters differ')
        tokens_before = tokens.clone()
        first = run_session(model, tokens, torch, cache_utils, modeling_llama)
        first_summary = summary(first, torch)
        require(first_summary == old['original_summary'], 'fresh fullnative session differs from frozen original')
        repeated = run_session(model, tokens, torch, cache_utils, modeling_llama)
        require(summary(repeated, torch) == first_summary, 'native repeat differs')
        require(state(model, torch) ==(params, buffers) and torch.equal(tokens, tokens_before),
            'parameters/buffers/tokens changed')
        destinations = {}
        for name, arrays in first['projections'].items():
            directory = out / name
            directory.mkdir()
            np.savez(directory / 'operands.npz', **arrays)
            manifest = {'schema_version': 1, 'source_dtype': 'float32', 'npz_sha256': identity(directory / 'operands.npz')[
                'sha256'],
                      'source': {'kind': 'fresh_native_BF16_exact_float32_views', 'original_precision': 'torch.bfloat16',
                'native_session_script': identity(__file__),
                                'checkpoint': q['source']['weights'], 'frozen_tokens': q['source']['tokens'],
                'no_FP32_native_reference_claim': True},
                      'site': {'module': 'model.layers.0.' + name, 'positions': [0, 128], 'M': 128, 'K': arrays['A'].shape[1],
                'N': arrays[
                'B'].shape[1]},
                      'arrays': {k: array_identity(a) for k, a in arrays.items()}}
            write_pretty_json(directory / 'manifest.json', manifest, sort_keys=False, allow_nan=False)
            destinations[name] = identity(directory / 'manifest.json')
        for key in ('A', 'B', 'native_C'):
            require(array_identity(first['projections']['self_attn.q_proj'][key]) == q['arrays'][key],
                'freshQ source differs from historical immutable source')
        np.savez(out / 'session.npz', logits_bf16=raw_array(first['logits'], torch), tokens=tokens.numpy(), entry_bf16=raw_array(
            first[
            'prefill']['entry'], torch))
        for r in records:
            identity(r['path'], r['sha256'])
        report.update(status='passed_frozen_native_BF16_session', source_records=records, full_parameters=params, buffers=buffers,
                      source_bookends_verified=True, native_repeat_exact=True, all_22_cache_lengths_prefixes_exact=True,
                      frozen_original_summary_exact=True, fresh_Q_views_equal_historical=True, original_summary=first_summary,
                      metrics=score(first['logits'], tokens, torch), projection_manifests=destinations,
                      native_precision='torch.bfloat16; FP32 view artifacts do not claim FP32-native arithmetic',
                      session=identity(out / 'session.npz'), full_model_executed=True, complete_atlas_execution=False)
    finally:
        torch.set_rng_state(rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        report['caller_torch_rng_restored'] = torch.equal(torch.get_rng_state(), rng)


def main():
    from workload_cli import run_report
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical', type=safe_path, required=True)
    p.add_argument('--out', type=safe_path, required=True)
    args = p.parse_args()
    report = {'status': 'failed', 'command': sys.argv, 'python': sys.version,
        'scope': 'fresh native BF16 provenance/repeat/capture only'}
    raise SystemExit(run_report(args, lambda args, report: native(args, args.out, report),
        filename='receipt.json', output=args.out, report=report,
        summary=('peak_rss_kib', 'status', 'wall_seconds')))
if __name__ == '__main__':
    main()
