"""Shared TinyLlama BF16 scale and native cache-tail checks."""
import numpy as np
import tiny_workload_native as native
from compare_tinyllama_attention import raw_array, tensor_identity
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
