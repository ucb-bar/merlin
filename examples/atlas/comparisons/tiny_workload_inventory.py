"""Declare complete pinned TinyLlama operators and bounded Atlas column shards."""
from merlin.common.jsonio import write_pretty_json
import argparse
import json
from pathlib import Path
import struct
from capture_tinyllama_native import safe_path, identity, WEIGHT_SHA


def column_shards(k, n, m=128):
    if (any(type(x) is not int for x in (m, k, n)) or m not in (32,
        128) or not 0 < k <= 8192 or k % 32 or not 0 < n <= 32000 or n % 32):
        raise ValueError('Tiny physical M 32/128, K <= 8192, N <= 32000 positive multiples of 32 required')
    # Match the unchanged producer's 32 MiB NPZ bound, including NPY headers.
    limit = 32 * 1024 ** 2 - 4096
    maximum = 2048 if k <= 2048 else 1024
    rows =[]
    for start in range(0, n, maximum):
        end = min(n, start + maximum)
        width = end - start
        source_bytes = 4 *(m * k + k * width + m * width)
        dram_bytes = m * k + k * width + 2 * m * width + 512
        if width % 32 or source_bytes > limit or dram_bytes > 64 * 1024 ** 2 or m * width > 1048576:
            raise ValueError('column shard exceeds original producer resource/source bound')
        rows.append({'columns': [start, end], 'physical_mkn': [m, k, width],
                     'source_array_bytes': source_bytes, 'maximum_npz_bytes': 32 * 1024 ** 2,
                     'dram_bytes': dram_bytes, 'K_order': 'unchanged full logical K',
                     'quantization': 'tensor-pow2 per actual column shard; global-scale equivalence must be measured'})
    return rows


def inventory(checkpoint):
    record = identity(checkpoint / 'model.safetensors', WEIGHT_SHA)
    with (checkpoint / 'model.safetensors').open('rb') as stream:
        extent = struct.unpack('<Q', stream.read(8))[0]
        if extent > 1024 ** 2:
            raise ValueError('checkpoint header exceeds 1 MiB')
        header = json.loads(stream.read(extent))
    expected = {'self_attn.q_proj': (2048, 2048), 'self_attn.k_proj': (2048, 256), 'self_attn.v_proj': (2048, 256),
              'self_attn.o_proj': (2048, 2048), 'mlp.gate_proj': (2048, 5632), 'mlp.up_proj': (2048, 5632), 'mlp.down_proj': (
        5632, 2048)}
    contractions =[]
    unique = set()
    all_linear = 0
    for layer in range(22):
        for site, (k, n) in expected.items():
            key=f'model.layers.{layer}.{site}.weight'
            if header[key]['dtype'] != 'BF16' or header[key]['shape'] !=[n, k]:
                raise ValueError('pinned weight shape/precision differs')
            shards = column_shards(k, n)
            for m in (128, 32):
                for shard in column_shards(k, n, m):
                    unique.add(tuple(shard['physical_mkn']))
            contractions.append({'site': key.removesuffix('.weight'), 'partition': 'Atlas', 'logical_prefill_mkn': [128, k, n],
                                 'logical_decode_mkn': [1, k, n], 'prefill_shards': shards,
                                 'decode_padding': 'logical 1 row then 31 positive-zero rows; not yet executed',
                'MACs_prefill': 128 * k * n, 'MACs_frozen128plus32': 160 * k * n})
            all_linear += 160 * k * n
    key = 'lm_head.weight'
    if header[key]['dtype'] != 'BF16' or header[key]['shape'] !=[32000, 2048]:
        raise ValueError('LM head shape differs')
    shards = column_shards(2048, 32000)
    for m in (128, 32):
        for shard in column_shards(2048, 32000, m):
            unique.add(tuple(shard['physical_mkn']))
    contractions.append({'site': 'lm_head', 'partition': 'Atlas planned via column shards; current baseline native host',
                         'logical_prefill_mkn': [128, 2048, 32000], 'logical_decode_mkn': [1, 2048, 32000],
                         'prefill_shards': shards, 'MACs_prefill': 128 * 2048 * 32000,
        'MACs_frozen128plus32': 160 * 2048 * 32000})
    all_linear += 160 * 2048 * 32000
    # Dense eager masked attention executes all 128x128 prefill entries.
    attention = 22 * 2 * 32 * 64 *(128 * 128 + sum(range(129, 161)))
    return {'status': 'declared_partition_only', 'checkpoint': record, 'configuration': identity(checkpoint / 'config.json'),
            'native_precision': 'pinned BF16; FP32 model arithmetic is a separate reference',
            'contractions': contractions, 'unique_physical_template_shapes': sorted(map(list, unique)),
            'host_operator_inventory': {'embedding_gather': 1, 'RMSNorm': 45, 'attention_blocks': 22, 'QK_AV_dense_matmuls': 44,
                                       'RoPE_Q_K_pairs': 22, 'mask_softmax': 22, 'KV_cache_updates': 22, 'SiLU': 22,
        'elementwise_MLP_multiply': 22,
                                       'residual_add': 44,
        'final_logits_and_scoring': 'native until selected Atlas LMhead integration'},
            'first_block_logical_device_contractions': 7, 'first_block_prefill_programs': sum(len(column_shards(k, n)) for k,
        n in expected.values()),
            'logical_device_contractions_per_forward': 155, 'linear_MACs_frozen128plus32': all_linear,
            'host_attention_MACs_frozen128plus32': attention, 'planned_Atlas_share_of_linear_plus_attention_MACs': all_linear /(
        all_linear + attention),
            'share_convention': (
                'logical useful MACs only; excludes unary, data movement, padding, host overhead '
                'and unsupported operators; no timing claim'
            ),
            'not_claimed': ['whole model compiled', 'whole model Atlas execution', 'decode kernel execution',
        'native quality gate passing']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=safe_path, required=True)
    p.add_argument('--out', type=safe_path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('fresh inventory output required')
    result = inventory(a.checkpoint)
    result['source_script'] = identity(__file__)
    write_pretty_json(a.out, result, sort_keys=False)
    print(json.dumps({k: result[k] for k in ('first_block_prefill_programs',
        'planned_Atlas_share_of_linear_plus_attention_MACs')}))
if __name__ == '__main__':
    main()
