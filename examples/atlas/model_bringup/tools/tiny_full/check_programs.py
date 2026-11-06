"""Regenerate and assemble every final TinyLlama full-model device program; no Atlas instruction executes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/host_iss')]
import kernel_library  # noqa: E402,F401
from atlas_attention_v2_program import AttentionV2Config, assembly_v2
from atlas_execution import Executor
from atlas_path_guard import checked
from atlas_program import assembly
from atlas_program_checks import check_matrix_registers
from atlas_rope_postpass import ResidentRopeEpilogue, append_postpass
from attention_bundle import AttentionBundle, require
from chipyard.arena import DEFAULT_ARENA, relocate_assembly
from tiny_full.fusion import epilogue_from
from tiny_full.norm import prepare_norm

ROLES = ('q', 'k', 'v', 'o_plain', 'gate_silu', 'up_mul_fp8', 'down_plain', 'lm_head')


def projection_source(item, **options):
    """Return the exported projection recipe's assembly, with any RoPE post-pass, before arena relocation."""
    m, k, n = item['physical_mkn']
    spec = epilogue_from(item['epilogue'])
    source = assembly(m, k, n, item['output_offset'], item['rescale'], traffic_reuse=True, epilogue=spec, **options)
    return append_postpass(source, spec) if isinstance(spec, ResidentRopeEpilogue) else source


def regenerate(bundle, item):
    if item['kind'] == 'attention':
        return assembly_v2(AttentionV2Config(**item['config']), item['offsets'], dram_base=item['dram_base'])
    if item['kind'] == 'norm':
        prepared = prepare_norm(item['physical_rows'], bundle.read(item['h_bf16_tiles']), bundle.read(item['norm_weight']),
                                item['norm_scale'], item['logical_rows'], arena_base=item['dram_base'])
        require(prepared['initial'] == bundle.read(item['initial']) and prepared['output_offset'] == item['output_offset'],
                'norm staging regeneration differs: ' + str(item['index']))
        return prepared['source']
    return relocate_assembly(projection_source(item), DEFAULT_ARENA, item['dram_base'])


def check(bundle, executor):
    data = bundle.data
    require(data['attention_version'] == 2 and data['fusions'] is True and data['block_only'] is False
            and data['include_lm_head'] is True and data['kernel_count'] == len(data['launches'])
            and [row['index'] for row in data['launches']] == list(range(data['kernel_count'])),
            'complete fused attention-v2 full-model export required')
    assembled, roles, epilogues = set(), Counter(), 0
    for item in data['launches']:
        source, binary = bundle.read(item['source']), bundle.read(item['original'])
        require(hashlib.sha256(binary).hexdigest() == item['program_sha256'], 'raw program SHA differs')
        check_matrix_registers(source.decode())
        regenerated = regenerate(bundle, item)
        require(regenerated.encode() == source, 'recipe regeneration differs from final source: ' + str(item['index']))
        key = (item['program_sha256'], hashlib.sha256(source).hexdigest())
        if key not in assembled:
            program, sha256 = executor.assemble(regenerated)
            require(b''.join(word.to_bytes(4, 'little') for word in program.assemble()) == binary
                    and sha256 == item['program_sha256'], 'original model assembly differs from final binary')
            assembled.add(key)
        roles[ROLES[item['logical']] if item['kind'] == 'projection' else item['kind']] += 1
        epilogues += item['kind'] == 'projection' and item['epilogue'] is not None
    bundle.verify_bookends()
    return dict(status='PASS', launches=len(data['launches']), unique_programs=len(assembled), roles=dict(roles),
                epilogue_launches=epilogues, assets_sha256sums_sha256=bundle.checksums_sha256,
                model=executor.binding()['revision'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=checked, required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--model-root', type=checked, required=True)
    args = parser.parse_args()
    bundle = AttentionBundle(args.assets)
    require(bundle.sums['contract.json'] == args.contract_sha256, 'export contract differs')
    print(json.dumps(check(bundle, Executor(args.model_root)), indent=2))


if __name__ == '__main__':
    main()
