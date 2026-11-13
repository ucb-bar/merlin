"""Exhaustive independent v3 proof and separate torch/archive cross-check."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import rotary_exact as exact


def digest(path):
    return sha256(exact.checked(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--rv64-output', type=Path, default=root/'out/rv64-rotary.bin')
    parser.add_argument('--archive-gate', type=Path, default=root/'out/rotary-libm-receipt.json')
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=root/'reference/out/rotary-v3')
    args = parser.parse_args()
    import torch
    import rotary_source
    import whole_reference
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    gate = json.loads(exact.checked(args.archive_gate).read_text())
    if gate.get('gate') != 'PASS' or gate.get('bf16_mismatch_count') != 0:
        raise ValueError('archive qualification has not passed')
    if digest(args.rv64_output) != gate['rv64-rotary.bin_sha256']:
        raise ValueError('RV64 output hash differs from passing archive gate')
    frequency = exact.inv_freq()
    positions = np.arange(2048, dtype=np.int64)
    reference = exact.prepare(positions, frequency)
    # Replay the captured original-ATen rotary dependency slice with just its
    # arange extent expanded to cover every qualified position, in the same v3
    # Interpreter used for whole-model source execution.
    outputs, source, interpreter = rotary_source.replay(args.trace, whole_reference.primitives.POLICIES,
                                                        positions_end=2048)
    source_mismatches = {name: int(np.count_nonzero(outputs[name].raw[0] != reference[name+'_bf16']))
                         for name in ('sin', 'cos')}
    rv = np.fromfile(exact.checked(args.rv64_output), dtype='<u4').reshape(2048, 32, 3)
    pos = torch.from_numpy(positions)
    freq = torch.from_numpy(frequency.copy()).view(torch.float32)
    angle = (freq[None, :, None] @ pos[None, None, :].float()).transpose(1, 2)
    cat = torch.cat((angle, angle), dim=-1)
    result = {}
    angle_mismatches = int(np.count_nonzero(reference['angles_f32'] != rv[:, :, 0]))
    torch_angle_mismatches = int(np.count_nonzero(reference['angles_f32'] != angle[0].numpy().view(np.uint32)))
    for name, lane in (('sin', 1), ('cos', 2)):
        archive = np.concatenate((rv[:, :, lane], rv[:, :, lane]), axis=-1)
        ordinary = getattr(torch, name)(cat)*1.0
        torch_bits = ordinary[0].numpy().view(np.uint32)
        torch_words = ordinary[0].bfloat16().view(torch.uint16).numpy()
        result[name] = {
            'archive_f32_mismatches': int(np.count_nonzero(reference[name+'_f32'] != archive)),
            'archive_bf16_mismatches': int(np.count_nonzero(reference[name+'_bf16'] != ((archive+0x7fff+((archive>>16)&1))>>16).astype(np.uint16))),
            'torch_f32_mismatches': int(np.count_nonzero(reference[name+'_f32'] != torch_bits)),
            'torch_bf16_mismatches': int(np.count_nonzero(reference[name+'_bf16'] != torch_words)),
            'f32_sha256': sha256(reference[name+'_f32'].astype('<u4').tobytes()).hexdigest(),
            'bf16_sha256': sha256(reference[name+'_bf16'].astype('<u2').tobytes()).hexdigest()}
    mismatches = angle_mismatches + torch_angle_mismatches + sum(source_mismatches.values()) + sum(
        v[k] for v in result.values() for k in ('archive_f32_mismatches', 'archive_bf16_mismatches', 'torch_bf16_mismatches'))
    receipt = {'schema': 'atlas.independent_rotary_reference.v3', 'reference_version': 3,
               'status': 'PASS' if mismatches == 0 else 'FAIL', 'mismatches': mismatches,
               'selected_policy': exact.POLICY, 'domain': {'positions': [0, 2047], 'frequencies': 32,
               'distinct_angles': 65536, 'bf16_words': 262144},
               'inv_freq_sha256': exact.INV_FREQ_SHA256,
               'proof': {'arithmetic': 'exact Python integers and Fraction; no libm or framework trig',
                         'pi': 'Machin identity; alternating arctangent consecutive sums; outward dyadic enclosure',
                         'fractional_bits': exact.PRECISION, 'taylor_terms_after_seed': exact.TERMS,
                         'remainder': '|reduced angle|<=1; Lagrange <=1/(2*TERMS+1)! < one dyadic unit',
                         'range_reduction_error': 'exact pi interval propagated and bounded with derivative absolute value <=1',
                         'rounding': 'interval endpoints must round to identical f32 RNE words; ambiguity refuses',
                         'ambiguous_roundings': 0},
               'ordinary_framework_kernels_called_by_oracle': False, 'torch_used_only_for_comparison': True,
               'torch_version': torch.__version__, 'source_slice': source,
               'source_slice_bf16_mismatches': source_mismatches,
               'angle_archive_mismatches': angle_mismatches, 'angle_torch_mismatches': torch_angle_mismatches,
               'archive_gate_path': str(args.archive_gate.resolve()), 'archive_gate_sha256': digest(args.archive_gate),
               'archive_sha256': gate['archive_sha256'], 'rv64_output_sha256': digest(args.rv64_output),
               'oracle_sha256': digest(exact.__file__),
               'source_adapter_sha256': digest(rotary_source.__file__),
               'whole_interpreter_sha256': digest(whole_reference.__file__),
               'results': result, 'elapsed_seconds': round(time.monotonic()-start, 3)}
    out = exact.checked(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out/'expected.npz', **reference)
    receipt['expected_npz_sha256'] = digest(out/'expected.npz')
    (out/'reference.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
