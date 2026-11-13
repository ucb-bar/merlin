"""Compare complete captured rotary domain; run with the source torch environment."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
out = root / 'out'
freq_bytes = (out / 'inv_freq.bin').read_bytes()
assert hashlib.sha256(freq_bytes).hexdigest() == '734f9e6b629f7baccc4de81b7e8d2982035ee1b81f73f158e7d33a4dcf13fa0e', 'inverse frequency bytes changed'
freq = torch.from_numpy(np.frombuffer(freq_bytes, dtype='<f4').copy())
pos = torch.arange(2048, dtype=torch.int64)
# Captured batch matmul K=1; transpose and duplicate by cat, then source trig,
# attention scaling 1.0 in f32, finally torch BF16 RNE.
angles = (freq[None, :, None] @ pos[None, None, :].float()).transpose(1, 2)
cat = torch.cat((angles, angles), dim=-1)
source = {'sin': torch.sin(cat) * 1.0, 'cos': torch.cos(cat) * 1.0}
rv = np.fromfile(out / 'rv64-rotary.bin', dtype='<u4').reshape(2048, 32, 3)
assert np.array_equal(rv[:, :, 0], angles[0].numpy().view(np.uint32)), 'angle mismatch'
def ordered(a):
 return np.where(a & 0x80000000, ~a, a ^ np.uint32(0x80000000)).astype(np.int64)
receipt = {'schema': 'rotary.rv64-libm-domain-check.v1',
 'torch_version': torch.__version__,
 'bf16_distinct_angle_mismatch_count': None,
 'qualification': 'failed until zero BF16 mismatches established',
 'cat_lane_note': 'each frequency appears twice; mismatch records use frequency modulo 32', 'torch_device': 'cpu',
 'torch_threads': torch.get_num_threads(),
 'archive_path': str((out/'core-math/libcore-math-rotary.a').resolve()),
 'core_math_revision': 'a055b9b9a86b857e49573e339cf68306ad92c831',
 'archive_sha256': (out / 'libm.sha256').read_text().split()[0],
 'isa': 'rv64gc', 'abi': 'lp64d', 'rounding': 'RNE',
 'emulator': str(Path(os.environ['RISCV'])/'bin/spike'),
 'domain': {'positions': [0, 2047], 'frequencies': 32, 'theta': 10000, 'head_dim': 64,
            'distinct_input_lanes': 65536, 'compared_bf16_words': 262144},
 'inv_freq_sha256': hashlib.sha256(freq_bytes).hexdigest(),
 'frequency_source': 'authenticated captured read-only inverse-frequency buffer; unchanged binary32 bytes',
 'arithmetic': 'i64 position -> f32; K=1 f32 multiply; positive-zero seed; transpose; cat; selected sinf/cosf; f32 scale 1.0; BF16 RNE',
 'angle_mismatch_count': 0, 'results': {}}
for name, lane in [('sin', 1), ('cos', 2)]:
 bits = np.concatenate((rv[:, :, lane], rv[:, :, lane]), axis=1)
 words = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
 refbits = source[name][0].numpy().view(np.uint32)
 refwords = source[name][0].bfloat16().view(torch.uint16).numpy()
 mismatch = np.argwhere(words != refwords)
 ulp = np.abs(ordered(bits) - ordered(refbits))
 examples=[]
 for p,j in mismatch[:16]:
  examples.append({'position': int(p), 'frequency': int(j%32), 'cat_lane': int(j),
   'angle_f32': hex(int(rv[p,j%32,0])), 'rv64_f32': hex(int(bits[p,j])),
   'torch_f32': hex(int(refbits[p,j])), 'rv64_bf16': hex(int(words[p,j])),
   'torch_bf16': hex(int(refwords[p,j])), 'f32_ulp': int(ulp[p,j])})
 receipt['results'][name] = {'bf16_mismatch_count': len(mismatch),
  'f32_mismatch_count': int(np.count_nonzero(ulp)), 'max_f32_ulp': int(ulp.max()),
  'f32_ulp_histogram': {str(int(k)): int(v) for k,v in zip(*np.unique(ulp, return_counts=True))},
  'first_mismatches': examples}
receipt['bf16_mismatch_count'] = sum(v['bf16_mismatch_count'] for v in receipt['results'].values())
receipt['bf16_distinct_angle_mismatch_count'] = receipt['bf16_mismatch_count'] // 2
receipt['gate'] = 'PASS' if receipt['bf16_mismatch_count']==0 else 'FAIL'
receipt['qualification'] = 'BF16 exact over positions 0..2047 only' if receipt['gate']=='PASS' else 'failed'
for key, path in [('emulator_sha256', Path(receipt['emulator'])), ('proxy_kernel_sha256', Path(receipt['emulator']).parents[1] / 'riscv64-unknown-elf/bin/pk')]:
 receipt[key] = hashlib.sha256(path.read_bytes()).hexdigest()
for f in ['rotary_probe','rv64-rotary.bin','inv_freq.bin']:
 receipt[f+'_sha256'] = hashlib.sha256((out/f).read_bytes()).hexdigest()
(out/'rotary-libm-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
