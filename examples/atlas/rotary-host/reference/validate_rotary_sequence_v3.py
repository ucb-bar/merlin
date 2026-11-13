"""Light production-sequence smoke: incorrect rotary fixtures must be ignored."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np
import rotary_exact
import whole_reference as ref


def main():
    import torch
    from safetensors.torch import save_file
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, default=Path(__file__).resolve().parent/'out/rotary-v3/reference.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    class Outputs(torch.nn.Module):
        def forward(self, cosine, sine):
            return cosine.clone(), sine.clone()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        capsule = root/'capsules/layer0'
        capsule.mkdir(parents=True)
        cosine = torch.zeros((1, 64, 64), dtype=torch.bfloat16)
        sine = cosine.clone()
        program = torch.export.export(Outputs(), (cosine, sine))
        torch.export.save(program, capsule/'model.pt2')
        save_file({'input_0': cosine, 'input_1': sine}, str(capsule/'tensors.safetensors'))
        (capsule/'source.json').write_text(json.dumps({
            'input_keys': ['input_0', 'input_1'], 'source_boundaries': ['to_7', 'to_8'],
            'model_sha256': sha256((capsule/'model.pt2').read_bytes()).hexdigest(),
            'tensors': {key: {'shape': [1, 64, 64]} for key in ('input_0', 'input_1')}}))
        (root/'capsules/manifest.json').write_text(json.dumps({
            'whole_model': {'regions': ['layer0'], 'links': [{'kind': 'layer0', 'inputs': []}]},
            'source_model_sha256': 'light-fixture-refusal-smoke'}))
        record = ref.run_sequence(root/'capsules', root/'result', ref.primitives.POLICIES,
                                  rotary_source_bundle=args.source_bundle, rotary_validation=args.qualification)
        outputs = np.load(root/'result/layer0/expected.npz')
        expected = rotary_exact.prepare(np.arange(64, dtype=np.int64)[None, :])
        for i, name in enumerate(('cos', 'sin')):
            np.testing.assert_array_equal(outputs['output_'+str(i)], expected[name+'_bf16'])
            if np.count_nonzero(outputs['output_'+str(i)]) == 0:
                raise ValueError('all-zero rotary fixture leaked into stage outputs')
            if record['stages'][0]['inputs'][i]['arithmetic'] != 'v3 independent original-ATen rotary interpreter':
                raise ValueError('rotary input origin does not identify independent computation')
        print(json.dumps({'status': 'PASS', 'reference_version': record['reference_version'],
                          'fixture_rotary_words': 'intentionally incorrect all-zero fixtures ignored',
                          'computed_outputs': 'exact independent cos/sin through stage interpreter',
                          'archive_gate_sha256': record['rotary_source_replay']['archive_gate_sha256']}))


if __name__ == '__main__':
    main()
