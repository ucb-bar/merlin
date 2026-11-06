"""Prepare source-bound native Smol function tables; no device execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from build_smol_full_host import relocation_pairs, relocation_mapper
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked, map_metadata_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=checked, required=True)
    parser.add_argument('--out', type=checked, required=True)
    parser.add_argument('--relocate', action='append', default=[])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    pins = {}
    def pin(path, expected=None):
        path = checked(path)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('Native table input changed: '+str(path))
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError('Native table source changed: '+str(path))
        pins[str(path)] = digest
        return digest
    def read(path, expected=None):
        pin(path, expected)
        return checked(path).read_bytes()
    pin(__file__)
    pin(Path(__file__).with_name('build_smol_full_host.py'))
    pin(kernel_library.source_path('atlas_path_guard.py'))
    raw = read(args.replay)
    replay = map_metadata_paths(json.loads(raw), relocation_mapper(relocation_pairs(args.relocate)))
    if replay['status'] != 'PASS_ATLAS_ARITHMETIC_FULL_POLICY_REPLAY':
        raise ValueError('Native table producer requires complete arithmetic replay')
    library = replay['source']['torch_cpu_library']
    pin(library['path'], library['sha256'])
    import torch
    expected_torch = checked(Path(library['path']).parents[1] / '__init__.py')
    if checked(torch.__file__) != expected_torch:
        raise ValueError('Native function tables imported a different Torch provider')
    for path in (torch.__file__, torch.nn.functional.__file__, torch._C.__file__, sys.executable):
        pin(path)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    from lerobot.policies.smolvla.modeling_smolvla import create_sinusoidal_pos_embedding
    pin(sys.modules[create_sinusoidal_pos_embedding.__module__].__file__)
    words = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    gelu = torch.nn.functional.gelu(words.view(torch.bfloat16), approximate='tanh').view(torch.uint16).numpy().copy()
    gelu[(np.arange(65536, dtype=np.uint16) & 0x7f80) == 0x7f80] = 0
    if np.any((gelu & 0x7f80) == 0x7f80):
        raise ValueError('Native finite-domain GELU table contains a nonfinite result')
    trajectory = replay['trajectory']
    read(trajectory['path'], trajectory['sha256'])
    with np.load(checked(trajectory['path']), allow_pickle=False) as archive:
        times = np.array([archive[f'step_{step:02d}_timestep'][0] for step in range(10)], np.float32)
    calls = [row for row in replay['calls'] if row['module'] == 'action_time_mlp_in']
    values = []
    for step, time_value in enumerate(times):
        value = create_sinusoidal_pos_embedding(torch.tensor([time_value]), 720, 0.004, 4.0,
                                               device=torch.device('cpu')).float().numpy()[0]
        row = calls[step]['input']
        member = Path(row['file'])
        if member.is_absolute() or '..' in member.parts:
            raise ValueError('Invalid captured timestep member')
        captured = read(args.replay.parent/member, row['sha256'])
        actual = np.frombuffer(captured, '<f4').reshape(row['shape']).reshape(50, 1440)[:, 720:]
        if actual.tobytes() != np.broadcast_to(value, (50, 720)).copy().tobytes():
            raise ValueError('Native timestep table differs from captured actual operator input')
        values.append(value)
    products = {}
    for name, value in (('gelu', gelu), ('timestep', times), ('time.embedding', np.array(values, np.float32))):
        path = args.out/(name+'.bin')
        payload = value.tobytes()
        path.write_bytes(payload)
        products[name] = dict(file=path.name, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                              shape=list(value.shape), dtype=str(value.dtype))
    for path, expected in list(pins.items()):
        pin(path, expected)
    receipt = dict(status='PREPARED_NATIVE_FUNCTION_TABLES_ONLY', source_bookends_equal=True,
                   replay_sha256=hashlib.sha256(raw).hexdigest(), sources_sha256=pins, products=products,
                   command=[sys.executable,*sys.argv], python_version=sys.version, torch_version=torch.__version__,
                   pythonpath=os.environ.get('PYTHONPATH'), cpu_threads=torch.get_num_threads(),
                   elapsed_seconds=time.monotonic()-started, device_execution=False)
    (args.out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
