"""Float Model2MLIR/torch-mlir capture, retaining unelided resource payloads.

Random DiT and text-projection weights, pretrained CLIP, no accuracy claim.
The Merlin worker currently selects fx_importer. This companion uses the same
public Model2MLIR API with its torch_mlir backend, without changing model math.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from unittest.mock import patch


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m2m-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--region', choices=['full', 'vision', 'text', 'block', 'dit'], default='full')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('--threads must be between 1 and 16')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.m2m_dir.resolve()))
    import numpy as np
    import torch
    import m2m
    from safetensors.torch import save_file
    from torch_mlir import fx, ir
    from torch_mlir.dialects import torch as torch_dialect
    import loader
    from piece_loaders import select_region

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    print('construct', args.region, flush=True)
    if args.region == 'full':
        model, inputs = loader.get_model_and_inputs()
        scope = loader.describe(model, inputs)
    else:
        model, inputs = select_region(args.region)
        scope = {'weight_scope': loader.WEIGHT_SCOPE, 'seed': 0,
                 'region': args.region, 'quantization': 'none',
                 'inputs': [{'shape': list(x.shape), 'dtype': str(x.dtype)} for x in inputs]}
    receipt = {
        'schema': 'merlin.multi_task_dit.float_capture.v1',
        'status': 'started', 'command': [sys.executable, *sys.argv],
        'cwd': str(Path.cwd()), 'scope': scope,
        'm2m_revision': revision(args.m2m_dir),
        'merlin_base_revision': '79f25c91d',
        'versions': {name: importlib.metadata.version(name) for name in
                     ['torch', 'torch-mlir', 'lerobot', 'transformers', 'diffusers', 'torchvision', 'xdsl', 'numpy']},
        'python': sys.version,
        'torch_git_revision': torch.version.git_version,
        'source_files': {name: {'sha256': digest(Path(__file__).with_name(name))}
                         for name in ['capture.py', 'loader.py', 'piece_loaders.py', 'inventory.py']},
        'clip_checkpoint': loader.checkpoint_metadata(),
        'environment': {k: os.environ.get(k) for k in ['TMPDIR', 'HF_HOME', 'HF_HUB_OFFLINE',
                        'TRANSFORMERS_OFFLINE', 'MTDIT_TORCHVISION_SCHEMA_COMPAT',
                        'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
        'quantization': 'none', 'seed': 0, 'threads': args.threads,
        'deterministic_algorithms': True,
        'host_numerical_execution': 'PENDING',
    }
    receipt_path = out / 'receipt.json'

    def record():
        receipt_path.write_text(json.dumps(receipt, indent=2, default=str) + '\n')

    record()
    try:
        with torch.no_grad():
            print('eager', flush=True)
            eager = model(*inputs)
            np.save(out / 'eager.npy', eager.detach().cpu().numpy())
            np.savez(out / 'inputs.npz', **{str(i): x.detach().cpu().numpy() for i, x in enumerate(inputs)})
            print('torch.export', flush=True)
            exported = torch.export.export(model, tuple(inputs), strict=False)
            replay = exported.module()(*inputs)
            receipt['export_check'] = {
                'equal': bool(torch.equal(eager, replay)),
                'max_abs': float((eager - replay).abs().max()),
                'output_shape': list(eager.shape), 'dtype': str(eager.dtype),
                'finite': bool(torch.isfinite(eager).all()),
                'aten_calls': sum(n.op == 'call_function' for n in exported.graph.nodes),
            }
            record()
            if not receipt['export_check']['equal']:
                raise AssertionError('exported program differs from eager')
            torch.export.save(exported, out / 'exported.pt2')
            (out / 'exported-graph.txt').write_text(str(exported.graph))
            save_file({k: v.detach().cpu().contiguous().clone() for k, v in exported.state_dict.items()},
                      str(out / 'weights.safetensors'))
            if args.region == 'block':
                receipt['block_output_equals_input'] = bool(torch.equal(eager, inputs[0]))
            original_import = fx.export_and_import

            def retain_module(*positional, **kwargs):
                module = original_import(*positional, **kwargs)
                if not module.operation.verify():
                    raise AssertionError('torch-mlir module verification failed')
                # Pinned M2M's structural printer elides resource bytes. Retain
                # the exact same module before that printer, with all resources.
                with (out / 'model.mlir').open('w') as stream:
                    module.operation.print(file=stream, enable_debug_info=False)
                    stream.write('\n')
                with (out / 'model.mlirbc').open('wb') as stream:
                    module.operation.write_bytecode(stream)
                receipt['torch_mlir_in_memory_verify'] = True
                return module

            print('Model2MLIR backend=torch_mlir', flush=True)
            with patch.object(fx, 'export_and_import', retain_module):
                result = m2m.convert(exported, tuple(inputs), backend='torch_mlir',
                                     allow_fallback=False, capture_trace=True,
                                     output_type='linalg-on-tensors', func_name='forward')
            receipt.update(path_taken=result.path_taken, diagnostics=result.diagnostics,
                           conversion_ok=result.ok)
            (out / 'structural.mlir').write_text(result.mlir_text)
            if result.capture_trace is not None:
                (out / 'frontend-trace.json').write_text(json.dumps(result.capture_trace, indent=2, default=str) + '\n')
            record()
            if not result.ok or result.path_taken != 'torch_mlir':
                raise RuntimeError('Model2MLIR torch-mlir conversion failed; see receipt diagnostics')
            # The prepared/decomposed EP is also retained and checked separately.
            if result.exported_program is None:
                raise AssertionError('Model2MLIR did not retain the prepared exported program')
            if result.exported_program is not None:
                prepared = result.exported_program.module()(*inputs)
                receipt['prepared_export_check'] = {
                    'equal': bool(torch.equal(eager, prepared)),
                    'max_abs': float((eager - prepared).abs().max()),
                }
            if not receipt['prepared_export_check']['equal']:
                raise AssertionError('prepared exported program differs from eager')
            print('parse and verify persisted bytecode', flush=True)
            with ir.Context() as context:
                torch_dialect.register_dialect(context)
                parsed = ir.Module.parse((out / 'model.mlirbc').read_bytes())
                receipt['persisted_bytecode_verify'] = bool(parsed.operation.verify())
            if not receipt['persisted_bytecode_verify']:
                raise AssertionError('persisted bytecode verification failed')
            # Inventory parses the complete textual artifact, not the elided view.
            from inventory import inventory
            print('text parse, verify, inventory', flush=True)
            report = inventory((out / 'model.mlir').read_bytes())
            (out / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
            receipt['persisted_text_verify'] = report['verified']
            receipt['status'] = 'PASS'
    except Exception as exc:
        receipt['status'] = 'FAILED'
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        record()
        raise
    finally:
        receipt['artifacts'] = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
                                for p in sorted(out.iterdir()) if p.is_file() and p != receipt_path}
        record()
    print(json.dumps({k: receipt[k] for k in ['status', 'export_check', 'path_taken']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
