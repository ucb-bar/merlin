"""Float capture through the Merlin worker, with exact export replay checks.

Random policies and strictly loaded local checkpoints use the same worker path.
The explicit torch_mlir companion backend is retained for independent diagnosis.
Synthetic observations do not establish policy accuracy.
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


def artifact_files(directory):
    """Yield generated regular files without following symlinks."""
    for path in sorted(directory.iterdir()):
        if any(token in path.name.lower() for token in ("hammer", "vlsi")):
            continue
        if path.is_symlink():
            continue
        if path.is_file():
            yield path


def unresolved_calls(mlir_text):
    """Count parsed calls to bodyless declarations, including mangled names."""
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func, FuncOp, CallOp
    from xdsl.parser import Parser
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(Func)
    module = Parser(context, mlir_text).parse_module()
    declarations = {op.sym_name.data for op in module.walk()
                    if isinstance(op, FuncOp) and not op.body.blocks}
    calls = {}
    for op in module.walk():
        if isinstance(op, CallOp):
            name = op.callee.root_reference.data
            if name in declarations:
                calls[name] = calls.get(name, 0) + 1
    return calls


def torch_mlir_main():
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
                                for p in artifact_files(out) if p != receipt_path}
        record()
    print(json.dumps({k: receipt[k] for k in ['status', 'export_check', 'path_taken']}, indent=2), flush=True)



def main():
    """Capture through Merlin's worker; companion backend remains explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['merlin', 'torch_mlir'], default='merlin')
    known, remaining = parser.parse_known_args()
    if known.backend == 'torch_mlir':
        sys.argv = [sys.argv[0], *remaining]
        return torch_mlir_main()
    parser = argparse.ArgumentParser(description='Exact export and Merlin worker float capture')
    parser.add_argument('--m2m-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--region', choices=['full'], default='full')
    args = parser.parse_args(remaining)
    if not 1 <= args.threads <= 8:
        parser.error('--threads must be between 1 and 8')
    if args.checkpoint:
        os.environ['MTDIT_CHECKPOINT'] = str(args.checkpoint.resolve())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.m2m_dir.resolve()))
    merlin_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(merlin_root / 'src'))
    import importlib.util
    import numpy as np
    import torch
    modules_before_loader = set(sys.modules)
    import loader
    from merlin.targetgen import _m2m_capture_worker as worker
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    receipt = {
        'schema': 'merlin.multi_task_dit.worker_capture.v1', 'status': 'started',
        'command': [sys.executable, *sys.argv], 'm2m_revision': revision(args.m2m_dir),
        'merlin_revision': revision(merlin_root),
        'versions': {n: importlib.metadata.version(n) for n in
                     ['torch', 'torch-mlir', 'xdsl', 'lerobot', 'transformers', 'diffusers']},
        'threads': args.threads, 'quantization': 'none', 'host_numerical_execution': 'PENDING',
        'sources': {n: digest(Path(__file__).with_name(n)) for n in ['loader.py', 'capture.py']},
    }
    source_dir = out / 'sources'
    source_dir.mkdir()
    for name in ['loader.py', 'capture.py']:
        (source_dir / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    def record():
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    record()
    try:
        print('construct', flush=True)
        model, inputs = loader.get_model_and_inputs()
        receipt['scope'] = loader.describe(model, inputs)
        receipt['runtime_versions'] = {n: importlib.metadata.version(n) for n in ['numpy', 'packaging', 'typing-extensions', 'pyyaml', 'immutabledict', 'ordered-set']}
        record()
        with torch.no_grad():
            print('eager', flush=True)
            eager = model(*inputs)
            if isinstance(model, loader.Inference):
                scheduler = model.scheduler
                model.scheduler = model.policy.objective.noise_scheduler
                original = model(*inputs)
                model.scheduler = scheduler
                receipt['original_scheduler_check'] = {
                    'equal': bool(torch.equal(eager, original)),
                    'max_abs': float((eager - original).abs().max())}
                if not receipt['original_scheduler_check']['equal']:
                    raise AssertionError('static addressing differs from original scheduler')
            if isinstance(model, loader.CheckpointInference):
                original = model.reference_forward(*inputs)
                receipt['original_objective_check'] = {
                    'equal': bool(torch.equal(eager, original)),
                    'max_abs': float((eager - original).abs().max())}
                if not receipt['original_objective_check']['equal']:
                    raise AssertionError('static Euler grid differs from original objective')
            np.save(out / 'eager.npy', eager.cpu().numpy())
            np.savez(out / 'inputs.npz', **{str(i): x.cpu().numpy() for i, x in enumerate(inputs)})
            print('torch.export', flush=True)
            exported = torch.export.export(model, tuple(inputs), strict=False)
            replay = exported.module()(*inputs)
            receipt['export_check'] = {'equal': bool(torch.equal(eager, replay)),
                'max_abs': float((eager - replay).abs().max()), 'finite': bool(torch.isfinite(eager).all()),
                'output_shape': list(eager.shape),
                'aten_calls': sum(n.op == 'call_function' for n in exported.graph.nodes)}
            torch.export.save(exported, out / 'exported.pt2')
            (out / 'exported-graph.txt').write_text(str(exported.graph))
            record()
            if not receipt['export_check']['equal'] or not receipt['export_check']['finite']:
                raise AssertionError('export replay differs from eager or eager contains nonfinite values')
            def completed(payload):
                result = payload['conversion_result']
                prepared = result.exported_program
                if prepared is None:
                    raise AssertionError('worker did not retain its prepared export')
                if prepared is not None:
                    observed = prepared.module()(*inputs)
                    receipt['prepared_export_check'] = {'equal': bool(torch.equal(eager, observed)),
                        'max_abs': float((eager - observed).abs().max())}
                    if not receipt['prepared_export_check']['equal']:
                        raise AssertionError('worker prepared export differs from eager')
                receipt['worker_path'] = result.path_taken
                receipt['worker_meta'] = payload['metadata']
                record()
            print('Merlin worker fx_importer', flush=True)
            code = worker.main(['--loader', str(Path(loader.__file__).resolve()), '--dtype', 'fp32',
                '--out', str(out / 'worker'), '--m2m-dir', str(args.m2m_dir.resolve()),
                '--materialize-bundle'], prepared_program={'module': model, 'inputs': tuple(inputs),
                    'loader': loader, 'dependencies': worker._loader_dependency_sources(
                        modules_before_loader, Path(loader.__file__)),
                    'provenance': receipt['scope'], 'session': None},
                completed=completed)
            receipt['worker_exit_code'] = code
            bundle_meta = out / 'worker' / 'bundle_meta.json'
            # The bundle's parsed census is authoritative; worker text census
            # alone misses generic-form func.call operations on this M2M pin.
            calls = unresolved_calls((out / 'worker' / 'model.mlir').read_text())
            receipt['parsed_unresolved_calls'] = calls
            receipt['status'] = 'PASS' if code == 0 and not calls and receipt['worker_meta']['opaque'] == 0 else 'DIAGNOSTIC'
            if calls:
                receipt['reason'] = 'unresolved frontend calls remain in parsed bundle'
    except Exception as exc:
        receipt['status'] = 'FAILED'
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        record()
        raise
    finally:
        receipt['artifacts'] = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in artifact_files(out) if p.name != 'receipt.json'}
        record()
    print(json.dumps({k: receipt.get(k) for k in ['status', 'export_check', 'parsed_unresolved_calls']}, indent=2), flush=True)
    return 0 if receipt['status'] == 'PASS' else 3


if __name__ == '__main__':
    raise SystemExit(main())
