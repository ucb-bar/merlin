"""Package a standard live application for the existing example execution harness.

This command never exports a graph or compiles source IR. Run stage-capture and
native-compile --application first, with application-name capsule.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np
import torch
from safetensors.torch import load_file

from compile_capsule import build_host_library, preflight_host_library, raw_bytes


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--compiled', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--staged', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--atlas-root', type=Path, required=True)
    parser.add_argument('--boundary-source', type=Path, required=True)
    parser.add_argument('--expected-boundary-source-sha256', required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    from merlin.targetgen.native_codegen_cli import add_io_memory_argument

    add_io_memory_argument(parser)
    parser.add_argument('--execution-instruction-limit', type=int,
                        help='per-launch executed-instruction budget selected during compilation')
    parser.add_argument('--exact-float-unroll', action='store_true',
                        help='replay the exact float unrolling selected during compilation')
    parser.add_argument('--functional-validation', action='store_true', help='enable validation-only execution-host math component text replacement')
    parser.add_argument('--execution-host-math-archive', type=Path)
    parser.add_argument('--execution-host-math-equivalence', type=Path)
    parser.add_argument('--execution-host-math-rv64-output', type=Path)
    args = parser.parse_args()
    if args.execution_instruction_limit is not None and args.execution_instruction_limit <= 0:
        parser.error('execution instruction limit must be positive')
    codegen_choices, codegen_flags = {}, []
    if args.execution_instruction_limit is not None:
        codegen_choices['execution_instruction_limit'] = args.execution_instruction_limit
        codegen_flags.extend(['--execution-instruction-limit', str(args.execution_instruction_limit)])
    if args.exact_float_unroll:
        codegen_choices['exact_float_unroll'] = True
        codegen_flags.append('--exact-float-unroll')
    math_selection = None
    if (args.functional_validation or args.execution_host_math_archive
            or args.execution_host_math_equivalence or args.execution_host_math_rv64_output):
        if not (args.functional_validation and args.execution_host_math_archive
                and args.execution_host_math_equivalence and args.execution_host_math_rv64_output):
            parser.error('execution-host math requires functional-validation, archive, equality receipt and RV64 output together')
        math_selection = dict(functional_validation=True, archive=args.execution_host_math_archive.resolve(),
                              equivalence=args.execution_host_math_equivalence.resolve(),
                              rv64_output=args.execution_host_math_rv64_output.resolve())
    root = args.compiled.resolve()
    if (root / 'compiled.json').exists():
        raise ValueError('execution publication already exists')
    source_bytes = args.boundary_source.read_bytes()
    if sha256(source_bytes).hexdigest() != args.expected_boundary_source_sha256:
        raise ValueError('boundary source digest differs from frozen selection')
    source = json.loads(source_bytes)
    tensors = load_file(str(args.inputs))
    if set(tensors) != set(source['tensors']):
        raise ValueError('boundary tensor roster differs')
    for key, tensor in tensors.items():
        meta = source['tensors'][key]
        if (sha256(raw_bytes(tensor)).hexdigest() != meta['sha256']
                or list(tensor.shape) != meta['shape'] or str(tensor.dtype) != meta['dtype']):
            raise ValueError(f'boundary tensor changed: {key}')
    from merlin.targetgen.application_inventory import verify_capture_receipt

    verification = verify_capture_receipt(args.capture / 'model.mlir')
    if verification.get('status') != 'verified_materialized':
        raise ValueError(f'live capture is not authenticated: {verification}')
    from merlin.frontends.compile_inputs import load_compile_inputs

    staged = load_compile_inputs(args.staged)
    if staged.original_source_sha256 != sha256((args.capture / 'model.mlir').read_bytes()).hexdigest():
        raise ValueError('staged source differs from the live capture')
    manifest = json.loads((root / 'application/manifest.json').read_text())
    if manifest.get('schema') != 'merlin.semantic_application.v1':
        raise ValueError('standard application compilation did not complete')
    from merlin.semantic_compiler.codegen import IO_MEMORY_OPTIONS, CodegenOptions

    memory_options = {name: manifest[name] for name in IO_MEMORY_OPTIONS if name in manifest}
    if not memory_options:
        memory_options = {'io_memory_bytes': 67108864}
    CodegenOptions(**memory_options, **codegen_choices)
    for name in IO_MEMORY_OPTIONS:
        selected = getattr(args, name)
        if selected is not None and selected != memory_options.get(name):
            raise ValueError('selected I/O memory geometry differs from compiled application: ' + name)
    memory_flags = [value for name, number in memory_options.items()
                    for value in ('--' + name.replace('_', '-'), str(number))]
    subprocess.run([
        sys.executable, '-m', 'merlin.targetgen.cli', 'native-verify',
        '--engine', 'merlin_native', '--support', 'atlas', '--snapshot', str(args.snapshot),
        '--target-source', str(args.atlas_root), '--compile-inputs', str(args.staged),
        '--application', '--already-quantized', '--application-parameters', 'external',
        *memory_flags, *codegen_flags, '--mode', 'diagnostic',
        '--artifact', str(root / 'application'), '--out', str(root / 'application-verified.json'),
    ], check=True)
    if json.loads((root / 'application/manifest.json').read_text()) != manifest:
        raise ValueError('application manifest changed during verification')
    interface = preflight_host_library(manifest, execution_host_math=math_selection)
    arguments = [tensors[key] for key in source['input_keys']]
    if len(arguments) != len(interface['inputs']):
        raise ValueError('runtime inputs differ from standard application ABI')
    # The worker writes exact BF16 values to its runtime NPZ as f32 carriers.
    # Compare values and dtype/shape against the authenticated source words;
    # the generated execution receives the original words below.
    with np.load(args.capture / 'inputs.npz', allow_pickle=False) as captured:
        if set(captured.files) != {f'in{i}' for i in range(len(arguments))}:
            raise ValueError('worker runtime input roster differs')
        for i, tensor in enumerate(arguments):
            carrier = tensor.float().numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy()
            if (captured[f'in{i}'].dtype != carrier.dtype
                    or captured[f'in{i}'].shape != carrier.shape
                    or captured[f'in{i}'].tobytes() != carrier.tobytes()):
                raise ValueError('worker inputs differ from selected source boundary')
    package = root / 'runtime/c-package'
    package.mkdir(parents=True, exist_ok=False)
    for member in manifest['stage_artifacts']:
        path = Path(member)
        if not member.startswith('c-package/'):
            continue
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('unsafe application C package member')
        destination = package / path.relative_to('c-package')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / 'application' / path, destination)
    if math_selection is None:
        build_host_library(root / 'application', manifest, package=package)
    else:
        build_host_library(root / 'application', manifest, package=package, execution_host_math=math_selection)
    arrays = {key: (tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy())
              for key, tensor in tensors.items()}
    arrays.update({f'input_{i}': (tensor.view(torch.uint16).numpy() if tensor.dtype == torch.bfloat16 else tensor.numpy())
                   for i, tensor in enumerate(arguments)})
    np.savez(root / 'tensors.npz', **arrays)
    record = dict(status='COMPILED', kind=source['kind'], capture=str(args.capture.resolve()),
                  source_model_sha256=source['model_sha256'],
                  capture_mlir_sha256=sha256((args.capture / 'model.mlir').read_bytes()).hexdigest(),
                  source_request_digest=manifest['request_digest'],
                  logical_source_tensors=source['tensors'], interface=interface,
                  **memory_options, **codegen_choices, application_memory_bytes=manifest['application_memory_bytes'],
                  segments=manifest['segments'], regions=manifest['regions'],
                  compilation_entry='merlin targetgen native-compile --application',
                  boundary_source_sha256=args.expected_boundary_source_sha256,
                  host_library='runtime/c-package/capsule.so',
                  host_library_sha256=sha256((package / 'capsule.so').read_bytes()).hexdigest(),
                  runtime_files={str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                                 for p in (package / 'shim.c', package / 'capsule.so')},
                  independent_comparison='PENDING')
    if math_selection is not None:
        selection = package / 'execution-host-math.json'
        record['execution_host_math'] = json.loads(selection.read_text())
        record['functional_validation_only'] = True
        record['runtime_files'][str(selection.relative_to(root))] = sha256(selection.read_bytes()).hexdigest()
    (root / 'compiled.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': 'PACKAGED', 'request_digest': manifest['request_digest']}))


if __name__ == '__main__':
    main()
