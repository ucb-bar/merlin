"""Sequential generated cache phases with actual BF16 parent-cache inputs.

Source captures and independent phase oracles are required. No oracle activation
or cached framework K/V is supplied as a generated execution input.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from composition import generated_stage_inputs, close_seams
from whole_model import file_hash, execution_backend_arguments

ORDER = [f"layer{i}" for i in range(22)] + ["norm", "head"]


def read(path):
    return json.loads(Path(path).read_text())


def adapter_source_selection(export, contract_sha256, policy_sha256):
    """Bind the requested source receipts before compiling or reusing stages."""
    if export is None:
        return None
    export = export.resolve(strict=True)
    if (file_hash(export / 'manifest.json') != contract_sha256
            or file_hash(export / 'policy.json') != policy_sha256):
        raise ValueError('adapter source receipt digest differs from requested pins')
    return dict(export=str(export), contract_sha256=contract_sha256, policy_sha256=policy_sha256)


def verify_compiled_selection(destination, compilation, source, args, adapter_selection):
    """Fresh and reused publications must describe the exact selected intake."""
    if compilation.get('status') != 'COMPILED':
        raise ValueError('compiled cache region does not report a completed compilation')
    if (compilation.get('io_memory_bytes') != args.io_memory_bytes
            or compilation.get('application_memory_bytes') != args.application_memory_bytes):
        raise ValueError('compiled cache region uses a different selected memory profile')
    if adapter_selection is None:
        return None
    adapter = read(destination / 'adapter-manifest.json')
    saved = adapter.get('atlas', {}).get('saved_native_program', {})
    calls = adapter.get('atlas', {}).get('calls')
    if (adapter.get('schema') != 'm2m.quantization_manifest.v2' or adapter.get('adapter_id') != 'atlas_fp8'
            or adapter.get('contract_sha256') != adapter_selection['contract_sha256']
            or adapter.get('policy_sha256') != adapter_selection['policy_sha256']
            or adapter != read(destination / 'capture/quantization-manifest.json')
            or saved.get('sha256') != source['model_sha256']
            or type(saved.get('no_native_calls')) is not bool or not isinstance(calls, list)
            or saved['no_native_calls'] is not (not calls)):
        raise ValueError('compiled cache region adapter source pins or attestation differ')
    return dict(adapter_manifest_sha256=file_hash(destination / 'adapter-manifest.json'),
                capture_manifest_sha256=file_hash(destination / 'capture/quantization-manifest.json'))


def phase_inputs(manifest, sources):
    """Require every temporal input to have one source-declared cache role."""
    whole = manifest['whole_model']
    phase = whole.get('cache_phase')
    if (whole.get('schema') != 'atlas.sequential_cache_phase_fx.v1' or not phase
            or phase != manifest.get('cache_phase') or whole['regions'] != ORDER):
        raise ValueError('authenticated full source cache phase manifest required')
    embedded = [row for row in manifest['regions'] if row['kind'] in ORDER]
    if len(embedded) != len(ORDER) or {row['kind']: row for row in embedded} != sources:
        raise ValueError('current source metadata differs from captured manifest')
    abi = phase['binding']['abi']
    if (abi['layer_count'] != 22 or abi['cache_representation'] != 'bf16_post_rope_pre_gqa'
            or phase['binding']['model_pt2_sha256'] != manifest['source_model_sha256']):
        raise ValueError('source cache phase identity or geometry differs')
    expected = {(v['layer'], v['role']): v for v in abi['inputs'][3:]}
    external = whole['external_cache_inputs']
    roles = [(v['layer'], v['role']) for v in external]
    implicit_empty = abi.get('initial_cache_empty') is True
    if implicit_empty and (abi['past_length'] != 0 or len(abi['inputs']) != 3 or expected):
        raise ValueError('implicit initial cache requires a zero-prefix three-input source ABI')
    count = 0 if implicit_empty else 44
    if len(roles) != count or len(set(roles)) != count or set(roles) != set(expected):
        raise ValueError('complete unique source cache input roster required')
    by_region = {kind: [] for kind in ORDER}
    for item in external:
        declaration = expected[item['layer'], item['role']]
        kind, index = item['region'], item['input_index']
        if (kind != f"layer{item['layer']}" or item['source_node'] != declaration['source_node']
                or item['global_input_index'] != declaration['index']
                or item['shape'] != declaration['shape'] or item['dtype'] != 'torch.bfloat16'
                or sources[kind]['source_boundaries'][index] != item['boundary']):
            raise ValueError('region cache input differs from source ABI')
        by_region[kind].append(item)
    links = {item['kind']: item['inputs'] for item in whole['links']}
    if len(whole['links']) != 24 or set(links) != set(ORDER):
        raise ValueError('complete unique generated hidden link roster required')
    allowed = set(whole['fixture_boundary_allowlist'])
    for position, kind in enumerate(ORDER):
        incoming = links[kind]
        if (position == 0 and incoming) or (position and (len(incoming) != 1 or incoming[0]['region'] != ORDER[position - 1])):
            raise ValueError('each source decoder boundary requires its immediate generated predecessor')
        covered = {item['input_index'] for item in incoming + by_region[kind]}
        for index, name in enumerate(sources[kind]['source_boundaries']):
            if index not in covered and name not in allowed:
                raise ValueError('undeclared boundary cannot use framework fixture')
    return abi, by_region


def parent_caches(execution, regions, current_manifest, current_abi):
    """Authenticate generated parent words without opening any oracle arrays."""
    parent_manifest = read(regions / 'manifest.json')
    receipt = read(execution / 'receipt.json')
    abi = parent_manifest['cache_phase']['binding']['abi']
    if (receipt.get('schema') != 'atlas.generated_cache_phase.v1' or receipt.get('status') != 'PASS'
            or receipt.get('mismatches') != 0 or receipt.get('requested_regions') != ORDER
            or receipt.get('temporal_sequential_execution') is not True or receipt.get('oracle_chain_inputs') is not False
            or receipt.get('cache_capture_sha256') != parent_manifest['cache_phase']['cache_capture_sha256']
            or receipt.get('manifest_sha256') != file_hash(regions / 'manifest.json')
            or receipt.get('source_model_sha256') != parent_manifest['source_model_sha256']
            or receipt.get('cache_capture_sha256') != current_manifest['cache_phase']['cache_capture_sha256']
            or abi['total_length'] != current_abi['past_length']
            or any(abi[key] != current_abi[key] for key in ('batch_size', 'kv_heads', 'head_dim', 'layer_count'))):
        raise ValueError('generated parent cache phase is incomplete or incompatible')
    rows = {row['kind']: row for row in receipt['regions']}
    if len(receipt['regions']) != 24 or len(rows) != 24 or set(rows) != set(ORDER):
        raise ValueError('generated parent region roster differs')
    embedded = {row['kind']: row for row in parent_manifest['regions'] if row['kind'] in ORDER}
    if len(embedded) != 24:
        raise ValueError('parent source manifest lacks region provenance')
    result = {}
    global_outputs = {v['source_node']: v for v in abi['outputs'][1:]}
    for kind in ORDER:
        source = read(regions / kind / 'source.json')
        if source != embedded[kind]:
            raise ValueError('parent region source metadata changed')
        stage = execution / kind
        row = rows[kind]
        publication = read(stage / 'execution.json')
        if (publication != row['execution'] or row['mismatches'] != 0
                or publication['source_model_sha256'] != source['model_sha256']):
            raise ValueError('generated parent execution publication differs')
        with np.load(stage / 'actual.npz', allow_pickle=False) as actual:
            for index, name in enumerate(source['source_outputs']):
                role = global_outputs.get(name)
                if role is None:
                    continue
                key = (role['layer'], role['role'].replace('updated_', 'past_'))
                value = np.ascontiguousarray(actual[f'output_{index}'])
                published = publication['outputs'][index]
                digest = sha256(value.tobytes()).hexdigest()
                if (kind != f"layer{role['layer']}" or key in result or value.dtype != np.uint16 or list(value.shape) != role['shape']
                        or digest != published['sha256'] or published['shape'] != role['shape']
                        or published['dtype'] != value.dtype.str):
                    raise ValueError('generated parent cache words or role changed')
                result[key] = (value, dict(parent_region=kind, parent_output_index=index,
                    parent_actual_sha256=file_hash(stage / 'actual.npz'), parent_sha256=digest,
                    parent_actual_file=str(stage / 'actual.npz')))
    if set(result) != {(v['layer'], v['role']) for v in current_abi['inputs'][3:]}:
        raise ValueError('generated parent omits a source cache role')
    return result, receipt, file_hash(execution / 'receipt.json')


def stage_inputs(source, links, captured, execution, records, cache_roles, cache_values, past_length):
    overrides, rows = generated_stage_inputs(source, links, captured, execution, records)
    used = set()
    for role in cache_roles:
        index = role['input_index']
        if index in used or any(link['input_index'] == index for link in links):
            raise ValueError('cache input overlaps another generated input')
        used.add(index)
        key = f'input_{index}'
        if past_length:
            value, origin = cache_values[role['layer'], role['role']]
            origin = dict(origin, input_origin='generated_cache_actual')
        else:
            value = np.empty(role['shape'], dtype=np.uint16)
            if value.size:
                raise ValueError('initial source cache must be empty')
            origin = dict(input_origin='empty_initial_cache')
        compiled = captured[key]
        if value.shape != compiled.shape or value.dtype.itemsize != compiled.dtype.itemsize:
            raise ValueError('generated cache input ABI differs')
        overrides[key] = value.view(compiled.dtype)
        rows[index] = dict(input_index=index, boundary=source['source_boundaries'][index],
            layer=role['layer'], role=role['role'], sha256=sha256(value.tobytes()).hexdigest(), **origin)
    return overrides, rows


def compare_outputs(stage, oracle_stage, reference):
    mismatches, outputs = 0, []
    publication = read(stage / 'execution.json')
    with np.load(stage / 'actual.npz', allow_pickle=False) as actual, np.load(oracle_stage / 'expected.npz', allow_pickle=False) as expected:
        if set(actual.files) != set(reference['outputs']) or len(publication['outputs']) != len(actual.files):
            raise ValueError('generated output roster differs from independent reference')
        for key, metadata in reference['outputs'].items():
            lhs, rhs = actual[key], expected[key]
            index = int(key.removeprefix('output_'))
            published = publication['outputs'][index]
            if (published['sha256'] != sha256(lhs.tobytes()).hexdigest()
                    or published['shape'] != list(lhs.shape) or published['dtype'] != lhs.dtype.str):
                raise ValueError('actual output words differ from execution publication')
            if (sha256(rhs.tobytes()).hexdigest() != metadata['sha256'] or lhs.shape != rhs.shape
                    or lhs.dtype != np.uint16 or rhs.dtype != np.uint16):
                raise ValueError('reference output changed or output ABI differs')
            different = lhs.reshape(-1).view(np.uint8).reshape(-1, lhs.dtype.itemsize) != rhs.reshape(-1).view(np.uint8).reshape(-1, rhs.dtype.itemsize)
            count = int(np.any(different, axis=1).sum())
            mismatches += count
            outputs.append(dict(key=key, elements=lhs.size, mismatches=count, sha256=sha256(lhs.tobytes()).hexdigest()))
    return mismatches, outputs


def authenticated_oracle(directory, manifest, manifest_path, parent_sha256):
    """Bind each expected array's stage metadata to a complete independent root."""
    raw = (directory / 'reference.json').read_bytes()
    oracle = json.loads(raw)
    if (oracle.get('status') != 'EXPECTED' or oracle.get('schema') != 'atlas.independent_original_fx_sequence.v1'
            or oracle.get('oracle_mode') != 'independent_cache_phase'
            or oracle.get('source_model_sha256') != manifest['source_model_sha256']
            or oracle.get('capsule_manifest_sha256') != file_hash(manifest_path)
            or oracle.get('parent_reference_receipt_sha256') != parent_sha256
            or oracle.get('cache_capture_sha256') != manifest['cache_phase']['cache_capture_sha256']
            or oracle.get('ordinary_framework_kernels_called') is not False
            or oracle.get('compiler_or_emitted_program_inputs') is not False
            or not oracle.get('model_arithmetic_sources') or not oracle.get('validation_sha256')):
        raise ValueError('independent cache oracle source, parent, or qualification differs')
    stages = oracle.get('stages', [])
    if [row.get('stage') for row in stages] != ORDER:
        raise ValueError('independent cache oracle lacks complete ordered stage records')
    for row in stages:
        if (read(directory / row['stage'] / 'reference.json') != row
                or row.get('ordinary_framework_kernels_called') is not False
                or row.get('semantic_json_or_emitted_program_inputs') is not False):
            raise ValueError('independent cache stage publication differs from qualified root')
    oracle['_receipt_sha256'] = sha256(raw).hexdigest()
    return oracle


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('regions', 'compiled', 'execution', 'atlas-root', 'model-root', 'model-python'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--oracle', type=Path)
    p.add_argument('--parent-execution', type=Path)
    p.add_argument('--parent-regions', type=Path)
    p.add_argument('--compile-only', action='store_true')
    p.add_argument('--adapter-export', type=Path, help='authenticate native capsules with atlas_fp8')
    p.add_argument('--expected-contract-sha256', default='', help='contract SHA-256 from the source receipt')
    p.add_argument('--expected-policy-sha256', default='', help='policy SHA-256 from the source receipt')
    p.add_argument('--compiler-python', type=Path, default=Path(sys.executable))
    p.add_argument('--reference-root', type=Path)
    p.add_argument('--max-instructions', type=int, default=1000000)
    p.add_argument('--backend', choices=('functional', 'qualified_native'), default='functional')
    for name in ('native-qualification', 'memory-qualification', 'native-cache'):
        p.add_argument('--' + name, type=Path)
    from precision_options import add_reduction_policy_argument, verify_reduction_policy

    add_reduction_policy_argument(p)
    p.add_argument("--allocation-timeout-ms", type=int, default=1000)
    p.add_argument('--io-memory-bytes', type=int, default=67108864)
    p.add_argument('--application-memory-bytes', type=int, default=536870912)
    p.add_argument("--merge-adjacent-device-regions", action="store_true",
                        help="Opt in to Atlas checked adjacent device-region alternatives")
    args = p.parse_args()
    selection = (args.adapter_export, args.expected_contract_sha256, args.expected_policy_sha256)
    if any(selection) and not all(selection):
        p.error('--adapter-export and both expected source SHA-256 values require each other')
    if args.adapter_export and any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
            for value in (args.expected_contract_sha256, args.expected_policy_sha256)):
        p.error('expected source SHA-256 values must be lowercase 64-character digests')
    if args.allocation_timeout_ms <= 0:
        p.error("allocation timeout must be positive")
    if args.io_memory_bytes <= 0 or args.application_memory_bytes <= 0:
        p.error('memory windows must be positive')
    adapter_selection = adapter_source_selection(*selection)
    backend = execution_backend_arguments(args.backend, args.native_qualification, args.memory_qualification, args.native_cache, max_instructions=args.max_instructions)
    regions, compiled, execution = (getattr(args, key).resolve() for key in ('regions', 'compiled', 'execution'))
    manifest = read(regions / 'manifest.json')
    sources = {kind: read(regions / kind / 'source.json') for kind in ORDER}
    abi, external = phase_inputs(manifest, sources)
    for kind in ORDER:
        if file_hash(regions / kind / 'model.pt2') != sources[kind]['model_sha256']:
            raise ValueError('current source capsule archive changed')
    if not args.compile_only and args.oracle is None:
        p.error('generated phase comparison requires --oracle')
    if (args.parent_execution is None) != (args.parent_regions is None):
        p.error('parent execution and regions must be provided together')
    cache_values, parent, parent_hash = {}, None, None
    if not args.compile_only:
        if abi['past_length']:
            if args.parent_execution is None:
                p.error('decode requires actual generated parent-cache execution')
            cache_values, parent, parent_hash = parent_caches(args.parent_execution, args.parent_regions, manifest, abi)
        elif args.parent_execution is not None:
            p.error('initial prefill cannot consume a nonempty parent phase')
    if parent and parent['completed_monotonic_ns'] >= time.monotonic_ns():
        raise ValueError('generated parent must complete before this phase starts')
    execution.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema='atlas.generated_cache_phase.v1', status='RUNNING', requested_regions=ORDER,
        source_model_sha256=manifest['source_model_sha256'], manifest_sha256=file_hash(regions / 'manifest.json'),
        driver_sha256=file_hash(Path(__file__)), cache_capture_sha256=manifest['cache_phase']['cache_capture_sha256'],
        cache_phase=manifest['cache_phase']['phase'], temporal_sequential_execution=True, oracle_chain_inputs=False,
        requested_allocation_timeout_ms=args.allocation_timeout_ms,
        requested_io_memory_bytes=args.io_memory_bytes, requested_application_memory_bytes=args.application_memory_bytes,
        requested_adapter_selection=adapter_selection,
        requested_merge_adjacent_device_regions=args.merge_adjacent_device_regions, requested_reduction_policy=args.reduction_policy,
        validation_backend=args.backend, functional_instruction_limit=args.max_instructions, parent_execution_receipt_sha256=parent_hash,
        parent_completed_monotonic_ns=parent['completed_monotonic_ns'] if parent else None,
        framework_activation_substitution=False,
        parent_oracle_receipt_sha256=parent['oracle_receipt_sha256'] if parent else None,
        started_monotonic_ns=time.monotonic_ns(), regions=[], generated_decode_exactness='PENDING')
    def save():
        (execution / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    records = {}
    tools = Path(__file__).resolve().parent
    env = dict(os.environ)
    imports = [str(args.model_root.resolve()), str(tools), str(args.atlas_root.resolve())]
    if args.reference_root:
        imports.append(str(args.reference_root.resolve()))
    if args.backend == 'qualified_native':
        spec = find_spec('merlin')
        if spec is None or not spec.submodule_search_locations:
            raise ValueError('qualified execution requires the Merlin runtime source package')
        imports.insert(0, str(Path(next(iter(spec.submodule_search_locations))).parent))
    env['PYTHONPATH'] = os.pathsep.join(imports)
    try:
        oracle = None if args.compile_only else authenticated_oracle(args.oracle, manifest, regions / 'manifest.json', receipt['parent_oracle_receipt_sha256'])
        oracle_hash = oracle['_receipt_sha256'] if oracle else None
        oracle_stages = {row['stage']: row for row in oracle['stages']} if oracle else {}
        for kind in ORDER:
            started = time.monotonic_ns()
            destination, source = compiled / kind, sources[kind]
            if not (destination / 'compiled.json').exists():
                command = [str(args.compiler_python), str(tools / 'compile_capsule.py'), '--capsule', str(regions / kind),
                    '--output', str(destination), '--atlas-root', str(args.atlas_root.resolve()), '--external-parameters',
                    '--softmax-mask-policy', 'bf16-minfinite-zero-rowspan80-nonempty-v1',
                    '--allocation-timeout-ms', str(args.allocation_timeout_ms), '--reduction-policy', args.reduction_policy,
                    '--io-memory-bytes', str(args.io_memory_bytes), '--application-memory-bytes', str(args.application_memory_bytes)]
                if adapter_selection is not None:
                    command += ['--adapter-export', adapter_selection['export'],
                                '--expected-contract-sha256', args.expected_contract_sha256,
                                '--expected-policy-sha256', args.expected_policy_sha256]
                if args.merge_adjacent_device_regions:
                    command.append('--merge-adjacent-device-regions')
                with (execution / f'{kind}-compile.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            compilation_raw = (destination / 'compiled.json').read_bytes()
            compilation = json.loads(compilation_raw)
            compilation_hash = sha256(compilation_raw).hexdigest()
            if compilation.get('merge_adjacent_device_regions', False) != args.merge_adjacent_device_regions:
                raise ValueError('compiled region uses a different adjacent-device selection')
            if (compilation['source_model_sha256'] != source['model_sha256'] or compilation['kind'] != kind
                    or compilation['constant_bytes_identical'] is not True):
                raise ValueError('compiled cache region differs from source')
            adapter_bindings = verify_compiled_selection(destination, compilation, source, args, adapter_selection)
            precision = read(destination / 'precision.json')
            verify_reduction_policy(kind, precision, compilation, args.reduction_policy)
            if args.compile_only:
                receipt['regions'].append(dict(kind=kind, status='COMPILED', compiled_receipt_sha256=compilation_hash,
                                               adapter_bindings=adapter_bindings))
                save()
                continue
            oracle_stage = args.oracle / kind
            reference_raw = (oracle_stage / 'reference.json').read_bytes()
            reference = json.loads(reference_raw)
            reference_hash = sha256(reference_raw).hexdigest()
            if reference != oracle_stages[kind]:
                raise ValueError('independent stage changed after root authentication')
            if (reference['status'] != 'EXPECTED' or reference['source_model_sha256'] != source['model_sha256']
                    or reference['constant_bytes_identical'] is not True):
                raise ValueError('independent cache stage source differs')
            stage = execution / kind
            stage.mkdir()
            with np.load(destination / 'tensors.npz', allow_pickle=False) as captured:
                overrides, inputs = stage_inputs(source, {v['kind']: v['inputs'] for v in manifest['whole_model']['links']}[kind],
                    captured, execution, records, external[kind], cache_values, abi['past_length'])
            override_path = stage / 'generated-predecessor-inputs.npz'
            np.savez(override_path, **overrides)
            command = [str(args.model_python), str(tools / 'execute_graph.py'), '--compiled', str(destination),
                '--model-root', str(args.model_root.resolve()), '--atlas-root', str(args.atlas_root.resolve()),
                '--compact-trace', '--input-overrides', str(override_path), '--execution-output', str(stage), *backend]
            with (execution / f'{kind}-execute.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            actual = read(stage / 'execution.json')
            if actual.get('source_model_sha256') != source['model_sha256'] or actual.get('immutable_inputs') is not True:
                raise ValueError('generated execution source identity or immutability differs')
            if any(actual['inputs'][row['input_index']]['sha256'] != row['sha256'] for row in inputs):
                raise ValueError('generated cache phase consumed different words')
            mismatches, outputs = compare_outputs(stage, oracle_stage, reference)
            row = dict(kind=kind, source_model_sha256=source['model_sha256'], inputs=inputs, execution=actual,
                outputs=outputs, mismatches=mismatches, started_monotonic_ns=started, completed_monotonic_ns=time.monotonic_ns(),
                compiled_receipt_sha256=compilation_hash, reference_receipt_sha256=reference_hash,
                actual_npz_sha256=file_hash(stage / 'actual.npz'), execution_receipt_sha256=file_hash(stage / 'execution.json'),
                input_overrides_sha256=file_hash(override_path), source_metadata_sha256=file_hash(regions / kind / 'source.json'))
            row['adapter_bindings'] = adapter_bindings
            records[kind] = row
            receipt['regions'].append(row)
            save()
            if mismatches:
                raise ValueError('generated cache phase differs from independent source oracle')
            print(json.dumps(dict(stage=kind, mismatches=mismatches)), flush=True)
        if adapter_source_selection(*selection) != adapter_selection:
            raise ValueError('adapter source receipts changed during phase')
        for row in receipt['regions']:
            kind = row['kind']
            if file_hash(compiled / kind / 'compiled.json') != row['compiled_receipt_sha256']:
                raise ValueError('compiled cache publication changed during phase')
            if verify_compiled_selection(compiled / kind, read(compiled / kind / 'compiled.json'),
                    sources[kind], args, adapter_selection) != row['adapter_bindings']:
                raise ValueError('compiled cache adapter publication changed during phase')
        if args.compile_only:
            receipt['status'] = 'COMPILED'
        else:
            if file_hash(args.oracle / 'reference.json') != oracle_hash:
                raise ValueError('independent oracle root changed during execution')
            final_oracle = authenticated_oracle(args.oracle, manifest, regions / 'manifest.json', receipt['parent_oracle_receipt_sha256'])
            if final_oracle != oracle:
                raise ValueError('independent oracle records changed during execution')
            if (oracle['status'] != 'EXPECTED' or oracle['source_model_sha256'] != manifest['source_model_sha256']
                    or oracle.get('oracle_mode') != 'independent_cache_phase'
                    or oracle.get('parent_reference_receipt_sha256') != receipt['parent_oracle_receipt_sha256']
                    or oracle.get('cache_capture_sha256') != receipt['cache_capture_sha256']
                    or not oracle.get('model_arithmetic_sources') or not oracle.get('validation_sha256')):
                raise ValueError('independent cache sequence qualification or parent differs')
            for row in receipt['regions']:
                kind = row['kind']
                stage = execution / kind
                bound = ((stage / 'actual.npz', 'actual_npz_sha256'),
                    (stage / 'execution.json', 'execution_receipt_sha256'),
                    (stage / 'generated-predecessor-inputs.npz', 'input_overrides_sha256'),
                    (regions / kind / 'source.json', 'source_metadata_sha256'),
                    (compiled / kind / 'compiled.json', 'compiled_receipt_sha256'))
                if any(file_hash(path) != row[key] for path, key in bound):
                    raise ValueError('bound phase artifact changed after execution')
                if file_hash(regions / kind / 'model.pt2') != sources[kind]['model_sha256']:
                    raise ValueError('source program changed during phase execution')
                if (read(execution / kind / 'execution.json') != row['execution']
                        or file_hash(args.oracle / kind / 'reference.json') != row['reference_receipt_sha256']):
                    raise ValueError('saved execution or reference publication changed after comparison')
                final_mismatches, final_outputs = compare_outputs(execution / kind, args.oracle / kind, oracle_stages[kind])
                if final_mismatches or final_outputs != row['outputs']:
                    raise ValueError('saved actual words changed after comparison')
            links = {v['kind']: v['inputs'] for v in manifest['whole_model']['links']}
            receipt['seam_closure'] = close_seams(ORDER, links, records)
            if parent_hash and file_hash(args.parent_execution / 'receipt.json') != parent_hash:
                raise ValueError('generated parent receipt changed during phase')
            for row in receipt['regions']:
                for incoming in row['inputs']:
                    if incoming.get('input_origin') == 'generated_cache_actual' and file_hash(Path(incoming['parent_actual_file'])) != incoming['parent_actual_sha256']:
                        raise ValueError('generated parent cache publication changed during phase')
            receipt.update(status='PASS', mismatches=0, oracle_receipt_sha256=oracle_hash,
                generated_decode_exactness='all selected hidden, logits, and cache output words exact', completed_monotonic_ns=time.monotonic_ns())
    except Exception as error:
        receipt.update(status='FAILED', error=repr(error))
        save()
        raise
    save()


if __name__ == '__main__':
    main()
