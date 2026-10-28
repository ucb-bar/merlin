"""Capability intersection for advice only; never a legality or performance verdict.

This module is also copied as an author-side, read-only suggestion CLI. It imports
no evaluator, executes no candidate code, and reads only explicitly named inputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

NAMESPACES = ('host', 'submission', 'compute', 'numeric', 'memory', 'scheduling',
              'runtime', 'workload', 'compiler')
STATES = {'supported', 'unsupported', 'unknown'}
# Existing catalog terms get one semantic home, rather than a chip/category tag.
TRAITS = {
    'host': 'scalar_host integer_addressing',
    'submission': 'CPU_issued host_submitted_queues',
    'compute': 'systolic vector_lanes predicate_masks vector_register_files',
    'numeric': 'INT8_storage INT32_accumulation packed_subbyte_float separate_accumulator_format scaling_metadata',
    'memory': 'explicit_local_storage scratchpad finite_register_files explicit_buffer_ownership explicit_DMA strided_local_rows local_memory resident_operands accumulator noncoherent_cached_host device_allocations',
    'scheduling': 'multiple_outstanding async_issue persistent_configuration queues dependency_tokens finite_queues',
    'runtime': 'repeat_invocations persistent_state async_transfers device_events',
    'workload': 'immutable_weights graph_fanout static_temporaries immutable_tensor_views convolution dense_contraction independent_buffers tiling affine_tile_nests repeated_encoded_stream large_functions mixed_graph ranked_tensors pointwise_epilogue multiple_live_outputs adjacent_elementwise reductions row_reductions buffer_ownership source_graph rotating_buffers split_reduction shared_input multiple_shapes',
    'compiler': 'editable_compiler selected_target_adapter whole_system_objective emitted_artifacts',
}
ALIASES = {term: f'{ns}.{term}' for ns, terms in TRAITS.items() for term in terms.split()}
ALIASES.update({'scalar_host': 'host.scalar', 'CPU_issued': 'submission.cpu_issued',
                'INT8_storage': 'numeric.int8_storage', 'INT32_accumulation': 'numeric.int32_accumulation',
                'explicit_DMA': 'memory.explicit_dma'})
#: Named linalg products without a window: a dense contraction (a matrix or matrix-vector product).
DENSE_CONTRACTIONS = frozenset({'linalg.matmul', 'linalg.batch_matmul', 'linalg.matvec', 'linalg.vecmat',
                                'linalg.batch_matvec', 'linalg.matmul_transpose_a', 'linalg.matmul_transpose_b'})
EXTENSIONS = {
    'coherent_zero_copy': ['memory.host_coherence', 'memory.acquire_release_visibility'],
    'interrupt_completion': ['runtime.completion_interrupt', 'runtime.interrupt_abi', 'runtime.race_free_wakeup'],
    'autonomous_command_buffers': ['submission.device_command_fetch', 'submission.command_list_abi', 'runtime.list_completion'],
    'programmable_vector_epilogues': ['compute.vector_lanes', 'compute.predicate_masks', 'compute.accumulator_vector_transport', 'numeric.vector_elementwise'],
    'floating_matrix_compute': ['compute.floating_matrix', 'numeric.floating_matrix_contract', 'numeric.format_conversion'],
    'scatter_gather_dma': ['memory.segmented_dma', 'memory.segment_bounds', 'memory.transfer_fault_completion'],
}
ABSTRACTIONS = [
    ('graph_and_contract', ['merlin.common.mlir_query', 'merlin.xdsl_dialects.contract'],
     'Typed source and explicit contracts; derive patterns, shapes, numeric semantics and complete live-outs.'),
    ('region_and_preparation', ['merlin.compile.scheduling.multioutput_fusion',
                              'merlin.compile.scheduling.shared_input_groups',
                              'merlin.llvmlower.prepared_operand_schedule'],
     'Existing restricted region matchers and preparation planning/install APIs; no blanket fusion proof or automatic wiring.'),
    ('compact_instruction_schedule', ['merlin.sched.ir.kernel', 'merlin.compile.scheduling.compact_dense',
                                     'merlin.compile.scheduling.llvm_program'],
     'Existing Kernel and counted-loop emission route; consume the transformed result, preserve dependencies and tails.'),
    ('finite_schedule', ['merlin.compile.scheduling.schedule_pipeline'],
     'Existing finite-region pipeline with effect/storage/encoding adapters; do not unroll large graphs to use it.'),
    ('optional_dialect_exploration', ['merlin.xdsl_dialects.schedule', 'merlin.xdsl_dialects.interface'],
     'Existing dialect representations; target lowering integration must be demonstrated, not assumed from availability.'),
    ('owned_host_and_boundary', ['merlin.llvmlower.host_tensor_codegen', 'merlin.compile.host_optimizations'],
     'Existing generic host codegen and optimization route; preserve host/device ownership, ABI and exact numerical seams.'),
]


def abstraction_map(public_modules):
    modules = set(public_modules)
    return [{'layer': layer, 'scope': scope, 'apis': [{'module': module, 'selected': module in modules}
                                                    for module in apis]}
            for layer, apis, scope in ABSTRACTIONS]


def declarations(document):
    """Only explicit namespaced declarations; absent fields never mean false."""
    result = {}
    for namespace, fields in document.items():
        if namespace not in NAMESPACES or not isinstance(fields, dict):
            raise ValueError('capabilities require known namespaces and mappings')
        for name, value in fields.items():
            if not isinstance(name, str) or not name.isidentifier() or value not in STATES:
                raise ValueError('capability values must be supported, unsupported or unknown')
            result[f'{namespace}.{name}'] = value
    return result


def software_traits(spec):
    values = declarations(spec.get('optimization_capabilities', {}))
    # Exact selected software semantics only: scale-field FP32 is not FP matrix support.
    semantics = spec.get('numerical_semantics', {})
    for key, expected, trait in [('operand_dtype', {'i8', 'int8'}, 'numeric.int8_storage'),
                                  ('accumulator_dtype', {'i32', 'int32'}, 'numeric.int32_accumulation')]:
        if semantics.get(key) in expected:
            if values.get(trait) == 'unsupported':
                raise ValueError('explicit capability contradicts selected numerical semantics')
            values[trait] = 'supported'
    return values


def contract_traits(contract):
    """Explicit selected fields only; no implications from a target name."""
    values, evidence = {}, {}
    def observe(key, path):
        values[key] = 'supported'
        evidence[key] = path
    if contract.get('endpoint_kind') == 'inline_asm_insn':
        observe('submission.cpu_issued', 'endpoint_kind=inline_asm_insn')
        observe('submission.instruction_issued', 'endpoint_kind=inline_asm_insn')
        # Inline instructions execute inside host code, so a scalar host core runs the program.
        observe('host.scalar', 'endpoint_kind=inline_asm_insn: accelerator instructions are issued by host code')
    # Declared instruction roles that move data into and out of accelerator storage are explicit,
    # compiler-issued transfers; the role names are the contract's own vocabulary, not a target tag.
    roles = (contract.get('corpus_instruction_roles') or {}).get('by_role') or {}
    if isinstance(roles, dict) and roles.get('memory') and (roles.get('acc_readout') or roles.get('readout')):
        observe('memory.explicit_dma', 'corpus_instruction_roles.by_role: explicit memory load and readout instructions')
    for index, unit in enumerate(contract.get('compute_units', [])):
        if unit.get('kind') == 'systolic':
            observe('compute.systolic', f'compute_units[{index}].kind=systolic')
    return values, evidence


def storage_traits(rtl):
    """Selected logical address-space roles do not imply overlap or coherence."""
    values, evidence = {}, {}
    for index, space in enumerate(rtl.get('facts', {}).get('memory_address_spaces', [])):
        role = space.get('name')
        if (role in {'scratchpad', 'accumulator'} and space.get('view_kind') == 'address_space'
                and type(space.get('bytes')) is int and space['bytes'] > 0):
            for trait in (f'memory.{role}', 'memory.explicit_local_storage', 'memory.local_memory'):
                values[trait] = 'supported'
                evidence[trait] = f'facts.memory_address_spaces[{index}]: declared positive-byte {role} address space'
    return values, evidence


def ir_traits(text):
    """Narrow structural observations from parsed typed IR, not names or comments.

Alias/immutability, runtime repetition, masks, legality and profitable fusion are
intentionally not inferred. Absence of these patterns leaves broader traits unknown.
"""
    from merlin.common import mlir_query as mq
    from xdsl.dialects.builtin import TensorType
    module = mq.parse(text)
    values = {'workload.source_graph': 'supported'}
    operations = list(module.walk())
    for op in operations:
        if op.name.startswith('linalg.conv_'):
            values['workload.convolution'] = 'supported'
        if op.name in DENSE_CONTRACTIONS:
            values['workload.dense_contraction'] = 'supported'
        if op.name == 'func.return' and len(op.operands) > 1:
            values['workload.multiple_live_outputs'] = 'supported'
        iterator_types = op.attributes.get('iterator_types', op.properties.get('iterator_types'))
        if iterator_types is not None and any('reduction' in str(item) for item in iterator_types):
            values['workload.reductions'] = 'supported'
        for value in (*op.operands, *op.results):
            if isinstance(value.type, TensorType):
                values['workload.ranked_tensors'] = 'supported'
            if len({use.operation for use in value.uses}) > 1:
                values['workload.graph_fanout'] = 'supported'
    return values


def select(catalog, *, capabilities, public_modules, workload=None, limit=12):
    if type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError('suggestion limit must be 1..64')
    # A target capability declaration cannot invent workload or compiler evidence.
    facts = {key: value for key, value in capabilities.items()
             if key.split('.')[0] not in {'workload', 'compiler'}}
    for key, value in (workload or {}).items():
        if not key.startswith('workload.') or value not in STATES:
            raise ValueError('workload observations must be namespaced IR traits')
        facts[key] = value
    modules = set(public_modules)
    rows = []
    for entry in catalog['mechanisms']:
        requirements = [ALIASES[t] for t in entry['required_traits']]
        api = entry['compiler']['module']
        states = {key: facts.get(key, 'unknown') for key in requirements}
        states['compiler.api:' + api] = 'supported' if api in modules else 'unsupported'
        status = ('unsupported' if 'unsupported' in states.values() else
                  'unknown' if 'unknown' in states.values() else 'supported')
        rows.append({'id': entry['id'], 'status': status, 'requirements': states,
                     'mechanism': entry['mechanism'], 'api': api,
                     'symbol': entry['compiler']['symbol'], 'family': entry['facet'],
                     'legality': entry['applicability'], 'evidence': entry['evidence_to_request'],
                     'implementation_status': entry['implementation_status'],
                     'adapter_requirements': entry['remaining_adapter_or_feedback'],
                     'registry_metadata': 'unknown until supplied by the concrete registered PassSpec; '
                                          'requires, produces, invalidates and ir_level are not inferred from imports'})
    starters = []
    for entry in catalog.get('starter_sets', []):
        states = {ALIASES[t]: facts.get(ALIASES[t], 'unknown') for t in entry['traits']}
        status = ('unsupported' if 'unsupported' in states.values() else
                  'unknown' if 'unknown' in states.values() else 'supported')
        starters.append({'id': entry['id'], 'status': status, 'requirements': states,
                         'scope': 'design directions only; no implementation or runtime binding granted',
                         'directions': entry['mechanisms'], 'gaps': entry['gaps']})
    for entry in catalog.get('extensions', []):
        states = {key: facts.get(key, 'unknown') for key in EXTENSIONS[entry['id']]}
        status = ('unsupported' if 'unsupported' in states.values() else
                  'unknown' if 'unknown' in states.values() else 'supported')
        starters.append({'id': entry['id'], 'status': status, 'requirements': states,
                         'scope': 'proposal only; no implementation or runtime binding granted',
                         'directions': [entry['mechanism']], 'gaps': entry['required_validation']})
    applicable = [row for row in rows if row['status'] == 'supported']
    # Unknown only because the workload has not been checked: every other requirement is supported,
    # so one `--ir` run on the actual source decides them. Kept apart from genuine capability questions.
    pending = [row for row in rows if row['status'] == 'unknown' and all(
        state == 'supported' for key, state in row['requirements'].items() if not key.startswith('workload.'))]
    pending_ids = {row['id'] for row in pending}
    discovery = [row for row in rows if row['status'] == 'unknown' and row['id'] not in pending_ids]
    return {'schema': 'merlin.optimization_suggestions.v1',
            'authority': 'advisory applicability only; no legality, execution or performance credit',
            'ordering': 'catalog order for display only, not pass order or ranking',
            'suggestions': applicable[:limit], 'suggestions_omitted': len(applicable[limit:]),
            'pending_workload_check': [{'id': row['id'], 'family': row['family'], 'api': row['api'],
                                        'symbol': row['symbol'],
                                        'workload_requirements': {key: state for key, state in row['requirements'].items()
                                                                  if key.startswith('workload.')}}
                                       for row in pending],
            'discovery': discovery[:limit], 'discovery_omitted': len(discovery[limit:]),
            'all_mechanisms': rows, 'starter_sets': starters,
            'abstractions': abstraction_map(modules),
            'facts': facts, 'workload_scope': 'explicit parsed IR only; absent at initial prompt',
            'catalog_sha256': hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()}


def prepare(workspace, run_dir, *, repo, inventory, contract_path=None, ordering=None):
    """Append-only initial prompt artifact, using already staged public data.

``ordering`` is an explicit command-ordering selection pin ({path, sha256}) derived from the same
facts and contract; without one a derived ``command_ordering`` section of the facts is used. Either
settles scheduling/memory capabilities only through the fact each derivation names.
"""
    import yaml
    path = Path(repo) / 'docs/design/joint_optimization_catalog.json'
    software = Path(workspace) / 'public_declarations/software-spec.yaml'
    if not path.is_file() or not software.is_file():
        return ''
    catalog = json.loads(path.read_text())
    extension_path = path.with_name('tightly_coupled_optimization_extensions.json')
    if extension_path.is_file():
        catalog['extensions'] = [{key: entry[key] for key in ('id', 'mechanism', 'required_validation')}
                                 for entry in json.loads(extension_path.read_text())['entries']]
    spec = yaml.safe_load(software.read_text())
    modules = [row['module'] for row in inventory['public_modules']]
    capabilities, evidence = {}, {}
    if contract_path is not None:
        contract_path = Path(contract_path)
        capabilities, evidence = contract_traits(yaml.safe_load(contract_path.read_text()))
    facts_path = Path(workspace) / 'rtl_facts/facts.json'
    provenance, ordering_provenance, ordering_bytes = {}, None, None
    if facts_path.is_file():
        from .capability_facts import derive, ordering_input

        document = json.loads(facts_path.read_text())
        observed, origins = storage_traits(document)
        capabilities.update(observed)
        evidence.update(origins)
        record, ordering_provenance = ordering_input(ordering, facts_document=document, facts_path=facts_path,
                                                     contract_path=contract_path)
        if record is not None and (ordering_provenance or {}).get('selection'):
            # Capabilities alone do not provide the source ordering needed by an
            # authored schedule adapter. Carry the exact checked public record
            # into the workspace; operator filesystem/environment are private.
            pin = ordering_provenance['selection']
            ordering_bytes = Path(pin['path']).read_bytes()
            if hashlib.sha256(ordering_bytes).hexdigest() != pin['sha256']:
                raise ValueError('command ordering selection changed during public staging')
        observed, origins, provenance = derive(document, record, ordering_provenance=ordering_provenance)
        capabilities.update(observed)
        evidence.update(origins)
    capabilities.update(software_traits(spec))
    result = select(catalog, capabilities=capabilities, public_modules=modules)
    result['input_pins'] = {'software_sha256': hashlib.sha256(software.read_bytes()).hexdigest(),
                            'catalog_bytes_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'compiler_library_sha256': inventory['contract_sha256']}
    if contract_path is not None:
        result['input_pins']['target_contract_sha256'] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if facts_path.is_file():
        result['input_pins']['rtl_facts_sha256'] = hashlib.sha256(facts_path.read_bytes()).hexdigest()
    if ordering_provenance is not None and ordering_provenance.get('selection'):
        result['input_pins']['command_ordering_selection_sha256'] = ordering_provenance['selection']['sha256']
    result['trait_evidence'] = evidence
    result['capability_provenance'] = {'ordering': ordering_provenance, 'derived': provenance}
    files = {'OPTIMIZATION_SUGGESTIONS.json': json.dumps(result, indent=2) + '\n',
             'OPTIMIZATION_CATALOG.json': json.dumps(catalog, indent=2) + '\n',
             'optimization_suggest.py': Path(__file__).read_text(),
             'OPTIMIZATION_CAPABILITIES.json': json.dumps({'traits': capabilities, 'evidence': evidence,
                                                           'provenance': result['capability_provenance'],
                                                           'pins': result['input_pins']}, indent=2) + '\n',
             'OPTIMIZATION_API_SELECTION.json': json.dumps({'public_modules': modules}) + '\n'}
    for root in (Path(workspace), Path(run_dir)):
        for name, text in files.items():
            (root / name).write_text(text)
        if ordering_bytes is not None:
            target = root / 'COMMAND_ORDERING_SELECTION.json'
            if target.exists() and target.read_bytes() != ordering_bytes:
                raise ValueError('staged public command ordering selection changed')
            if not target.exists():
                target.write_bytes(ordering_bytes)
                target.chmod(0o444)
    return ('\n## Capability-selected optimization suggestions\n\n'
            'Read `OPTIMIZATION_SUGGESTIONS.json`: supported suggestions and unknown discovery '
            'questions are separate. This is advice, not pass order, legality or performance proof. '
            'The initial prompt has no workload IR and cannot establish its prerequisites. '
            'When selected, `COMMAND_ORDERING_SELECTION.json` carries the exact public RTL-derived '
            'ordering record for schedule law construction; the convolution/dense epoch recipes '
            'discover it without access to operator paths or environment. This is source evidence, '
            'not a selected schedule or permission to ignore dependencies. '
            'Recompute for the actual public source with `python optimization_suggest.py '
            '--ir PATH --output suggestions.json`. Only selected API modules are import grants; '
            'design starter sets grant no new implementation. Missing declarations remain unknown. '
            'The abstractions map connects typed graph/contracts, legal region fusion and preparation, '
            'compact Kernel scheduling, target emission and shared host codegen. Locate their editable '
            'sources in COMPILER_LIBRARY_API.json. Distill useful changes into general passes and '
            'typed-IR/resource heuristics for additional models; do not dispatch on model or benchmark names. '
            'Dialect availability does not require using every dialect or replacing the existing compiler.\n')


def main():
    import argparse
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog', type=Path, default=Path('OPTIMIZATION_CATALOG.json'))
    p.add_argument('--software', type=Path, default=Path('public_declarations/software-spec.yaml'))
    p.add_argument('--library', type=Path, default=Path('OPTIMIZATION_API_SELECTION.json'))
    p.add_argument('--capabilities', type=Path, default=Path('OPTIMIZATION_CAPABILITIES.json'))
    p.add_argument('--ir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    catalog = json.loads(args.catalog.read_text())
    spec = yaml.safe_load(args.software.read_text())
    library = json.loads(args.library.read_text())
    text = args.ir.read_text() if args.ir else None
    capabilities = json.loads(args.capabilities.read_text())['traits']
    capabilities.update(software_traits(spec))
    report = select(catalog, capabilities=capabilities, public_modules=library['public_modules'],
                    workload=ir_traits(text) if text is not None else None)
    report['input_pins'] = {key: hashlib.sha256(path.read_bytes()).hexdigest()
                            for key, path in [('catalog', args.catalog), ('software', args.software),
                                               ('library', args.library), ('capabilities', args.capabilities),
                                               ('ir', args.ir)] if path is not None}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
