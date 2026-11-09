"""Publish the selected caller's complete software ABI, without hardware facts.

The private target contract remains the caller/build-recipe authority. This
projection supplies the external callable names in the existing public
pointer protocol; it never guesses from candidate output or a target name.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

from merlin.common.digest import is_sha256
from merlin.common.text_tokens import is_identifier, source_tokens

SCHEMA = 'merlin.public_runtime_callable_abi.v2'
KEY = 'public_runtime_abi'
ALIAS = 'mlir_oot_backend_contract.yaml'
DECLARED = 'merlin/contract/' + ALIAS
NOTE = 'selected runtime callable ABI'


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _regular(path):
    path = Path(path)
    if (not path.is_absolute() or '..' in path.parts or path.is_symlink()
            or not path.is_file() or path.resolve(strict=True) != path):
        raise ValueError('runtime ABI requires a canonical regular input file')
    return path


def selected_symbol(contract):
    """Use the exact contract reader also used by the native caller and recipe."""
    from merlin.targetgen.contract.harness_abi import from_contract

    if not isinstance(contract, dict):
        raise ValueError('selected runtime ABI contract is not a mapping')
    symbol = from_contract(contract, target=str(contract.get('name', 'selected'))).entry_symbol
    if not symbol.isascii() or not symbol.isidentifier():
        raise ValueError('selected runtime callable must be a plain C ABI identifier')
    runner = contract.get('runner') or {}
    implementation = runner.get(runner.get('backend'), {})
    stack = (implementation.get('toolchain') or {}).get('kernel_stack_frame')
    if stack is not None and (not isinstance(stack, dict) or stack.get('entry_symbol') != symbol):
        raise ValueError('selected runtime caller and stack recipe entry symbols differ')
    return symbol


def _void_signature(signature, symbol):
    """Check the displayed protocol signature, which can contain placeholder parameters."""
    tokens = [token.text for token in source_tokens(signature)]
    name = [token.text for token in source_tokens(symbol)]
    prefix = ["void", *name, "("]
    return tokens[:len(prefix)] == prefix and ")" in tokens[len(prefix):]


def _extern_declaration(declaration):
    """Read the entire restricted external void-pointer declaration."""
    tokens = [token.text for token in source_tokens(declaration)]
    if (len(tokens) < 6 or tokens[:2] != ["extern", "void"]
            or not is_identifier(tokens[2]) or tokens[3] != "("
            or tokens[-2:] != [")", ";"] or any(t in ("(", ")") for t in tokens[4:-2])):
        raise ValueError('selected runtime has an unreviewed extra external declaration')
    return tokens[2], tokens[4:-2]


def _pointer_parameters(tokens):
    if tokens in ([], ["void"]):
        return True
    arguments, argument = [], []
    for token in [*tokens, ","]:
        if token == ",":
            arguments.append(argument)
            argument = []
        else:
            argument.append(token)
    return all(arg[:2] == ["void", "*"] and
               (len(arg) == 2 or len(arg) == 3 and is_identifier(arg[2])) for arg in arguments)


def selected_callables(contract):
    """Every unconditional candidate-defined function called by the native harness.

    The logical pointer protocol owns per-case entry arity. Completion has no
    arguments or result. The separate lowered representation metadata states
    the selected fixed runtime completion sequence.
    """
    from merlin.targetgen.contract.harness_abi import from_contract

    entry = selected_symbol(contract)
    abi = from_contract(contract, target=str(contract.get('name', 'selected')))
    fence = abi.fence_symbol
    if fence is not None and (not fence.isascii() or not fence.isidentifier() or fence == entry):
        raise ValueError('selected completion callable must be a distinct plain C ABI identifier')
    if (contract.get('harness_abi') or {}).get('main_convention', 'void') != 'void':
        raise ValueError('public logical pointer ABI requires its declared void entry convention')
    result = [dict(role='entry', symbol=entry, return_type='void', linkage='C',
                   parameters='per_case_logical_tensor_pointers', defined_by='candidate_package')]
    if fence is not None:
        result.append(dict(role='completion', symbol=fence, return_type='void', linkage='C', parameters=[],
                           defined_by='candidate_package',
                           signature=f'void {fence}(void)',
                           called_by='runner_after_entry_before_cycle_end_and_readback'))
    # The selected generic caller invokes entry and optional fence only. An
    # extra private extern cannot silently become an undisclosed ABI obligation.
    for declaration in abi.extern_decls:
        name, parameters = _extern_declaration(declaration)
        if name not in {entry, fence}:
            raise ValueError('selected runtime has an unreviewed extra external declaration')
        if name == fence and parameters not in ([], ['void']):
            raise ValueError('selected completion declaration must have void(void) prototype')
        if name == entry and not _pointer_parameters(parameters):
            raise ValueError('selected entry declaration contradicts the logical pointer protocol')
    return result


def selected_software_limits(contract):
    """Runtime stack admission is a software constraint, not an RTL capacity."""
    selected_symbol(contract)
    runner = contract.get('runner') or {}
    toolchain = (runner.get(runner.get('backend'), {}) or {}).get('toolchain') or {}
    stack = toolchain.get('kernel_stack_frame')
    if stack is None:
        return {}
    maximum = stack.get('max_static_bytes')
    if type(maximum) is not int or maximum <= 0:
        raise ValueError('selected runtime requires a positive static entry stack budget')
    return dict(entry_max_static_stack_bytes=maximum,
                dynamic_stack_allocation_allowed=False,
                stack_report_scope='all_functions_in_lowered_translation_unit')


def selected_lowered_representation(contract):
    """Project only the existing selected software calling-convention gate."""
    from merlin.targetgen.contract.harness_abi import lowered_plan_abi

    abi = lowered_plan_abi(contract, target=str(contract.get('name', 'selected')))
    if abi['completion_symbol'] is not None and (
            not isinstance(abi['completion_asm'], str) or not abi['completion_asm'].strip()):
        raise ValueError('selected lowered completion ABI lacks its fixed runtime sequence')
    return dict(
        selected_abi=abi,
        scope='normalized LLVM-dialect artifact subject to source/task ownership validation',
        functions=('Inline all ordinary helper definitions into the selected entry. The only permitted '
                   'additional defined llvm.func is the optional selected completion shim.'),
        entry=('Define the selected entry with external linkage, ccc convention, void result, no '
               'variadic arguments, and exactly the declared number of ordinary address-space-zero '
               'pointer arguments. Entry argument and result attribute dictionaries must be empty.'),
        completion_shim=dict(
            purpose='fixed runtime ABI completion sequence; not a kernel or scheduling implementation',
            signature='external ccc void(void)', blocks=1,
            operations=['llvm.inline_asm', 'llvm.return'],
            inline_asm_constraints='~{memory}', inline_asm_has_side_effects=True,
            inline_asm_operands=0,
            inline_asm_results='none, or one unused LLVM void result', return_operands=0,
            task_attributes='No merlin.global_task or merlin.source_op_index on the shim or its operations.',
            instruction='Use exactly selected_abi.completion_asm, without an added or substituted sequence.'))


def source_inventory_rules(input_format):
    """Explain the selected source representation without supplying compiler code."""
    generic = ('For generic mathematical MLIR, use the unique defined single-block func: ordered '
               'block arguments are entry arguments; number every direct operation except func.return, '
               'including constants, tensor.empty, tensor.splat and linalg.fill. Do not number nested '
               'scalar-body operations. Result indices follow each operation result order; returns '
               'follow func.return operand order. Preserve existing prov.region_id values.')
    if input_format == 'neutral_math_v1':
        selected = ('For numerical_program.v1 JSON, source_op_count is len(expressions); '
                    'source_op_indices are zero-based expressions array positions, including explicit '
                    'epilogues and casts. Inputs and outputs records are not source operations. Every '
                    'expression has one result, so source_values.result_index is 0. entry_bindings '
                    'follow inputs array order and bind the corresponding input id to its materialized '
                    'tensor. output_bindings follow outputs array order and each output value id must '
                    'resolve to the same source_values operation result. JSON expressions have no '
                    'region IDs; do not invent source_region_ids.')
    elif input_format == 'shared_interface_v1':
        selected = ('For module-level merlin_iface input, ordered merlin_iface.tensor declarations '
                    'form the entry arguments and are excluded from source_op_count and '
                    'source_op_indices. Number every other module-level operation in source order, '
                    'including resident_pack, matmul and evict lifetime/state operations. '
                    'source_values.result_index follows each operation result order. entry_bindings '
                    'follow the tensor declarations. output_bindings follow published tensor results '
                    'in module order: resident_pack, matmul and evict do not publish tensor results. '
                    'Opaque resident/accumulator results do not authorize shaped tensor storage. '
                    'Preserve existing prov.region_id values.')
    else:
        raise ValueError('unknown selected public source input format')
    return dict(input_format=input_format,
                identity='Hash the exact delivered source file bytes consumed by the compiler, without reserialization.',
                selected_representation=selected, generic_mathematical_mlir=generic,
                bindings=('entry_bindings covers source arguments exactly once in order. source_values '
                          'binds materialized source results by op_index/result_index with matching '
                          'shape and dtype. output_bindings matches source return order and '
                          'kernel_abi.outputs; each output is write-only and has its unique writer in '
                          'the task owning that source result. kernel_abi.args covers every materialized '
                          'tensor exactly once. Compiler temporaries require the schema-declared '
                          'source-result provenance and an owning writer before any read.'))


def rewrite_protocol(raw, *, contract, input_format='shared_interface_v1'):
    """Keep the numerical pointer protocol; disclose the complete selected linkage."""
    document = yaml.safe_load(raw)
    if not isinstance(document, dict) or not isinstance(document.get('logical_kernel_abi'), dict):
        raise ValueError('public protocol lacks its logical kernel ABI')
    output = copy.deepcopy(document)
    block = output['logical_kernel_abi']
    original, signature = block.get('symbol'), block.get('signature')
    if (not isinstance(original, str) or not original or not isinstance(signature, str)
            or signature.count(original) != 1
            or not _void_signature(signature, original)):
        raise ValueError('public callable symbol and signature are inconsistent')
    symbol = selected_symbol(contract)
    block['symbol'] = symbol
    block['signature'] = signature.replace(original, symbol)
    callables = selected_callables(contract)
    block['required_callables'] = callables
    block['software_limits'] = selected_software_limits(contract)
    block['normalized_lowered_representation'] = selected_lowered_representation(contract)
    tokens = block.get('tokens', {})
    if 'declared_whole_program_args' in tokens:
        tokens['declared_whole_program_args'] = (
            'kernel_abi.args[].tensor in array order (kind = whole_program). Outputs must be '
            'write-only. readwrite does not confer source initialization or a seed: scratch requires '
            'a source-owned task writer before its first read. A reset allocation is not source provenance.')
    attribution = output.setdefault('whole_program_attribution', {})
    attribution['public_preflight'] = (
        'Use schemas/mixed_program_plan.schema.json and the selected_source_inventory rules below. '
        'Bind exact source bytes and source operation indices, not payload-only operation indices. '
        'This is structural provenance validation, not numerical or execution evidence. A helper '
        'library is supplied only if present in the selected environment source/tool inventory.')
    attribution['selected_source_inventory'] = source_inventory_rules(input_format)
    attribution['required_plan'] = (
        'params.global_program_plan is mandatory for a kernel_abi.kind whole_program buffer with '
        'any role=intermediate tensor, even when the mathematical case has a single operation. '
        'Use the complete mixed_program_plan_v1 schema, covering every source operation exactly once. '
        'readwrite access grants no seed or initialization; mutable scratch needs a source-owned '
        'writer before its first read. Do not invent source regions or rely on allocation reset bytes.')
    if isinstance(attribution.get('source_to_task'), str):
        attribution['source_to_task'] = attribution['source_to_task'].replace(
            "full source function's operation order", 'selected source inventory operation order')
    if isinstance(attribution.get('host_compute_boundary'), str):
        attribution['host_compute_boundary'] += (
            ' In the normalized source/task-audited LLVM artifact, ordinary helper definitions must '
            'be inlined into the selected entry; the only permitted additional definition is the '
            'optional fixed-ABI completion shim described in logical_kernel_abi.')
    completion = next((row for row in callables if row['role'] == 'completion'), None)
    if completion is None:
        block['completion'] = ('The entry must return only when its results are visible in memory. '
                               'The selected runner calls no separate candidate completion function.')
    else:
        block['completion'] = (
            f"The candidate package must also define `{completion['signature']}` with external C linkage. "
            'After each entry invocation the runner calls this function before the closing cycle read '
            'and before any readback. It must wait until all results are visible in memory. '
            'The runner does not supply this function. Its normalized LLVM form must obey '
            'normalized_lowered_representation, including the exact fixed runtime ABI completion '
            'sequence in selected_abi.completion_asm. This metadata is supplied runtime support, '
            'not a kernel implementation or schedule.')
    return yaml.safe_dump(output, sort_keys=False).encode()


def is_projected_protocol(row):
    return row.get('note') == NOTE and row.get('as') == ALIAS and row.get('mode') == 'ro'


def prepare(common, *, destination, repo, input_format='shared_interface_v1'):
    """Pin a fresh public protocol from the already selected private caller."""
    from merlin.targetgen.sandbox.bwrap import resolve_grant

    selected = [row for row in common if row.get('note') == 'selected capability contract (ALL arms)']
    protocols = [row for row in common if row.get('path') == DECLARED]
    if len(selected) != 1 or len(protocols) != 1 or selected[0].get('mode') != 'ro':
        raise ValueError('runtime ABI projection requires one selected caller and public protocol')
    contract_path = _regular(resolve_grant(selected[0]['path'], Path(repo)).absolute())
    protocol_path = _regular(resolve_grant(DECLARED, Path(repo)).absolute())
    contract_raw = contract_path.read_bytes()
    contract = yaml.safe_load(contract_raw)
    symbol = selected_symbol(contract)
    raw = rewrite_protocol(protocol_path.read_bytes(), contract=contract, input_format=input_format)
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    target = destination / ALIAS
    target.write_bytes(raw)
    target.chmod(0o444)
    destination.chmod(0o555)
    replacement = dict(path=str(target.absolute()), mode='ro', **{'as': ALIAS}, note=NOTE)
    projected = [replacement if row.get('path') == DECLARED else copy.deepcopy(row) for row in common]
    record = dict(schema=SCHEMA, input_format=input_format, entry_symbol=symbol, callables=selected_callables(contract),
                  software_limits=selected_software_limits(contract),
                  target_contract=dict(path=str(contract_path), sha256=_sha(contract_raw)),
                  protocol=dict(path=str(target.absolute()), sha256=_sha(raw), alias=ALIAS))
    return projected, record


def bind_protocol(record, common):
    """Bind an EL1 neutral reserialization to the same selected callable name."""
    rows = [row for row in common if row.get('as') == ALIAS]
    if len(rows) != 1 or rows[0].get('mode') != 'ro':
        raise ValueError('runtime ABI lacks one exact public protocol alias')
    path = _regular(Path(rows[0]['path']))
    result = copy.deepcopy(record)
    result['protocol'] = dict(path=str(path), sha256=_sha(path.read_bytes()), alias=ALIAS)
    return result


def binding(bundle):
    """Validate pinned declarations without reading a mutable input fallback."""
    record = bundle.get(KEY)
    if not isinstance(record, dict) or set(record) != {
            'schema', 'input_format', 'entry_symbol', 'callables', 'software_limits', 'target_contract', 'protocol'}:
        raise ValueError('raw-RTL public runtime ABI binding is missing or malformed')
    if record['schema'] != SCHEMA:
        raise ValueError('public runtime ABI binding schema differs')
    source_inventory_rules(record['input_format'])
    from . import author_profiles
    declared_profile = bundle.get('author_profile')
    profile = author_profiles.resolve(declared_profile['name']) if declared_profile else None
    if (profile is not None and profile.raw_rtl_study
            and profile.disclosure['input_format'] != record['input_format']):
        raise ValueError('public source protocol differs from its selected author profile')
    symbol = record['entry_symbol']
    if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isidentifier():
        raise ValueError('public runtime ABI binding has no plain callable symbol')
    for kind in ('target_contract', 'protocol'):
        pin = record[kind]
        expected = {'path', 'sha256', 'alias'} if kind == 'protocol' else {'path', 'sha256'}
        if (not isinstance(pin, dict) or set(pin) != expected
                or not isinstance(pin.get('path'), str) or not Path(pin['path']).is_absolute()
                or '..' in Path(pin['path']).parts
                or not is_sha256(str(pin.get('sha256')))):
            raise ValueError('public runtime ABI has an invalid content pin')
    pin = record['protocol']
    if pin['alias'] != ALIAS:
        raise ValueError('public runtime ABI uses a different protocol alias')
    rows = [row for row in bundle.get('allowed', ()) if row.get('path') == pin['path']]
    aliases = [row for row in bundle.get('allowed', ())
               if row.get('as', Path(str(row.get('path', ''))).name) == ALIAS]
    if (len(rows) != 1 or len(aliases) != 1 or aliases[0] != rows[0]
            or rows[0].get('as') != ALIAS or rows[0].get('mode') != 'ro'
            or any(row.get('path') == DECLARED for row in bundle.get('allowed', ()))):
        raise ValueError('selected public runtime ABI is not the unique exact public protocol')
    return record


def verify(bundle, *, workspace=None, repo=None):
    """Compare frozen public bytes with the same selected native caller contract.

    Before freezing, preparation reads its own newly published files. A native
    caller must supply workspace+repo and resolves both pins only through the
    verified private/public snapshot owner, never live fallback files.
    """
    record = binding(bundle)
    declared = [Path(record[key]['path']) for key in ('target_contract', 'protocol')]
    if workspace is not None:
        if repo is None:
            raise ValueError('frozen runtime ABI verification requires its repository')
        from merlin.targetgen.sandbox.bwrap import snapshot_input_paths
        paths = snapshot_input_paths(Path(workspace), bundle, declared, repo=Path(repo))
    elif repo is not None:
        raise ValueError('native runtime ABI verification requires its workspace')
    else:
        paths = declared
    payloads = [_regular(path).read_bytes() for path in paths]
    for key, raw in zip(('target_contract', 'protocol'), payloads, strict=True):
        if _sha(raw) != record[key]['sha256']:
            raise ValueError('public runtime ABI content differs from its selected pin')
    contract, protocol = [yaml.safe_load(raw) for raw in payloads]
    selected = selected_symbol(contract)
    callables, limits = selected_callables(contract), selected_software_limits(contract)
    block = (protocol or {}).get('logical_kernel_abi') or {}
    signature = block.get('signature')
    if (selected != record['entry_symbol'] or block.get('symbol') != selected
            or record['callables'] != callables or block.get('required_callables') != callables
            or record['software_limits'] != limits or block.get('software_limits') != limits
            or not isinstance(signature, str)
            or not _void_signature(signature, selected)):
        raise ValueError('public callable ABI differs from the selected native caller')
    # Completion responsibility is part of the frozen software protocol too;
    # a correct symbol list beside the old contradictory prose is insufficient.
    rewritten = yaml.safe_load(rewrite_protocol(payloads[1], contract=contract, input_format=record['input_format']))
    if (block.get('completion') != rewritten['logical_kernel_abi']['completion']
            or block.get('normalized_lowered_representation') != selected_lowered_representation(contract)
            or protocol.get('whole_program_attribution', {}).get('selected_source_inventory')
               != source_inventory_rules(record['input_format'])
            or protocol.get('whole_program_attribution', {}).get('required_plan')
               != rewritten['whole_program_attribution']['required_plan']):
        raise ValueError('public callable ABI has contradictory completion responsibility')
    return dict(schema=SCHEMA, entry_symbol=selected, callables=callables, software_limits=limits,
                protocol=copy.deepcopy(record['protocol']),
                target_contract=copy.deepcopy(record['target_contract']))
