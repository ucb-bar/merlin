"""Resumable source-local fusion frontiers; proposals carry no legality authority."""
from __future__ import annotations

from collections import Counter
from itertools import islice
from math import prod

from merlin.common.digest import is_sha256
from merlin.transforms.certificates import ir_digest
from merlin.verify.lean_contract import document_digest


def _materialization(operations, ordinals):
    """Logical unfused tensor payloads, without inferring physical storage.

    One producer write and one read per distinct consumer describe a source
    screening model. Only a value with every observer inside the proposal can
    disappear entirely. Views, initializers, scalar SSA and target padding are
    not charged as materialized compute results here.
    """
    from xdsl.dialects.builtin import TensorType

    members = {child for op in operations for child in op.walk()}
    rows, unknown, saved, live = [], [], 0, 0
    for op in operations:
        if _structural_family(op) is None:
            continue
        for index, value in enumerate(op.results):
            ty = value.type
            if not isinstance(ty, TensorType):
                continue
            shape, width = tuple(ty.get_shape()), getattr(ty.element_type, 'bitwidth', None)
            if any(d < 0 for d in shape) or type(width) is not int or width < 1:
                unknown.append({'producer_ordinal': ordinals[op], 'result': index, 'type': str(ty)})
                continue
            size = (prod(shape) * width + 7) // 8
            users = {use.operation for use in value.uses}
            external = users - members
            removable = bool(users) and not external
            benefit = size * (1 + len(users)) if removable else 0
            saved += benefit
            live += size if external else 0
            rows.append({'producer_ordinal': ordinals[op], 'result': index,
                         'payload_bytes': size, 'distinct_consumers': len(users),
                         'external_consumer_ordinals': sorted(ordinals[u] for u in external),
                         'all_observers_inside': removable, 'potential_saved_logical_bytes': benefit})
    return {'basis': 'static_logical_tensor_materialization',
            'potential_saved_logical_bytes': None if unknown else saved,
            'known_potential_saved_logical_bytes': saved,
            'live_output_payload_bytes': None if unknown else live,
            'known_live_output_payload_bytes': live,
            'values': rows, 'unknown_values': unknown,
            'physical_savings_claimed': False,
            'limitations': 'source screening only; excludes target padding, views, packing, operand reuse, storage pressure and execution costs'}


def plan_materialization_frontier(report, *, exact_cluster_limit=8):
    """Prioritize nonoverlapping source proposals for actual realization.

    Small conflict clusters use bounded subset search; larger ones use a
    deterministic greedy choice. All original candidates remain available.
    Neither this source-byte objective nor a chosen group authorizes a rewrite
    or promotes a performance winner. Current native legality and same-region
    cycle comparisons remain the execution/selection path.
    """
    if (type(report) is not dict or report.get('schema') != 'merlin.fusion_workflow.v1'
            or type(exact_cluster_limit) is not int or not 1 <= exact_cluster_limit <= 10
            or not isinstance(report.get('candidates'), (list, tuple))
            or len(report['candidates']) > 1024):
        raise ValueError('fusion screening requires bounded current discovery and conflict search')
    rows, excluded = [], []
    for row in report['candidates']:
        cost = row.get('materialization', {}).get('potential_saved_logical_bytes')
        if (row.get('source_legality') or {}).get('status') == 'declined' or row.get('status') == 'unsupported':
            excluded.append({'candidate_index': row['candidate_index'], 'reason': 'native source reader declined'})
        elif type(cost) is not int:
            excluded.append({'candidate_index': row['candidate_index'], 'reason': 'unknown materialization payload'})
        elif cost > 0 and row['operation_ordinals']:
            rows.append(row)
    if len({row['candidate_index'] for row in rows}) != len(rows):
        raise ValueError('fusion screening requires distinct discovery candidates')
    clusters, positions = [], {}
    for row in rows:
        touched = {positions[op] for op in row['operation_ordinals'] if op in positions}
        combined = [row]
        for index in sorted(touched):
            combined.extend(clusters[index])
            clusters[index] = []
        index = len(clusters)
        clusters.append(combined)
        for member in combined:
            for op in member['operation_ordinals']:
                positions[op] = index
    selected, methods = [], []
    for cluster in filter(None, clusters):
        cluster.sort(key=lambda row: row['candidate_index'])
        if len(cluster) <= exact_cluster_limit:
            best, best_score = (), (-1, -1)
            for bits in range(1 << len(cluster)):
                choice, occupied, score, source_checked = [], set(), 0, 0
                for index, row in enumerate(cluster):
                    if bits & (1 << index):
                        region = set(row['operation_ordinals'])
                        if occupied & region:
                            break
                        occupied.update(region)
                        choice.append(row)
                        score += row['materialization']['potential_saved_logical_bytes']
                        source_checked += (row.get('source_legality') or {}).get('status') == 'source_eligible'
                else:
                    if (score, source_checked) > best_score:
                        best, best_score = tuple(choice), (score, source_checked)
            selected.extend(best)
            methods.append('bounded_subset')
        else:
            occupied = set()
            for row in sorted(cluster, key=lambda r: (-r['materialization']['potential_saved_logical_bytes'],
                    (r.get('source_legality') or {}).get('status') != 'source_eligible', r['candidate_index'])):
                if not occupied.intersection(row['operation_ordinals']):
                    selected.append(row)
                    occupied.update(row['operation_ordinals'])
            methods.append('greedy')
    order = sorted(rows, key=lambda r: (-r['materialization']['potential_saved_logical_bytes'],
                   (r.get('source_legality') or {}).get('status') != 'source_eligible', r['candidate_index']))
    ordered_indices = {row['candidate_index'] for row in order}
    selected_indices = {row['candidate_index'] for row in selected}
    return {'basis': 'static_logical_tensor_materialization',
            'binding_sha256': report.get('binding_sha256'),
            'selected_candidate_indices': sorted(selected_indices),
            'potential_saved_logical_bytes': sum(row['materialization']['potential_saved_logical_bytes'] for row in selected),
            'evaluation_order': [row['candidate_index'] for row in order]
                + [row['candidate_index'] for row in report['candidates'] if row['candidate_index'] not in ordered_indices],
            'conflict_cluster_methods': methods, 'excluded_from_byte_selection': excluded,
            'scope': 'this discovery page; unvisited or other optimization benefits remain unranked',
            'rewrite_authority': False, 'performance_ranking': False}

def _boundary(operations, ordinals):
    # Regions may capture external SSA values without listing them as parent
    # operands. Include the entire owned tree, excluding its own block args.
    nested = tuple(dict.fromkeys(child for root in operations for child in root.walk()))
    members = set(nested)
    inputs, outputs = {}, {}
    for op in nested:
        for value in op.operands:
            owner = value.owner
            local_block_arg = hasattr(owner, "parent_op") and owner.parent_op() in members
            if owner not in members and not local_block_arg:
                inputs[value] = None
        for value in op.results:
            if any(use.operation not in members for use in value.uses):
                outputs[value] = None

    def value_record(value):
        owner = value.owner
        if owner in ordinals:
            identity = {"producer_ordinal": ordinals[owner], "result": list(owner.results).index(value)}
        else:
            parent = owner.parent_op() if hasattr(owner, "parent_op") else None
            identity = {"block_owner_ordinal": ordinals.get(parent), "argument": getattr(value, "index", None)}
        return {**identity, "type": str(value.type)}

    return {"live_inputs": [value_record(v) for v in inputs],
            "live_outputs": [value_record(v) for v in outputs],
            "source_ids": [{"ordinal": ordinals[op],
                            "region_id": getattr(op.attributes.get("prov.region_id"), "data", None),
                            "node_ids": [getattr(a, "data", None) for a in
                                         getattr(op.attributes.get("prov.source_node_ids"), "data", ())]}
                           for op in operations]}


def _structural_family(op):
    """Small public IR observation, never a hardware eligibility decision.

    Named matrix operations have declared contraction semantics. A generic is
    observed as elementwise only for parallel iterators and identity maps. Signed
    contractions use the existing exact scalar recurrence reader. Other forms
    remain unknown; selected realization readers own storage/target proofs.
    """
    if op.name in {"linalg.matmul", "linalg.batch_matmul", "linalg.matvec", "linalg.vecmat", "linalg.dot"}:
        return "contraction"
    if op.name != "linalg.generic":
        return None
    iterators = op.properties.get("iterator_types")
    maps = op.properties.get("indexing_maps")
    if not iterators or not maps or not iterators.data or not maps.data:
        return None
    if (all(getattr(item.data, "value", None) == "parallel" for item in iterators.data)
            and all(item.data == item.data.identity(item.data.num_dims) for item in maps.data)):
        return "elementwise_map"
    from xdsl.dialects.linalg.ops import GenericOp
    from merlin.transforms.dense_ir import recognize_signed_integer_matmul

    # Parser subclasses are normalized only at this read-only observation
    # boundary, retaining exact scalar operations, maps and unknown attributes.
    if isinstance(op, GenericOp) and len(op.operands) == 3:
        formats = tuple(str(getattr(value.type, 'element_type', '')) for value in op.operands)
        detached = GenericOp.create(operands=op.operands, result_types=[v.type for v in op.results],
            attributes=dict(op.attributes), properties=dict(op.properties), successors=op.successors,
            regions=[region.clone() for region in op.regions])
        try:
            recognize_signed_integer_matmul(detached, formats)
            return 'contraction'
        except ValueError:
            pass
        finally:
            detached.drop_all_references()
    return None



def _frontier(module, members, *, max_region_operations):
    from merlin.compile.scheduling.shared_input_groups import find_shared_input_groups, grow_shared_input_group
    from merlin.compile.scheduling.multioutput_fusion import read_source_multioutput
    from merlin.compile.scheduling.fusion_semantics import UnsupportedElementwise
    from xdsl.dialects import func

    ordinals = {op: i for i, op in enumerate(members)}
    typed = {op: family for op in members if (family := _structural_family(op)) is not None}
    # One deterministic forward frontier per root, not every connected subset.
    for producer, family in typed.items():
        region = [producer]
        while len(region) < max_region_operations:
            users = {use.operation for op in region for value in op.results for use in value.uses
                     if use.operation in typed and use.operation not in region
                     and use.operation.parent is producer.parent
                     and ordinals[use.operation] > ordinals[producer]}
            if not users:
                break
            consumer = min(users, key=ordinals.__getitem__)
            region.append(consumer)
            block = tuple(producer.parent.ops)
            adjacent = len(region) == 2 and block.index(consumer) == block.index(producer) + 1
            route = ('epilogue' if len(region) == 2 and family == 'contraction'
                     and typed[consumer] != 'contraction' else 'adjacent_elementwise'
                     if adjacent and all(typed[op] != 'contraction' for op in region)
                     else 'connected_residency')
            successors = {use.operation for op in region for value in op.results for use in value.uses
                          if use.operation in typed and use.operation not in region
                          and use.operation.parent is producer.parent
                          and ordinals[use.operation] > ordinals[producer]}
            growth_status = ('region_budget_exhausted' if successors and len(region) == max_region_operations
                             else 'expandable' if successors else 'complete')
            yield route, tuple(region), {'adjacent': adjacent, 'growth': 'forward_frontier',
                'growth_status': growth_status,
                'remaining_successor_ordinals': sorted(ordinals[op] for op in successors)}, None
    # The closed fanout reader owns observer/numeric legality, including refusals.
    for block in dict.fromkeys(op.parent for op in members if op.parent is not None):
        operations = tuple(block.ops)
        for position, producer in enumerate(operations):
            if producer.name != 'linalg.generic' or len(producer.results) != 1:
                continue
            users = {use.operation for use in producer.results[0].uses}
            adjacent = operations[position:position + 3]
            if (len(users) < 2 or len(adjacent) != 3
                    or any(typed.get(op) != 'elementwise_map' for op in adjacent)):
                continue
            detail = {'producer_observers': len(users), 'legal_scope': 'adjacent_pointwise_two_output_fanout'}
            # Defer the potentially expensive closed reader until this page consumes the row.
            def check(operations=adjacent):
                try:
                    selected = read_source_multioutput(module, operations)
                    return {'status': 'source_eligible', 'reason': 'native source dependencies and scalar bodies checked',
                            'shape': list(selected.shape), 'live_outputs': len(selected.outputs)}
                except (ValueError, UnsupportedElementwise) as error:
                    return {'status': 'declined', 'reason': str(error)[:300]}
            yield 'multioutput', adjacent, detail, check
    try:
        groups = find_shared_input_groups(module)
    except ValueError as error:
        yield 'shared_input', (), {'reader_status': 'unsupported', 'reason': str(error)[:240]}, None
        groups = ()
    if groups:
        function = next(op for op in module.body.block.ops if type(op) is func.FuncOp)
        operations = tuple(function.body.block.ops)
        for group in groups:
            for prefix in grow_shared_input_group(group):
                yield 'shared_input', tuple(operations[i] for i in prefix.source_op_indices), prefix.record(), None


def discover(module, *, capability_contract, numerical_policy_sha256, preparation_requests=(),
             max_candidates=256, cursor=None, max_region_operations=8):
    """Return a bounded page and explicit continuation, without mutating source.

    Resume with ``cursor=report['next_cursor']`` and unchanged inputs. The cursor
    binds exact source, capability/numeric contracts and preparation requests.
    Pages report exhaustion rather than silently dropping candidates or failing
    a compilation. Traversal grows one forward frontier per producer and each
    shared-input prefix; it deliberately does not enumerate all graph subsets.
    Row indices are stable within this bound scope, not persistent operation IDs.
    """
    from merlin.compile import compiler_observations as C
    from merlin.compile.fusion_workflow import ROUTES

    if (not is_sha256(numerical_policy_sha256) or type(max_candidates) is not int
            or not 0 < max_candidates <= 1024 or type(max_region_operations) is not int
            or not 2 <= max_region_operations <= 64):
        raise ValueError('fusion discovery requires pinned numerics and bounded page/region sizes')
    members = tuple(module.walk())
    if len(members) > 10000:
        raise ValueError('fusion source operation budget exceeded')
    source = ir_digest(module)
    ordinals = {op: i for i, op in enumerate(members)}
    preparation_requests = tuple(preparation_requests)
    packing = C.packing(module, preparation_requests)
    requests = [{'consumer_ordinal': ordinals[request.consumer],
                 'operand_indices': [i for i, value in enumerate(request.consumer.operands)
                                     if value is request.value],
                 'format_sha256': request.format_sha256} for request in preparation_requests]
    binding = {'source_ir_sha256': source, 'capability_sha256': document_digest(dict(capability_contract)),
               'numerical_policy_sha256': numerical_policy_sha256,
               'preparation_sha256': document_digest({'observation': packing, 'requests': requests}), 'max_region_operations': max_region_operations}
    binding_sha = document_digest(binding)
    offset = 0
    if cursor is not None:
        if (type(cursor) is not dict or set(cursor) != {'schema', 'binding_sha256', 'offset'}
                or cursor['schema'] != 'merlin.fusion_cursor.v1' or cursor['binding_sha256'] != binding_sha
                or type(cursor['offset']) is not int or not 0 <= cursor['offset'] <= 1_000_000):
            raise ValueError('fusion cursor differs from source, contracts or discovery scope')
        offset = cursor['offset']
    caps = {}
    for unit in capability_contract.get('compute_units', ()):
        if not isinstance(unit, dict):
            raise ValueError('public compute-unit declarations must be objects')
        for declaration in unit.get('semantic_capabilities', ()):
            if not isinstance(declaration, dict) or not isinstance(declaration.get('family'), str):
                raise ValueError('public semantic declarations must explicitly name a family')
            caps.setdefault(declaration['family'], []).append({'unit': unit.get('name'), 'declaration': dict(declaration)})

    def proposals():
        yield from _frontier(module, members, max_region_operations=max_region_operations)
        for opportunity in packing['opportunities']:
            yield ('preparation_reuse', tuple(members[i] for i in opportunity['consumer_ordinals']),
                   opportunity, None)

    stream = islice(proposals(), offset, None)
    rows, fanout, reader = [], [], None
    for index, (route, operations, detail, check) in enumerate(islice(stream, max_candidates)):
        if not operations:
            reader = detail
        families = sorted({_structural_family(op) for op in operations} - {None})
        observations = []
        for family in families:
            declarations = caps.get(family, [])
            observations.append({'family': family, 'status': 'declared' if declarations else 'unknown',
                'basis': 'explicit_public_family_declaration_only', 'declarations': declarations,
                'composed_with': [row['declaration'].get('composed_with') for row in declarations],
                'forms': [row['declaration'].get('forms') for row in declarations],
                'applicability': 'unknown_until_native_legality_checks'})
        checked = check() if check else None
        row = {'candidate_index': offset + index, 'route': route, 'entrypoint': ROUTES[route],
               'status': 'unsupported' if not operations else 'unknown',
               'reason': detail.get('reason') if not operations else 'source opportunity; native realization required',
               'operation_ordinals': [ordinals[op] for op in operations], **_boundary(operations, ordinals),
               'materialization': _materialization(operations, ordinals),
               'capability_observations': observations, 'detail': detail,
               'source_legality': checked, 'binding_sha256': binding_sha,
               'storage_lifetimes': 'unproved', 'schedule_sha256': None, 'whole_region_cost': None,
               'legality_premises': ['retain every live output and external observer',
                   'preserve exact numeric order', 'check storage, effects, completion and resources',
                   'consume actual realization through ordinary emission']}
        rows.append(row)
        if checked:
            fanout.append({**detail, **checked, 'operation_ordinals': row['operation_ordinals']})
    more = next(stream, None) is not None
    if ir_digest(module) != source or document_digest(dict(capability_contract)) != binding['capability_sha256']:
        raise ValueError('fusion discovery source or contracts changed')
    counts = Counter(row['route'] for row in rows)
    report = {'schema': 'merlin.fusion_workflow.v1', **binding, 'binding_sha256': binding_sha,
            'planned': {'source_regions': sum(bool(row['operation_ordinals']) for row in rows)}, 'emitted': {},
            'maturity': 'source_opportunities_not_legalized', 'physical_savings_claimed': False,
            'status': 'budget_exhausted' if more else 'complete', 'page_offset': offset,
            'next_cursor': {'schema': 'merlin.fusion_cursor.v1', 'binding_sha256': binding_sha,
                            'offset': offset + len(rows)} if more else None,
            'search_scope': 'bounded forward frontiers, adjacent fanouts and consecutive shared-input prefixes; not all subsets',
            'structural_reader_scope': 'named matrices, exact signed contraction generics and parallel identity-map generics',
            'unclassified_tensor_operations': [{'ordinal': ordinals[op], 'operation': op.name}
                for op in members if any(hasattr(v.type, 'get_shape') for v in op.results)
                and _structural_family(op) is None],
            'metric_definitions': {'planned.source_regions': 'page source opportunities, not legalized regions'},
            'routes': {route: {'status': 'unknown' if counts[route] else 'no-match', 'candidates': counts[route]}
                       for route in ROUTES}, 'shared_input_reader': reader, 'candidates': rows,
            'pointwise_fanout': {'planned': {'source_regions': sum(row['status'] == 'source_eligible' for row in fanout)},
                                'candidates': fanout, 'emitted': {}},
            'fallback': 'preserve original source and unfused schedule; no discovery mutation',
            'missing': ['general connected graph residency realization', 'native proof for each selected route',
                        'complete-region numerical checks and same-scope runtime cost']}
    report['materialization_frontier'] = plan_materialization_frontier(report)
    return report
