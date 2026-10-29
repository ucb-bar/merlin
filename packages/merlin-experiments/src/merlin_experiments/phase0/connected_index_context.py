"""Reprove index premises on an exact cut with separately pinned capture metadata.

Only closed range/count/extract forms are handled. This is a source observation,
not a serialized host grant or whole capture/preparation equivalence claim.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from merlin.common import mlir_query as mq
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.llvmlower.target_data_layout import selected_index_bits
from merlin.xdsl_dialects._common import text
from merlin_experiments.phase1.feedback import private_index_source as index
from merlin_experiments.phase1.feedback import private_literal_arange as arange


def _read(path, expected):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("connected index context requires direct source files")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("connected index context source pin differs")
    return raw



def _root_body(op):
    """Canonical typed body with explicit external SSA; provenance is checked separately."""
    from xdsl.ir import Block, Operation
    from merlin.xdsl_dialects._common import text
    inside = set(op.walk())
    blocks = {block for nested in inside for region in nested.regions for block in region.blocks}
    external = []
    for nested in op.walk():
        for value in nested.operands:
            if ((isinstance(value.owner, Operation) and value.owner in inside)
                    or isinstance(value.owner, Block) and value.owner in blocks):
                continue
            if value not in external:
                external.append(value)
    boundary = Block(arg_types=[value.type for value in external])
    clone = op.clone(value_mapper=dict(zip(external, boundary.args, strict=True)))
    for nested in clone.walk():
        nested.attributes = {k:v for k,v in nested.attributes.items() if not k.startswith('prov.')}
    boundary.add_op(clone)
    return ([str(value.type) for value in external], text(clone, generic=True))


def _capture_join(model, trace, roots):
    """Require selected cut roots to retain the exact captured typed bodies and trace owners."""
    normalized, _ = normalize_capture_mlir(model.decode())
    parsed = tuple(mq.walk(mq.parse(normalized)))
    mlir = trace.get('mlir') or {}
    entries, joins = mlir.get('operations'), mlir.get('source_correspondence')
    if not isinstance(entries, list) or len(entries) != len(parsed) or not isinstance(joins, list):
        raise ValueError('connected index capture operation roster differs')
    result = []
    for node_id, projected in roots.items():
        selected = [r for r in joins if isinstance(r, dict) and r.get('node_id') == node_id]
        owned = {i for i, op in enumerate(parsed) if index._ids(op) == (node_id,)}
        if (len(selected) != 1 or selected[0].get('status') != 'lowered'
                or type(selected[0].get('mlir_ordinals')) is not list
                or any(type(i) is not int for i in selected[0]['mlir_ordinals'])
                or set(selected[0]['mlir_ordinals']) != owned
                or len(selected[0]['mlir_ordinals']) != len(owned)
                or any(r.get('node_id') != node_id and owned.intersection(r.get('mlir_ordinals') or [])
                       for r in joins if isinstance(r, dict))):
            raise ValueError('connected index capture source correspondence differs')
        captured = [(i, parsed[i]) for i in sorted(owned) if parsed[i].name in {'linalg.generic','linalg.reduce'}]
        if len(captured) != len(projected):
            raise ValueError('connected index captured compute roster differs')
        for (old_i, old), (new_i, new) in zip(captured, projected, strict=True):
            if old.name != new.name or _root_body(old) != _root_body(new):
                raise ValueError('connected index typed capture body differs')
            result.append({'source_node_id':node_id, 'capture_ordinal':old_i, 'projected_ordinal':new_i,
                           'typed_body_sha256':sha256(json.dumps(_root_body(new),sort_keys=True).encode()).hexdigest()})
    return result


def prove(context, program: Path, capture: Path, *, capture_model_sha256: str,
          frontend_trace_sha256: str, capture_receipt_sha256: str):
    """Recompute subset facts, never remap saved full-program proof ordinals."""
    from .connected_source_context import require_context

    require_context(context, {"region": program})
    source_binding = context.source_binding(program)
    selected = context.for_program(program)["selected_index_observation"]
    width = selected_index_bits(selected)
    capture = Path(capture)
    model_path, trace_path, receipt_path = (capture / n for n in ("model.mlir", "frontend-trace.json", "capture_receipt.json"))
    model = _read(model_path, capture_model_sha256)
    trace_bytes = _read(trace_path, frontend_trace_sha256)
    receipt_bytes = _read(receipt_path, capture_receipt_sha256)
    trace, receipt = json.loads(trace_bytes), json.loads(receipt_bytes)
    artifacts = receipt.get("artifacts") or {}
    if (receipt.get("schema") != "m2m.capture-receipt.v1"
            or artifacts.get("model.mlir", {}).get("sha256") != capture_model_sha256
            or artifacts.get("frontend-trace.json", {}).get("sha256") != frontend_trace_sha256):
        raise ValueError("connected index capture receipt does not bind source and trace")
    prepared = (trace.get("graphs") or {}).get("prepared") or {}
    if (trace.get("schema") != "m2m.frontend_trace.v1" or trace.get("status") != "complete"
            or prepared.get("status") != "complete" or (trace.get("mlir") or {}).get("sha256") != capture_model_sha256
            or (trace.get("mlir") or {}).get("bytes") != len(model)):
        raise ValueError("connected index trace is incomplete or belongs to another capture")
    nodes = prepared.get("nodes")
    if not isinstance(nodes, list) or any(not isinstance(n, dict) or not isinstance(n.get("id"), str) for n in nodes):
        raise ValueError("connected index trace lacks prepared node identities")
    by_id = {n['id']: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("connected index trace repeats a prepared node")
    raw = program.read_bytes()
    normalized, normalization = normalize_capture_mlir(raw.decode())
    module = mq.parse(normalized)
    module.verify()
    parsed = tuple(mq.walk(module))
    ordinals = {id(op): i for i, op in enumerate(parsed)}
    ranges, literals, range_rows = {}, {}, []
    roots = {}
    for ordinal, op in enumerate(parsed):
        target = mq.attr_str(op, 'prov.aten')
        if op.name not in {'linalg.generic', 'linalg.reduce'} or target not in {
                'aten.arange.default', 'aten.arange.start', 'aten.arange.start_step', 'aten.index.Tensor'}:
            continue
        ids = index._ids(op)
        if len(ids) != 1 or ids[0] not in by_id or by_id[ids[0]].get('target') != target:
            raise ValueError("connected index source identity has no exact prepared trace node")
        if target == 'aten.index.Tensor':
            roots.setdefault(ids[0], []).append((ordinal, op))
            continue
        start, end, step, extent = arange._range(by_id[ids[0]])
        arange._body(op, start, step, extent, width)
        literal = sha256(json.dumps([start, end, step, extent], separators=(',', ':')).encode()).hexdigest()
        ranges[id(op)], literals[id(op)] = (start, end, step, extent), literal
        range_rows.append({'ordinal':ordinal, 'source_node_id':ids[0], 'range':[start,end,step,extent], 'literal_sha256':literal})
    joins = _capture_join(model, trace, roots)
    records = []
    for node_id, owned_roots in roots.items():
        owned = {i for i, op in enumerate(parsed) if index._ids(op) == (node_id,)}
        if len(owned_roots) == 2 and {op.name for _, op in owned_roots} == {'linalg.generic', 'linalg.reduce'}:
            reduction = next(op for _, op in owned_roots if op.name == 'linalg.reduce')
            fact = index._mask_count(reduction, owned, ordinals, width)
        elif len(owned_roots) == 1 and owned_roots[0][1].name == 'linalg.generic':
            fact = index._indexed_extract(owned_roots[0][1], index_bits=width, ranges=ranges,
                                          range_literals=literals, ordinals=ordinals)
        else:
            raise ValueError("connected index node has an unmatched or partial compute roster")
        if not set(fact['component_ordinals']) <= owned:
            raise ValueError("connected index components belong to another source node")
        records.append({'source_node_id':node_id, 'compute_ordinals':[i for i, _ in owned_roots], **fact})
    require_context(context, {"region": program})
    for path, pin in ((model_path,capture_model_sha256),(trace_path,frontend_trace_sha256),(receipt_path,capture_receipt_sha256)):
        _read(path, pin)
    return {'schema':'merlin.connected_index_source_observation.v1', **source_binding,
            'status':'source_proved_pending_capture_preparation_join_host_and_build',
            'normalized_projected_sha256': normalization['output_sha256'],
            'capture_model_sha256':capture_model_sha256,'frontend_trace_sha256':frontend_trace_sha256,
            'capture_receipt_sha256':capture_receipt_sha256, 'selected_index_observation':selected,
            'ranges':range_rows,'records':records, 'capture_root_joins':joins,
            'parsed_module_sha256':sha256(text(module, generic=True).encode()).hexdigest(),
            'n_source_operations':len(parsed),
            'scope':'closed cut index forms and literal graph metadata only; full capture/preparation equivalence, host placement and linked numerics remain unproved'}
