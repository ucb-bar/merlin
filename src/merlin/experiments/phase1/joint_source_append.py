"""Explicit source-only member admission; no runtime or calibration authority."""
from __future__ import annotations

import json
from pathlib import Path

from ..corpus.projection_shapes import contractions
from ..phase2.contracts import document_sha256
from .feedback import joint_performance as PERF


def verified_members(evidence):
    """Recompute each reviewed independent integer contraction's geometry."""
    from merlin.llvmlower.device_shim import logical_buffers
    from merlin.targetgen.contract.interface_emit import parse_interface_mlir
    from .joint_source_revision import _capsule

    if set(evidence) != {'schema', 'members'} or evidence['schema'] != 'merlin.source_member_append_review.v1':
        raise ValueError('source append requires an explicit member review')
    rows = evidence['members']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 64:
        raise ValueError('source append requires a bounded nonempty roster')
    result = {}
    projections = {}
    for row in rows:
        if set(row) != {'capsule', 'projection', 'member'}:
            raise ValueError('source append has unreviewed fields')
        capsule, files = _capsule(row['capsule'])
        name = capsule.get('name')
        if not isinstance(name, str) or not name or name in result:
            raise ValueError('source append has duplicate or missing identity')
        source = capsule.get('interface_mlir')
        screen = capsule.get('software_screen', {})
        if (not isinstance(source, str) or source not in files
                or screen.get('status') != 'admitted'
                or screen.get('program_sha256') != files[source]
                or capsule.get('numeric_policy') != {'compare': 'exact_int', 'dtype': 'i32'}
                or capsule.get('required_oracle_tiers') != ['L0', 'L1', 'L2']):
            raise ValueError('source append lacks admitted independent development contract')
        operation = capsule.get('operation', {})
        attrs = operation.get('attributes', {})
        if operation.get('op') != 'matmul' or attrs.get('epilogue') != [] or attrs.get('output_dtype') != 'i32':
            raise ValueError('source append supports only raw independent integer contractions')
        root = Path(row['capsule']['path']).parent
        interface = (root / source).read_text()
        commands = parse_interface_mlir(interface)['commands']
        if [c['opcode'] for c in commands] != ['RES_PACK', 'MATMUL_RESIDENT', 'COMMIT', 'EVICT']:
            raise ValueError('source append must contain exactly one raw contraction')
        pack, multiply, commit, evict = commands
        if (pack['operands']['src'] != attrs['weight']
                or multiply['operands']['lhs'] != attrs['lhs']
                or multiply['operands']['rhs'] != pack['operands']['dst']
                or commit['operands']['src'] != multiply['operands']['dst']
                or commit['operands']['dst'] != attrs['out']
                or commit.get('attributes') != {'epilogue': [], 'output_dtype': 'i32'}
                or evict['operands']['handle'] != pack['operands']['dst']):
            raise ValueError('source append differs from its raw contraction contract')
        buffers, reason = logical_buffers(interface)
        if not buffers or len(buffers) != 3:
            raise ValueError('source append interface has unsupported logical buffers: ' + reason)
        by_name = {b.name: b for b in buffers}
        a, b, out = (by_name[attrs[key]] for key in ('lhs', 'weight', 'out'))
        if (a.kind != 'input' or b.kind != 'input' or out.kind != 'output'
                or a.dtype != 'i8' or b.dtype != 'i8' or out.dtype != 'i32'
                or len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]
                or out.shape != (a.shape[0], b.shape[1])):
            raise ValueError('source append has unsupported interface arithmetic')
        m, k, n = a.shape[0], a.shape[1], b.shape[1]
        projected = row['projection']
        key = (projected['path'], projected['sha256'])
        if key not in projections:
            from merlin.runtime import _native_files as files_api
            if len(projections) >= 4:
                raise ValueError('source append exceeds graph projection budget')
            before = files_api.pin(projected['path'], max_bytes=64 << 20)
            if before != projected:
                raise ValueError('source append graph projection changed')
            graph = json.loads(Path(projected['path']).read_bytes())
            if files_api.pin(projected['path'], max_bytes=64 << 20) != before:
                raise ValueError('source append graph changed during admission')
            projections[key] = contractions(graph)
        matches = [g for g in projections[key] if g['shape_mkn'] == [m, k, n]]
        if len(matches) != 1:
            raise ValueError('source append shape absent from approved graph')
        binding_path = root / 'source-shape-binding.json'
        if binding_path.name not in files:
            raise ValueError('source append lacks exact occurrence binding')
        binding = PERF._read({'path': str(binding_path), 'sha256': files[binding_path.name]})
        if (any(binding.get(key) != value for key, value in matches[0].items())
                or binding.get('projection_sha256') != projected['sha256']
                or any(binding.get(key) is not False for key in ('captured_boundary_values', 'connected_cut', 'full_model'))
                or binding.get('development_engine') != 'spike'):
            raise ValueError('source append occurrence or execution scope differs')
        member = dict(capsule_sha256=document_sha256(capsule), source_sha256=files[source],
                      contractions=[[m, k, n]], logical_read_bytes=m*k+k*n,
                      logical_write_bytes=4*m*n, scalar_products=m*k*n,
                      source_scope='mathematical logical tensor footprint only; no physical DMA/reuse claim')
        if row['member'] != member:
            raise ValueError('source append geometry differs from admitted interface')
        result[name] = member
    return result
