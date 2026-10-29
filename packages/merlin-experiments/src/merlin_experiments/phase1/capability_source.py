"""Public interface structure for capability planning, without executing source."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path


def analyze(capsule):
    """Extract operation/type/dataflow requirements through existing source parsers.

    Only the declared interface is opened. No tensor inputs, golden outputs,
    reference implementations or model artifacts are inspected. Parse failures
    remain explicit unknowns and never make a capability appear supported.
    """
    filename, directory = capsule.get('interface_mlir'), capsule.get('__dir__')
    if not filename or not directory:
        return {'status': 'unknown', 'reason': 'no declared public interface source'}
    root = Path(directory).resolve()
    source = root / filename
    if not source.resolve().is_relative_to(root):
        raise ValueError('capability interface source escapes its public capsule')
    if not source.is_file():
        return {'status': 'unknown', 'reason': 'declared public interface is unavailable'}
    if source.stat().st_size > 8 << 20:
        return {'status': 'unknown', 'reason': 'public interface exceeds bounded analysis size'}
    payload = source.read_bytes()
    identity = {'file': str(Path(filename)), 'sha256': hashlib.sha256(payload).hexdigest()}
    try:
        text = payload.decode()
        if 'merlin_iface.' in text:
            from merlin.targetgen.contract.interface_emit import parse_interface_mlir
            program = parse_interface_mlir(text)
            counts = Counter(row['opcode'] for row in program['commands'])
            return {**identity, 'status': 'parsed', 'representation': 'merlin_iface',
                    'operations': dict(sorted(counts.items())),
                    'tensors': [{key: value[key] for key in ('name', 'shape', 'dtype', 'role') if key in value}
                                for name, spec in sorted(program['tensors'].items())
                                for value in [dict(spec, name=name)]],
                    'graph': _interface_graph(program['commands'])}
        from merlin.frontends.linalg_mlir import parse_mlir_text
        module = parse_mlir_text(text)
        operations = list(module.walk())
        counts = Counter(op.name for op in operations if op.name != 'builtin.module')
        positions = {op: i for i, op in enumerate(operations)}
        edges = set()
        for i, op in enumerate(operations):
            for operand in op.operands:
                owner = operand.owner
                if owner in positions:
                    edges.add((positions[owner], i))
        tensor_types = {}
        for op in operations:
            for value in (*op.operands, *op.results):
                typ = value.type
                if hasattr(typ, 'get_shape') and hasattr(typ, 'get_element_type'):
                    record = {'shape': list(typ.get_shape()), 'dtype': str(typ.get_element_type())}
                    tensor_types[str(typ)] = record
        return {**identity, 'status': 'parsed', 'representation': 'typed_mlir',
                'operations': dict(sorted(counts.items())),
                'tensor_types': [tensor_types[k] for k in sorted(tensor_types)],
                'graph': {'nodes': len(operations), 'operations': [op.name for op in operations],
                          'producer_consumer_edges': sorted(edges),
                          'shared_producers': _fanout(edges)}}
    except Exception as error:  # Parser coverage is descriptive; never silently drop a source.
        return {**identity, 'status': 'unknown', 'reason': type(error).__name__,
                'note': 'Inspect the public interface; unsupported parsing does not remove obligations.'}


def _fanout(edges):
    users = Counter(a for a, _ in edges)
    return sorted(a for a, count in users.items() if count > 1)


def _interface_graph(commands):
    producers, edges = {}, set()
    for i, command in enumerate(commands):
        operands = command.get('operands') or {}
        for name in operands.values():
            if isinstance(name, str) and name in producers:
                edges.add((producers[name], i))
        output = command.get('result')
        if isinstance(output, str):
            producers[output] = i
        for key in ('out', 'dst'):
            name = operands.get(key)
            if isinstance(name, str):
                producers[name] = i
    return {'nodes': len(commands), 'operations': [c['opcode'] for c in commands],
            'producer_consumer_edges': sorted(edges),
            'shared_producers': _fanout(edges)}
