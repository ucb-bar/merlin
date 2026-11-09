"""Bitwise boundary comparison. Exit status 1 means a mismatch, 2 bad input."""
import json
from pathlib import Path
import numpy as np
from .io import checked_path, digest


def compare_arrays(expected, actual):
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        return {'passed': False, 'reason': 'shape or dtype differs',
                'expected_shape': list(expected.shape), 'actual_shape': list(actual.shape),
                'expected_dtype': expected.dtype.str, 'actual_dtype': actual.dtype.str}
    eb = np.ascontiguousarray(expected).view(np.uint8).reshape(-1, expected.dtype.itemsize)
    ab = np.ascontiguousarray(actual).view(np.uint8).reshape(-1, actual.dtype.itemsize)
    different = np.any(eb != ab, axis=1)
    count = int(np.count_nonzero(different))
    result = {'passed': count == 0, 'elements': int(expected.size),
              'mismatching_elements': count, 'expected_sha256': digest(eb.tobytes()),
              'actual_sha256': digest(ab.tobytes())}
    if count:
        index = int(np.flatnonzero(different)[0])
        coord = list(np.unravel_index(index, expected.shape))
        result['first_mismatch'] = {'index': [int(i) for i in coord],
                                    'expected_bytes': eb[index].tobytes().hex(),
                                    'actual_bytes': ab[index].tobytes().hex()}
    return result


def compare_boundary(reference, actual, *, node, unit='mxu0', case=0,
                     boundary='output_bf16', raw=False, actual_dtype=None, operands=None):
    reference = checked_path(reference)
    manifest = json.loads(reference.read_text())
    if manifest.get('complete') is False:
        raise ValueError('reference run is incomplete')
    matches = [r for r in manifest['runs'] if r['unit'] == unit and r['case'] == case]
    if len(matches) != 1:
        raise ValueError('expected exactly one matching reference run')
    run = matches[0]
    if node.startswith('layer:'):
        entries = [r for r in run['layers'] if r['layer'] == node[6:]]
        key = 'output'
    elif node.startswith('final:'):
        index = int(node[6:])
        if index < 0 or index >= len(run['final']):
            raise ValueError('final output index outside reference')
        entries, key = [{boundary: run['final'][index],
                         'scope': 'Atlas quantized ops, unchanged PyTorch ordinary nodes'}], boundary
    else:
        entries, key = [e for e in run['entries'] if e['node'] == node], boundary
    if len(entries) != 1 or key not in entries[0]:
        raise ValueError('missing reference node or boundary')
    if operands is not None:
        if not entries[0].get('operands'):
            raise ValueError('operand provenance check requires a contraction boundary')
        aliases = {'a_codes': ('a_codes', 'lhs_codes', 'input_0'),
                   'a_exp': ('a_exp', 'lhs_exp', 'input_1'),
                   'b_codes': ('b_codes', 'rhs_codes', 'input_2'),
                   'b_exp': ('b_exp', 'rhs_exp', 'input_3'),
                   'bias_bf16': ('bias_bf16',)}
        differences = []
        with np.load(checked_path(operands), allow_pickle=False) as archive:
            for name, expected_operand in entries[0].get('operands', {}).items():
                selected = next((k for k in aliases.get(name, (name,)) if k in archive), None)
                if selected is None:
                    raise ValueError('observed operands missing: ' + name)
                value = archive[selected]
                observed = {'shape': list(value.shape), 'dtype': value.dtype.str,
                            'sha256': digest(value.tobytes())}
                if observed != expected_operand:
                    differences.append({'operand': name, 'expected': expected_operand,
                                        'actual': observed})
        if differences:
            return {'passed': False, 'node': node, 'unit': unit, 'case': case,
                    'reason': 'operand provenance differs; output comparison was not performed',
                    'operand_differences': differences}
    record = entries[0][key]
    path = checked_path(reference.parent / run['directory'] / record['file'])
    if not path.resolve().is_relative_to(reference.parent.resolve()):
        raise ValueError('reference boundary escapes its output directory')
    expected = np.load(path, allow_pickle=False)
    if not isinstance(expected, np.ndarray):
        raise ValueError('reference boundary must be a single NPY array')
    if (digest(expected.tobytes()) != record['sha256'] or list(expected.shape) != record['shape']
            or expected.dtype.str != record['dtype']):
        raise ValueError('reference boundary integrity failure')
    actual_path = checked_path(actual)
    if raw:
        dtype = np.dtype(actual_dtype or record['dtype'])
        actual_array = np.fromfile(actual_path, dtype=dtype)
        if actual_array.size == expected.size:
            actual_array = actual_array.reshape(expected.shape)
    else:
        actual_array = np.load(actual_path, allow_pickle=False)
        if not isinstance(actual_array, np.ndarray):
            raise ValueError('actual capture must be a single NPY array')
    return {'node': node, 'unit': unit, 'case': case, 'boundary': boundary,
            'scope': entries[0].get('scope', 'Atlas quantized op in unchanged PyTorch context'),
            **compare_arrays(expected, actual_array)}
