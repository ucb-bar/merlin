"""Standalone read-only queries over the staged public capability inventory.

This exact file is served to the author. Keep it standard-library-only: it needs
no private experiment imports and can open only its adjacent public inventory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

INPUT = 'PUBLIC_CAPABILITIES.json'
MAX_INPUT_BYTES = 16 << 20
MAX_OUTPUT_BYTES = 32 << 10


def read_inventory(directory):
    path = Path(directory) / INPUT
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError('expected a bounded ordinary adjacent public inventory')
    value = json.loads(path.read_text())
    if value.get('schema') != 'merlin.public_capability_inventory.v1':
        raise ValueError('unsupported public inventory schema')
    return value


def query(value, *, group=None, capsule=None, offset=0, limit=40):
    if group is not None and capsule is not None:
        raise ValueError('select one group or one capsule')
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError('offset must be nonnegative and limit must be in 1..100')
    if capsule is not None:
        member = value['members'].get(capsule)
        if member is None:
            raise ValueError('capsule is not in the selected public inventory')
        source = member['source_analysis']
        graph = source.get('graph') or {}
        fields = ('cohort', 'family', 'declared_operation', 'source_sha256', 'numeric_policy',
                  'layout_requirements', 'semantic_requirements', 'declared_tiers',
                  'development_engines', 'gsim_role', 'performance_priority')
        return {'capsule': capsule, **{k: member[k] for k in fields if k in member},
                'input_tensors': _page(member['input_tensors'], offset, limit),
                'source': {k: source[k] for k in ('status', 'reason', 'file', 'sha256', 'representation', 'operations')
                           if k in source},
                'graph_summary': {'nodes': graph.get('nodes'),
                    'producer_consumer_edges': len(graph.get('producer_consumer_edges', [])),
                    'shared_producers': len(graph.get('shared_producers', []))},
                'full_graph': 'on-demand source_analysis.graph in the adjacent public inventory',
                'qualification_authority': False}
    groups = value['capability_groups']
    if group is not None:
        matches = [key for key in groups if key.startswith(group)]
        if not group or len(matches) != 1:
            raise ValueError('group prefix must identify exactly one public capability group')
        key = matches[0]
        row = groups[key]
        return {'group': key, 'requirements': row['requirements'],
                'members': _page(row['members'], offset, limit),
                'small_canary_candidate': row['small_canary_candidate'],
                'large_generalization_candidate': row['large_generalization_candidate'],
                'selection_basis': row['selection_basis'], 'qualification_authority': False}
    rows = [{'group': key[:12], 'family': row['requirements']['family'],
             'members': len(row['members']), 'dtypes': row['requirements']['input_dtypes'],
             'operation_kinds': len(row['requirements']['operations']),
             'small_canary': row['small_canary_candidate']}
            for key, row in sorted(groups.items())]
    return {'public_cases': value['total'], 'cohorts': value['cohorts'],
            'groups': _page(rows, offset, limit), 'qualification_authority': False}


def _page(rows, offset, limit):
    selected = rows[offset:offset + limit]
    return {'total': len(rows), 'offset': offset, 'items': selected,
            'next_offset': offset + len(selected) if offset + len(selected) < len(rows) else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--groups', action='store_true', help='list compact capability groups (default)')
    selection.add_argument('--group', help='inspect one exact group or unique digest prefix')
    selection.add_argument('--capsule', help='inspect one selected public capsule')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=40, help='page size, 1..100')
    args = parser.parse_args(argv)
    try:
        value = read_inventory(Path(__file__).resolve().parent)
        result = query(value, group=args.group, capsule=args.capsule, offset=args.offset, limit=args.limit)
        text = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
        if len(text.encode()) > MAX_OUTPUT_BYTES:
            raise ValueError('query exceeds 32 KiB; request a smaller page or a single capsule')
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'capability query refused: {error}', file=sys.stderr)
        return 2
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
