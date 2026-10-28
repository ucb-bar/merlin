"""Reindex frozen chunks with the current ordinary TensorIR extraction path.

No compilation, timing, calibration, reference generation or input mutation.
Invoke with the selected TVM python/apps and Merlin runner folders on PYTHONPATH.
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import measure
import tvm
from firesim_job_pool import identity
from tune_firesim_chunks import measurement_sources
from tune_models import extract_inventory
from tvm.relax.backend.contrib.gemmini_search import WORKLOAD_ATTR
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.input_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    catalog = json.loads((root / 'catalog.json').read_text())
    old = json.loads((root / 'task-owners.json').read_text())
    caps = GemminiCapabilities()
    owners, representatives, chunks, inputs = {}, {}, [], []
    graph = dict(graph_fusion_level=2, prepack_weights=False, bias_preload=False, graph_cse=False)
    for row in sorted(catalog['chunks'], key=lambda r: r['id']):
        if row['status'] != 'ready_for_tuning':
            continue
        started = time.monotonic()
        program, fixture = root / row['program'], root / row['fixture']
        inputs.append(dict(chunk=row['id'], program=identity(program), fixture=identity(fixture)))
        mod = tvm.ir.load_json(program.read_text())
        prepared, tasks, inventory = extract_inventory(mod, caps, fusion_level=2,
            prepack_weights=False, bias_preload=False, eliminate_common_subexpr=False)
        kinds = {item['task']: item['kind'] for item in inventory}
        keys, owned = [], []
        for task in tasks:
            if kinds[task.task_name] != 'contraction':
                continue
            key = str(task.dispatched[0]['main'].attrs[WORKLOAD_ATTR])
            keys.append(key)
            if key not in owners:
                owned.append(key)
                owners[key] = dict(owner=row['id'], model=row['model'], program=row['program'],
                    fixture=row['fixture'], occurrences=[], **graph)
                representatives[key] = task.dispatched[0]
                (output / (key + '.tir.json')).write_text(tvm.ir.save_json(task.dispatched[0]))
            elif not tvm.ir.structural_equal(representatives[key], task.dispatched[0]):
                raise ValueError('Structural-key collision: ' + key)
            owners[key]['occurrences'].append(dict(chunk=row['id'], task=task.task_name, weight=int(task.weight)))
        record = dict(id=row['id'], model=row['model'], program=row['program'], fixture=row['fixture'],
            tasks=inventory, contraction_keys=keys, owned_keys=owned,
            prepared_sha256=__import__('hashlib').sha256(tvm.ir.save_json(prepared).encode()).hexdigest(),
            extraction_seconds=time.monotonic()-started)
        chunks.append(record)
        write(output / 'chunks.json', chunks)
        print(row['id'], len(keys), 'contractions;', len(owners), 'distinct total', flush=True)
    campaigns = [dict(id=row['id'], program=row['program'], fixture=row['fixture'],
                      model=row['model'], workload_keys=row['owned_keys'], **graph)
                 for row in chunks if row['owned_keys']]
    write(output / 'task-owners.json', owners)
    write(output / 'campaign-catalog.json', dict(schema='firesim_owner_catalog_v1',
        status='extraction_only_not_execution_qualified', input_root=str(root), campaigns=campaigns))
    write(output / 'bindings.json', dict(catalog=identity(root / 'catalog.json'), inputs=inputs,
        compiler=identity(tvm._ffi.base._LIB._name), python_sources=measurement_sources(tvm, measure),
        extractor=identity(__file__), capabilities=caps.fingerprint))
    write(output / 'summary.json', dict(status='extraction_only', graph_options=graph,
        chunks_by_model=dict(Counter(r['model'] for r in chunks)),
        owners_by_model=dict(Counter(r['model'] for r in campaigns)),
        tasks_by_model=dict(Counter(r['model'] for r in owners.values())),
        distinct_contractions=len(owners), owners=len(campaigns),
        matching_historical_keys=sorted(set(owners) & set(old)),
        excluded_chunks=[r['id'] for r in catalog['chunks'] if r['status']!='ready_for_tuning'],
        limits=['No new numerical or hardware qualification',
                'Historical key intersection is not a validated trace replay',
                'Set input paths from input-root when constructing the remote study',
                'No trial counts or fresh budget allocated']))


if __name__ == '__main__':
    main()
