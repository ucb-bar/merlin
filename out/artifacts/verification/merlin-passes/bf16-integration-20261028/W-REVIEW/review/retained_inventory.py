"""Independently census historical exported IR as data; never replay or promote a receipt."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
OLD = Path('/home/jeremy/Projects/merlin/out/artifacts/verification/merlin-passes/frontend/bf16-f32acc')


def load_bound(ref):
    path = Path(ref['path'])
    assert path.suffix in ('.json', '.mlir'), 'Only source/data reads permitted'
    data = path.read_bytes()
    assert len(data) == ref['bytes'] and hashlib.sha256(data).hexdigest() == ref['sha256'], path
    return json.loads(data) if path.suffix == '.json' else None


def body(op):
    return op['regions'][0]['blocks'][0]['ops']


def candidate(op):
    results = op['results']
    if len(results) != 1: return False
    t = results[0]['type']
    if t.get('kind') != 'tensor' or any(d is None for d in t['shape']): return False
    if t.get('element', {}).get('format') not in ('bf16', 'f16'): return False
    if op['name'] == 'linalg.matmul': return True
    attrs = op['attrs']
    if op['name'] != 'linalg.generic' or attrs.get('operandSegmentSizes', {}).get('values', [])[:1] != ['2']:
        return False
    if not any(i.get('value') == 'reduction' for i in attrs.get('iterator_types', {}).get('elements', [])):
        return False
    return any({'arith.mulf', 'arith.addf'} <= {i['name'] for i in block['ops']}
               for reg in op['regions'] for block in reg['blocks'])


def recursive_sites(ops, prefix=()):
    found = []
    for index, op in enumerate(ops):
        here = prefix + (index,)
        if candidate(op): found.append(here)
        for ri, region in enumerate(op['regions']):
            for bi, block in enumerate(region['blocks']):
                found.extend(recursive_sites(block['ops'], here + ('region', ri, 'block', bi)))
    return found


def main():
    all_records = []
    for name in ('SY_model_smolvla', 'SY_model_smolvla-segment', 'SY_model_tiny_llama'):
        receipt_path = OLD/name/'receipt.json'
        receipt = json.loads(receipt_path.read_text())
        source = load_bound(receipt['exports']['before'])
        final = load_bound(receipt['exports']['after'])
        for endpoint in ('before', 'after'): load_bound(receipt[endpoint])
        source_ops, final_ops = source['module']['ops'], final['module']['ops']
        assert len(source_ops) == len(final_ops)
        rows, sample = [], None
        for source_fn, final_fn in zip(source_ops, final_ops):
            if source_fn['name'] != 'func.func' or not source_fn['regions']: continue
            if not source_fn['regions'][0]['blocks']: continue
            name_attr = source_fn['attrs']['sym_name']['value']
            before, after = body(source_fn), body(final_fn)
            cursor = 0
            types = {v['id']: v['type'] for v in source_fn['regions'][0]['blocks'][0]['args']}
            owners = {}
            for index, op in enumerate(before):
                if candidate(op):
                    replacement = after[cursor:cursor+6]
                    assert [o['name'] for o in replacement] == ['tensor.empty','arith.constant','linalg.fill',
                                                               'linalg.generic','tensor.empty','linalg.generic']
                    destination = owners[op['operands'][2]]
                    literal = owners[destination['operands'][0]]
                    attrs = op['attrs']
                    row = {'function':name_attr,'source_operation':index,
                           'source_results':[v['id'] for v in op['results']],
                           'operand_types':[types[i] for i in op['operands']],
                           'maps':attrs.get('indexing_maps'),'iterators':attrs.get('iterator_types'),
                           'initialization':destination['name'],'zero_literal':literal['attrs']['value'],
                           'target_start':cursor, 'source_body_order':[o['operands'] for o in body(op)],
                           'target_scalar_ops':[o['name'] for o in body(replacement[3])],
                           'target_flags':[o['attrs'] for o in body(replacement[3])[2:4]]}
                    rows.append(row)
                    if sample is None:
                        sample={'kind':'actual historical IR JSON fragment; not a new compiler capture',
                                'receipt_path':str(receipt_path), 'source_raw':receipt['before'], 'final_raw':receipt['after'],
                                'source_export':receipt['exports']['before'], 'final_export':receipt['exports']['after'],
                                'row':row,'source_initializer':[literal,destination],'source_op':op,'target_ops':replacement}
                    cursor += 6
                else: cursor += 1
                for value in op['results']:
                    types[value['id']]=value['type']; owners[value['id']]=op
            assert cursor == len(after)
        old_report = load_bound(receipt['cert'])
        old_boundary = old_report.get('policy_boundary',old_report)['pass_boundary']
        roster = [{k:r[k] for k in ('function','source_operation','source_results')} for r in rows]
        old_roster = [{k:r[k] for k in ('function','source_operation','source_results')}
                      for r in old_boundary['site_obligations']]
        all_records.append({'name':name,'source_raw':receipt['before'],'final_raw':receipt['after'],
                            'source_export':receipt['exports']['before'], 'final_export':receipt['exports']['after'],
                            'top_level_sites':len(rows),'recursive_candidate_paths':recursive_sites(source_ops),
                            'historical_report_roster_matches_independent_census':roster == old_roster,
                            'sites':rows,'native_replay':False,'fresh_qualification':False})
        if name=='SY_model_smolvla':
            (OUT/'ACTUAL_SITE_0.json').write_text(json.dumps(sample,indent=2)+'\n')
    assert [r['top_level_sites'] for r in all_records] == [24,24,0]
    assert all(r['historical_report_roster_matches_independent_census'] for r in all_records)
    assert all_records[0]['source_export']['sha256']==all_records[1]['source_export']['sha256']
    (OUT/'RETAINED_IR_INVENTORY.json').write_text(json.dumps({'kind':'historical source/data census, not checker execution',
        'distinct_positive_source_boundaries':1,'distinct_sites':24,'records':all_records},indent=2)+'\n')
    print('Historical data census: 24/24/0; one distinct positive boundary, all exact roster comparisons match.')


if __name__ == '__main__': main()
