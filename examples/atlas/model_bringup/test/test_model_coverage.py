"""Compare authored projection and attention plans with pinned export tensor evidence.

Checks cover ordered projections, Smol input dtypes, complete Tiny projection output shapes,
Tiny cache/logit geometry, and attention batch, heads, head dimension, query/key extents,
mask tensor extents, and recorded insertion order through production planning/admission geometry.
Aggregated fixtures do not establish mask values, causal semantics, host operations, fusion
boundaries beyond the plan, scale contracts, or complete producer/consumer graph edges.
The fixtures are atlas-kernels coverage/fixture/{smolvla,tinyllama}.json. Regenerate them
with coverage/export_ops.py from the captured checkpoints and inputs, never by hand.
"""
import copy
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import kernel_library  # noqa: E402
import smol_attention_assets as smol_attention  # noqa: E402
from atlas_attention_v2_program import AttentionV2Config  # noqa: E402
from smol_full_plan import calls, reference_geometry  # noqa: E402
sys.path.insert(0, str(ROOT / 'tools/tiny_full'))
from capture_boundaries import validate_inventory  # noqa: E402
from attention_bundle import attention_before_projection, model_attention_geometry  # noqa: E402

# Use the fixtures in the pinned submodule, including after generator path moves.
FIXTURE = kernel_library.LIBRARY / 'coverage' / 'fixture'


PROJECTION = ('aten.linear.default', 'aten.conv2d.default', 'aten.conv2d.padding')
ATTENTION = ('aten.softmax.int', 'aten.scaled_dot_product_attention.default')


def module_name(row, layer):
    return row['module'].replace('layers.*', f'layers.{layer}') if layer is not None else row['module']


def load(model):
    """Fixture plus its ordered projection calls and attention cores, rebuilt from the rows' call indices."""
    inventory = json.loads((FIXTURE / f'{model}.json').read_text())
    if inventory.get('schema') != 'atlas.coverage.inventory.v2' or inventory.get('model') != model:
        raise ValueError('unexpected coverage fixture')
    calls, attention = {}, []
    for row in inventory['ops']:
        for index, layer, step in row.get('calls', []):
            entry = {'module': module_name(row, layer), 'module_pattern': row['module'],
                     'phase': row['phase'], 'step': step, 'source': row['source']}
            if row['op'] in PROJECTION:
                calls[index] = dict(entry, input=row['inputs'][0], output=row['outputs'][0])
            elif row['op'] in ATTENTION:
                kind = 'sdpa' if row['op'] == ATTENTION[1] else 'eager'
                shapes = dict(q=row['inputs'][0], k=row['inputs'][1], v=row['inputs'][2],
                              mask=row['inputs'][3], output=row['outputs'][0]) if kind == 'sdpa' else dict(scores=row['inputs'][0])
                attention.append(dict(entry, kind=kind, calls_before=index, **shapes))
    if sorted(calls) != list(range(len(calls))):
        raise ValueError('fixture projection call indices are not contiguous')
    inventory['calls'] = [calls[index] for index in range(len(calls))]
    inventory['attention'] = sorted(attention, key=lambda row: row['calls_before'])
    return inventory


def smol_plan_geometry():
    """smol_full_plan.calls() in the export's terms, with validate_reference's input and output shapes."""
    result = []
    for call in calls():
        rows, k, n = call['rows'], call['input_width'], call['output_width']
        shapes = ([1, k], [1, n]) if call['site'] == 0 else ([1, rows, k], [1, rows, n])
        if call['site'] == 7 and call['role'] == 0:
            shapes = [1, 3, 512, 512], [1, 768, 32, 32]
        dtype = 'bfloat16' if call['input_dtype'] == 0 else 'float32'
        result.append((call['module'], shapes[0], shapes[1], dtype, call.get('step')))
    return result


def unique_shape(shapes):
    shapes = {tuple(shape) for shape in shapes}
    if len(shapes) != 1:
        raise ValueError('missing or ambiguous aggregated tensor geometry')
    return list(shapes.pop())


def attention_geometry(inventory):
    """Tensor extents only. Related aggregate rows supply no new SSA or mask-value claim."""
    result = []
    for row in inventory['attention']:
        before = row['calls_before']
        if row['kind'] == 'sdpa':
            q, k, v, mask = [row[key][0] for key in ('q', 'k', 'v', 'mask')]
            if row['output'][0] != q or k != v:
                raise ValueError('SDPA output or KV shape differs')
        else:
            batch, heads, rows, keys = row['scores'][0]
            projections = inventory['calls'][before-3:before]
            if [p['module'].rsplit('.', 1)[-1] for p in projections] != ['q_proj', 'k_proj', 'v_proj']:
                raise ValueError('attention is not recorded after Q/K/V projections')
            related = [op for op in inventory['ops'] if op['phase'] == row['phase']
                       and op['module'] == row['module_pattern'] and row['step'] in op['steps']]

            def view(projection):
                return unique_shape(op['outputs'][0][0] for op in related
                                    if op['op'] == 'aten.view.default' and len(op['outputs'][0][0]) == 4
                                    and op['inputs'][0][0] == projection['output'][0]
                                    and (inventory['model'] == 'tinyllama' or op['source'] == row['source']))

            qview, kview, vview = map(view, projections)
            if (qview[:2] != [batch, rows] or qview[2] != heads or kview != vview
                    or kview[0] != batch or qview[3] != kview[3]):
                raise ValueError('eager scores and projected head geometry differ')
            q = [batch, heads, rows, qview[3]]
            k, v = [batch, kview[2], keys, kview[3]], [batch, vview[2], keys, vview[3]]
            mask_op, mask_input = (('aten.add.Tensor', 1) if inventory['model'] == 'tinyllama'
                                   else ('aten.where.ScalarOther', 0))
            mask = unique_shape(op['inputs'][mask_input][0] for op in related
                                if op['op'] == mask_op and op['source'] == row['source']
                                and op['outputs'][0][0] == row['scores'][0])
        result.append(dict(q=q, k=k, v=v, mask=mask, before_call=before, step=row['step'],
                           module=row['module']))
    return result


def check_attention_shapes(row, config, batch):
    # Retain the batch admitted by the production reference/cache geometry.
    q = [batch, config.query_heads, config.query_rows, config.head_dim]
    k = [batch, config.kv_heads, config.keys, config.head_dim]
    if row['q'] != q or row['k'] != k or row['v'] != k:
        raise ValueError('export and production attention tensor geometry differ')
    if row['mask'] != [q[0], 1, config.query_rows, config.keys]:
        raise ValueError('export and production attention mask extent differ')
    # Only the shape of a logical Boolean mask is available from the export.
    smol_attention.mask_tiles(config, np.zeros(row['mask'][2:], dtype=bool))


def validate_smol_attention(inventory):
    derived = attention_geometry(inventory)
    planned = [event for event in smol_attention.event_plan() if event['kind'] == 'attention']
    if [row['before_call'] for row in derived] != [event['before_call'] for event in planned]:
        raise ValueError('Smol attention insertion order differs')
    graph = calls()
    batch = reference_geometry()['prefix_key_cache'][0][1]
    for phase, row in enumerate(derived):
        # Vision records retain their module identity; eager Smol records have none.
        call = graph[row['before_call']]
        if (row['step'] != call.get('step', 0)
                or row['module'] and row['module'] != call['module'].rsplit('.', 1)[0]):
            raise ValueError('Smol attention module order differs')
        check_attention_shapes(row, AttentionV2Config(*smol_attention.attention_geometry(phase)), batch)


def validate_tiny_attention(inventory):
    # The inventory admission proves the site indices used by the actual exporter.
    capture = validate_tiny_inventory(inventory)
    planned = [index for index, call in enumerate(capture['calls'])
               if attention_before_projection(call['site_index'])]
    derived = attention_geometry(inventory)
    if [row['before_call'] for row in derived] != planned:
        raise ValueError('Tiny attention insertion order differs')
    for row in derived:
        call = capture['calls'][row['before_call']]
        if row['step'] != call['step'] or row['module'] != call['site'].rsplit('.', 1)[0]:
            raise ValueError('Tiny attention module/step order differs')
        batch = capture['target_trace'][row['step']]['cache'][0]['shape'][0]
        check_attention_shapes(row, AttentionV2Config(**model_attention_geometry(row['step'])), batch)


def tiny_capture_view(inventory):
    """The capture fields validate_inventory reads, filled from the export.

    Logical fields (step, site order, module, rows, K, output width, KV cache shapes, logits width) come from the
    export. Physical panels are the plan's column split and M32 decode padding, the part under test elsewhere.
    """
    data, steps = {'calls': []}, {}
    for row in inventory['calls']:
        step, (shape, _), width = row['step'], row['input'], row['output'][0][-1]
        site = steps.get(step, 0)
        steps[step] = site + 1
        rows, k = math.prod(shape[:-1]), shape[-1]
        m, panels, column = (128 if rows == 128 else 32), [], 0
        while column < width:
            n = min(1024 if k == 5632 else 2048, width - column)
            panels.append({'columns': [column, column + n], 'logical_mkn': [rows, k, n], 'physical_mkn': [m, k, n],
                           'expected_bf16': {'shape': [m, n], 'dtype': 'uint16', 'bytes': m * n * 2}})
            column += n
        data['calls'].append({'step': step, 'site_index': site, 'site': row['module'],
                              'A_bf16': {'shape': [rows, k], 'dtype': 'uint16', 'bytes': rows * k * 2},
                              'panels': panels})
    data['target_trace'] = [{'step': step, 'cache': [], 'logits_payload': None} for step in sorted(steps)]
    for row in inventory['ops']:
        if row['module'] == 'model.layers.*.self_attn' and row['source'].endswith(':270') and row['op'] == 'aten.cat.default':
            if row['count'] != 2 * len(row['layers']) * len(row['steps']):
                raise ValueError('KV cache appends are not one key and one value per layer')
            for step in row['steps']:
                data['target_trace'][step]['cache'] += [
                    {'layer': layer, 'field': field, 'shape': row['outputs'][0][0], 'dtype': 'uint16',
                     'bytes': math.prod(row['outputs'][0][0]) * 2} for layer in row['layers'] for field in ('keys', 'values')]
    for row in inventory['calls']:
        if row['module'] == 'lm_head':
            width = row['output'][0][-1]
            data['target_trace'][row['step']]['logits_payload'] = {'shape': [width], 'dtype': 'float32', 'bytes': 4 * width}
    return data


def validate_tiny_inventory(inventory):
    data = tiny_capture_view(inventory)
    validate_inventory(data)
    # Admission fixes logical rows, widths, and batch through the cache geometry.
    # Compare every exported dimension, including lm_head, before dropping the batch.
    for exported, admitted in zip(inventory['calls'], data['calls']):
        batch = data['target_trace'][admitted['step']]['cache'][0]['shape'][0]
        rows, k = admitted['A_bf16']['shape']
        width = admitted['panels'][-1]['columns'][1]
        if exported['input'][0] != [batch, rows, k] or exported['output'][0] != [batch, rows, width]:
            raise ValueError('Tiny complete logical projection output shape differs')
    return data


class SmolCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = load('smolvla')

    def test_capture_crosscheck_passed(self):
        check = self.inventory['capture_crosscheck']
        self.assertEqual((check['status'], check['exported_calls'], check['captured_calls']), ('PASS', 1421, 1421))

    def test_plan_calls_equal_export(self):
        export = [(row['module'], row['input'][0], row['output'][0], row['input'][1],
                   row['step'] if row['phase'] == 'denoise' else None) for row in self.inventory['calls']]
        self.assertEqual(smol_plan_geometry(), export)

    def test_plan_attention_equals_export(self):
        validate_smol_attention(self.inventory)

    def test_production_head_mutation_is_detected(self):
        original = smol_attention.attention_geometry

        def changed(phase):
            rows, keys, hq, hkv = original(phase)
            return (rows, keys, 6, 6) if phase < 24 else (rows, keys, hq, hkv)

        with patch.object(smol_attention, 'attention_geometry', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'tensor geometry'):
                validate_smol_attention(self.inventory)

    def test_sdpa_head_dimension_mutation_is_detected(self):
        changed = copy.deepcopy(self.inventory)
        for row in changed['attention']:
            if row['kind'] == 'sdpa':
                for field in ('q', 'k', 'v', 'output'):
                    row[field][0][-1] = 32
        with self.assertRaises(ValueError):
            validate_smol_attention(changed)

    def test_total_preserving_attention_reorder_is_detected(self):
        changed = copy.deepcopy(self.inventory)
        first, second = changed['attention'][:2]
        first['calls_before'], second['calls_before'] = second['calls_before'], first['calls_before']
        changed['attention'].sort(key=lambda row: row['calls_before'])
        self.assertEqual(len(changed['attention']), len(self.inventory['attention']))
        self.assertEqual(sum(row['calls_before'] for row in changed['attention']),
                         sum(row['calls_before'] for row in self.inventory['attention']))
        self.assertCountEqual([row['q'] for row in changed['attention'] if row['kind'] == 'sdpa'],
                              [row['q'] for row in self.inventory['attention'] if row['kind'] == 'sdpa'])
        with self.assertRaisesRegex(ValueError, 'module order'):
            validate_smol_attention(changed)

    def test_attention_tensor_extent_mutations_are_detected(self):
        for field, axis in (('q', 0), ('q', 1), ('q', 2), ('k', 1), ('k', 2), ('mask', 2), ('mask', 3)):
            with self.subTest(field=field, axis=axis):
                changed = copy.deepcopy(self.inventory)
                changed['attention'][0][field][0][axis] += 1
                with self.assertRaises(ValueError):
                    validate_smol_attention(changed)

    def test_mutated_export_is_detected(self):
        export = [(row['module'], row['input'][0], row['output'][0], row['input'][1],
                   row['step'] if row['phase'] == 'denoise' else None) for row in self.inventory['calls']]
        for index, field, value in ((5, 0, 'renamed'), (300, 1, [1, 177, 961]), (1420, 3, 'bfloat16')):
            changed = copy.deepcopy(export)
            changed[index] = changed[index][:field] + (value,) + changed[index][field + 1:]
            self.assertNotEqual(smol_plan_geometry(), changed)
        self.assertNotEqual(smol_plan_geometry(), export[:-1])


class TinyCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = load('tinyllama')

    def test_capture_crosscheck_passed(self):
        check = self.inventory['capture_crosscheck']
        self.assertEqual((check['status'], check['exported_calls'], check['captured_calls']), ('PASS', 465, 465))

    def test_validate_inventory_admits_export(self):
        validate_tiny_inventory(self.inventory)

    def test_attention_geometry(self):
        validate_tiny_attention(self.inventory)

    def test_attention_tensor_extent_mutations_are_rejected(self):
        for axis in range(4):
            with self.subTest(score_axis=axis):
                changed = copy.deepcopy(self.inventory)
                changed['attention'][0]['scores'][0][axis] += 1
                with self.assertRaises(ValueError):
                    validate_tiny_attention(changed)
        changed = copy.deepcopy(self.inventory)
        first, second = changed['attention'][:2]
        first['calls_before'], second['calls_before'] = second['calls_before'], first['calls_before']
        changed['attention'].sort(key=lambda row: row['calls_before'])
        with self.assertRaisesRegex(ValueError, 'module/step order'):
            validate_tiny_attention(changed)

    def test_complete_projection_output_mutation_is_rejected(self):
        changed = copy.deepcopy(self.inventory)
        for row in changed['calls']:
            row['output'][0][:-1] = [99, 99]
        with self.assertRaisesRegex(ValueError, 'complete logical projection output'):
            validate_tiny_inventory(changed)

    def test_mutated_export_is_rejected(self):
        def mutate(change):
            inventory = copy.deepcopy(self.inventory)
            change(inventory)
            with self.assertRaises(ValueError):
                validate_tiny_inventory(inventory)

        mutate(lambda d: d['calls'][3].update(module='model.layers.0.self_attn.o_projection'))
        mutate(lambda d: d['calls'][6]['input'][0].__setitem__(-1, 2048))
        mutate(lambda d: d['calls'][200]['output'][0].__setitem__(-1, 2560))
        mutate(lambda d: d['calls'].pop(154))
        mutate(lambda d: next(row for row in d['ops'] if row['source'].endswith(':270'))['outputs'][0][0].__setitem__(2, 160))


if __name__ == '__main__':
    unittest.main()
