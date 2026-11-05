"""Focused attention build admission tests without model assets or toolchains."""
import copy
import hashlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/host_iss'), str(ROOT / 'tools/tiny_full')]
from attention_bundle import (MODEL_PIN, MODEL_SOURCE_MEMBERS, validate_fused_output, validate_gate,
                              validate_model_sources, validate_scope)


def _bind_members(data, item, row, sums, index):
    for key, member in (('initial_sha256', 'initial'), ('output_sha256', 'expected'),
                        ('final_arena_sha256', 'expected_arena'), ('program_source_sha256', 'source')):
        item[member] = f'{index}/{member}'
        sums[item[member]], row[key] = 'f'*64, 'f'*64
    item['original'] = f'{index}/program'
    sums[item['original']] = '0'*64
    item['requested_dma'] = dict(load_bytes=32, store_bytes=32)
    data[item['kind'] + '_kernels'][item['table_index']].update(
        {key: value for key, value in item.items() if key not in ('index', 'table_index')})


def fixture():
    projections, attentions, launches, plan = [], [], [], []
    for step in range(3):
        for layer in range(22):
            # This fixture exercises interleaving admission, not projection arithmetic.
            for local in range(12 if layer < 21 else 28):
                item = dict(index=len(projections), kind='projection', step=step, layer=layer)
                projections.append(item)
                launches.append(dict(item, index=len(launches), table_index=item['index']))
                plan.append(dict(launch_index=len(plan), kind='projection', index=item['index']))
            item = dict(index=len(attentions), kind='attention', step=step, layer=layer, numerics='rtl',
                        model_commit=MODEL_PIN, config=dict(query_rows=128 if step == 0 else 1,
                        keys=128 + step, query_heads=32, kv_heads=4, head_dim=64, mask='causal',
                        query_start=0 if step == 0 else 127 + step, scale=0.125))
            attentions.append(item)
            launches.append(dict(item, index=len(launches), table_index=item['index']))
            plan.append(dict(launch_index=len(plan), kind='attention', index=item['index']))
    refs = []
    for step in range(3):
        for layer in range(22):
            for field in ('keys', 'values'):
                refs.append(dict(actual_symbol=f'tiny_full_cache_{step:02d}_{layer:02d}_{field}',
                                 file=f'cache-{step}-{layer}-{field}', bytes=4*160*64*2, dtype='uint16', sha256='a'*64))
        refs.append(dict(actual_symbol=f'tiny_full_logits_{step:02d}', file=f'logits-{step}',
                         bytes=32000*4, dtype='float32', sha256='b'*64))
    for name, size in (('tiny_full_greedy_tokens', 12), ('tiny_full_decode_inputs', 8)):
        refs.append(dict(actual_symbol=name, file=name, bytes=size, dtype='uint32', sha256='c'*64))
    data = dict(schema='atlas.tiny_attention_contract.v1', workload='tiny-attention', layer_count=22,
                decode_count=2, include_lm_head=True, block_only=False, kernel_count=906, projection_calls=465,
                projection_kernel_count=840, attention_kernel_count=66, launches=launches, launch_plan=plan,
                projection_kernels=projections, attention_kernels=attentions, reference_buffers=refs,
                greedy_tokens=[278, 25269, 505])
    replay = dict(schema='atlas.v2.captured_reply.v1', launches=launches,
                  final_buffers=[dict(symbol=r['actual_symbol'], expected=r['file'], expected_sha256=r['sha256'])
                                 for r in refs])
    sums, rows = {}, []
    for index, item in enumerate(launches):
        row = dict(index=index, kind=item['kind'], status='PASS', exact=True, full_arena_exact=True,
                   accelerated=False, native_dispatch_enabled=False, model=dict(commit=MODEL_PIN, numerics='rtl'),
                   source_sha256={str(i): 'd'*64 for i in range(45)}, harness_source_sha256={'harness': 'e'*64})
        _bind_members(data, item, row, sums, index)
        row['execution'] = dict(program_sha256='0'*64, observed=item['requested_dma'])
        rows.append(row)
    gate = dict(status='PASS', model_pin=MODEL_PIN, numerics='rtl', native_not_used=True,
                all_launches_original_npu_model_rtl_exact=True, attention_native_allowed=False, launch_count=906,
                verified_indices=list(range(906)), assets_replay_sha256='1'*64, launches=rows)
    return data, replay, gate, sums


class AttentionBuildTests(unittest.TestCase):
    def setUp(self):
        self.data, self.replay, self.gate, self.sums = fixture()
        for name in ('validate_model_sources', 'validate_source_pins'):
            mocked = patch('attention_bundle.' + name)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_complete_scope(self):
        validate_scope(self.data, self.replay)

    def test_scope_rejects_schema_plan_and_missing_final(self):
        for mutation in (lambda d: d.update(schema='atlas.full_kernel_assets.v1'),
                         lambda d: d['launch_plan'][12].update(kind='projection'),
                         lambda d: d['reference_buffers'].pop()):
            data = copy.deepcopy(self.data)
            mutation(data)
            with self.assertRaises((ValueError, KeyError, IndexError)):
                validate_scope(data, self.replay)

    def test_complete_original_gate(self):
        validate_gate(self.gate, self.data, '1'*64, self.sums)

    def test_output_coverage_inventory_and_exactness(self):
        gate = copy.deepcopy(self.gate)
        gate['all_program_output_spans_written'] = True
        row = gate['launches'][0]
        gate['output_coverage'] = [dict(program_sha256=program, status='PASS', output_coverage=True,
            exact=True, full_arena_exact=True, model_sources_before=row['source_sha256'],
            model_sources_after=row['source_sha256'], harness_sources_before=row['harness_source_sha256'],
            harness_sources_after=row['harness_source_sha256'])
            for program in {item['execution']['program_sha256'] for item in gate['launches']}]
        validate_gate(gate, self.data, '1'*64, self.sums)
        gate['output_coverage'][0]['full_arena_exact'] = False
        with self.assertRaisesRegex(ValueError, 'coverage exactness'):
            validate_gate(gate, self.data, '1'*64, self.sums)
        gate['output_coverage'] = []
        with self.assertRaisesRegex(ValueError, 'unique-program output coverage'):
            validate_gate(gate, self.data, '1'*64, self.sums)

    def test_gate_rejects_partial_native_dma_or_changed_bytes(self):
        mutations = (lambda g: g['verified_indices'].pop(),
                     lambda g: g['launches'][0].update(accelerated=True),
                     lambda g: g['launches'][12].update(full_arena_exact=False),
                     lambda g: g['launches'][905]['execution'].update(observed=dict(load_bytes=64, store_bytes=32)),
                     lambda g: g['launches'][42].update(final_arena_sha256='9'*64),
                     lambda g: g.update(assets_replay_sha256='2'*64))
        for mutate in mutations:
            gate = copy.deepcopy(self.gate)
            mutate(gate)
            with self.assertRaises(ValueError):
                validate_gate(gate, self.data, '1'*64, self.sums)


def block_fixture():
    projections, attentions, launches, plan = [], [], [], []
    order = [('projection', role) for role in (0, 1, 2)] + [('attention', None)]
    order += [('projection', role) for role in (3, 4, 4, 4, 5, 5, 5, 6, 6)]
    for kind, role in order:
        table = projections if kind == 'projection' else attentions
        item = dict(index=len(table), kind=kind, step=0, layer=0)
        if kind == 'projection':
            item['logical'] = role
        else:
            item.update(numerics='rtl', model_commit=MODEL_PIN,
                        config=dict(query_rows=128, keys=128, query_heads=32, kv_heads=4,
                                    head_dim=64, mask='causal', query_start=0, scale=0.125))
        table.append(item)
        launches.append(dict(item, index=len(launches), table_index=item['index']))
        plan.append(dict(launch_index=len(plan), kind=kind, index=item['index']))
    refs = [dict(actual_symbol='tiny_full_hidden_00', file='hidden', bytes=128*2048*2, dtype='uint16', sha256='a'*64)]
    refs += [dict(actual_symbol='tiny_full_cache_00_00_'+field, file=field, bytes=4*160*64*2,
                  dtype='uint16', sha256='b'*64) for field in ('keys', 'values')]
    data = dict(schema='atlas.tiny_attention_contract.v1', workload='tiny-attention', layer_count=1,
                decode_count=0, include_lm_head=False, block_only=True, kernel_count=13, projection_calls=7,
                projection_kernel_count=12, attention_kernel_count=1, launches=launches, launch_plan=plan,
                projection_kernels=projections, attention_kernels=attentions, reference_buffers=refs, greedy_tokens=[])
    replay = dict(schema='atlas.v2.captured_reply.v1', launches=launches,
                  final_buffers=[dict(symbol=r['actual_symbol'], expected=r['file'], expected_sha256=r['sha256']) for r in refs])
    sources, harness = {str(i): 'd'*64 for i in range(45)}, {'harness': 'e'*64}
    sums, rows = {}, []
    for index, item in enumerate(launches):
        item.update(output_offset=256, output_extent_bytes=32)
        row = dict(index=index, kind=item['kind'], status='PASS', exact=True, full_arena_exact=True,
                   accelerated=False, native_dispatch_enabled=False, model_pin=MODEL_PIN, numerics='rtl',
                   model_root='/qualified/npu-model', model_sources_before=copy.deepcopy(sources),
                   model_sources_after=copy.deepcopy(sources), harness_sources_before=dict(harness),
                   harness_sources_after=dict(harness), output_offset=256, output_bytes=32,
                   method='executed_original_model')
        _bind_members(data, item, row, sums, index)
        row.update(program_sha256='0'*64, requested_dma=dict(item['requested_dma']),
                   execution=dict(program_sha256='0'*64, observed=dict(item['requested_dma'])))
        rows.append(row)
    gate = dict(schema='atlas.d1.all_launch_original.v1', status='PASS', model_pin=MODEL_PIN, numerics='rtl',
                native_not_used=True, accelerated=False, smoke_scope=True, all_launches_original_npu_model_rtl_exact=True,
                launch_count=13, verified_indices=list(range(13)), assets_replay_sha256='1'*64, launches=rows,
                model_sources_before=copy.deepcopy(sources), model_sources_after=copy.deepcopy(sources),
                harness_sources_before=dict(harness), harness_sources_after=dict(harness))
    return data, replay, gate, sums


class CompleteBlockAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.data, self.replay, self.gate, self.sums = block_fixture()
        for name in ('validate_model_sources', 'validate_source_pins'):
            mocked = patch('attention_bundle.' + name)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_complete_block_and_original_gate(self):
        self.assertEqual(validate_scope(self.data, self.replay), 'layer0-block')
        validate_gate(self.gate, self.data, '1'*64, self.sums)
        self.gate['launches'][7]['execution']['observed']['fp8_output'] = {'values': 32}
        validate_gate(self.gate, self.data, '1'*64, self.sums)

    def test_block_rejects_prefix_missing_hidden_decode_and_tokens(self):
        for mutate in (lambda d, r: r['final_buffers'].pop(),
                       lambda d, r: d.update(decode_count=1),
                       lambda d, r: d.update(layer_count=True),
                       lambda d, r: d.update(greedy_tokens=[278]),
                       lambda d, r: d['launches'][6].update(logical=5)):
            data, replay = copy.deepcopy((self.data, self.replay))
            mutate(data, replay)
            with self.assertRaises(ValueError):
                validate_scope(data, replay)

    def test_block_gate_rejects_legacy_prefix_native_and_changed_bookends(self):
        for mutate in (lambda g: g.pop('schema'),
                       lambda g: g.update(status='PASS_NATIVE_RECORD_ADMITTED_REQUIRES_ORIGINAL47_BACKFILL'),
                       lambda g: g['launches'][0]['model_sources_after'].update({'0': '9'*64}),
                       lambda g: g['launches'][0]['harness_sources_after'].update(harness='9'*64),
                       lambda g: g['launches'][0].update(output_bytes=64),
                       lambda g: g['launches'][0]['execution']['observed'].update(native_mxu0_calls=1),
                       lambda g: g['harness_sources_after'].update(harness='9'*64)):
            gate = copy.deepcopy(self.gate)
            mutate(gate)
            with self.assertRaises(ValueError):
                validate_gate(gate, self.data, '1'*64, self.sums)


class OriginalSourceClosureTests(unittest.TestCase):
    def test_every_required_module_table_and_curve_is_required(self):
        self.assertEqual(len(MODEL_SOURCE_MEMBERS), 45)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'npu-model'
            (root / '.git').mkdir(parents=True)
            sources = {str(root / name): 'a'*64 for name in MODEL_SOURCE_MEMBERS}
            with patch('attention_bundle.subprocess.check_output') as git:
                for missing in MODEL_SOURCE_MEMBERS:
                    incomplete = dict(sources)
                    del incomplete[str(root / missing)]
                    incomplete[str(root / 'README.md')] = 'b'*64
                    with self.assertRaisesRegex(ValueError, 'source or unary table closure missing'):
                        validate_model_sources(root, incomplete)
                git.assert_not_called()

    def test_gitfile_root_authenticates_head_and_blobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'npu-model'
            subprocess.run(['git', 'init', '--quiet', '--separate-git-dir', str(Path(temporary)/'metadata'),
                            str(root)], check=True)
            sources = {}
            for name in MODEL_SOURCE_MEMBERS:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'authenticated fixture')
                sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            subprocess.run(['git', '-C', str(root), 'add', 'npu_model'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test', '-c', 'user.email=test@example.com',
                            'commit', '--quiet', '-m', 'test: fixture'], check=True)
            head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            with patch('attention_bundle.MODEL_PIN', head):
                validate_model_sources(root, sources)
                next(iter(root.glob('npu_model/*.py'))).write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    validate_model_sources(root, sources)


class FusedExtentTests(unittest.TestCase):
    def fixture(self):
        offset, extent, extra = 51456, 360448, 196608
        operand = bytes([0x77]) * extra
        initial = bytes(offset - 256) + b'\xa5'*256 + b'\xcd'*extent + b'\x5a'*256 + operand
        output = bytearray(b'\xcd'*extent)
        output[131072:180224] = b'\x13'*49152
        output[311296:360448] = b'\x29'*49152
        final = initial[:offset] + output + initial[offset+extent:]
        item = dict(physical_mkn=[64, 32, 1536], logical=5, columns=[4096, 5632], width=5632,
                    fusion_role=5, output_offset=offset, output_extent_bytes=extent,
                    operand_offset=412160, total_dram_bytes=len(initial), operand='operand',
                    written_spans=[dict(offset=131072, bytes=49152), dict(offset=311296, bytes=49152)],
                    requested_dma=dict(load_bytes=247808, store_bytes=98304), output_dtype='fp8',
                    output_row_stride=180224, output_column_tile=128, output_scale=-9)
        return item, initial, bytes(output), bytes(final), lambda name: operand

    def test_strided_extent_and_dma_are_distinct(self):
        validate_fused_output(*self.fixture())

    def test_fused_extent_operand_store_stride_and_hole_corruption_rejected(self):
        for mutate in (lambda x: x.update(output_extent_bytes=98304),
                       lambda x: x.update(operand_offset=412192),
                       lambda x: x['requested_dma'].update(store_bytes=360448),
                       lambda x: x.update(output_row_stride=65536)):
            item, initial, output, final, read = self.fixture()
            mutate(item)
            with self.assertRaises(ValueError):
                validate_fused_output(item, initial, output, final, read)
        item, initial, output, final, read = self.fixture()
        output, final = bytearray(output), bytearray(final)
        output[1] ^= 1
        final[item['output_offset']+1] ^= 1
        with self.assertRaisesRegex(ValueError, 'strided hole'):
            validate_fused_output(item, initial, bytes(output), bytes(final), read)


if __name__ == '__main__':
    unittest.main()
