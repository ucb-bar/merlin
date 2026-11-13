"""Exact mutable status and readonly exponent publication for Atlas examples."""
import copy
import unittest
from model_runtime import output_ranges, active_dram_prefix, check_immutable_memory

BASE = 0x90000000


def plan():
    return dict(schema='atlas.quantized_loop_execution.v1',
                input_addresses=[BASE, BASE + 2048], input_buffer_bytes=[2048, 2048],
                exponent_addresses=[BASE + 4096, BASE + 6144], exponent_buffer_bytes=[2048, 2048],
                output_addresses=[BASE + 8192], output_bytes=2048,
                status_address=BASE + 12288, status_bytes=2048)


def exact_float_plan(operation="bf16_to_f32"):
    # Literal ABI sizes from the accepted R3 layer0 images; include all unary
    # and binary families without relying on the target provider's builder.
    logical, sizes, result = {
        "bf16_to_f32": (8192, [4096], 8192),
        "f32_to_bf16": (4096, [8192], 4096),
        "bf16_neg": (4096, [4096], 4096),
        "f32_neg": (8192, [8192], 8192),
        "bf16_add": (2048, [2048, 2048], 2048),
        "bf16_sub": (2048, [2048, 2048], 2048),
    }[operation]
    output = BASE + sum(sizes)
    return dict(schema="atlas.exact_float.v1", operation=operation,
                input_addresses=[BASE + sum(sizes[:i]) for i in range(len(sizes))],
                input_payload_bytes=sizes, output_addresses=[output],
                logical_output_bytes=logical, result_buffer_bytes=result,
                output_bytes=result + 32, status_address=output + result,
                workspace_bytes=sum(sizes) + result + 32)


def quantizer_plan(dynamic=False):
    count, group = (32768, 4096) if dynamic else (131072, 131072)
    return dict(schema="atlas.activation_quantize_vector_plan.v1", mode="dynamic" if dynamic else "static",
                count=count, input_dtype="bf16", group_elements=group, head_elements=count,
                row_elements=group, input_address=BASE, input_bytes=2 * count,
                input_addresses=[BASE], input_payload_bytes=[2 * count],
                code_address=BASE + 2 * count, code_bytes=count,
                exponent_address=BASE + 3 * count, exponent_bytes=32,
                output_addresses=[BASE + 2 * count], output_bytes=count + 32,
                workspace_bytes=3 * count + 32)


class MemoryFootprintTests(unittest.TestCase):
    def test_row_loop_versions_keep_identical_mutable_and_readonly_windows(self):
        p = dict(schema='atlas.row_loop_execution_plan.v1', groups=3,
                 readonly_base=BASE, readonly_stride_bytes=2048,
                 output_base=BASE + 6144, output_stride_bytes=4096,
                 scratch_address=BASE + 8192, scratch_stride_bytes=4096,
                 scratch_bytes=2048)
        expected = ((6144, 18432), (8192, 18432))
        for version in (1, 2):
            with self.subTest(version=version):
                record = dict(p, schema=f'atlas.row_loop_execution_plan.v{version}')
                if version == 2:
                    record['dispatch_instruction_budget'] = 128000000
                ranges = output_ranges(record, BASE, 20480)
                self.assertEqual(ranges, expected)
                self.assertEqual(active_dram_prefix(record, BASE, 20480), 18432)
                before = bytes(20480)
                after = bytearray(before)
                after[6144:18432] = bytes([1]) * 12288
                check_immutable_memory(before, after, ranges)
                for index in (0, 6143, 18432, 20479):
                    broken = bytearray(after)
                    broken[index] ^= 1
                    with self.subTest(index=index), self.assertRaises(AssertionError):
                        check_immutable_memory(before, broken, ranges)
                with self.assertRaises(AssertionError):
                    output_ranges(record, BASE, 18431)
        for budget in (None, True, 0, -1, 1.5, '128000000'):
            with self.subTest(budget=budget), self.assertRaises(AssertionError):
                output_ranges(dict(p, schema='atlas.row_loop_execution_plan.v2',
                                   dispatch_instruction_budget=budget), BASE, 20480)
        with self.assertRaises(AssertionError):
            output_ranges(dict(p, schema='atlas.row_loop_execution_plan.v3'), BASE, 20480)

    def test_exact_status_write_with_guards_and_readonly_exponents(self):
        ranges = output_ranges(plan(), BASE, 16384)
        self.assertEqual(ranges, ((8192, 10240), (12288, 14336)))
        before = bytes([0xa5]) * 16384
        after = bytearray(before)
        after[8192:10240] = bytes(2048)
        after[12288:14336] = bytes(2048)
        check_immutable_memory(before, after, ranges)
        for index in (0, 2048, 4096, 6144, 10240, 12287, 14336, 16383):
            broken = bytearray(after)
            broken[index] ^= 1
            with self.subTest(index=index), self.assertRaises(AssertionError):
                check_immutable_memory(before, broken, ranges)

    def assert_act_writes_only_its_complete_output(self, p):
        memory = p['workspace_bytes'] + 64
        start = p['output_addresses'][0] - BASE
        end = start + p['output_bytes']
        ranges = output_ranges(p, BASE, memory)
        self.assertEqual(ranges, ((start, end),))
        self.assertEqual(active_dram_prefix(p, BASE, memory), end)
        before = bytes([0xa5]) * memory
        after = bytearray(before)
        after[start:end] = bytes(p['output_bytes'])
        check_immutable_memory(before, after, ranges)
        for address in (0, start - 1, end, memory - 1):
            broken = bytearray(after)
            broken[address] ^= 1
            with self.subTest(address=address), self.assertRaises(AssertionError):
                check_immutable_memory(before, broken, ranges)

    def test_exact_float_combined_status_and_all_operations(self):
        for operation in ('bf16_to_f32', 'f32_to_bf16', 'bf16_neg', 'f32_neg', 'bf16_add', 'bf16_sub'):
            with self.subTest(operation=operation):
                self.assert_act_writes_only_its_complete_output(exact_float_plan(operation))
        p = exact_float_plan('bf16_add')
        p.update(input_addresses=[BASE], input_payload_bytes=[2048], output_addresses=[BASE + 2048],
                 status_address=BASE + 4096, workspace_bytes=4128)
        self.assert_act_writes_only_its_complete_output(p)
        # A ragged logical conversion still owns its exact padded result/tail.
        p = dict(exact_float_plan('f32_to_bf16'), logical_output_bytes=66,
                 input_payload_bytes=[256], output_addresses=[BASE + 256],
                 result_buffer_bytes=128, output_bytes=160, status_address=BASE + 384, workspace_bytes=416)
        self.assert_act_writes_only_its_complete_output(p)

    def test_exact_float_rejects_status_size_alias_and_geometry_mutations(self):
        p = exact_float_plan()
        changes = [dict(status_address=p['status_address'] + 1), dict(status_address=p['status_address'] - 32),
                   dict(status_bytes=32), dict(status_bytes=2048), dict(output_bytes=p['output_bytes'] + 32),
                   dict(result_buffer_bytes=True), dict(result_buffer_bytes=4096), dict(workspace_bytes=p['workspace_bytes'] + 32),
                   dict(input_addresses=[p['output_addresses'][0]]), dict(input_addresses=[BASE + 1]),
                   dict(input_payload_bytes=[]), dict(input_payload_bytes=[True]), dict(input_payload_bytes=[4097]),
                   dict(input_addresses=[], input_payload_bytes=[]), dict(output_addresses=[BASE + 1]),
                   dict(exponent_addresses=[BASE]), dict(exponent_buffer_bytes=[2048]),
                   dict(operation='unknown'), dict(logical_output_bytes=8193), dict(schema='unknown')]
        for change in changes:
            broken = dict(p, **change)
            with self.subTest(change=change), self.assertRaises(AssertionError):
                output_ranges(broken, BASE, 32768)
        for key in ('status_address', 'result_buffer_bytes', 'output_bytes', 'workspace_bytes', 'input_payload_bytes'):
            broken = dict(p); del broken[key]
            with self.subTest(missing=key), self.assertRaises(AssertionError):
                active_dram_prefix(broken, BASE, 32768)
        with self.assertRaises(AssertionError):
            output_ranges(p, BASE, p['workspace_bytes'] - 1)

    def test_quantizer_combined_codes_and_exponents_remain_mutable(self):
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                self.assert_act_writes_only_its_complete_output(quantizer_plan(dynamic))
        # The same ABI permits complete padded per-row and per-head exponents.
        p = quantizer_plan(True)
        p.update(row_elements=64, group_elements=4096, exponent_bytes=2048,
                 output_bytes=p['code_bytes'] + 2048, workspace_bytes=3 * p['count'] + 2048)
        self.assert_act_writes_only_its_complete_output(p)

    def test_quantizer_rejects_inconsistent_or_overlapping_publications(self):
        for dynamic in (False, True):
            p = quantizer_plan(dynamic)
            changes = [dict(status_address=p['exponent_address']), dict(status_bytes=32),
                       dict(exponent_addresses=[BASE]), dict(exponent_buffer_bytes=[2048]),
                       dict(input_address=BASE + 32), dict(input_payload_bytes=[]),
                       dict(input_addresses=[p['code_address']]), dict(input_payload_bytes=[p['input_bytes'] - 32]),
                       dict(code_address=BASE), dict(code_bytes=p['code_bytes'] + 32),
                       dict(exponent_address=p['code_address']), dict(exponent_bytes=0), dict(exponent_bytes=True),
                       dict(exponent_bytes=64, output_bytes=p['code_bytes'] + 64, workspace_bytes=3 * p['count'] + 64),
                       dict(count=True), dict(group_elements=0), dict(head_elements=p['count'] - 1),
                       dict(row_elements=True), dict(input_dtype='i8'), dict(mode='unknown'),
                       dict(workspace_bytes=p['workspace_bytes'] - 32)]
            for change in changes:
                with self.subTest(dynamic=dynamic, change=change), self.assertRaises(AssertionError):
                    output_ranges(dict(p, **change), BASE, p['workspace_bytes'] + 4096)

    def test_undeclared_status_is_immutable(self):
        p = plan()
        del p['status_address'], p['status_bytes']
        ranges = output_ranges(p, BASE, 16384)
        self.assertEqual(ranges, ((8192, 10240),))
        after = bytearray(16384)
        after[12288] = 1
        with self.assertRaises(AssertionError):
            check_immutable_memory(bytes(16384), after, ranges)

    def test_compact_prefix_includes_every_exponent_page(self):
        p = plan()
        del p['status_address'], p['status_bytes']
        p['exponent_addresses'][1] = BASE + 14336
        self.assertEqual(active_dram_prefix(p, BASE, 16384), 16384)
        self.assertEqual(active_dram_prefix(plan(), BASE, 16384), 14336)

    def test_third_epilogue_input_stays_readonly_and_bounds_compact_prefix(self):
        p = plan()
        p['input_addresses'].append(p['status_address'] + p['status_bytes'])
        p['input_buffer_bytes'].append(p['output_bytes'])
        ranges = output_ranges(p, BASE, 16384)
        self.assertEqual(ranges, ((8192, 10240), (12288, 14336)))
        self.assertEqual(active_dram_prefix(p, BASE, 16384), 16384)
        before = bytes(16384)
        after = bytearray(before)
        after[8192:10240] = bytes([1]) * 2048
        after[12288:14336] = bytes([1]) * 2048
        check_immutable_memory(before, after, ranges)
        for index in (14336, 16383):
            broken = bytearray(after)
            broken[index] = 1
            with self.subTest(index=index), self.assertRaises(AssertionError):
                check_immutable_memory(before, broken, ranges)
        for address, size in ((BASE + 14336, 0), (BASE + 14336, 2049),
                              (BASE + 14336, 4096), (BASE + 12288, 2048),
                              (BASE + 8192, 2048), (BASE + 4096, 2048)):
            broken = copy.deepcopy(p)
            broken['input_addresses'][2] = address
            broken['input_buffer_bytes'][2] = size
            with self.subTest(address=address, size=size), self.assertRaises(AssertionError):
                output_ranges(broken, BASE, 16384)

    def test_invalid_footprints_reject(self):
        changes = [dict(status_bytes=32), dict(status_address=BASE + 15360),
                   dict(status_address=BASE + 8192), dict(status_address=BASE + 4096),
                   dict(status_address=BASE + 2048), dict(status_address=BASE + 12289),
                   dict(exponent_addresses=[BASE + 4096]),
                   dict(exponent_buffer_bytes=[2048]),
                   dict(exponent_buffer_bytes=[0, 2048]),
                   dict(exponent_addresses=[BASE + 4096, BASE + 4096]),
                   dict(exponent_addresses=[BASE, BASE + 6144]),
                   dict(exponent_addresses=[BASE + 8192, BASE + 6144]),
                   dict(exponent_addresses=[BASE + 16384, BASE + 6144]),
                   dict(exponent_buffer_bytes=[True, 2048])]
        for change in changes:
            p = plan()
            p.update(copy.deepcopy(change))
            with self.subTest(change=change), self.assertRaises(AssertionError):
                output_ranges(p, BASE, 16384)
        for key in ('status_address', 'status_bytes', 'exponent_addresses', 'exponent_buffer_bytes'):
            p = plan()
            del p[key]
            with self.subTest(missing=key), self.assertRaises(AssertionError):
                output_ranges(p, BASE, 16384)


if __name__ == '__main__':
    unittest.main()
