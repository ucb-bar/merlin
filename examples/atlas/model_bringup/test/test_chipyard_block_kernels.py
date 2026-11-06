"""Admission and receipt controls for independent raw block SoC kernels."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools/chipyard'))
import block_kernels as kernels
import kernel_library  # noqa: E402,F401
from atlas_tooling import Admission, sha
from full_bundle_fixture import full_bundle_fixture


class BlockKernelsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='atlas-block-soc-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.program = struct.pack('<2I', 0x13, 0x73)
        self.initial = b'guard' + b'123' + b'input' + b'45' + b'tail'
        self.expected = b'guard' + b'abc' + b'input' + b'de' + b'tail'
        self.row = dict(id='tiny-00', dram_base=0x100000000,
                        output_ranges=[dict(offset=5, size=3), dict(offset=13, size=2)])
        for key, raw in [('program', self.program), ('initial_dram', self.initial), ('expected_dram', self.expected)]:
            path = self.root / (key + '.bin'); path.write_bytes(raw)
            self.row[key + '_bin'] = str(path)
            self.row[key + '_sha256'] = sha(raw)
        self.manifest = self.root / 'manifest.json'
        self.write_manifest([self.row])
        self.row = kernels.read_manifest(self.manifest)[0]
        self.compiler = self.root / 'cc'; self.compiler.write_bytes(b'fixture compiler')

    def write_manifest(self, rows):
        self.manifest.write_text(json.dumps(dict(schema=kernels.SCHEMA, programs=rows)))

    def changed_asset(self, key, raw):
        Path(self.row[key + '_bin']).write_bytes(raw)
        self.row[key + '_sha256'] = sha(raw)

    def test_multi_output_and_preserved_regions(self):
        a = Admission()
        program, runtime, initial, expected, base, ranges = kernels.admit(self.row, a)
        self.assertEqual(program, self.program)
        self.assertEqual(runtime[:-24], program[:-4])
        self.assertEqual(len(runtime), len(program) + 20)
        self.assertEqual((initial, expected, base, ranges),
                         (self.initial, self.expected, 0x100000000, [(5, 3), (13, 2)]))
        self.assertTrue(a.recheck())

    def test_manifest_requires_hashes_and_unique_safe_names(self):
        for change in [dict(id='../bad'), dict(id='aVLSIb'), dict(program_sha256='0'), dict(id='')]:
            row = dict(self.row, **change); self.write_manifest([row])
            with self.subTest(change=change), self.assertRaises(ValueError): kernels.read_manifest(self.manifest)
        self.write_manifest([self.row, self.row])
        with self.assertRaises(ValueError): kernels.read_manifest(self.manifest)

    def test_hash_and_manifest_mutation_refused(self):
        Path(self.row['program_bin']).write_bytes(b'bad')
        with self.assertRaises(ValueError): kernels.admit(self.row, Admission())
        self.write_manifest([])
        with self.assertRaises(ValueError): kernels.admit(self.row, Admission())

    def test_restricted_and_symlink_assets_refused_before_read(self):
        row = dict(self.row, program_bin='/tmp/HAMMER-private/program.bin')
        with mock.patch.object(Admission, 'read', wraps=Admission().read) as read:
            with self.assertRaises(ValueError): kernels.admit(row, Admission())
            self.assertEqual(read.call_count, 1)  # Manifest is admitted first.
        alias = self.root / 'alias'; alias.symlink_to(Path(self.row['program_bin']))
        with self.assertRaises(ValueError): kernels.admit(dict(self.row, program_bin=str(alias)), Admission())

    def test_output_extents_and_input_preservation_refused(self):
        for ranges in [[], [dict(offset=-1, size=1)], [dict(offset=5, size=0)],
                       [dict(offset=5, size=4), dict(offset=8, size=1)], [dict(offset=100, size=1)],
                       [dict(offset=True, size=2)]]:
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                kernels.admit(dict(self.row, output_ranges=ranges), Admission())
        self.changed_asset('expected_dram', b'X' + self.expected[1:])
        with self.assertRaises(ValueError): kernels.admit(self.row, Admission())

    def test_raw_completion_and_reserved_handshake_refused(self):
        for words in [(0x73, 0x73), (0x13,), (0xc1009073, 0x73), (0xc1109073, 0x73), (0xc0109073, 0x73)]:
            self.changed_asset('program', struct.pack('<' + 'I' * len(words), *words))
            with self.subTest(words=words), self.assertRaises(ValueError): kernels.admit(self.row, Admission())

    def test_build_failure_and_source_bookends(self):
        def compiler(command, **kwargs):
            Path(command[-1]).write_bytes(b'synthetic ELF')
            self.compiler.write_bytes(b'changed')
            return SimpleNamespace(returncode=0)
        with mock.patch.object(kernels.subprocess, 'run', side_effect=compiler), mock.patch.object(kernels, 'audit_loadmem'):
            result = kernels.build_kernel(self.row, self.root / 'out', self.compiler)
        self.assertNotEqual(result['returncode'], 0)
        self.assertFalse(result['source_bookends_equal'])
        self.assertNotIn('elf_sha256', result)
        with mock.patch.object(kernels.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['cc'], 1)):
            result = kernels.build_kernel(self.row, self.root / 'timeout', self.compiler, wall_seconds=1)
        self.assertEqual(result['returncode'], 124)
        with self.assertRaises(ValueError): kernels.build_kernel(self.row, self.root / 'out', self.compiler)

    def test_real_compiler_embeds_exact_assets(self):
        compiler = os.environ.get('RISCV_CC') or shutil.which('riscv64-unknown-elf-gcc')
        if compiler is None: self.skipTest('set RISCV_CC for actual ELF validation')
        for fast in (False, True):
            with self.subTest(fast_check=fast):
                self._real_compiler_embeds_exact_assets(compiler, fast)

    def _real_compiler_embeds_exact_assets(self, compiler, fast):
        out = self.root / ('real-fast' if fast else 'real-default')
        result = kernels.build_kernel(self.row, out, compiler, max_polls=7, wall_seconds=30, fast_check=fast)
        self.assertEqual(result['returncode'], 0, (out / 'build.log').read_text())
        self.assertEqual(result.get('fast_check', False), fast)
        self.assertEqual('-DATLAS_FAST_CHECK=1' in result['command'], fast)
        self.assertEqual(str(kernels.ROOT / 'runtime/atlas_host_check.h') in result['inputs_sha256'], fast)
        kernels.audit_loadmem((out / 'host.elf').read_bytes(), self.program, self.initial, self.expected, 0x100000000)
        self.assertEqual((out / 'program.bin').read_bytes(), self.program)
        self.assertEqual((out / 'expected-dram.bin').read_bytes(), self.expected)
        self.assertIn('#define PAYLOAD_OUTPUT_RANGES 2u', (out / 'payload.h').read_text())
        broken = bytearray((out / 'host.elf').read_bytes())
        phoff, = struct.unpack_from('<Q', broken, 32)
        size, count = struct.unpack_from('<HH', broken, 54)
        for index in range(count):
            offset = phoff + index * size
            kind, = struct.unpack_from('<I', broken, offset)
            pa, = struct.unpack_from('<Q', broken, offset + 24)
            if kind == 1 and pa == 0x100000000:
                struct.pack_into('<Q', broken, offset + 24, pa + 64)
        with self.assertRaises(ValueError):
            kernels.audit_loadmem(bytes(broken), self.program, self.initial, self.expected, 0x100000000)
        broken = bytearray((out / 'host.elf').read_bytes())
        for index in range(count):
            offset = phoff + index * size
            kind, = struct.unpack_from('<I', broken, offset)
            pa, = struct.unpack_from('<Q', broken, offset + 24)
            if kind == 1 and pa == 0x100000000:
                struct.pack_into('<QQ', broken, offset + 16, 0x80000000, 0x80000000)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            kernels.audit_loadmem(bytes(broken), self.program, self.initial, self.expected, 0x100000000)

    def test_fast_check_requires_a_boolean(self):
        with self.assertRaises(ValueError):
            kernels.build_kernel(self.row, self.root / 'invalid', self.compiler, fast_check='yes')

    def test_fast_result_requires_byte_metrics_and_rejects_byte_failure(self):
        run, meta, log = self.valid_run()
        build = dict(elf_sha256='bound', fast_check=True)
        self.assertFalse(kernels.kernel_result('tiny-00', run, build)['exact'])
        log = log.replace('_unit_mismatches', '_byte_mismatches').replace(
            'ATLAS_KERNEL_CHECK ', 'ATLAS_KERNEL_CHECK_TIMING fast=1 check_cpu_cycles=42\nATLAS_KERNEL_CHECK ')
        (run / 'sim.log').write_text(log)
        self.assertTrue(kernels.kernel_result('tiny-00', run, build)['exact'])
        for field in ('output_byte_mismatches', 'preserved_byte_mismatches'):
            (run / 'sim.log').write_text(log.replace(field + '=0', field + '=1'))
            self.assertFalse(kernels.kernel_result('tiny-00', run, build)['exact'])

    def valid_run(self):
        run = self.root / 'run'; run.mkdir()
        (run / 'exit-status.txt').write_text('0\n')
        meta = dict(elf_sha256='bound', returncode=0, timed_out=False)
        (run / 'run.json').write_text(json.dumps(meta))
        log = ('ATLAS_KERNEL_PHASE result=0 active=0 sequence=1 device_cycles=5376\n'
               'ATLAS_KERNEL_CHECK output_unit_mismatches=0 preserved_unit_mismatches=0 dbg0=1 dbg1=1\n'
               'ATLAS_KERNEL_PASS name=tiny-00 device_cycles=5376\n')
        (run / 'sim.log').write_text(log)
        return run, meta, log

    def test_result_requires_complete_identity_completion_and_zero_exit(self):
        run, meta, log = self.valid_run(); build = dict(elf_sha256='bound')
        self.assertTrue(kernels.kernel_result('tiny-00', run, build)['exact'])
        for bad in [log.replace('sequence=1', 'sequence=0'), log.replace('dbg1=1', 'dbg1=0'),
                    log.replace('preserved_unit_mismatches=0', 'preserved_unit_mismatches=1'),
                    log.replace('output_unit_mismatches=0', 'output_unit_mismatches=1'),
                    log.replace('result=0', 'result=2'), log + log.splitlines()[-1] + '\n',
                    log.replace('name=tiny-00', 'name=other'), log + 'ATLAS_KERNEL_FAIL\n',
                    log.replace('device_cycles=5376', 'device_cycles=5375', 1), log.splitlines()[-1] + '\n']:
            (run / 'sim.log').write_text(bad)
            with self.subTest(log=bad): self.assertFalse(kernels.kernel_result('tiny-00', run, build)['exact'])
        (run / 'sim.log').write_text(log)
        for changed in [dict(meta, timed_out=True), dict(meta, elf_sha256='other'), dict(meta, returncode=2)]:
            (run / 'run.json').write_text(json.dumps(changed))
            self.assertFalse(kernels.kernel_result('tiny-00', run, build)['exact'])
        (run / 'run.json').write_text(json.dumps(meta)); (run / 'exit-status.txt').write_text('2')
        self.assertFalse(kernels.kernel_result('tiny-00', run, build)['exact'])

    def test_batch_detects_changed_elf_and_missing_runs(self):
        out = self.root / 'batch'; path = out / 'tiny-00'; path.mkdir(parents=True)
        (path / 'host.elf').write_bytes(b'elf')
        build = dict(kernel='tiny-00', returncode=0, source_bookends_equal=True, elf_sha256=sha(b'elf'))
        (path / 'build.json').write_text(json.dumps(build))
        (out / 'batch.json').write_text(json.dumps(dict(schema=kernels.SCHEMA, kernels=['tiny-00'], kernel_count=1)))
        self.assertEqual(list(kernels.load_batch(out)), ['tiny-00'])
        report = kernels.collect_results(out, self.root / 'absent')
        self.assertFalse(report['all_kernels_exact']); self.assertIsNone(report['device_cycles'])
        (path / 'host.elf').write_bytes(b'changed')
        with self.assertRaises(ValueError): kernels.load_batch(out)


class FullPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='atlas-full-soc-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def bundle(self, *, kernel_only=False):
        root, initial = full_bundle_fixture(self.root / 'bundle', kernel_only=kernel_only)
        return root, (root / 'assets/program.bin').read_bytes(), initial

    def test_staged_kernel_assets_prepare_without_a_host_elf(self):
        root, program, _ = self.bundle(kernel_only=True)
        receipt = kernels.prepare_full(root, 'tiny-full', self.root / 'staged')
        self.assertEqual(receipt['source_metadata'], 'kernels.json')
        self.assertEqual(receipt['kernel_count'], 2)
        rows = kernels.read_manifest(Path(receipt['manifest']))
        self.assertEqual([Path(row['program_bin']).read_bytes() for row in rows], [program, program])

    def test_full_images_preserve_program_inputs_guards_and_duplicate_launches(self):
        root, program, initial = self.bundle()
        receipt = kernels.prepare_full(root, 'tiny-full', self.root / 'prepared')
        self.assertEqual(receipt['status'], 'PASS_PREPARED_INPUTS_ONLY')
        self.assertTrue(receipt['source_bookends_equal'])
        self.assertFalse(receipt['hardware_execution'])
        for relative in ('tools/atlas_host_layout.py', 'runtime/atlas_host_layout.h',
                         'tools/chipyard/run_sim.py'):
            source = kernels.ROOT / relative
            self.assertEqual(receipt['source_sha256'][str(source)], sha(source.read_bytes()))
        rows = kernels.read_manifest(Path(receipt['manifest']))
        self.assertEqual([row['id'] for row in rows], ['tiny-full-00000', 'tiny-full-00001'])
        for index, row in enumerate(rows):
            original, _, before, after, base, ranges = kernels.admit(row, Admission())
            self.assertEqual(original, program)
            self.assertEqual(before, initial)
            self.assertEqual(after, initial[:2304] + b'e' * 2048 + initial[4352:])
            self.assertEqual((base, ranges), (0x100000000, [(2304, 2048)]))
            self.assertEqual(row['source_index'], index)

    def test_stale_layout_fails_before_output(self):
        root, _, _ = self.bundle()
        kernels.full_bundle(root, 'tiny-full')
        with mock.patch.object(sys.modules['atlas_host_layout'], 'HEADER_SHA256', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'layout identity'):
                kernels.prepare_full(root, 'tiny-full', self.root / 'stale-layout')
        self.assertFalse((self.root / 'stale-layout').exists())
    def test_late_bookend_failure_retains_failed_evidence_without_manifest(self):
        root, _, _ = self.bundle()
        provider = kernels.full_bundle(root, 'tiny-full')
        out = self.root / 'partial'
        with mock.patch.object(kernels, 'full_bundle', return_value=provider), \
             mock.patch.object(provider, 'verify_bookends', side_effect=ValueError('changed source')):
            with self.assertRaises(ValueError):
                kernels.prepare_full(root, 'tiny-full', out)
        self.assertFalse((out / 'manifest.json').exists())
        self.assertEqual(json.loads((out / 'preparation.json').read_text())['status'], 'FAILED')
        self.assertTrue((out / 'tiny-full-00000/program.bin').is_file())

    def test_existing_and_restricted_outputs_are_refused_before_bundle_read(self):
        out = self.root / 'exists'
        out.mkdir()
        with mock.patch.object(kernels, 'full_bundle') as provider:
            for path in (out, self.root / 'forbidden-HaMmEr-copy'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    kernels.prepare_full(self.root / 'absent', 'tiny-full', path)
            provider.assert_not_called()


if __name__ == '__main__':
    unittest.main()
