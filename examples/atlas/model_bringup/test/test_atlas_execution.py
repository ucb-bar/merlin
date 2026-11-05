"""Real selected-model checks for the shared execution API and JSONL worker."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

TOOLS = Path(__file__).absolute().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import kernel_library  # noqa: E402,F401
from atlas_execution import Executor, MODEL_PIN


@unittest.skipUnless(os.environ.get('ATLAS_EXECUTION_MODEL_ROOT'), 'set explicit 47b8599 model checkout')
class ExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['ATLAS_EXECUTION_MODEL_ROOT']).absolute()
        cls.executor = Executor(cls.root)

    def test_full_arena_dma_binary_and_rtl_binding(self):
        from atlas_attention_program import AttentionConfig, assembly, pack_tiles, stage
        from tiny_full.attention_worker import dma_counts

        config = AttentionConfig(query_rows=1, keys=1, query_heads=2, kv_heads=1, mask='causal')
        q = np.zeros((2, 1, 64), dtype=np.uint16)
        k = np.zeros((1, 1, 64), dtype=np.uint16)
        v = np.full((1, 1, 64), 0x3f80, dtype=np.uint16)
        arena, offsets = stage(config, q, k, v)
        source = assembly(config, offsets)
        _, digest = self.executor.assemble(source)
        expected = bytearray(arena)
        start, size = offsets['output'], offsets['output_bytes']
        expected[start:start + size] = pack_tiles(np.full_like(q, 0x3f80))
        requested = dma_counts(config, arena, offsets)
        actual = self.executor.execute(source, arena, start, size, expected_arena=bytes(expected),
                                       requested_dma=requested, program_sha256=digest)
        self.assertEqual(actual.model['revision'], MODEL_PIN)
        self.assertEqual(actual.model['numerics'], 'rtl')
        self.assertEqual(len(actual.model['unary_table_sha256']), 11)
        self.assertEqual(actual.arena, expected)
        self.assertEqual(actual.observed, requested)
        self.assertTrue(actual.implementation_source_sha256)
        model_sources = list(actual.model['imported_source_sha256'])
        self.assertEqual(subprocess.run(['git', '-C', str(self.root), 'diff', '--quiet', 'HEAD', '--',
                                         *model_sources], check=False).returncode, 0)

    def test_invalid_dma_full_arena_program_and_halt_are_rejected(self):
        arena = bytes(64)
        with self.assertRaisesRegex(AssertionError, 'DMA byte counts'):
            self.executor.execute('ecall\n', arena, 32, 0, requested_dma=dict(load_bytes=32, store_bytes=0))
        with self.assertRaisesRegex(AssertionError, 'full staged arena'):
            self.executor.execute('ecall\n', arena, 32, 32, expected_arena=bytes(63) + b'\1')
        with self.assertRaisesRegex(ValueError, 'binary identity'):
            self.executor.execute('ecall\n', arena, 32, 0, program_sha256='0' * 64)
        with self.assertRaisesRegex(AssertionError, 'ECALL'):
            self.executor.execute('nop\n', arena, 32, 0)

    def test_modified_model_source_or_unary_table_is_rejected_without_editing_checkout(self):
        original_read = Path.read_bytes
        for relative in ('npu_model/configs/data/exp.bin.gz', 'npu_model/configs/numerics.py'):
            selected = self.root / relative

            def altered_read(path):
                raw = original_read(path)
                return raw + b'tampered' if path == selected else raw

            with self.subTest(path=relative), mock.patch.object(Path, 'read_bytes', altered_read):
                with self.assertRaisesRegex(ValueError, 'differs from pinned revision'):
                    self.executor.execute('ecall\n', bytes(64), 32, 0)

    def test_persistent_worker_uses_explicit_root_and_capture_outside_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            descriptors = {}
            for role, shape in (('q', (2, 1, 64)), ('k', (1, 1, 64)), ('v', (1, 1, 64))):
                words = np.full(shape, 0x3f80 if role == 'v' else 0, dtype='<u2')
                raw = words.tobytes()
                (root / (role + '.bin')).write_bytes(raw)
                descriptors[role + '_bf16'] = dict(file=role + '.bin', dtype='uint16', shape=list(shape),
                                                   bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            command = [sys.executable, '-B', str(TOOLS / 'tiny_full/attention_worker.py'), '--model-root', str(self.root)]
            environment = dict(os.environ)
            environment.pop('PYTHONPATH', None)
            worker = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, cwd=directory, env=environment)
            try:
                for index in range(2):
                    request = dict(root=str(root), out=str(root / str(index)), **descriptors,
                                   config=dict(query_rows=1, keys=1, query_heads=2, kv_heads=1, mask='causal'))
                    worker.stdin.write(json.dumps(request) + '\n')
                    worker.stdin.flush()
                    answer = json.loads(worker.stdout.readline())
                    self.assertEqual(answer['status'], 'ok', answer)
                    record = answer['record']
                    self.assertEqual(record['model_commit'], MODEL_PIN)
                    self.assertEqual(record['requested_dma'], record['execution']['observed'])
                    output = np.fromfile(root / record['output_bf16']['file'], dtype='<u2')
                    self.assertTrue(np.all(output == 0x3f80))
                worker.stdin.write(json.dumps(dict(command='close')) + '\n')
                worker.stdin.flush()
                self.assertEqual(json.loads(worker.stdout.readline()), dict(status='closed'))
                self.assertEqual(worker.wait(timeout=30), 0)
            finally:
                if worker.poll() is None:
                    worker.kill()
                    worker.wait(timeout=30)
                worker.stdin.close()
                worker.stdout.close()
                worker.stderr.close()


if __name__ == '__main__':
    unittest.main()
