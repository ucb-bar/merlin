"""Storage-only differential against an explicitly pinned original source class.

An optional CLI emits a source-bound receipt after every check passes. The
reference class is compiled verbatim from its original AST, with the same
torch module; no model source is changed and no instruction is substituted.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

import torch

from merlin.runtime.uniform_sparse_memory import (
    QUALIFICATION_SCHEMA, REQUIRED_CASES, UniformSparseByteMemory,
    adapter_identity, reference_source_identity, verify_qualification,
)

RANDOM_ITERATIONS = 256


class UniformStorageTest(unittest.TestCase):
    def test_default_is_lazy_and_read_tensors_are_snapshots(self):
        memory = UniformSparseByteMemory(2*4096+13)
        self.assertEqual(memory._pages, {})
        read = memory[:]
        self.assertTrue(torch.equal(read, torch.full((memory.size,), 0xCD, dtype=torch.uint8)))
        read.view(torch.uint8).zero_()
        self.assertEqual(memory._pages, {})
        self.assertEqual(memory[0:1].item(), 0xCD)
        memory[0:1] = torch.tensor([0], dtype=torch.uint8)
        self.assertEqual(read[0].item(), 0)
        self.assertEqual(memory[1:2].item(), 0xCD)

    def test_zero_pages_survive_until_all_bytes_are_default(self):
        memory = UniformSparseByteMemory(4096+17)
        memory[:] = torch.zeros(memory.size, dtype=torch.uint8)
        self.assertEqual(len(memory._pages), 2)
        self.assertEqual(memory[:].count_nonzero().item(), 0)
        memory[:4095] = torch.full((4095,), 0xCD, dtype=torch.uint8)
        self.assertIn(0, memory._pages)
        self.assertEqual(memory[4095:4096].item(), 0)
        memory[4095:4096] = torch.tensor([0xCD], dtype=torch.uint8)
        self.assertNotIn(0, memory._pages)
        memory[4096:] = torch.full((17,), 0xCD, dtype=torch.uint8)
        self.assertEqual(memory._pages, {})

    def test_explicit_default_and_empty_memory(self):
        memory = UniformSparseByteMemory(17, default_byte=0)
        memory[:] = torch.zeros(17, dtype=torch.uint8)
        self.assertEqual(memory._pages, {})
        empty = UniformSparseByteMemory(0)[:]
        self.assertEqual(empty.dtype, torch.uint8)
        self.assertEqual(empty.numel(), 0)
        for invalid in (-1, 256, True, 1.5):
            with self.subTest(default_byte=invalid), self.assertRaises(ValueError):
                UniformSparseByteMemory(4, default_byte=invalid)


class QualificationMetadataTest(unittest.TestCase):
    def test_verifier_binds_both_sources_default_and_required_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'reference.py'
            source.write_text('class OriginalMemory: pass\n')
            expected = sha256(source.read_bytes()).hexdigest()
            receipt = {**adapter_identity(), 'schema': QUALIFICATION_SCHEMA, 'status': 'passed',
                       'reference_source_sha256': expected, 'checks': dict.fromkeys(REQUIRED_CASES, 1)}
            verified = verify_qualification(receipt, reference_source=source, expected_reference_sha256=expected)
            self.assertFalse(verified['hardware_or_arithmetic_qualification'])
            for changed in ({'adapter_source_sha256': '0'*64}, {'default_byte': 0},
                            {'status': 'pending'}, {'checks': {}}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    verify_qualification({**receipt, **changed}, reference_source=source,
                                         expected_reference_sha256=expected)
            source.write_text('class ChangedMemory: pass\n')
            with self.assertRaisesRegex(ValueError, 'reference source'):
                verify_qualification(receipt, reference_source=source, expected_reference_sha256=expected)


class ReferenceDifferentialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        selected = os.environ.get('MERLIN_SPARSE_MEMORY_REFERENCE')
        expected = os.environ.get('MERLIN_SPARSE_MEMORY_REFERENCE_SHA256')
        if not selected or not expected:
            raise unittest.SkipTest('select the original memory source and its pinned SHA256')
        cls.reference_source = Path(selected)
        cls.reference_sha256 = reference_source_identity(cls.reference_source, expected)
        text = cls.reference_source.read_text()
        tree = ast.parse(text)
        definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SparseByteMemory']
        if len(definitions) != 1:
            raise ValueError('reference source needs exactly one original SparseByteMemory class')
        definition = definitions[0]
        cls.reference_class_sha256 = sha256(ast.get_source_segment(text, definition).encode()).hexdigest()
        namespace = {'torch': torch, '__name__': 'selected_original_sparse_memory'}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(cls.reference_source), 'exec'), namespace)
        cls.reference_type = namespace['SparseByteMemory']
        if cls.reference_type.PAGE_BYTES != UniformSparseByteMemory.PAGE_BYTES:
            raise ValueError('selected reference has a different page-size contract')
        cls.checks = dict.fromkeys(REQUIRED_CASES, 0)

    def pair(self, size=4*4096+31):
        reference = self.reference_type(size)
        reference[:] = torch.full((size,), 0xCD, dtype=torch.uint8)
        return reference, UniformSparseByteMemory(size)

    def equal(self, reference, memory, key=slice(None)):
        actual, expected = memory[key], reference[key]
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertTrue(torch.equal(actual, expected))

    def test_randomized_cross_page_reads_and_writes(self):
        rng = random.Random(34019)
        generator = torch.Generator().manual_seed(34019)
        reference, memory = self.pair()
        for iteration in range(RANDOM_ITERATIONS):
            if iteration % 3:
                start = rng.randrange(memory.size+1)
                count = rng.randrange(min(memory.size-start, 4096+47)+1)
                if iteration % 11 == 0:
                    value = torch.zeros(count, dtype=torch.uint8)
                elif iteration % 13 == 0:
                    value = torch.full((count,), 0xCD, dtype=torch.uint8)
                else:
                    value = torch.randint(256, (count,), dtype=torch.uint8, generator=generator)
                key = slice(start, start+count) if iteration % 2 else slice(start, None)
                reference[key] = value
                memory[key] = value
            boundary = rng.choice((4096, 8192, 12288))
            self.equal(reference, memory, slice(boundary-rng.randrange(47), boundary+rng.randrange(47)))
            if iteration % 16 == 0:
                self.equal(reference, memory)
            self.checks['randomized_cross_page'] += 1
        self.equal(reference, memory)

    def test_zero_writes_and_uniform_page_reclamation(self):
        reference, memory = self.pair(2*4096+13)
        for key in (slice(0, 4096), slice(4093, 4101), slice(8192, 8205)):
            value = torch.zeros(key.stop-key.start, dtype=torch.uint8)
            reference[key] = value
            memory[key] = value
            self.equal(reference, memory)
            self.checks['zero_writes'] += 1
        self.assertIn(0, memory._pages)
        for key in (slice(0, 4095), slice(4095, 8192), slice(8192, 8205)):
            value = torch.full((key.stop-key.start,), 0xCD, dtype=torch.uint8)
            reference[key] = value
            memory[key] = value
            self.equal(reference, memory)
            self.checks['uniform_page_reclamation'] += 1
        self.assertEqual(memory._pages, {})

    def test_bounds_and_error_identity_before_any_write(self):
        reference, memory = self.pair(4096+17)
        value = torch.tensor([1, 2], dtype=torch.uint8)
        operations = [
            lambda m: m[-1:0], lambda m: m[2:1], lambda m: m[:m.size+1],
            lambda m: m[::2], lambda m: m[::-1], lambda m: m[0],
            lambda m: m.__setitem__(slice(-1, 1), value),
            lambda m: m.__setitem__(slice(0, 1), value),
            lambda m: m.__setitem__(slice(m.size, None), value),
            lambda m: m.__setitem__(slice(None, None, 2), value),
        ]
        for operation in operations:
            errors = []
            for target in (reference, memory):
                try:
                    operation(target)
                except Exception as error:
                    errors.append((type(error), str(error)))
            self.assertEqual(len(errors), 2)
            self.assertEqual(errors[0], errors[1])
            self.equal(reference, memory)
            self.checks['bounds_and_errors'] += 1
        for key in (slice(None, 0), slice(memory.size, memory.size), slice(4096, 4096, 1)):
            self.equal(reference, memory, key)
            reference[key] = torch.empty(0, dtype=torch.uint8)
            memory[key] = torch.empty(0, dtype=torch.uint8)
            self.equal(reference, memory)
            self.checks['bounds_and_errors'] += 1

    def test_tensor_views_source_mutation_and_deepcopy_are_isolated(self):
        reference, memory = self.pair(4096+17)
        base = torch.arange(48, dtype=torch.int64).reshape(6, 8)
        view = base[:, ::2]
        reference[4080:] = view
        memory[4080:] = view
        base.zero_()
        self.equal(reference, memory)
        left, right = reference[4080:4104], memory[4080:4104]
        left.view(torch.int16).fill_(-1)
        right.view(torch.int16).fill_(-1)
        self.equal(reference, memory)
        reference[4080:4104] = torch.zeros(24, dtype=torch.uint8)
        memory[4080:4104] = torch.zeros(24, dtype=torch.uint8)
        self.assertTrue(torch.equal(left, right))
        self.equal(reference, memory)
        first, second = deepcopy(reference), deepcopy(memory)
        first[0:1] = torch.tensor([0], dtype=torch.uint8)
        second[0:1] = torch.tensor([0], dtype=torch.uint8)
        self.equal(first, second)
        self.assertEqual(memory[0:1].item(), 0xCD)
        self.checks['mutation_isolation'] += 4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-source')
    parser.add_argument('--reference-sha256')
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--iterations', type=int, default=RANDOM_ITERATIONS)
    args = parser.parse_args()
    if args.reference_source:
        os.environ['MERLIN_SPARSE_MEMORY_REFERENCE'] = args.reference_source
    if args.reference_sha256:
        os.environ['MERLIN_SPARSE_MEMORY_REFERENCE_SHA256'] = args.reference_sha256
    if args.iterations <= 0:
        parser.error('iterations must be positive')
    RANDOM_ITERATIONS = args.iterations
    torch.set_num_threads(1)
    result = unittest.main(argv=[__file__], exit=False).result
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.receipt:
        if result.skipped or not all(ReferenceDifferentialTest.checks.values()):
            raise SystemExit('cannot emit a qualification receipt with skipped reference checks')
        receipt = {**adapter_identity(), 'schema': QUALIFICATION_SCHEMA, 'status': 'passed',
                   'reference_source_sha256': ReferenceDifferentialTest.reference_sha256,
                   'reference_class_sha256': ReferenceDifferentialTest.reference_class_sha256,
                   'reference_execution': 'verbatim-original-class-ast-with-original-torch-operations',
                   'checks': ReferenceDifferentialTest.checks, 'random_seed': 34019,
                   'tests_run': result.testsRun}
        verify_qualification(receipt, reference_source=ReferenceDifferentialTest.reference_source,
                             expected_reference_sha256=ReferenceDifferentialTest.reference_sha256)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
