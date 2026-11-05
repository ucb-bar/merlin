"""Run the required export/capture source joins without optional skips."""

import os
from pathlib import Path
import sys
import unittest

import kernel_library
from atlas_path_guard import checked


ROOT = Path(__file__).absolute().parents[1]
COVERAGE = kernel_library.LIBRARY / 'coverage'


class RequiredResult(unittest.TextTestResult):
    def addSkip(self, test, reason):
        self.addFailure(test, (AssertionError, AssertionError('required check skipped: ' + reason), None))


def main():
    required = [(COVERAGE / 'fixture' / f'{model}.json', f'{model} pinned export fixture')
                for model in ('tinyllama', 'smolvla')]
    for name, suffix in (('SMOL_REPLAY', 'receipt.json'), ('SMOL_KERNEL_ASSETS', 'kernels.json'),
                         ('TINY_FULL_CAPTURE', '')):
        value = os.environ.get(name)
        if not value:
            print(f'{name} is required; see test/FAITHFULNESS.md', file=sys.stderr)
            return 1
        required.append((Path(value) / suffix if suffix else Path(value), name))
    for path, description in required:
        if not checked(path).is_file():
            print(f'Missing required {description}: {path}', file=sys.stderr)
            return 1

    sys.path[:0] = [str(ROOT / 'test'), str(COVERAGE / 'tests')]
    # Name the real-reference cases explicitly so their removal fails discovery.
    # Numerical bench execution is a separate gate.
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'test_model_coverage',
        'test_smol_full_plan.PlanTest.test_full_architecture_extents',
        'test_smol_full_plan.PlanTest.test_actual_reference_plan_and_mutations',
        'test_tiny_full_segments.InventoryTest.test_complete_inventory_and_missing_kv_panel_or_logit_controls',
        'test_build_table',
        'test_bench_model.PlanTest',
    ])
    result = unittest.TextTestRunner(verbosity=2, resultclass=RequiredResult).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
