import copy
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
from smol_full_plan import calls, panels, validate_reference
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked


class PlanTest(unittest.TestCase):
    def test_full_architecture_extents(self):
        graph, kernels = calls(), panels()
        self.assertEqual((len(graph), len(kernels)), (1421, 2715))
        self.assertEqual(sum(row['site'] == 7 for row in graph), 148)
        self.assertEqual(sum(row['site'] == 5 for row in graph), 112)
        self.assertEqual(sum(row['site'] == 6 for row in graph), 1120)
        self.assertEqual(max(row['physical_mkn'][1] for row in kernels), 12288)

    @unittest.skipUnless(os.environ.get('SMOL_REPLAY') and os.environ.get('SMOL_KERNEL_ASSETS'),
                         'Optional full-policy reference is not configured')
    def test_actual_reference_plan_and_mutations(self):
        def load(value):
            path = Path(value)
            if any(word in part.lower() for part in path.parts for word in ('hammer', 'vlsi')):
                raise ValueError('restricted path')
            return json.loads(checked(path).read_bytes())
        replay = load(Path(os.environ['SMOL_REPLAY']) / 'receipt.json')
        manifest = load(Path(os.environ['SMOL_KERNEL_ASSETS']) / 'kernels.json')
        self.assertEqual(len(validate_reference(replay, manifest)), 1421)
        for field, value in (('call', 1), ('logical_row_start', 128), ('a_scale', 100),
                             ('physical_mkn', [32, 768, 768])):
            changed = copy.deepcopy(manifest)
            changed['kernels'][0][field] = value
            with self.assertRaises(ValueError): validate_reference(replay, changed)
        changed = copy.deepcopy(replay)
        changed['calls'][262]['input']['dtype'] = 'torch.bfloat16'
        with self.assertRaises(ValueError): validate_reference(changed, manifest)


if __name__ == '__main__':
    unittest.main()
