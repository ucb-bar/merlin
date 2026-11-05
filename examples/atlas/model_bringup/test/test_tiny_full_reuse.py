"""Full Tiny policy controls preserve baseline admission and reject dishonest DMA."""
import copy
import sys
from pathlib import Path
import unittest

TOOLS = Path(__file__).absolute().parents[1]/'tools'
sys.path[:0] = [str(TOOLS),str(TOOLS/'host_iss')]
import kernel_library  # noqa: E402,F401
from tiny_full_contract import validate_plan
from tiny_contract_fixture import tiny_contract_fixture


def fixture(reuse):
    return tiny_contract_fixture(1, (0, 0, 0), [0, 0], 310, full_provenance=True, reuse=reuse)


class FullReuseTest(unittest.TestCase):
    def test_original_and_public_reuse_have_distinct_valid_dma(self):
        original,old_provenance = fixture(False)
        reuse,new_provenance = fixture(True)
        validate_plan(original,old_provenance)
        validate_plan(reuse,new_provenance)
        self.assertTrue(any(a['requested_dma'] != b['requested_dma']
                            for a,b in zip(original['kernels'],reuse['kernels'])))
        self.assertEqual([x['physical_mkn'] for x in original['kernels']],
                         [x['physical_mkn'] for x in reuse['kernels']])

    def test_reuse_cannot_retain_original_dma_or_change_policy(self):
        original,_ = fixture(False)
        reuse,provenance = fixture(True)
        index = next(i for i,(a,b) in enumerate(zip(original['kernels'],reuse['kernels']))
                     if a['requested_dma'] != b['requested_dma'])
        reuse['kernels'][index]['requested_dma'] = original['kernels'][index]['requested_dma']
        provenance['kernels'] = copy.deepcopy(reuse['kernels'])
        with self.assertRaisesRegex(ValueError,'physical layout or DMA'):
            validate_plan(reuse,provenance)
        reuse,provenance = fixture(True)
        reuse['program_policy'] = 'original'
        with self.assertRaisesRegex(ValueError,'policy binding'):
            validate_plan(reuse,provenance)

    def test_reuse_cannot_claim_instruction_execution(self):
        reuse,provenance = fixture(True)
        reuse['reuse']['instruction_execution'] = True
        provenance['reuse'] = copy.deepcopy(reuse['reuse'])
        with self.assertRaisesRegex(ValueError,'preparation binding'):
            validate_plan(reuse,provenance)


if __name__ == '__main__':
    unittest.main()
