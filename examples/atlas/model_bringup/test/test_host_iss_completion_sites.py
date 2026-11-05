"""Accept exact outlined completion instructions and reject changed trap stops."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

TOOLS = Path(__file__).absolute().parents[1] / "tools"
sys.path[:0] = [str(TOOLS), str(TOOLS / "host_iss")]
from run import resnet_stop_sites
from tiny_full_contract import stop_sites


class CompletionSitesTest(unittest.TestCase):
    def test_resnet_outlined_finish_includes_success_stop(self):
        image = Mock()
        image.symbols = {"finish": {}}
        instructions = {"atlas_workload_main": [(0x100, 4, 0x10500073)],
                        "atlas_block_trap_report": [], "finish": [(0x180, 4, 0x10500073), (0x184, 4, 0x13)]}
        image.instructions.side_effect = lambda name: instructions[name]
        self.assertEqual(resnet_stop_sites(image), {0x100: 4, 0x180: 4})

    def test_tiny_outlined_complete_and_exact_named_trap(self):
        image = Mock()
        image.symbols = dict(complete={}, trap_entry=dict(address=0x200, size=80),
                             tiny_full_trap_stop=dict(address=0x240, size=2))
        image.instructions.side_effect = lambda name: [] if name == "main" else [(0x180, 4, 0x10500073)]
        image.bytes_at.return_value = b"\x01\xa0"
        self.assertEqual(stop_sites(image), {0x180: 4, 0x240: 2})
        image.bytes_at.return_value = b"\x01\x00"
        with self.assertRaisesRegex(ValueError, "trap stop changed"):
            stop_sites(image)
        image.bytes_at.return_value = b"\x01\xa0"
        image.symbols["tiny_full_trap_stop"]["address"] = 0x300
        with self.assertRaisesRegex(ValueError, "trap stop changed"):
            stop_sites(image)


if __name__ == "__main__":
    unittest.main()
