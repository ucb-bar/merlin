"""Policy and reservation checks; no compiler, simulator or board is required."""

import copy
import importlib.util
import json
import unittest

from merlin.common.paths import repo_root


class TargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = repo_root() / "examples/gemmini/comparisons/tvm"
        spec = importlib.util.spec_from_file_location("tvm_target_check", directory / "check_target.py")
        cls.checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.checker)
        cls.template = cls.checker.yaml.load((directory / "target.yaml").read_text(), Loader=cls.checker.UniqueLoader)

    def setUp(self):
        self.config = copy.deepcopy(self.template)
        self.memory = self.config["development"]["memory"]
        self.base = self.memory["base_address"]

    def plan(self, *intervals):
        return {"regions": [{"name": name, "address": self.base + offset, "size_bytes": size} for name, offset, size in intervals]}

    def test_unknown_capacity_and_absent_plan_stay_unknown(self):
        result = self.checker.check_target(self.config)
        self.assertIsNone(result["memory"]["declared_confirmed_size_bytes"])
        self.assertIsNone(result["memory"]["fits_requested_map"])
        self.assertIsNone(result["memory"]["reserved_bytes"])
        self.assertFalse(result["deployment_qualified"])
        self.assertFalse(result["paper_ready"])
        self.assertEqual(result["compile_defines"], ["TVM_GEMMINI_FORBID_HW_LOOPS=1", "TVM_GEMMINI_PE_OUTPUT_BITS=20"])
        self.assertTrue(result["instruction_policy_confirmed"])

    def test_adjacent_and_boundary_reservations(self):
        size = self.memory["requested_size_bytes"]
        result = self.checker.check_target(self.config, self.plan(("code", 0, 4096), ("arena", 4096, size - 4096)))
        self.assertEqual(result["memory"]["reserved_bytes"], size)
        self.assertEqual(result["memory"]["unreserved_bytes"], 0)
        self.assertTrue(result["memory"]["fits_requested_map"])
        self.assertFalse(result["memory"]["deployment_fit_verified"])

    def test_invalid_reservations(self):
        cases = [
            self.plan(("code", 0, 4096), ("arena", 4095, 4096)),
            self.plan(("below", -1, 1)),
            self.plan(("beyond", self.memory["requested_size_bytes"], 1)),
            self.plan(("empty", 0, 0)),
            self.plan(("negative", 0, -1)),
            self.plan(("bool", 0, True)),
            self.plan(("same", 0, 1), ("same", 1, 1)),
            self.plan((" ", 0, 1)),
            {"regions": []},
            {"regions": [{"name": "arena", "address": 2**64 - 1, "size_bytes": 2}]},
        ]
        for plan in cases:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                self.checker.check_target(self.config, plan)

    def test_confirmed_capacity_must_cover_request(self):
        self.memory["confirmed_size_bytes"] = 4 * 2**30
        with self.assertRaisesRegex(ValueError, "exceeds declared confirmed"):
            self.checker.check_target(self.config)
        self.memory["confirmed_size_bytes"] = 8 * 2**30
        result = self.checker.check_target(self.config, self.plan(("code", 0, 4096)))
        self.assertFalse(result["memory"]["deployment_fit_verified"])

    def test_invalid_map(self):
        for key, value in (("base_address", True), ("base_address", -1), ("base_address", 2**64 - 1), ("requested_size_bytes", 0), ("confirmed_size_bytes", True)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                config = copy.deepcopy(self.config)
                config["development"]["memory"][key] = value
                self.checker.check_target(config)

    def test_unsupported_policy_and_arithmetic(self):
        cases = [
            ("instructions", "hardware_loops", "allow"),
            ("instructions", "dataflow", "WS"),
            ("instructions", "paper_confirmed", "false"),
            ("arithmetic", "input_dtype", "uint8"),
            ("arithmetic", "output_dtype", "uint64"),
            ("arithmetic", "full_accumulator_readout", False),
            ("arithmetic", "dim", True),
            ("arithmetic", "dim", 3),
            ("arithmetic", "pe_output_bits", 19),
            ("arithmetic", "scratchpad_bytes", 1023),
            ("arithmetic", "accumulator_bytes", 64),
            ("runtime", "kind", "linux"),
            ("sources", "params_sha256", "unknown"),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                config = copy.deepcopy(self.config)
                config["development"][section][key] = value
                self.checker.check_target(config)

    def test_duplicate_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            self.checker.yaml.load("hardware_loops: allow\nhardware_loops: forbid\n", Loader=self.checker.UniqueLoader)
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            json.loads('{"size_bytes": 4, "size_bytes": 8}', object_pairs_hook=self.checker.unique_mapping)


if __name__ == "__main__":
    unittest.main()
