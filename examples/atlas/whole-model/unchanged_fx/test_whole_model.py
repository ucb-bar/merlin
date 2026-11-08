"""Seam checks must fail before a later stage can use a framework fixture."""
import hashlib
import unittest
from pathlib import Path
import numpy as np
from whole_model import activation_overrides, verify_fixture_boundaries, execution_backend_arguments, require_stateless_manifest


class StatelessManifestTests(unittest.TestCase):
    def test_original_manifest_is_admitted(self):
        require_stateless_manifest({'whole_model': {'schema': 'atlas.sequential_unchanged_fx.v1'}})

    def test_cached_markers_refuse_without_generated_state_authentication(self):
        for manifest in ({'cache_phase': {}}, {'whole_model': {'cache_phase': {}}},
                         {'whole_model': {'schema': 'atlas.sequential_cache_phase_fx.v1'}}):
            with self.subTest(manifest=manifest), self.assertRaisesRegex(ValueError, 'parent-cache'):
                require_stateless_manifest(manifest)


class ActivationContinuityTests(unittest.TestCase):
    def setUp(self):
        self.value = np.arange(12, dtype=np.uint16).reshape(1, 3, 4)
        self.digest = hashlib.sha256(self.value.tobytes()).hexdigest()
        self.link = {"input_index": 2, "region": "layer0", "output_index": 0,
                     "boundary": "add_6", "source_node": "add_6"}
        self.previous = {"layer0": {"outputs": [{"sha256": self.digest}]}}
        self.actual = {"layer0": {"output_0": self.value}}

    def test_exact_generated_words_fill_selected_boundary(self):
        overrides, seams = activation_overrides("layer1", [self.link], self.previous, self.actual)
        self.assertEqual(set(overrides), {"input_2"})
        self.assertEqual(overrides["input_2"].tobytes(), self.value.tobytes())
        self.assertEqual(seams[0]["sha256"], self.digest)

    def test_tampered_generated_words_refuse(self):
        self.actual["layer0"]["output_0"] = self.value + 1
        with self.assertRaisesRegex(ValueError, "output changed"):
            activation_overrides("layer1", [self.link], self.previous, self.actual)

    def test_missing_generated_predecessor_refuses(self):
        with self.assertRaisesRegex(ValueError, "has not executed"):
            activation_overrides("layer1", [self.link], {}, self.actual)

    def test_later_stage_without_seam_refuses(self):
        with self.assertRaisesRegex(ValueError, "no generated predecessor"):
            activation_overrides("layer1", [], self.previous, self.actual)

    def test_only_initial_region_accepts_capture_boundaries(self):
        self.assertEqual(activation_overrides("layer0", [], {}, {}), ({}, []))


class FixtureBoundaryTests(unittest.TestCase):
    def test_later_undeclared_hidden_state_cannot_fall_back_to_fixture(self):
        sources = {"layer0": {"source_boundaries": ["embedding", "mask", "rope"]},
                   "layer1": {"source_boundaries": ["generated_hidden", "mask", "missed_generated_value"]}}
        links = {"layer0": [], "layer1": [{"input_index": 0}]}
        with self.assertRaisesRegex(ValueError, "cannot use framework fixture"):
            verify_fixture_boundaries(sources, links)

    def test_only_known_preamble_inputs_remain_fixtures(self):
        sources = {"layer0": {"source_boundaries": ["embedding", "mask", "rope"]},
                   "layer1": {"source_boundaries": ["generated_hidden", "mask", "rope"]}}
        verify_fixture_boundaries(sources, {"layer0": [], "layer1": [{"input_index": 0}]})


class ExecutionBackendTests(unittest.TestCase):
    def test_original_functional_backend_is_explicit_default_without_accelerators(self):
        self.assertEqual(execution_backend_arguments('functional'), ['--backend', 'functional'])

    def test_explicit_functional_budget_is_forwarded_and_bounded(self):
        self.assertEqual(execution_backend_arguments('functional', max_instructions=2000000),
                         ['--backend', 'functional', '--max-instructions', '2000000'])
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                execution_backend_arguments('functional', max_instructions=value)

    def test_qualified_backend_forwards_separate_receipts_and_cache_paths(self):
        native, memory, cache = (Path(name) for name in ('native.json', 'memory.json', 'native-cache'))
        self.assertEqual(execution_backend_arguments('qualified_native', native, memory, cache),
            ['--backend', 'qualified_native', '--native-qualification', str(native.resolve()),
             '--memory-qualification', str(memory.resolve()), '--native-cache', str(cache.resolve())])

    def test_missing_qualification_or_unselected_backend_never_falls_back(self):
        for backend, native, memory, cache in (
                ('qualified_native', None, Path('memory.json'), None),
                ('qualified_native', Path('native.json'), None, None),
                ('functional', Path('native.json'), Path('memory.json'), None),
                ('functional', None, None, Path('cache')), ('unknown', None, None, None)):
            with self.subTest(backend=backend, native=native, memory=memory, cache=cache), self.assertRaises(ValueError):
                execution_backend_arguments(backend, native, memory, cache)


if __name__ == "__main__":
    unittest.main()
