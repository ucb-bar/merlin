"""The full-session edge preserves target indexing and native cache evidence."""

import importlib.util
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from merlin.common.paths import repo_root


path = repo_root() / 'examples/atlas/comparisons/compare_tinyllama_session.py'
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location('native_session', path)
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)


class NativeSessionTest(unittest.TestCase):
    def test_rng_is_restored_on_success_and_failed_evaluation(self):
        state = np.arange(8, dtype=np.uint8)
        fake_torch = SimpleNamespace(
            get_rng_state=lambda: SimpleNamespace(numpy=lambda: state),
            from_numpy=lambda value: value,
            set_rng_state=lambda value: np.copyto(state, value))
        before = edge.rng_snapshot(fake_torch)
        for fails in (False, True):
            def run(*_):
                state[0] ^= np.uint8(1)
                np.random.random()
                edge.random.random()
                if fails:
                    raise ValueError('directed native failure')
            report = {'status': 'failed'}
            with self.subTest(fails=fails), patch.dict(sys.modules, {'torch': fake_torch}), patch.object(edge, '_evaluate',
                side_effect=run):
                if fails:
                    with self.assertRaisesRegex(ValueError, 'directed native failure'):
                        edge.evaluate(None, None, report)
                else:
                    edge.evaluate(None, None, report)
            self.assertFalse(report['rng_bookends']['evaluation_unchanged'])
            self.assertTrue(report['rng_bookends']['caller_restored'])
            self.assertEqual(edge.rng_fingerprints(before), edge.rng_fingerprints(edge.rng_snapshot(fake_torch)))

    def test_exact_continuation_targets_exclude_final_unscored_distribution(self):
        logits = np.broadcast_to(np.arange(33)[:, None], (33, 32000))
        tokens = np.arange(160)[None]
        selected, targets = edge.scoring_rows(logits, tokens)
        np.testing.assert_array_equal(selected[:, 0], np.arange(32))
        np.testing.assert_array_equal(targets, np.arange(128, 160))
        with self.assertRaises(ValueError):
            edge.scoring_rows(logits[:32], tokens)

    def test_cache_append_refuses_changed_old_prefix_and_wrong_geometry(self):
        old = np.arange(16, dtype=np.uint16).reshape(1, 2, 4, 2)
        new = np.concatenate((old, np.zeros((1, 2, 1, 2), np.uint16)), axis=-2)
        edge.require_cache_prefix(new, old, lambda value: value)
        changed = new.copy()
        changed[0, 0, 1, 0] ^= np.uint16(1)
        for bad in (changed, old, new[:, :1]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                edge.require_cache_prefix(bad, old, lambda value: value)

    def test_native_descriptor_includes_dtype_shape_and_signed_zero(self):
        original = np.array([0.0], dtype=np.float32)
        edge.require_native_array(original.copy(), original, 'native')
        for changed in (np.array([-0.0], dtype=np.float32), original.astype(np.float64), original.reshape(1, 1)):
            with self.subTest(dtype=changed.dtype), self.assertRaises(ValueError):
                edge.require_native_array(changed, original, 'native')

    def test_descriptor_archive_rejects_path_duplicate_and_unexpected_members(self):
        record = {'path': '/not-opened/descriptor.npz', 'sha256': 'unused'}
        with patch.object(edge, 'identity', return_value=record), \
                patch.object(edge.zipfile, 'ZipFile') as archive, \
                patch.object(edge.np, 'load', side_effect=AssertionError('member data read')):
            for names in (('keys.npy', '../values.npy'), ('keys.npy', 'keys.npy'), ('keys.npy', 'other.npy')):
                with self.subTest(names=names):
                    archive.return_value.__enter__.return_value.infolist.return_value = [
                        SimpleNamespace(filename=name, file_size=1) for name in names]
                    with self.assertRaisesRegex(ValueError, 'Unexpected native descriptor archive members'):
                        edge.load_descriptor(record, {'keys', 'values'})


if __name__ == '__main__':
    unittest.main()
