"""Historical capture metadata follows only explicit lexical prefix mappings."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
from smol_capture import capture


class CaptureRelocationTests(unittest.TestCase):
    def setUp(self):
        self.saved_map, self.saved_bundle = capture.PATH_MAP, capture.BUNDLE
        self.addCleanup(setattr, capture, 'PATH_MAP', self.saved_map)
        self.addCleanup(setattr, capture, 'BUNDLE', self.saved_bundle)
        capture.BUNDLE = Path('/synthetic/input-bundle')
        capture.PATH_MAP = [('/historical', '/synthetic/general'),
                            ('/historical/provider', '/synthetic/provider')]

    def test_distinct_provider_and_bundle_use_longest_prefix_without_access(self):
        original = dict(source=dict(weights=dict(path='/historical/provider/weights.bin')),
                        source_bookends=[dict(path='/historical/provider/weights.bin', sha256='saved')],
                        command=['capture', '--observation-input', '/historical/observations.npz'],
                        hashes={'/historical/provider/model.c': 'saved'}, relative='host/bias.bin')
        frozen = copy.deepcopy(original)
        with patch.object(Path, 'lstat', side_effect=AssertionError('metadata must remain lexical')), \
             patch.object(Path, 'stat', side_effect=AssertionError('metadata must remain lexical')), \
             patch.object(Path, 'open', side_effect=AssertionError('metadata must remain lexical')):
            relocated = capture.relocate_historical_metadata(original)
            bookends = capture.compile_bookends(dict(source_sha256=original['hashes']))
            mappings = capture.metadata_relocation()
        self.assertEqual(relocated['source']['weights']['path'], '/synthetic/provider/weights.bin')
        self.assertEqual(relocated['source_bookends'][0]['path'], '/synthetic/provider/weights.bin')
        self.assertEqual(relocated['command'][-1], '/synthetic/general/observations.npz')
        self.assertEqual(relocated['hashes'], bookends)
        self.assertEqual(bookends, {'/synthetic/provider/model.c': 'saved'})
        self.assertEqual(relocated['relative'], 'host/bias.bin')
        self.assertEqual(original, frozen)
        self.assertEqual(mappings, dict(mappings=[dict(source=old, destination=new) for old, new in capture.PATH_MAP]))

    def test_exact_prefix_and_sibling_boundary(self):
        self.assertEqual(capture.migrated('/historical/provider'), '/synthetic/provider')
        self.assertEqual(capture.migrated('/historical/providers/file'), '/synthetic/general/providers/file')
        self.assertEqual(capture.migrated('/historical-other/file'), '/historical-other/file')

    def test_no_mapping_has_no_implicit_bundle_destination(self):
        capture.PATH_MAP = []
        historical = dict(path='/historical/provider/weights.bin')
        self.assertEqual(capture.relocate_historical_metadata(historical), historical)
        self.assertEqual(capture.metadata_relocation(), dict(mappings=[]))

    def test_ambiguous_relocated_metadata_is_rejected(self):
        capture.PATH_MAP = [('/historical/a', '/synthetic/shared'),
                            ('/historical/b', '/synthetic/shared')]
        with self.assertRaisesRegex(ValueError, 'mapping collision'):
            capture.relocate_historical_metadata(['/historical/a/file', '/historical/b/file'])


if __name__ == '__main__':
    unittest.main()
