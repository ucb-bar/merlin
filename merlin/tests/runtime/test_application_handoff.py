"""Keep image reuse separate from source-call device launch multiplicity."""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from merlin.compile.application_handoff import application_handoff
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType


class ApplicationHandoffLaunchCountTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        tensor = TensorType((1,), 'f32', 'raw-bit-pattern')
        self.source = KernelRequest((SemanticNode('x', 'input', (), tensor, effect='input'),
            SemanticNode('y', 'identity', ('x',), tensor)), ('y',), ('external',), (('x', 'external'),), 'test-target')
        self.interface = dict(source='app.c', header='app.h', entry='app_run', inputs=['x'])
        self.files = {'c-package/app.c': b'int app_run(void){return 0;}\n', 'c-package/app.h': b'int app_run(void);\n'}
        self.binary = b'\x01\0\0\0'
        self.digest = sha256(self.binary).hexdigest()
        folder = self.root/'program'
        folder.mkdir()
        (folder/'program.bin').write_bytes(self.binary)
        (folder/'manifest.json').write_text(json.dumps({'binary_sha256': self.digest}))
        (folder/'execution_plan.json').write_text(json.dumps({'schema': 'test', 'streaming': {'launch_count': 9}}))

    def tearDown(self):
        self.temp.cleanup()

    def emit(self, regions):
        result = application_handoff(self.source, self.interface, regions, self.files, self.root)
        return json.loads(result['handoff/abi.json'])

    def region(self, name, count=None):
        description = {} if count is None else {'launches_per_call': count}
        return dict(id=name, placement='accelerator', native_programs=['program'], description=description)

    def test_shared_image_keeps_distinct_per_call_counts_and_unused_host_call(self):
        host = dict(id='host0', placement='host', native_programs=[], description={})
        abi = self.emit([self.region('short', 2), host, self.region('long', 7)])
        self.assertEqual(len(abi['programs']), 1)
        self.assertEqual(abi['calls'][0]['launches_per_call'], 2)
        self.assertEqual(abi['calls'][2]['launches_per_call'], 7)
        self.assertEqual(abi['calls'][0]['images'], [self.digest])
        self.assertNotIn('launches_per_call', abi['programs'][0])
        self.assertNotIn('launches_per_call', abi['calls'][1])
        self.assertEqual(abi['calls'][1]['images'], [])

    def test_bad_explicit_counts_and_ambiguous_or_host_images_refuse(self):
        for count in (0, -1, True, 1.5, '2', None):
            region = self.region('bad')
            region['description']['launches_per_call'] = count
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'positive integer'):
                self.emit([region])
        for region in (dict(id='host', placement='host', native_programs=[], description={'launches_per_call': 2}),
                       dict(id='two', placement='accelerator', native_programs=['program', 'program'], description={'launches_per_call': 2})):
            with self.subTest(region=region['id']), self.assertRaisesRegex(ValueError, 'exactly one device image'):
                self.emit([region])

    def test_absent_count_preserves_existing_streaming_image_metadata(self):
        abi = self.emit([self.region('old')])
        self.assertNotIn('launches_per_call', abi['calls'][0])
        self.assertEqual(abi['programs'][0]['execution']['streaming']['launch_count'], 9)


if __name__ == '__main__':
    unittest.main()
