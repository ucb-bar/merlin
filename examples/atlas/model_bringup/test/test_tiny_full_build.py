"""Full Tiny builder physical placement and source binding controls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).absolute().parents[1]
spec = importlib.util.spec_from_file_location('tiny_full_builder', ROOT/'tools/tiny_full/build_host.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PlacementTest(unittest.TestCase):
    def fixture(self, root):
        (root/'programs').mkdir()
        raw = struct.pack('<4I', 0x00100f13, 30 << 15 | 0x7f,
                          3 << 20 | 2 << 15 | 1 << 7 | 0x7b, 0x73)
        source = b'li x30, 1\ndma.config x30, ch0\ndma.load x1, x2, x3, ch0\necall\n'
        item = dict(dram_base=0x100000000,total_dram_bytes=4096)
        sums = {}
        for key, value, suffix in [('original',raw,'.bin'), ('source',source,'.S')]:
            digest = hashlib.sha256(value).hexdigest()
            name = 'programs/'+digest+suffix
            (root/name).write_bytes(value);sums[name] = digest;item[key] = name
        for name in ('kernels.json','provenance.json','contract.json'):
            value = json.dumps(dict(kernels=[item])).encode()
            (root/name).write_bytes(value);sums[name] = hashlib.sha256(value).hexdigest()
        (root/'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name,digest in sums.items()))
        return item

    def test_relocation_binds_programs_contract_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); old = self.fixture(root)
            builder.place_bundle(root,0x190000000,0x80000000,8 << 30)
            documents = [json.loads((root/name).read_bytes()) for name in
                         ('kernels.json','provenance.json','contract.json')]
            row = documents[0]['kernels'][0]
            self.assertEqual(row['dram_base'],0x190000000)
            self.assertNotEqual(row['original'],old['original'])
            self.assertNotEqual(row['source'],old['source'])
            self.assertEqual(documents[0]['kernels'],documents[2]['kernels'])
            self.assertEqual(documents[1]['placement'],documents[2]['placement'])
            self.assertEqual(documents[0]['contract_sha256'],hashlib.sha256((root/'contract.json').read_bytes()).hexdigest())
            for line in (root/'SHA256SUMS').read_text().splitlines():
                digest,name = line.split('  ',1)
                self.assertEqual(digest,hashlib.sha256((root/name).read_bytes()).hexdigest())

    def test_default_keeps_metadata_and_program_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);self.fixture(root)
            names = [line.split('  ',1)[1] for line in (root/'SHA256SUMS').read_text().splitlines()]
            names.append('SHA256SUMS')
            before = {name:(root/name).read_bytes() for name in names}
            builder.place_bundle(root,0x100000000,0x80000000,64 << 30)
            self.assertEqual(before,{name:(root/name).read_bytes() for name in names})

    def test_rejects_outside_dram_and_low32_wrap(self):
        for base,size in [(0x190000000,1 << 30),(0x1ffffffc0,8 << 30)]:
            with self.subTest(base=base),tempfile.TemporaryDirectory() as directory:
                root = Path(directory);self.fixture(root)
                with self.assertRaises(ValueError):builder.place_bundle(root,base,0x80000000,size)


if __name__ == '__main__':
    unittest.main()
