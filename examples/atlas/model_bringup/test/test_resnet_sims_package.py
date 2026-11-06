"""ResNet50 host packaging keeps default and fast tables aligned and attaches only its board-support target."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from resnet_sims import build, package


class ResnetSimsPackageTest(unittest.TestCase):
    def model(self,root):
        def blob(name,raw):
            (root/name).write_bytes(raw)
            return name
        launch=dict(index=0,program=blob('program.bin',bytes(8)+bytes.fromhex('73000000')),source=blob('program.S',b'ecall\n'),
                    initial=blob('initial.bin',bytes(4096)),expected=blob('expected.bin',bytes(1024)),arena_bytes=4096,
                    output_bytes=1024,output_offset=2048,input_tensor='input',output_tensor='out',m=32,k=32,n=32)
        model=dict(arena_base=0x90000000,input=dict(path=blob('input.bin',bytes(48)),shape=[3,2,2],tensor='input'),
                   tensors=[dict(id='out',dtype='fp8_nhwc',channels=32,h=1,w=1)],fc_bias=blob('bias.bin',bytes(8)),
                   expected_logits=blob('logits.bin',bytes(8)),launches=[launch],nodes=[dict(op='launch',launch=0)])
        (root/'model.json').write_text(json.dumps(model))
        return root/'model.json'

    def test_fast_tables_mirror_default_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'package'
            package.package(self.model(root),out)
            text=(out/'assets.c').read_text()
            self.assertIn('{m2_fast_program_0,12,m2_asset_004,',text)
            self.assertIn('{m2_asset_003,12,m2_asset_004,',text)
            self.assertIn('const m2_fast_launch m2_fast_launches[] = {\n};',text)
            self.assertIn('.incbin "fast-program-00.bin"\n#endif',(out/'images.S').read_text())
            self.assertIn('M2_FAST=1',(out/'CMakeLists.txt').read_text())
            self.assertEqual(json.loads((out/'package.json').read_text())['initial_symbols'],['m2_asset_004'])
            with self.assertRaisesRegex(ValueError,'already exists'):
                package.package(root/'model.json',out)

    def test_binding_rewrites_fast_sizes_and_bases(self):
        try:
            from resnet_sims import fast
        except ImportError as error:
            self.skipTest(f'fast binding dependencies unavailable: {error}')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'package'
            package.package(self.model(root),out)
            (out/'data/fast-program-00.bin').write_bytes(bytes(20))
            fast.write_tables(out,[dict(initial_symbol='m2_asset_004',weight_offset=1024,weight_bytes=1024,weight_base=0x80001400)])
            text=(out/'assets.c').read_text()
            self.assertIn('{m2_fast_program_0,20,m2_asset_004,',text)
            self.assertIn('{m2_asset_003,12,m2_asset_004,',text)
            self.assertIn('const m2_fast_launch m2_fast_launches[] = {\n  {m2_asset_004+1024,1024,1024,0x80001400UL},\n};',text)

    def test_platform_root_attaches_resnet_once_and_drops_delivery_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'platform_root'
            (source/'penguin26-tests/atlas_delivery').mkdir(parents=True)
            (source/'penguin26-tests/atlas_delivery/CMakeLists.txt').write_text('include(/machine/local/smoke.cmake)\n')
            (source/'penguin26-tests/CMakeLists.txt').write_text('add_subdirectory(hello_world)\n')
            (source/'build-sims').mkdir()
            first=build.prepare_platform_root(source,root/'first')
            tests=(first/'penguin26-tests/CMakeLists.txt').read_text()
            self.assertEqual(tests.count('PENGUIN26_TESTS_ATLAS_RESNET50_DIR'),2)
            self.assertNotIn('/machine/local',(first/'penguin26-tests/atlas_delivery/CMakeLists.txt').read_text())
            self.assertFalse((first/'build-sims').exists())
            second=build.prepare_platform_root(first,root/'second')
            self.assertEqual((second/'penguin26-tests/CMakeLists.txt').read_text(),tests)


if __name__=='__main__':
    unittest.main()
