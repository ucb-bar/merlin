"""SIMS packaging carries the authenticated host execution profile to CMake."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from smol_full_phases import V1,V2
from smol_full_plan import panels
from smol_sims import package


def linker_fixture():
    return ('SECTIONS {\n  __stack_size = 128K;\n' + '}> DRAM\n' * 2
            + '  /* Section for initialized data */\n' + '}> DRAM\n' * 7
            + '  .bss : ALIGN (8)\n  PROVIDE(__stack_shift = 17);\n > ASSETS\n')


class SimsPackageTest(unittest.TestCase):
    def manifest(self,profile):
        result=dict(kernel_count=2715,kernels=panels())
        if profile==V2:
            result.update(attention_count=200,device_event_count=2915,device_events=profile.events(),
                attention_kernels=[dict(index=e['kernel'],phase=e['kernel'],before_call=e['before_call'])
                                   for e in profile.events() if e['kind']=='attention'])
        return result

    def bundle(self,root,contract,provenance=None):
        (root/'host').mkdir(parents=True)
        raw=(json.dumps(dict(contract,schema='atlas.smol_full_contract.v1'))+'\n').encode()
        (root/'host/smol-full-contract.json').write_bytes(raw)
        value=dict(contract if provenance is None else provenance,schema='atlas.full_bundle.v1',
                   workload='smol-full',contract='host/smol-full-contract.json',
                   contract_sha256=hashlib.sha256(raw).hexdigest())
        (root/'provenance.json').write_text(json.dumps(value))
        (root/'host/host.elf').write_bytes(b'synthetic packaging fixture')

    def test_v2_cmake_remains_byte_identical(self):
        text=(package.PACKAGE/'templates/CMakeLists.txt').read_text()
        expected=text.replace('smol_full_driver.c smol_vision_attention.c atlas_host.c',
            'smol_full_driver.c smol_attention.c smol_vision_attention.c atlas_host.c')
        expected=expected.replace('ATLAS_SMOL_LAUNCH_HOST_TIMING=1)',
            'ATLAS_SMOL_LAUNCH_HOST_TIMING=1 SMOL_ATTENTION_RUNTIME=1)')
        self.assertEqual(package.sims_cmake(text,V2),expected)

    def test_real_package_cmake_preserves_both_driver_profiles(self):
        for profile in (V1,V2):
            with self.subTest(profile=profile.name),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);bundle=root/'bundle';work=root/'build';out=root/'package'
                self.bundle(bundle,self.manifest(profile))
                platform=root/'platform_root';linker=platform/'platform/penguin26/penguin26-smolvla.ld'
                linker.parent.mkdir(parents=True);linker.write_text(linker_fixture())
                for name in ('source','support'): (work/name).mkdir(parents=True)
                (work/'source/smol_full_driver.c').write_bytes((ROOT/'runtime/smol_full_driver.c').read_bytes())
                (work/'source/smol_full_driver.h').write_bytes((ROOT/'runtime/smol_full_driver.h').read_bytes())
                (work/'main.c').write_text('/* prepared main */\n')
                (work/'assets.S').write_text('.incbin "/prepared/assets/blob.bin"\n')
                with patch.object(sys,'argv',['package.py','--bundle',str(bundle),'--build-work',str(work),'--out',str(out),'--platform-root',str(root/'platform_root')]):
                    package.main()
                text=(out/'platform_root/CMakeLists.txt').read_text()
                self.assertIn('SMOL_ATTENTION_RUNTIME=1',text)
                self.assertEqual('SMOL_FULL_LAUNCH_COUNT=2715' in text,profile==V1)
                self.assertEqual('SMOL_FULL_ATTENTION=0' in text,profile==V1)
                self.assertEqual((out/'source/smol_full_driver.c').read_bytes(),(ROOT/'runtime/smol_full_driver.c').read_bytes())

    def test_partial_or_disagreeing_profiles_fail_before_output_creation(self):
        v1,v2=self.manifest(V1),self.manifest(V2)
        partial=copy.deepcopy(v2);partial['attention_kernels'].pop()
        for contract,provenance in ((partial,partial),(v1,v2),(dict(v1,device_event_count=2915),v1)):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);bundle=root/'bundle';out=root/'package'
                self.bundle(bundle,contract,provenance)
                with patch.object(sys,'argv',['package.py','--bundle',str(bundle),'--build-work',str(root/'unused'),'--out',str(out),'--platform-root',str(root/'platform_root')]):
                    with self.assertRaises(ValueError): package.main()
                self.assertFalse(out.exists())

    def test_changed_contract_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.bundle(root,self.manifest(V1))
            with (root/'host/smol-full-contract.json').open('ab') as stream: stream.write(b' ')
            with self.assertRaisesRegex(ValueError,'identity'): package.bundle_profile(root)

    def test_package_adds_fusion_only_explicitly_and_preserves_low_window(self):
        template=(ROOT/'tools/smol_sims/templates/CMakeLists.txt').read_text()
        plain=package.sims_cmake(template,V2);enabled=package.sims_cmake(template,V2,'fc1-bias-gelu-fp8')
        self.assertNotIn('SMOL_FUSION_RUNTIME',plain);self.assertNotIn('smol_fusion.c',plain)
        self.assertIn('SMOL_FUSION_RUNTIME=1',enabled);self.assertEqual(enabled.count('smol_fusion.c'),1)
        linker=package.sims_linker(linker_fixture())
        self.assertIn('> ASSETS :assets',linker);self.assertIn('64K;',linker)
        self.assertIn('__stack_bottom = __stack_start;',linker)


if __name__=='__main__': unittest.main()
