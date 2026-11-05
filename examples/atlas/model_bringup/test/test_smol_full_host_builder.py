"""Compile actual generated host declarations and verify bounded placement/progress."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import build_smol_full_host as builder
from smol_full_plan import panels, reference_geometry
from chipyard.result_transport import write_config


class BuilderTest(unittest.TestCase):
    def test_progress_covers_each_layer_and_step(self):
        progress = builder.progress_launches(dict(kernels=panels()))
        self.assertEqual(len(progress), 212)
        self.assertEqual(progress[-1], 2715)
        self.assertEqual(progress, sorted(set(progress)))
        self.assertIn(585, progress)
        self.assertIn(1170, progress)
        self.assertIn(1395, progress)

    def test_arena_segment_relocated_without_flattening(self):
        template = (ROOT/'runtime/host_templates/smol_link.ld').read_text()
        result = builder.full_link(template, 12706304, 0x90000000)
        self.assertIn('dram PT_LOAD FLAGS(6);', result)
        self.assertIn('.atlas_dram 2415919104', result)
        self.assertIn('. += 12706304;', result)
        self.assertIn('__stack_top < 2415919104', result)
        moved = builder.full_link(template, 12706304, 0x90000000, rodata_after_arena=True, dram_end=0x100000000)
        self.assertIn('. = MAX(., 2428625408);', moved)
        self.assertIn('__stack_top <= 4294967296', moved)
        with self.assertRaises(ValueError):
            builder.full_link(template, 128, 0xffffffc0)

    def test_explicit_metadata_mapping_rejects_aliases_and_preserves_original(self):
        import kernel_library  # noqa: E402,F401
        from atlas_path_guard import map_metadata_paths
        original = {'path':'/old/a', 'inputs':{'/old/b':'digest'}}
        pairs = builder.relocation_pairs(['/old=/scratch2/reednicolas/oct27-fm/smol'])
        result = map_metadata_paths(original,builder.relocation_mapper(pairs))
        self.assertEqual(original['path'],'/old/a')
        self.assertEqual(result['path'],'/scratch2/reednicolas/oct27-fm/smol/a')
        with self.assertRaises(ValueError):
            builder.relocation_pairs(['/old=/scratch2/reednicolas/oct27-fm/smol',
                                      '/old/sub=/scratch2/reednicolas/oct27-fm/smol'])
        with self.assertRaises(ValueError):
            map_metadata_paths({'a':'/old/a','b':'/other/a'},
                               builder.relocation_mapper([(Path('/old'),Path('/new')),
                                                          (Path('/other'),Path('/new'))]))

    def test_quoted_header_closure_rejects_conflicting_sources(self):
        import shutil
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, first, second = [root/name for name in ('target','first','second')]
            for path in (target,first,second): path.mkdir()
            (target/'unit.c').write_text('#include "top.h"\n')
            (first/'top.h').write_text('#include "leaf.h"\n')
            (first/'leaf.h').write_text('typedef int leaf;\n')
            copy = lambda source,destination: (shutil.copyfile(source,destination),destination)[1]
            builder.copy_header_closure([target/'unit.c'],target,(first,second),copy,Path.read_bytes)
            self.assertTrue((target/'leaf.h').exists())
            (target/'top.h').unlink()
            (target/'leaf.h').unlink()
            (second/'top.h').write_text('typedef float changed;\n')
            with self.assertRaisesRegex(ValueError,'conflicting'):
                builder.copy_header_closure([target/'unit.c'],target,(first,second),copy,Path.read_bytes)

    @unittest.skipUnless(os.environ.get('SMOL_MERLIN'), 'Smol C graph sources are not configured')
    def test_generated_real_host_has_current_c_abi_and_uart_closure(self):
        names = ['vision.position','gelu','cosine','sine','language.embedding','timestep','time.embedding']
        names += [f'vision.{layer}.norm{n}.{field}' for layer in range(12) for n in (1,2)
                  for field in ('weight','bias')]
        names += ['vision.post.weight','vision.post.bias']
        names += [f'{family}.{layer}.{field}' for family in ('prefix','expert') for layer in range(16)
                  for field in ('input','post')]
        names += ['prefix.final','expert.final']
        names += ['observation.'+name for name in ('images','state','noise','image_masks',
                                                  'language_mask','language_tokens')]
        parameters = {name:dict(symbol=f'param_{i}') for i,name in enumerate(names)}
        references = []
        for name,(shape,dtype) in reference_geometry().items():
            elements = 1
            for extent in shape: elements *= extent
            references.append(dict(name=name,actual_symbol='actual_'+name,expected_symbol='expected_'+name,
                bytes=elements*dict(float32=4,uint16=2,bool=1)[dtype],dtype=dtype))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            write_config(out,argparse.Namespace(uart_kind='ns16550',uart_base=0x10000000,uart_divisor=None))
            for attention_count in (0,200):
                source = builder.make_globals(2715,parameters,references,[],[],
                    dict(dram=12706304,vocabulary=32000,arena=0x100000000),
                    [dict(c='{0}',symbols=[])]*attention_count)
                self.assertIn('smol_full_vision_frame',source)
                self.assertIn('smol_full_policy_frame',source)
                self.assertIn('smol_full_cycles['+str(2715+attention_count)+']',source)
                self.assertIn('.attention_panels='+('smol_full_attention_panels' if attention_count else '0'),source)
                (out/'main.c').write_text(source)
                flags=['cc','-std=c11','-Wall','-Wextra','-Werror','-fsyntax-only',
                    '-DSMOL_FULL_LAUNCH_COUNT='+str(2715+attention_count),
                    '-DSMOL_FULL_ATTENTION='+str(int(bool(attention_count))),
                    '-I'+str(out),'-I'+str(ROOT/'runtime'),'-I'+os.environ['SMOL_MERLIN']]
                subprocess.run([*flags,str(out/'main.c')],check=True)
                subprocess.run([*flags,str(ROOT/'runtime/smol_full_driver.c')],check=True)
                if attention_count:
                    fused=builder.make_globals(2715,parameters,references,[],[],
                        dict(dram=12706304,vocabulary=32000,arena=0x90000000),
                        [dict(c='{0}',symbols=[])]*attention_count,
                        [dict(c='{0,0,0,0,0}',symbols=[])]*2715)
                    self.assertIn('const smol_fusion_panel smol_full_fusion_panels[2715]',fused)
                    self.assertNotIn('smol_full_fusion_panels',source)
                    (out/'fused.c').write_text(fused)
                    subprocess.run([*flags,'-DSMOL_FUSION_RUNTIME=1',str(out/'fused.c')],check=True)


if __name__ == '__main__':
    unittest.main()
