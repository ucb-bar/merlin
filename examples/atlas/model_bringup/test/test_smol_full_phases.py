"""Complete captured phase coverage and immutable live-inventory controls."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).absolute().parents[1]/'tools'),
              str(Path(__file__).absolute().parents[1]/'tools/host_iss')]
import smol_full_phases as phases
from smol_full_plan import calls, panels
from elf_image import Image


class PhaseTests(unittest.TestCase):
    def test_complete_plan_and_folded_boundary_geometry(self):
        ranges=phases.segments()
        self.assertEqual(len(ranges),200)
        self.assertEqual([sum(x['kind']==k for x in ranges) for k in ('vision','prefix','expert')],[24,16,160])
        self.assertEqual([i for r in ranges for i in range(r['launch_start'],r['launch_end'])],list(range(2715)))
        self.assertEqual([i for r in ranges for i in range(r['call_start'],r['call_end'])],list(range(1421)))
        self.assertEqual((ranges[0]['launch_start'],ranges[0]['launch_end']),(0,56))
        self.assertEqual((ranges[23]['launch_end'],ranges[24]['launch_start']),(1170,1170))
        self.assertEqual((ranges[-1]['launch_start'],ranges[-1]['launch_end']),(2705,2715))
        for r in ranges:
            actual=[p['index'] for p in panels() if r['call_start']<=p['call']<r['call_end']]
            self.assertEqual(actual,list(range(r['launch_start'],r['launch_end'])))

    def test_missing_tail_and_reordered_call_fail(self):
        changed=calls()[:-1]
        with patch.object(phases,'calls',return_value=changed),self.assertRaises(ValueError): phases.segments()
        changed=calls();changed[7],changed[13]=changed[13],changed[7]
        with patch.object(phases,'calls',return_value=changed),self.assertRaises(ValueError): phases.segments()

    def test_current_observations_and_canonical_history(self):
        values=[0]*2715
        values[56:104]=[7]*48
        self.assertEqual(phases.check_observations(values,56,104),[7]*48)
        for index,value in ((0,1),(104,1),(56,0),(56,True),(56,-1)):
            changed=values.copy();changed[index]=value
            with self.subTest(index=index,value=value),self.assertRaises(ValueError):
                phases.check_observations(changed,56,104)

    def test_attention_profile_plan_and_observations(self):
        plan=phases.segments(phases.V2)
        self.assertEqual((plan[0]['launch_start'],plan[0]['launch_end']),(0,57))
        self.assertEqual((plan[23]['launch_end'],plan[24]['launch_start']),(1194,1194))
        self.assertEqual((plan[-1]['launch_start'],plan[-1]['launch_end']),(2904,2915))
        self.assertEqual([i for p in plan for i in range(p['launch_start'],p['launch_end'])],list(range(2915)))
        for p in plan:
            events=phases.V2.events()[p['launch_start']:p['launch_end']]
            self.assertEqual(sum(e['kind']=='attention' for e in events),1)
            self.assertEqual([e['kernel'] for e in events if e['kind']=='projection'],
                             list(range(p['panel_start'],p['panel_end'])))
        values=[0]*2915;values[57:106]=[7]*49
        self.assertEqual(phases.check_observations(values,57,106,phases.V2),[7]*49)
        with self.assertRaises(ValueError): phases.check_observations(values,57,106,phases.V1)

    def test_manifest_selects_only_complete_profiles(self):
        v1=dict(kernel_count=2715,kernels=panels())
        attention=[dict(index=e['kernel'],phase=e['kernel'],before_call=e['before_call'])
                   for e in phases.V2.events() if e['kind']=='attention']
        v2=dict(v1,attention_kernels=attention,attention_count=200,device_event_count=2915,
                device_events=phases.V2.events())
        self.assertEqual(phases.execution_profile(v1),phases.V1)
        self.assertEqual(phases.execution_profile(v2),phases.V2)
        for bad in (dict(v1,device_event_count=2915),dict(v1,attention_count=200),
                    dict(v2,attention_kernels=attention[:-1]),dict(v2,device_events=list(reversed(v2['device_events']))),
                    dict(v2,execution_profile='v1'),dict(v1,kernels=panels()[:-1])):
            with self.subTest(bad=list(bad)),self.assertRaises(ValueError): phases.execution_profile(bad)

    def test_live_cycle_extent_follows_manifest(self):
        from smol_full_plan import reference_geometry
        references=[]
        for name,(shape,dtype) in reference_geometry().items():
            size={'float32':4,'uint16':2,'bool':1}[dtype]
            for dim in shape: size*=dim
            references.append(dict(name=name,actual_symbol='actual_'+name,bytes=size))
        for profile in (phases.V1,phases.V2):
            data=dict(kernel_count=2715,kernels=panels(),reference_buffers=references)
            if profile==phases.V2:
                data.update(attention_count=200,device_event_count=2915,device_events=profile.events(),
                    attention_kernels=[dict(index=e['kernel'],phase=e['kernel'],before_call=e['before_call'])
                                       for e in profile.events() if e['kind']=='attention'])
            symbols={name:dict(size=size) for name,size in phases.extra_live(profile).items()}
            symbols.update((r['actual_symbol'],dict(size=r['bytes'])) for r in references)
            self.assertEqual(phases.live_regions(data,SimpleNamespace(symbols=symbols))['smol_full_cycles'],profile.cycle_bytes)
            symbols['smol_full_cycles']['size']=phases.V2.cycle_bytes if profile==phases.V1 else phases.V1.cycle_bytes
            with self.assertRaises(ValueError): phases.live_regions(data,SimpleNamespace(symbols=symbols))

    def test_emitter_carries_profile_to_callback_cursors_and_completion(self):
        from smol_full_driver_emit import emit
        parameters={'observation.'+name:dict(symbol='param_'+name) for name in
                    ('images','state','noise','image_masks','language_mask','language_tokens')}
        for profile in (phases.V1,phases.V2):
            inventory=phases.extra_live(profile)
            regions={name:dict(file=name,bytes=size,sha256='unused') for name,size in inventory.items()}
            evidence=dict(status='PASS_NATIVE_PHASE_REFERENCE',mode='phased',
                          initial_boundary=dict(regions=regions),boundaries=[dict(regions=regions)]*200)
            def read(path,expected=None):
                return json.dumps(evidence).encode() if path.name=='reference.json' else bytes(inventory[path.name])
            source,data=emit('',parameters,[],32000,Path('reference.json'),read,lambda *args:None,profile=profile)
            self.assertNotIn('smol_fusion_configure',source)
            self.assertEqual(data['inventory']['smol_full_cycles'],profile.cycle_bytes)
            self.assertIn('expected='+str(profile.launch_count)+',calls=1421',source)
            self.assertIn('i<'+str(profile.launch_count)+';++i)if(!smol_full_cycles',source)
            if profile.attention_count:
                self.assertIn('smol_attention_configure(smol_policy_raw_attention,&smol_full_raw)',source)
                self.assertIn('attention_cursor=first;smol_full_raw.panel_cursor=p->launch_start-first',source)
                fused,_=emit('',parameters,[],32000,Path('reference.json'),read,lambda *args:None,
                    profile=profile,fusion_set='fc1-bias-gelu-fp8')
                guard='if((smol_full_segment_request==UINT64_MAX||smol_full_segment_request<24)&&smol_fusion_configure('
                self.assertEqual(fused.count('smol_fusion_configure('),1)
                self.assertIn(guard,fused)
                self.assertLess(fused.index('smol_timing_start_instret=smol_timing_rdinstret()'),fused.index(guard))
                self.assertLess(fused.index('if(atlas_host_init('),fused.index(guard))
                self.assertLess(fused.index(guard),fused.index('smol_full_segment_ready();'))
                self.assertLess(fused.index(guard),fused.index('smol_full_raw.call_cursor=p->call_start'))
            else:
                self.assertIn('smol_attention_configure(0,0)',source)
                self.assertIn('attention_cursor=0;smol_full_raw.panel_cursor=p->launch_start;',source)

    @unittest.skipUnless(os.environ.get('SMOL_PHASE_BUNDLE'),'actual baseline ELF fixture required')
    def test_actual_full_elf_live_buffers_and_omission_controls(self):
        root=Path(os.environ['SMOL_PHASE_BUNDLE'])
        data=json.loads((root/'host/smol-full-contract.json').read_text())
        image=Image((root/'host/host.elf').read_bytes())
        image.symbols['smol_full_phase_state']=dict(size=40,address=0,section=0)
        live=phases.live_regions(data,image)
        self.assertEqual(len(live),56)
        self.assertEqual(live['smol_full_actual_prefix_key_cache'],1812480)
        self.assertIn('smol_b_output',live)
        scratch=phases.scratch_regions(image,live)
        self.assertNotIn('smol_b_output',scratch)
        self.assertIn('smol_g_positions',scratch)
        self.assertIn('smol_full_input_copy',scratch)
        for mutate in ('reference','size','symbol'):
            wrong=copy.deepcopy(data);symbols=copy.deepcopy(image.symbols)
            if mutate=='reference': wrong['reference_buffers'].pop()
            elif mutate=='size': wrong['reference_buffers'][0]['bytes']+=1
            else: symbols.pop('smol_b_output')
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                phases.live_regions(wrong,SimpleNamespace(symbols=symbols))


if __name__=='__main__': unittest.main()
