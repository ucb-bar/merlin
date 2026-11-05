"""Prepared fusion source, arena, and typed-view contracts; no numerical execution."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/host_iss')]
import kernel_library  # noqa: E402,F401
from atlas_program import assembly,dma_load_bytes
from chipyard.arena import relocate_assembly
from smol_fusion_program import stage_panel
from smol_fusion_fp8 import stage_fc1_fp8
import smol_fusion_assets as fusion
from export_smol_policy_kernels import split_initial
from full_bundle import KernelAssets


class FusionAssetTest(unittest.TestCase):
    def setUp(self):self.payloads={}

    def put(self,raw,suffix='.bin'):
        name='assets/'+hashlib.sha256(raw).hexdigest()+suffix
        self.payloads[name]=raw
        return name

    def tensor(self,raw,dtype,shape):return fusion.record(raw,self.put(raw),dtype=dtype,shape=shape)

    def panel(self,index,call,start,m,k,n,*,output='bf16',scale=-5,ops=('bias','gelu_tanh')):
        module='vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.'+('fc1' if call==5 else 'fc2')
        offset=m*k+k*n+256
        row=dict(index=index,call=call,module=module,logical_row_start=start,logical_mkn=[m,k,n],physical_mkn=[m,k,n],
            output_offset=offset,total_dram_bytes=offset+2*m*n+256,dram_bytes=offset+2*m*n+256,
            a_scale=0,b_scale=0,output_rescale_exponent=0,recipe='traffic-reuse-v1',dram_base=0x90000000,
            requested_dma=dict(load_bytes=dma_load_bytes(m,k,n,traffic_reuse=True),store_bytes=2*m*n))
        initial=bytes(offset-256)+b'\xa5'*256+b'\xcd'*(2*m*n)+b'Z'*256
        bias=bytes(2*n) if ops else None
        if output=='fp8':staged=stage_fc1_fp8(row,initial,bias_bf16=bias,output_scale_exponent=scale)
        elif ops:staged=stage_panel(row,initial,operations=ops,bias_bf16=bias)
        else:staged=stage_panel(row,initial)
        row.update(requested_dma=staged.metadata['requested_dma'],total_dram_bytes=len(staged.arena),dram_bytes=len(staged.arena),
            initial_sha256=fusion.digest(staged.arena),source=self.put(staged.source.encode(),'.S'),original=self.put(struct.pack('<I',0x73)),
            activation=self.put(bytes(m*k)),weights=self.put(bytes(k*n)),expected=self.put(bytes(staged.metadata['output_bytes'])))
        if ops:
            meta=dict(staged.metadata)
            extra=staged.arena[meta['extra_offset']:meta['extra_offset']+meta['extra_bytes']]
            meta['extra']=fusion.record(extra,self.put(extra))
            if bias is not None:meta['bias']=self.tensor(bias,'uint16',[n])
            row['fusion']=meta
        return row,staged

    def test_bf16_and_fp8_regenerate_exact_source_arena_and_dma(self):
        for output,flags in [('bf16',3),('fp8',11)]:
            with self.subTest(output=output):
                row,staged=self.panel(0,5,0,128,768,3072,output=output)
                actual=fusion.validate_program(row,self.payloads.__getitem__)
                self.assertEqual(actual.source,staged.source);self.assertEqual(actual.arena,staged.arena)
                self.assertEqual(fusion.runtime_fields(row)[0],flags)
                a,w=split_initial(staged.arena,self.payloads[row['expected']],row)
                self.assertEqual((len(a),len(w)),(128*768,768*3072))

    def test_arbitrary_source_and_resigned_dma_refused(self):
        row,staged=self.panel(0,5,0,128,768,3072,output='fp8')
        changed=copy.deepcopy(row);changed['source']=self.put(b'# changed\n'+staged.source.encode(),'.S')
        with self.assertRaisesRegex(ValueError,'source differs'):fusion.validate_program(changed,self.payloads.__getitem__)
        changed=copy.deepcopy(row);changed['requested_dma']['load_bytes']+=32
        changed['fusion']['requested_dma']=dict(changed['requested_dma'])
        with self.assertRaisesRegex(ValueError,'metadata differs'):fusion.validate_program(changed,self.payloads.__getitem__)

    def test_extra_bytes_cannot_replace_actual_bias_parameter(self):
        row,staged=self.panel(0,5,0,128,768,3072,output='fp8')
        changed=copy.deepcopy(row);meta=changed['fusion'];raw=bytearray(staged.arena)
        raw[meta['extra_offset']]^=1;extra=bytes(raw[meta['extra_offset']:meta['extra_offset']+meta['extra_bytes']])
        meta['extra']=fusion.record(extra,self.put(extra));meta['extra_sha256']=fusion.digest(extra)
        changed['initial_sha256']=meta['initial_sha256']=fusion.digest(raw)
        with self.assertRaisesRegex(ValueError,'staging differs'):fusion.validate_program(changed,self.payloads.__getitem__)

    def fp8_bundle(self):
        rows=[]
        for i in range(8):rows.append(self.panel(i,5,128*i,128,768,3072,output='fp8')[0])
        for i in range(8):
            row,_=self.panel(i+8,6,128*i,128,3072,768,ops=())
            row['a_scale']=-5;row['output_rescale_exponent']=-5
            # The base producer used exponent0; regenerate this ordinary consumer.
            source=relocate_assembly(assembly(128,3072,768,row['output_offset'],-5,traffic_reuse=True),0x100000000,0x90000000)
            row['source']=self.put(source.encode(),'.S');rows.append(row)
        packed=bytes(1024*3072);diag=bytes(2*1024*3072)
        views=[]
        for index,side in [(5,'output'),(6,'input')]:
            module='vlm_with_expert.vlm.model.vision_model.encoder.layers.0.mlp.'+('fc1' if index==5 else 'fc2')
            views.append(dict(index=index,module=module,**{
                side+'_fp8':self.tensor(packed,'uint8',[32,96,32,32]),
                side+'_diagnostic':self.tensor(diag,'torch.bfloat16',[1,1024,3072]),
                side+'_decode_exponent':-5,side+'_source_format':'fp8_tiles',side+'_diagnostic_only':True}))
        return dict(schema='atlas.full_kernel_assets.v1',workload='smol-full',kernel_count=16,kernels=rows,
            fusion_set='fc1-bias-gelu-fp8',call_views=views,placement=dict(arena_base=0x90000000,dram_base=0x80000000,dram_bytes=0x80000000))

    def test_typed_fp8_is_actual_input_and_output_not_diagnostic(self):
        data=self.fp8_bundle();fusion.validate_views(data,self.payloads.__getitem__)
        changed=copy.deepcopy(data);changed['call_views'][1]['input_fp8']=changed['call_views'][1]['input_diagnostic']
        with self.assertRaises(ValueError):fusion.validate_views(changed,self.payloads.__getitem__)
        changed=copy.deepcopy(data);changed['kernels'][8]['activation']=self.put(b'\x01'+bytes(128*3072-1))
        with self.assertRaisesRegex(ValueError,'A panels differ'):fusion.validate_views(changed,self.payloads.__getitem__)
        changed=copy.deepcopy(data);changed['call_views'][1]['input_decode_exponent']=-4
        with self.assertRaises(ValueError):fusion.validate_views(changed,self.payloads.__getitem__)
        changed=copy.deepcopy(data);changed['fusion_set']='none'
        with self.assertRaises(ValueError):fusion.validate_views(changed,self.payloads.__getitem__)

    def test_kernel_assets_reconstruct_compact_arenas_and_extras(self):
        data=self.fp8_bundle()
        temporary=tempfile.TemporaryDirectory(prefix='fusion-assets-');self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)
        files=dict(self.payloads);files['kernels.json']=(json.dumps(data,indent=2)+'\n').encode()
        for name,raw in files.items():
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        (root/'SHA256SUMS').write_text(''.join(fusion.digest(raw)+'  '+name+'\n' for name,raw in sorted(files.items())))
        bundle=KernelAssets(root,'smol-full')
        initial=bundle.projection_initial_dram(0)
        self.assertEqual(len(initial),data['kernels'][0]['total_dram_bytes'])
        self.assertEqual(fusion.digest(initial),data['kernels'][0]['initial_sha256'])
        self.assertEqual(data['kernels'][0]['requested_dma']['store_bytes'],128*3072)



if __name__=='__main__':unittest.main()
