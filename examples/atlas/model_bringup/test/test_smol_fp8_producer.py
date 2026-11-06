"""FP8 producer layout, diagnostic decode, and read-only model observations."""
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np

sys.path[:0] = [str(Path(__file__).absolute().parents[1]/'tools')]
import kernel_library  # noqa: E402,F401
from atlas_epilogue import Epilogue
from atlas_program import dma_load_bytes
from smol_fusion_fp8 import stage_fc1_fp8
from smol_capture.fusion import operations_for_call, fp8_diagnostic


class FP8ProducerTests(unittest.TestCase):
    def test_first_set_selects_only_actual_fc1(self):
        for site in (5, 6, 7):
            for role in range(8):
                ops = operations_for_call('fc1-bias-gelu-fp8', SimpleNamespace(site=site, role=role))
                self.assertEqual(ops, ('bias', 'gelu_tanh') if (site, role)==(7, 5) else ())

    def test_fp8_output_and_extras_geometry(self):
        m,k,n = 128,768,3072
        offset=m*k+k*n+256
        initial=bytes(offset-256)+b'\xa5'*256+b'\xcd'*(2*m*n)+b'Z'*256
        row=dict(module='x.vision_model.encoder.layers.0.mlp.fc1',physical_mkn=[m,k,n],
                 output_offset=offset,output_rescale_exponent=-10,a_scale=-5,b_scale=-5,
                 program_policy='traffic-reuse-v1',requested_dma=dict(
                     load_bytes=dma_load_bytes(m,k,n,traffic_reuse=True),store_bytes=2*m*n))
        for scale in (-7,-6,-5,-4,-3,-2):
            case=stage_fc1_fp8(row,initial,bias_bf16=bytes(n*2),output_scale_exponent=scale)
            meta=case.metadata
            self.assertEqual(meta['output_bytes'],m*n)
            self.assertEqual(meta['requested_dma']['store_bytes'],m*n)
            self.assertEqual(meta['extra_offset'],offset+m*n+256)
            self.assertEqual(case.arena[:offset],initial[:offset])
            self.assertEqual(case.arena[offset:offset+m*n],b'\xcd'*(m*n))
            self.assertEqual(case.arena[offset+m*n:meta['extra_offset']],b'Z'*256)
            self.assertEqual(case.arena[-256:],b'Z'*256)
            self.assertEqual(meta['output_dtype'],'fp8')
            self.assertIn(f'seli e0, {127-scale}',case.source)
            self.assertEqual(initial[-256:],b'Z'*256)

    def test_diagnostic_untile_and_exponent(self):
        matrix=np.arange(64*64,dtype=np.uint32).reshape(64,64)%112+8
        matrix=matrix.astype(np.uint8)
        packed=matrix.reshape(2,32,2,32).transpose(0,2,1,3).tobytes()
        result=np.frombuffer(fp8_diagnostic(packed,64,64,-5),'<u2')
        expected=(1+(matrix&7)/8)*np.exp2(((matrix>>3)&15).astype(np.int32)-7-5)
        expected=np.asarray(expected,dtype='<f4').view('<u4')>>16
        np.testing.assert_array_equal(result.reshape(64,64),expected)
        for raw in (bytes([0x7f])*1024,bytes([0xff])*1024,bytes([1])*1024):
            with self.assertRaises(ValueError):fp8_diagnostic(raw,32,32,0)

    @unittest.skipUnless(os.environ.get('ATLAS_NPU_MODEL_ROOT'),'original model fixture required')
    def test_read_only_observer_preserves_actual_original_execution(self):
        from smol_capture import model
        from smol_capture.stats import observe_fp8
        epilogue=Epilogue(32,32,0x90000000,output_dtype='fp8',scratch_registers=(10,11))
        source='li x3, 327680\nli x9, 327936\nli x20, 2415919104\nvli.all m2, 17402\ndelay 66\n'
        source+='\n'.join(epilogue.store(20))+'\necall\n'
        initial=b'\xcd'*1024+b'Z'*256
        ordinary=model.execute(source,initial,0,1024)
        original_assemble=model.assemble
        with observe_fp8(model,0,1024) as stats:
            observed=model.execute(source,initial,0,1024)
        self.assertIs(model.assemble,original_assemble)
        self.assertEqual(observed.arena,ordinary.arena)
        self.assertEqual(observed.program_sha256,ordinary.program_sha256)
        self.assertEqual(observed.observed,ordinary.observed)
        report=stats.report(observed.output)
        self.assertEqual(report['physical_values'],1024)
        self.assertEqual(report['preclip_saturation_count'],1024)
        self.assertEqual(report['preclip_nonfinite_count'],0)
        self.assertEqual(report['emitted_nan_count'],0)
        self.assertEqual(observed.output,bytes([0x7e])*1024)
        with self.assertRaisesRegex(RuntimeError,'control'):
            with observe_fp8(model,0,1024):raise RuntimeError('control')
        self.assertIs(model.assemble,original_assemble)


if __name__=='__main__':unittest.main()
