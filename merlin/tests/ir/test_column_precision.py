"""Independent threshold controls for finite FP8 conversion and scale axes."""
from dataclasses import replace
import unittest
import numpy as np

from merlin.semantic_compiler.precision_values import FloatBoundarySpec
from merlin.semantic_compiler.precision_codegen import finite_f32_codebook


class ColumnPrecisionTests(unittest.TestCase):
    def test_normal_significand_underflow_tie(self):
        original=FloatBoundarySpec('control','fp8_e4m3','encode',max_normal_exponent_code=15,
            flush_below_normal=True,canonicalize_negative_zero=True)
        selected=replace(original,underflow_rounding='round-normal-significand-then-flush')
        # Normal-significand rounding promotes the 31/32 threshold to 2^-6;
        # the old source-domain flush excludes every value below 2^-6.
        below=np.nextafter(np.float32(31/2048),np.float32(0))
        values=np.array([[below,31/2048,-31/2048,2**-6,448,-448,480,-480]],dtype=np.float32)
        self.assertEqual(selected.encode(values).tolist(),[[0,8,136,8,126,254,126,254]])
        self.assertEqual(original.encode(values).tolist(),[[0,0,0,8,126,254,126,254]])
        self.assertNotIn('underflow_rounding',original.record())
        gate=finite_f32_codebook(selected).zero_gate_bits
        self.assertEqual(gate,np.array([31/2048],dtype=np.float32).view(np.uint32)[0])

    def test_scale_axis_and_restore(self):
        base=FloatBoundarySpec('scale','fp8_e4m3','encode',max_normal_exponent_code=15,
            flush_below_normal=True,canonicalize_negative_zero=True,
            column_scale_exponents=((0,2,1),(2,4,2)))
        values=np.ones((3,4),dtype=np.float32)
        expected=np.array([[64,64,72,72]]*3,dtype=np.uint8)
        np.testing.assert_array_equal(base.encode(values),expected)
        np.testing.assert_array_equal(replace(base,scale_axis=0).encode(values.T),expected.T)
        restored=FloatBoundarySpec('restore','bf16','decode',column_scale_exponents=((0,2,-2),(2,4,-3)))
        np.testing.assert_array_equal(restored.decode(np.full((1,4),0x3f80,dtype=np.uint16)),[[.25,.25,.125,.125]])
        with self.assertRaisesRegex(ValueError,'partition'):
            base.encode(np.ones((3,5),dtype=np.float32))
        with self.assertRaisesRegex(ValueError,'partition'):
            replace(base,column_scale_exponents=((0,2,1),(3,4,2)))


if __name__=='__main__':
    unittest.main()
