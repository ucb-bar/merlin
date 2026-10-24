"""Independent layout and partition-boundary controls for dynamic native examples."""
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from merlin.common.paths import repo_root

REFRESH=repo_root()/'examples/atlas/comparisons/refresh'
with patch.object(sys,'path',[str(REFRESH),*sys.path]):
    spec=importlib.util.spec_from_file_location('resnet_partition_controls',REFRESH/'resnet_graph_inputs.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)


def scalar_patches(value, module, shape):
    result = []
    for h in range(shape[-2]):
        for w in range(shape[-1]):
            row = []
            for c in range(value.shape[1]):
                for r in range(module.kernel_size[0]):
                    for s in range(module.kernel_size[1]):
                        ih = h * module.stride[0] + r * module.dilation[0] - module.padding[0]
                        iw = w * module.stride[1] + s * module.dilation[1] - module.padding[1]
                        row.append(value[0, c, ih, iw] if 0 <= ih < value.shape[2] and 0 <= iw < value.shape[3] else 0)
            result.append(row)
    return np.asarray(result,dtype=np.float32)


class ResnetPartitionTest(unittest.TestCase):
    def test_explicit_patches_match_scalar_rectangular_stride_dilation_and_padding(self):
        values=np.arange(1,1+2*7*9,dtype=np.float32).reshape(1,2,7,9)
        for kernel,stride,padding,dilation in (((1,1),(1,1),(0,0),(1,1)),((3,3),(1,1),(1,1),(1,1)),((2,3),(2,1),(1,2),(2,1)),((7,
            7),(2,2),(3,3),(1,1))):
            module=SimpleNamespace(groups=1,padding_mode='zeros',kernel_size=kernel,stride=stride,padding=padding,
                dilation=dilation)
            shape=(1,3,*[(values.shape[2+i]+2*padding[i]-dilation[i]*(kernel[i]-1)-1)//stride[i]+1 for i in range(2)])
            with self.subTest(kernel=kernel,stride=stride):
                actual=runner.explicit_patches(values,module,shape,np)
                np.testing.assert_array_equal(actual,scalar_patches(values,module,shape))
                self.assertTrue(actual.flags.c_contiguous)

    def test_boundary_and_channel_order_are_observable(self):
        values=np.arange(1,1+2*3*4,dtype=np.float32).reshape(1,2,3,4)
        module=SimpleNamespace(groups=1,padding_mode='zeros',kernel_size=(3,3),stride=(1,1),padding=(1,1),dilation=(1,1))
        actual=runner.explicit_patches(values,module,(1,2,3,4),np)
        self.assertEqual(actual[0,:9].tolist(),[0,0,0,0,1,2,0,5,6])
        self.assertEqual(actual[0,9:].tolist(),[0,0,0,0,13,14,0,17,18])
        self.assertFalse(np.array_equal(actual,actual.reshape(12,2,3,3).transpose(0,2,3,1).reshape(12,18)))

    def test_unsupported_batch_groups_and_padding_refused(self):
        value=np.zeros((1,2,3,4),dtype=np.float32)
        module=SimpleNamespace(groups=1,padding_mode='zeros',kernel_size=(1,1),stride=(1,1),padding=(0,0),dilation=(1,1))
        for changed,inputs in (({'groups':2},value),({'padding_mode':'reflect'},value),({},np.repeat(value,2,axis=0))):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                runner.explicit_patches(inputs,SimpleNamespace(**{**vars(module),**changed}),(1,2,3,4),np)

    def test_tail_padding_preserves_all_source_words_and_is_positive_zero(self):
        value=np.arange(35*19,dtype=np.float32).reshape(35,19)
        padded=runner.pad_matrix(value,64,32,np)
        np.testing.assert_array_equal(padded[:35,:19].view(np.uint32),value.view(np.uint32))
        self.assertTrue(np.all(padded[35:].view(np.uint32)==0))
        self.assertTrue(np.all(padded[:35,19:].view(np.uint32)==0))
        self.assertEqual(padded.dtype,np.float32)


if __name__=="__main__":
    unittest.main()
