"""Launch census conservation and captured-boundary regression tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from launch_inventory import capture_boundaries, capture_live_boundaries, classify, document, load_exporter, phase_records, projection_shapes


class LaunchInventoryTest(unittest.TestCase):
    def test_iteration_boundary_and_layernorm_gap(self):
        records=[{'module':'policy.noise_predictor.time_mlp.1','op':'aten.linear.default'},
                 {'module':'policy.noise_predictor.transformer_blocks.0.attn','op':'aten.scaled_dot_product_attention.default'},
                 {'module':'policy.noise_predictor.time_mlp.1','op':'aten.linear.default'}]
        phase_records(records)
        self.assertEqual([r['step'] for r in records],[0,0,1])
        self.assertEqual(classify({'op':'aten.layer_norm.default'})[0],[])
        self.assertIn('not RMSNorm',classify({'op':'aten.layer_norm.default'})[1])

    def test_live_hooks_preserve_forward_and_count_attention(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            loader=root/'loader.py'
            loader.write_text("""import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(4,4)
    def forward(self,x):
        a=self.linear(x)
        q=a.view(1,1,3,4)
        return torch.nn.functional.scaled_dot_product_attention(q,q,q)+torch.nn.functional.scaled_dot_product_attention(q,q,q)
def get_model_and_inputs():
    torch.manual_seed(0)
    return Model().eval(),(torch.randn(1,3,4),)
""")
            spec=importlib.util.spec_from_file_location('reference_loader',loader)
            module=importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model,inputs=module.get_model_and_inputs()
            with torch.inference_mode():
                np.save(root/'expected.npy',model(*inputs).numpy())
            result=capture_live_boundaries(loader,root/'boundaries',root/'expected.npy')
            self.assertEqual(result['full_forward_eager_check']['status'],'PASS')
            self.assertEqual(result['projection_logical_occurrences'],1)
            self.assertEqual(result['attention_logical_occurrences'],2)
            self.assertEqual(result['attention'][0]['count'],2)

    def test_full_boundary_geometry_and_counts(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first=torch.nn.Linear(5,7)
                self.second=torch.nn.Linear(5,7)
            def forward(self,x):
                return self.first(x)+self.second(x)
        torch.manual_seed(7)
        model=Model().eval().requires_grad_(False)
        inputs=(torch.randn(2,3,5),)
        program=torch.export.export(model,inputs,strict=False)
        library=Path(__file__).resolve().parents[5]/'atlas-kernels'
        exporter=load_exporter(library)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            export=root/'model.pt2'
            torch.export.save(program,export)
            np.savez(root/'inputs.npz',x=inputs[0].numpy())
            np.save(root/'eager.npy',model(*inputs).numpy())
            doc,records=document(program,exporter,export)
            self.assertEqual(doc['summary']['linear_calls'],2)
            self.assertEqual(doc['summary']['qualified_fp32_flops'],0)
            self.assertEqual(doc['summary']['linear_candidate_flops_excluding_bias'],840)
            result=capture_boundaries(program,root/'inputs.npz',root/'boundaries',records,root/'eager.npy')
            self.assertEqual(result['projection_count'],1)
            self.assertEqual(result['projection_logical_occurrences'],2)
            self.assertEqual(result['full_forward_eager_check']['status'],'PASS')
            row=result['projections'][0]
            self.assertEqual((row['m'],row['k'],row['n']),(6,5,7))
            with np.load(row['artifact']['path']) as values:
                self.assertTrue(np.array_equal(values['A'],inputs[0].reshape(6,5).numpy()))
                self.assertTrue(np.array_equal(values['W'],model.first.weight.T.numpy()))
                self.assertTrue(np.array_equal(values['expected_without_bias'],torch.mm(torch.from_numpy(values['A']),torch.from_numpy(values['W'])).numpy()))
                self.assertTrue(np.allclose(values['original_output'],values['expected_without_bias']+values['bias']))


if __name__=='__main__':
    unittest.main()
