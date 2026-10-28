"""Kernel handoff keeps all outputs, signed inputs, VALID borders and exact exits."""
import importlib

import numpy as np
import pytest
from merlin.common.paths import repo_root


@pytest.fixture
def kernels(monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root()/"examples/gemmini/comparisons/tvm"))
    return importlib.import_module("prepare_kernel_benchmarks")


def evaluate(kernels,spec):
    tvm=pytest.importorskip("tvm")
    from tvm import relax
    data=importlib.import_module("kernel_benchmark_data")
    inputs,expected=data.fixtures(spec)
    vm=relax.VirtualMachine(relax.build(kernels.build_program(spec),target="llvm"),tvm.cpu())
    output=vm["main"](*[tvm.nd.array(v) for v in inputs.values()])
    actual=[output] if len(expected)==1 else list(output)
    assert len(actual)==len(expected)
    for got,want in zip(actual,expected.values()):
        np.testing.assert_array_equal(got.numpy(),want)
    return inputs,expected


@pytest.mark.parametrize("case",["T01","T02"])
def test_complete_attention_group_has_all_32_exact_outputs(kernels,case):
    spec=kernels.case_spec(case)
    inputs,expected=evaluate(kernels,spec)
    assert len(inputs)==36 and len(expected)==32
    a,k=inputs["A31"],inputs["K3"]
    acc=sum(int(a[0,i])*int(k[i,0]) for i in range(a.shape[1]))
    value=acc if case=="T01" else int(np.clip(np.rint(np.float32(acc)/128),-128,127))
    assert expected["Y31"][0,0]==value


def test_biased_valid_convolution_preserves_nonzero_border(kernels):
    spec=dict(id="control",kind="conv",scale=1.,bias_seed=0x6000,
              roster=[("IFM",(1,5,6,2)),("W",(18,3))])
    inputs,expected=evaluate(kernels,spec)
    assert expected["Y0"].shape==(12,3)
    assert np.any(inputs["IFM"][:,0,:,:]!=0)
    value=int(inputs["B"][0])
    for y in range(3):
        for x in range(3):
            for c in range(2):
                value += int(inputs["IFM"][0,y,x,c])*int(inputs["W"][(y*3+x)*2+c,0])
    assert expected["Y0"][0,0]==max(-128,min(127,value))


def test_projection_groups_keep_every_independent_weight_and_output(kernels):
    for case,count in (("T04",2),("T06",3)):
        spec=kernels.case_spec(case)
        assert len(spec["products"])==count
        names=[name for name,_ in spec["roster"]]
        spec["roster"]=[(name,(3,5) if name=="A0" else (5,7+i)) for i,name in enumerate(names)]
        _,outputs=evaluate(kernels,spec)
        assert len(outputs)==count and all(v.dtype==np.int8 for v in outputs.values())


def test_catalog_shapes_exits_and_generator_are_exact(kernels):
    data=importlib.import_module("kernel_benchmark_data")
    assert len(data.CASES)==16
    assert dict(data.case_spec("G03")["roster"])["W"]==(64,256)
    assert dict(data.case_spec("T07")["roster"])["A0"]==(1,2048)
    assert data.case_spec("T07")["scale"] is None
    assert [data.case_spec(f"C{i:02d}")["bias_seed"] for i in range(1,4)]==[0x6000,0x5000,0x4000]
    values=data.make_inputs([("a",(17,19)),("b",(3,7))])
    assert all(v.dtype==np.int8 and v.min()>=-8 and v.max()<=7 for v in values.values())
    np.testing.assert_array_equal(data.q8(np.array([-320,-192,-64,64,192,320]),1/128),[-2,-2,0,0,2,2])
