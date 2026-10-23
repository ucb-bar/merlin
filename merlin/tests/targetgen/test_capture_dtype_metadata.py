"""Metadata casts keep no evaluated tensor payload in the prepared graph."""
import pytest


def test_real_metadata_is_converted_to_fake_before_a_dtype_change():
    torch=pytest.importorskip('torch')
    from torch._subclasses.fake_tensor import FakeTensor
    from merlin.targetgen.declared_calibration import dtype_metadata
    results=[dtype_metadata(torch.full((2,3),value),torch.int8) for value in (0.,float('nan'))]
    assert all(isinstance(x,FakeTensor) for x in results)
    assert [(x.dtype,tuple(x.shape),x.stride()) for x in results]==[(torch.int8,(2,3),(3,1))]*2
