"""Unproved native IEEE sources refuse before any device proof or eager readout."""

import os
from pathlib import Path

import numpy as np
import pytest
from mlir_fixtures import generic
from targetgen_fixtures import assert_host_refusal, eager_record


def test_unproved_native_source_refuses_without_reading_eager_or_inputs(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("declared capture interpreter unavailable")
    typ = "tensor<1xf16>"
    body = generic([("%x", typ), ("%zeros", typ)], typ, "%v=arith.subf %a,%b : f16")
    source = f"""module {{func.func @forward(%x:{typ}) -> ({typ},{typ}) {{
      %z=arith.constant 0.0 : f16
      %zeros=tensor.splat %z : {typ}
      {body}
      return %r,%x : {typ},{typ}
    }} }}"""
    # This independent CPU boundary expects +0; the captured equation gives -0.
    # The reference remains qualification data, never a generated readout value.
    record = eager_record([dict(dtype="f16")], ["0000"], ["0080"], [1])
    inputs = [np.array([-0.0], np.float16)]
    refusal = assert_host_refusal(tmp_path, source, inputs, [record], "native_component_transfer_unadmitted")
    assert "source_agreement" not in refusal


def test_native_reduced_storage_requires_complete_source_observation(tmp_path):
    if not os.environ.get("MERLIN_M2M_PYTHON"):
        pytest.skip("declared capture interpreter unavailable")
    typ = "tensor<1xf16>"
    body = generic([("%x", typ), ("%y", typ), ("%z", typ)], typ, "%v=math.fma %a,%b,%c : f16")
    source = f"module {{func.func @forward(%x:{typ},%y:{typ},%z:{typ}) -> {typ} {{\n{body}\nreturn %r : {typ}\n}} }}"
    record = eager_record([dict(dtype="f16")] * 3, [np.array([2], np.float16).tobytes().hex()], [], [])
    inputs = [np.array([1], np.float16)] * 3
    assert_host_refusal(tmp_path, source, inputs, [record], "native_parallel_runtime_product_construction")
