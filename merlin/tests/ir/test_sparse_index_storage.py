"""Bounded private support indices preserve words, fenv and capacity refusals."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest
from test_guarded_zero_contraction import source

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import build_host_shared
from merlin.llvmlower.guarded_zero_contraction import INDEX_FEATURE
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scalar_contraction import EIGHT_OUTPUTS_FEATURE
from merlin.llvmlower.toolchain import clang


@pytest.mark.parametrize("k", [0, 1, 4096])
def test_exact_storage_boundary_indices_modes_and_flags(tmp_path, monkeypatch, k):
    monkeypatch.setenv("MERLIN_GENERALIZE_BEFORE_FUSE", "1")
    text = source(m=3, n=8, k=k)
    models = []
    for name, features in [("control", set()), ("selected", {EIGHT_OUTPUTS_FEATURE, INDEX_FEATURE})]:
        d = tmp_path / name
        d.mkdir()
        ll = lower_to_llvm_ir(text, workdir=d, features=features)
        p = d / "model.ll"
        p.write_text(ll)
        models.append(HostModel.load(str(build_host_shared(p, d / "model.so"))))
        if name == "selected" and k > 1:
            assert "load i16" in ll
    c = tmp_path / "env.c"
    c.write_text(
        """#include <fenv.h>
int mode(int n){
  int m[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
  return fesetround(m[n]);
}
int flags(void){return fetestexcept(FE_ALL_EXCEPT);}
int clear(void){return feclearexcept(FE_ALL_EXCEPT);}
"""
    )
    so = tmp_path / "env.so"
    subprocess.run([str(clang()), "-shared", "-fPIC", str(c), "-lm", "-o", str(so)], check=True, capture_output=True)
    env = ctypes.CDLL(str(so))
    try:
        for mode in range(4):
            assert env.mode(mode) == 0
            for full in [False, True]:
                p = np.zeros((3, k), dtype="f4")
                b = np.arange(k * 8, dtype="i1").reshape(k, 8)
                if k:
                    if full:
                        p[:] = np.float32(2**-12)
                    else:
                        p[:, -1] = 0.5
                        p[1, -1] = -0.5
                        p[2, -1] = np.array([1], dtype="u4").view("f4")[0]
                output = []
                flags = []
                for model in models:
                    store = np.full(40, 0xA5A5A5A5, dtype="u4")
                    out = store[8:32].view("f4").reshape(3, 8)
                    env.clear()
                    model([(p.ctypes.data, p.shape), (b.ctypes.data, b.shape), (out.ctypes.data, out.shape)])
                    flags.append(env.flags())
                    output.append(out.copy())
                    assert np.all(store[:8] == 0xA5A5A5A5) and np.all(store[32:] == 0xA5A5A5A5)
                assert flags[0] == flags[1]
                np.testing.assert_array_equal(output[0].view("u4"), output[1].view("u4"))
    finally:
        assert env.mode(0) == 0
