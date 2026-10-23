"""Returned memref descriptors carry data-dependent output sizes."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.c_runtime import generate
from merlin.runtime.backends.spike_model import SpikeModelError, parse_console


def test_returned_descriptor_c_abi_handles_empty_and_nonempty_results(tmp_path):
    if not shutil.which("cc"):
        pytest.skip("native C compiler unavailable")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.mlir").write_text("""builtin.module {
      func.func @forward() -> (tensor<?x2xi64>, tensor<1xi64>) {
        func.return
      }
    }""")
    (model / "weights.safetensors.manifest.json").write_text(json.dumps({}))
    np.savez(model / "inputs.npz")
    out = tmp_path / "generated"
    generate(model, out, model / "inputs.npz", dump_all_outputs=True)
    driver = tmp_path / "driver.c"
    driver.write_text("""#include <assert.h>
    #include <stddef.h>
    #include <stdint.h>
    typedef struct {void *a,*p; int64_t offset,size[2],stride[2];} R2;
    typedef struct {void *a,*p; int64_t offset,size[1],stride[1];} R1;
    typedef struct {R2 r0; R1 r1;} Results;
    static int64_t data[16]; static int extent;
    void _mlir_ciface_forward(Results *r) {
      r->r0=(R2){data,data,0,{extent,2},{2,1}};
      r->r1=(R1){data,data,0,{1},{1}};
    }
    extern void merlin_invoke(void **);
    extern void **merlin_result_ptrs(void);
    extern size_t MERLIN_OUTPUT_NBYTES[2];
    extern long merlin_result_extent(int,int);
    int main(void) {
      for(extent=0;extent<7;extent++) {
        merlin_invoke(0);
        assert(MERLIN_OUTPUT_NBYTES[0]==(size_t)extent*2*sizeof(int64_t));
        assert(MERLIN_OUTPUT_NBYTES[1]==sizeof(int64_t));
        assert(merlin_result_extent(0,0)==extent);
        assert(merlin_result_ptrs()[0]==data);
      }
    }""")
    executable = tmp_path / "check"
    subprocess.run(
        ["cc", "-std=c11", "-Werror", str(driver), str(out / "model_call.c"), "-o", str(executable)], check=True
    )
    subprocess.run([str(executable)], check=True)


def test_shape_frames_must_be_complete_and_ordered():
    result = parse_console("OUT_SHAPE 0 2 0 2\nOUT_BYTES 0 0\nDONE\n")
    assert result["output_shapes"] == [[0, 2]]
    assert result["output_bytes"] == [b""]
    for frame in ("OUT_SHAPE 1 1 3", "OUT_SHAPE 0 2 3", "OUT_SHAPE 0 1 -1"):
        with pytest.raises(SpikeModelError):
            parse_console(frame + "\nOUT_BYTES 0 0\nDONE\n")
