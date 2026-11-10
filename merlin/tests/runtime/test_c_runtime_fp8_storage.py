"""FP8 C payloads are original one-byte words, never converted numeric values."""

import shutil
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.c_runtime import C_OF, DT_BYTES, FP8_DTYPES, _embed_array, _storage_bytes


@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_all_fp8_words_survive_generated_c_storage(tmp_path, dtype):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("host C compiler required")
    raw = np.arange(256, dtype=np.uint8)
    assert DT_BYTES[dtype] == 1
    assert _storage_bytes(raw, dtype) == raw.tobytes()
    source = '#include <stdio.h>\n'+f"static const {C_OF[dtype]} data[]={{{_embed_array(raw, dtype)}}};\n"
    source += "int main(void) {return fwrite(data,1,sizeof(data),stdout)!=sizeof(data);}\n"
    executable = tmp_path / "fp8-bytes"
    compiled = subprocess.run([compiler, "-O2", "-x", "c", "-", "-o", str(executable)],
        input=source, text=True, capture_output=True, timeout=20)
    assert compiled.returncode == 0, compiled.stderr
    actual = subprocess.run([str(executable)], capture_output=True, timeout=10)
    assert actual.returncode == 0
    assert actual.stdout == raw.tobytes()
    for function in (_storage_bytes, _embed_array):
        with pytest.raises(ValueError, match="raw uint8"):
            function(raw.astype(np.float32), dtype)


@pytest.mark.parametrize("dtype", FP8_DTYPES)
def test_fp8_value_footprints_are_one_byte_per_element(dtype):
    pytest.importorskip("xdsl")
    from xdsl.dialects.builtin import TensorType
    from merlin.common.mlir_query import value_bytes
    from merlin.xdsl_dialects.fp8 import _FP8_BY_NAME
    assert value_bytes(TensorType(_FP8_BY_NAME[dtype], [2, 7])) == 14
