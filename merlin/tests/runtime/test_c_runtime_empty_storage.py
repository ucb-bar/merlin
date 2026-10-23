"""Empty logical arguments retain distinct native caller-storage identities."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower.c_runtime import generate


@pytest.mark.parametrize("dtype,width", [("f16", 2), ("f32", 4), ("i1", 1)])
def test_empty_input_storage_is_distinct(tmp_path, dtype, width):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler required")
    model = tmp_path / "model"
    model.mkdir()
    tensor = f"tensor<0x{dtype}>"
    (model / "model.mlir").write_text(
        f"module {{ func.func @forward(%a: {tensor}, %b: {tensor}, %c: {tensor}) -> {tensor} {{"
        f" func.return %c : {tensor} }} }}"
    )
    (model / "weights.safetensors.manifest.json").write_text(
        json.dumps({str(i): dict(kind="input", name=f"arg{i}") for i in range(3)})
    )
    np.savez(model / "inputs.npz", **{f"in{i}": np.empty(0, np.float32) for i in range(3)})
    out = tmp_path / "generated"
    info = generate(model, out, model / "inputs.npz", dump_all_outputs=True)
    assert info["static_io_bytes"] >= 3 * width
    source = out / "check.c"
    source.write_text(
        '#include "model_gen.h"\n#include "model_io.h"\n'
        "#include <assert.h>\n"
        "int main(void) {\n"
        "assert(MERLIN_INPUT_PTR[0] != MERLIN_INPUT_PTR[1]);\n"
        "assert(MERLIN_INPUT_PTR[0] != MERLIN_INPUT_PTR[2]);\n"
        "assert(MERLIN_INPUT_PTR[1] != MERLIN_INPUT_PTR[2]);\n"
        "return 0; }\n"
    )
    exe = out / "check"
    subprocess.run(
        [cc, "-O2", "-std=c11", "-I" + str(runtime_dir() / "c"), "-I" + str(out), str(source), "-o", str(exe)],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(exe)], check=True, capture_output=True)
