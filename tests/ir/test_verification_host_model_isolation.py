from __future__ import annotations

from verification_witnesses.host_model_isolation import HostModel, ctypes, pytest, shutil, subprocess


@pytest.mark.parametrize("n_args", [1025])
def test_trampolines_bind_the_loaded_model_not_the_first_global_symbol(tmp_path, n_args):
    if shutil.which("cc") is None:
        pytest.skip("host C compiler unavailable")
    models = []
    params = ", ".join((f"void *p{i}" for i in range(n_args)))
    for value in (17, 29):
        src = tmp_path / f"model_{value}.c"
        lib = src.with_suffix(".so")
        src.write_text(
            f"typedef struct {{ void *allocated; long *aligned; long offset; }} Desc;\nvoid forward(void *p) {{ Desc *d = p; d->aligned[d->offset] = {value}; }}\nvoid _mlir_ciface_forward({params}) {{ forward(p0); }}\n"
        )
        subprocess.run(["cc", "-fPIC", "-shared", str(src), "-o", str(lib)], check=True)
        models.append(HostModel.load(str(lib), n_args=n_args))
    output = ctypes.c_long()
    buffers = [(ctypes.addressof(output), [])] * n_args
    for model, expected in ((models[0], 17), (models[1], 29), (models[0], 17)):
        model(buffers)
        assert output.value == expected
