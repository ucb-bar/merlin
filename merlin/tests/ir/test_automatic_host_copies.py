"""Ordinary host copy selection preserves explicit opt-outs and other families."""

from merlin.llvmlower.automatic_copies import FEATURES, select
from merlin.llvmlower.optional_passes import Selection, applied


def test_serial_default_selects_existing_checked_passes_only():
    features, report = select(None, serial=True)
    assert FEATURES <= features and report["enabled"]
    assert select(features, serial=True)[0] == features
    assert select(None, serial=False)[0] == frozenset()


def test_named_and_environment_opt_out_keep_explicit_existing_features(monkeypatch):
    with applied(Selection.parse("-automatic-host-copies")):
        assert select(None, serial=True)[0] == frozenset()
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_COPIES", "0")
    assert select(None, serial=True)[0] == frozenset()
    features, report = select({"fold_uniform_fill_copy"}, serial=True)
    assert "fold_uniform_fill_copy" in features and not report["enabled"]


def test_ordinary_lowering_realizes_the_pass_and_preserves_native_guarded_outputs(tmp_path):
    import subprocess

    import numpy as np

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.toolchain import m2m_python
    from merlin.tests.ir.test_uniform_fill_copy import fixture

    # The xDSL reader accepts generic memref.copy; canonicalize only the
    # fixture's printer syntax before exercising the ordinary front door.
    source = "\n".join(line for line in fixture().splitlines() if "memref.dealloc" not in line)
    printed = subprocess.run([str(m2m_python()), "-c",
        "import sys; from torch_mlir import ir\nwith ir.Context():\n"
        " m=ir.Module.parse(sys.stdin.read()); print(m.operation.get_asm(print_generic_op_form=True))"],
        input=source, text=True, capture_output=True, check=True)
    results = []
    for label, selection in (("control", "-automatic-host-copies"), ("candidate", "automatic-host-copies")):
        with applied(Selection.parse(selection)):
            result = lower_model(printed.stdout, tmp_path / label)
        results.append(result)
        invoke = HostModel.load(str(result.host_so))
        for word in (0, 127, 128, 255):
            value = np.array([word], dtype=np.uint8)
            guarded = np.full(5*7+16, 77, dtype=np.uint8)
            output = guarded[8:-8].reshape(5, 7)
            expected = guarded.copy()
            expected[8:-8].reshape(5, 7)[1:3, 1:4] = word
            expected[8:-8].reshape(5, 7)[3:5, 3:6] = word
            for _ in range(2):
                invoke([(value.ctypes.data, value.shape), (output.ctypes.data, output.shape)])
                np.testing.assert_array_equal(guarded, expected)
    assert not results[0].stats["automatic_host_copies"]["enabled"]
    assert results[1].stats["automatic_host_copies"]["uniform_fill_copy"]["copies_folded"] == 2
