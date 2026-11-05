"""Automatic host composition must execute real checked source transforms."""

import hashlib
import json

import numpy as np
import pytest

from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.automatic_host_features import FEATURE, choices, require_report, select
from merlin.llvmlower.lower import lower_model
from merlin.llvmlower.optional_passes import Selection, applied
from merlin.tests.ir.test_scalar_contraction import ordered, source


def test_normal_selection_respects_family_and_named_opt_outs(monkeypatch):
    assert FEATURE in select(None, serial=True)
    assert FEATURE not in select(None, serial=False)
    explicit = {"scalar_contraction_accumulator_2x4_outputs", "hoist_broadcast_source_rsqrt"}
    assert choices(select(explicit, serial=True)) == {
        "contraction": False,
        "broadcast_math": False,
        "squared_sum": True,
    }
    with applied(Selection.parse("-automatic-host-optimizations")):
        assert select(None, serial=True) == frozenset()
    with applied(Selection.parse("-scalar-contraction,-broadcast-math-hoist,-scalar-squared-sum")):
        assert not any(choices(select(None, serial=True)).values())
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_OPTIMIZATIONS", "0")
    assert select(None, serial=True) == frozenset()


@pytest.mark.parametrize("transpose", [False, True])
def test_actual_native_contraction_retains_order_and_all_words(tmp_path, transpose):
    rng = np.random.default_rng(52)
    a = rng.normal(size=(3, 7)).astype(np.float32)
    b = rng.normal(size=(7, 5)).astype(np.float32)
    c = rng.normal(size=(3, 5)).astype(np.float32)
    a[0] = [16777216, 1, -16777216, 0, 0, 0, 0]
    b[:, 0] = 1
    c[0, 0] = 0
    expected = ordered(a, b, c)
    storage = np.ascontiguousarray(b.T) if transpose else b
    text = source(a.shape, storage.shape, c.shape, transpose=transpose)
    source_key = hashlib.sha256(text.encode()).hexdigest()[:12]
    for label, setting in (("control", "-automatic-host-optimizations"), ("automatic", "automatic-host-optimizations")):
        with applied(Selection.parse(setting)):
            result = lower_model(text, tmp_path / (label + "_" + source_key))
        guarded = np.full(31, 79, dtype=np.float32)
        output = guarded[8:-8].reshape(c.shape)
        invoke = HostModel.load(str(result.host_so))
        for _ in range(2):
            seed = c.copy()
            invoke([(x.ctypes.data, x.shape) for x in (a, storage, seed, output)])
            np.testing.assert_array_equal(output.view(np.uint32), expected.view(np.uint32))
            np.testing.assert_array_equal(guarded[:8], np.full(8, 79, np.float32))
            np.testing.assert_array_equal(guarded[-8:], np.full(8, 79, np.float32))
        if label == "automatic":
            report = result.stats["automatic_host_optimizations"]
            assert report["rewrites"]["contraction"] == 1
            assert report["hardware_cycles"] == "not_measured"
            assert "fmul float" in result.ll_path.read_text()
            assert "llvm.fma" not in result.ll_path.read_text()
            assert json.loads((result.workdir / "automatic-host-optimizations.json").read_text()) == report


def test_selected_report_cannot_silently_disappear_or_change_choices():
    with pytest.raises(ValueError, match="absent"):
        require_report("", features={FEATURE})
    report = {"schema": "merlin.automatic_host_features.v1", "choices": {}, "rewrites": {}}
    with pytest.raises(ValueError, match="differs"):
        require_report("OK automatic_scalar_host " + json.dumps(report), features={FEATURE})
    report.update(
        choices=choices({FEATURE}),
        rewrites=dict.fromkeys(choices({FEATURE}), 0),
        source_sha256="0" * 64,
        output_sha256="1" * 64,
        hardware_cycles="not_measured",
    )
    token = "OK automatic_scalar_host " + json.dumps(report)
    assert require_report(token, features={FEATURE}) == report
    with pytest.raises(ValueError, match="ambiguous"):
        require_report(token + "\n" + token, features={FEATURE})
    report["source_sha256"] = "missing"
    with pytest.raises(ValueError, match="differs"):
        require_report("OK automatic_scalar_host " + json.dumps(report), features={FEATURE})
