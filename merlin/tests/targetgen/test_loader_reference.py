"""A float capsule whose corpus ships no golden is graded against its own PyTorch loader's output."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from merlin.targetgen import capsule_source, golden_store, loader_reference


def _capsule(tmp_path, *, dtype="f32", loader=True):
    d = tmp_path / "SY_host_lane_map_f32"
    d.mkdir()
    cap = {
        "name": d.name,
        "kind": "model_slice",
        "inputs": [{"name": "X", "role": "input", "shape": [2, 3], "dtype": dtype}],
        "operation": {"op": "gelu", "attributes": {"out": "Y0", "arg_order": ["X"]}},
        "numeric_policy": {"compare": "tolerance_float", "dtype": dtype, "atol": 0.03, "rtol": 0.02},
    }
    if loader:
        cap["pytorch_ref"] = {"op": "gelu", "dtype": dtype, "loader": "capsule.pytorch.py"}
        (d / "capsule.pytorch.py").write_text("# loader\n")
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap))
    return d, cap


class _Source:
    calls = 0

    def available(self):
        return True

    def capture_loader(self, loader, dtype, *, workdir=None):
        type(self).calls += 1
        x = [[0.5, -1.0, 0.25], [1.5, 2.0, -0.75]]
        return SimpleNamespace(inputs=[x], golden=[[v * 2 for v in row] for row in x], meta={"path_taken": "stub"})


@pytest.fixture
def stub_source(monkeypatch, tmp_path):
    _Source.calls = 0
    monkeypatch.setattr(capsule_source, "PytorchRefSource", _Source)
    monkeypatch.setenv(loader_reference.CACHE_ENV, str(tmp_path / "cache"))
    return _Source


def test_the_loader_defines_the_reference_and_its_inputs_once(tmp_path, stub_source):
    from merlin.targetgen import capsule_golden as CG
    from merlin.targetgen import capsule_inputs as CI

    d, cap = _capsule(tmp_path)
    assert golden_store.load_golden(d) is None
    assert CG.golden({**cap, "__dir__": str(d)}, d) == {"Y0": [[1.0, -2.0, 0.5], [3.0, 4.0, -1.5]]}
    assert CG.golden_source(cap, d) == "host_torch_eager"
    assert CI.canonical_input_values(cap, d)["X"]["values"] == [0.5, -1.0, 0.25, 1.5, 2.0, -0.75]
    assert stub_source.calls == 1, "captured once and then read from the content-addressed cache"
    assert not any(p.name.startswith("golden") for p in d.iterdir()), "the corpus is never written"
    (d / "capsule.pytorch.py").write_text("# a different loader\n")
    CG.golden({**cap, "__dir__": str(d)}, d)
    assert stub_source.calls == 2, "a changed loader is a different reference"


def test_integer_capsules_and_loaderless_float_capsules_are_untouched(tmp_path, stub_source):
    from merlin.targetgen import capsule_golden as CG

    d, cap = _capsule(tmp_path, dtype="i8")
    assert loader_reference.captured_reference(d) is None
    (tmp_path / "b").mkdir()
    d2, cap2 = _capsule(tmp_path / "b", loader=False)
    with pytest.raises(CG.UnsupportedGoldenFormat):
        CG.golden({**cap2, "__dir__": str(d2)}, d2)
    assert stub_source.calls == 0


def test_a_shipped_golden_always_wins(tmp_path, stub_source):
    from merlin.targetgen import capsule_golden as CG

    d, cap = _capsule(tmp_path)
    golden_store.write_golden(d, {"golden_source": "host_torch_eager", "outputs": {"Y0": [[9.0]]}})
    assert CG.golden({**cap, "__dir__": str(d)}, d) == {"Y0": [[9.0]]}
    assert stub_source.calls == 0
