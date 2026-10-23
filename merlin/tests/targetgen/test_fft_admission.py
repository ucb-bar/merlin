"""No operator name selects an FFT preparation path; FFT placement comes from the IR detectors."""

from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.software_spec import load_software_spec


def _selection():
    root = repo_root() / "examples" / "gemmini" / "target"
    spec, host = root / "software-spec.yaml", root.parent / "platform" / "host-capabilities-core-aten.yaml"
    return {
        "spec": load_software_spec(spec),
        "host": validate_host_capabilities(load_yaml(host)),
        "identities": {"software_spec": {"path": str(spec)}, "host_capabilities": {"path": str(host)}},
    }


@pytest.mark.parametrize("recipe_count", [0, 2])
def test_fft_overload_name_does_not_trigger_a_recapture(tmp_path, monkeypatch, recipe_count):
    """No source overload selects a preparation path (source_overload_dispatch: forbidden).

    A program with no admitted contraction, no prospective native site and no
    unique static recipe keeps its original host source, whatever the case's
    overload name says.
    """
    from merlin.targetgen import core_aten_numeric, core_aten_spec_int8
    from merlin.targetgen.capsule_source import PytorchRefSource

    selection = _selection()
    selection["recipes"] = [
        {
            "activation": dict(
                dtype="int8", mode="static", granularity="tensor", symmetric=True, quant_min=-128, quant_max=127
            )
        }
        for _ in range(recipe_count)
    ]
    monkeypatch.setattr(core_aten_spec_int8, "select_spec", lambda *args: selection)

    def capture(*args, **kwargs):
        raise AssertionError("ambiguous or absent recipes cannot recapture the source")

    monkeypatch.setattr(PytorchRefSource, "capture_loader", capture)
    (tmp_path / "spec.yaml").write_text(Path(selection["identities"]["software_spec"]["path"]).read_text())
    original = "builtin.module {}\n"
    (tmp_path / "model.mlir").write_text(original)
    report = {
        "cases": [
            {
                "overload": "aten._fft_r2c.default",
                "status": "bundled",
                "input_abi": [{"dtype": "f32", "shape": [2, 4]}],
                "output_abi": [{"dtype": "f32", "shape": [2, 3, 2]}],
            }
        ]
    }
    plan = core_aten_numeric.prepare_spec_int8(
        tmp_path,
        report,
        target="selected",
        facts={},
        software_spec=tmp_path / "spec.yaml",
        host_capabilities=tmp_path / "host.yaml",
        provider=None,
        package=tmp_path,
    )
    assert not plan.get("quantized_contractions")
    assert plan["native_admission"]["status"] == "host"
    assert (tmp_path / "model.mlir").read_text() == original
