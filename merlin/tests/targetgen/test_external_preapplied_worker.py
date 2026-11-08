"""External selection must be complete and cannot trigger numerical rewrites."""
import pytest

from merlin.targetgen._m2m_capture_worker import main


@pytest.mark.parametrize("options", [
    ["--expected-contract-sha256", "0" * 64],
    ["--already-quantized", "--quant-adapter", "external", "--contract", "contract.json",
     "--policy", "policy.json", "--expected-policy-sha256", "incorrect"],
    ["--quant-adapter", "external"],
    ["--contract", "contract.json", "--policy", "policy.json"],
    ["--quant-adapter", "external", "--contract", "contract.json", "--policy", "policy.json"],
    ["--already-quantized", "--quant-adapter", "external"],
    ["--already-quantized", "--quant-adapter", "external", "--contract", "contract.json",
     "--policy", "policy.json", "--scheme", "int8_weight_only"],
    ["--already-quantized", "--quant-adapter", "external", "--contract", "contract.json",
     "--policy", "policy.json", "--stage-fp32"],
])
def test_incomplete_or_conflicting_external_selection_refuses_before_loading(options):
    with pytest.raises(SystemExit) as error:
        main(["--loader", "absent.py", "--dtype", "bf16", "--out", "absent-out", *options])
    assert error.value.code == 2


@pytest.mark.parametrize("expected", ["valid", "invalid", None])
def test_exact_export_reaches_normalization_and_bundle(tmp_path, expected):
    from hashlib import sha256
    import json

    contract, policy = tmp_path / "contract.json", tmp_path / "policy.json"
    contract.write_text("{}")
    policy.write_text("{}")
    manifest = {"schema": "m2m.quantization_manifest.v2", "adapter_id": "external_fixture",
                "contract_sha256": sha256(contract.read_bytes()).hexdigest(),
                "policy_sha256": sha256(policy.read_bytes()).hexdigest(),
                "sites": [{"site_id": "identity", "status": "host"}]}
    loader = tmp_path / "loader.py"
    loader.write_text('''import torch
from xdsl.dialects.builtin import StringAttr
class Model(torch.nn.Module):
    def forward(self, x):
        return x + x + torch.tensor([2., 3., 4.])
def get_model_and_inputs():
    inputs = (torch.ones(2, 3),)
    ep = torch.export.export(Model().eval(), inputs)
    ep._m2m_external_quantization_manifest = ''' + repr(manifest) + '''
    return ep, inputs
def normalize_capture(module, manifest):
    assert manifest['adapter_id'] == 'external_fixture'
    module.attributes['fixture.normalized'] = StringAttr('yes')
    return module
''')
    out = tmp_path / "capture"
    options = ["--loader", str(loader), "--dtype", "bf16", "--out", str(out),
                 "--already-quantized", "--quant-adapter", "external_fixture",
                 "--contract", str(contract), "--policy", str(policy), "--materialize-bundle",
                 *(["--expected-contract-sha256", "0" * 64 if expected == "invalid" else manifest["contract_sha256"],
                    "--expected-policy-sha256", manifest["policy_sha256"]] if expected else [])]
    if expected == "invalid":
        with pytest.raises(ValueError, match="selected contract sha256 differs"):
            main(options)
        return
    assert main(options) == 0
    assert 'fixture.normalized = "yes"' in (out / "model.mlir").read_text()
    assert json.loads((out / "quantization-manifest.json").read_text()) == manifest
    metadata = json.loads((out / "meta.json").read_text())
    assert metadata["opaque"] == 0
    import numpy as np
    with np.load(out / "extra.npz", allow_pickle=False) as leaves:
        assert np.array_equal(leaves["c_lifted_tensor_0"], np.array([2., 3., 4.]))


def test_runpy_import_finds_owned_helper_from_an_unrelated_directory(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path

    import merlin.targetgen._m2m_capture_worker as worker

    script = Path(worker.__file__).resolve()
    helper_name = "_external_quantization_capture"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p).resolve() != script.parent])
    # Register the previous state with the fixture before allowing runpy to
    # create a standalone helper module; teardown restores or removes it.
    monkeypatch.setitem(sys.modules, helper_name, None)
    del sys.modules[helper_name]
    loaded = runpy.run_path(str(script), run_name="capture_worker_import_control")
    assert Path(loaded["EQ"].__file__).resolve() == script.with_name(helper_name + ".py")
