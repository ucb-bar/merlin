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


@pytest.mark.parametrize("expected", ["valid", "invalid", "manifest-mismatch", "missing-manifest", None])
def test_exact_export_reaches_normalization_and_bundle(tmp_path, monkeypatch, expected):
    from functools import wraps
    from hashlib import sha256
    import json

    import m2m
    import torch
    from merlin.targetgen import _m2m_capture_worker as worker

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
    def __init__(self):
        super().__init__()
        self.register_buffer('offset', torch.tensor([5., 6., 7.]))
    def forward(self, x):
        return x + x + self.offset + torch.tensor([2., 3., 4.])
def get_model_and_inputs():
    inputs = (torch.ones(2, 3),)
    ep = torch.export.export(Model().eval(), inputs)
    ep._m2m_external_quantization_manifest = ''' + repr(
        None if expected == "missing-manifest" else {
            **manifest, **({"adapter_id": "other"} if expected == "manifest-mismatch" else {})
        }
    ) + '''
    return ep, inputs
def normalize_capture(module, manifest):
    assert manifest['adapter_id'] == 'external_fixture'
    module.attributes['fixture.normalized'] = StringAttr('yes')
    return module
''')
    selected = {}
    prepare = worker.EQ.prepared_module

    def observe_prepared(model, **options):
        exported, module = prepare(model, **options)
        selected.update(exported=exported, graph=str(exported.graph),
                        weights={name: value.clone() for name, value in exported.state_dict.items()})
        return exported, module

    monkeypatch.setattr(worker.EQ, "prepared_module", observe_prepared)
    convert = m2m.convert
    conversions = []

    @wraps(convert)
    def observe_conversion(model, inputs, **options):
        conversions.append(model)
        assert model is selected["exported"]
        assert str(model.graph) == selected["graph"]
        assert all(torch.equal(value, selected["weights"][name])
                   for name, value in model.state_dict.items())
        assert inputs[0].dtype == torch.float32  # --dtype bf16 must not cast the source.
        assert options["original_frontend_snapshot"]["status"] == "complete"
        return convert(model, inputs, **options)

    monkeypatch.setattr(m2m, "convert", observe_conversion)
    out = tmp_path / "capture"
    options = ["--loader", str(loader), "--dtype", "bf16", "--out", str(out),
                 "--already-quantized", "--quant-adapter", "external_fixture",
                 "--contract", str(contract), "--policy", str(policy), "--materialize-bundle",
                 *(["--expected-contract-sha256", "0" * 64 if expected == "invalid" else manifest["contract_sha256"],
                    "--expected-policy-sha256", manifest["policy_sha256"]] if expected else [])]
    if expected in {"invalid", "manifest-mismatch", "missing-manifest"}:
        error = {"invalid": "selected contract sha256 differs",
                 "manifest-mismatch": "manifest adapter_id differs",
                 "missing-manifest": "manifest has an invalid schema"}[expected]
        with pytest.raises(ValueError, match=error):
            main(options)
        assert not conversions
        assert not (out / "capture_receipt.json").exists()
        return
    assert main(options) == 0
    assert len(conversions) == 1
    assert 'fixture.normalized = "yes"' in (out / "model.mlir").read_text()
    assert json.loads((out / "quantization-manifest.json").read_text()) == manifest
    metadata = json.loads((out / "meta.json").read_text())
    assert metadata["opaque"] == 0
    assert metadata["read_only_buffer_arguments"] == [0]
    receipt = json.loads((out / "capture_receipt.json").read_text())
    assert receipt["artifacts"]["meta.json"]["sha256"] == sha256(
        (out / "meta.json").read_bytes()
    ).hexdigest()
    import numpy as np
    with np.load(out / "extra.npz", allow_pickle=False) as leaves:
        assert np.array_equal(leaves["c_lifted_tensor_0"], np.array([2., 3., 4.]))
        assert leaves["c_lifted_tensor_0"].dtype == np.float32
    trace = json.loads((out / "frontend-trace.json").read_text())
    assert trace["status"] == "complete", trace
    assert trace["graphs"]["original"]["status"] == "complete"
    assert trace["graphs"]["original"]["by_target"] == trace["graphs"]["quantized"]["by_target"]
    assert all(value["dtype"] in {None, "float32"}
               for node in trace["graphs"]["original"]["nodes"] for value in node["results"])
    from merlin.targetgen.cli import main as targetgen_main

    staged, status = tmp_path / "compiler-inputs", tmp_path / "stage-status.json"
    assert targetgen_main(["stage-capture", "--capture", str(out), "--out", str(staged),
                           "--status-file", str(status)]) == 0
    assert json.loads(status.read_text())["status"] == "staged"
    assert (staged / "frontend-trace.json").read_bytes() == (out / "frontend-trace.json").read_bytes()
    assert (staged / "weights.safetensors").read_bytes() == (out / "weights.safetensors").read_bytes()


@pytest.mark.parametrize("external,native", [(False, True), (True, False), (False, False)])
def test_other_preapplied_sources_keep_original_unavailable(tmp_path, monkeypatch, external, native):
    from functools import wraps
    from types import SimpleNamespace

    import m2m
    import torch
    from merlin.targetgen import _m2m_capture_worker as worker

    class Model(torch.nn.Module):
        def forward(self, x):
            return x + x

    inputs = (torch.ones(2, 3),)
    source = torch.export.export(Model(), inputs) if native else Model()
    loader = tmp_path / "loader.py"
    loader.write_text("# test-owned loader\n")
    monkeypatch.setattr(worker, "_load_loader", lambda path: SimpleNamespace(
        get_model_and_inputs=lambda: (source, inputs), normalize_capture=lambda module, manifest: module,
    ))

    class ConversionObserved(Exception):
        pass

    @wraps(m2m.convert)
    def observe_conversion(model, example_inputs, **options):
        assert model is source
        assert options["original_frontend_snapshot"]["status"] == "unavailable"
        raise ConversionObserved

    monkeypatch.setattr(m2m, "convert", observe_conversion)
    selection = []
    if external:
        contract, policy = tmp_path / "contract.json", tmp_path / "policy.json"
        contract.write_text("{}")
        policy.write_text("{}")
        selection = ["--quant-adapter", "external_fixture", "--contract", str(contract),
                     "--policy", str(policy)]
    with pytest.raises(ConversionObserved):
        main(["--loader", str(loader), "--dtype", "bf16", "--out", str(tmp_path / "capture"),
              "--already-quantized", *selection])


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
