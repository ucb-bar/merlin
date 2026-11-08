"""Exercise the example loader through real raw-state reconstruction and export."""
from hashlib import sha256
import json
from unittest.mock import patch

import algos
import pytest
import quant_export
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, LlamaConfig

import adapter_loader


@pytest.fixture
def source(tmp_path, monkeypatch):
    import fx_export

    torch.manual_seed(19)
    architecture, export = tmp_path / "architecture", tmp_path / "export"
    architecture.mkdir()
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                         tie_word_embeddings=False, use_cache=False)
    config._attn_implementation = "eager"
    config.save_pretrained(architecture)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="eager").eval()
    for module in model.modules():
        module._non_persistent_buffers_set.clear()
    algo = algos.make("granularity:w=channel,a=token,rule=nc")
    options = {"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"}
    report = algos.quantize(model, "ref_conv", ["lm_head"], options, algo=algo)
    quant_export.export(model, export, mode="ref_conv", report=report, algo=algo,
                        algo_spec="granularity:w=channel,a=token,rule=nc", options=options, skip=["lm_head"])
    manifest_path = export / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["model"] = {"checkpoint": {"config_json_sha256": sha256((architecture / "config.json").read_bytes()).hexdigest()}}
    manifest_path.write_text(json.dumps(manifest))
    inputs = torch.arange(8, dtype=torch.int64).reshape(1, 8)
    with torch.no_grad():
        expected = fx_export.CausalLogits(model).eval()(inputs).clone()
    inputs_path = tmp_path / "inputs.safetensors"
    save_file({"input_ids": inputs}, inputs_path)
    values = {"ATLAS_EXPORT_DIR": str(export), "ATLAS_ARCHITECTURE": str(architecture),
              "ATLAS_INPUTS_SAFETENSORS": str(inputs_path),
              "ATLAS_EXPECTED_CONTRACT_SHA256": sha256(manifest_path.read_bytes()).hexdigest(),
              "ATLAS_EXPECTED_POLICY_SHA256": sha256((export / "policy.json").read_bytes()).hexdigest()}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return export, architecture, inputs_path, inputs, expected


def test_loader_reconstructs_native_attention_without_saved_program_or_weight_conversion(source):
    from ref_profile import RefQuantLinear
    import raw_capture
    from safetensors.torch import load_file

    export, _, _, inputs, expected = source
    before = raw_capture.inventory(load_file(str(export / "model.safetensors")))
    with patch.object(torch.export, "load", side_effect=AssertionError("saved source program loaded")), \
         patch.object(RefQuantLinear, "__init__", side_effect=AssertionError("quantizing constructor called")):
        program, actual_inputs = adapter_loader.get_model_and_inputs()
    assert isinstance(program, torch.export.ExportedProgram)
    assert torch.equal(actual_inputs[0], inputs)
    assert torch.equal(program.module()(*actual_inputs), expected)
    assert raw_capture.inventory(load_file(str(export / "model.safetensors"))) == before
    calls = program._m2m_external_quantization_manifest["atlas"]["calls"]
    assert {"atlas.qlinear.default", "atlas.qk_matmul.default", "atlas.pv_matmul.default"} <= {
        call["target"] for call in calls}
    assert "saved_native_program" not in program._m2m_external_quantization_manifest["atlas"]


@pytest.mark.parametrize("role", ["contract", "policy"])
def test_loader_refuses_changed_receipt_bytes_before_architecture_construction(source, role):
    export, *_ = source
    path = export / ("manifest.json" if role == "contract" else "policy.json")
    path.write_bytes(path.read_bytes() + b"\n")
    with patch("transformers.AutoModelForCausalLM.from_config", side_effect=AssertionError("model constructed")):
        with pytest.raises(ValueError, match="source receipt digest"):
            adapter_loader.get_model_and_inputs()


def test_loader_refuses_unattributed_architecture(source):
    _, architecture, *_ = source
    path = architecture / "config.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="architecture differs"):
        adapter_loader.get_model_and_inputs()


def test_loader_refuses_noninteger_token_archive(source):
    _, _, inputs_path, inputs, _ = source
    save_file({"input_ids": inputs.float()}, inputs_path)
    with pytest.raises(ValueError, match="only int64 input_ids"):
        adapter_loader.get_model_and_inputs()
