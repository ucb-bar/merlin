"""TinyLlama raw-state loader for the external adapter capture worker.

Select ATLAS_EXPORT_DIR, ATLAS_ARCHITECTURE, and ATLAS_INPUTS_SAFETENSORS.
Set ATLAS_EXPECTED_CONTRACT_SHA256 and ATLAS_EXPECTED_POLICY_SHA256 from the
authenticated source receipt. The input archive contains input_ids. Source
modules must be on PYTHONPATH.
Invoke the worker with --already-quantized --quant-adapter atlas_fp8 and the
export's manifest.json and policy.json as --contract and --policy, with the
same expected values as --expected-contract-sha256 and --expected-policy-sha256.
"""
from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

from atlas_mlir_native.capture import normalize_capture, prepare_model


def get_model_and_inputs():
    import algos
    import fx_export
    import load_quantized
    import ref_attention
    import ref_profile
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    import importlib.util
    spec = importlib.util.spec_from_file_location("_adapter_raw_capture", Path(__file__).with_name("raw_capture.py"))
    raw_capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(raw_capture)
    inventory = raw_capture.inventory
    prohibit_weight_conversion = raw_capture.prohibit_weight_conversion
    reconstruct = raw_capture.reconstruct

    export = Path(os.environ["ATLAS_EXPORT_DIR"])
    architecture = Path(os.environ["ATLAS_ARCHITECTURE"])
    expected_contract = os.environ["ATLAS_EXPECTED_CONTRACT_SHA256"]
    expected_policy = os.environ["ATLAS_EXPECTED_POLICY_SHA256"]
    for path, expected in ((export / "manifest.json", expected_contract),
                           (export / "policy.json", expected_policy)):
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"selected source receipt digest differs: {path.name}")
    # read_export authenticates files before any model reconstruction.
    tensors, policy, manifest = load_quantized.read_export(export, verify=True)
    if sha256((architecture / "config.json").read_bytes()).hexdigest() != manifest["model"]["checkpoint"]["config_json_sha256"]:
        raise ValueError("architecture differs from authenticated export")
    before = inventory(tensors)
    config = AutoConfig.from_pretrained(str(architecture), local_files_only=True)
    config.use_cache, config._attn_implementation = False, "eager"
    with prohibit_weight_conversion(), torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="eager")
    for module in model.modules():
        module._non_persistent_buffers_set.clear()
    replay = load_quantized.PolicyAlgo(policy, tensors)
    with prohibit_weight_conversion(), algos.active(replay):
        reconstruct(model, policy, tensors, replay=replay, linear_class=ref_profile.RefQuantLinear,
                    patch_attention=ref_attention.patch_attention)
    if inventory(tensors) != before:
        raise ValueError("raw source state changed during reconstruction")
    boundary = fx_export.CausalLogits(model).eval()
    inputs = load_file(os.environ["ATLAS_INPUTS_SAFETENSORS"])
    if set(inputs) != {"input_ids"} or inputs["input_ids"].dtype != torch.int64:
        raise ValueError("input archive must contain only int64 input_ids")
    with prohibit_weight_conversion(), algos.active(replay):
        return prepare_model(boundary, (inputs["input_ids"],), export / "manifest.json",
                             export / "policy.json", export_prefix="model.",
                             expected_contract_sha256=expected_contract,
                             expected_policy_sha256=expected_policy)
