"""Capture native layer-zero attention/residual; optionally substitute admitted Q.

The pinned LlamaModel owns masks, positions, RoPE, GQA and cache updates. A hook
stops before post-attention normalization, so no MLP or later layer executes.
This is a local source-fidelity experiment, never a language-quality verdict.
"""

import argparse
import inspect
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from capture_tinyllama_native import (CACHE_SHA, CORPUS_SHA, LLAMA_SHA, TOKEN_SHA,
    WEIGHT_SHA, array_identity, check_memory, identity, safe_path,
    validate_arrays, verify_declared_sources, artifact_output, load_provider_validator)


def raw_array(tensor, torch):
    tensor = tensor.detach().cpu().contiguous()
    return (tensor.view(torch.uint16) if tensor.dtype == torch.bfloat16 else tensor).numpy().copy()


def tensor_identity(tensor, torch):
    result = array_identity(raw_array(tensor, torch))
    result["native_dtype"] = str(tensor.dtype)
    return result


def admit_query(inputs, output, source, torch):
    if (inputs.shape != (1, 128, 2048) or output.shape != (1, 128, 2048) or
            inputs.dtype != torch.bfloat16 or output.dtype != torch.bfloat16 or
            not np.array_equal(raw_array(inputs[0], torch), source["A_bf16"]) or
            not np.array_equal(raw_array(output[0], torch), source["native_C_bf16"])):
        raise ValueError("Native Q invocation differs from the original full capture")


def error(actual, reference):
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Nonfinite native attention comparison")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    norm = float(np.linalg.norm(reference.astype(np.float64)))
    delta_norm = float(np.linalg.norm(delta))
    return {"relative_l2": delta_norm / norm if norm else (0.0 if not delta_norm else None),
            "zero_reference": norm == 0,
            "max_abs": float(np.abs(delta).max()),
            "allclose_rtol1e4_atol1e4": bool(np.allclose(actual, reference, rtol=1e-4, atol=1e-4)),
            "violating_values": int(np.count_nonzero(np.abs(delta) > 1e-4 + 1e-4 * np.abs(reference.astype(np.float64))))}


def evaluate(args, out, report):
    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoTokenizer, cache_utils
    from transformers.models.llama import modeling_llama

    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    if transformers.__version__ != "5.4.0":
        raise ValueError("Pinned Transformers 5.4.0 is required")
    bundle = safe_path(args.native_bundle)
    manifest_path = bundle / "manifest.json"
    manifest_id = identity(manifest_path, args.native_manifest_sha256)
    native = json.loads(manifest_path.read_text())
    site = native["site"]
    if (native["status"] != "native_selective_capture_passed" or native["source_dtype"] != "bfloat16" or
            site["module"] != "model.layers.0.self_attn.q_proj" or site["batch"] != 0 or
            site["positions"] != [0, 128] or site["output_channels"] != [0, 2048] or
            site["k_range"] != [0, 2048] or site["full_k_reduction"] is not True or
            not all(native["negative_controls"].values())):
        raise ValueError("Require the complete source-native 128-position Q projection")
    source = native["source"]
    if source["python"] != sys.version or source["torch"] != torch.__version__ or source["transformers"] != transformers.__version__:
        raise ValueError("Native interpreter/framework versions differ from source capture")
    verify_declared_sources(source)
    operands_id = identity(bundle / "operands.npz", native["npz_sha256"])
    required = {name + ".npy" for name in ("A", "B", "native_C", "A_bf16", "B_bf16", "native_C_bf16")}
    with zipfile.ZipFile(operands_id["path"]) as archive:
        members = archive.infolist()
        if len(members) != len(required) or {x.filename for x in members} != required or sum(x.file_size for x in members) > 32 * 1024**2:
            raise ValueError("Unexpected native archive members")
    with np.load(operands_id["path"], allow_pickle=False) as archive:
        operands = {key: archive[key].copy() for key in archive.files}
    validate_arrays(operands, full_query=True)
    if any(array_identity(value) != native["arrays"][key] for key, value in operands.items()):
        raise ValueError("Native operand array identities differ")
    checkpoint = safe_path(source["weights"]["path"]).parent
    corpus = safe_path(source["corpus"]["path"])
    identity(checkpoint / "model.safetensors", WEIGHT_SHA)
    identity(corpus, CORPUS_SHA)
    records = [manifest_id, operands_id, identity(__file__),
               identity(inspect.getfile(validate_arrays)), identity(modeling_llama.__file__, LLAMA_SHA),
               identity(cache_utils.__file__, CACHE_SHA), identity(inspect.getfile(modeling_llama.create_causal_mask)),
               identity(inspect.getfile(modeling_llama.LlamaConfig)),
               identity(inspect.getfile(torch.nn.Linear)), identity(Path(torch.__file__).parent / "lib/libtorch_cpu.so")]
    if identity(inspect.getfile(torch.nn.Linear)) != source["native_linear"] or identity(Path(torch.__file__).parent / "lib/libtorch_cpu.so") != source["torch_cpu_library"]:
        raise ValueError("Native CPU implementation differs from the captured source")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    tokens = tokenizer(corpus.read_text(encoding="utf-8"), return_tensors="pt", add_special_tokens=True)["input_ids"][:, :160].contiguous()
    if tokens.shape != (1, 160) or array_identity(tokens.numpy())["sha256"] != TOKEN_SHA:
        raise ValueError("Original raw-text/token positions differ")
    token_before = tokens.clone()
    config = modeling_llama.LlamaConfig.from_pretrained(str(checkpoint), local_files_only=True)
    if (config.hidden_size, config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads) != (2048, 22, 32, 4):
        raise ValueError("Original TinyLlama structure differs")
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = modeling_llama.LlamaModel(config)
    model.rotary_emb = modeling_llama.LlamaRotaryEmbedding(config=config, device="cpu")
    names = ("embed_tokens.weight", "layers.0.input_layernorm.weight") + tuple(
        "layers.0.self_attn." + name + ".weight" for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt", device="cpu") as handle:
        loaded = {"model." + name: handle.get_tensor("model." + name) for name in names}
    for name in names:
        value = loaded["model." + name]
        if value.dtype != torch.bfloat16:
            raise ValueError("Selective parameters must retain original BF16 words")
        module, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(module), attribute, torch.nn.Parameter(value, requires_grad=False))
    model.eval()
    parameters_before = {key: tensor_identity(value, torch) for key, value in loaded.items()}
    buffers_before = {key: tensor_identity(value, torch) for key, value in model.rotary_emb.named_buffers()}
    if not np.array_equal(raw_array(model.layers[0].self_attn.q_proj.weight.T, torch), operands["B_bf16"]):
        raise ValueError("Native Q weights differ from captured full B")
    layer = model.layers[0]

    class AttentionCaptured(Exception):
        pass

    def forward(replacement=None):
        observed, counts = {}, {"query": 0, "attention": 0, "stop": 0}
        cache = cache_utils.DynamicCache(config=config)
        if cache.get_seq_length() != 0:
            raise AssertionError("Fresh native cache is not empty")

        def query_hook(_module, inputs, output):
            admit_query(inputs[0], output, operands, torch)
            counts["query"] += 1
            observed["normalized_input"] = inputs[0].detach().clone()
            observed["native_query"] = output.detach().clone()
            effective = output if replacement is None else replacement.clone()
            observed["effective_query"] = effective.detach().clone()
            return effective

        def attention_pre(_module, _inputs, kwargs):
            observed["position_ids"] = kwargs["position_ids"].detach().clone()
            observed["attention_mask"] = kwargs["attention_mask"].detach().clone()
            observed["rope_cos"], observed["rope_sin"] = (value.detach().clone() for value in kwargs["position_embeddings"])

        def attention_hook(_module, _inputs, output):
            observed["attention_output"] = output[0].detach().clone()
            counts["attention"] += 1

        def stop_hook(_module, inputs):
            observed["residual_output"] = inputs[0].detach().clone()
            counts["stop"] += 1
            raise AttentionCaptured()

        hooks = [layer.self_attn.q_proj.register_forward_hook(query_hook),
                 layer.self_attn.register_forward_pre_hook(attention_pre, with_kwargs=True),
                 layer.self_attn.register_forward_hook(attention_hook),
                 layer.post_attention_layernorm.register_forward_pre_hook(stop_hook)]
        try:
            with torch.inference_mode():
                try:
                    model(input_ids=tokens[:, :128], past_key_values=cache, use_cache=True)
                except AttentionCaptured:
                    pass
                else:
                    raise AssertionError("Native model did not stop at first attention residual")
        finally:
            for hook in hooks:
                hook.remove()
        if counts != {"query": 1, "attention": 1, "stop": 1} or cache.get_seq_length() != 128:
            raise AssertionError("Wrong native invocation count or cache length")
        if any(cache.get_seq_length(index) for index in range(1, 22)):
            raise AssertionError("A later native decoder layer executed")
        observed["keys"], observed["values"] = cache.layers[0].keys.detach().clone(), cache.layers[0].values.detach().clone()
        if any(observed[key].shape != (1, 4, 128, 64) for key in ("keys", "values")):
            raise AssertionError("Native GQA cache shape differs")
        return observed

    reference = forward()
    repeated = forward()
    descriptor = lambda values: {key: tensor_identity(value, torch) for key, value in values.items()}
    if descriptor(reference) != descriptor(repeated):
        raise AssertionError("Native attention repeat changed raw bytes")
    np.savez(out / "native-attention.npz", **{key: raw_array(value, torch) for key, value in reference.items()})
    report.update(status="passed_native_attention_baseline", source_manifest=manifest_id,
                  source_npz=operands_id, native_repeat_raw_exact=True,
                  native_arrays=descriptor(reference), native_attention_npz=identity(out / "native-attention.npz"),
                  materialized_parameters=parameters_before, materialized_parameter_bytes=sum(v.numel() * v.element_size() for v in loaded.values()),
                  native_source_policies=source["source_policies"], attention_implementation="eager",
                  rotary_buffers=buffers_before,
                  scope="layer0 RMSNorm/QKV/RoPE/GQA/causal-attention/O/residual; stops before post_attention_layernorm",
                  native_model_configuration={"hidden_size": 2048, "layers": 22, "query_heads": 32, "kv_heads": 4, "positions": [0, 128]},
                  full_model_executed=False, mlp_executed=False, language_quality_claim=False)
    if args.provider_receipt:
        if safe_path(args.provider_validator) != args.producer_root / "tools/validate_model_contraction.py":
            raise ValueError("Validator must belong to selected producer root")
        validator, validator_id = load_provider_validator(
            args.producer_root, args.provider_validator_sha256,
            args.provider_receipt, args.provider_receipt_sha256)
        admitted_native, admitted_source, target, provider_records, details = validator.admit_contraction(
            receipt_path=args.provider_receipt, receipt_sha256=args.provider_receipt_sha256,
            native_bundle=bundle, native_manifest_sha256=args.native_manifest_sha256,
            producer_root=args.producer_root, model_root=args.model_root)
        if admitted_native != native or any(admitted_source[key].tobytes() != operands[key].tobytes() for key in operands):
            raise ValueError("Admitted provider source differs from native Q capture")
        words = (target.view(np.uint32) >> 16).astype(np.uint16)
        replacement = torch.from_numpy(words.copy()).view(torch.bfloat16).reshape(1, 128, 2048)
        hybrid, hybrid_repeat = forward(replacement), forward(replacement)
        if not np.array_equal(raw_array(hybrid["effective_query"], torch), words[None]):
            raise AssertionError("Actual target Q words changed at the native adapter")
        if descriptor(hybrid) != descriptor(hybrid_repeat):
            raise AssertionError("Substituted native attention repeat changed raw bytes")
        restored = forward()
        if descriptor(restored) != descriptor(reference):
            raise AssertionError("Removing substitution did not restore native attention")
        for key in ("keys", "values", "position_ids", "attention_mask", "rope_cos", "rope_sin", "normalized_input", "native_query"):
            if tensor_identity(hybrid[key], torch) != tensor_identity(reference[key], torch):
                raise AssertionError("Q substitution changed unrelated native inputs/cache")
        errors = {key: error(hybrid[key].float().numpy(), reference[key].float().numpy())
                  for key in ("attention_output", "residual_output")}
        np.savez(out / "hybrid-attention.npz", **{key: raw_array(value, torch) for key, value in hybrid.items()})
        gate = all(value["allclose_rtol1e4_atol1e4"] for value in errors.values())
        report.update(status="passed_native_attention_gate" if gate else "rejected_native_attention_gate",
                      native_attention_gate_passed=gate, provider=details, provider_validator=validator_id,
                      query_diagnostic=error(target, operands["native_C"]),
                      errors=errors, unchanged_native_inputs_and_cache=True, hybrid_repeat_raw_exact=True,
                      actual_query_adapter_raw_exact=True,
                      native_restored_after_substitution_raw_exact=True,
                      hybrid_attention_npz=identity(out / "hybrid-attention.npz"), provider_bindings=provider_records)
        validator.recheck(provider_records)
        records.append(validator_id)
    if not torch.equal(tokens, token_before) or parameters_before != {key: tensor_identity(value, torch) for key, value in loaded.items()} or buffers_before != {key: tensor_identity(value, torch) for key, value in model.rotary_emb.named_buffers()}:
        raise AssertionError("Native source inputs/parameters/rotary buffers changed")
    verify_declared_sources(source)
    for record in records:
        identity(record["path"], record["sha256"])
    report.update(source_bookends_verified=True, source_records=records, caller_inputs_preserved=True,
                  parameters_and_rotary_buffers_preserved=True, wall_seconds=time.monotonic() - started,
                  peak_rss_bytes=check_memory(), cpu_threads=1,
                  claim="Pinned local native attention experiment; no full model or language-quality qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bundle", type=safe_path, required=True)
    parser.add_argument("--native-manifest-sha256", required=True)
    parser.add_argument("--out", type=artifact_output, required=True)
    parser.add_argument("--provider-receipt", type=safe_path)
    parser.add_argument("--provider-receipt-sha256")
    parser.add_argument("--provider-validator", type=safe_path)
    parser.add_argument("--provider-validator-sha256")
    parser.add_argument("--producer-root", type=safe_path)
    parser.add_argument("--model-root", type=safe_path)
    args = parser.parse_args()
    provider = (args.provider_receipt, args.provider_receipt_sha256, args.provider_validator,
                args.provider_validator_sha256, args.producer_root, args.model_root)
    if any(provider) and not all(provider):
        parser.error("All six provider admission arguments are required together")
    args.out = artifact_output(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "argv": sys.argv, "python": sys.version}
    try:
        evaluate(args, args.out, report)
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (args.out / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "wall_seconds", "peak_rss_bytes")}))
    if report.get("native_attention_gate_passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
