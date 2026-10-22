"""Full, locally supplied TinyLlama prefill/decode session for verify_session.py.

The checkpoint is the official unmodified BF16 safetensors artifact at the pinned
revision below. ``precision="float32"`` explicitly converts those trained weights
for a CPU frontend diagnostic. It is not original-precision or model-quality
qualification. No model artifacts are downloaded and no layer count is changed.
"""

import hashlib
import json
import sys
from contextlib import contextmanager
from verify_resnet import file_identity, safe_path


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
WEIGHT_SHA256 = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"
MODEL_SOURCE = f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}"
GEOMETRY = {"model_type": "llama", "num_hidden_layers": 22,
            "hidden_size": 2048, "intermediate_size": 5632,
            "num_attention_heads": 32, "num_key_value_heads": 4,
            "vocab_size": 32000, "tie_word_embeddings": False}
MODEL_SEMANTICS = {**GEOMETRY, "hidden_act": "silu", "attention_bias": False,
                   "max_position_embeddings": 2048, "rms_norm_eps": 1e-5,
                   "rope_theta": 10000.0, "rope_scaling": None,
                   "bos_token_id": 1, "eos_token_id": 2, "pretraining_tp": 1}


def checkpoint_artifacts(checkpoint_dir):
    """Validate only the explicit local artifact set before Transformers can read it."""
    root = safe_path(checkpoint_dir)
    identities = {}
    for name in ("config.json", "model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json"):
        path = safe_path(root / name)
        if not path.is_file():
            raise FileNotFoundError(f"Required full TinyLlama artifact is absent: {path}")
        identities[name] = {**file_identity(path, safe_path), "bytes": path.stat().st_size}
    if identities["model.safetensors"]["sha256"] != WEIGHT_SHA256:
        raise ValueError("TinyLlama weights differ from the pinned official full checkpoint")
    config = json.loads(safe_path(root / "config.json").read_text())
    if any(config.get(name) != value for name, value in MODEL_SEMANTICS.items()):
        raise ValueError("TinyLlama config differs from the complete pinned model geometry")
    if config.get("torch_dtype", config.get("dtype")) != "bfloat16":
        raise ValueError("Pinned TinyLlama checkpoint configuration must declare bfloat16")
    # Transformers can optionally read these; record them when present and validate
    # their paths before the library sees the enclosing directory.
    for name in ("generation_config.json", "tokenizer.model", "added_tokens.json"):
        path = safe_path(root / name)
        if path.is_file():
            identities[name] = {**file_identity(path, safe_path), "bytes": path.stat().st_size}
    return root, identities


@contextmanager
def capture_last_token_inputs(lm):
    """Observe bounded real inputs without changing model arithmetic or state."""
    import tinyllama_integer as integer
    trace, attention, handles = {}, {}, []
    for name, module in lm.named_modules():
        if isinstance(module, integer.IntegerProjection):
            def record(projection, args, selected_name=name):
                if selected_name in trace:
                    raise AssertionError("TinyLlama diagnostic requires one invocation per named projection")
                trace[selected_name] = args[0].detach().reshape(-1, projection.in_features)[-1:].clone()
            handles.append(module.register_forward_pre_hook(record))
        elif name == "model.layers.0.self_attn":
            def record_attention(_module, _args, kwargs):
                mask = kwargs.get("attention_mask")
                attention["mask"] = None if mask is None else mask.detach()[..., -1:, :].clone()
                embeddings = kwargs.get("position_embeddings")
                attention["rope"] = None if embeddings is None else tuple(value.detach()[..., -1:, :].clone() for value in embeddings)
            handles.append(module.register_forward_pre_hook(record_attention, with_kwargs=True))
    try:
        yield trace, attention
    finally:
        for handle in handles:
            handle.remove()


def compare_cache_attention(actual, source, length, torch):
    """Compare the final token's active causal mask and RoPE in both routes."""
    result = {"observed": bool(actual) and bool(source)}
    if not result["observed"]:
        return result
    source_mask, actual_mask = source["mask"], actual["mask"]
    if source_mask is None or actual_mask is None:
        result["mask_comparison"] = "not_observed_both_routes"
    else:
        torch.testing.assert_close(actual_mask[..., :length], source_mask[..., :length], rtol=0, atol=0)
        unused = actual_mask[..., length:]
        blocked = torch.isneginf(unused) | (unused == torch.finfo(unused.dtype).min)
        if unused.numel() and not bool(blocked.all()):
            raise AssertionError("Functional TinyLlama attention exposes unused cache slots")
        result.update({"active_mask_exact": True, "unused_slots_masked": True,
                       "source_mask_shape": list(source_mask.shape), "actual_mask_shape": list(actual_mask.shape)})
    if source["rope"] is not None and actual["rope"] is not None:
        if len(actual["rope"]) != len(source["rope"]):
            raise AssertionError("Functional TinyLlama RoPE components differ")
        for actual_value, source_value in zip(actual["rope"], source["rope"]):
            torch.testing.assert_close(actual_value, source_value, rtol=0, atol=0)
        result["last_token_rope_exact"] = True
    return result


def compare_projection_traces(actual, source, lm):
    """Locate the first changed incoming float and independently encoded INT8 row."""
    import tinyllama_integer as integer
    if list(actual) != list(source):
        raise AssertionError("TinyLlama projection invocation order or coverage differs")
    checks, first_float, first_code = [], None, None
    for name in actual:
        projection = lm.get_submodule(name)
        codes, scales = projection.integer_inputs(actual[name])
        check = {"projection": name, **integer.diagnose_activation_pair(source[name].numpy(), actual[name].numpy(), projection.activation_scale,
                                                                       actual_codes=codes.numpy(), actual_scales=scales.numpy())}
        checks.append(check)
        if check["input_mismatches"] and first_float is None:
            first_float = name
        if check["code_mismatches"] and first_code is None:
            first_code = {"projection": name, **check["first_code_difference"]}
    return {"scope": "last_token_all_invoked_projections_not_complete_session_acceptance", "projections": len(checks),
            "first_float_input_difference": first_float, "first_quantized_input_difference": first_code, "checks": checks}


def validate_reference_stages(capture, causal, torch):
    """Check functional cache wrappers against the original full-sequence HF model."""
    capacity = capture.prefill_tokens + capture.decode_tokens
    prefill = causal.FixedCachePrefillStage(capture.lm, capacity).eval()
    decode = causal.FixedCacheDecodeStep(capture.lm, capacity).eval()
    errors = []
    # BF16 is a separate explicit reference option; this CPU ONNX verifier uses FP32.
    rtol = atol = 1e-4 if next(capture.lm.parameters()).dtype == torch.float32 else 0.05
    with torch.no_grad(), capture_last_token_inputs(capture.lm) as (trace, attention):
        logits, cache, position = prefill(capture.tokens[:, :capture.prefill_tokens])
        for length in range(capture.prefill_tokens, capacity + 1):
            cached_trace, cached_attention = dict(trace), dict(attention)
            trace.clear()
            attention.clear()
            positions = torch.arange(length, device=capture.tokens.device).unsqueeze(0)
            full_logits = capture.lm(
                input_ids=capture.tokens[:, :length], position_ids=positions, use_cache=False).logits[:, -1, :]
            full_trace, full_attention = dict(trace), dict(attention)
            expected_position = torch.full_like(position, length)
            if not torch.equal(position, expected_position):
                raise AssertionError("Functional TinyLlama cache used lengths are incorrect")
            if bool(cache[..., length:, :].count_nonzero()):
                raise AssertionError("Functional TinyLlama cache padding is not zero")
            attention_check = compare_cache_attention(cached_attention, full_attention, length, torch)
            try:
                torch.testing.assert_close(logits, full_logits, rtol=rtol, atol=atol)
            except AssertionError as error:
                diagnostic = {"passed": False, "failed_tokens": length, "rtol": rtol, "atol": atol,
                              "matched_tokens": capture.tokens[:, :length].tolist(), "position": position.tolist(),
                              "unused_cache_slots_zero": True, "attention": attention_check, "checks_before_failure": errors,
                              "max_absolute_logit_error": float((logits - full_logits).abs().max()),
                              "projection_inputs": compare_projection_traces(cached_trace, full_trace, capture.lm)}
                capture.provenance["functional_cache_reference"] = diagnostic
                raise AssertionError(f"TinyLlama full-sequence cache gate failed at {length} tokens: {error}") from error
            errors.append({"tokens": length, "max_absolute_logit_error":
                           float((logits - full_logits).abs().max()),
                           "position": position.tolist(), "unused_cache_slots_zero": True, "attention": attention_check})
            if length < capacity:
                before_cache, before_position = cache.clone(), position.clone()
                trace.clear()
                attention.clear()
                logits, next_cache, next_position = decode(
                    capture.tokens[:, length:length + 1], cache, position)
                if not torch.equal(cache, before_cache) or not torch.equal(position, before_position):
                    raise AssertionError("Functional TinyLlama decode mutated caller state")
                torch.testing.assert_close(next_cache[..., :length, :], cache[..., :length, :],
                                           rtol=0, atol=0)
                cache, position = next_cache, next_position
    reference = "candidate_full_sequence_hf_use_cache_false" if capture.provenance.get("arithmetic_policy", {}).get("name") in ("fixed_tree_fp32_v1", "fixed_tree_fp32_v2_pv", "fixed_tree_fp32_v3_softmax_reduce") else "original_full_sequence_hf_use_cache_false"
    return {"reference": reference, "passed": True,
            "rtol": rtol, "atol": atol, "checks": errors}


def get_model_and_inputs(*, checkpoint_dir, model2mlir_root, corpus_path,
                         prefill_tokens=8, decode_tokens=3, precision="float32",
                         cpu_threads=2, quantized=False, quality_corpus_path=None,
                         workload="frontend_diagnostic", activation_scale="power_of_two", arithmetic_policy="native", rotary_policy="native", silu_export_policy="native"):
    """Build compiled prefill plus teacher-forced recurrent decode on actual text.

    ``corpus_path`` is a local UTF-8 text file tokenized with this checkpoint's
    tokenizer. All selected IDs are recorded. Short inputs establish session and
    frontend semantics only; ``paper_ready`` remains false.
    """
    if precision not in ("float32", "bfloat16"):
        raise ValueError("Precision must be explicitly float32 or bfloat16")
    if type(quantized) is not bool or quantized and precision != "float32":
        raise ValueError("W8A8 requires explicit CPU FP32 surrounding operations")
    if activation_scale not in ("maxabs", "power_of_two"):
        raise ValueError("Unknown integer activation scale policy")
    if arithmetic_policy not in ("native", "fixed_tree_fp32_v1", "fixed_tree_fp32_v2_pv", "fixed_tree_fp32_v3_softmax_reduce"):
        raise ValueError("Unknown TinyLlama arithmetic policy")
    if rotary_policy not in ("native", "source_table_v1") or rotary_policy != "native" and precision != "float32":
        raise ValueError("Source rotary preservation requires explicit CPU FP32 and source_table_v1")
    if silu_export_policy not in ("native", "torch_avx512_sleef_v1") or silu_export_policy != "native" and (not quantized or precision != "float32" or cpu_threads != 2):
        raise ValueError("Source SiLU export preservation requires explicit W8A8 CPU FP32 and two threads")
    if arithmetic_policy != "native" and (not quantized or precision != "float32"):
        raise ValueError("Fixed tree candidate requires explicit W8A8 CPU FP32 execution")
    if quality_corpus_path is not None and not quantized:
        raise ValueError("Language quantization quality requires the explicit W8A8 recipe")
    if workload not in ("frontend_diagnostic", "development_language", "synthetic_feasibility"):
        raise ValueError("Unknown TinyLlama workload contract")
    if workload == "development_language" and (
            prefill_tokens != 128 or decode_tokens != 32 or quality_corpus_path is None):
        raise ValueError("Development language workload requires 128 prefill, 32 decode and quality corpus")
    if workload == "synthetic_feasibility" and (
            (prefill_tokens, decode_tokens) not in ((8, 3), (128, 32)) or quality_corpus_path is not None):
        raise ValueError("Synthetic feasibility requires explicit 8+3 or 128+32 and no quality corpus")
    if (type(prefill_tokens) is not int or type(decode_tokens) is not int
            or prefill_tokens < 1 or decode_tokens < 2):
        raise ValueError("Require a positive prefill and at least two recurrent decode steps")
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
        raise ValueError("cpu_threads must be between 1 and 8")
    if prefill_tokens + decode_tokens > 2048:
        raise ValueError("Session exceeds the checkpoint context capacity")
    if (arithmetic_policy in ("fixed_tree_fp32_v2_pv", "fixed_tree_fp32_v3_softmax_reduce") or rotary_policy != "native") and prefill_tokens + decode_tokens > 160:
        raise ValueError("Fixed tree PV session exceeds declared common capacity160")
    checkpoint, artifacts = checkpoint_artifacts(checkpoint_dir)
    corpus = safe_path(corpus_path)
    text = corpus.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Token corpus must contain actual text")
    corpus_identity = file_identity(corpus, safe_path)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import cache_utils
    from verify_session import load_module, load_protocol

    if not isinstance(getattr(cache_utils.StaticLayer(prefill_tokens + decode_tokens),
                              "cumulative_length", None), torch.Tensor):
        raise RuntimeError("The selected Transformers cache API is incompatible with m2m; use the prepared Transformers 5.4.0 environment")
    torch.set_num_threads(cpu_threads)
    silu_record = None
    if silu_export_policy != "native":
        import tinyllama_integer as silu_integer
        silu_record = silu_integer.source_silu_contract(torch, silu_export_policy)
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False, use_fast=True)
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    capacity = prefill_tokens + decode_tokens
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < capacity:
        raise ValueError(f"Actual token corpus must contain at least {capacity} token IDs")
    tokens = tokens[:, :capacity].contiguous()
    if tokens.dtype != torch.int64 or bool((tokens < 0).any()) or bool((tokens >= 32000).any()):
        raise ValueError("Tokenizer emitted an invalid TinyLlama token ABI")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[precision]
    lm, loading = AutoModelForCausalLM.from_pretrained(
        str(checkpoint), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=dtype, attn_implementation="eager",
        output_loading_info=True)
    if any(loading.get(name) for name in
           ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Full TinyLlama checkpoint did not load exactly")
    if any(getattr(lm.config, name, None) != value for name, value in MODEL_SEMANTICS.items()
           if name not in ("rope_theta", "rope_scaling")):
        raise ValueError("Loaded TinyLlama model geometry differs from the pinned checkpoint")
    rope = getattr(lm.config, "rope_parameters", None)
    if rope is None:
        valid_rope = lm.config.rope_theta == 10000.0 and lm.config.rope_scaling is None
    else:
        valid_rope = (isinstance(rope, dict) and rope.get("rope_type") == "default"
                      and rope.get("rope_theta") == 10000.0
                      and rope.get("partial_rotary_factor", 1.0) == 1.0)
    if not valid_rope:
        raise ValueError("Loaded TinyLlama RoPE semantics differ from the pinned checkpoint")
    if any(value.dtype != dtype or value.device.type != "cpu" for value in lm.parameters()):
        raise ValueError("Loaded model violates explicit CPU precision selection")
    lm.eval().requires_grad_(False)
    if silu_record is not None:
        from transformers.activations import SiLUActivation
        if any(not isinstance(lm.get_submodule(f"model.layers.{layer}.mlp.act_fn"), SiLUActivation) for layer in range(22)):
            raise ValueError("Source SiLU preservation requires unchanged native SiLU activation modules")
    source_parameter_count = sum(value.numel() for value in lm.parameters())
    source_parameter_bytes = sum(value.numel() * value.element_size() for value in lm.parameters())
    integer = None
    integer_records, integer_checks = [], []
    language_contract, language_check, arithmetic_record, rotary_record = None, None, None, None
    if quantized:
        import tinyllama_integer as integer
        if quality_corpus_path is not None:
            quality_tokens, language_contract = integer.language_inputs(
                quality_corpus_path, tokenizer, safe_path, file_identity, activation_scale=activation_scale, arithmetic_policy=arithmetic_policy)
            # Freeze IDs, source, selection and fixed thresholds before quantization.
            quality_reference = integer.language_reference(lm, quality_tokens)
        integer_records = integer.install_projections(lm, activation_scale=activation_scale)
        if arithmetic_policy != "native":
            arithmetic_record = integer.install_fixed_tree_arithmetic(lm, arithmetic_policy)
        if rotary_policy != "native":
            rotary_record = integer.install_source_rotary_table(lm)
        if quality_corpus_path is not None:
            language_check = integer.language_quality(
                lm, quality_tokens, quality_reference, language_contract)
            del quality_tokens, quality_reference
    if rotary_policy != "native" and rotary_record is None:
        import tinyllama_integer as rotary_integer
        rotary_record = rotary_integer.install_source_rotary_table(lm)
    model_root = safe_path(model2mlir_root)
    load_protocol(model_root)
    causal_path = safe_path(model_root / "m2m/capture/causal_session.py")
    causal = load_module("m2m.capture.causal_session", causal_path)
    provenance = {
        "checkpoint": MODEL_ID, "checkpoint_revision": REVISION,
        "checkpoint_source": MODEL_SOURCE, "full_checkpoint": True,
        "checkpoint_artifacts": artifacts, "checkpoint_tensor_dtype": "bfloat16",
        "execution_precision": precision,
        "weight_conversion": "bfloat16_to_float32" if precision == "float32" else "none",
        "reference_precision": precision, "reference_role": "compiler_correctness",
        "quality_reference": "not_evaluated", "synthetic_tokens": workload == "synthetic_feasibility",
        "token_source": "synthetic_utf8_feasibility_corpus_with_pinned_checkpoint_tokenizer"
        if workload == "synthetic_feasibility" else "local_utf8_corpus_with_pinned_checkpoint_tokenizer",
        "corpus": corpus_identity, "token_ids": tokens.tolist(),
        "token_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        "causal_session_source": file_identity(causal_path, safe_path),
        "transformers_version": transformers.__version__,
        "transformers_cache_source": file_identity(cache_utils.__file__, safe_path),
        "transformers_model_source": file_identity(sys.modules[type(lm).__module__].__file__, safe_path),
        "tokenizer_version": tokenizer.__class__.__name__,
        "decode_policy": "teacher_forced_actual_tokens",
        "parameter_count": source_parameter_count,
        "parameter_bytes": source_parameter_bytes,
        "parameter_bytes_role": "checkpoint_after_explicit_precision_selection_before_optional_quantization",
        "kv_cache_bytes": 2 * 22 * 4 * capacity * 64 * torch.empty((), dtype=dtype).element_size(),
        "position_bytes": 22 * 8, "cpu_threads": cpu_threads,
        "workload_contract": {"name": workload, "batch": 1,
                              "prefill_tokens": prefill_tokens, "decode_tokens": decode_tokens,
                              "kv_capacity": capacity, "decode_policy": "teacher_forced_actual_tokens",
                              "repeats": 2, "reset": "between_repeats",
                              "application_quality": "separate_language_corpus" if language_contract else "not_evaluated"},
    }
    if rotary_record is not None:
        provenance["rotary_policy"] = rotary_record
        provenance["rotary_source"] = file_identity(sys.modules["tinyllama_integer"].__file__, safe_path)
    if silu_record is not None:
        provenance["silu_export_policy"] = {**silu_record,
                                          "native_activation_source": file_identity(sys.modules[SiLUActivation.__module__].__file__, safe_path)}
    if integer is not None:
        provenance.update({"execution_precision": "w8a8_projections_float32_cpu_surrounding_ops",
                           "weight_conversion": "bfloat16_to_float32_then_per_channel_int8_projections",
                           "reference_precision": "source_bound_w8a8_integer_projection_reference",
                           "integer_recipe": integer.recipe(activation_scale, arithmetic_policy),
                           "integer_source": file_identity(integer.__file__, safe_path),
                           "integer_projections": integer_records,
                           "operation_placement": integer.operation_placement(),
                           "memory_plan": integer.memory_plan(lm, integer_records, prefill_tokens, decode_tokens),
                           "language_quality_contract": language_contract,
                           "language_quality": language_check or {"evaluated": False, "passed": False},
                           "quality_reference": integer.QUALITY_CONTRACT["reference"] if language_contract else "not_evaluated",
                           "checkpoint_parameter_count": 1100048384,
                           "state_dict_bytes": sum(value.numel() * value.element_size() for value in lm.state_dict().values())})
        if arithmetic_record is not None:
            provenance["arithmetic_policy"] = arithmetic_record
            provenance["reference_precision"] = "source_bound_w8a8_fixed_tree_fp32_candidate_reference"
            provenance["operation_placement"]["qk_contractions"]["arithmetic"] = "fixed_tree_fp32_v1"
            provenance["operation_placement"]["rms_norm"]["arithmetic"] = "fixed_tree_mean_squares_fp32_v1_other_norm_ops_unchanged"
            if arithmetic_policy in ("fixed_tree_fp32_v2_pv", "fixed_tree_fp32_v3_softmax_reduce"):
                provenance["operation_placement"]["pv_contractions"]["arithmetic"] = "fixed_tree_fp32_v2_pv_capacity160_tree256"
            if arithmetic_policy == "fixed_tree_fp32_v3_softmax_reduce":
                provenance["operation_placement"]["softmax"]["arithmetic"] = "fixed_tree_sum256_distinct_native_compiled_fp32_exp"
                provenance["operation_placement"]["softmax"]["runtime_admission"] = "eager_mask_checks_before_cache_update;compiled_static_source_generated_zero_or_minimum_masks_no_general_runtime_value_guard"
        # Every trained projection is checked over its full reduction dimension at
        # representative rows/columns on the first actual prefill call.
        handles = []
        for name, module in lm.named_modules():
            if isinstance(module, integer.IntegerProjection):
                def check_once(projection, inputs, selected_name=name):
                    if selected_name not in checked:
                        integer_checks.append({"name": selected_name,
                                               **integer.check_projection(projection, inputs[0])})
                        checked.add(selected_name)
                handles.append(module.register_forward_pre_hook(check_once))
        checked = set()
    capture = causal.CausalMultiProgramCapture(
        lm, tokens, checkpoint=MODEL_ID, full_checkpoint=True, paper_ready=False,
        provenance=provenance, prefill_tokens=prefill_tokens, decode_tokens=decode_tokens)
    try:
        capture.provenance["functional_cache_reference"] = validate_reference_stages(
            capture, causal, torch)
    except AssertionError as error:
        # Retain quality/source/memory evidence even when the independent cache
        # gate fails. This attaches diagnostics and never accepts the model.
        error.tinyllama_provenance = capture.provenance
        error.tinyllama_integer_projection_reference = integer_checks
        raise
    finally:
        if integer is not None:
            for handle in handles:
                handle.remove()
    if integer is not None:
        if len(integer_checks) != 155:
            raise AssertionError("The complete trained projection arithmetic was not checked")
        capture.provenance["integer_projection_reference"] = integer_checks
    return capture, ()


def get_verification_artifacts(model, _inputs):
    """Selected checkpoint/corpus/source closure for verifier pre/post file hashes."""
    provenance = model.provenance
    identities = [*provenance["checkpoint_artifacts"].values(),
                  provenance["corpus"], provenance["causal_session_source"],
                  provenance["transformers_cache_source"], provenance["transformers_model_source"]]
    if provenance.get("integer_source"):
        identities.append(provenance["integer_source"])
    if provenance.get("rotary_source") and provenance["rotary_source"] != provenance.get("integer_source"):
        identities.append(provenance["rotary_source"])
    if provenance.get("silu_export_policy"):
        identities.extend(provenance["silu_export_policy"][name] for name in ("native_library", "native_vector_header", "native_loop_header", "native_sleef_header", "native_activation_source"))
    if provenance.get("language_quality_contract"):
        identities.append(provenance["language_quality_contract"]["corpus"])
        if provenance["language_quality_contract"].get("dataset_artifact"):
            identities.append(provenance["language_quality_contract"]["dataset_artifact"])
    return tuple(safe_path(identity["path"]) for identity in identities)


def get_onnx_export_options(torch, model=None):
    """Select translations from this model's declared arithmetic policy."""
    from tinyllama_integer import export_options
    policy = model.provenance.get("arithmetic_policy", {}).get("name", "native") if model is not None else "native"
    silu_policy = model.provenance.get("silu_export_policy", {}).get("name", "native") if model is not None else "native"
    return export_options(torch, policy, silu_policy)
