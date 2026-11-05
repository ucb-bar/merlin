"""Explicit TinyLlama W8A8 projection reference and development quality contract.

This module owns a numerical recipe, not a device runtime. Learned projections
use signed int8 operands and an actual int32 contraction; attention and surrounding
model operations retain the factory's explicit CPU FP32 recipe. No calibration,
evaluation-selected scales, checkpoint clipping or source-quality claim is hidden.
"""
import hashlib
import json
import math
import re

import numpy as np
import torch
from torch import nn


RECIPE = {
    "name": "tinyllama_w8a8_dynamic_token_v1",
    "weight": "per_output_channel_maxabs_div127",
    "activation": "dynamic_per_token_maxabs_div127",
    "rounding": "nearest_ties_even",
    "range": [-127, 127], "zero_point": 0,
    "zero_scale": 1.0, "minimum_nonzero_scale": torch.finfo(torch.float32).tiny,
    "accumulator": "int32", "maximum_k": 131071,
    "dequantization": "float32(accumulator)*activation_scale*weight_scale",
    "bias": "none", "attention_and_state": "float32_cpu",
}


FIXED_TREE_POLICY = {
    "name": "fixed_tree_fp32_v1", "status": "opt_in_candidate_not_adopted",
    "domain": "merlin.fixed_order", "version": 1,
    "order": "adjacent_pairs_each_level_no_initial_zero_addition",
    "rounding": "float32_each_product_and_pair_add_no_contraction_or_reassociation",
    "mean": "multiply_tree_sum_by_exact_float32_reciprocal_power_of_two_k",
    "qualified_operands": "finite_float32_inputs_without_intermediate_overflow",
    "maximum_k": 4096, "native_query_block": 4, "inference_only": True,
    "rms_norm_scope": "mean_squares_only_rsqrt_epsilon_and_weight_multiplication_unchanged",
    "attention_scope": "qk_only_scaling_mask_softmax_and_pv_unchanged",
}


FIXED_TREE_PV_POLICY = {
    **FIXED_TREE_POLICY, "name": "fixed_tree_fp32_v2_pv",
    "attention_scope": "qk_and_pv_scaling_mask_softmax_unchanged",
    "pv_capacity": 160, "pv_tree_k": 256,
    "pv_capacity_scope": "common_bound_for_short8plus3_representative128plus32_and_quality128_not_selected_from_each_call_length",
    "pv_padding": "positive_zero_probability_and_transposed_value_extension_to_declared_tree_k_unused_cache_tails_remain_zero",
    "pv_signed_zero": "positive_zero_extension_can_canonicalize_negative_zero_not_prior_matmul_bit_preservation",
    "pv_cache_interface": "none_or_initialized_static_cache_capacity_at_most160",
}


FIXED_TREE_SOFTMAX_POLICY = {
    **FIXED_TREE_PV_POLICY, "name": "fixed_tree_fp32_v3_softmax_reduce",
    "attention_scope": "qk_pv_and_softmax_reduction_order_exp_backends_remain_distinct",
    "softmax_capacity": 160, "softmax_tree_k": 256,
    "softmax_mask": "additive_zero_or_float32_minimum_at_least_one_active_entry_per_row",
    "softmax_order": "active_logits_max_then_fp32_subtract_exp_mask_to_positive_zero_tree_sum_fp32_divide",
    "softmax_sum": "adjacent_pair_tree256_via_ordered_dot_with_exact_one_products_positive_zero_tail",
    "softmax_native_exp": "torch_exp_cpu_float32_backend_approximation",
    "softmax_compiled_exp": "onnx_Exp_tvm_cpu_float32_backend_approximation",
    "softmax_exp_equivalence": "not_bit_deterministic_or_correctly_rounded_qualified_only_by_independent_exp_and_unchanged_model_gates",
    "softmax_signed_zero": "masked_and_padded_leaves_are_positive_zero_active_equal_max_subtract_follows_fp32",
    "compiled_value_admission": "only_source_generated_static_zero_minimum_masks;native_value_checks_are_not_exported_runtime_guards",
    "native_softmax_arrays_bytes": "logits_shifted_exponent_each_B*H*M*L*4;leaves_B*H*M*256*4;ones_B*H*256*4;row_max_and_total_each_B*H*M*4;mask_broadcast_and_allocator_overhead_separate",
    "native_softmax_sum_scratch_bytes": "B*H*min(M,4)*256*4_plus_tree_and_output_temporaries",
    "compiled_softmax_sum_private_scratch_bytes": 1024,
    "softmax_target_workspace": "unqualified;graph_fusion_materialization_and_static_export_not_measured",
}


def arithmetic_contract(arithmetic_policy="native"):
    if arithmetic_policy == "native":
        return {"name": "native", "arithmetic": "unchanged_cpu_fp32_eager"}
    if arithmetic_policy == FIXED_TREE_POLICY["name"]:
        return dict(FIXED_TREE_POLICY)
    if arithmetic_policy == FIXED_TREE_PV_POLICY["name"]:
        return dict(FIXED_TREE_PV_POLICY)
    if arithmetic_policy == FIXED_TREE_SOFTMAX_POLICY["name"]:
        return dict(FIXED_TREE_SOFTMAX_POLICY)
    raise ValueError("Unknown TinyLlama arithmetic policy")


def _fixed_tree_shape(value):
    k = value.shape[-1] if value.ndim else 0
    if value.dtype != torch.float32 or value.device.type != "cpu" or not 1 <= k <= 4096 or k & (k - 1) or any(dimension < 1 for dimension in value.shape):
        raise ValueError("Fixed tree requires positive static CPU FP32 shapes with power-of-two K <= 4096")


def _pair_tree(value):
    while value.shape[-1] > 1:
        value = value[..., 0::2] + value[..., 1::2]
    return value


def _fixed_tree_dot_native(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Bounded native oracle implementation; products and tree adds are separate FP32 ops."""
    _fixed_tree_shape(left)
    _fixed_tree_shape(right)
    if left.ndim != 4 or right.ndim != 4 or left.shape[:2] != right.shape[:2] or left.shape[-1] != right.shape[-1]:
        raise ValueError("Fixed tree dot requires matching [B,H,M,K] and [B,H,N,K] operands")
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("Fixed tree requires finite operands")
    blocks = [_pair_tree(left[..., begin:begin + 4, None, :] * right[..., None, :, :]).squeeze(-1) for begin in range(0, left.shape[-2], 4)]
    result = torch.cat(blocks, dim=-2)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Fixed tree qualified operands must not overflow")
    return result


def _fixed_tree_dot_fake(left, right):
    _fixed_tree_shape(left)
    _fixed_tree_shape(right)
    if left.ndim != 4 or right.ndim != 4 or left.shape[:2] != right.shape[:2] or left.shape[-1] != right.shape[-1]:
        raise ValueError("Fixed tree dot requires matching [B,H,M,K] and [B,H,N,K] operands")
    return left.new_empty((*left.shape[:2], left.shape[-2], right.shape[-2]))


def _fixed_tree_mean_squares_native(value: torch.Tensor) -> torch.Tensor:
    _fixed_tree_shape(value)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Fixed tree requires finite operands")
    result = _pair_tree(value * value) * (1.0 / value.shape[-1])
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Fixed tree qualified operands must not overflow")
    return result


def _fixed_tree_mean_squares_fake(value):
    _fixed_tree_shape(value)
    return value.new_empty((*value.shape[:-1], 1))


_FIXED_TREE_OPS = None


def _ensure_fixed_tree_ops():
    """Register only for explicit candidate use; aliases reuse existing dispatcher ops."""
    global _FIXED_TREE_OPS
    if _FIXED_TREE_OPS is None:
        namespace = torch.ops.merlin_fixed_order
        if not hasattr(namespace, "dot"):
            dot = torch.library.custom_op("merlin_fixed_order::dot", mutates_args=())(_fixed_tree_dot_native)
            dot.register_fake(_fixed_tree_dot_fake)
        if not hasattr(namespace, "mean_squares"):
            mean = torch.library.custom_op("merlin_fixed_order::mean_squares", mutates_args=())(_fixed_tree_mean_squares_native)
            mean.register_fake(_fixed_tree_mean_squares_fake)
        _FIXED_TREE_OPS = (namespace.dot.default, namespace.mean_squares.default)
    return _FIXED_TREE_OPS


def fixed_tree_dot(left, right):
    return _ensure_fixed_tree_ops()[0](left, right)


def fixed_tree_mean_squares(value):
    return _ensure_fixed_tree_ops()[1](value)


def fixed_tree_pv(probabilities, values):
    """V2-only PV: every call uses the declared160-token bound and256-leaf tree."""
    if (probabilities.dtype != torch.float32 or values.dtype != torch.float32 or probabilities.device.type != "cpu" or values.device.type != "cpu"
            or probabilities.ndim != 4 or values.ndim != 4 or probabilities.shape[:2] != values.shape[:2] or probabilities.shape[-1] != values.shape[-2]
            or any(dimension < 1 for dimension in (*probabilities.shape, *values.shape)) or probabilities.shape[-2] > 160 or probabilities.shape[-1] > 160):
        raise ValueError("Fixed tree PV requires positive aligned CPU FP32 shapes and query/KV lengths <= declared capacity160")
    padded_probabilities = nn.functional.pad(probabilities, (0, 256 - probabilities.shape[-1]), value=0.0)
    transposed_values = values.transpose(-1, -2)
    padded_values = nn.functional.pad(transposed_values, (0, 256 - values.shape[-2]), value=0.0)
    return fixed_tree_dot(padded_probabilities, padded_values)


def validate_softmax_mask(mask):
    if mask.dtype != torch.float32 or mask.device.type != "cpu" or mask.ndim != 4 or any(extent < 1 for extent in mask.shape) or mask.shape[-1] > 160 or mask.shape[-2] > 160:
        raise ValueError("Ordered softmax requires positive CPU FP32 mask dimensions within capacity160")
    if not torch.compiler.is_compiling() and (not bool(((mask == 0) | (mask == torch.finfo(torch.float32).min)).all()) or not bool((mask == 0).any(dim=-1).all())):
        raise ValueError("Ordered softmax requires zero/minimum masks and at least one active entry per row")


def fixed_tree_softmax(logits, mask):
    """V3 fixes the sum order; native and compiled exp approximations remain distinct."""
    validate_softmax_mask(mask)
    if logits.dtype != torch.float32 or logits.device.type != "cpu" or logits.ndim != 4 or any(extent < 1 for extent in logits.shape) or logits.shape[-1] > 160 or logits.shape[-2] > 160 or mask.shape != (logits.shape[0], 1, logits.shape[-2], logits.shape[-1]):
        raise ValueError("Ordered softmax requires aligned CPU FP32 logits/mask within capacity160")
    if not torch.compiler.is_compiling() and not bool(torch.isfinite(logits).all()):
        raise ValueError("Ordered softmax requires finite logits")
    active = mask == 0
    maximum = torch.where(active, logits, torch.finfo(torch.float32).min).amax(dim=-1, keepdim=True)
    shifted = torch.where(active, logits - maximum, torch.zeros_like(logits))
    exponent = torch.where(active, torch.exp(shifted), torch.zeros_like(logits))
    leaves = nn.functional.pad(exponent, (0, 256 - logits.shape[-1]), value=0.0)
    ones = torch.ones((logits.shape[0], logits.shape[1], 1, 256), dtype=torch.float32, device=logits.device)
    total = fixed_tree_dot(leaves, ones)
    return exponent / total


class SourceRotaryTable(nn.Module):
    """Freeze source-generated words; exporter sees literal buffers and Gather only."""
    def __init__(self, original):
        super().__init__()
        settings = getattr(original.config, "rope_parameters", None)
        if not isinstance(settings, dict) or settings.get("rope_type") != "default" or settings.get("rope_theta") != 10000.0 or settings.get("partial_rotary_factor", 1.0) != 1.0 or original.inv_freq.dtype != torch.float32 or original.inv_freq.device.type != "cpu":
            raise ValueError("Source rotary table requires unscaled default10000 CPU FP32 RoPE")
        positions = torch.arange(160, dtype=torch.int64).reshape(1, -1)
        value = torch.zeros((1, 160, original.config.hidden_size), dtype=torch.float32)
        with torch.no_grad():
            cos, sin = original(value, positions)
            for selected in (positions[:, :8], positions[:, :128], positions, *(positions[:, index:index + 1] for index in range(160))):
                actual = original(value[:, :selected.shape[-1]], selected)
                if any(not torch.equal(output.contiguous().view(torch.int32), table[:, selected[0]].contiguous().view(torch.int32)) for output, table in zip(actual, (cos, sin))):
                    raise ValueError("Native RoPE words depend on full/cached call shape; table is not source-faithful")
        self.register_buffer("cos_table", cos[0].contiguous())
        self.register_buffer("sin_table", sin[0].contiguous())
        self.training = original.training

    def forward(self, value, position_ids):
        if value.dtype != torch.float32 or value.device.type != "cpu" or position_ids.dtype != torch.int64 or position_ids.device.type != "cpu" or position_ids.ndim != 2 or position_ids.shape[0] != 1 or not 1 <= position_ids.shape[-1] <= 160:
            raise ValueError("Source rotary table requires batch1 CPU FP32/int64 positions within capacity160")
        if not torch.compiler.is_compiling() and bool(((position_ids < 0) | (position_ids >= 160)).any()):
            raise ValueError("Source rotary table positions must be within0:159")
        return self.cos_table[position_ids], self.sin_table[position_ids]


def install_source_rotary_table(lm):
    table = SourceRotaryTable(lm.model.rotary_emb)
    lm.model.rotary_emb = table
    return {"name": "source_table_v1", "status": "opt_in_export_preservation_not_default", "positions": [0, 159],
            "source": "original_loaded_native_RoPE_forward_position0_to159", "export": "literal_FP32_model_buffers_Gather_by_int64_positions_no_trigonometric_recomputation",
            "shape_invariance_checked": "full8_full128_full160_all160_single_cached_positions_raw_exact",
            "compiled_value_admission": "only_source_generated_int64_positions0_to159;eager_value_checks_are_not_exported_runtime_guards",
            "cos": identity(table.cos_table), "sin": identity(table.sin_table), "buffer_bytes": 2 * table.cos_table.numel() * 4}


class FixedTreeRMSNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.variance_epsilon = original.variance_epsilon
        self.training = original.training

    def forward(self, hidden_states):
        variance = fixed_tree_mean_squares(hidden_states)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


def fixed_tree_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    from transformers.models.llama.modeling_llama import repeat_kv
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    weights = fixed_tree_dot(query, key_states) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    weights = fixed_tree_softmax(weights, attention_mask) if getattr(module, "arithmetic_policy", None) == "fixed_tree_fp32_v3_softmax_reduce" else nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = nn.functional.dropout(weights, p=dropout, training=module.training)
    output = (fixed_tree_pv(weights, value_states) if getattr(module, "arithmetic_policy", "fixed_tree_fp32_v1") in ("fixed_tree_fp32_v2_pv", "fixed_tree_fp32_v3_softmax_reduce") else torch.matmul(weights, value_states)).transpose(1, 2).contiguous()
    return output, weights


class FixedTreeAttention(nn.Module):
    """Candidate-only eager forward; model mask construction remains on the eager path."""
    def __init__(self, original, arithmetic_policy="fixed_tree_fp32_v1"):
        super().__init__()
        if arithmetic_policy not in ("fixed_tree_fp32_v1", "fixed_tree_fp32_v2_pv", "fixed_tree_fp32_v3_softmax_reduce"):
            raise ValueError("Fixed tree attention requires an explicit candidate policy")
        self.arithmetic_policy = arithmetic_policy
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, getattr(original, name))
        for name in ("config", "layer_idx", "head_dim", "num_key_value_groups", "scaling", "attention_dropout", "is_causal"):
            setattr(self, name, getattr(original, name))
        self.training = original.training

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if attention_mask is None:
            raise ValueError("Fixed tree static inference requires an explicit causal attention mask")
        if self.arithmetic_policy in ("fixed_tree_fp32_v2_pv", "fixed_tree_fp32_v3_softmax_reduce"):
            if (hidden_states.ndim != 3 or hidden_states.dtype != torch.float32 or hidden_states.device.type != "cpu" or hidden_states.shape[-1] != self.config.hidden_size or not 1 <= hidden_states.shape[-2] <= 160
                    or attention_mask.ndim != 4 or attention_mask.dtype != torch.float32 or attention_mask.device.type != "cpu" or not 1 <= attention_mask.shape[-1] <= 160
                    or attention_mask.shape[:3] != (hidden_states.shape[0], 1, hidden_states.shape[-2])):
                raise ValueError("Fixed tree PV query/mask exceeds declared capacity160")
            if past_key_values is None:
                if attention_mask.shape[-1] != hidden_states.shape[-2]:
                    raise ValueError("Fixed tree PV requires a matching full-sequence mask within capacity160")
            else:
                layers = getattr(past_key_values, "layers", ())
                if not 0 <= self.layer_idx < len(layers):
                    raise ValueError("Fixed tree PV requires an initialized static cache within capacity160")
                layer = layers[self.layer_idx]
                capacity = getattr(layer, "max_cache_len", None)
                keys = getattr(layer, "keys", None)
                cache_values = getattr(layer, "values", None)
                if (type(capacity) is not int or not 1 <= capacity <= 160 or not isinstance(keys, torch.Tensor) or not isinstance(cache_values, torch.Tensor)
                        or keys.dtype != torch.float32 or cache_values.dtype != torch.float32 or keys.device.type != "cpu" or cache_values.device.type != "cpu"
                        or keys.shape != (hidden_states.shape[0], self.config.num_key_value_heads, capacity, self.head_dim) or cache_values.shape != keys.shape or attention_mask.shape[-1] != capacity):
                    raise ValueError("Fixed tree PV requires an initialized static cache within capacity160")
        if self.arithmetic_policy == "fixed_tree_fp32_v3_softmax_reduce":
            if self.training:
                raise ValueError("Ordered softmax candidate is inference-only")
            validate_softmax_mask(attention_mask)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        output, weights = fixed_tree_attention(self, query, key, value, attention_mask, scaling=self.scaling, dropout=0.0 if not self.training else self.attention_dropout, **kwargs)
        output = self.o_proj(output.reshape(*input_shape, -1).contiguous())
        return output, weights


def install_fixed_tree_arithmetic(lm, arithmetic_policy="fixed_tree_fp32_v1"):
    """Isolate the candidate on the supplied model, retaining trained norm parameters."""
    if arithmetic_policy not in ("fixed_tree_fp32_v1", "fixed_tree_fp32_v2_pv", "fixed_tree_fp32_v3_softmax_reduce"):
        raise ValueError("Fixed tree installer requires an explicit candidate policy")
    from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaAttention
    names = [f"model.layers.{index}.{name}" for index in range(22) for name in ("input_layernorm", "post_attention_layernorm")] + ["model.norm"]
    norms = {name: module for name, module in lm.named_modules() if isinstance(module, LlamaRMSNorm)}
    attentions = {name: module for name, module in lm.named_modules() if isinstance(module, LlamaAttention)}
    if set(norms) != set(names) or any(module.weight.dtype != torch.float32 or module.weight.device.type != "cpu" or module.weight.shape != (2048,) for module in norms.values()):
        raise ValueError("Fixed tree candidate requires the complete 45 CPU FP32 TinyLlama norms")
    if set(attentions) != {f"model.layers.{index}.self_attn" for index in range(22)} or lm.config._attn_implementation != "eager":
        raise ValueError("Fixed tree candidate requires the complete 22 eager TinyLlama attention modules")
    for name in names:
        parent, _, leaf = name.rpartition(".")
        setattr(lm.get_submodule(parent), leaf, FixedTreeRMSNorm(norms[name]))
    for name, module in attentions.items():
        parent, _, leaf = name.rpartition(".")
        setattr(lm.get_submodule(parent), leaf, FixedTreeAttention(module, arithmetic_policy))
    metadata = {**arithmetic_contract(arithmetic_policy), "rms_norms": len(names), "attention_layers": 22,
            "native_qk_product_scratch_bytes": "B*H*min(M,4)*N*K*4_per_block",
            "native_qk_output_bytes": "B*H*M*N*4_plus_block_results_during_concatenation",
            "native_qk_tree_and_output_upper_bound_bytes": "1.5*product_scratch_bytes+2*output_bytes_excluding_inputs_finiteness_checks_and_allocator_overhead",
            "native_rms_scratch_bytes": "input_elements*4_plus_halving_tree_temporaries",
            "cost": "same_arithmetic_count_tree_elementwise_kernels_instead_of_blas_qk;not_timing_qualified"}
    if arithmetic_policy in ("fixed_tree_fp32_v2_pv", "fixed_tree_fp32_v3_softmax_reduce"):
        metadata.update({"native_pv_product_scratch_bytes": "B*H*min(M,4)*D*256*4_per_block",
                         "native_pv_output_bytes": "B*H*M*D*4_plus_block_results_and_output_transpose",
                         "native_pv_padding_bytes": "B*H*M*256*4+B*H*D*256*4_excluding_inputs_and_repeat_kv",
                         "compiled_pv_helper_scratch_bytes": 256 * 4,
                         "cost": "qk_and_pv_use_tree_elementwise_kernels;pv_padding_increases_products_to256_per_output;not_timing_qualified"})
    return metadata


def recipe(activation_scale="maxabs", arithmetic_policy="native"):
    if activation_scale not in ("maxabs", "power_of_two"):
        raise ValueError("Unknown integer activation scale policy")
    selected = dict(RECIPE) if activation_scale == "maxabs" else {**RECIPE, "name": "tinyllama_w8a8_dynamic_token_pot_v1",
            "activation": "dynamic_per_token_ceil_power_of_two_maxabs_div127",
            "scale_policy": "smallest_float32_normal_power_of_two_at_least_maxabs_div127"}
    contract = arithmetic_contract(arithmetic_policy)
    if arithmetic_policy != "native":
        selected = {**selected, "name": selected["name"] + "_" + contract["name"], "arithmetic_policy": contract}
    return selected
QUALITY_CONTRACT = {
    "name": "tinyllama_language_development_v1",
    "reference": "unchanged_checkpoint_explicit_bfloat16_to_float32_cpu_eager",
    "minimum_documents": 8, "tokens_per_document": 128,
    "maximum_mean_nll_increase_nats": 0.20,
    "minimum_next_token_top1_agreement": 0.80,
    "purpose": "development_quantization_loss_check_not_paper_language_benchmark",
    "selection": "first128_token_ids_each_document_in_declared_order_no_filtering",
    "session": {"batch": 1, "prefill_tokens": 128, "decode_tokens": 32,
                "kv_capacity": 160, "repeats": 2, "reset": "between_repeats"},
}
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-2-raw-v1/validation-00000-of-00001.parquet"
WIKITEXT_SHA256 = "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c"
WIKITEXT_SOURCE = f"https://huggingface.co/datasets/Salesforce/wikitext/blob/{WIKITEXT_REVISION}/{WIKITEXT_FILE}"


def prepare_wikitext_corpus(parquet_path, checkpoint_dir, output_dir):
    """Prepare the frozen development corpus from one user-supplied pinned artifact.

    Selection depends only on dataset order and pinned tokenizer length: first eight
    Wikipedia articles with at least160 IDs. Quality scores consume first128 IDs per
    article; the first article also supplies the128+32teacher-forced session text.
    No model logits or quantization feedback participate in input selection.
    """
    from verify_resnet import safe_path, file_identity
    source = safe_path(parquet_path)
    artifact = file_identity(source, safe_path)
    if artifact["sha256"] != WIKITEXT_SHA256:
        raise ValueError("WikiText validation artifact differs from the pinned source bytes")
    from transformers import AutoTokenizer
    import pyarrow.parquet as parquet
    checkpoint = safe_path(checkpoint_dir)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        safe_path(checkpoint / name)
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    table = parquet.read_table(source, columns=["text"])
    articles, current, title, start = [], [], None, 0
    for row, text in enumerate(table.column("text").to_pylist()):
        if not isinstance(text, str):
            raise ValueError("Pinned WikiText text column contains a non-string value")
        if re.fullmatch(r"\s*=\s+[^=].*[^=]\s+=\s*", text):
            if title is not None:
                articles.append((start, title, "".join(current)))
            title, start, current = text.strip(), row, [text]
        elif title is not None:
            current.append(text)
    if title is not None:
        articles.append((start, title, "".join(current)))
    documents = []
    for row, title, text in articles:
        if len(tokenizer(text, add_special_tokens=True)["input_ids"]) >= 160:
            documents.append({"id": f"validation_article_row_{row}", "text": text,
                              "source": WIKITEXT_SOURCE + f"#article_start_row={row}", "title": title})
            if len(documents) == 8:
                break
    if len(documents) != 8:
        raise ValueError("Pinned validation corpus lacks eight eligible articles")
    payload = {"source": WIKITEXT_SOURCE,
               "selection": "first8_top_level_articles_in_parquet_order_with_at_least160_pinned_tokenizer_ids;first128_ids_each_for_quality",
               "dataset_artifact": artifact, "dataset_revision": WIKITEXT_REVISION,
               "dataset_config": "wikitext-2-raw-v1", "split": "validation",
               "documents": documents}
    destination = safe_path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    quality = safe_path(destination / "language-quality.json")
    session = safe_path(destination / "session.txt")
    quality.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    session.write_text(documents[0]["text"], encoding="utf-8")
    return {"quality_corpus_path": str(quality), "corpus_path": str(session),
            "workload": "development_language", "prefill_tokens": 128,
            "decode_tokens": 32, "quantized": True}


def identity(value):
    value = value.detach().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": value.numel() * value.element_size(),
            "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def independent_quantize_rows(value, activation_scale="maxabs"):
    """Independent encoder: frexp scale selection and explicit floor/parity RNE.

    Division retains the recipe's FP32 rounding before integer rounding. Neither
    Torch round/log2 nor NumPy rint supplies the integer rounding oracle.
    """
    value = np.asarray(value)
    if value.dtype != np.float32 or value.ndim < 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("Encoding oracle requires a nonempty finite FP32 tensor")
    maximum = np.max(np.abs(value), axis=-1, keepdims=True)
    scale = np.where(maximum == 0, np.float32(1),
                     np.maximum(maximum / np.float32(127), np.finfo(np.float32).tiny))
    if activation_scale == "power_of_two":
        # Frexp is independent of the exported log2/pow expression and is exact for
        # finite float32 input, including immediately adjacent dyadic boundaries.
        mantissa, exponent = np.frexp(scale)
        scale = np.ldexp(np.ones_like(scale), exponent - (mantissa == np.float32(.5)))
    elif activation_scale != "maxabs":
        raise ValueError("Unknown integer activation scale policy")
    ratio = (value / scale).astype(np.float64)
    lower = np.floor(ratio)
    fraction = ratio - lower
    rounded = lower + ((fraction > .5) | ((fraction == .5) & (np.remainder(lower, 2) != 0)))
    return np.clip(rounded, -127, 127).astype(np.int8), scale


def check_operand_encoding(value, codes, scales, activation_scale="maxabs"):
    """Check emitted codes and scales against actual inputs, before any dot check."""
    value, codes, scales = np.asarray(value), np.asarray(codes), np.asarray(scales)
    expected_codes, expected_scales = independent_quantize_rows(value, activation_scale)
    if codes.dtype != np.int8 or codes.shape != value.shape:
        raise ValueError("Emitted activation codes must have exact INT8 shape")
    if scales.dtype != np.float32 or scales.shape != expected_scales.shape or not np.isfinite(scales).all():
        raise ValueError("Emitted activation scales must have exact finite FP32 shape")
    if not np.array_equal(scales, expected_scales):
        raise AssertionError("Activation scale encoding disagrees with independent oracle")
    if not np.array_equal(codes, expected_codes):
        raise AssertionError("Activation code rounding/clipping disagrees with independent oracle")
    return {"passed": True, "oracle": "numpy_fp32_division_frexp_floor_parity_rne",
            "checked_codes": int(codes.size), "checked_scales": int(scales.size),
            "rounding": "nearest_ties_even", "range": [-127, 127], "scale_policy": activation_scale}


def diagnose_activation_pair(source, actual, activation_scale="power_of_two", *, actual_codes=None, actual_scales=None, maximum_elements=160 * 5632):
    """Bounded local encoder checks and propagated input/threshold diagnostics.

    This reports a changed quantized program, never a model acceptance verdict.
    Supplying emitted codes/scales checks an implementation on its actual inputs.
    """
    source, actual = np.asarray(source), np.asarray(actual)
    if type(maximum_elements) is not int or maximum_elements < 1 or source.shape != actual.shape or source.size > maximum_elements:
        raise ValueError("Activation diagnostic requires matching shapes within its element budget")
    source_codes, source_scales = independent_quantize_rows(source, activation_scale)
    expected_codes, expected_scales = independent_quantize_rows(actual, activation_scale)
    if (actual_codes is None) != (actual_scales is None):
        raise ValueError("Actual emitted codes and scales must be supplied together")
    actual_codes = expected_codes if actual_codes is None else np.asarray(actual_codes)
    actual_scales = expected_scales if actual_scales is None else np.asarray(actual_scales)
    local = check_operand_encoding(actual, actual_codes, actual_scales, activation_scale)
    differences = actual_codes != source_codes
    result = {"scope": "local_actual_input_encoding_and_propagated_source_input",
              "source_input": identity(torch.from_numpy(source.copy())),
              "actual_input": identity(torch.from_numpy(actual.copy())),
              "local_actual_encoding": local,
              "input_mismatches": int(np.count_nonzero(source != actual)),
              "max_absolute_input_error": float(np.max(np.abs(source.astype(np.float64) - actual.astype(np.float64)))),
              "scale_mismatches": int(np.count_nonzero(source_scales != actual_scales)),
              "code_mismatches": int(np.count_nonzero(differences)), "first_code_difference": None}
    if np.any(differences):
        index = tuple(int(v) for v in np.argwhere(differences)[0])
        scale_index = (*index[:-1], 0)
        source_ratio = float(np.float32(source[index] / source_scales[scale_index]))
        actual_ratio = float(np.float32(actual[index] / actual_scales[scale_index]))
        boundary = math.floor(min(source_ratio, actual_ratio)) + .5
        result["first_code_difference"] = {"index": list(index), "source_code": int(source_codes[index]),
                                          "actual_code": int(actual_codes[index]), "source_input": float(source[index]),
                                          "actual_input": float(actual[index]), "source_scale": float(source_scales[scale_index]),
                                          "actual_scale": float(actual_scales[scale_index]), "source_ratio": source_ratio,
                                          "actual_ratio": actual_ratio, "nearest_lower_half_integer": boundary,
                                          "same_scale_crosses_rounding_boundary": bool(source_scales[scale_index] == actual_scales[scale_index]
                                                                                       and min(source_ratio, actual_ratio) <= boundary <= max(source_ratio, actual_ratio))}
    return result


def quantize_rows(value, activation_scale="maxabs"):
    """Symmetric row-wise quantization; finite zero/subnormal rows are well defined."""
    maximum = value.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum == 0, torch.ones_like(maximum),
                        (maximum / 127.0).clamp(min=torch.finfo(torch.float32).tiny))
    if activation_scale == "power_of_two":
        power = torch.pow(2.0, torch.ceil(torch.log2(scale)))
        # Float32 log2 can round a just-above-dyadic input down to an integer.
        # Correct that boundary using the original scale, with no model-specific rule.
        scale = torch.where(power < scale, power * 2.0, power)
    elif activation_scale != "maxabs":
        raise ValueError("Unknown integer activation scale policy")
    quantized = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


class IntegerProjection(nn.Module):
    """OI FP32 learned weight becomes one contiguous KN int8 constant and N scales."""
    def __init__(self, weight, activation_scale="maxabs"):
        super().__init__()
        recipe(activation_scale)
        self.activation_scale = activation_scale
        if (weight.ndim != 2 or weight.dtype != torch.float32 or weight.device.type != "cpu"
                or not bool(torch.isfinite(weight).all())):
            raise ValueError("Integer projection requires a finite CPU FP32 weight matrix")
        self.out_features, self.in_features = weight.shape
        if not 1 <= self.in_features <= RECIPE["maximum_k"]:
            raise ValueError("Integer projection reduction exceeds the signed-int32 bound")
        packed, scales = quantize_rows(weight.detach())
        self.register_buffer("weight_kn", packed.t().contiguous())
        self.register_buffer("weight_scale", scales.flatten().contiguous())

    def integer_inputs(self, value):
        if not torch.compiler.is_compiling():
            if (value.dtype != torch.float32 or value.device.type != "cpu"
                    or value.ndim < 1 or value.shape[-1] != self.in_features
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Integer projection requires finite CPU FP32 activations with matching K")
        quantized, scale = quantize_rows(value, self.activation_scale)
        return quantized.reshape(-1, self.in_features).contiguous(), scale

    def forward(self, value):
        quantized, scale = self.integer_inputs(value)
        accumulator = torch._int_mm(quantized, self.weight_kn)
        result = accumulator.reshape(*value.shape[:-1], self.out_features).to(torch.float32)
        return result * scale * self.weight_scale


SOURCE_SILU_ARTIFACT_HASHES = {
    "native_library": "4207b9ed0c3c86b5a99150637aef72e26e49a0629f02faf21f37313505b09d6e",
    "native_vector_header": "0150e7eebb07a5235f19ce5c759368ac3c3bf93f65ff459632a30bcf99d5896c",
    "native_loop_header": "a6f2ab2e6e7175454f1c4226994177cd2e934d844fe89b09dc049e7a77a7872d",
    "native_sleef_header": "2955f7429d04a05c5534a634f1aa77519510c90c603b4cbb4afa29d01e9a5923",
}


def _source_silu_artifacts(torch_module):
    from pathlib import Path
    from verify_resnet import file_identity, safe_path
    root = safe_path(Path(torch_module.__file__).parent)
    names = {"native_library": "lib/libtorch_cpu.so", "native_vector_header": "include/ATen/cpu/vec/vec512/vec512_float.h",
             "native_loop_header": "include/ATen/native/cpu/Loops.h", "native_sleef_header": "include/sleef.h"}
    return {name: file_identity(root / relative, safe_path) for name, relative in names.items()}


def source_silu_contract(torch_module, policy="native"):
    """Admit an export-only reproduction of the bound native vector SiLU kernel."""
    if policy == "native":
        return None
    if policy != "torch_avx512_sleef_v1":
        raise ValueError("Unknown source SiLU export policy")
    if (torch_module.version.git_version != "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
            or torch_module.backends.cpu.get_cpu_capability() != "AVX512" or torch_module.get_num_threads() != 2):
        raise ValueError("Source SiLU requires pinned PyTorch AVX512 CPU dispatch and two threads")
    artifacts = _source_silu_artifacts(torch_module)
    if any(artifacts[name]["sha256"] != expected for name, expected in SOURCE_SILU_ARTIFACT_HASHES.items()):
        raise ValueError("Source SiLU requires the independently proved native library and header identities")
    return {"name": policy, "status": "opt_in_source_preserving_export_not_qualified",
            "native": "unchanged_aten_silu_cpu_float32_contiguous_Tiny5632_columns_two_threads_no_scalar_tail",
            "domain": "merlin.source_preserving", "version": 1, "operator": "TorchSiluAvx512F32",
            "torch_git": torch_module.version.git_version, "cpu_capability": "AVX512", "cpu_threads": 2,
            "native_loop": "paired_16_word_vectors;admitted_contiguous_chunks_divisible_by32;scalar_std_exp_tail_not_admitted",
            "exp_source": "https://github.com/shibatch/sleef/blob/3.8/src/libm/sleefsimdsp.c#L1856",
            "silu_source": "https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/Activation.cpp#L1098",
            "exp_algorithm": "sleef3.8_avx512_u10_fp32_RNE_range_reduction_eight_explicit_FMA_separate_square_add_two_exponent_scaling_products",
            "silu_algorithm": "FP32_x_div_FP32_one_plus_source_exp_negative_x;no_reciprocal_multiply",
            "compiled_nonfinite": "sanitize_NaN_before_int_cast;NaN_output;infinity_classification_only_no_payload_equality_claim",
            "compiled_value_admission": "source_session_finite_activations_indirectly_checked_by_down_projection_encoder;no_direct_SiLU_input_or_exported_general_runtime_refusal_guard",
            "lowering": "LLVM_CPU_explicit_llvm_fma;no_Torch_or_Sleef_runtime_dependency;C_codegen_unsupported",
            "correct_rounding": "no_universal_correctly_rounded_exp_or_silu_claim",
            "workspace": "one_FP32_output_same_shape;scalar_local_temporaries_no_private_heap_scratch;target_cost_and_execution_unqualified", **artifacts}


def export_options(torch_module, arithmetic_policy="native", silu_export_policy="native"):
    """Modern exporter preserves signed-int8 operands with standard MatMulInteger."""
    from onnxscript import opset18 as op
    from onnxscript.onnx_types import INT8, INT32, FLOAT
    from onnxscript.values import Opset, Op
    from onnx import defs
    ordered = Opset("merlin.fixed_order", 1)
    def ordered_op(name, inputs):
        schema = defs.OpSchema(name, "merlin.fixed_order", 1, inputs=[defs.OpSchema.FormalParameter(value, "T") for value in inputs], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(float)"], "Declared ordered FP32 arithmetic")])
        return Op(ordered, name, schema)
    dot_op = ordered_op("FixedTreeDot", ("a", "b"))
    mean_op = ordered_op("FixedTreeMeanSquares", ("a",))

    def integer_mm(a: INT8, b: INT8) -> INT32:
        return op.MatMulInteger(a, b)

    arithmetic_contract(arithmetic_policy)
    translations = {torch_module.ops.aten._int_mm.default: integer_mm}
    if source_silu_contract(torch_module, silu_export_policy) is not None:
        source = Opset("merlin.source_preserving", 1)
        schema = defs.OpSchema("TorchSiluAvx512F32", "merlin.source_preserving", 1, inputs=[defs.OpSchema.FormalParameter("a", "T")], outputs=[defs.OpSchema.FormalParameter("result", "T")], type_constraints=[("T", ["tensor(float)"], "Bound source FP32 SiLU")])
        silu_op = Op(source, "TorchSiluAvx512F32", schema)
        def source_silu(self: FLOAT) -> FLOAT:
            return silu_op(self)
        translations[torch_module.ops.aten.silu.default] = source_silu
    if arithmetic_policy == "native":
        return {"custom_translation_table": translations}
    _ensure_fixed_tree_ops()

    def ordered_dot(a: FLOAT, b: FLOAT) -> FLOAT:
        return dot_op(a, b)

    def ordered_mean_squares(a: FLOAT) -> FLOAT:
        return mean_op(a)

    translations.update({torch_module.ops.merlin_fixed_order.dot.default: ordered_dot, torch_module.ops.merlin_fixed_order.mean_squares.default: ordered_mean_squares})
    return {"custom_translation_table": translations}


def projection_names():
    return [f"model.layers.{layer}.{group}.{name}_proj"
            for layer in range(22)
            for group, names in (("self_attn", ("q", "k", "v", "o")),
                                 ("mlp", ("gate", "up", "down")))
            for name in names] + ["lm_head"]


def projection_shape(name):
    if name == "lm_head":
        return 32000, 2048
    leaf = name.rsplit(".", 1)[-1]
    return {"q_proj": (2048, 2048), "k_proj": (256, 2048), "v_proj": (256, 2048),
            "o_proj": (2048, 2048), "gate_proj": (5632, 2048),
            "up_proj": (5632, 2048), "down_proj": (2048, 5632)}[leaf]


def install_projections(lm, activation_scale="maxabs"):
    """Replace exactly the complete 155 learned contractions, never an attention op."""
    linears = {name: module for name, module in lm.named_modules() if isinstance(module, nn.Linear)}
    expected = projection_names()
    if (set(linears) != set(expected)
            or any(module.bias is not None or tuple(module.weight.shape) != projection_shape(name)
                   for name, module in linears.items())):
        raise ValueError("TinyLlama must expose exactly 155 complete bias-free learned projections")
    records = []
    for name in expected:
        original = linears[name]
        projection = IntegerProjection(original.weight, activation_scale=activation_scale).eval()
        channels = np.unique(np.linspace(0, projection.out_features - 1,
                                         min(7, projection.out_features), dtype=int))
        q_expected, scale_expected = independent_quantize_rows(original.weight.detach().numpy()[channels])
        if (not np.array_equal(projection.weight_kn.numpy()[:, channels].T, q_expected)
                or not np.array_equal(projection.weight_scale.numpy()[channels], scale_expected.flatten())):
            raise AssertionError("Trained weight quantization disagrees with independent NumPy oracle")
        parent, _, leaf = name.rpartition(".")
        setattr(lm.get_submodule(parent) if parent else lm, leaf, projection)
        records.append({"name": name, "source_weight": identity(original.weight),
                        "packed_weight": identity(projection.weight_kn),
                        "weight_scale": identity(projection.weight_scale),
                        "input_k": projection.in_features, "output_n": projection.out_features,
                        "weight_quantization_reference": {"passed": True, "oracle": "numpy_fp32_division_floor_parity_rne",
                                                          "channels": channels.tolist(), "full_k": projection.in_features},
                        "placement": "gemmini_int8_int32_candidate"})
        # Release original FP32 projection promptly rather than retaining a second checkpoint.
        linears[name] = None
    return records


def check_projection(projection, value, *, rows=3, columns=7):
    """Independent NumPy int64 dot checks sampled outputs over the complete K axis."""
    if type(rows) is not int or type(columns) is not int or rows < 1 or columns < 1:
        raise ValueError("Projection reference requires positive integer row/column budgets")
    quantized, _scale = projection.integer_inputs(value)
    row_indices = np.unique(np.linspace(0, quantized.shape[0] - 1,
                                        min(rows, quantized.shape[0]), dtype=int))
    column_indices = np.unique(np.linspace(0, projection.out_features - 1,
                                           min(columns, projection.out_features), dtype=int))
    a = quantized.numpy()[row_indices].astype(np.int64)
    encoding = check_operand_encoding(value.detach().reshape(-1, projection.in_features).numpy()[row_indices],
                                      quantized.numpy()[row_indices], _scale.detach().reshape(-1, 1).numpy()[row_indices], projection.activation_scale)
    b = projection.weight_kn.numpy()[:, column_indices].astype(np.int64)
    expected = a @ b
    actual = torch._int_mm(quantized, projection.weight_kn).numpy()[np.ix_(row_indices, column_indices)]
    if not np.array_equal(actual.astype(np.int64), expected):
        raise AssertionError("Native integer contraction disagrees with independent int64 reference")
    return {"passed": True, "oracle": "numpy_int64_complete_k_sampled_rows_columns",
            "k": projection.in_features, "rows": row_indices.tolist(),
            "columns": column_indices.tolist(), "checked_elements": int(expected.size),
            "max_absolute_error": 0, "activation_encoding": encoding}


def language_inputs(path, tokenizer, safe_path, file_identity, activation_scale="maxabs", arithmetic_policy="native"):
    """Freeze an attributed local corpus before any quality inference or quantization.

    JSON owns source/selection and documents [{id, text, source}]. This declaration
    and its content hash are recorded; attribution truth is not automatically proven.
    The corpus cannot silently collapse into the short frontend diagnostic.
    """
    path = safe_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if (not isinstance(payload.get("source"), str) or not payload["source"].strip()
            or not isinstance(payload.get("selection"), str) or not payload["selection"].strip()
            or not isinstance(documents, list) or len(documents) < QUALITY_CONTRACT["minimum_documents"]):
        raise ValueError("Language quality requires an attributed corpus with at least eight documents")
    ids, digests, tokens, records = set(), set(), [], []
    count = QUALITY_CONTRACT["tokens_per_document"]
    for document in documents:
        if (not isinstance(document, dict) or not isinstance(document.get("id"), str)
                or not document["id"].strip() or document["id"] in ids
                or not isinstance(document.get("source"), str) or not document["source"].strip()
                or not isinstance(document.get("text"), str) or not document["text"].strip()):
            raise ValueError("Language quality documents require unique IDs, text and attribution")
        encoded = tokenizer(document["text"], return_tensors="pt", add_special_tokens=True)["input_ids"]
        if (encoded.ndim != 2 or encoded.shape[0] != 1 or encoded.shape[1] < count
                or encoded.dtype != torch.int64 or bool((encoded < 0).any())
                or bool((encoded >= 32000).any())):
            raise ValueError("Each language quality document must provide at least 128 valid token IDs")
        selected = encoded[:, :count].contiguous()
        selected_identity = identity(selected)
        if selected_identity["sha256"] in digests:
            raise ValueError("Language quality documents must contain distinct selected token sequences")
        ids.add(document["id"])
        digests.add(selected_identity["sha256"])
        tokens.append(selected)
        records.append({"id": document["id"], "source": document["source"],
                        "text_sha256": hashlib.sha256(document["text"].encode()).hexdigest(),
                        "token_ids": selected.tolist(), "token_identity": selected_identity})
    contract = {**QUALITY_CONTRACT, "corpus": file_identity(path, safe_path),
                "corpus_source": payload["source"], "selection_declaration": payload["selection"],
                "documents": records, "calibration": "none_dynamic_activation_recipe",
                "quantization_recipe": recipe(activation_scale, arithmetic_policy)}
    if payload.get("dataset_artifact") is not None:
        selected = payload["dataset_artifact"]
        if (not isinstance(selected, dict) or set(selected) != {"path", "sha256"}
                or file_identity(safe_path(selected["path"]), safe_path) != selected):
            raise ValueError("Language corpus source dataset identity differs from its preparation")
        contract["dataset_artifact"] = selected
    contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return tokens, contract


def language_reference(lm, token_documents):
    """Retain only next-token NLL/top1 labels, avoiding full vocabulary trajectories."""
    result = []
    with torch.no_grad():
        for tokens in token_documents:
            logits = lm(input_ids=tokens, use_cache=False).logits[:, :-1, :].float()
            labels = tokens[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.flatten(), reduction="none")
            result.append({"nll": nll.detach().clone(), "top1": logits.argmax(-1).detach().clone()})
    return result


def language_quality(lm, token_documents, reference, contract):
    closure = {key: value for key, value in contract.items() if key != "sha256"}
    if hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest() != contract["sha256"]:
        raise ValueError("Frozen language quality contract changed before evaluation")
    if len(token_documents) != len(reference) or len(reference) != len(contract["documents"]):
        raise ValueError("Language quality reference must cover the complete frozen document list")
    candidate = language_reference(lm, token_documents)
    differences, agreements, records = [], [], []
    for document, expected, actual in zip(contract["documents"], reference, candidate):
        difference = actual["nll"] - expected["nll"]
        agreement = actual["top1"] == expected["top1"]
        if not bool(torch.isfinite(difference).all()):
            raise AssertionError("Language quality produced nonfinite likelihoods")
        differences.append(difference)
        agreements.append(agreement.flatten())
        records.append({"id": document["id"], "predicted_tokens": difference.numel(),
                        "reference_mean_nll": float(expected["nll"].mean()),
                        "integer_mean_nll": float(actual["nll"].mean()),
                        "next_token_top1_agreement": float(agreement.float().mean())})
    increase = float(torch.cat(differences).mean())
    agreement = float(torch.cat(agreements).float().mean())
    return {"evaluated": True, "contract_sha256": contract["sha256"],
            "predicted_tokens": sum(item.numel() for item in differences),
            "mean_nll_increase_nats": increase, "perplexity_ratio": math.exp(increase),
            "next_token_top1_agreement": agreement, "documents": records,
            "passed": increase <= QUALITY_CONTRACT["maximum_mean_nll_increase_nats"]
            and agreement >= QUALITY_CONTRACT["minimum_next_token_top1_agreement"],
            "paper_language_quality": False}


def operation_placement():
    return {"learned_projections": {"count": 155, "arithmetic": "int8_int8_int32",
                                    "placement": "gemmini_candidate"},
            **{name: {"placement": "cpu", "arithmetic": arithmetic}
               for name, arithmetic in (
                   ("token_embedding", "int64_gather_float32_weights"),
                   ("rms_norm", "float32"), ("rope", "float32_sin_cos_and_matmul"),
                   ("qk_contractions", "float32_batched_matmul"),
                   ("attention_mask", "boolean_int64_float32"),
                   ("softmax", "float32"), ("pv_contractions", "float32_batched_matmul"),
                   ("silu_gate_residual", "float32"),
                   ("activation_quantization_dequantization", "float32_round_clip_int8_cast"),
                   ("kv_state_update", "float32_functional_scatter"),
                   ("positions_and_token_selection", "int64"))}}


def memory_plan(lm, records, prefill_tokens, decode_tokens):
    """Actual constant bytes plus explicit deployment scenarios; never a peak/fit claim."""
    constants = sum(value.numel() * value.element_size() for value in lm.state_dict().values())
    packed = sum(item["packed_weight"]["bytes"] for item in records)
    scales = sum(item["weight_scale"]["bytes"] for item in records)
    capacity = prefill_tokens + decode_tokens
    cache = 2 * 22 * 4 * capacity * 64 * 4
    positions = 22 * 8
    io = {"prefill_input_tokens": prefill_tokens * 8, "decode_input_token": 8,
          "returned_logits": 32000 * 4, "simultaneous_kv_input_output": 2 * cache,
          "simultaneous_position_input_output": 2 * positions}
    return {"kind": "constant_and_session_io_lower_bound_not_target_peak",
            "constants_shared_once_bytes": constants,
            "constants_duplicated_two_stage_bytes": 2 * constants,
            "int8_packed_weights_bytes": packed, "float32_weight_scales_bytes": scales,
            "unquantized_constants_bytes": constants - packed - scales,
            "packing": "KN_contiguous_one_byte_no_extra_device_copy_in_this_plan",
            "one_kv_state_bytes": cache, "kv_capacity": capacity, "io_bytes": io,
            "shared_constants_plus_io_lower_bound_bytes": constants + sum(io.values()),
            "duplicated_constants_plus_io_lower_bound_bytes": 2 * constants + sum(io.values()),
            "missing_peak_terms": ["graph_workspace", "packing_copies_if_required",
                                   "linked_image", "stack", "platform_reservations", "fixtures"],
            "stage_constant_sharing_implemented": False, "actual_platform_fit": "unqualified",
            "whole_model_spike": "pending_runtime_and_static_export_feasibility"}
