"""Local, complete pretrained SmolVLA action-chunk factory for verify_session.py.

The default observation supplies every configured camera. Inputs are explicitly
preprocessed tensors; image/token preprocessing, robot execution and application
quality are outside this neural policy session. Original LeRobot precision is
preserved, including its BF16 backbone and prefix cache. A runtime unable to
represent those tensors must refuse this session rather than cast them.
"""

import importlib.metadata
import importlib.util
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

from verify_resnet import file_identity, safe_path


MODEL_ID = "lerobot/smolvla_base"
MODEL_REVISION = "c83c3163b8ca9b7e67c509fffd9121e66cb96205"
# Official pinned model.safetensors page, SHA256 field (not its Xet hash).
WEIGHT_SHA256 = "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb"
MODEL_SOURCE = f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/model.safetensors"
BACKBONE_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
GEOMETRY = {
    "num_vlm_layers": 16, "num_expert_layers": 0, "expert_width_multiplier": 0.75,
    "num_steps": 10, "chunk_size": 50, "n_action_steps": 50,
    "max_state_dim": 32, "max_action_dim": 32, "tokenizer_max_length": 48,
    "resize_imgs_with_padding": [512, 512], "attention_mode": "cross_attn",
    "self_attn_every_n_layers": 2, "use_cache": True, "load_vlm_weights": True,
    "vlm_model_name": BACKBONE_ID, "adapt_to_pi_aloha": False,
    "use_amp": False, "n_obs_steps": 1, "empty_cameras": 0,
    "add_image_special_tokens": False, "prefix_length": 0, "pad_language_to": "max_length",
    "min_period": 0.004, "max_period": 4.0,
}


# Keep policy admission, source closure and export selection in the same order.
SOURCE_POLICIES = {
    "mean": ("mean", ("native", "torch_cpu_mean720_v1", "torch_cpu_mean720_960_v1")),
    "softmax": ("softmax", ("native", "torch_cpu_softmax177_v1", "torch_cpu_softmax177_227_v1", "torch_cpu_softmax177_227_prefix_v1")),
    "layernorm": ("LayerNorm", ("native", "torch_cpu_layernorm768_v1")),
    "conv": ("convolution", ("native", "torch_cpu_conv3x16_v1")),
    "sgemm": ("SGEMM", ("native", "torch_cpu_sgemm177_v1", "torch_cpu_attention177_227_v1", "torch_cpu_attention177_227_prefix_v1")),
    "scalar": ("scalar", ("native", "torch_cpu_language_scale960_v1")),
    "rotary": ("rotary", ("native", "torch_cpu_denoise_rotary64_table_v1")),
    "flash": ("Flash", ("native", "torch_cpu_flash_bf16_vision1024_v1")),
    "silu": ("SiLU", ("native", "torch_cpu_silu_fp32_v1")),
}


def validate_source_policy(name, policy, quantized, cpu_threads):
    label, choices = SOURCE_POLICIES[name]
    if policy not in choices or policy != "native" and (not quantized or cpu_threads != 1):
        raise ValueError(f"Source {label} preservation requires the original-precision integer recipe and one native thread")


def source_policy_records(torch, policies):
    """Bind every selected source contract before constructing the policy model."""
    records = {}
    for name, policy in policies.items():
        if policy != "native":
            import bf16_export
            records[f"source_{name}_policy"] = getattr(bf16_export, f"source_{name}_contract")(torch, policy)
    return records


def source_policy_artifacts(records):
    """Include every active contract's bindings and its shared implementation."""
    if not records:
        return []
    import bf16_export
    return [identity for record in records.values() for identity in record["bindings"]] + [
        file_identity(bf16_export.__file__, safe_path)]


def directory_artifacts(directory):
    """Hash a supplied model directory, pruning prohibited names before traversal."""
    root = safe_path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Local model directory is absent: {root}")
    result = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories
                                if not any(word in name.lower() for word in ("hammer", "vlsi")))
        for name in sorted(files):
            if any(word in name.lower() for word in ("hammer", "vlsi")):
                continue
            result.append(file_identity(Path(current) / name, safe_path))
    return result


def checkpoint_artifacts(directory):
    root = safe_path(directory)
    for name in ("config.json", "model.safetensors"):
        if not safe_path(root / name).is_file():
            raise FileNotFoundError(f"Pinned complete SmolVLA checkpoint requires {root / name}")
    weight = file_identity(root / "model.safetensors", safe_path)
    if weight["sha256"] != WEIGHT_SHA256:
        raise ValueError("Weights are not the pinned official full SmolVLA checkpoint")
    config = json.loads(safe_path(root / "config.json").read_text())
    if config.get("type") != "smolvla" or any(config.get(key) != value for key, value in GEOMETRY.items()):
        raise ValueError("Configuration differs from the complete pinned SmolVLA geometry/precision")
    camera_names = [f"observation.images.camera{index}" for index in range(1, 4)]
    features = config.get("input_features", {})
    if any(features.get(name) != {"type": "VISUAL", "shape": [3, 256, 256]} for name in camera_names):
        raise ValueError("Pinned SmolVLA requires its three configured camera features")
    if features.get("observation.state") != {"type": "STATE", "shape": [6]} or \
            config.get("output_features", {}).get("action") != {"type": "ACTION", "shape": [6]}:
        raise ValueError("Pinned SmolVLA physical state/action feature extent differs")
    return config, directory_artifacts(root)


def fixture_inputs(config, camera_count, *, input_npz=None, seed=0):
    """Validate exact tensor ABI; never silently coerce a supplied fixture's dtype."""
    import numpy as np
    import torch

    height, width = config.resize_imgs_with_padding
    specs = {
        "images": ((1, camera_count, 3, height, width), np.dtype("float32")),
        "image_masks": ((1, camera_count), np.dtype("bool")),
        "language_tokens": ((1, config.tokenizer_max_length), np.dtype("int64")),
        "language_mask": ((1, config.tokenizer_max_length), np.dtype("bool")),
        "state": ((1, config.max_state_dim), np.dtype("float32")),
        "noise": ((1, config.chunk_size, config.max_action_dim), np.dtype("float32")),
    }
    if input_npz is not None:
        path = safe_path(input_npz)
        with np.load(path, allow_pickle=False) as fixture:
            if set(fixture.files) != set(specs):
                raise ValueError("SmolVLA fixture requires exactly images, image_masks, language_tokens, language_mask, state, noise")
            arrays = [np.array(fixture[name], copy=True, order="C") for name in specs]
        for (name, (shape, dtype)), value in zip(specs.items(), arrays):
            if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError(f"SmolVLA fixture shape/dtype/finite-value mismatch: {name}")
        inputs = tuple(torch.from_numpy(value) for value in arrays)
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        state = torch.zeros(1, config.max_state_dim)
        state[:, :config.robot_state_feature.shape[0]] = torch.randn(
            1, config.robot_state_feature.shape[0], generator=generator)
        inputs = (
            torch.rand(specs["images"][0], generator=generator) * 2 - 1,
            torch.ones(specs["image_masks"][0], dtype=torch.bool),
            torch.randint(0, 256, specs["language_tokens"][0], generator=generator),
            torch.ones(specs["language_mask"][0], dtype=torch.bool), state,
            torch.randn(specs["noise"][0], generator=generator),
        )
    if not torch.all((inputs[0] >= -1) & (inputs[0] <= 1)):
        raise ValueError("Preprocessed SmolVLA images must be in [-1,1]")
    if torch.any(inputs[2] < 0):
        raise ValueError("Language token IDs must be nonnegative")
    if torch.count_nonzero(inputs[4][:, config.robot_state_feature.shape[0]:]):
        raise ValueError("SmolVLA physical state must be zero padded to max_state_dim")
    return inputs


def observation_manifest(input_npz, manifest_path):
    """Bind prepared recorded observations; attribution text alone is not schema proof."""
    if manifest_path is None:
        return None
    if input_npz is None:
        raise ValueError("Recorded observation manifest requires its exact NPZ")
    from prepare_smolvla_observation import CAMERA_MAP, DATASET_ID, DATASET_REVISION, STATE_NAMES, episode_split

    path = safe_path(manifest_path)
    value = json.loads(path.read_text())
    expected = {"version": 1, "fixture_kind": "dataset", "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION, "checkpoint_revision": MODEL_REVISION,
                "precision": "original_lerobot", "schema_verified": True,
                "selected_cameras": list(CAMERA_MAP.values()), "camera_mapping": CAMERA_MAP,
                "state_names": STATE_NAMES, "active_gate_changed": False,
                "model_quality_qualified": False}
    if any(value.get(key) != item for key, item in expected.items()) or \
            value.get("split") != episode_split(value.get("episode_index")):
        raise ValueError("Recorded observation manifest differs from the pinned preparation recipe")
    if value.get("input_fixture") != file_identity(input_npz, safe_path):
        raise ValueError("Recorded observation NPZ identity differs from its preparation manifest")
    bindings = value.get("bindings")
    if not isinstance(bindings, list) or not bindings or any(
            file_identity(item["path"], safe_path) != item for item in bindings):
        raise ValueError("Recorded observation source/processor bindings changed")
    # Verify the manifest's record of the original official parquet identities.
    from prepare_smolvla_observation import PINNED_FILES
    for relative, digest in PINNED_FILES.items():
        if not any(Path(item["path"]).as_posix().endswith("/" + relative)
                   and item["sha256"] == digest for item in bindings):
            raise ValueError("Recorded observation lacks pinned official parquet bindings")
    return value


def install_integer_projections(model, torch):
    """Declare W8A8 learned linears, preserving original floating tensor interfaces.

    The caller owns a separate target copy. CPU attention/normalization/convolution
    retain original LeRobot arithmetic. FP32 dequantization and bias addition precede
    exactly one output cast to the original linear weight/input dtype.
    """
    import tinyllama_integer as shared
    import numpy as np

    class OriginalDtypeProjection(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            if original.weight.dtype not in (torch.bfloat16, torch.float32) or \
                    original.bias is not None and original.bias.dtype != original.weight.dtype:
                raise ValueError("Smol integer recipe requires matching original BF16/FP32 linear operands")
            self.original_dtype = original.weight.dtype
            self.integer = shared.IntegerProjection(original.weight.detach().float())
            self.register_buffer("bias", None if original.bias is None else original.bias.detach().clone())
            # LeRobot reads q/k/v/o_proj.weight.dtype to retain its explicit cast
            # boundaries. This zero-element marker preserves those reads without
            # retaining the original floating weights on the target.
            self.register_buffer("original_dtype_marker", torch.empty(0, dtype=self.original_dtype))

        @property
        def weight(self):
            return self.original_dtype_marker

        def forward(self, value):
            if not torch.compiler.is_compiling() and value.dtype != self.original_dtype:
                raise ValueError("Integer projection input differs from its original floating interface")
            result = self.integer(value.float())
            if self.bias is not None:
                result = result + self.bias.float()
            return result.to(self.original_dtype)

    linears = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    if not linears:
        raise ValueError("Integer Smol reference requires learned linear modules")
    records, projections = [], {}
    for name, original in linears.items():
        projection = OriginalDtypeProjection(original)
        channels = np.unique(np.linspace(0, projection.integer.out_features - 1,
                                         min(7, projection.integer.out_features), dtype=int))
        quantized, scales = shared.independent_quantize_rows(original.weight.detach().float().numpy()[channels])
        if not np.array_equal(projection.integer.weight_kn.numpy()[:, channels].T, quantized) or \
                not np.array_equal(projection.integer.weight_scale.numpy()[channels], scales.flatten()):
            raise AssertionError("Smol weight quantization differs from the independent NumPy reference")
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, projection)
        projections[name] = projection
        records.append({"name": name, "source_weight": shared.identity(original.weight),
                        "source_bias": None if original.bias is None else shared.identity(original.bias),
                        "packed_weight": shared.identity(projection.integer.weight_kn),
                        "weight_scale": shared.identity(projection.integer.weight_scale),
                        "input_k": projection.integer.in_features, "output_n": projection.integer.out_features,
                        "float_interface_dtype": str(projection.original_dtype),
                        "bias_boundary": "FP32 add after FP32 dequantization, before single original-dtype output cast",
                        "weight_oracle": {"passed": True, "channels": channels.tolist(), "full_k": projection.integer.in_features},
                        "placement": "CPU eager INT8/INT32 reference; Gemmini candidate only after export/device checks",
                        "calls_by_stage": {}, "integer_operator_reference": None})
        # Do not retain a second floating copy of the target's quantized weights.
        linears[name] = None
    recipe = {**shared.RECIPE, "name": "smolvla_w8a8_original_float_interfaces_v1",
              "bias": "original BF16/FP32 bias converted exactly to FP32; add after dequantization; then original-dtype cast",
              "attention_and_state": "original LeRobot mixed BF16/FP32 CPU",
              "source_conversion": "BF16 operands widen exactly to FP32 only inside declared projection quantization"}
    recipe["sha256"] = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    return records, projections, recipe


def check_integer_projection(projection, value, output, torch):
    """Check quantization, complete-K int64 contraction and explicit bias/cast order."""
    import tinyllama_integer as shared
    import numpy as np

    result = shared.check_projection(projection.integer, value.detach().float())
    rows, columns = result["rows"], result["columns"]
    quantized, activation_scale = projection.integer.integer_inputs(value.detach().float())
    accumulator = quantized.numpy()[rows].astype(np.int64) @ projection.integer.weight_kn.numpy()[:, columns].astype(np.int64)
    expected = accumulator.astype(np.float32) * activation_scale.reshape(-1, 1).numpy()[rows] \
        * projection.integer.weight_scale.numpy()[columns]
    if projection.bias is not None:
        expected = expected + projection.bias.float().numpy()[columns]
    expected = torch.from_numpy(expected).to(projection.original_dtype)
    actual = output.detach().reshape(-1, projection.integer.out_features)[rows][:, columns]
    if not torch.equal(expected, actual):
        raise AssertionError("Smol projection dequantization/bias/cast differs from its independent reference")
    return {**result, "float_bias_and_output_cast_exact": True,
            "float_interface_dtype": str(projection.original_dtype)}


def integer_memory_plan(source, candidate, records, inputs, captures, torch):
    """Actual constants and resident diagnostic scenarios; not a deployed peak/fit."""
    def constants(model):
        storages, groups = {}, {}
        for value in model.state_dict().values():
            storage = value.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            if key not in storages:
                storages[key] = storage.nbytes()
                groups[str(value.dtype)] = groups.get(str(value.dtype), 0) + storage.nbytes()
        return sum(storages.values()), groups

    source_bytes, source_groups = constants(source)
    target_bytes, target_groups = constants(candidate)
    packed = sum(record["packed_weight"]["bytes"] for record in records)
    scales = sum(record["weight_scale"]["bytes"] for record in records)
    cache = captures["prefix_key_cache"].numel() * captures["prefix_key_cache"].element_size() \
        + captures["prefix_value_cache"].numel() * captures["prefix_value_cache"].element_size()
    fixture = sum(value.numel() * value.element_size() for value in inputs)
    flow = inputs[5].numel() * inputs[5].element_size()
    return {
        "kind": "actual_constant_inventory_and_explicit_scenarios_not_target_peak",
        "source_constants_bytes": source_bytes, "source_bytes_by_dtype": source_groups,
        "target_constants_bytes": target_bytes, "target_bytes_by_dtype": target_groups,
        "target_packed_int8_weights_bytes": packed, "target_fp32_weight_scales_bytes": scales,
        "target_cpu_constants_including_bias_embedding_convolution_bytes": target_bytes - packed - scales,
        "host_two_model_diagnostic_constants_bytes": source_bytes + target_bytes,
        "host_clone_before_quantization_constants_bytes": 2 * source_bytes,
        "one_prefix_cache_bytes": cache, "simultaneous_cache_input_output_bytes": 2 * cache,
        "fixture_bytes": fixture, "simultaneous_flow_input_output_bytes": 2 * flow,
        "shared_target_constants_plus_declared_io_bytes": target_bytes + 2 * cache + fixture + 2 * flow,
        "possible_additional_device_packing_copy_bytes": packed,
        "naive_full_module_constants_for_three_stages_bytes": 3 * target_bytes,
        "stage_constant_sharing_implemented": False,
        "missing_peak_terms": ["graph workspace", "captured diagnostics", "largest conversion temporaries",
                               "linked image", "stack", "platform reservations", "runtime packing alignment"],
        "baremetal_missing_dtype_support": ["bfloat16"], "actual_platform_fit": "unqualified",
        "device_execution": False, "source_model_is_diagnostic_only_not_deployed": True,
    }


def run_integer_policy_diagnostic(source, candidate, inputs, records, projections, directory,
                                  *, seed, fixture_kind, recipe, torch):
    """Keep original reference intact; qualify only the declared eager integer session."""
    import numpy as np
    import tinyllama_integer as shared
    import diagnose_smolvla_backends as diagnostic

    root = safe_path(directory)
    from merlin.common.paths import artifacts_dir
    if root.exists() or not root.is_relative_to(safe_path(artifacts_dir())):
        raise ValueError("Choose a new integer diagnostic directory inside the configured artifacts root")
    root.mkdir(parents=True, exist_ok=False)
    source_dir, candidate_dir = root / "source", root / "integer"
    source_dir.mkdir()
    candidate_dir.mkdir()
    source_before = diagnostic.state_identity(source, torch)
    helper_before = file_identity(shared.__file__, safe_path)
    sources = [file_identity(path, safe_path) for path in (__file__, diagnostic.__file__)]
    original, source_receipt = diagnostic.run_source_policy(source, inputs, "default", seed, 0, source_dir, np, torch)
    stage = None

    @contextmanager
    def observer(name):
        nonlocal stage
        previous, stage = stage, name
        try:
            yield
        finally:
            stage = previous

    by_name = {record["name"]: record for record in records}
    handles = []
    for name, projection in projections.items():
        def inspect(module, arguments, output, *, name=name):
            record = by_name[name]
            if stage not in ("prefix_encode", "flow_denoise"):
                raise AssertionError("Integer projection ran outside a declared policy stage")
            record["calls_by_stage"][stage] = record["calls_by_stage"].get(stage, 0) + 1
            if record["integer_operator_reference"] is None:
                record["integer_operator_reference"] = check_integer_projection(module, arguments[0], output, torch)
        handles.append(projection.register_forward_hook(inspect))
    try:
        integer, target_receipt = diagnostic.run_source_policy(candidate, inputs, "default", seed, 0,
                                                              candidate_dir, np, torch, stage_observer=observer)
    finally:
        for handle in handles:
            handle.remove()
    for name in ("prefix_pad_mask", "prefix_attention_mask"):
        if not torch.equal(original[name], integer[name]):
            raise AssertionError("Quantized policy changed source mask/routing semantics")
    replays = {}
    for name, prefix in (
        ("original_prefix", None),
        ("integer_prefix", (integer["prefix_pad_mask"], integer["prefix_key_cache"], integer["prefix_value_cache"])),
    ):
        values, replay = diagnostic.compare_identical_state_velocities(
            source.denoise_step, candidate.denoise_step, original, torch,
            expected_steps=int(source.config.num_steps), candidate_prefix=prefix)
        replay_path = safe_path(root / ("identical-state-" + name + ".npz"))
        np.savez_compressed(replay_path, **{key: value.numpy() for key, value in values.items()})
        replays[name] = {**replay, "output_file": file_identity(replay_path, safe_path)}
    if diagnostic.state_identity(source, torch) != source_before:
        raise AssertionError("Integer preparation/execution changed the original source model")
    bindings = [helper_before, *sources]
    if any(file_identity(item["path"], safe_path) != item for item in bindings):
        raise AssertionError("Integer policy source/shared helper changed during execution")
    executed = [record for record in records if record["integer_operator_reference"] is not None]
    report = {
        "status": "eager_integer_session_checked", "fixture_kind": fixture_kind,
        "synthetic_inputs": fixture_kind == "synthetic", "recipe": recipe,
        "source_reference_unchanged": True, "source_state_sha256": source_before,
        "source_bindings": bindings, "source_run": source_receipt, "integer_run": target_receipt,
        "denoising_steps": candidate.config.num_steps, "source_mask_routing_exact": True,
        "projection_inventory": records, "executed_projection_count": len(executed),
        "unexecuted_projection_names": [record["name"] for record in records if not record["calls_by_stage"]],
        "operator_reference": "sampled rows/columns with complete K; independent NumPy int64 and explicit float bias/output cast",
        "comparisons": {name: diagnostic.statistics(original[name], integer[name], torch) for name in original},
        "action_quality_diagnostic": diagnostic.action_quality_diagnostic(original["final_actions"], integer["final_actions"], torch),
        "identical_state_velocity_replay": replays,
        "memory": integer_memory_plan(source, candidate, records, inputs, integer, torch),
        "placement": {"learned_linear": "eager CPU int8/int32; Gemmini candidate, not device qualified",
                      "convolution_embedding_qk_pv_rope_norm_softmax": "original mixed BF16/FP32 CPU",
                      "state_cache_masks_euler_action_decode": "original source dtypes and routes"},
        "active_gate_changed": False, "original_bf16_compiler_gate_passed": False,
        "model_quality_qualified": False, "quantized_recipe_quality_qualified": False,
        "device_execution": False, "timing_qualified": False, "source_closure_verified": False,
    }
    path = safe_path(root / "results.json")
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report, file_identity(path, safe_path)


class SessionProvider:
    def __init__(self, session, artifacts):
        self.session, self.artifacts = session, tuple(artifacts)

    def external_runtime_session(self):
        return self.session


def make_policy_session(model, inputs, loader, provenance):
    """Split upstream inference, preserving exact timesteps and policy action slicing."""
    import torch
    from torch import nn
    from m2m.capture.external_runtime import ExternalRuntimeProgram, make_external_runtime_session

    class PrefixStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, images, image_masks, tokens, language_mask, state):
            embeds, masks, attention_masks = self.model.embed_prefix(
                [images[:, index] for index in range(images.shape[1])],
                [image_masks[:, index] for index in range(images.shape[1])],
                tokens, language_mask, state=state)
            _, cache = self.model.vlm_with_expert.forward(
                attention_mask=loader.make_att_2d_masks(masks, attention_masks),
                position_ids=torch.cumsum(masks, dim=1) - 1, past_key_values=None,
                inputs_embeds=[embeds, None], use_cache=True, fill_kv_cache=True)
            layers = range(int(self.model.config.num_vlm_layers))
            keys = torch.stack([cache[index]["key_states"] for index in layers])
            values = torch.stack([cache[index]["value_states"] for index in layers])
            return masks, torch.stack((keys, values))

    prefix = PrefixStage().eval().requires_grad_(False)
    with torch.no_grad():
        masks, cache = prefix(*inputs[:5])
    steps = int(model.config.num_steps)
    dt = -1.0 / steps
    # Upstream uses 1 + step*dt, not repeated FP32 addition of dt.
    times = torch.stack([torch.tensor(1.0 + step * dt, dtype=torch.float32).expand(1)
                         for step in range(steps)])
    flow = loader.SmolVLAFlowSession(model).eval()
    flow_metadata = {
        "kind": "action_chunk", "steps": steps,
        "states": [{"name": "prefix_kv_cache", "input_index": 1, "output_index": 1},
                   {"name": "flow_state", "input_index": 2, "output_index": 0}],
        "streams": [{"name": "upstream_timestep", "input_index": 3, "values": times}],
        "quality": {"key": "actions", "output_index": 0},
    }
    action_dim = int(model.config.action_feature.shape[0])
    action = loader.SmolVLAActionDecodeStage(model.config.chunk_size, action_dim).eval()
    metadata = {
        "kind": "action_chunk", "paper_ready": False,
        "stages": ["prefix_encode", "flow_denoise", "action_decode"],
        "stage_schedule": [{"name": name, "steps": count, "execution": execution, "timed": True}
                           for name, count, execution in [("prefix_encode", 1, "compiled"),
                               ("flow_denoise", steps, "compiled_recurrent"),
                               ("action_decode", 1, "compiled")]],
        "bindings": [
            {"name": "prefix_pad_masks", "from": {"program": "prefix_encode", "output_index": 0},
             "to": {"program": "flow_denoise", "input_index": 0}},
            {"name": "prefix_kv_cache", "from": {"program": "prefix_encode", "output_index": 1},
             "to": {"program": "flow_denoise", "input_index": 1}},
            {"name": "decoded_flow_state", "from": {"program": "flow_denoise", "output_index": 0},
             "to": {"program": "action_decode", "input_index": 0}}],
        "quality_program": "flow_denoise",
        "parameters": {"denoise_steps": steps, "action_horizon": int(model.config.chunk_size),
                       "physical_action_dim": action_dim, "camera_count": inputs[0].shape[1],
                       "batch": 1, "semantic_observations": "denoising iterations for one action chunk",
                       "input_boundary": "preprocessed normalized images/tokens/zero-padded state",
                       "output_boundary": "policy action coordinates before original postprocessing",
                       "action_normalization_qualified": False, "physical_action_units_qualified": False},
        "provenance": provenance,
    }
    session = make_external_runtime_session(version=2, programs=(
        ExternalRuntimeProgram("prefix_encode", prefix, inputs[:5], 1),
        ExternalRuntimeProgram("flow_denoise", flow, (masks, cache, inputs[5], times[0]), steps, flow_metadata),
        ExternalRuntimeProgram("action_decode", action, (inputs[5],), 1)), metadata=metadata)
    # An independent upstream full inference is the splitting correctness gate.
    with torch.no_grad():
        expected = model.sample_actions(
            [inputs[0][:, index] for index in range(inputs[0].shape[1])],
            [inputs[1][:, index] for index in range(inputs[1].shape[1])],
            inputs[2], inputs[3], inputs[4], noise=inputs[5])[:, :, :action_dim]
        state = inputs[5]
        for timestep in times:
            state, cache, _ = flow(masks, cache, state, timestep)
        actual = action(state)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    provenance["upstream_sample_actions_equivalence"] = {"passed": True, "exact": True,
                                                        "shape": list(actual.shape)}
    return session


def get_model_and_inputs(*, checkpoint_dir, backbone_dir, model2mlir_root,
                         input_npz=None, fixture_kind="synthetic", input_source=None,
                         input_manifest=None, selected_cameras=None, seed=0, cpu_threads=1, precision="original",
                         quantized=False, integer_diagnostic_dir=None, source_mean_policy="native", source_softmax_policy="native", source_layernorm_policy="native", source_conv_policy="native", source_sgemm_policy="native", source_scalar_policy="native", source_rotary_policy="native", source_flash_policy="native", source_silu_policy="native"):
    if precision != "original":
        raise ValueError("Only original LeRobot precision is supported; implicit BF16 conversion is forbidden")
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 4 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid explicit CPU thread/fixture seed budget")
    if type(quantized) is not bool or quantized != (integer_diagnostic_dir is not None):
        raise ValueError("Declared integer policy requires quantized=True and a new integer_diagnostic_dir")
    source_policies = {
        "mean": source_mean_policy, "softmax": source_softmax_policy,
        "layernorm": source_layernorm_policy, "conv": source_conv_policy,
        "sgemm": source_sgemm_policy, "scalar": source_scalar_policy,
        "rotary": source_rotary_policy, "flash": source_flash_policy,
        "silu": source_silu_policy,
    }
    for name, policy in source_policies.items():
        if name != "silu":
            validate_source_policy(name, policy, quantized, cpu_threads)
    if fixture_kind not in ("synthetic", "dataset") or (fixture_kind == "dataset" and (not input_npz or not input_source)):
        raise ValueError("Dataset fixtures require both a local NPZ and attributed input_source")
    validate_source_policy("silu", source_silu_policy, quantized, cpu_threads)
    if input_manifest is not None and fixture_kind != "dataset":
        raise ValueError("A recorded observation manifest must be labeled as a dataset fixture")
    fixture_manifest = observation_manifest(input_npz, input_manifest)
    config_json, checkpoint_files = checkpoint_artifacts(checkpoint_dir)
    backbone = safe_path(backbone_dir)
    backbone_files = directory_artifacts(backbone)
    if not safe_path(backbone / "config.json").is_file() or not any(
            Path(item["path"]).name.endswith(".safetensors") for item in backbone_files):
        raise FileNotFoundError("Original-precision SmolVLA requires local backbone config, processor and weights")
    loader_path = safe_path(Path(model2mlir_root) / "workloads/smolvla/loader.py")
    if selected_cameras is None:
        selected_cameras = fixture_manifest["selected_cameras"] if fixture_manifest else [
            name for name in config_json["input_features"] if name.startswith("observation.images.")]
    if fixture_manifest and selected_cameras != fixture_manifest["selected_cameras"]:
        raise ValueError("Selected camera order differs from recorded observation preparation")
    configured = {name for name in config_json["input_features"] if name.startswith("observation.images.")}
    if not isinstance(selected_cameras, list) or not selected_cameras or \
            len(set(selected_cameras)) != len(selected_cameras) or any(name not in configured for name in selected_cameras):
        raise ValueError("Selected cameras must be an explicit nonempty unique configured subset")
    if importlib.metadata.version("lerobot") != "0.5.1":
        raise ValueError("This original policy recipe requires lerobot==0.5.1")
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla import smolvlm_with_expert

    torch.set_num_threads(cpu_threads)
    policy_records = source_policy_records(torch, source_policies)
    config = PreTrainedConfig.from_pretrained(str(safe_path(checkpoint_dir)), local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError("Pinned checkpoint did not resolve to SmolVLAConfig")
    config.device = "cpu"
    config.vlm_model_name = str(backbone)
    policy = SmolVLAPolicy.from_pretrained(str(safe_path(checkpoint_dir)), config=config,
                                         local_files_only=True, strict=True).eval().requires_grad_(False)
    model = policy.model
    spec = importlib.util.spec_from_file_location("tvm_smolvla_loader", loader_path)
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    inputs = fixture_inputs(config, len(selected_cameras), input_npz=input_npz, seed=seed)
    if torch.any(inputs[2] >= model.vlm_with_expert.config.text_config.vocab_size):
        raise ValueError("Fixture token ID is outside the pinned backbone vocabulary")
    import diagnose_smolvla_backends as diagnostic
    processor_diagnostic = diagnostic.original_processor_diagnostic(checkpoint_dir, backbone, torch, physical_dim=int(config.action_feature.shape[0]))
    integer_report, integer_receipt, helper_identity = None, None, None
    if quantized:
        import tinyllama_integer as shared
        original_state = diagnostic.state_identity(model, torch)
        candidate = copy.deepcopy(model)
        records, projections, recipe = install_integer_projections(candidate, torch)
        if diagnostic.state_identity(model, torch) != original_state:
            raise AssertionError("Integer projection installation changed original source weights")
        integer_report, integer_receipt = run_integer_policy_diagnostic(
            model, candidate, inputs, records, projections, integer_diagnostic_dir,
            seed=seed, fixture_kind=fixture_kind, recipe=recipe, torch=torch)
        helper_identity = file_identity(shared.__file__, safe_path)
        model = candidate
    sources = [file_identity(path, safe_path) for path in (
        __file__, loader_path, sys.modules[SmolVLAPolicy.__module__].__file__,
        sys.modules[SmolVLAConfig.__module__].__file__, smolvlm_with_expert.__file__,
        sys.modules[PreTrainedConfig.__module__].__file__)]
    files = checkpoint_files + backbone_files + sources
    files.extend(source_policy_artifacts(policy_records))
    files.extend(processor_diagnostic["bindings"])
    if quantized:
        files.extend([integer_receipt, helper_identity, file_identity(diagnostic.__file__, safe_path)])
    if input_npz:
        files.append(file_identity(input_npz, safe_path))
    if input_manifest:
        files.append(file_identity(input_manifest, safe_path))
        files.extend(fixture_manifest["bindings"])
    dtype_bytes = {}
    for value in model.state_dict().values():
        dtype = str(value.dtype)
        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + value.numel() * value.element_size()
    provenance = {"checkpoint": MODEL_ID, "checkpoint_revision": MODEL_REVISION,
                  "official_weight_sha256": WEIGHT_SHA256, "official_weight_source": MODEL_SOURCE,
                  "full_checkpoint": True, "precision": "original_lerobot",
                  "synthetic_inputs": fixture_kind == "synthetic", "fixture_seed": seed if not input_npz else None,
                  "input_source": input_source or "seeded synthetic preprocessed tensors",
                  "observation_manifest": fixture_manifest, "observation_schema_verified": fixture_manifest is not None,
                  "selected_cameras": selected_cameras, "all_configured_cameras": set(selected_cameras) == configured,
                  "backbone": BACKBONE_ID, "local_backbone": str(backbone),
                  "backbone_revision_verified": False, "declared_artifacts": files,
                  "dependencies": {name: importlib.metadata.version(name) for name in
                                   ("lerobot", "torch", "transformers", "safetensors", "numpy", "onnx")},
                  "model_state_bytes_by_dtype": dtype_bytes, "model_quality_qualified": False,
                  "source_closure_verified": False}
    provenance["original_processor_diagnostic"] = processor_diagnostic
    provenance.update(policy_records)
    if quantized:
        provenance.update(precision="declared_w8a8_with_original_lerobot_float_interfaces",
                          source_precision="original_lerobot", quantized_recipe_quality_qualified=False,
                          integer_reference=integer_report, integer_reference_receipt=integer_receipt,
                          compiler_numerical_reference="declared integer policy; original policy is separate quality reference")
    session = make_policy_session(model, inputs, loader, provenance)
    for identity in files:
        if file_identity(identity["path"], safe_path) != identity:
            raise ValueError("Selected SmolVLA artifact/source changed while loading the policy session")
    return SessionProvider(session, [Path(item["path"]) for item in files]), ()


def get_verification_artifacts(model, _inputs):
    # The optional exporter hook imports this adjacent shared helper even when a
    # non-integer graph has no matching nodes. Bind those selected bytes too.
    return (*model.artifacts, safe_path(Path(__file__).with_name("tinyllama_integer.py")))


def get_onnx_export_options(torch, model=None):
    """Select integer contractions and declared source-preserving reductions."""
    import tinyllama_integer as shared
    if safe_path(shared.__file__) != safe_path(Path(__file__).with_name("tinyllama_integer.py")):
        raise ValueError("Shared integer projection helper differs from the selected comparison source")
    options = shared.export_options(torch)
    if model is not None:
        provenance = model.session.metadata.get("provenance", {})
        for name in SOURCE_POLICIES:
            key = f"source_{name}_policy"
            policy = provenance.get(key, {}).get("name", "native")
            if policy == "native":
                continue
            if name in ("scalar", "rotary", "flash", "silu"):
                options[key] = policy
            else:
                import bf16_export
                options["custom_translation_table"].update(
                    getattr(bf16_export, f"source_{name}_translations")(torch, policy))
    return options
