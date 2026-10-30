"""Replay genuine SmolVLA inference from the retained original prefix and noise.

The native baseline must close every archived cache/trajectory tensor before
actual-Q candidates run their own complete native ten-step trajectories. Vision
is bypassed explicitly; outputs remain policy coordinates, without physical-unit
or task-quality qualification.
"""

from merlin.common.digest import sha256_bytes

import argparse
import hashlib
import importlib.metadata
from pathlib import Path
import sys

import numpy as np

import source_relocation
from source_relocation import argument_relocator, read_metadata

from merlin.common.paths import artifacts_dir
from capture_smolvla import (CHECKPOINT_REVISION, WEIGHTS_SHA, RECEIPT_SHA,
                            ARCHIVE_SHA, OBSERVATION_SHA, EXPERT_SOURCE_SHA, LLAMA_SHA)
from capture_smolvla_full_query import validate_padding
from capture_tinyllama_native import (identity, safe_path, array_identity, verify_declared_sources,
                                      bf16_words_to_float, load_provider_validator)
from evaluate_smolvla_attention import MASK_SOURCE_SHA, require_same

CONFIG_SHA = "650584b56c104720f7a3c91d1ec6bec9e8de8ac11e60c92ba2fa82d93eda147d"
INPUT_SHA = "167d7bbec48362a3dbb177b9afc97ec6ebfe50855e07f3858e40369de87f36e9"
STATE_SHA = "fc23e35fe27e16f828e099f26c32630b0138f64034518fe3c99f3d5682451140"
TOP_RECEIPT_SHA = "a0d2900e31f1eebf0af9ac6bb689e8955669c3d45f8ae462ced0e8090701cc13"
HISTORICAL_HELPERS = {"smolvla_session.py": "19cbb1b1a73b9885a771cabc8713d73be7dfa98d85a1b3b833a34cb0d024ddf5",
                      "diagnose_smolvla_backends.py": "155a38b251036af89e6869ca042967a7a786d29fc467590581664877711f1412"}
INPUT_NAMES = ("images", "image_masks", "language_tokens", "language_mask", "state", "noise")


def tensor_words(value, torch):
    # Same flat-byte contract as the receipt-bound historical diagnostic,
    # including expanded tensors whose singleton stride need not be one.
    if value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError("Native replay requires strided CPU tensors")
    flat = torch.empty(value.numel(), dtype=value.dtype, device="cpu")
    flat.copy_(value.detach().reshape(-1))
    return flat.view(torch.uint8).numpy().tobytes()


def tensor_id(value, torch):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": sha256_bytes(tensor_words(value, torch))}


def state_sha(model, torch):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(tensor_words(value, torch))
    return digest.hexdigest()


def evaluate(args, report, *, candidate_plan=None):
    producer = safe_path(args.producer_root)
    validator, validator_id = load_provider_validator(
        producer, args.validator_sha256, args.provider_receipt, args.provider_receipt_sha256)
    manifest, arrays, target, provider_records, admission = validator.admit_contraction(
        receipt_path=safe_path(args.provider_receipt), receipt_sha256=args.provider_receipt_sha256,
        native_bundle=safe_path(args.native_bundle), native_manifest_sha256=args.native_manifest_sha256,
        producer_root=producer, model_root=safe_path(args.model_root),
        metadata_path_mapper=argument_relocator(args).path)
    source, site = manifest["source"], manifest["site"]
    if (manifest["status"] != "native_selective_capture_passed" or manifest["source_dtype"] != "bfloat16"
            or source["revision"] != CHECKPOINT_REVISION or site["matrix_shape"] != [192, 960, 960]
            or site["module"] != "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj"
            or site["positions"] != [0, 177] or site["output_channels"] != [0, 960]
            or site["k_range"] != [0, 960] or site["native_rows"] != 177 or site["valid_rows"] != 177
            or site["attention_valid_positions"] != 141 or site["padding"]["native_masks_or_positions_extended"] is not False):
        raise ValueError("Require the original complete 177-position query with export-only padding")
    verify_declared_sources(source)
    bindings = [*args.workload_helper_bindings, identity(__file__), identity(source_relocation.__file__), validator_id,
                identity(source["weights"]["path"], WEIGHTS_SHA),
                identity(source["historical_source_receipt"]["path"], RECEIPT_SHA),
                identity(source["historical_original_archive"]["path"], ARCHIVE_SHA),
                identity(source["observation_receipt"]["path"], OBSERVATION_SHA),
                identity(args.observation_input, INPUT_SHA),
                identity(manifest["native_query"]["path"], manifest["native_query"]["sha256"])]
    for name, sha in HISTORICAL_HELPERS.items():
        bindings.append(identity(safe_path(args.historical_source_root) / name, sha))
    historical = read_metadata(args, report, source["historical_source_receipt"]["path"], RECEIPT_SHA)
    if (not historical["source_reference_unchanged"] or historical["source_state_sha256"] != STATE_SHA
            or historical["source_run"]["output_file"]["sha256"] != ARCHIVE_SHA or historical["denoising_steps"] != 10):
        raise ValueError("Wrong original native trajectory reference")
    top_receipt = safe_path(args.top_receipt)
    bindings.append(identity(top_receipt, TOP_RECEIPT_SHA))
    top = read_metadata(args, report, top_receipt, TOP_RECEIPT_SHA)
    policy_dir = safe_path(source["weights"]["path"]).parent
    bindings.append(identity(policy_dir / "config.json", CONFIG_SHA))
    config_json = read_metadata(args, report, policy_dir / "config.json", CONFIG_SHA)
    geometry = {"num_vlm_layers": 16, "num_expert_layers": 0, "expert_width_multiplier": 0.75,
                "num_steps": 10, "chunk_size": 50, "max_state_dim": 32, "max_action_dim": 32,
                "self_attn_every_n_layers": 2, "attention_mode": "cross_attn", "use_cache": True,
                "use_amp": False, "load_vlm_weights": True}
    if any(config_json[key] != value for key, value in geometry.items()):
        raise ValueError("Original native policy configuration changed")
    with np.load(source["historical_original_archive"]["path"], allow_pickle=False) as archive:
        reference = {name: archive[name].copy() for name in historical["source_run"]["captures"]}
    with np.load(manifest["native_query"]["path"], allow_pickle=False) as archive:
        native_q = archive["Q_bf16"].copy()
    validate_padding(arrays, native_q, arrays["A_bf16"][:177])
    if np.any(target[177:].view(np.uint32)):
        raise ValueError("Synthetic physical query rows are not positive zero")
    actual_q = target[:177].copy()
    actual_words = (actual_q.view(np.uint32) >> 16).astype(np.uint16)
    require_same(actual_q.view(np.uint32), bf16_words_to_float(actual_words).view(np.uint32),
        "Actual query is not an exact BF16 view")
    with np.load(args.observation_input, allow_pickle=False) as archive:
        if set(archive.files) != set(INPUT_NAMES):
            raise ValueError("Unexpected observation fixture fields")
        input_arrays = {name: archive[name].copy() for name in INPUT_NAMES}
    import torch
    from transformers.models.llama import modeling_llama
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla import smolvlm_with_expert, modeling_smolvla
    versions = {key: importlib.metadata.version(key) for key in ("torch", "transformers", "lerobot", "numpy")}
    if versions != source["versions"] or sys.version != source["python"]:
        raise ValueError("Native environment differs from the source reference")
    bindings += [identity(smolvlm_with_expert.__file__, EXPERT_SOURCE_SHA), identity(modeling_llama.__file__, LLAMA_SHA),
                 identity(modeling_smolvla.__file__, MASK_SOURCE_SHA),
                 identity(Path(torch.__file__).parent / "lib/libtorch_cpu.so", source["torch_cpu_library"]["sha256"])]
    # Bind the additional original policy/configuration modules from the retained
    # run records, without importing any TVM/compiler helper or scanning packages.
    package_root = safe_path(smolvlm_with_expert.__file__).parents[3]
    for record in top["factory_artifacts"]:
        path = Path(record["path"])
        if path.suffix == ".py" and path.is_relative_to(package_root):
            bindings.append(identity(safe_path(path), record["sha256"]))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = PreTrainedConfig.from_pretrained(str(policy_dir), local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError("Pinned local config did not resolve to SmolVLAConfig")
    before_overrides = {"device": str(config.device), "vlm_model_name": config.vlm_model_name}
    config.device = "cpu"
    config.vlm_model_name = str(safe_path(source["backbone_config"]["path"]).parent)
    effective_config = {"original_config": identity(policy_dir / "config.json", CONFIG_SHA),
                        "raw_values": {key: config_json.get(key) for key in before_overrides},
                        "loaded_values_before_overrides": before_overrides,
                        "intentional_overrides": {"device": config.device, "vlm_model_name": config.vlm_model_name},
                        "local_backbone_config": source["backbone_config"],
                        "reason": "Match the recovered original CPU/offline native loader semantics"}
    policy = SmolVLAPolicy.from_pretrained(str(policy_dir), config=config, local_files_only=True, strict=True).eval(
        ).requires_grad_(False)
    model = policy.model
    if model._rtc_enabled() or config.compile_model or state_sha(model, torch) != STATE_SHA:
        raise ValueError("Original native state/topology/compilation/RTC settings differ")
    if model.vlm_with_expert.num_vlm_layers != 16 or model.vlm_with_expert.num_expert_layers != 16:
        raise ValueError("Original prefix/expert layer count changed")
    inputs = tuple(torch.from_numpy(input_arrays[name]) for name in INPUT_NAMES)
    input_ids = {name: tensor_id(value, torch) for name, value in zip(INPUT_NAMES, inputs)}
    if input_ids != historical["source_run"]["input_bindings"]:
        raise ValueError("Observation tensors are not the exact original replay inputs")
    if not np.array_equal(input_arrays["noise"].view(np.uint32), reference["step_00_input_state"].view(np.uint32)):
        raise ValueError("Initial noise is not the archived initial native state")
    prefix_names = ("prefix_embeddings", "prefix_pad_mask", "prefix_attention_mask")
    if (reference["prefix_embeddings"].shape != (1, 177, 960) or reference["prefix_embeddings"].dtype != np.float32
            or reference["prefix_pad_mask"].shape != (1, 177) or reference["prefix_pad_mask"].dtype != np.bool_
            or int(reference["prefix_pad_mask"].sum()) != 141 or reference["prefix_attention_mask"].shape != (1, 177)):
        raise ValueError("Original retained prefix extent/dtype/masking changed")
    prefix = tuple(torch.from_numpy(reference[name].copy()) for name in prefix_names)
    query_module = model.vlm_with_expert.get_vlm_model().text_model.layers[0].self_attn.q_proj
    require_same(query_module.weight.detach().t().contiguous().view(torch.uint16).numpy(), arrays["B_bf16"],
        "Original Q weights changed")
    replacement = torch.from_numpy(actual_words.copy()).view(torch.bfloat16).reshape(1, 177, 960)
    current = {}
    candidate_prefix = "actual-query" if candidate_plan is None else "actual-" + candidate_plan.roles_name
    candidate_names = tuple(candidate_prefix + "-repeat" + str(i) for i in range(2))
    if candidate_plan is not None:
        candidate_plan.bind_prefix(model, torch, manifest, arrays, actual_words, prefix, smolvlm_with_expert.apply_rope)

    def q_hook(_, incoming, output):
        if current["q_calls"] != 0 or incoming[0].dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise ValueError("Candidate query callback invocation/dtype changed")
        require_same(incoming[0].view(torch.uint16).numpy()[0], arrays["A_bf16"][:177], "Candidate query input changed")
        require_same(output.view(torch.uint16).numpy()[0], native_q, "Candidate native query output changed")
        current["q_calls"] += 1
        return replacement.clone()

    def cache_ids(cache):
        if set(cache) != set(range(16)):
            raise ValueError("Native prefix cache must contain exactly 16 layers")
        if any(set(layer) != {"key_states", "value_states"} for layer in cache.values()):
            raise ValueError("Unexpected native prefix cache fields")
        if any(value.shape != (1, 177, 5, 64) or value.dtype != torch.bfloat16 or value.device.type != "cpu"
               for layer in cache.values() for value in layer.values()):
            raise ValueError("Native prefix cache geometry/dtype/device changed")
        return {str(i) + ":" + name: tensor_id(cache[i][name], torch)
                for i in range(16) for name in ("key_states", "value_states")}

    def save_array(value):
        flat = torch.empty(value.numel(), dtype=value.dtype)
        flat.copy_(value.detach().reshape(-1))
        if value.dtype == torch.bfloat16:
            return flat.view(torch.uint16).reshape(value.shape).numpy().copy()
        return flat.reshape(value.shape).numpy().copy()

    def verify_trajectory(captures, *, original):
        if set(captures) != set(reference):
            raise ValueError("Incomplete or additional trajectory captures")
        for name, array in captures.items():
            expected = historical["source_run"]["captures"][name]
            dtype = {"torch.bfloat16": np.uint16, "torch.float32": np.float32,
                     "torch.bool": np.bool_, "torch.int64": np.int64}[expected["dtype"]]
            values = bf16_words_to_float(array) if expected["dtype"] == "torch.bfloat16" else array
            if array.dtype != dtype or list(array.shape) != expected["shape"] or not np.isfinite(values).all():
                raise ValueError("Native trajectory geometry/finite check failed: " + name)
            if original:
                require_same(array, reference[name], "Original native trajectory no longer closes archive: " + name)
        dt = -1.0 / 10
        for index in range(10):
            stem = f"step_{index:02d}"
            expected_time = torch.tensor(1.0 + index * dt, dtype=torch.float32).expand(1).numpy()
            require_same(captures[stem + "_timestep"], expected_time, "Original timestep stream changed")
            expected_update = torch.from_numpy(captures[stem + "_input_state"]) + dt * torch.from_numpy(captures[
                stem + "_velocity"])
            require_same(captures[stem + "_output_state"], expected_update.numpy(), "Native closed Euler update changed")
            next_name = f"step_{index + 1:02d}_input_state" if index < 9 else "final_full_state"
            require_same(captures[stem + "_output_state"], captures[next_name],
                "Candidate reused another trajectory's later state")
        require_same(captures["final_actions"], captures["final_full_state"][:, :, :6],
            "Native policy-coordinate action extent changed")

    def run(name, candidate):
        owners = ((model, "embed_prefix"), (model, "denoise_step"), (model.vlm_with_expert, "forward"))
        if any(method in owner.__dict__ for owner, method in owners):
            raise ValueError("Native source methods already have instance overrides")
        denoise_original, forward_original = model.denoise_step, model.vlm_with_expert.forward
        captures, steps, initial_cache, prefix_calls = {}, [], {}, []
        current.clear()
        current.update(q_calls=0)
        rng = torch.random.get_rng_state().clone()

        def embed(*args_, **kwargs):
            if set(kwargs) != {"state"} or len(args_) != 4 or prefix_calls:
                raise ValueError("Retained prefix called with unexpected arguments/count")
            observed = (torch.stack(args_[0], dim=1), torch.stack(args_[1], dim=1), *args_[2:], kwargs["state"])
            if {key: tensor_id(value, torch) for key, value in zip(INPUT_NAMES[:5], observed)} != {key: input_ids[
                key] for key in INPUT_NAMES[:5]}:
                raise ValueError("Retained prefix caller is not the original observation")
            prefix_calls.append(True)
            values = tuple(value.clone() for value in prefix)
            captures.update({key: save_array(value) for key, value in zip(prefix_names, values)})
            return values

        def encode(*args_, **kwargs):
            result = forward_original(*args_, **kwargs)
            if kwargs.get("fill_kv_cache"):
                if initial_cache:
                    raise ValueError("Native prefix was encoded more than once")
                initial_cache.update(cache_ids(result[1]))
                for field in ("key_states", "value_states"):
                    captures["prefix_" + field.replace("states", "cache")] = save_array(torch.stack([result[1][i][
                        field] for i in range(16)]))
            return result

        def denoise(*args_, **kwargs):
            index = len(steps)
            if args_ or index >= 10 or cache_ids(kwargs["past_key_values"]) != initial_cache:
                raise ValueError("Native denoising count/cache/argument contract changed")
            state_id, time_id = tensor_id(kwargs["x_t"], torch), tensor_id(kwargs["timestep"], torch)
            stem = f"step_{index:02d}"
            captures[stem + "_input_state"] = save_array(kwargs["x_t"])
            captures[stem + "_timestep"] = save_array(kwargs["timestep"])
            result = denoise_original(**kwargs)
            if (cache_ids(kwargs["past_key_values"]) != initial_cache or state_id != tensor_id(kwargs["x_t"], torch)
                    or time_id != tensor_id(kwargs["timestep"], torch)):
                raise ValueError("Native denoiser mutated its cache/state/timestep inputs")
            captures[stem + "_velocity"] = save_array(result)
            steps.append(index)
            return result

        model.embed_prefix, model.denoise_step, model.vlm_with_expert.forward = embed, denoise, encode
        handle = query_module.register_forward_hook(q_hook) if candidate else None
        try:
            if candidate and candidate_plan is not None:
                candidate_plan.begin_candidate()
            with torch.no_grad():
                full = model.sample_actions([inputs[0][:, i] for i in range(2)], [inputs[1][:, i] for i in range(2)], *inputs[
                    2:5], noise=inputs[5])
        finally:
            rng_after = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng)
            try:
                if candidate and candidate_plan is not None:
                    candidate_plan.remove_candidate_hooks()
            finally:
                if handle is not None:
                    handle.remove()
                for owner, method in owners:
                    delattr(owner, method)
        if len(steps) != 10 or len(prefix_calls) != 1 or current["q_calls"] != int(candidate):
            raise ValueError("Incomplete native prefix/trajectory/query traversal")
        captures["final_full_state"] = save_array(full)
        captures["final_actions"] = save_array(full[:, :, :6])
        for index in steps:
            captures[f"step_{index:02d}_output_state"] = captures[f"step_{index + 1:02d}_input_state"].copy(
                ) if index < 9 else captures["final_full_state"].copy()
        if candidate and candidate_plan is not None:
            candidate_plan.finish_candidate(captures)
        verify_trajectory(captures, original=not candidate)
        if not torch.equal(rng, rng_after):
            raise ValueError("Explicit-noise native replay consumed RNG")
        if input_ids != {key: tensor_id(value, torch) for key, value in zip(INPUT_NAMES, inputs)} or state_sha(model,
            torch) != STATE_SHA:
            raise ValueError("Native replay changed source inputs or parameter/buffer state")
        require_same(replacement.view(torch.uint16).numpy()[0], actual_words,
            "Actual query adaptation changed during native replay")
        np.savez_compressed(args.output / (name + ".npz"), **captures)
        return captures

    # A source refusal here prevents every candidate run.
    original = run("original-native", False)
    original_repeat = run("original-native-repeat", False)
    candidates = [run(name, True) for name in candidate_names]
    restored = run("restored-native", False)
    for left, right in ((original, original_repeat), (original, restored), (candidates[0], candidates[1])):
        for key in left:
            require_same(left[key], right[key], "Native replay repeat/restoration changed: " + key)
    if candidate_plan is None:
        for field in ("prefix_key_cache", "prefix_value_cache"):
            require_same(candidates[0][field][0], original[field][0], "Layer-zero K/V changed after a Q-only substitution")
    else:
        candidate_plan.check_layer0_caches(candidates)
    changed = {key: value.copy() for key, value in candidates[0].items()}
    changed["step_01_input_state"].view(np.uint32).flat[0] ^= 1
    try:
        verify_trajectory(changed, original=False)
    except ValueError:
        controls = [{"control": "changed_later_candidate_state_rejected_by_closed_trajectory", "refused": True}]
    else:
        raise AssertionError("Broken candidate trajectory was admitted")
    verify_declared_sources(source)
    for binding in bindings:
        identity(binding["path"], binding["sha256"])
    validator.recheck(provider_records)
    if candidate_plan is not None:
        candidate_plan.recheck()
    metrics = {}
    for key in ("final_full_state", "final_actions"):
        native, actual = original[key], candidates[0][key]
        delta = actual.astype(np.float64) - native.astype(np.float64)
        close = np.isclose(actual, native, rtol=1e-4, atol=1e-4)
        norm = float(np.linalg.norm(native.astype(np.float64)))
        metrics[key] = {"rtol": 1e-4, "atol": 1e-4, "passed": bool(close.all()), "failed_values": int((~close).sum()),
                        "max_abs": float(np.abs(delta).max()), "relative_l2": float(np.linalg.norm(
            delta) / norm) if norm else None}
    report.update(status="completed_retained_prefix_hybrid_diagnostic", source=source, native_environment=versions,
                  effective_configuration=effective_config,
                  physical_admission=admission, native_state_sha256=STATE_SHA, original_replay_inputs={"prefix": {
        key: array_identity(reference[key]) for key in prefix_names}, "noise": input_ids["noise"]},
                  checks={"all_16_original_cache_layers_raw_exact": True,
        "all_10_original_states_timesteps_velocities_updates_actions_raw_exact": True,
                          "original_baseline_passed_before_candidates": True,
        "candidate_later_caches_recomputed_by_original_native_forward": True,
                          "candidate_own10step_closed_trajectory": True, "layer0_KV_preserved_under_Q_only_substitution": True,
                          "candidate_repeats_and_original_restoration_raw_exact": True,
        "input_state_parameter_RNG_bookends": True},
                  negative_controls=controls, source_fidelity=metrics,
                  artifacts={name: identity(args.output / (name + ".npz")) for name in ("original-native",
        "original-native-repeat", *candidate_names, "restored-native")},
                  source_bookends=bindings, provider_bookends=provider_records,
                  claim_limits=manifest["claim_limits"] + ["retained_prefix_replay_not_fresh_vision",
        "policy_coordinates_before_original_postprocessing", "physical_units_unqualified"])
    if candidate_plan is not None:
        candidate_plan.extend_report(report)


def main():
    from workload_cli import run_report
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-bundle", "provider-receipt", "producer-root", "model-root", "historical-source-root",
        "observation-input", "top-receipt", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("native-manifest-sha256", "provider-receipt-sha256", "validator-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args()
    helper_sources = ("capture_tinyllama_native", "capture_smolvla", "capture_smolvla_full_query", "evaluate_smolvla_attention")
    args.workload_helper_bindings = [identity(sys.modules[name].__file__) for name in helper_sources]
    args.output = safe_path(args.output_dir)
    if args.output == safe_path(artifacts_dir()) or not args.output.is_relative_to(safe_path(artifacts_dir())):
        parser.error("Output must be a new directory beneath the configured artifact root")
    report = {"status": "failed", "claim": __doc__, "command": sys.argv, "cwd": str(Path.cwd()),
              "source_script": identity(__file__), "python_executable": sys.executable,
              "complete_atlas_execution": False, "timing_qualified": False, "application_quality_qualified": False,
              "vision_prefix_recomputed": False, "physical_units_qualified": False}
    return run_report(args, evaluate,
        filename='receipt.json', output=args.output, report=report, atomic=True)


if __name__ == "__main__":
    raise SystemExit(main())
