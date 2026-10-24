"""Run one dynamic Atlas transformer block inside retained-prefix native policy.

This experiment replaces seven real selected-layer Linear invocations, including
their current inputs, using an explicitly pinned workload producer. Native
normalization, attention, residuals, nonlinearities, later layers and ten Euler
updates execute through the original source. The retained prefix is an archived
intermediate, so this does not establish fresh vision or a complete device policy.
"""

from merlin.common.jsonio import write_pretty_json
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

import source_relocation
from source_relocation import argument_relocator, read_metadata

import workload_replay_admission as child_admission

from capture_smolvla import (CHECKPOINT_REVISION, WEIGHTS_SHA, RECEIPT_SHA,
                            ARCHIVE_SHA, OBSERVATION_SHA, EXPERT_SOURCE_SHA, LLAMA_SHA)
from capture_tinyllama_native import (identity, safe_path, array_identity,
                                      verify_declared_sources, bf16_words_to_float)
from evaluate_smolvla_attention import MASK_SOURCE_SHA, require_same
from evaluate_smolvla_retained_prefix import (CONFIG_SHA, INPUT_SHA, STATE_SHA,
                                             TOP_RECEIPT_SHA, INPUT_NAMES,
                                             tensor_id, state_sha)


ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
EXPECTED_DIMS = ((960, 960), (960, 320), (960, 320), (960, 960),
                 (960, 2560), (960, 2560), (2560, 960))


def select_vlm_layer(layers, index):
    if len(layers) != 16 or type(index) is not int or not 0 <= index < len(layers):
        raise ValueError("Require one of the original sixteen VLM layers")
    return layers[index]


def rng_identity():
    state = np.random.get_state()
    return {"python": repr(random.getstate()), "numpy": [state[0], array_identity(state[1]), *state[2:]]}


def attention_control_identity(arguments, kwargs, torch):
    if arguments or set(kwargs) != {"attention_mask", "position_ids", "past_key_values", "inputs_embeds", "use_cache",
        "fill_kv_cache"}:
        raise ValueError("Native transformer control ABI changed")
    controls = {key: tensor_id(kwargs[key], torch) for key in ("attention_mask", "position_ids")}
    controls.update({key: kwargs[key] for key in ("use_cache", "fill_kv_cache")})
    if any(type(controls[key]) is not bool for key in ("use_cache", "fill_kv_cache")):
        raise ValueError("Native transformer cache flags changed")
    return controls


def log(args, message):
    stamp = datetime.now(timezone.utc).isoformat()
    with safe_path(args.output / "LOG.md").open("a") as stream:
        stream.write(f"\n- {stamp} {message}\n")
    print(message, flush=True)


def compare(actual, native):
    actual, native = np.asarray(actual, np.float64), np.asarray(native, np.float64)
    delta = actual - native
    close = np.isclose(actual, native, rtol=1e-4, atol=1e-4)
    denominator = float(np.linalg.norm(native))
    return {"rtol": 1e-4, "atol": 1e-4, "passed": bool(close.all()),
            "failed_values": int((~close).sum()), "values": int(close.size),
            "max_abs": float(np.abs(delta).max()),
            "relative_l2": float(np.linalg.norm(delta) / denominator) if denominator else None}


class DynamicBlock:
    """Trusted experiment adapter; every producer call sees runtime Linear input."""

    def __init__(self, args, layer, torch, source, report):
        self.args, self.torch, self.source, self.report = args, torch, source, report
        self.code_identities = args.producer_identities
        self.modules = {}
        for role, shape in zip(ROLES, EXPECTED_DIMS):
            owner, name = role.split(".")
            module = getattr(getattr(layer, owner), name)
            if (module.in_features, module.out_features) != shape or module.bias is not None:
                raise ValueError("Selected checkpoint Linear geometry/bias changed: " + role)
            if module.weight.dtype != torch.bfloat16:
                raise ValueError("Selected native Linear dtype changed: " + role)
            self.modules[role] = module
        self.weights = {role: tensor_id(module.weight, torch) for role, module in self.modules.items()}
        self.handles, self.admissions, self.observations =[], {}, []
        self.repeat = None

    def begin(self, repeat):
        if self.handles:
            raise ValueError("Candidate hooks already installed")
        self.repeat, self.seen, self.order = repeat, set(), []
        try:
            for role, module in self.modules.items():
                self.handles.append(module.register_forward_hook(self.hook(role)))
        except BaseException:
            self.remove()
            raise

    def remove(self):
        while self.handles:
            self.handles.pop().remove()

    def hook(self, role):
        def dispatch(module, incoming, native_output):
            torch = self.torch
            if (role in self.seen or len(incoming) != 1
                    or incoming[0].shape !=(1, 177, module.in_features)
                    or native_output.shape !=(1, 177, module.out_features)
                    or incoming[0].dtype != torch.bfloat16 or native_output.dtype != torch.bfloat16
                    or tensor_id(module.weight, torch) != self.weights[role]):
                raise ValueError("Runtime Linear count/geometry/dtype/weight differs: " + role)
            a = incoming[0].detach().float().numpy()[0].copy()
            b = module.weight.detach().float().t().numpy().copy()
            native = native_output.detach().float().numpy()[0].copy()
            if not all(np.isfinite(x).all() for x in (a, b, native)):
                raise ValueError("Nonfinite runtime projection: " + role)
            current = {"A": array_identity(a), "B": array_identity(b), "native_C": array_identity(native)}
            if self.repeat == 0:
                kernel = self.args.output / "kernels" / role
                kernel.mkdir(parents=True, exist_ok=False)
                arrays = {"A": np.pad(a, ((0, 15), (0, 0))), "B": b,
                          "native_C": np.pad(native, ((0, 15), (0, 0)))}
                np.savez(kernel / "operands.npz", **arrays)
                manifest = {"schema_version": 1, "source_dtype": "float32",
                            "npz_sha256": identity(kernel / "operands.npz")["sha256"],
                            "source": {"kind": "runtime_original_native_Linear_invocation",
                                       "checkpoint": self.source["weights"],
                                       "native_dtype": "bfloat16",
                                       "runtime_arrays": current,
                                       "upstream_roles_executed": list(self.order),
                                       "retained_prefix_not_fresh_vision": True},
                            "site": {"module": "model.vlm_with_expert.vlm.model.text_model.layers." + str(
                    self.args.vlm_layer) + "." + role,
                                     "native_rows": 177, "export_rows": 192,
                                     "shape_M_K_N": [192, module.in_features, module.out_features],
                                     "native_masks_positions_extended": False}}
                write_pretty_json(safe_path(kernel / "manifest.json"), manifest, sort_keys=False, allow_nan=False)
                identity(self.args.producer, self.args.producer_sha256)
                command =[str(self.args.model_python), str(self.args.producer),
                           "--model-root", str(self.args.model_root),
                           "--input", str(kernel / "operands.npz"),
                           "--manifest", str(kernel / "manifest.json"),
                           "--out", str(kernel / "model"), "--repeats", "1",
                           "--max-cycles", str(self.args.max_cycles)]
                write_pretty_json(safe_path(kernel / "command.json"), {"argv": command, "cwd": str(self.args.output),
                                                      "timeout_seconds": self.args.kernel_timeout}, sort_keys=False,
                    allow_nan=False)
                log(self.args, "Started dynamic " + role + " " + str(manifest["site"]["shape_M_K_N"]))
                with (kernel / "model.log").open("w") as stream:
                    result = subprocess.run(command, cwd=self.args.output, stdout=stream,
                                            stderr=subprocess.STDOUT, timeout=self.args.kernel_timeout,
                                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                                                     OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
                if result.returncode != 0:
                    raise ValueError("Dynamic kernel model process failed: " + role)
                words, receipt, admission = self.admit_role(role, current)
                receipt_id = admission["receipt"]
                self.admissions[role] = {"runtime_arrays": current, "receipt": receipt_id,
                                         "strict_child_admission": admission,
                                         "result": identity(kernel / "model/result.npz"),
                                         "words": words[:177].copy(),
                                         "cycles": [x["cycles"] for x in receipt["runs"]],
                                         "target_vs_native": compare(bf16_words_to_float(words[:177]), native),
                                         "target_vs_FP32_matmul": compare(bf16_words_to_float(words[:177]), a @ b),
                                         "upstream_roles": list(self.order)}
                log(self.args, "Passed exact model/oracle " + role + " cycles=" + str(self.admissions[role]["cycles"]))
            else:
                item = self.admissions[role]
                if item["runtime_arrays"] != current or item["upstream_roles"] != self.order:
                    raise ValueError("Second source continuation has another dynamic input: " + role)
                identity(item["receipt"]["path"], item["receipt"]["sha256"])
                identity(item["result"]["path"], item["result"]["sha256"])
            self.seen.add(role)
            self.order.append(role)
            return torch.from_numpy(self.admissions[role]["words"].copy()).view(torch.bfloat16).reshape(1, 177,
                module.out_features)
        return dispatch

    def admit_role(self, role, runtime_arrays):
        module = self.modules[role]
        return child_admission.admit_child_result(
            self.args.output / "kernels" / role / "model", self.code_identities, runtime_arrays,
            (192, module.in_features, module.out_features), 177)

    def finish(self):
        if self.handles or self.seen != set(ROLES) or tuple(self.order) != ROLES:
            raise ValueError("Incomplete/reordered dynamic selected block: " + str(self.order))
        for role in self.modules:
            item = self.admissions[role]
            words, _, binding = self.admit_role(role, item["runtime_arrays"])
            if binding["receipt"] != item["receipt"]:
                raise ValueError("Child receipt changed during source continuation: " + role)
            require_same(words[:177], item["words"], "Child physical result changed during continuation")
        self.observations.append(list(self.order))

    def results(self):
        return {role: {key: value for key, value in item.items() if key != "words"}
                for role, item in self.admissions.items()}


def evaluate(args, report):
    manifest_id = identity(args.native_manifest, args.native_manifest_sha256)
    manifest = read_metadata(args, report, args.native_manifest, manifest_id["sha256"])
    source = manifest["source"]
    if source["revision"] != CHECKPOINT_REVISION:
        raise ValueError("Unadmitted checkpoint revision")
    verify_declared_sources(source)
    bindings =[identity(__file__), identity(source_relocation.__file__), identity(child_admission.__file__), manifest_id,
                *args.producer_identities.values(),
                args.model_interpreter_binding["physical_binary"], args.native_interpreter_binding["physical_binary"],
                identity(source["weights"]["path"], WEIGHTS_SHA),
                identity(source["historical_source_receipt"]["path"], RECEIPT_SHA),
                identity(source["historical_original_archive"]["path"], ARCHIVE_SHA),
                identity(source["observation_receipt"]["path"], OBSERVATION_SHA),
                identity(args.observation_input, INPUT_SHA)]
    for name in ("capture_smolvla.py", "capture_tinyllama_native.py", "capture_smolvla_full_query.py",
                 "evaluate_smolvla_attention.py", "evaluate_smolvla_retained_prefix.py"):
        bindings.append(identity(Path(__file__).with_name(name)))
    historical = read_metadata(args, report, source["historical_source_receipt"]["path"], RECEIPT_SHA)
    if (not historical["source_reference_unchanged"] or historical["source_state_sha256"] != STATE_SHA
            or historical["source_run"]["output_file"]["sha256"] != ARCHIVE_SHA or historical["denoising_steps"] != 10):
        raise ValueError("Unadmitted original native trajectory")
    top_path = safe_path(args.top_receipt)
    bindings.append(identity(top_path, TOP_RECEIPT_SHA))
    top = read_metadata(args, report, top_path, TOP_RECEIPT_SHA)
    policy_dir = safe_path(source["weights"]["path"]).parent
    bindings.append(identity(policy_dir / "config.json", CONFIG_SHA))
    with np.load(source["historical_original_archive"]["path"], allow_pickle=False) as archive:
        reference = {name: archive[name].copy() for name in historical["source_run"]["captures"]}
    with np.load(args.observation_input, allow_pickle=False) as archive:
        if set(archive.files) != set(INPUT_NAMES):
            raise ValueError("Observation fields changed")
        input_arrays = {name: archive[name].copy() for name in INPUT_NAMES}
    import torch
    from transformers.models.llama import modeling_llama
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla import smolvlm_with_expert, modeling_smolvla
    versions = {key: importlib.metadata.version(key) for key in ("torch", "transformers", "lerobot", "numpy")}
    if versions != source["versions"] or sys.version != source["python"]:
        raise ValueError("Native environment differs")
    bindings +=[identity(smolvlm_with_expert.__file__, EXPERT_SOURCE_SHA),
                 identity(modeling_llama.__file__, LLAMA_SHA), identity(modeling_smolvla.__file__, MASK_SOURCE_SHA),
                 identity(Path(torch.__file__).parent / "lib/libtorch_cpu.so", source["torch_cpu_library"]["sha256"])]
    package_root = safe_path(smolvlm_with_expert.__file__).parents[3]
    for record in top["factory_artifacts"]:
        path = safe_path(record["path"])
        if path.suffix == ".py" and path.is_relative_to(package_root):
            bindings.append(identity(path, record["sha256"]))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = PreTrainedConfig.from_pretrained(str(policy_dir), local_files_only=True)
    if not isinstance(config, SmolVLAConfig):
        raise ValueError("Pinned local configuration is not SmolVLA")
    config.device = "cpu"
    config.vlm_model_name = str(safe_path(source["backbone_config"]["path"]).parent)
    log(args, "Loading exact original policy locally, CPU/offline")
    policy = SmolVLAPolicy.from_pretrained(str(policy_dir), config=config, local_files_only=True, strict=True).eval(
        ).requires_grad_(False)
    model = policy.model
    if model._rtc_enabled() or config.compile_model or state_sha(model, torch) != STATE_SHA:
        raise ValueError("Original model state/compilation/RTC differ")
    if (model.vlm_with_expert.num_vlm_layers, model.vlm_with_expert.num_expert_layers) !=(16, 16):
        raise ValueError("Original prefix/expert layers differ")
    inputs = tuple(torch.from_numpy(input_arrays[name]) for name in INPUT_NAMES)
    input_ids = {name: tensor_id(value, torch) for name, value in zip(INPUT_NAMES, inputs)}
    if input_ids != historical["source_run"]["input_bindings"]:
        raise ValueError("Observation does not match exact original replay inputs")
    prefix_names =("prefix_embeddings", "prefix_pad_mask", "prefix_attention_mask")
    prefix = tuple(torch.from_numpy(reference[name].copy()) for name in prefix_names)
    if prefix[0].shape !=(1, 177, 960) or prefix[1].shape !=(1, 177) or int(prefix[1].sum()) != 141:
        raise ValueError("Retained prefix extent changed")
    layer = select_vlm_layer(model.vlm_with_expert.get_vlm_model().text_model.layers, args.vlm_layer)
    block = DynamicBlock(args, layer, torch, source, report)
    report.update(source=source, source_bookends=bindings, native_state_sha256=STATE_SHA,
                  native_environment=versions, original_replay_inputs=input_ids,
                  selected_vlm_layer=args.vlm_layer)
    write_pretty_json(safe_path(args.output / "receipt.json"), report, sort_keys=False, allow_nan=False)

    def save_array(value):
        flat = torch.empty(value.numel(), dtype=value.dtype)
        flat.copy_(value.detach().reshape(-1))
        if value.dtype == torch.bfloat16:
            return flat.view(torch.uint16).reshape(value.shape).numpy().copy()
        return flat.reshape(value.shape).numpy().copy()

    def cache_ids(cache):
        if set(cache) != set(range(16)) or any(set(x) != {"key_states", "value_states"} for x in cache.values()):
            raise ValueError("Native prefix cache fields/layer count changed")
        if any(x.shape !=(1, 177, 5, 64) or x.dtype != torch.bfloat16
               for item in cache.values() for x in item.values()):
            raise ValueError("Native prefix cache geometry/dtype changed")
        return {str(i) + ":" + field: tensor_id(cache[i][field], torch)
                for i in range(16) for field in ("key_states", "value_states")}

    def verify_trajectory(captures, original):
        if set(captures) != set(reference):
            raise ValueError("Incomplete trajectory capture")
        for name, value in captures.items():
            expected = reference[name]
            decoded = bf16_words_to_float(value) if value.dtype == np.uint16 else value
            if value.shape != expected.shape or value.dtype != expected.dtype or not np.isfinite(decoded).all():
                raise ValueError("Trajectory geometry/dtype/finite check failed: " + name)
            if original:
                require_same(value, expected, "Original native archive did not close: " + name)
        dt = -1.0 / 10
        for index in range(10):
            stem = f"step_{index:02d}"
            expected_time = torch.tensor(1.0 + index * dt, dtype=torch.float32).expand(1).numpy()
            require_same(captures[stem + "_timestep"], expected_time, "Euler timestep changed")
            expected = torch.from_numpy(captures[stem + "_input_state"]) + dt * torch.from_numpy(captures[stem + "_velocity"])
            require_same(captures[stem + "_output_state"], expected.numpy(), "Euler update did not close")
            next_name = f"step_{index + 1:02d}_input_state" if index < 9 else "final_full_state"
            require_same(captures[stem + "_output_state"], captures[next_name], "Later state belongs to another trajectory")
        require_same(captures["final_actions"], captures["final_full_state"][:,:,:6], "Action extent changed")

    control_records = {}

    def run(name, candidate, repeat=0):
        owners = ((model, "embed_prefix"), (model, "denoise_step"), (model.vlm_with_expert, "forward"))
        if any(method in owner.__dict__ for owner, method in owners):
            raise ValueError("Original methods already overridden")
        denoise_original, forward_original = model.denoise_step, model.vlm_with_expert.forward
        captures, steps, initial_cache, prefix_calls, attention_controls = {}, [], {}, [], []
        rng = torch.random.get_rng_state().clone()
        other_rng = rng_identity()

        def embed(*arguments, **kwargs):
            if set(kwargs) != {"state"} or len(arguments) != 4 or prefix_calls:
                raise ValueError("Retained prefix call count/arguments differ")
            observed =(torch.stack(arguments[0], dim=1), torch.stack(arguments[1], dim=1), *arguments[2:], kwargs["state"])
            if {key: tensor_id(value, torch) for key, value in zip(INPUT_NAMES[:5], observed)} != {key: input_ids[
                key] for key in INPUT_NAMES[:5]}:
                raise ValueError("Retained prefix caller differs from pinned observation")
            prefix_calls.append(True)
            values = tuple(value.clone() for value in prefix)
            captures.update({key: save_array(value) for key, value in zip(prefix_names, values)})
            return values

        def encode(*arguments, **kwargs):
            controls = attention_control_identity(arguments, kwargs, torch)
            result = forward_original(*arguments, **kwargs)
            if controls != attention_control_identity(arguments, kwargs, torch):
                raise ValueError("Native transformer mutated mask/position/cache controls")
            attention_controls.append(controls)
            if kwargs.get("fill_kv_cache"):
                if initial_cache:
                    raise ValueError("Duplicate native prefix encoding")
                initial_cache.update(cache_ids(result[1]))
                for field in ("key_states", "value_states"):
                    captures["prefix_" + field.replace("states", "cache")] = save_array(torch.stack([result[1][i][
                        field] for i in range(16)]))
            return result

        def denoise(*arguments, **kwargs):
            index = len(steps)
            if arguments or index >= 10 or cache_ids(kwargs["past_key_values"]) != initial_cache:
                raise ValueError("Denoising count/cache/arguments differ")
            state_id, time_id = tensor_id(kwargs["x_t"], torch), tensor_id(kwargs["timestep"], torch)
            stem = f"step_{index:02d}"
            captures[stem + "_input_state"] = save_array(kwargs["x_t"])
            captures[stem + "_timestep"] = save_array(kwargs["timestep"])
            result = denoise_original(**kwargs)
            if (cache_ids(kwargs["past_key_values"]) != initial_cache or state_id != tensor_id(kwargs["x_t"], torch)
                    or time_id != tensor_id(kwargs["timestep"], torch)):
                raise ValueError("Denoiser changed its cache/state/time inputs")
            captures[stem + "_velocity"] = save_array(result)
            steps.append(index)
            return result

        model.embed_prefix, model.denoise_step, model.vlm_with_expert.forward = embed, denoise, encode
        log(args, "Started source policy continuation " + name)
        try:
            if candidate:
                block.begin(repeat)
            with torch.no_grad():
                full = model.sample_actions([inputs[0][:, i] for i in range(2)], [inputs[1][:, i] for i in range(2)],
                                            *inputs[2:5], noise=inputs[5])
        finally:
            rng_after = torch.random.get_rng_state().clone()
            torch.random.set_rng_state(rng)
            block.remove()
            for owner, method in owners:
                delattr(owner, method)
        if len(steps) != 10 or len(prefix_calls) != 1 or len(attention_controls) != 11:
            raise ValueError("Incomplete prefix/ten-step trajectory")
        if attention_controls[0]["fill_kv_cache"] is not True or any(x["fill_kv_cache"] for x in attention_controls[1:]):
            raise ValueError("Native prefix/suffix cache phase changed")
        if control_records and attention_controls != control_records["original-native"]:
            raise ValueError("Masks/positions/cache flags differ from original native trajectory")
        control_records[name] = attention_controls
        if candidate:
            block.finish()
        captures["final_full_state"] = save_array(full)
        captures["final_actions"] = save_array(full[:,:,:6])
        for index in steps:
            captures[f"step_{index:02d}_output_state"] = captures[f"step_{index + 1:02d}_input_state"].copy(
                ) if index < 9 else captures["final_full_state"].copy()
        verify_trajectory(captures, not candidate)
        if not torch.equal(rng, rng_after) or rng_identity() != other_rng or input_ids != {key: tensor_id(value, torch) for key,
            value in zip(INPUT_NAMES, inputs)}:
            raise ValueError("RNG/input bookends differ")
        if state_sha(model, torch) != STATE_SHA:
            raise ValueError("Native parameter/buffer state changed")
        np.savez_compressed(args.output /(name + ".npz"), **captures)
        log(args, "Passed source/closed trajectory " + name)
        return captures

    original = run("original-native", False)
    original_repeat = run("original-native-repeat", False)
    report["original_baseline_passed_before_candidates"] = True
    write_pretty_json(safe_path(args.output / "receipt.json"), report, sort_keys=False, allow_nan=False)
    candidates =[run("dynamic-block-repeat" + str(i), True, i) for i in range(2)]
    restored = run("restored-native", False)
    for left, right in ((original, original_repeat), (original, restored), (candidates[0], candidates[1])):
        for key in left:
            require_same(left[key], right[key], "Repeat/restoration changed: " + key)
    changed = {key: value.copy() for key, value in candidates[0].items()}
    changed["step_01_input_state"].view(np.uint32).flat[0] ^= 1
    try:
        verify_trajectory(changed, False)
    except ValueError:
        controls =[{"control": "later_state_payload_bit_flip", "refused": True}]
    else:
        raise AssertionError("Broken later state was accepted")
    for item in bindings:
        identity(item["path"], item["sha256"])
    verify_declared_sources(source)
    report.update(status="passed_dynamic_block_model_and_retained_policy_continuation",
                  kernels=block.results(), negative_controls=controls,
                  checks={"native_baseline_full_archive_exact": True,
                          "seven_actual_runtime_Linear_inputs": True,
                          "seven_full_target_model_oracle_equal": True,
                          "candidate_own_ten_Euler_steps_closed": True,
                          "candidate_repeats_native_restoration_exact": True,
                          "input_parameter_RNG_source_bookends_unchanged": True},
                  attention_controls=control_records,
                  quality_informational_relaxation_of_failing_native_gates={key: compare(candidates[0][key], original[key])
                      for key in ("final_full_state", "final_actions")},
                  quality_reference=(
                      'unchanged original mixed BF16/FP32 CPU policy, final action arrays FP32; not an '
                      'all-FP32 policy'
                  ),
                  model_cycles_total=sum(sum(x["cycles"]) for x in block.admissions.values()),
                  artifacts={name: identity(args.output /(name + ".npz")) for name in (
                      "original-native", "original-native-repeat", "dynamic-block-repeat0", "dynamic-block-repeat1",
        "restored-native")})


def main():
    from workload_cli import run_report
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-manifest", "observation-input", "producer", "model-root", "top-receipt", "output"):
        parser.add_argument("--" + name, type=safe_path, required=True)
    for name in ("native-manifest-sha256", "producer-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--model-python", type=Path, required=True)
    parser.add_argument("--vlm-layer", type=int, choices=range(16), default=0)
    parser.add_argument("--max-cycles", type=int, default=100000000)
    parser.add_argument("--kernel-timeout", type=int, default=5400)
    parser.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW")
    args = parser.parse_args()
    if args.max_cycles <= 0 or args.kernel_timeout <= 0:
        parser.error("Cycle and wall bounds must be positive")
    if safe_path(child_admission.__file__) != safe_path(Path(__file__).with_name("workload_replay_admission.py")):
        parser.error("Child admission helper imported outside the bound experiment source")
    try:
        args.model_interpreter_binding = child_admission.admit_executable(args.model_python)
        args.native_interpreter_binding = child_admission.admit_executable(sys.executable)
        args.model_python = Path(args.model_interpreter_binding["entry"])
        args.producer_identities = child_admission.producer_identities(args.producer, args.producer_sha256)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report = {"schema_version": 1, "status": "failed", "command": sys.argv, "cwd": os.getcwd(),
              "python": sys.executable, "source_script": identity(__file__),
              "model_interpreter": args.model_interpreter_binding,
              "native_interpreter": args.native_interpreter_binding,
              "admission_helper": identity(child_admission.__file__),
              "complete_atlas_execution": False, "full_policy_quality_qualified": False,
              "vision_prefix_recomputed": False, "physical_units_qualified": False,
              "full_SoC_executed": False, "timing_qualified": False,
              "scope": "one dynamically executed complete VLM block at index " + str(
        args.vlm_layer) + ", retained-prefix host policy continuation",
              "claim_limits": ["retained TRAIN-prefix intermediate, not fresh model input",
                               "host Python continuation, not native C/SoC execution",
                               "model timing excludes host conversion/packing/continuation",
                               "native 1e-4 tolerances and reference unchanged",
                               "quality numbers are informational relaxation of native fidelity gates",
                               "all other VLM/expert/vision operators are not Atlas executed"]}
    def failed(error, report):
        log(args, "Gate failed: " + str(error))
    return run_report(args, evaluate,
        filename='receipt.json', output=args.output, report=report,
        summary=lambda report: {"status": report["status"],
            "error": report.get("error"), "cycles": report.get("model_cycles_total")},
        on_error=failed)


if __name__ == "__main__":
    raise SystemExit(main())
