"""Evaluate two actual model-result convolutions inside unchanged native ResNet.

Conv2 must belong to the admitted actual-conv1/native-BN/ReLU candidate chain.
Native BN2/ReLU/conv3/residual and the remaining graph execute on the host. This
semantic hybrid computes/checks native convolutions before replacing outputs;
it establishes neither runtime offload, application quality, nor timing.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
from types import SimpleNamespace

from merlin.common.paths import artifacts_dir, repo_root
from capture_resnet import (RECEIPT_SHA256, identity, require_identity, safe_path,
                            load_module, load_native_admission, tensor_identity, git_head, write_manifest)
from capture_resnet_3x3 import patch_rows
from evaluate_resnet_substitution import load_results
from capture_tinyllama_native import load_provider_validator


def validate_chain(first, first_admission, second):
    chain = second["source"]["candidate_chain"]
    site = second["site"]
    if (second["status"] != "passed" or second["source_dtype"] != "float32"
            or site["module"] != "layer1.0.conv2" or site["input_shape"] != [1, 64, 56, 56]
            or site["weight_shape"] != [64, 64, 3, 3] or site["spatial_flat_range"] != [0, 3136]
            or site["output_channel_range"] != [0, 64] or site["k_range"] != [0, 576] or not site["complete_k"]
            or site["stride"] != [1, 1] or site["padding"] != [1, 1] or site["dilation"] != [1, 1]
            or site["groups"] != 1 or site["output_shape"] != [1, 64, 56, 56]
            or chain["provider_receipt"] != first_admission["provider_receipt"]
            or chain["provider_result"] != first_admission["provider_result"]
            or chain["provider_recipe"] != first_admission["provider_recipe"]
            or chain["provider_output_formatter"] != first_admission["provider_output_formatter"]
            or chain["original_native_logits"] != first["source"]["native_logits"]
            or any(second["source"][key] != first["source"][key] for key in
                   ("checkpoint", "input_stream", "input", "state_dict_sha256", "versions", "sample_index", "sample", "split"))):
        raise ValueError("Conv2 source is not the complete admitted conv1 candidate chain")


def validate_third_chain(first, first_admission, second_details, third, args):
    third_chain = third["source"]["candidate_chain"]
    third_site = third["site"]
    if (third["status"] != "passed" or third["source_dtype"] != "float32"
            or third_site["module"] != "layer1.0.conv3"
            or third_site["input_shape"] != [1, 64, 56, 56]
            or third_site["weight_shape"] != [256, 64, 1, 1]
            or third_site["output_shape"] != [1, 256, 56, 56]
            or third_site["spatial_flat_range"] != [0, 3136]
            or third_site["output_channel_range"] != [0, 256]
            or third_site["k_range"] != [0, 64] or not third_site["complete_k"]
            or third_chain["conv1_admission"] != first_admission
            or third_chain["conv2_admission"] != second_details
            or third_chain["conv2_native_manifest"] != {"path": str(safe_path(args.conv2_native_bundle) / "manifest.json"), "sha256": args.conv2_native_manifest_sha256}
            or third_chain["conv2_provider_receipt"] != require_identity({"path": str(safe_path(args.conv2_provider_receipt)), "sha256": args.conv2_provider_receipt_sha256})
            or third_chain["conv2_provider_result"] != identity(safe_path(args.conv2_provider_receipt).parent / "result.npz")
            or third_chain["original_native_logits"] != first["source"]["native_logits"]
            or any(third["source"][key] != first["source"][key] for key in
                   ("checkpoint", "input_stream", "input", "state_dict_sha256", "versions", "sample_index", "sample", "split"))):
        raise ValueError("Conv3 source is not the complete admitted two-operator candidate chain")


def require_tensor(actual, expected, message):
    if tensor_identity(actual) != tensor_identity(expected):
        raise ValueError(message)


def validate_shortcut_source(first, shortcut):
    site = shortcut["site"]
    if (shortcut["status"] != "passed" or shortcut["source_dtype"] != "float32"
            or site["module"] != "layer1.0.downsample.0" or site["input_shape"] != [1, 64, 56, 56]
            or site["weight_shape"] != [256, 64, 1, 1] or site["spatial_flat_range"] != [0, 3136]
            or site["output_channel_range"] != [0, 256] or site["k_range"] != [0, 64] or not site["complete_k"]
            or shortcut["tensors"]["A"] != first["tensors"]["A"]
            or any(shortcut["source"][key] != first["source"][key] for key in
                   ("checkpoint", "input_stream", "input", "state_dict_sha256", "versions", "sample_index", "sample", "split", "native_logits"))):
        raise ValueError("Shortcut source is not the complete original shared-input convolution")


def evaluate(args, report, *, capture_sink=None, third_operator=None, shortcut_operator=None):
    import numpy as np
    if capture_sink is not None and third_operator is not None:
        raise ValueError("Conv3 capture requires the unchanged two-operator native suffix")
    if shortcut_operator is not None and (third_operator is None or capture_sink is not None):
        raise ValueError("Shortcut substitution requires the three admitted main convolutions")
    third_controls = []
    historical_id = require_identity({"path": str(args.conv1_hybrid_receipt), "sha256": args.conv1_hybrid_receipt_sha256})
    historical = json.loads(Path(historical_id["path"]).read_text())
    if historical["status"] != "completed_hybrid_diagnostic":
        raise ValueError("Require the explicitly pinned prior one-operator diagnostic")
    first_admission = {}
    first, first_arrays, first_target, records = load_results(SimpleNamespace(
        native_bundle=Path(historical["source_bundle"]["path"]).parent,
        native_manifest_sha256=historical["source_bundle"]["sha256"],
        provider_receipt=Path(historical["provider_receipt"]["path"]),
        provider_receipt_sha256=historical["provider_receipt"]["sha256"],
        provider_arithmetic=Path(historical["provider_output_formatter"]["path"]),
        site="layer1.0.conv1", sample_index=0), first_admission)
    producer = safe_path(args.producer_root)
    validator, validator_id = load_provider_validator(
        producer, args.validator_sha256, args.conv2_provider_receipt, args.conv2_provider_receipt_sha256)
    second, second_arrays, second_target, second_records, second_details = validator.admit_contraction(
        receipt_path=safe_path(args.conv2_provider_receipt), receipt_sha256=args.conv2_provider_receipt_sha256,
        native_bundle=safe_path(args.conv2_native_bundle), native_manifest_sha256=args.conv2_native_manifest_sha256,
        producer_root=producer, model_root=safe_path(args.model_root))
    validate_chain(first, first_admission, second)
    if third_operator is not None:
        third = third_operator["manifest"]
        validate_third_chain(first, first_admission, second_details, third, args)
        third_chain = third["source"]["candidate_chain"]
        changed_third = copy.deepcopy(third)
        changed_third["source"]["candidate_chain"]["conv2_provider_result"]["sha256"] = "0" * 64
        try:
            validate_third_chain(first, first_admission, second_details, changed_third, args)
        except ValueError:
            third_controls.append({"control": "different_conv2_result_in_conv3_source_chain", "refused": True})
        else:
            raise AssertionError("A different conv2 source chain was admitted")
    if shortcut_operator is not None:
        shortcut = shortcut_operator["manifest"]
        validate_shortcut_source(first, shortcut)
        changed_shortcut = copy.deepcopy(shortcut)
        changed_shortcut["tensors"]["A"]["raw_sha256"] = "0" * 64
        try:
            validate_shortcut_source(first, changed_shortcut)
        except ValueError:
            third_controls.append({"control": "different_shared_input_in_shortcut_source", "refused": True})
        else:
            raise AssertionError("A different shortcut source input was admitted")
    changed_chain = copy.deepcopy(second)
    changed_chain["source"]["candidate_chain"]["provider_result"]["sha256"] = "0" * 64
    try:
        validate_chain(first, first_admission, changed_chain)
    except ValueError:
        chain_control = {"control": "different_conv1_result_in_conv2_source_chain", "refused": True}
    else:
        raise AssertionError("A different conv1 source chain was admitted")
    source = first["source"]
    chain = second["source"]["candidate_chain"]
    records += [historical_id, validator_id, identity(__file__), identity(Path(__file__).with_name("capture_resnet_3x3.py")),
                identity(Path(__file__).with_name("evaluate_resnet_substitution.py"))]
    records += [require_identity(record) for record in source["immutable_bindings"]]
    records += [require_identity(record) for record in source["original_preparation_bindings"]]
    records += [require_identity(record) for record in second["source"]["immutable_bindings"]]
    original_reference = require_identity(second["original_reference"])
    records.append(original_reference)
    with np.load(original_reference["path"], allow_pickle=False) as archive:
        original_second = archive["native_C"].copy()
        original_saved_logits = archive["native_logits"].copy()
    receipt_id = require_identity(source["receipt"])
    if receipt_id["sha256"] != RECEIPT_SHA256:
        raise ValueError("Wrong original native development source")
    receipt = json.loads(Path(receipt_id["path"]).read_text())
    helper = load_native_admission(receipt, records, report)
    import torch
    import torchvision
    import torchvision.models.resnet as implementation
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__}
    if versions != source["versions"] or identity(implementation.__file__)["sha256"] != receipt["torchvision_resnet_source"]["sha256"]:
        raise ValueError("Original native environment differs from the source capture")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model = torchvision.models.resnet50(weights=None).eval()
    helper.load_checkpoint(model, source["checkpoint"]["path"], torch)
    loader = load_module("two_operator_source_loader", receipt["loader_source"]["path"])
    state_before = loader._state_dict_sha256(model)
    if state_before != source["state_dict_sha256"]:
        raise ValueError("Original checkpoint parameters/buffers differ")
    images, _ = helper.load_images(source["input_stream"]["path"], np)
    image = torch.from_numpy(images[0].copy())
    if tensor_identity(image) != source["input"]:
        raise ValueError("Wrong original development image")
    block = model.layer1[0]
    original_capture, original_third = [], []

    def original_hook(_, inputs, output):
        original_capture.append((inputs[0].detach().clone(), output.detach().clone()))

    def original_third_hook(_, inputs, output):
        original_third.append((inputs[0].detach().clone(), output.detach().clone()))

    original_handles = [block.conv2.register_forward_hook(original_hook)]
    if capture_sink is not None or third_operator is not None:
        original_handles.append(block.conv3.register_forward_hook(original_third_hook))
    try:
        original = model(image).detach().clone()
    finally:
        for handle in original_handles:
            handle.remove()
    if (len(original_capture) != 1 or tensor_identity(original) != source["native_logits"]
            or not np.array_equal(original.numpy().view(np.uint32), original_saved_logits.view(np.uint32))
            or not np.array_equal(original_capture[0][1][0].flatten(1).t().contiguous().numpy().view(np.uint32), original_second.view(np.uint32))):
        raise AssertionError("Original full native reference no longer matches")
    replacements = [torch.from_numpy(value.copy()).t().reshape(1, 64, 56, 56).contiguous() for value in (first_target, second_target)]
    for value, matrix in zip(replacements, (first_target, second_target)):
        if not np.array_equal(value[0].flatten(1).t().contiguous().numpy().view(np.uint32), matrix.view(np.uint32)):
            raise AssertionError("Actual target output layout changed during NCHW reshape")
    if tensor_identity(replacements[0]) != chain["actual_conv1_output"]:
        raise ValueError("Admitted first output differs from the captured conv2 source chain")
    if third_operator is not None:
        third_replacement = torch.from_numpy(third_operator["target"].copy()).t().reshape(1, 256, 56, 56).contiguous()
        if not np.array_equal(third_replacement[0].flatten(1).t().contiguous().numpy().view(np.uint32), third_operator["target"].view(np.uint32)):
            raise ValueError("Actual conv3 output layout changed during NCHW reshape")
        if third_chain["actual_outputs"] != [tensor_identity(value) for value in replacements]:
            raise ValueError("Conv3 capture used different actual prefix outputs")
        if (len(original_third) != 1 or tensor_identity(original_third[0][0]) != third_chain["original_conv3_activation"]
                or tensor_identity(original_third[0][1]) != third_chain["original_conv3_native_output"]):
            raise ValueError("Original conv3 reference differs from the captured source")
        records += [require_identity(record) for record in third["source"]["immutable_bindings"]]
    if shortcut_operator is not None:
        shortcut_replacement = torch.from_numpy(shortcut_operator["target"].copy()).t().reshape(1, 256, 56, 56).contiguous()
        if not np.array_equal(shortcut_replacement[0].flatten(1).t().contiguous().numpy().view(np.uint32), shortcut_operator["target"].view(np.uint32)):
            raise ValueError("Actual shortcut output layout changed during NCHW reshape")
        if not np.array_equal(shortcut_operator["arrays"]["A"].view(np.uint32), first_arrays["A"].view(np.uint32)):
            raise ValueError("Shortcut operands do not share the original conv1 input")
        records += [require_identity(record) for record in shortcut["source"]["immutable_bindings"]]
    for module, arrays in ((block.conv1, first_arrays), (block.conv2, second_arrays)):
        weights = module.weight.flatten(1).t().contiguous().numpy()
        if not np.array_equal(weights.view(np.uint32), arrays["B"].view(np.uint32)):
            raise ValueError("Native convolution weights differ from admitted operands")
    current, traces, outputs, observations = {}, [], [], []
    expected_trace = ["actual_conv1", "native_bn1", "actual_conv2", "native_bn2", "native_conv3", "native_bn3", "native_skip", "native_residual"]
    if third_operator is not None:
        expected_trace[4] = "actual_conv3"
    if shortcut_operator is not None:
        expected_trace[6:6] = ["actual_shortcut_conv", "native_shortcut_bn"]

    def inject_first(_, inputs, output):
        a = inputs[0][0].flatten(1).t().contiguous().numpy()
        c = output[0].flatten(1).t().contiguous().numpy()
        if (not np.array_equal(a.view(np.uint32), first_arrays["A"].view(np.uint32))
                or not np.array_equal(c.view(np.uint32), first_arrays["native_C"].view(np.uint32))):
            raise ValueError("First callback is not the admitted original native invocation")
        current["trace"].append("actual_conv1")
        current["block_input"] = inputs[0].detach().clone()
        return replacements[0].clone()

    def observe_bn1(_, inputs, output):
        require_tensor(inputs[0], replacements[0], "BN1 did not receive actual conv1 output")
        if tensor_identity(output) != chain["native_bn1_output"]:
            raise ValueError("Native BN1 output differs from the captured candidate chain")
        current["trace"].append("native_bn1")

    def inject_second(_, inputs, output):
        if (tensor_identity(inputs[0]) != chain["conv2_activation"] or tensor_identity(output) != chain["conv2_native_output"]):
            raise ValueError("Conv2 callback is not the captured changed-prefix invocation")
        a = patch_rows(inputs[0].numpy(), 0, 3136, np)
        c = output[0].flatten(1).t().contiguous().numpy()
        if (not np.array_equal(a.view(np.uint32), second_arrays["A"].view(np.uint32))
                or not np.array_equal(c.view(np.uint32), second_arrays["native_C"].view(np.uint32))):
            raise ValueError("Conv2 callback operands/native output differ from the full admitted contraction")
        current["trace"].append("actual_conv2")
        return replacements[1].clone()

    def observe_bn2(_, inputs, output):
        require_tensor(inputs[0], replacements[1], "BN2 did not receive actual conv2 output")
        current["bn2"] = output.detach().clone()
        current["trace"].append("native_bn2")

    def observe_conv3(module, inputs, output):
        require_tensor(inputs[0], torch.relu(current["bn2"]), "Conv3 did not receive native ReLU(BN2(actual conv2))")
        if third_operator is not None:
            if (tensor_identity(current["bn2"]) != third_chain["native_bn2_output"]
                    or tensor_identity(inputs[0]) != third_chain["conv3_activation"]
                    or tensor_identity(output) != third_chain["conv3_native_output"]):
                raise ValueError("Conv3 callback is not the captured changed-prefix invocation")
            observed = {"A": inputs[0][0].flatten(1).t().contiguous().numpy(),
                        "B": module.weight.flatten(1).t().contiguous().numpy(),
                        "native_C": output[0].flatten(1).t().contiguous().numpy()}
            if any(not np.array_equal(value.view(np.uint32), third_operator["arrays"][name].view(np.uint32)) for name, value in observed.items()):
                raise ValueError("Conv3 callback operands/native output differ from the full admitted contraction")
            current["conv3"] = third_replacement.clone()
            current["trace"].append("actual_conv3")
            return third_replacement.clone()
        current["conv3"] = output.detach().clone()
        if capture_sink is not None:
            current["conv3_input"] = inputs[0].detach().clone()
            current["conv3_weight"] = module.weight.detach().clone()
        current["trace"].append("native_conv3")

    def observe_bn3(_, inputs, output):
        require_tensor(inputs[0], current["conv3"], "Native BN3 input changed")
        current["bn3"] = output.detach().clone()
        current["trace"].append("native_bn3")

    def observe_skip(_, inputs, output):
        require_tensor(inputs[0], current["block_input"], "Native residual skip input changed")
        if shortcut_operator is not None:
            require_tensor(output, current["shortcut_bn"], "Shortcut sequential output differs from native BN(actual shortcut)")
        current["skip"] = output.detach().clone()
        current["trace"].append("native_skip")

    def inject_shortcut(module, inputs, output):
        require_tensor(inputs[0], current["block_input"], "Shortcut convolution did not receive the original block input")
        observed = {"A": inputs[0][0].flatten(1).t().contiguous().numpy(),
                    "B": module.weight.flatten(1).t().contiguous().numpy(),
                    "native_C": output[0].flatten(1).t().contiguous().numpy()}
        if any(not np.array_equal(value.view(np.uint32), shortcut_operator["arrays"][name].view(np.uint32)) for name, value in observed.items()):
            raise ValueError("Shortcut original checkpoint operands/native output differ from the admitted source")
        current["trace"].append("actual_shortcut_conv")
        return shortcut_replacement.clone()

    def observe_shortcut_bn(_, inputs, output):
        require_tensor(inputs[0], shortcut_replacement, "Shortcut BN did not receive actual physical shortcut output")
        current["shortcut_bn"] = output.detach().clone()
        current["trace"].append("native_shortcut_bn")

    def observe_block(_, inputs, output):
        require_tensor(output, torch.relu(current["bn3"] + current["skip"]), "Bottleneck output is not native residual addition/ReLU")
        current["block_output"] = output.detach().clone()
        current["trace"].append("native_residual")

    handles = [module.register_forward_hook(callback) for module, callback in (
        (block.conv1, inject_first), (block.bn1, observe_bn1), (block.conv2, inject_second), (block.bn2, observe_bn2),
        (block.conv3, observe_conv3), (block.bn3, observe_bn3), (block.downsample, observe_skip), (block, observe_block))]
    if shortcut_operator is not None:
        handles += [block.downsample[0].register_forward_hook(inject_shortcut), block.downsample[1].register_forward_hook(observe_shortcut_bn)]
    try:
        for repeat in range(2):
            current.clear()
            current["trace"] = []
            outputs.append(model(image).detach().clone())
            if current["trace"] != expected_trace:
                raise AssertionError("Native candidate callbacks did not occur once in the required order")
            traces.append(list(current["trace"]))
            observations.append({key: tensor_identity(current[key]) for key in ("bn2", "conv3", "bn3", "skip", "block_output")})
    finally:
        for handle in handles:
            handle.remove()
    if (tensor_identity(outputs[0]) != tensor_identity(outputs[1]) or observations[0] != observations[1]
            or not all(torch.isfinite(value).all() for value in outputs)):
        raise AssertionError("Native candidate suffix/repeated terminal changed or became nonfinite")
    restored = model(image).detach().clone()
    if (tensor_identity(restored) != tensor_identity(original) or tensor_identity(image) != source["input"]
            or loader._state_dict_sha256(model) != state_before):
        raise AssertionError("Hook removal did not restore original terminal/input/state")
    if capture_sink is not None:
        if len(original_third) != 1:
            raise AssertionError("Original conv3 reference must be observed exactly once")
        require_tensor(block.conv3(current["conv3_input"]), current["conv3"], "Candidate native conv3 replay changed output")
        if loader._state_dict_sha256(model) != state_before:
            raise AssertionError("Conv3 replay changed native state")
        capture_sink.update(original=original_third[0], candidate=(current["conv3_input"], current["conv3"]),
                            weight=current["conv3_weight"], replay_raw_exact=True,
                            actual_outputs=[tensor_identity(value) for value in replacements])
    controls = [chain_control] + third_controls
    for name, call in (("original_conv2_input_instead_of_candidate", lambda: inject_second(block.conv2, (original_capture[0][0],), original_capture[0][1])),):
        try:
            call()
        except ValueError:
            controls.append({"control": name, "refused": True})
        else:
            raise AssertionError("Callback negative admitted: " + name)
    wrong_second = replacements[1].clone()
    wrong_second.view(torch.int32).flatten()[0] ^= 1
    try:
        observe_bn2(block.bn2, (wrong_second,), current["bn2"])
    except ValueError:
        controls.append({"control": "changed_actual_conv2_words_at_bn2_callback", "refused": True})
    else:
        raise AssertionError("BN2 admitted changed target words")
    if third_operator is not None:
        wrong_third = third_replacement.clone()
        wrong_third.view(torch.int32).flatten()[0] ^= 1
        for name, call in (
                ("original_conv3_input_instead_of_candidate", lambda: observe_conv3(block.conv3, (original_third[0][0],), original_third[0][1])),
                ("changed_actual_conv3_words_at_bn3", lambda: observe_bn3(block.bn3, (wrong_third,), current["bn3"]))):
            try:
                call()
            except ValueError:
                controls.append({"control": name, "refused": True})
            else:
                raise AssertionError("Conv3 callback negative admitted: " + name)
    if shortcut_operator is not None:
        wrong_shortcut = shortcut_replacement.clone()
        wrong_shortcut.view(torch.int32).flatten()[0] ^= 1
        try:
            observe_shortcut_bn(block.downsample[1], (wrong_shortcut,), current["shortcut_bn"])
        except ValueError:
            controls.append({"control": "changed_actual_shortcut_words_at_native_bn", "refused": True})
        else:
            raise AssertionError("Shortcut BN admitted changed physical output words")
    for record in records:
        require_identity(record)
    validator.recheck(second_records)
    if third_operator is not None:
        third_operator["validator"].recheck(third_operator["records"])
    if shortcut_operator is not None:
        shortcut_operator["validator"].recheck(shortcut_operator["records"])
    label = source["sample"]["label"]

    def classify(value):
        order = np.lexsort((np.arange(value.shape[1]), -value.numpy()[0]))[:5].tolist()
        return {"top1": order[0], "top5": order, "label": label, "top1_hit": order[0] == label, "top5_hit": label in order}

    delta = outputs[0].numpy().astype(np.float64) - original.numpy().astype(np.float64)
    norm = float(np.linalg.norm(original.numpy().astype(np.float64)))
    close = np.isclose(outputs[0].numpy(), original.numpy(), rtol=1e-4, atol=1e-4)
    output_name = "two_operator_hybrid_logits" if third_operator is None else "three_main_convolution_hybrid_logits"
    if shortcut_operator is not None:
        output_name = "four_target_contraction_hybrid_logits"
    np.savez(args.output / "outputs.npz", original_native_logits=original.numpy(), **{output_name: np.stack([value.numpy() for value in outputs])})
    report.update(status="completed_hybrid_diagnostic", source=source, original_native_logits=tensor_identity(original),
                  hybrid_logits=[tensor_identity(value) for value in outputs], native_classification=classify(original), hybrid_classification=classify(outputs[0]),
                  source_fidelity={"rtol": 1e-4, "atol": 1e-4, "passed": bool(close.all()), "failed_values": int(np.sum(~close)),
                                   "max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta * delta))),
                                   "relative_l2": float(np.linalg.norm(delta) / norm) if norm else None},
                  first_operator_admission=first_admission, second_operator_admission=second_details,
                  candidate_chain=chain, callback_traces=traces, native_suffix_observations=observations, negative_controls=controls,
                  checks={"original_full_reference_raw_exact": True, "both_actual_physical_outputs_admitted": True,
                          "conv2_uses_changed_conv1_prefix": True, "native_bn_relu_conv3_residual_checked": True,
                          "candidate_repeats_raw_exact": True, "original_terminal_restored_raw_exact": True, "source_input_state_bookends": True},
                  source_bookends=records, conv2_admission_bookends=second_records, outputs=identity(args.output / "outputs.npz"),
                  native_environment=versions, native_state_sha256=state_before, merlin_revision=git_head(repo_root()))
    if third_operator is not None:
        report["checks"].pop("both_actual_physical_outputs_admitted")
        report["checks"].pop("native_bn_relu_conv3_residual_checked")
        report["checks"].update(three_actual_physical_outputs_admitted=True, conv3_uses_changed_conv1_conv2_prefix=True,
                               native_bn3_shortcut_bn_residual_checked=True)
        report.update(third_operator_admission=third_operator["details"], conv3_candidate_chain=third_chain,
                      conv3_admission_bookends=third_operator["records"], native_shortcut_convolution=True,
                      whole_bottleneck_offloaded=False)
    if shortcut_operator is not None:
        report["checks"].pop("three_actual_physical_outputs_admitted")
        report["checks"].update(four_actual_physical_outputs_admitted=True, shortcut_uses_original_shared_block_input=True,
                               native_shortcut_bn_and_residual_order_checked=True)
        report.update(shortcut_operator_admission=shortcut_operator["details"], shortcut_source=shortcut["source"],
                      shortcut_admission_bookends=shortcut_operator["records"], native_shortcut_convolution=False,
                      target_contractions=4, native_normalization_relu_residual_suffix=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("conv1-hybrid-receipt", "conv2-native-bundle", "conv2-provider-receipt", "producer-root", "model-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("conv1-hybrid-receipt-sha256", "conv2-native-manifest-sha256", "conv2-provider-receipt-sha256", "validator-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    args.output = safe_path(args.output_dir)
    if args.output == safe_path(artifacts_dir()) or not args.output.is_relative_to(safe_path(artifacts_dir())):
        parser.error("Output must be a new directory beneath the configured artifact root")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"status": "failed", "claim": __doc__, "command": sys.argv, "cwd": str(Path.cwd()), "source_script": identity(__file__),
              "python_executable": sys.executable, "complete_atlas_execution": False, "timing_qualified": False, "application_quality_qualified": False,
              "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    code = 1
    try:
        evaluate(args, report)
        code = 0
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    report.update(wall_seconds=time.monotonic() - start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_manifest(args.output / "receipt.json", report)
    print(json.dumps({"status": report["status"], "receipt": str(args.output / "receipt.json"), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
