"""Replay admitted ResNet contraction prefixes through unchanged native continuation."""
import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
from merlin.common.paths import artifacts_dir, repo_root
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).parent.parent))
from capture_tinyllama_native import load_provider_validator
from capture_resnet import (RECEIPT_SHA256, identity, require_identity, safe_path,
    load_module, load_native_admission, tensor_identity, git_head, write_manifest,
    read_source_metadata, relocation_source_identity, source_relocator, patch_rows, require_tensor, restore_native_source)

def admit_linkage(native, provider, native_manifest_hash, npz_hash, site, sample):
    if (native.get("status") != "passed" or native.get("schema_version") != 1
            or native.get("source_dtype") != "float32"):
        raise ValueError("Require a passing native FP32 source capture")
    source, location = native["source"], native["site"]
    if location["module"] != site or source["sample_index"] != sample or source["split"] != "development":
        raise ValueError("Wrong native site, sample or development role")
    _, k, height, width = location["input_shape"]
    n = location["weight_shape"][0]
    if (location["input_shape"][0] != 1 or location["weight_shape"] != [n, k, 1, 1]
            or not location["complete_k"] or location["k_range"] != [0, k]
            or location["spatial_flat_range"] != [0, height * width]
            or location["output_channel_range"] != [0, n]):
        raise ValueError("Substitution requires the complete declared native convolution")
    if provider.get("status") != "passed_model_only" or provider["shape"] != [height * width, k, n]:
        raise ValueError("Wrong provider status or full contraction shape")
    expected = {"input": npz_hash, "manifest": native_manifest_hash}
    if (provider["source_inputs"] != expected or native["npz_sha256"] != npz_hash
            or not provider["source_bookends"]["unchanged"]
            or provider["source_bookends"]["input_before"] != expected
            or provider["source_bookends"]["input_after"] != expected
            or provider["code_before"] != provider["code_after"]):
        raise ValueError("Provider/source bundle identities or bookends disagree")
    return height * width, k, n

def admit_result_arrays(result, shape, np):
    m, _, n = shape
    required = {"target_C", "actual_bits", "expected_bits"}
    if not required <= result.keys() or result.keys() - required - {"raw_accumulator_bits"}:
        raise ValueError("Require actual and independently expected operator words")
    target, actual, expected = (result[key] for key in ("target_C", "actual_bits", "expected_bits"))
    if (target.shape != (m, n) or target.dtype != np.float32 or not np.isfinite(target).all()
            or any(value.shape != (m, n) or value.dtype != np.uint16 for value in (actual, expected))):
        raise ValueError("Wrong target or word-array shape/dtype/finiteness")
    if not np.array_equal(actual, expected):
        raise ValueError("Actual operator words differ from the independent oracle")
    if not np.array_equal(target.view(np.uint32), actual.astype(np.uint32) << 16):
        raise ValueError("Target float32 payload is not the exact actual BF16 word view")
    raw = result.get("raw_accumulator_bits")
    if raw is not None and (raw.shape != (m, n) or raw.dtype != np.uint16 or np.any(((raw >> 7) & 255) == 255)):
        raise ValueError("Optional pre-rescale oracle accumulator requires finite BF16 words of the full shape")
    return target

def admit_tile_binding(actual_bits, tile, payload, output_halves):
    r0, r1 = tile["row_range"]
    c0, c1 = tile["column_range"]
    if output_halves(actual_bits[r0:r1, c0:c1].tolist()) != payload:
        raise ValueError("Global actual words disagree with the authenticated physical tile output")

def negative_controls(native, provider, manifest_hash, npz_hash, shape, arrays, args, np, first_tile, first_payload,
    output_halves):
    def refuses(name, call):
        try:
            call()
        except ValueError:
            return {"control": name, "refused": True}
        raise AssertionError("Negative control admitted: " + name)

    link = lambda n, p: admit_linkage(n, p, manifest_hash, npz_hash, args.site, args.sample_index)
    controls = [refuses("wrong_site", lambda: admit_linkage(native, provider, manifest_hash, npz_hash, "wrong.site",
        args.sample_index))]
    wrong = copy.deepcopy(provider)
    wrong["source_inputs"]["input"] = "0" * 64
    controls.append(refuses("wrong_source_bundle", lambda: link(native, wrong)))
    for name in ("shape", "actual_word", "oracle_word", "target_view"):
        changed = {key: value.copy() for key, value in arrays.items()}
        if name == "shape":
            changed["target_C"] = changed["target_C"][:-1]
        elif name == "target_view":
            changed["target_C"].view(np.uint32)[0, 0] ^= np.uint32(1)
        else:
            changed["actual_bits" if name == "actual_word" else "expected_bits"][0, 0] ^= np.uint16(1)
        controls.append(refuses("wrong_" + name, lambda: admit_result_arrays(changed, shape, np)))
    changed = {key: value.copy() for key, value in arrays.items()}
    row, column = first_tile["row_range"][0], first_tile["column_range"][0]
    changed["actual_bits"][row, column] ^= np.uint16(1)
    changed["expected_bits"][row, column] = changed["actual_bits"][row, column]
    changed["target_C"][row, column] = np.array([np.uint32(changed["actual_bits"][row, column]) << np.uint32(16)],
        dtype=np.uint32).view(np.float32)[0]
    admit_result_arrays(changed, shape, np)
    controls.append(refuses("coherently_changed_aggregate_against_unchanged_tile", lambda: admit_tile_binding(changed[
        "actual_bits"], first_tile, first_payload, output_halves)))
    for name in ("shape", "dtype", "nonfinite", "unknown_field"):
        changed = {key: value.copy() for key, value in arrays.items()}
        changed["raw_accumulator_bits"] = arrays["actual_bits"].copy()
        if name == "shape":
            changed["raw_accumulator_bits"] = changed["raw_accumulator_bits"][:-1]
        elif name == "dtype":
            changed["raw_accumulator_bits"] = changed["raw_accumulator_bits"].astype(np.uint32)
        elif name == "nonfinite":
            changed["raw_accumulator_bits"][0, 0] = np.uint16(0x7f80)
        else:
            changed["unsupported_diagnostic"] = np.zeros(1)
        controls.append(refuses("optional_raw_" + name, lambda: admit_result_arrays(changed, shape, np)))
    return controls

def load_results(args, report):
    import numpy as np
    native_dir, provider_path = safe_path(args.native_bundle), safe_path(args.provider_receipt)
    native_path = native_dir / "manifest.json"
    native_id = require_identity({"path": str(native_path), "sha256": args.native_manifest_sha256})
    provider_id = require_identity({"path": str(provider_path), "sha256": args.provider_receipt_sha256})
    native = read_source_metadata(args, report, native_path, native_id["sha256"])
    provider = read_source_metadata(args, report, provider_path, provider_id["sha256"])
    arithmetic_id = require_identity({"path": str(args.provider_arithmetic), "sha256": provider["code_before"][
        "atlas_tile_arithmetic.py"]})
    arithmetic = load_module("hybrid_provider_output_format", arithmetic_id["path"])
    operands_id = require_identity({"path": str(native_dir / "operands.npz"), "sha256": native["npz_sha256"]})
    shape = admit_linkage(native, provider, native_id["sha256"], operands_id["sha256"], args.site, args.sample_index)
    records = [native_id, provider_id, operands_id, arithmetic_id, relocation_source_identity(), identity(__file__), identity(
        Path(__file__).with_name("capture_resnet.py"))]
    with np.load(operands_id["path"], allow_pickle=False) as archive:
        operands = {key: archive[key].copy() for key in archive.files}
    if set(operands) != {"A", "B", "native_C"}:
        raise ValueError("Unexpected native operand bundle fields")
    for name, expected_shape in zip(("A", "B", "native_C"), ((shape[0], shape[1]), (shape[1], shape[2]), (shape[0], shape[2]))):
        value = operands[name]
        descriptor = {"shape": list(value.shape), "dtype": str(value.dtype), "raw_sha256": hashlib.sha256(value.tobytes(
            )).hexdigest()}
        if value.shape != expected_shape or value.dtype != np.float32 or not np.isfinite(value).all() or descriptor != native[
            "tensors"][name]:
            raise ValueError("Native operand shape/bytes differ from captured source")
    for name in ("result.npz", "source.npz", "source-manifest.json"):
        records.append(require_identity({"path": str(provider_path.parent / name), "sha256": provider["artifacts"][name]}))
    if identity(provider_path.parent / "source-manifest.json")["sha256"] != native_id["sha256"]:
        raise ValueError("Provider retained another source manifest")
    with np.load(provider_path.parent / "source.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(operands) or any(not np.array_equal(archive[key].view(np.uint32), operands[key].view(
            np.uint32)) for key in operands):
            raise ValueError("Provider consumed another native operand payload")
    with np.load(provider_path.parent / "result.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    target = admit_result_arrays(arrays, shape, np)
    coverage = np.zeros((shape[0], shape[2]), dtype=np.bool_)
    for index, tile in enumerate(provider["tiles"]):
        r0, r1 = tile["row_range"]
        c0, c1 = tile["column_range"]
        if (not 0 <= r0 < r1 <= shape[0] or not 0 <= c0 < c1 <= shape[2]
                or tile["k_range"] != [0, shape[1]] or coverage[r0:r1, c0:c1].any()):
            raise ValueError("Provider tiles overlap or omit the declared complete reduction")
        execution = tile["execution"]
        if any(execution[key] is not True for key in ("all_output_bytes_exact", "drained_before_halt_observation",
            "input_and_guards_preserved")):
            raise ValueError("Provider tile lacks exact guarded completed execution")
        directory = provider_path.parent / f"tile-{index:04d}"
        for name, key in (("actual-bf16-halves-0.bin", "actual_sha256"), ("expected-bf16-halves.bin", "expected_sha256")):
            bound = require_identity({"path": str(directory / name), "sha256": tile[key]})
            if bound["bytes"] != (r1 - r0) * (c1 - c0) * 2:
                raise ValueError("Provider tile output size disagrees with its ranges")
            records.append(bound)
            if key == "actual_sha256":
                payload = Path(bound["path"]).read_bytes()
                admit_tile_binding(arrays["actual_bits"], tile, payload, arithmetic.output_halves)
                if index == 0:
                    first_payload = payload
        if tile["actual_sha256"] != tile["expected_sha256"]:
            raise ValueError("Provider actual/oracle output payload hashes disagree")
        coverage[r0:r1, c0:c1] = True
    if not coverage.all():
        raise ValueError("Provider did not execute the complete named convolution")
    report.update(source_bundle=native_id, provider_receipt=provider_id, provider_result=identity(
        provider_path.parent / "result.npz"),
                  shape=list(shape), source_site=native["site"], source_sample=native["source"]["sample"],
                  provider_recipe=provider["recipe"], provider_execution_scope=provider["invocations"],
                  provider_output_formatter=arithmetic_id,
                  optional_raw_accumulator_role=(
                      'Independent pre-rescale oracle diagnostic only; never substituted or treated as '
                      'physical output'
                  ) if "raw_accumulator_bits" in arrays else None,
                  negative_controls=negative_controls(native, provider, native_id["sha256"], operands_id["sha256"], shape, arrays,
        args, np, provider["tiles"][0], first_payload, arithmetic.output_halves))
    for record in records:
        require_identity(record)
    return native, operands, target, records

def evaluate_single(args, report):
    import numpy as np
    native, operands, target, records = load_results(args, report)
    if args.validate_only:
        report["status"] = "passed_bundle_admission_only"
        return
    source = native["source"]
    receipt_id = require_identity(source["receipt"])
    if receipt_id["sha256"] != RECEIPT_SHA256:
        raise ValueError("Another native development source receipt was selected")
    receipt = read_source_metadata(args, report, receipt_id["path"], receipt_id["sha256"])
    records += [require_identity(record) for record in source["immutable_bindings"]]
    records += [require_identity(record) for record in source["original_preparation_bindings"]]
    torch, model, loader, state_before, image, versions = restore_native_source(
        source, receipt, records, report, sample_index=args.sample_index)
    module = model.get_submodule(args.site)
    location = native["site"]
    if (type(module) is not torch.nn.Conv2d or module.kernel_size != (1, 1) or module.stride != (1, 1)
            or module.padding != (0, 0) or module.dilation != (1, 1) or module.groups != 1 or module.bias is not None
            or list(module.weight.shape) != location["weight_shape"]):
        raise ValueError("Native operation boundary differs from source capture")
    weight = module.weight[:, :, 0, 0].t().contiguous().numpy()
    if not np.array_equal(weight.view(np.uint32), operands["B"].view(np.uint32)):
        raise ValueError("Native weights differ from provider source operands")
    original = model(image).detach().clone()
    if tensor_identity(original) != source["native_logits"]:
        raise AssertionError("Unchanged native terminal no longer matches the frozen capture")
    _, _, height, width = location["input_shape"]
    replacement = torch.from_numpy(target.copy()).t().reshape(1, target.shape[1], height, width).contiguous()
    if not np.array_equal(replacement[0].flatten(1).t().contiguous().numpy().view(np.uint32), target.view(np.uint32)):
        raise AssertionError("Matrix-to-native output layout changed actual target words")
    invocations = []

    def substitute(_, inputs, output):
        activation = inputs[0][0].flatten(1).t().contiguous().numpy()
        current = output[0].flatten(1).t().contiguous().numpy()
        if (list(inputs[0].shape) != location["input_shape"] or output.shape != replacement.shape
                or not np.array_equal(activation.view(np.uint32), operands["A"].view(np.uint32))
                or not np.array_equal(current.view(np.uint32), operands["native_C"].view(np.uint32))):
            raise AssertionError("Wrong invocation/operands/native output at the substitution boundary")
        invocations.append(tensor_identity(output))
        return replacement.clone()

    hybrid = []
    handle = module.register_forward_hook(substitute)
    try:
        for repeat in range(2):
            before = len(invocations)
            hybrid.append(model(image).detach().clone())
            if len(invocations) != before + 1:
                raise AssertionError("Substitution must occur exactly once per source forward")
    finally:
        handle.remove()
    if tensor_identity(hybrid[0]) != tensor_identity(hybrid[1]):
        raise AssertionError("Hybrid logits changed across repeated forwards")
    restored = model(image).detach().clone()
    if (tensor_identity(restored) != tensor_identity(original) or tensor_identity(image) != source["input"]
            or loader._state_dict_sha256(model) != state_before):
        raise AssertionError("Hook removal, caller or native state preservation failed")
    for record in records:
        require_identity(record)
    label = source["sample"]["label"]

    def classify(value):
        order = np.lexsort((np.arange(value.shape[1]), -value.numpy()[0]))[:5].tolist()
        return {"top1": order[0], "top5": order, "label": label, "top1_hit": order[0] == label, "top5_hit": label in order}

    delta = hybrid[0].numpy().astype(np.float64) - original.numpy().astype(np.float64)
    reference_norm = float(np.linalg.norm(original.numpy().astype(np.float64)))
    close = np.isclose(hybrid[0].numpy(), original.numpy(), rtol=1e-4, atol=1e-4)
    np.savez(args.output / "outputs.npz", native_logits=original.numpy(), hybrid_logits=np.stack([value.numpy(
        ) for value in hybrid]))
    report.update(status="completed_hybrid_diagnostic", native_classification=classify(original), hybrid_classification=classify(
        hybrid[0]),
                  source_fidelity={"rtol": 1e-4, "atol": 1e-4, "passed": bool(close.all()),
                                   "max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta * delta))),
                                   "relative_l2": float(np.linalg.norm(delta) / reference_norm) if reference_norm else None,
                                   "zero_reference": reference_norm == 0, "failed_values": int(np.sum(~close))},
                  checks={"native_terminal_raw_exact": True, "actual_target_words_admitted": True,
        "full_operator_coverage_verified": True,
                          "substitution_count_per_forward": 1, "hybrid_repeats_raw_exact": True,
        "native_after_hook_removal_raw_exact": True,
                          "source_input_state_bookends": True},
                  native_environment=versions, native_state_sha256=state_before, source_bookends=records,
                  native_logits=tensor_identity(original), hybrid_logits=[tensor_identity(value) for value in hybrid],
                  outputs=identity(args.output / "outputs.npz"), merlin_revision=git_head(repo_root()))

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
            or third_chain["conv2_native_manifest"] != {"path": str(safe_path(args.conv2_native_bundle) / "manifest.json"),
        "sha256": args.conv2_native_manifest_sha256}
            or third_chain["conv2_provider_receipt"] != require_identity({"path": str(safe_path(args.conv2_provider_receipt)),
        "sha256": args.conv2_provider_receipt_sha256})
            or third_chain["conv2_provider_result"] != identity(safe_path(args.conv2_provider_receipt).parent / "result.npz")
            or third_chain["original_native_logits"] != first["source"]["native_logits"]
            or any(third["source"][key] != first["source"][key] for key in
                   ("checkpoint", "input_stream", "input", "state_dict_sha256", "versions", "sample_index", "sample", "split"))):
        raise ValueError("Conv3 source is not the complete admitted two-operator candidate chain")

def validate_shortcut_source(first, shortcut):
    site = shortcut["site"]
    if (shortcut["status"] != "passed" or shortcut["source_dtype"] != "float32"
            or site["module"] != "layer1.0.downsample.0" or site["input_shape"] != [1, 64, 56, 56]
            or site["weight_shape"] != [256, 64, 1, 1] or site["spatial_flat_range"] != [0, 3136]
            or site["output_channel_range"] != [0, 256] or site["k_range"] != [0, 64] or not site["complete_k"]
            or shortcut["tensors"]["A"] != first["tensors"]["A"]
            or any(shortcut["source"][key] != first["source"][key] for key in
                   ("checkpoint", "input_stream", "input", "state_dict_sha256", "versions", "sample_index", "sample", "split",
        "native_logits"))):
        raise ValueError("Shortcut source is not the complete original shared-input convolution")


def admit_provider_contraction(args, role):
    producer = safe_path(args.producer_root)
    receipt = getattr(args, role + "_provider_receipt")
    receipt_sha256 = getattr(args, role + "_provider_receipt_sha256")
    validator, validator_id = load_provider_validator(producer, args.validator_sha256, receipt, receipt_sha256)
    admitted = validator.admit_contraction(
        receipt_path=safe_path(receipt), receipt_sha256=receipt_sha256,
        native_bundle=safe_path(getattr(args, role + "_native_bundle")),
        native_manifest_sha256=getattr(args, role + "_native_manifest_sha256"),
        producer_root=producer, model_root=safe_path(args.model_root),
        metadata_path_mapper=source_relocator(args).path)
    return validator, validator_id, admitted


def evaluate_chain(args, report, *, capture_sink=None, third_operator=None, shortcut_operator=None):
    import numpy as np
    if capture_sink is not None and third_operator is not None:
        raise ValueError("Conv3 capture requires the unchanged two-operator native suffix")
    if shortcut_operator is not None and (third_operator is None or capture_sink is not None):
        raise ValueError("Shortcut substitution requires the three admitted main convolutions")
    third_controls = []
    historical_id = require_identity({"path": str(args.conv1_hybrid_receipt), "sha256": args.conv1_hybrid_receipt_sha256})
    historical = read_source_metadata(args, report, historical_id["path"], historical_id["sha256"])
    if historical["status"] != "completed_hybrid_diagnostic":
        raise ValueError("Require the explicitly pinned prior one-operator diagnostic")
    first_admission = {}
    first, first_arrays, first_target, records = load_results(SimpleNamespace(
        native_bundle=Path(historical["source_bundle"]["path"]).parent,
        native_manifest_sha256=historical["source_bundle"]["sha256"],
        provider_receipt=Path(historical["provider_receipt"]["path"]),
        provider_receipt_sha256=historical["provider_receipt"]["sha256"],
        provider_arithmetic=Path(historical["provider_output_formatter"]["path"]),
        site="layer1.0.conv1", sample_index=0, relocate=getattr(args, "relocate", ()),
        _source_relocation=source_relocator(args)), first_admission)
    producer = safe_path(args.producer_root)
    validator, validator_id, admitted = admit_provider_contraction(args, "conv2")
    second, second_arrays, second_target, second_records, second_details = admitted
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
    records += [historical_id, validator_id, identity(__file__), identity(Path(__file__).with_name("capture_resnet.py")),
                identity(Path(__file__).with_name("evaluate_resnet.py"))]
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
    receipt = read_source_metadata(args, report, receipt_id["path"], receipt_id["sha256"])
    torch, model, loader, state_before, image, versions = restore_native_source(
        source, receipt, records, report, sample_index=0)
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
            or not np.array_equal(original_capture[0][1][0].flatten(1).t().contiguous().numpy().view(np.uint32),
        original_second.view(np.uint32))):
        raise AssertionError("Original full native reference no longer matches")
    replacements = [torch.from_numpy(value.copy()).t().reshape(1, 64, 56, 56).contiguous() for value in (first_target,
        second_target)]
    for value, matrix in zip(replacements, (first_target, second_target)):
        if not np.array_equal(value[0].flatten(1).t().contiguous().numpy().view(np.uint32), matrix.view(np.uint32)):
            raise AssertionError("Actual target output layout changed during NCHW reshape")
    if tensor_identity(replacements[0]) != chain["actual_conv1_output"]:
        raise ValueError("Admitted first output differs from the captured conv2 source chain")
    if third_operator is not None:
        third_replacement = torch.from_numpy(third_operator["target"].copy()).t().reshape(1, 256, 56, 56).contiguous()
        if not np.array_equal(third_replacement[0].flatten(1).t().contiguous().numpy().view(np.uint32), third_operator[
            "target"].view(np.uint32)):
            raise ValueError("Actual conv3 output layout changed during NCHW reshape")
        if third_chain["actual_outputs"] != [tensor_identity(value) for value in replacements]:
            raise ValueError("Conv3 capture used different actual prefix outputs")
        if (len(original_third) != 1 or tensor_identity(original_third[0][0]) != third_chain["original_conv3_activation"]
                or tensor_identity(original_third[0][1]) != third_chain["original_conv3_native_output"]):
            raise ValueError("Original conv3 reference differs from the captured source")
        records += [require_identity(record) for record in third["source"]["immutable_bindings"]]
    if shortcut_operator is not None:
        shortcut_replacement = torch.from_numpy(shortcut_operator["target"].copy()).t().reshape(1, 256, 56, 56).contiguous()
        if not np.array_equal(shortcut_replacement[0].flatten(1).t().contiguous().numpy().view(np.uint32), shortcut_operator[
            "target"].view(np.uint32)):
            raise ValueError("Actual shortcut output layout changed during NCHW reshape")
        if not np.array_equal(shortcut_operator["arrays"]["A"].view(np.uint32), first_arrays["A"].view(np.uint32)):
            raise ValueError("Shortcut operands do not share the original conv1 input")
        records += [require_identity(record) for record in shortcut["source"]["immutable_bindings"]]
    for module, arrays in ((block.conv1, first_arrays), (block.conv2, second_arrays)):
        weights = module.weight.flatten(1).t().contiguous().numpy()
        if not np.array_equal(weights.view(np.uint32), arrays["B"].view(np.uint32)):
            raise ValueError("Native convolution weights differ from admitted operands")
    current, traces, outputs, observations = {}, [], [], []
    expected_trace = ["actual_conv1", "native_bn1", "actual_conv2", "native_bn2", "native_conv3", "native_bn3", "native_skip",
        "native_residual"]
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
            if any(not np.array_equal(value.view(np.uint32), third_operator["arrays"][name].view(np.uint32)) for name,
                value in observed.items()):
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
        if any(not np.array_equal(value.view(np.uint32), shortcut_operator["arrays"][name].view(np.uint32)) for name,
            value in observed.items()):
            raise ValueError("Shortcut original checkpoint operands/native output differ from the admitted source")
        current["trace"].append("actual_shortcut_conv")
        return shortcut_replacement.clone()

    def observe_shortcut_bn(_, inputs, output):
        require_tensor(inputs[0], shortcut_replacement, "Shortcut BN did not receive actual physical shortcut output")
        current["shortcut_bn"] = output.detach().clone()
        current["trace"].append("native_shortcut_bn")

    def observe_block(_, inputs, output):
        require_tensor(output, torch.relu(current["bn3"] + current["skip"]),
            "Bottleneck output is not native residual addition/ReLU")
        current["block_output"] = output.detach().clone()
        current["trace"].append("native_residual")

    handles = [module.register_forward_hook(callback) for module, callback in (
        (block.conv1, inject_first), (block.bn1, observe_bn1), (block.conv2, inject_second), (block.bn2, observe_bn2),
        (block.conv3, observe_conv3), (block.bn3, observe_bn3), (block.downsample, observe_skip), (block, observe_block))]
    if shortcut_operator is not None:
        handles += [block.downsample[0].register_forward_hook(inject_shortcut), block.downsample[1].register_forward_hook(
            observe_shortcut_bn)]
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
    for name, call in (("original_conv2_input_instead_of_candidate", lambda: inject_second(block.conv2, (original_capture[0][0],),
        original_capture[0][1])),):
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
                ("original_conv3_input_instead_of_candidate", lambda: observe_conv3(block.conv3, (original_third[0][0],),
            original_third[0][1])),
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
    np.savez(args.output / "outputs.npz", original_native_logits=original.numpy(), **{output_name: np.stack([value.numpy(
        ) for value in outputs])})
    report.update(status="completed_hybrid_diagnostic", source=source, original_native_logits=tensor_identity(original),
                  hybrid_logits=[tensor_identity(value) for value in outputs], native_classification=classify(original),
        hybrid_classification=classify(outputs[0]),
                  source_fidelity={"rtol": 1e-4, "atol": 1e-4, "passed": bool(close.all()), "failed_values": int(np.sum(~close)),
                                   "max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta * delta))),
                                   "relative_l2": float(np.linalg.norm(delta) / norm) if norm else None},
                  first_operator_admission=first_admission, second_operator_admission=second_details,
                  candidate_chain=chain, callback_traces=traces, native_suffix_observations=observations,
        negative_controls=controls,
                  checks={"original_full_reference_raw_exact": True, "both_actual_physical_outputs_admitted": True,
                          "conv2_uses_changed_conv1_prefix": True, "native_bn_relu_conv3_residual_checked": True,
                          "candidate_repeats_raw_exact": True, "original_terminal_restored_raw_exact": True,
        "source_input_state_bookends": True},
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

def evaluate_three(args, report):
    own_id = identity(__file__)
    producer = safe_path(args.producer_root)
    validator, validator_id, admitted = admit_provider_contraction(args, "conv3")
    manifest, arrays, target, records, details = admitted
    shortcut = None
    if args.shortcut_native_bundle is not None:
        shortcut_manifest, shortcut_arrays, shortcut_target, shortcut_records, shortcut_details = validator.admit_contraction(
            receipt_path=safe_path(args.shortcut_provider_receipt), receipt_sha256=args.shortcut_provider_receipt_sha256,
            native_bundle=safe_path(args.shortcut_native_bundle), native_manifest_sha256=args.shortcut_native_manifest_sha256,
            producer_root=producer, model_root=safe_path(args.model_root),
            metadata_path_mapper=source_relocator(args).path)
        shortcut = {"manifest": shortcut_manifest, "arrays": shortcut_arrays, "target": shortcut_target,
                    "records": shortcut_records, "details": shortcut_details, "validator": validator}
    evaluate_chain(args, report, third_operator={"manifest": manifest, "arrays": arrays, "target": target,
             "records": records, "details": details, "validator": validator}, shortcut_operator=shortcut)
    require_identity(own_id)
    require_identity(validator_id)
    report["source_bookends"] += [own_id, validator_id]

def main():
    root = argparse.ArgumentParser(description=__doc__)
    modes = root.add_subparsers(dest="mode", required=True)
    parser = modes.add_parser("single")
    parser.add_argument("--native-bundle", required=True, type=Path)
    parser.add_argument("--native-manifest-sha256", required=True)
    parser.add_argument("--provider-receipt", required=True, type=Path)
    parser.add_argument("--provider-receipt-sha256", required=True)
    parser.add_argument("--provider-arithmetic", required=True, type=Path,
        help="Exact hash-bound OOT arithmetic/output formatter used by the provider receipt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--site", default="layer1.0.conv1")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW")
    parser = modes.add_parser("coupled")
    for name in ("conv1-hybrid-receipt", "conv2-native-bundle", "conv2-provider-receipt", "producer-root", "model-root",
        "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("conv1-hybrid-receipt-sha256", "conv2-native-manifest-sha256", "conv2-provider-receipt-sha256",
        "validator-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW")
    parser = modes.add_parser("three")
    for name in ("conv1-hybrid-receipt", "conv2-native-bundle", "conv2-provider-receipt", "conv3-native-bundle",
                 "conv3-provider-receipt", "producer-root", "model-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("conv1-hybrid-receipt-sha256", "conv2-native-manifest-sha256", "conv2-provider-receipt-sha256",
                 "conv3-native-manifest-sha256", "conv3-provider-receipt-sha256", "validator-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("shortcut-native-bundle", "shortcut-provider-receipt"):
        parser.add_argument("--" + name, type=Path)
    for name in ("shortcut-native-manifest-sha256", "shortcut-provider-receipt-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--relocate", action="append", default=[], metavar="OLD=NEW")
    args = root.parse_args()
    if args.mode == "three":
        shortcut = [getattr(args, name) for name in ("shortcut_native_bundle", "shortcut_provider_receipt",
                    "shortcut_native_manifest_sha256", "shortcut_provider_receipt_sha256")]
        if any(value is not None for value in shortcut) and not all(value is not None for value in shortcut):
            root.error("The optional shortcut requires both paths and both hashes")
    from workload_cli import run_report
    callbacks = {"single": evaluate_single, "coupled": evaluate_chain, "three": evaluate_three}
    return run_report(args, callbacks[args.mode], filename="receipt.json", claim=args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
