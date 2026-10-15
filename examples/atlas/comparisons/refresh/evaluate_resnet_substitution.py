"""Evaluate one actual model-result convolution inside unchanged native ResNet.

This is a hybrid diagnostic: all remaining operators execute on the native CPU.
It supplies neither complete Atlas execution nor a new quality/timing verdict.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from merlin.common.paths import artifacts_dir, repo_root
from capture_resnet import (RECEIPT_SHA256, identity, require_identity, safe_path,
                            load_module, load_native_admission, tensor_identity, git_head, write_manifest)


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


def negative_controls(native, provider, manifest_hash, npz_hash, shape, arrays, args, np, first_tile, first_payload, output_halves):
    def refuses(name, call):
        try:
            call()
        except ValueError:
            return {"control": name, "refused": True}
        raise AssertionError("Negative control admitted: " + name)

    link = lambda n, p: admit_linkage(n, p, manifest_hash, npz_hash, args.site, args.sample_index)
    controls = [refuses("wrong_site", lambda: admit_linkage(native, provider, manifest_hash, npz_hash, "wrong.site", args.sample_index))]
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
    changed["target_C"][row, column] = np.array([np.uint32(changed["actual_bits"][row, column]) << np.uint32(16)], dtype=np.uint32).view(np.float32)[0]
    admit_result_arrays(changed, shape, np)
    controls.append(refuses("coherently_changed_aggregate_against_unchanged_tile", lambda: admit_tile_binding(changed["actual_bits"], first_tile, first_payload, output_halves)))
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
    native, provider = json.loads(native_path.read_text()), json.loads(provider_path.read_text())
    arithmetic_id = require_identity({"path": str(args.provider_arithmetic), "sha256": provider["code_before"]["atlas_tile_arithmetic.py"]})
    arithmetic = load_module("hybrid_provider_output_format", arithmetic_id["path"])
    operands_id = require_identity({"path": str(native_dir / "operands.npz"), "sha256": native["npz_sha256"]})
    shape = admit_linkage(native, provider, native_id["sha256"], operands_id["sha256"], args.site, args.sample_index)
    records = [native_id, provider_id, operands_id, arithmetic_id, identity(__file__), identity(Path(__file__).with_name("capture_resnet.py"))]
    with np.load(operands_id["path"], allow_pickle=False) as archive:
        operands = {key: archive[key].copy() for key in archive.files}
    if set(operands) != {"A", "B", "native_C"}:
        raise ValueError("Unexpected native operand bundle fields")
    for name, expected_shape in zip(("A", "B", "native_C"), ((shape[0], shape[1]), (shape[1], shape[2]), (shape[0], shape[2]))):
        value = operands[name]
        descriptor = {"shape": list(value.shape), "dtype": str(value.dtype), "raw_sha256": hashlib.sha256(value.tobytes()).hexdigest()}
        if value.shape != expected_shape or value.dtype != np.float32 or not np.isfinite(value).all() or descriptor != native["tensors"][name]:
            raise ValueError("Native operand shape/bytes differ from captured source")
    for name in ("result.npz", "source.npz", "source-manifest.json"):
        records.append(require_identity({"path": str(provider_path.parent / name), "sha256": provider["artifacts"][name]}))
    if identity(provider_path.parent / "source-manifest.json")["sha256"] != native_id["sha256"]:
        raise ValueError("Provider retained another source manifest")
    with np.load(provider_path.parent / "source.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(operands) or any(not np.array_equal(archive[key].view(np.uint32), operands[key].view(np.uint32)) for key in operands):
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
        if any(execution[key] is not True for key in ("all_output_bytes_exact", "drained_before_halt_observation", "input_and_guards_preserved")):
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
    report.update(source_bundle=native_id, provider_receipt=provider_id, provider_result=identity(provider_path.parent / "result.npz"),
                  shape=list(shape), source_site=native["site"], source_sample=native["source"]["sample"],
                  provider_recipe=provider["recipe"], provider_execution_scope=provider["invocations"],
                  provider_output_formatter=arithmetic_id,
                  optional_raw_accumulator_role="Independent pre-rescale oracle diagnostic only; never substituted or treated as physical output" if "raw_accumulator_bits" in arrays else None,
                  negative_controls=negative_controls(native, provider, native_id["sha256"], operands_id["sha256"], shape, arrays, args, np, provider["tiles"][0], first_payload, arithmetic.output_halves))
    for record in records:
        require_identity(record)
    return native, operands, target, records


def evaluate(args, report):
    import numpy as np
    native, operands, target, records = load_results(args, report)
    if args.validate_only:
        report["status"] = "passed_bundle_admission_only"
        return
    source = native["source"]
    receipt_id = require_identity(source["receipt"])
    if receipt_id["sha256"] != RECEIPT_SHA256:
        raise ValueError("Another native development source receipt was selected")
    receipt = json.loads(Path(receipt_id["path"]).read_text())
    records += [require_identity(record) for record in source["immutable_bindings"]]
    records += [require_identity(record) for record in source["original_preparation_bindings"]]
    helper = load_native_admission(receipt, records, report)
    import torch
    import torchvision
    import torchvision.models.resnet as implementation
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__}
    if versions != source["versions"] or identity(implementation.__file__)["sha256"] != receipt["torchvision_resnet_source"]["sha256"]:
        raise ValueError("Imported native environment differs from the frozen capture")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model = torchvision.models.resnet50(weights=None).eval()
    helper.load_checkpoint(model, source["checkpoint"]["path"], torch)
    loader = load_module("hybrid_native_loader", receipt["loader_source"]["path"])
    state_before = loader._state_dict_sha256(model)
    if state_before != source["state_dict_sha256"]:
        raise ValueError("Native parameters/buffers differ from the captured checkpoint")
    images, _ = helper.load_images(source["input_stream"]["path"], np)
    image = torch.from_numpy(images[args.sample_index].copy())
    if tensor_identity(image) != source["input"]:
        raise ValueError("Wrong native input image")
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
    np.savez(args.output / "outputs.npz", native_logits=original.numpy(), hybrid_logits=np.stack([value.numpy() for value in hybrid]))
    report.update(status="completed_hybrid_diagnostic", native_classification=classify(original), hybrid_classification=classify(hybrid[0]),
                  source_fidelity={"rtol": 1e-4, "atol": 1e-4, "passed": bool(close.all()),
                                   "max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta * delta))),
                                   "relative_l2": float(np.linalg.norm(delta) / reference_norm) if reference_norm else None,
                                   "zero_reference": reference_norm == 0, "failed_values": int(np.sum(~close))},
                  checks={"native_terminal_raw_exact": True, "actual_target_words_admitted": True, "full_operator_coverage_verified": True,
                          "substitution_count_per_forward": 1, "hybrid_repeats_raw_exact": True, "native_after_hook_removal_raw_exact": True,
                          "source_input_state_bookends": True},
                  native_environment=versions, native_state_sha256=state_before, source_bookends=records,
                  native_logits=tensor_identity(original), hybrid_logits=[tensor_identity(value) for value in hybrid],
                  outputs=identity(args.output / "outputs.npz"), merlin_revision=git_head(repo_root()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bundle", required=True, type=Path)
    parser.add_argument("--native-manifest-sha256", required=True)
    parser.add_argument("--provider-receipt", required=True, type=Path)
    parser.add_argument("--provider-receipt-sha256", required=True)
    parser.add_argument("--provider-arithmetic", required=True, type=Path, help="Exact hash-bound OOT arithmetic/output formatter used by the provider receipt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--site", default="layer1.0.conv1")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    args.output = safe_path(args.output_dir)
    if args.output == safe_path(artifacts_dir()) or not args.output.is_relative_to(safe_path(artifacts_dir())):
        parser.error("Output must be a new directory beneath the configured artifact root")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"status": "failed", "claim": __doc__, "command": sys.argv, "cwd": str(Path.cwd()),
              "scope": "One frozen development image; one whole convolution uses actual prior model outputs; all remaining operators native CPU",
              "complete_atlas_execution": False, "timing_qualified": False, "application_quality_qualified": False,
              "python_executable": sys.executable, "source": identity(__file__),
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
