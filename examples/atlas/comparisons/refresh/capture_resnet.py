"""Capture native FP32 ResNet contractions from the frozen development stream.

This workload edge preserves the original source. Target encodings, arithmetic,
launch support and performance evaluation belong to the companion provider.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from merlin.common.paths import artifacts_dir, repo_root

# This explicit example sibling owns shared source admission; no target code is imported.
sys.path.insert(0, str(Path(__file__).parent.parent))
from capture_tinyllama_native import identity, safe_path


RECEIPT = "out/build/baselines/tvm-gemmini/resnet50-residual-accumulator-v6-development20-finalsource-20261010/results.json"
RECEIPT_SHA256 = "7630b69889495a657a2396fcfe2cee0a909cd6f6b7aa92e5390095e47c1d9f8a"
# This identifies the historical verifier declared by the frozen receipt. It is
# separate from the current native helper source identity recorded at admission.
HISTORICAL_VERIFIER_SHA256 = "5ca2ca510969d9f8237183ae028d49bb9a8d1ba9c7fbfb8cff11e4ffbe8a406b"
PREPARATION_SHA256 = "8c0876d4092485b3e6686500c0dd2b833767861acfba75f452926ec3ceff1c85"


def require_identity(record):
    actual = identity(record["path"])
    if actual["sha256"] != record["sha256"]:
        raise ValueError("Source binding changed: " + actual["path"])
    return actual


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, safe_path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_native_admission(receipt, records, report):
    """Preserve historical receipt provenance and bind the current native helper."""
    historical = receipt["verifier_source"]
    if historical["sha256"] != HISTORICAL_VERIFIER_SHA256:
        raise ValueError("Frozen receipt declares a different historical verifier")
    helper_path = Path(__file__).with_name("resnet_native.py")
    helper_id = identity(helper_path)
    helper = load_module("atlas_resnet_native_admission", helper_path)
    require_identity(helper_id)
    records.append(helper_id)
    report["native_admission"] = {"historical_verifier_source": dict(historical),
                                "current_helper": helper_id}
    return helper


def git_head(path):
    return subprocess.check_output(["git", "-C", str(safe_path(path)), "rev-parse", "HEAD"], text=True).strip()


def tensor_identity(value):
    array = value.detach().cpu().contiguous().numpy()
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "raw_sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def write_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def capture(args, report):
    baseline = safe_path(args.baseline_root)
    receipt_path = baseline / RECEIPT
    receipt_id = require_identity({"path": str(receipt_path), "sha256": RECEIPT_SHA256})
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "passed" or receipt["evaluation_role"] != "development":
        raise ValueError("Expected the frozen passing development receipt")
    preparation_path = safe_path(receipt["artifacts"]["inputs"]["path"]).parent / "preparation.json"
    preparation_id = require_identity({"path": str(preparation_path), "sha256": PREPARATION_SHA256})
    preparation = json.loads(preparation_path.read_text())
    records = [receipt_id, preparation_id, identity(__file__)]
    records += [require_identity(receipt["artifacts"][name]) for name in ("checkpoint", "inputs", "labels")]
    records += [require_identity(receipt[name]) for name in ("loader_source", "torchvision_resnet_source")]
    helper = load_native_admission(receipt, records, report)
    import numpy as np
    import torch
    import torchvision
    import torchvision.models.resnet as resnet_source

    expected_versions = receipt["versions"]
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "torchvision": torchvision.__version__, "numpy": np.__version__}
    if any(versions[name] != expected_versions[name] for name in versions):
        raise ValueError("Native environment differs from the frozen source versions")
    if identity(resnet_source.__file__)["sha256"] != receipt["torchvision_resnet_source"]["sha256"]:
        raise ValueError("Imported torchvision implementation differs from the frozen source")
    prepared_bindings = {str(safe_path(record["path"])): record for record in preparation["source_bindings"]}
    for path in (torch.__file__, np.__file__, Path(torch.__file__).parent / "lib/libtorch_cpu.so"):
        record = prepared_bindings.get(str(safe_path(path)))
        if record is None:
            raise ValueError("Imported native dependency is outside the frozen preparation bindings: " + str(path))
        records.append(require_identity(record))
    if not 0 <= args.sample_index < len(receipt["images"]):
        raise ValueError("Sample index is outside the frozen development stream")
    sample = preparation["samples"]["evaluation"][args.sample_index]
    if sample["id"] != receipt["label_declarations"]["samples"][args.sample_index]["id"]:
        raise ValueError("Development sample identity differs between receipts")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    torch.manual_seed(194)
    model = torchvision.models.resnet50(weights=None).eval()
    helper.load_checkpoint(model, receipt["artifacts"]["checkpoint"]["path"], torch)
    loader = load_module("refresh_resnet_source_loader", receipt["loader_source"]["path"])
    state_before = loader._state_dict_sha256(model)
    if state_before != receipt["model"]["state_dict_sha256"]:
        raise ValueError("Loaded native model state differs from the frozen model")
    module = model.get_submodule(args.site)
    if (type(module) is not torch.nn.Conv2d or module.kernel_size != (1, 1)
            or module.stride != (1, 1) or module.padding != (0, 0)
            or module.dilation != (1, 1) or module.groups != 1 or module.bias is not None):
        raise ValueError("Capture requires a native bias-free stride-one 1x1 convolution")
    images, _ = helper.load_images(receipt["artifacts"]["inputs"]["path"], np)
    image = images[args.sample_index].copy()
    tagged = hashlib.sha256(str(image.dtype).encode() + str(image.shape).encode() + image.tobytes()).hexdigest()
    if tagged != receipt["images"][args.sample_index]["input_sha256"]:
        raise ValueError("Selected native input tensor differs from its frozen digest")
    native_path = receipt_path.parent / ("image_" + str(args.sample_index) + ".npz")
    saved_native_id = identity(native_path)
    records.append(saved_native_id)
    with np.load(native_path, allow_pickle=False) as archive:
        historical_logits = archive["torch"].copy()
        if not np.array_equal(archive["image"], image):
            raise ValueError("Historical native image differs from the selected stream input")
    tensor = torch.from_numpy(image.copy())
    input_before = tensor_identity(tensor)
    logits = model(tensor).detach().clone()
    if not np.array_equal(logits.numpy(), historical_logits):
        raise AssertionError("Native source differs from the currently saved native logits")
    captured = []

    def hook(_, inputs, output):
        captured.append((inputs[0].detach().clone(), output.detach().clone()))

    handle = module.register_forward_hook(hook)
    try:
        instrumented_logits = model(tensor).detach().clone()
    finally:
        handle.remove()
    if len(captured) != 1 or tensor_identity(logits) != tensor_identity(instrumented_logits):
        raise AssertionError("Hook changed the source terminal or site invocation count")
    activation, convolution = captured[0]
    replay = module(activation).detach().clone()
    if tensor_identity(replay) != tensor_identity(convolution):
        raise AssertionError("Same-input native convolution does not reproduce its captured output")
    repeated_logits = model(tensor).detach().clone()
    if tensor_identity(repeated_logits) != tensor_identity(logits):
        raise AssertionError("Native repeat after hook removal changed the terminal")
    if input_before != tensor_identity(tensor) or loader._state_dict_sha256(model) != state_before:
        raise AssertionError("Native capture changed input, parameter or buffer bytes")
    if activation.dtype != torch.float32 or convolution.dtype != torch.float32:
        raise ValueError("Native contraction must retain FP32 source dtype")
    channels, height, width = activation.shape[1:]
    row_stop = args.spatial_start + args.rows
    channel_stop = args.output_channel_start + args.columns
    if (activation.shape[0] != 1 or args.spatial_start < 0 or row_stop > height * width
            or args.output_channel_start < 0 or channel_stop > module.out_channels):
        raise ValueError("Requested tile lies outside the native contraction")
    rows = slice(args.spatial_start, row_stop)
    outputs = slice(args.output_channel_start, channel_stop)
    a = activation[0].flatten(1).t()[rows].contiguous()
    b = module.weight[outputs, :, 0, 0].t().contiguous()
    c = convolution[0].flatten(1).t()[rows, outputs].contiguous()
    # This independent indexing/reconstruction gate checks the matrix contract,
    # while exact source replay above preserves the actual native conv algorithm.
    matrix = torch.mm(a, b)
    if not torch.allclose(matrix, c, rtol=1e-4, atol=1e-4):
        raise AssertionError("Matrix decomposition fails the unchanged native floating gate")
    if any(not torch.isfinite(value).all() for value in (a, b, c)):
        raise ValueError("Nonfinite native contraction operands or outputs")
    for record in records:
        require_identity(record)
    for record in preparation["source_bindings"]:
        # Preserve all originally bound decode/transform/native library bytes.
        require_identity(record)
    np.savez(args.output / "operands.npz", A=a.numpy(), B=b.numpy(), native_C=c.numpy())
    report.update(status="passed", source_dtype="float32", npz_sha256=identity(args.output / "operands.npz")["sha256"],
                  site={"module": args.site, "boundary": "native convolution before BatchNorm; no bias, cast or folding",
                        "operation": "C=A@B", "activation_layout": "NCHW", "weight_layout": "OIHW",
                        "input_shape": list(activation.shape), "weight_shape": list(module.weight.shape),
                        "spatial_flat_range": [args.spatial_start, row_stop], "output_channel_range": [args.output_channel_start, channel_stop],
                        "k_range": [0, channels], "complete_k": True, "matrix_shapes": {"A": list(a.shape), "B": list(b.shape), "native_C": list(c.shape)},
                        "mapping": "A[p,c]=X[0,c,h,w], B[c,o]=W[o,c,0,0], native_C[p,o]=Y[0,o,h,w]; h,w=divmod(spatial_start+p,W)"},
                  source={"workload": "torchvision ResNet50 v1.5 IMAGENET1K_V2; modern pinned comparison, not original-paper reproduction",
                          "receipt": receipt_id, "preparation": preparation_id, "checkpoint": receipt["artifacts"]["checkpoint"],
                          "input_stream": receipt["artifacts"]["inputs"], "sample_index": args.sample_index, "sample": sample,
                          "split": "development", "selection": "frozen class-table/lexical development order; index selected before candidate execution",
                          "transform": preparation["transform"], "state_dict_sha256": state_before,
                          "native_logits": tensor_identity(logits), "input": input_before,
                          "saved_native_archive": saved_native_id,
                          "saved_native_archive_provenance": "Archive hash is bound by this capture, not authenticated by the original numerical receipt",
                          "immutable_bindings": records, "original_preparation_bindings": preparation["source_bindings"],
                          "versions": versions, "merlin_revision": git_head(repo_root()), "baseline_merlin_revision": git_head(baseline)},
                  checks={"saved_native_logits_raw_exact": True, "instrumented_terminal_raw_exact": True,
                          "native_site_replay_raw_exact": True, "repeat_after_hook_removal_raw_exact": True,
                          "input_state_preserved": True, "bindings_unchanged": True,
                          "matrix_decomposition_rtol": 1e-4, "matrix_decomposition_atol": 1e-4,
                          "matrix_decomposition_max_abs": float((matrix - c).abs().max()),
                          "cpu_threads": 1, "cpu_interop_threads": 1},
                  tensors={name: tensor_identity(value) for name, value in (("A", a), ("B", b), ("native_C", c))},
                  target_arithmetic_qualified=False, target_or_timing_qualified=False, application_quality_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--site", default="layer1.0.conv1")
    parser.add_argument("--sample-index", default=0, type=int)
    parser.add_argument("--spatial-start", default=0, type=int)
    parser.add_argument("--output-channel-start", default=0, type=int)
    parser.add_argument("--rows", default=32, type=int)
    parser.add_argument("--columns", default=32, type=int)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    args.output = safe_path(args.output_dir)
    if args.output == safe_path(artifacts_dir()) or not args.output.is_relative_to(safe_path(artifacts_dir())):
        parser.error("--output-dir must be a new directory beneath the configured artifacts root")
    if args.rows <= 0 or args.columns <= 0:
        parser.error("Tile dimensions must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"schema_version": 1, "status": "failed", "scope": __doc__, "invocation": sys.argv,
              "cwd": str(Path.cwd()), "python_executable": sys.executable,
              "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MERLIN_OUT_ROOT")}}
    code = 1
    try:
        capture(args, report)
        code = 0
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    report.update(wall_seconds=time.monotonic() - start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_manifest(args.output / "manifest.json", report)
    print(json.dumps({"status": report["status"], "manifest": str(args.output / "manifest.json"), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
