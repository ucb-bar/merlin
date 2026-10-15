"""Capture an unchanged native ResNet 3x3 convolution as explicit im2col operands.

Host patch extraction is workload preparation. The reference remains the actual
native convolution, whose FP32 accumulation can differ from a separate GEMM.
"""

import argparse
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


def patch_rows(activation, start, count, np, padding=1):
    """A[p,(c*3+kh)*3+kw], with explicit out-of-image zeros."""
    if activation.ndim != 4 or activation.shape[0] != 1:
        raise ValueError("Patch extraction requires one NCHW image")
    _, channels, height, width = activation.shape
    if start < 0 or count <= 0 or start + count > height * width:
        raise ValueError("Patch rows lie outside the native spatial output")
    positions = np.arange(start, start + count)
    h, w = positions // width, positions % width
    result = np.zeros((count, channels * 9), dtype=activation.dtype)
    for channel in range(channels):
        for kh in range(3):
            for kw in range(3):
                ih, iw = h + kh - padding, w + kw - padding
                valid = (ih >= 0) & (ih < height) & (iw >= 0) & (iw < width)
                result[valid, (channel * 3 + kh) * 3 + kw] = activation[0, channel, ih[valid], iw[valid]]
    return result


def require_same_patches(actual, expected, np):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
        raise ValueError("Explicit patch layout differs from independent native unfold")


def packing_controls(activation, expected, start, count, np):
    controls = []
    wrong_padding = patch_rows(activation, start, count, np, padding=0)
    correct = patch_rows(activation, start, count, np)
    channels = activation.shape[1]
    wrong_order = correct.reshape(count, channels, 3, 3).transpose(0, 2, 3, 1).reshape(count, channels * 9)
    for name, changed in (("missing_zero_padding", wrong_padding), ("kernel_major_instead_of_channel_major", wrong_order)):
        try:
            require_same_patches(changed.copy(), expected, np)
        except ValueError:
            controls.append({"control": name, "refused": True})
        else:
            raise AssertionError("Padding/layout negative control did not disagree: " + name)
    return controls


def capture(args, report):
    import numpy as np
    source_dir = safe_path(args.source_bundle)
    source_id = require_identity({"path": str(source_dir / "manifest.json"), "sha256": args.source_manifest_sha256})
    previous = json.loads(Path(source_id["path"]).read_text())
    if previous.get("status") != "passed" or previous.get("source_dtype") != "float32":
        raise ValueError("Require the original passing FP32 native source capture")
    source = previous["source"]
    if source["split"] != "development" or source["sample_index"] != args.sample_index:
        raise ValueError("Require the same frozen development image")
    receipt_id = require_identity(source["receipt"])
    if receipt_id["sha256"] != RECEIPT_SHA256:
        raise ValueError("Wrong frozen native source receipt")
    receipt = json.loads(Path(receipt_id["path"]).read_text())
    records = [source_id, receipt_id, identity(__file__), identity(Path(__file__).with_name("capture_resnet.py"))]
    records += [require_identity(record) for record in source["immutable_bindings"]]
    records += [require_identity(record) for record in source["original_preparation_bindings"]]
    helper = load_native_admission(receipt, records, report)
    import torch
    import torchvision
    import torchvision.models.resnet as implementation
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__}
    if versions != source["versions"] or identity(implementation.__file__)["sha256"] != receipt["torchvision_resnet_source"]["sha256"]:
        raise ValueError("Native source environment differs from the frozen capture")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model = torchvision.models.resnet50(weights=None).eval()
    helper.load_checkpoint(model, source["checkpoint"]["path"], torch)
    loader = load_module("resnet_3x3_native_loader", receipt["loader_source"]["path"])
    state_before = loader._state_dict_sha256(model)
    if state_before != source["state_dict_sha256"]:
        raise ValueError("Native state differs from the frozen checkpoint")
    images, _ = helper.load_images(source["input_stream"]["path"], np)
    image = torch.from_numpy(images[args.sample_index].copy())
    if tensor_identity(image) != source["input"]:
        raise ValueError("Native input differs from the frozen development image")
    module = model.get_submodule(args.site)
    if (type(module) is not torch.nn.Conv2d or module.kernel_size != (3, 3) or module.stride != (1, 1)
            or module.padding != (1, 1) or module.padding_mode != "zeros" or module.dilation != (1, 1) or module.groups != 1 or module.bias is not None):
        raise ValueError("Capture requires native bias-free stride-one zero-padded 3x3 convolution")
    original = model(image).detach().clone()
    if tensor_identity(original) != source["native_logits"]:
        raise AssertionError("Unchanged native terminal differs from the frozen capture")
    captured = []

    def hook(_, inputs, output):
        captured.append((inputs[0].detach().clone(), output.detach().clone()))

    handle = module.register_forward_hook(hook)
    try:
        instrumented = model(image).detach().clone()
    finally:
        handle.remove()
    if len(captured) != 1 or tensor_identity(instrumented) != tensor_identity(original):
        raise AssertionError("Capture hook changed native invocation count or terminal")
    activation, convolution = captured[0]
    if activation.dtype != torch.float32 or convolution.dtype != torch.float32:
        raise ValueError("Native convolution must retain original FP32 arithmetic")
    if tensor_identity(module(activation)) != tensor_identity(convolution):
        raise AssertionError("Native same-input convolution replay differs from captured output")
    repeated = model(image).detach().clone()
    if (tensor_identity(repeated) != tensor_identity(original) or tensor_identity(image) != source["input"]
            or loader._state_dict_sha256(model) != state_before):
        raise AssertionError("Native repeat/input/state preservation failed")
    _, channels, height, width = activation.shape
    end_column = args.output_channel_start + args.columns
    if (args.output_channel_start < 0 or args.columns <= 0 or end_column > module.out_channels
            or tuple(convolution.shape) != (1, module.out_channels, height, width)):
        raise ValueError("Requested output columns or native output shape are invalid")
    a = patch_rows(activation.numpy(), args.spatial_start, args.rows, np)
    native_unfold = torch.nn.functional.unfold(activation, kernel_size=3, padding=1)[0].t()[args.spatial_start:args.spatial_start + args.rows].contiguous().numpy()
    require_same_patches(a, native_unfold, np)
    columns = slice(args.output_channel_start, end_column)
    b = module.weight[columns].flatten(1).t().contiguous().numpy()
    c = convolution[0].flatten(1).t()[args.spatial_start:args.spatial_start + args.rows, columns].contiguous().numpy()
    matrix = torch.mm(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    if any(not np.isfinite(value).all() for value in (a, b, c, matrix)):
        raise ValueError("Native operands, output or decomposition contain nonfinite values")
    # Always exercise the actual top/left padding edge, independently of selected rows.
    edge_count = min(32, height * width)
    edge_unfold = torch.nn.functional.unfold(activation, kernel_size=3, padding=1)[0].t()[:edge_count].contiguous().numpy()
    controls = packing_controls(activation.numpy(), edge_unfold, 0, edge_count, np)
    for record in records:
        require_identity(record)
    np.savez(args.output / "operands.npz", A=a, B=b, native_C=c)
    delta = matrix.astype(np.float64) - c.astype(np.float64)
    close = np.isclose(matrix, c, rtol=1e-4, atol=1e-4)
    report.update(schema_version=1, status="passed", source_dtype="float32", npz_sha256=identity(args.output / "operands.npz")["sha256"],
                  source={**source, "source_capture": source_id, "immutable_bindings": records,
                          "merlin_revision": git_head(repo_root()),
                          "candidate_chain": None, "activation_role": "Unchanged native model prefix; no earlier target substitutions"},
                  site={"module": args.site, "boundary": "Native 3x3 convolution before BatchNorm; no bias, cast or folding",
                        "operation": "native_C=Conv2d(X,W); A=explicit_host_im2col(X), B=flatten_native_weights(W)",
                        "activation_layout": "NCHW", "weight_layout": "OIHW", "input_shape": list(activation.shape),
                        "weight_shape": list(module.weight.shape), "output_shape": list(convolution.shape),
                        "stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1,
                        "spatial_flat_range": [args.spatial_start, args.spatial_start + args.rows],
                        "output_channel_range": [args.output_channel_start, end_column], "k_range": [0, channels * 9], "complete_k": True,
                        "matrix_shapes": {"A": list(a.shape), "B": list(b.shape), "native_C": list(c.shape)},
                        "mapping": "p=spatial_start+row; h,w=divmod(p,W); q=(c*3+kh)*3+kw; A[row,q]=X[0,c,h+kh-1,w+kw-1] or zero outside image; B[q,o]=weight[output_start+o,c,kh,kw]; native_C[row,o]=actual_Y[0,output_start+o,h,w]"},
                  tensors={name: {"shape": list(value.shape), "dtype": str(value.dtype), "raw_sha256": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in (("A", a), ("B", b), ("native_C", c))},
                  checks={"native_terminal_raw_exact": True, "instrumented_terminal_raw_exact": True, "native_site_replay_raw_exact": True,
                          "repeat_after_hook_removal_raw_exact": True, "input_state_preserved": True, "source_bookends_unchanged": True,
                          "patches_vs_native_unfold_raw_exact": True, "negative_controls": controls, "cpu_threads": 1, "cpu_interop_threads": 1},
                  fp32_decomposition={"role": "Diagnostic only: separate GEMM accumulation may differ; actual native convolution output is retained unchanged",
                                      "rtol": 1e-4, "atol": 1e-4, "allclose": bool(close.all()), "failed_values": int(np.sum(~close)),
                                      "raw_exact": bool(np.array_equal(matrix.view(np.uint32), c.view(np.uint32))), "max_abs": float(np.abs(delta).max()),
                                      "rms": float(np.sqrt(np.mean(delta * delta)))},
                  target_arithmetic_qualified=False, target_or_timing_qualified=False, application_quality_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site", default="layer1.0.conv2")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--spatial-start", type=int, default=0)
    parser.add_argument("--output-channel-start", type=int, default=0)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--columns", type=int, default=32)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    args.output = safe_path(args.output_dir)
    if args.output == safe_path(artifacts_dir()) or not args.output.is_relative_to(safe_path(artifacts_dir())):
        parser.error("Output must be a new directory beneath the configured artifact root")
    if args.rows <= 0 or args.columns <= 0:
        parser.error("Rows and columns must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"schema_version": 1, "status": "failed", "scope": __doc__, "command": sys.argv, "cwd": str(Path.cwd()),
              "source_script": identity(__file__), "python_executable": sys.executable,
              "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    code = 1
    try:
        capture(args, report)
        code = 0
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    report.update(wall_seconds=time.monotonic() - start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_manifest(args.output / "manifest.json", report)
    print(json.dumps({"status": report["status"], "manifest": str(args.output / "manifest.json"), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
