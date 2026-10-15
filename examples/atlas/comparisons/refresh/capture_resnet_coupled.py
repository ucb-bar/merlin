"""Capture conv2 after actual conv1 model output and native BatchNorm/ReLU.

This is one frozen development image and a hybrid candidate prefix. Original
unhooked native results remain separate; no full-target or quality claim follows.
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
from capture_resnet_3x3 import patch_rows, require_same_patches, packing_controls
from evaluate_resnet_substitution import load_results


def require_tensor(actual, expected, message):
    if tensor_identity(actual) != tensor_identity(expected):
        raise ValueError(message)


def capture(args, report):
    import numpy as np
    # Reuse full physical-output admission, including producer/source hashes,
    # complete tile coverage, payload linkage and its actual negative controls.
    admission = {}
    native, operands, target, records = load_results(args, admission)
    source = native["source"]
    if args.site != "layer1.0.conv1" or native["site"]["module"] != args.site:
        raise ValueError("This bounded candidate chain starts at layer1.0.conv1")
    records += [identity(__file__), identity(Path(__file__).with_name("capture_resnet_3x3.py")),
                identity(Path(__file__).with_name("evaluate_resnet_substitution.py"))]
    receipt_id = require_identity(source["receipt"])
    if receipt_id["sha256"] != RECEIPT_SHA256:
        raise ValueError("Wrong frozen original development receipt")
    receipt = json.loads(Path(receipt_id["path"]).read_text())
    records += [require_identity(record) for record in source["immutable_bindings"]]
    records += [require_identity(record) for record in source["original_preparation_bindings"]]
    helper = load_native_admission(receipt, records, report)
    import torch
    import torchvision
    import torchvision.models.resnet as implementation
    versions = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__, "torchvision": torchvision.__version__}
    if versions != source["versions"] or identity(implementation.__file__)["sha256"] != receipt["torchvision_resnet_source"]["sha256"]:
        raise ValueError("Native environment differs from the original source capture")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    model = torchvision.models.resnet50(weights=None).eval()
    helper.load_checkpoint(model, source["checkpoint"]["path"], torch)
    loader = load_module("coupled_resnet_native_loader", receipt["loader_source"]["path"])
    state_before = loader._state_dict_sha256(model)
    if state_before != source["state_dict_sha256"]:
        raise ValueError("Native parameters/buffers differ from the original source")
    images, _ = helper.load_images(source["input_stream"]["path"], np)
    image = torch.from_numpy(images[args.sample_index].copy())
    if tensor_identity(image) != source["input"]:
        raise ValueError("Wrong frozen source image")
    block = model.layer1[0]
    conv1, bn1, conv2 = block.conv1, block.bn1, block.conv2
    if (type(conv1) is not torch.nn.Conv2d or type(bn1) is not torch.nn.BatchNorm2d
            or type(block.relu) is not torch.nn.ReLU or type(conv2) is not torch.nn.Conv2d
            or conv1.kernel_size != (1, 1) or conv1.stride != (1, 1) or conv1.bias is not None
            or conv2.kernel_size != (3, 3) or conv2.stride != (1, 1) or conv2.padding != (1, 1)
            or conv2.padding_mode != "zeros" or conv2.dilation != (1, 1) or conv2.groups != 1 or conv2.bias is not None):
        raise ValueError("Pinned native bottleneck boundary has unexpected operations")
    original_capture = []

    def original_hook(_, inputs, output):
        original_capture.append((inputs[0].detach().clone(), output.detach().clone()))

    handle = conv2.register_forward_hook(original_hook)
    try:
        original = model(image).detach().clone()
    finally:
        handle.remove()
    if len(original_capture) != 1 or tensor_identity(original) != source["native_logits"]:
        raise AssertionError("Unchanged original reference differs from the frozen source")
    height, width = native["site"]["input_shape"][2:]
    replacement = torch.from_numpy(target.copy()).t().reshape(1, target.shape[1], height, width).contiguous()
    if not np.array_equal(replacement[0].flatten(1).t().contiguous().numpy().view(np.uint32), target.view(np.uint32)):
        raise AssertionError("Actual target words changed during matrix-to-NCHW reshape")
    traces, captures, batchnorms, terminals = [], [], [], []
    current_trace, current_bn = [], []

    def inject(_, inputs, output):
        a = inputs[0][0].flatten(1).t().contiguous().numpy()
        c = output[0].flatten(1).t().contiguous().numpy()
        if (list(inputs[0].shape) != native["site"]["input_shape"] or output.shape != replacement.shape
                or not np.array_equal(a.view(np.uint32), operands["A"].view(np.uint32))
                or not np.array_equal(c.view(np.uint32), operands["native_C"].view(np.uint32))):
            raise ValueError("Conv1 callback no longer receives the admitted original invocation")
        current_trace.append("actual_conv1_output")
        return replacement.clone()

    def observe_bn(_, inputs, output):
        if current_trace != ["actual_conv1_output"]:
            raise ValueError("Native BatchNorm callback occurred outside the candidate chain")
        require_tensor(inputs[0], replacement, "Native BatchNorm input is not the actual target conv1 result")
        current_bn.append(output.detach().clone())
        current_trace.append("native_bn1")

    def observe_conv2(_, inputs, output):
        if current_trace != ["actual_conv1_output", "native_bn1"] or len(current_bn) != 1:
            raise ValueError("Conv2 callback occurred outside the admitted prefix")
        require_tensor(inputs[0], torch.relu(current_bn[0]), "Conv2 input is not native ReLU(native BN(actual conv1))")
        captures.append((inputs[0].detach().clone(), output.detach().clone()))
        batchnorms.append(current_bn[0])
        current_trace.append("native_relu_to_conv2")

    handles = [conv1.register_forward_hook(inject), bn1.register_forward_hook(observe_bn), conv2.register_forward_hook(observe_conv2)]
    try:
        for repeat in range(2):
            current_trace.clear()
            current_bn.clear()
            terminals.append(model(image).detach().clone())
            if current_trace != ["actual_conv1_output", "native_bn1", "native_relu_to_conv2"]:
                raise AssertionError("Candidate callbacks must occur exactly once and in order")
            traces.append(list(current_trace))
    finally:
        for handle in handles:
            handle.remove()
    if (len(captures) != 2 or tensor_identity(terminals[0]) != tensor_identity(terminals[1])
            or not all(torch.isfinite(value).all() for value in terminals)):
        raise AssertionError("Candidate capture count or repeated terminal changed")
    activation, convolution = captures[0]
    for left, right in zip(captures[0], captures[1]):
        require_tensor(left, right, "Candidate activation/native output changed across repeats")
    if tensor_identity(conv2(activation)) != tensor_identity(convolution):
        raise AssertionError("Native conv2 replay on candidate activation differs")
    restored = model(image).detach().clone()
    if (tensor_identity(restored) != tensor_identity(original) or tensor_identity(image) != source["input"]
            or loader._state_dict_sha256(model) != state_before):
        raise AssertionError("Original terminal/input/state not restored after hook removal")
    # Invoke the actual callbacks with wrong data, without running extra graphs.
    wrong_replacement = replacement.clone()
    wrong_replacement.view(torch.int32).flatten()[0] ^= 1
    current_trace[:] = ["actual_conv1_output"]
    current_bn.clear()
    try:
        observe_bn(bn1, (wrong_replacement,), batchnorms[0])
    except ValueError:
        wrong_bn_control = {"control": "changed_actual_conv1_words_at_bn_callback", "refused": True}
    else:
        raise AssertionError("BatchNorm callback admitted changed target output words")
    current_trace[:] = ["actual_conv1_output", "native_bn1"]
    current_bn[:] = [batchnorms[0]]
    try:
        observe_conv2(conv2, (original_capture[0][0],), original_capture[0][1])
    except ValueError:
        original_activation_control = {"control": "original_later_site_activation", "refused": True}
    else:
        raise AssertionError("Selected candidate did not distinguish the original later-site activation")
    _, channels, height, width = activation.shape
    end = args.output_channel_start + args.columns
    if args.output_channel_start < 0 or args.columns <= 0 or end > conv2.out_channels:
        raise ValueError("Requested output columns lie outside native conv2")
    a = patch_rows(activation.numpy(), args.spatial_start, args.rows, np)
    unfold = torch.nn.functional.unfold(activation, kernel_size=3, padding=1)[0].t().contiguous().numpy()
    require_same_patches(a, unfold[args.spatial_start:args.spatial_start + args.rows], np)
    controls = packing_controls(activation.numpy(), unfold[:32], 0, 32, np) + [wrong_bn_control, original_activation_control]
    b = conv2.weight[args.output_channel_start:end].flatten(1).t().contiguous().numpy()
    c = convolution[0].flatten(1).t()[args.spatial_start:args.spatial_start + args.rows, args.output_channel_start:end].contiguous().numpy()
    matrix = torch.mm(torch.from_numpy(a), torch.from_numpy(b)).numpy()
    if any(not np.isfinite(value).all() for value in (a, b, c, matrix)):
        raise ValueError("Nonfinite candidate/native operands or outputs")
    for record in records:
        require_identity(record)
    np.savez(args.output / "operands.npz", A=a, B=b, native_C=c)
    np.savez(args.output / "terminals.npz", original_native_logits=original.numpy(), candidate_prefix_native_logits=terminals[0].numpy())
    np.savez(args.output / "original-reference.npz", native_logits=original.numpy(),
             native_C=original_capture[0][1][0].flatten(1).t().contiguous().numpy())
    delta = matrix.astype(np.float64) - c.astype(np.float64)
    close = np.isclose(matrix, c, rtol=1e-4, atol=1e-4)
    report.update(status="passed", source_dtype="float32", npz_sha256=identity(args.output / "operands.npz")["sha256"],
                  source={**source, "source_capture": admission["source_bundle"], "immutable_bindings": records,
                          "merlin_revision": git_head(repo_root()), "activation_role": "Native BN/ReLU applied to actual prior target-model conv1 physical output",
                          "candidate_chain": {"sites": ["layer1.0.conv1", "layer1.0.bn1", "native ReLU", "layer1.0.conv2"],
                                              "provider_receipt": admission["provider_receipt"], "provider_result": admission["provider_result"],
                                              "provider_recipe": admission["provider_recipe"], "provider_output_formatter": admission["provider_output_formatter"],
                                              "actual_conv1_output": tensor_identity(replacement), "native_bn1_output": tensor_identity(batchnorms[0]),
                                              "conv2_activation": tensor_identity(activation), "conv2_native_output": tensor_identity(convolution),
                                              "original_conv2_activation": tensor_identity(original_capture[0][0]),
                                              "original_conv2_native_output": tensor_identity(original_capture[0][1]),
                                              "original_native_logits": tensor_identity(original), "candidate_prefix_native_logits": tensor_identity(terminals[0])}},
                  site={"module": "layer1.0.conv2", "boundary": "Native conv2 before BatchNorm, on the explicitly admitted candidate prefix",
                        "operation": "native_C=Conv2d(candidate_X,W); explicit_host_im2col(candidate_X)@flatten_native_weights(W)",
                        "activation_layout": "NCHW", "weight_layout": "OIHW", "input_shape": list(activation.shape), "weight_shape": list(conv2.weight.shape),
                        "output_shape": list(convolution.shape), "stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1,
                        "spatial_flat_range": [args.spatial_start, args.spatial_start + args.rows], "output_channel_range": [args.output_channel_start, end],
                        "k_range": [0, channels * 9], "complete_k": True,
                        "matrix_shapes": {"A": list(a.shape), "B": list(b.shape), "native_C": list(c.shape)},
                        "mapping": "q=(c*3+kh)*3+kw; h,w=divmod(spatial_start+p,W); A[p,q]=candidate_X[0,c,h+kh-1,w+kw-1] or zero outside image; B[q,o]=weight[output_start+o,c,kh,kw]; native_C[p,o]=actual_native_conv2(candidate_X)[0,output_start+o,h,w]"},
                  tensors={name: {"shape": list(value.shape), "dtype": str(value.dtype), "raw_sha256": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in (("A", a), ("B", b), ("native_C", c))},
                  checks={"original_native_reference_raw_exact": True, "actual_conv1_output_physically_admitted": True, "native_bn_relu_callback_chain_exact": True,
                          "callback_traces": traces, "candidate_repeats_raw_exact": True, "candidate_native_conv2_replay_raw_exact": True,
                          "original_terminal_restored_raw_exact": True, "source_input_state_bookends": True, "patches_vs_unfold_raw_exact": True,
                          "negative_controls": controls, "provider_admission_controls": admission["negative_controls"]},
                  fp32_decomposition={"role": "Diagnostic only; native conv2 on the candidate activation is retained unchanged", "rtol": 1e-4, "atol": 1e-4,
                                      "allclose": bool(close.all()), "failed_values": int(np.sum(~close)), "max_abs": float(np.abs(delta).max())},
                  terminals=identity(args.output / "terminals.npz"), original_reference=identity(args.output / "original-reference.npz"),
                  target_arithmetic_qualified=False, target_or_timing_qualified=False,
                  application_quality_qualified=False, complete_atlas_execution=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--native-manifest-sha256", required=True)
    parser.add_argument("--provider-receipt", type=Path, required=True)
    parser.add_argument("--provider-receipt-sha256", required=True)
    parser.add_argument("--provider-arithmetic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--site", default="layer1.0.conv1", choices=["layer1.0.conv1"])
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
    report = {"schema_version": 1, "status": "failed", "scope": __doc__, "command": sys.argv, "cwd": str(Path.cwd()), "source_script": identity(__file__),
              "python_executable": sys.executable, "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    start = time.monotonic()
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
