"""Capture native conv3 after actual conv1/conv2 model outputs and native BN/ReLU.

The unchanged two-operator source admission and native control path supplies the
boundary. Original conv3/logits remain separate from this candidate reference.
"""

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import traceback

from merlin.common.paths import artifacts_dir, repo_root
from capture_resnet import identity, require_identity, safe_path, tensor_identity, git_head, write_manifest
from evaluate_resnet_coupled import evaluate


def capture(args, report):
    import numpy as np
    import torch
    sink, hybrid = {}, {"status": "failed", "claim": "Two actual model-result convolutions inside unchanged native ResNet",
                       "complete_atlas_execution": False, "timing_qualified": False, "application_quality_qualified": False,
                       "command": sys.argv, "source_script": identity(Path(__file__).with_name("evaluate_resnet_coupled.py"))}
    evaluate(args, hybrid, capture_sink=sink)
    activation, convolution = sink["candidate"]
    original_activation, original_convolution = sink["original"]
    weight = sink["weight"]
    if (list(activation.shape) != [1, 64, 56, 56] or list(convolution.shape) != [1, 256, 56, 56]
            or list(weight.shape) != [256, 64, 1, 1] or not sink["replay_raw_exact"]
            or any(value.dtype != torch.float32 for value in (activation, convolution, weight))):
        raise ValueError("Unexpected candidate conv3 geometry/dtype/replay")
    stop_row, stop_channel = args.spatial_start + args.rows, args.output_channel_start + args.columns
    if (args.spatial_start < 0 or stop_row > 3136 or args.output_channel_start < 0 or stop_channel > 256):
        raise ValueError("Conv3 tile lies outside the native operation")
    a = activation[0].flatten(1).t()[args.spatial_start:stop_row].contiguous()
    b = weight[args.output_channel_start:stop_channel, :, 0, 0].t().contiguous()
    c = convolution[0].flatten(1).t()[args.spatial_start:stop_row, args.output_channel_start:stop_channel].contiguous()
    matrix = torch.mm(a, b)
    if any(not torch.isfinite(value).all() for value in (a, b, c, matrix)):
        raise ValueError("Nonfinite native candidate conv3 operands/output")
    # These source controls are additional to the actual chain/callback negatives
    # executed by the shared native evaluation path above.
    wrong = a.numpy().astype(np.float64) @ b.numpy()[::-1].astype(np.float64)
    controls = {"wrong_k_order_detected": not bool(np.allclose(wrong, c.numpy(), rtol=1e-4, atol=1e-4)),
                "original_later_site_input_distinguished": tensor_identity(activation) != tensor_identity(original_activation),
                "wrong_output_order_detected": not bool(np.array_equal(c.numpy().view(np.uint32), np.roll(c.numpy(), 1, axis=1).view(np.uint32)))}
    if not all(controls.values()):
        raise AssertionError("Candidate conv3 layout/source control failed: " + str(controls))
    source_records = hybrid["source_bookends"] + hybrid["conv2_admission_bookends"]
    source_records += [identity(__file__), identity(Path(__file__).with_name("evaluate_resnet_coupled.py"))]
    for record in source_records:
        require_identity(record)
    np.savez(args.output / "operands.npz", A=a.numpy(), B=b.numpy(), native_C=c.numpy())
    with np.load(args.output / "outputs.npz", allow_pickle=False) as archive:
        original_logits = archive["original_native_logits"].copy()
    np.savez(args.output / "original-reference.npz", native_logits=original_logits,
             A=original_activation[0].flatten(1).t().contiguous().numpy(),
             native_C=original_convolution[0].flatten(1).t().contiguous().numpy())
    write_manifest(args.output / "hybrid-receipt.json", hybrid)
    close = torch.isclose(matrix, c, rtol=1e-4, atol=1e-4)
    second_receipt = require_identity({"path": str(args.conv2_provider_receipt), "sha256": args.conv2_provider_receipt_sha256})
    second = json.loads(Path(second_receipt["path"]).read_text())
    second_result = require_identity({"path": str(Path(second_receipt["path"]).parent / "result.npz"), "sha256": second["artifacts"]["result.npz"]})
    report.update(status="passed", source_dtype="float32", npz_sha256=identity(args.output / "operands.npz")["sha256"],
                  source={**hybrid["source"], "immutable_bindings": source_records, "merlin_revision": git_head(repo_root()),
                          "activation_role": "Native BN2/ReLU after actual admitted conv1 and coupled conv2 model outputs",
                          "candidate_chain": {"conv1_admission": hybrid["first_operator_admission"],
                                              "conv2_native_manifest": {"path": str(safe_path(args.conv2_native_bundle) / "manifest.json"), "sha256": args.conv2_native_manifest_sha256},
                                              "conv2_provider_receipt": second_receipt, "conv2_provider_result": second_result,
                                              "conv2_admission": hybrid["second_operator_admission"], "actual_outputs": sink["actual_outputs"],
                                              "native_bn2_output": hybrid["native_suffix_observations"][0]["bn2"],
                                              "conv3_activation": tensor_identity(activation), "conv3_native_output": tensor_identity(convolution),
                                              "original_conv3_activation": tensor_identity(original_activation),
                                              "original_conv3_native_output": tensor_identity(original_convolution),
                                              "original_native_logits": hybrid["original_native_logits"], "two_operator_native_logits": hybrid["hybrid_logits"][0]}},
                  site={"module": "layer1.0.conv3", "boundary": "Native convolution before BN3/residual, on admitted two-operator candidate prefix",
                        "operation": "C=A@B", "activation_layout": "NCHW", "weight_layout": "OIHW",
                        "input_shape": list(activation.shape), "weight_shape": list(weight.shape), "output_shape": list(convolution.shape),
                        "spatial_flat_range": [args.spatial_start, stop_row], "output_channel_range": [args.output_channel_start, stop_channel],
                        "k_range": [0, 64], "complete_k": True, "matrix_shapes": {"A": list(a.shape), "B": list(b.shape), "native_C": list(c.shape)},
                        "mapping": "h,w=divmod(spatial_start+p,56); A[p,k]=candidate_X[0,k,h,w]; B[k,o]=W[output_start+o,k,0,0]; native_C[p,o]=actual_native_conv3(candidate_X)[0,output_start+o,h,w]"},
                  tensors={name: tensor_identity(value) for name, value in (("A", a), ("B", b), ("native_C", c))},
                  checks={**hybrid["checks"], "candidate_native_conv3_replay_raw_exact": True, "two_operator_callback_negatives": hybrid["negative_controls"]},
                  negative_controls=controls, original_reference=identity(args.output / "original-reference.npz"),
                  hybrid_receipt=identity(args.output / "hybrid-receipt.json"),
                  fp32_decomposition={"role": "Diagnostic only; actual native conv3 on candidate input retained unchanged", "rtol": 1e-4, "atol": 1e-4,
                                      "allclose": bool(close.all()), "failed_values": int((~close).sum()), "max_abs": float((matrix - c).abs().max())},
                  target_arithmetic_qualified=False, target_or_timing_qualified=False, application_quality_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("conv1-hybrid-receipt", "conv2-native-bundle", "conv2-provider-receipt", "producer-root", "model-root", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("conv1-hybrid-receipt-sha256", "conv2-native-manifest-sha256", "conv2-provider-receipt-sha256", "validator-sha256"):
        parser.add_argument("--" + name, required=True)
    for name, default in (("rows", 32), ("columns", 32), ("spatial-start", 0), ("output-channel-start", 0)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    args.output = safe_path(args.output_dir)
    if args.output == safe_path(artifacts_dir()) or not args.output.is_relative_to(safe_path(artifacts_dir())):
        parser.error("Output must be a fresh directory beneath the configured artifact root")
    if args.rows <= 0 or args.columns <= 1:
        parser.error("Rows must be positive and columns must exceed one for the output-order control")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {"schema_version": 1, "status": "failed", "scope": __doc__, "command": sys.argv, "cwd": str(Path.cwd()),
              "source_script": identity(__file__), "python_executable": sys.executable}
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
