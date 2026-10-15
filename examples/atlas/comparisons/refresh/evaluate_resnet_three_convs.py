"""Evaluate three actual main-path convolutions inside unchanged native ResNet.

Conv3 must belong to the admitted actual-conv1/conv2 candidate chain. BN/ReLU,
the downsample convolution/BN shortcut, residual addition and the remaining
graph execute natively. This semantic hybrid checks native convolutions before
replacing outputs; it establishes no runtime offload, quality, or timing claim.
"""

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from merlin.common.paths import artifacts_dir
from capture_resnet import identity, require_identity, safe_path, write_manifest
from evaluate_resnet_coupled import evaluate
from capture_tinyllama_native import load_provider_validator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    shortcut_args = [getattr(args, name) for name in ("shortcut_native_bundle", "shortcut_provider_receipt",
                     "shortcut_native_manifest_sha256", "shortcut_provider_receipt_sha256")]
    if any(value is not None for value in shortcut_args) and not all(value is not None for value in shortcut_args):
        parser.error("The optional shortcut requires both native/provider paths and both hashes")
    sys.dont_write_bytecode = True
    args.output = safe_path(args.output_dir)
    if args.output == safe_path(artifacts_dir()) or not args.output.is_relative_to(safe_path(artifacts_dir())):
        parser.error("Output must be a fresh directory beneath the configured artifact root")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    own_id = identity(__file__)
    report = {"status": "failed", "claim": __doc__, "command": sys.argv, "cwd": str(Path.cwd()), "source_script": own_id,
              "python_executable": sys.executable, "complete_atlas_execution": False, "timing_qualified": False,
              "application_quality_qualified": False, "whole_bottleneck_offloaded": False,
              "environment": {key: os.environ.get(key) for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    if args.shortcut_native_bundle is not None:
        report["claim"] = "Four actual target contractions in native ResNet surroundings; normalization, ReLU, residual and suffix stay native. No whole-bottleneck hardware offload, runtime, timing or application-quality qualification."
    code = 1
    try:
        producer = safe_path(args.producer_root)
        validator, validator_id = load_provider_validator(
            producer, args.validator_sha256, args.conv3_provider_receipt, args.conv3_provider_receipt_sha256)
        manifest, arrays, target, records, details = validator.admit_contraction(
            receipt_path=safe_path(args.conv3_provider_receipt), receipt_sha256=args.conv3_provider_receipt_sha256,
            native_bundle=safe_path(args.conv3_native_bundle), native_manifest_sha256=args.conv3_native_manifest_sha256,
            producer_root=producer, model_root=safe_path(args.model_root))
        shortcut = None
        if args.shortcut_native_bundle is not None:
            shortcut_manifest, shortcut_arrays, shortcut_target, shortcut_records, shortcut_details = validator.admit_contraction(
                receipt_path=safe_path(args.shortcut_provider_receipt), receipt_sha256=args.shortcut_provider_receipt_sha256,
                native_bundle=safe_path(args.shortcut_native_bundle), native_manifest_sha256=args.shortcut_native_manifest_sha256,
                producer_root=producer, model_root=safe_path(args.model_root))
            shortcut = {"manifest": shortcut_manifest, "arrays": shortcut_arrays, "target": shortcut_target,
                        "records": shortcut_records, "details": shortcut_details, "validator": validator}
        evaluate(args, report, third_operator={"manifest": manifest, "arrays": arrays, "target": target,
                 "records": records, "details": details, "validator": validator}, shortcut_operator=shortcut)
        require_identity(own_id)
        require_identity(validator_id)
        report["source_bookends"] += [own_id, validator_id]
        code = 0
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    report.update(wall_seconds=time.monotonic() - start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_manifest(args.output / "receipt.json", report)
    print(json.dumps({"status": report["status"], "receipt": str(args.output / "receipt.json"), "error": report.get("error")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
