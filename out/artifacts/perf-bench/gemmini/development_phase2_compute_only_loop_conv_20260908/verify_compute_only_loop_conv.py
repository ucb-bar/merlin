#!/usr/bin/env python3
"""Public, payload-free verifier for the compute-only LOOP_CONV checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SKIP = {"build", "__pycache__", ".git"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_tree(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or SKIP.intersection(path.parts):
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
        files += 1
    return digest.hexdigest(), files


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise SystemExit(detail)


def verify_private_candidate(candidate: Path, receipt: dict) -> None:
    """Verify non-public model payloads when the private deployment bundle is present."""
    payload = receipt["payload"]
    require(sha256(candidate / "payload/const_blob.bin") == payload["const_blob_sha256"],
            "candidate constant blob changed")
    model = receipt["whole_model_spike"]
    elf = candidate / "resnet50_merlin_compute_only_loop_conv_w8a8_warm_measured.elf"
    require(sha256(elf) == model["elf_sha256"], "final clean ELF changed")
    require(sha256(candidate / "compiler/kernel.o") ==
            model["measured_kernel_object_sha256"], "candidate measured kernel changed")
    require(sha256(candidate / "compiler/kernel_warm_progress.o") ==
            model["warm_progress_kernel_object_sha256"], "candidate warm kernel changed")
    full_log_path = candidate / "validation/compute_only_final_clean.log"
    require(sha256(full_log_path) == model["log_sha256"], "final Spike log changed")
    full_log = full_log_path.read_text()
    for witness in (
        "MERLIN_METRIC cycles=496240002",
        "MERLIN_METRIC instret=496240008",
        "MERLIN_RESULT checksum_fnv1a64=c6e777c3fe0aae90",
        "MERLIN_RESULT logits_checked=1000 bad=0 nonfinite=0 top1=258 expected_top1=258",
        "PASS: warm-then-measured Merlin W8A8 ResNet-50 and all-logit check",
    ):
        require(witness in full_log, f"missing final Spike witness: {witness}")
    require(full_log.count("MERLIN_WARM_PROGRESS completed_task=") == 8,
            "warm progress is incomplete")
    require("MERLIN_STEM" not in full_log, "diagnostic stem dump leaked into final log")
    readiness = json.loads(
        (candidate / "validation/firesim_submission_readiness.json").read_text())
    require(readiness["safe_to_submit"] is False and readiness["firesim_run"] is True and
            readiness["independent_audit"] == "passed" and
            readiness["status"] == "hardware_tested_rejected_perf",
            "candidate audit/readiness state changed")
    hardware = json.loads(
        (candidate / "validation/firesim_queue_job_545_rejected_perf.json").read_text())
    require(hardware["status"] == "hardware_tested_rejected_perf" and
            hardware["promoted"] is False and hardware["resubmission_blocked"] is True,
            "q545 disposition changed")
    require(hardware["queue_job"]["state"] == "DONE" and
            hardware["queue_job"]["return_code"] == 0,
            "q545 did not complete successfully")
    require(hardware["measured_metrics"]["cycles"] == 1383906735 and
            hardware["q535_comparison"]["additional_cycles"] == 67287036,
            "q545 performance result changed")
    require(hardware["correctness"]["logits_checked"] == 1000 and
            hardware["correctness"]["bad"] == 0 and
            hardware["correctness"]["top1"] == 258,
            "q545 correctness changed")
    evidence = hardware["evidence"]
    for relative, key in (
        ("validation/firesim_queue_job_545_runworkload_full.json",
         "runworkload_full_json_sha256"),
        ("validation/firesim_queue_job_545_stdout.log", "stdout_sha256"),
        ("validation/firesim_queue_job_545_stderr.log", "stderr_sha256"),
        ("validation/firesim_queue_job_545_uart.log", "uart_sha256"),
        ("validation/firesim_queue_job_545_config_runtime.yaml", "config_runtime_sha256"),
    ):
        require(sha256(candidate / relative) == evidence[key],
                f"q545 evidence changed: {relative}")
    authority = readiness["hardware_authority"]
    for relative, key in (
        ("firesim.tar.gz", "firesim_tar_sha256"),
        ("runtime/config_hwdb.yaml", "config_hwdb_sha256"),
        ("queue_launcher/firesim", "queue_firesim_sha256"),
        ("queue_launcher/make", "queue_make_sha256"),
    ):
        require(sha256(candidate / relative) == authority[key],
                f"inherited hardware authority changed: {relative}")
    for line in (candidate / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ./", 1)
        require(sha256(candidate / relative) == expected,
                f"candidate manifest mismatch: {relative}")
    abi = json.loads((candidate / "payload/abi_layout.json").read_text())
    require(abi["kernel_arg_count"] == 393 and
            abi["const_blob_bytes"] == payload["const_blob_bytes"],
            "candidate ABI/blob extent changed")
    arg162 = next(row for row in abi["args"] if row["tensor"] == "arg162")
    require(arg162["prepack_recipe"]["source_layout"] == "OIHW" and
            arg162["prepack_recipe"]["packed_layout"] == "HWIO_dim_padded" and
            arg162["prepack_recipe"]["permutation"] == [2, 3, 1, 0],
            "stem weight prepack contract changed")
    harness = (candidate / "payload/single_run_harness_warm_progress.c").read_text()
    require(harness.count("  run_model_warm_progress();") == 1 and
            harness.count("  run_model();") == 1,
            "warm/measured call path changed")
    begin = harness.index("const uint64_t cycle_start = read_cycles();")
    measured = harness.index("  run_model();", begin)
    end = harness.index("const uint64_t cycle_end = read_cycles();", measured)
    require(begin < measured < end, "measured invocation escaped cycle interval")
    prepare = (candidate / "prepare_queue_chipyard_view.sh").read_text()
    require("resnet50_merlin_compute_only_loop_conv_w8a8_warm_measured.elf" in prepare,
            "queue preparation targets the wrong ELF")
    require("affine-im2col" not in prepare and "affine_im2col" not in prepare,
            "stale q535 queue identity remains")


def main() -> int:
    receipt = json.loads(
        (ROOT / "validation/compute_only_loop_conv_receipt.json").read_text())
    require(receipt["status"] == "hardware_tested_rejected_perf",
            "unexpected qualification status")
    require(hash_tree(ROOT / "compiler") ==
            (receipt["compiler"]["tree_sha256"], receipt["compiler"]["tree_files"]),
            f"compiler tree changed: {hash_tree(ROOT / 'compiler')}")

    canonical = ROOT / "validation/canonical_resnet50"
    candidate = ROOT / "validation/full_spike_candidate"
    compiler = receipt["compiler"]
    require(sha256(canonical / "target.mlir") == compiler["target_mlir_sha256"],
            "canonical target changed")
    require(sha256(canonical / "command_buffer.json") ==
            compiler["command_buffer_sha256"], "canonical command buffer changed")
    require(sha256(canonical / "object_build/kernel.o") ==
            compiler["kernel_object_sha256"], "canonical kernel object changed")
    command_buffer = json.loads((canonical / "command_buffer.json").read_text())
    census = command_buffer["params"]["convolution_lowering"]
    require(census["native_loop_conv_compute_only_count"] == 1,
            "actual command buffer does not select exactly one compute-only conv")
    require(census["native_loop_conv_narrow_count"] == 0,
            "unexpected narrow LOOP_CONV selection")
    require(census["fallback_count"] == 52, "actual fallback census changed")
    selections = census["selections"]
    require(len(selections) == 53, "actual source-convolution census changed")
    require(sum(row.get("selected") == "gemmini_loop_conv_ws_compute_only"
                for row in selections) == 1, "selection receipt changed")
    require(sum(row.get("reason") == "loop_conv_store_is_narrow_only"
                for row in selections) == 52, "fallback reasons changed")

    model = receipt["whole_model_spike"]
    if (candidate / "firesim.tar.gz").exists():
        verify_private_candidate(candidate, receipt)

    micro = ROOT / "validation/two_descriptor_null_output"
    gate = receipt["two_descriptor_exact_gate"]
    require(sha256(micro / "target.mlir") == gate["target_mlir_sha256"],
            "two-descriptor target changed")
    require(sha256(micro / "two_descriptor.elf") == gate["elf_sha256"],
            "two-descriptor ELF changed")
    require(sha256(micro / "spike.repeat1.log") == gate["repeat_log_sha256"],
            "two-descriptor Spike evidence changed")
    log = (micro / "spike.repeat1.log").read_text()
    require("descriptors=2 checked=512 bad=0" in log and "\nPASS\n" in log,
            "two-descriptor exact witness absent")

    require(receipt["coverage"]["source_convolutions"] == 53, "wrong conv census")
    require(receipt["coverage"]["native_compute_only"] == 1, "wrong native census")
    require(receipt["coverage"]["fallback"] == 52, "wrong fallback census")
    require(model["logits_checked"] == 1000,
            "whole-model check incomplete")
    require(model["mismatches"] == 0,
            "whole-model mismatch recorded")
    require(model["top1"] == 258,
            "wrong whole-model top1")
    require(receipt["hardware"]["firesim_run"] is True and
            receipt["hardware"]["safe_to_submit"] is False and
            receipt["hardware"]["promoted"] is False and
            receipt["hardware"]["cycles"] == 1383906735,
            "candidate hardware rejection receipt changed")

    public_hardware_path = ROOT / "validation/firesim_queue_job_545_rejected_perf.json"
    public_hardware = json.loads(public_hardware_path.read_text())
    require(public_hardware["status"] == "hardware_tested_rejected_perf" and
            public_hardware["promoted"] is False and
            public_hardware["resubmission_blocked"] is True,
            "public q545 disposition changed")
    require(public_hardware["queue_job"]["id"] == 545 and
            public_hardware["queue_job"]["state"] == "DONE" and
            public_hardware["queue_job"]["return_code"] == 0,
            "public q545 queue result changed")
    metrics = public_hardware["measured_metrics"]
    comparison = public_hardware["q535_comparison"]
    require(metrics["cycles"] == 1383906735 and
            comparison["q535_cycles"] == 1316619699 and
            comparison["additional_cycles"] == 67287036 and
            abs(comparison["regression_percent"] - 5.1105901006) < 1e-10,
            "public q545 cycle comparison changed")
    require(comparison["counter_deltas"]["main_ex_cycles"]["delta"] == 39100793 and
            comparison["counter_deltas"]["reservation_station_active_cycles"]["delta"] == 40037604 and
            comparison["counter_deltas"]["rdma_bytes_rec"]["delta"] == 3144320 and
            comparison["counter_deltas"]["wdma_bytes_sent"]["delta"] == 0,
            "public q545 counter diagnosis changed")
    correctness = public_hardware["correctness"]
    require(correctness["logits_checked"] == 1000 and correctness["bad"] == 0 and
            correctness["nonfinite"] == 0 and correctness["top1"] == 258 and
            correctness["checksum_fnv1a64"] == "c6e777c3fe0aae90",
            "public q545 correctness changed")
    public_uart_path = ROOT / "validation/firesim_queue_job_545_uart_public.log"
    require(sha256(public_uart_path) ==
            public_hardware["public_evidence"]["sanitized_uart_log_sha256"],
            "public q545 UART summary changed")
    public_uart = public_uart_path.read_text()
    for witness in (
        "MERLIN_METRIC cycles=1383906735",
        "MERLIN_RESULT checksum_fnv1a64=c6e777c3fe0aae90",
        "MERLIN_RESULT logits_checked=1000 bad=0 nonfinite=0 top1=258 expected_top1=258",
        "MERLIN_DISPOSITION status=hardware_tested_rejected_perf promoted=false resubmission_blocked=true",
    ):
        require(witness in public_uart, f"missing public q545 witness: {witness}")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "compiler"), str(ROOT / "runtime_target")])
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=ROOT, env=env, check=True)
    print("Compute-only LOOP_CONV gate passed: exact micro + 51 tests; "
          "q545 exact hardware result is performance-rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
