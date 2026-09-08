#!/usr/bin/env python3
"""Reproduce the q535/q545 hardware and issued-command differential."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
Q535_RECEIPT = ROOT / (
    "out/artifacts/perf-bench/gemmini/"
    "resnet50_merlin_phase2_affine_im2col_spans_w8a8_warm_measured_"
    "firesim_candidate_20260908/validation/firesim_queue_job_535_success.json"
)
Q535_MLIR = ROOT / (
    "out/artifacts/perf-bench/gemmini/"
    "resnet50_merlin_phase2_affine_im2col_spans_w8a8_warm_measured_"
    "firesim_candidate_20260908/compiler/target.mlir"
)
Q545_UART = Path("/scratch/firesim_queue/jobs/545/simulation/sim_slot_0/uartlog")
Q545_MLIR = ROOT / (
    "out/artifacts/perf-bench/gemmini/development_phase2_compute_only_loop_conv_20260908/"
    "validation/full_spike_candidate/compiler/target.mlir"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def q545_metrics() -> dict[str, int]:
    result = {}
    for name, value in re.findall(r"MERLIN_METRIC (\w+)=(\d+)", Q545_UART.read_text()):
        result[name] = int(value)
    return result


def commands(path: Path, task: int = 1) -> Counter[str]:
    result: Counter[str] = Counter()
    marker = f"merlin.global_task = {task} "
    for line in path.read_text().splitlines():
        if "llvm.inline_asm" not in line or marker not in line:
            continue
        asm = re.search(r'asm_string = "([^"]+)', line)
        if asm is None:
            raise RuntimeError(f"unparsed inline assembly: {line}")
        result[asm.group(1)] += 1
    return result


def main() -> None:
    q535 = json.loads(Q535_RECEIPT.read_text())["measured_metrics"]
    q545 = q545_metrics()
    keys = sorted(set(q535) & set(q545))
    delta = {
        key: {
            "absolute": q545[key] - q535[key],
            "percent_of_q535": 100.0 * (q545[key] - q535[key]) / q535[key],
        }
        for key in keys
    }
    print(json.dumps({
        "schema": "gemmini_q545_vs_q535_differential_v1",
        "inputs": {
            "q535_receipt": str(Q535_RECEIPT), "q535_receipt_sha256": sha256(Q535_RECEIPT),
            "q535_mlir": str(Q535_MLIR), "q535_mlir_sha256": sha256(Q535_MLIR),
            "q545_uart": str(Q545_UART), "q545_uart_sha256": sha256(Q545_UART),
            "q545_mlir": str(Q545_MLIR), "q545_mlir_sha256": sha256(Q545_MLIR),
        },
        "q535": {key: q535[key] for key in keys},
        "q545": {key: q545[key] for key in keys},
        "delta": delta,
        "q535_task1_issued": commands(Q535_MLIR),
        "q545_task1_issued": commands(Q545_MLIR),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
