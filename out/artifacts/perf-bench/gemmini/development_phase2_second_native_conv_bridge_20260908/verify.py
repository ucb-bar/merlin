#!/usr/bin/env python3
"""Payload-free verifier for the q545 cost guard artifact."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(message)


def census(path: Path) -> tuple[int, int, list[str]]:
    data = json.loads(path.read_text())["params"]["convolution_lowering"]
    return (int(data["native_loop_conv_count"]), int(data["fallback_count"]),
            [row["reason"] for row in data["selections"]])


def main() -> int:
    status = json.loads((ROOT / "STATUS.json").read_text())
    require(status["status"] == "hardware_tested_rejected_perf_guarded",
            "unexpected artifact status")
    prod_target = ROOT / "validation/canonical_resnet50/production_target.mlir"
    diag_target = ROOT / "validation/canonical_resnet50/diagnostic_target.mlir"
    require(sha(prod_target) == status["target_identity"]["q535_target_sha256"],
            "production output is not byte-identical to q535")
    require(sha(diag_target) == status["target_identity"]["q545_target_sha256"],
            "diagnostic output is not byte-identical to q545")
    pn, pf, pr = census(
        ROOT / "validation/canonical_resnet50/production_command_buffer.json")
    dn, df, dr = census(
        ROOT / "validation/canonical_resnet50/diagnostic_command_buffer.json")
    require((pn, pf) == (0, 53), "production convolution census changed")
    require(pr.count("hardware_cost_guard_transposed_nchw_underfills_systolic_rows") == 1,
            "production hardware guard is absent")
    require((dn, df) == (1, 52), "diagnostic convolution census changed")
    require(dr.count("selected_compute_only_full_width_mvout") == 1,
            "diagnostic opt-in no longer selects the exact mechanism")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "compiler"), str(REPO / "merlin/python")]
    )
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT,
                   env=env, check=True)
    print("q545 cost guard verified: production 0/53 native, diagnostic 1/53 native")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
