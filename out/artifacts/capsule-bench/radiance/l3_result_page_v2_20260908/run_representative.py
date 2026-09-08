#!/usr/bin/env python3
"""Bounded replay of the production Radiance GSIM result-page adapter."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.runtime.backends import base as backend_base
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen.capsule_common import load_capsule


CASES = {
    "rp10_pass": 120_000,
    "rp10_negative_control": 120_000,
    "r4_rmsnorm_observed_fail": 360_000,
    "rp12_embed_scale": 360_000,
}
CAPSULES = {
    "rp10_pass": "RP10_gemv_batched_fp16_pt",
    "rp10_negative_control": "RP10_gemv_batched_fp16_pt",
    "r4_rmsnorm_observed_fail": "R4_rmsnorm_fp32",
    "rp12_embed_scale": "RP12_embed_scale_fp32_pt",
}


def _perturb_first(value):
    if isinstance(value, list):
        out = copy.deepcopy(value)
        out[0] = _perturb_first(out[0])
        return out
    return float(value) + 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--emulator", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=280)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout must be in [1, 300] seconds")
    if not args.emulator.is_file():
        parser.error(f"emulator does not exist: {args.emulator}")

    root = repo_root()
    here = Path(__file__).resolve().parent
    case_dir = here / "cases" / args.case
    # The trusted answer key stays in the local corpus and is deliberately not
    # published with this evidence artifact.
    capsule_dir = (root / "merlin/contract/capsules/radiance/model_slices"
                   / CAPSULES[args.case])
    cap = load_capsule(capsule_dir, contract=root / "merlin/contract")
    cb = json.loads((case_dir / "command_buffer.json").read_text(encoding="utf-8"))
    golden = CG.golden(cap)
    if args.case == "rp10_negative_control":
        name = next(iter(golden))
        golden = copy.deepcopy(golden)
        golden[name] = _perturb_first(golden[name])
    # Mirror capsule_runner's production binding: the kernel must execute on
    # the exact canonical operands from which golden.yaml was produced.
    canonical = CG.canonical_input_values(cap, capsule_dir)
    if canonical:
        cb["canonical_inputs"] = canonical
    raws = CG.canonical_input_raws(cap, capsule_dir)
    for name, spec in (cb.get("tensors") or {}).items():
        if name in raws and spec.get("role") in ("input", "weight", "bias"):
            spec["preload_b64"] = base64.b64encode(raws[name]).decode()
    cb["_oracle_expected_outputs"] = golden
    cb["_oracle_numeric_policy"] = cap["numeric_policy"]

    backend_base._load_oot_backend(
        "muon", root / "merlin/targets/muon/backend", ns="merlin._oot_sim_oracles")
    from merlin._oot_sim_oracles.muon import muon_oracles

    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    os.environ["MERLIN_MUON_GSIM_EMU"] = str(args.emulator.resolve())
    os.environ["MERLIN_MUON_GSIM_MAXCYCLES"] = str(CASES[args.case])
    result = muon_oracles.gsim_muon_adapter("radiance")(
        cb,
        (case_dir / "lowered.llvm.mlir").read_text(encoding="utf-8"),
        args.work,
        args.timeout,
    )
    (args.work / "adapter_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "numeric_verdict": result["numeric_verdict"],
        "timing": result["timing"],
        "bounded_observation": result["bounded_observation"],
        "work": str(args.work.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
