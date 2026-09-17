#!/usr/bin/env python3
"""Replay the exact model-derived multiply capsule on the Radiance L2 Cyclotron oracle."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path

import yaml

from merlin.common.paths import repo_root
from merlin.runtime.backends import base as backend_base
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
from merlin.targetgen.linalg_lower import lower_linalg_to_cb


CAPSULE = "SY_app_elementwise_mul_f32_rank1_32_l2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _perturb(value):
    out = copy.deepcopy(value)
    cursor = out
    while isinstance(cursor[0], list):
        cursor = cursor[0]
    cursor[0] = float(cursor[0]) + 100.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=("positive", "negative"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--publish", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout must be in [1, 300]")

    root = repo_root()
    capsule_dir = root / "merlin/contract/capsules/radiance/layers" / CAPSULE
    cap = load_capsule(capsule_dir, contract=root / "merlin/contract")
    parsed = parse_linalg_mlir((capsule_dir / "capsule.interface.mlir").read_text(encoding="utf-8"))
    cb = lower_linalg_to_cb(parsed, target="radiance")

    backend_base._load_oot_backend(
        "muon", root / "merlin/targets/muon/backend", ns="merlin._oot_l2_application_audit")
    from merlin._oot_l2_application_audit.muon import muon_codegen_mlir, muon_oracles

    lowered = muon_codegen_mlir.emit_kernel_mlir(cb)
    golden_doc = yaml.safe_load((capsule_dir / "golden.yaml").read_text(encoding="utf-8")) or {}
    expected = copy.deepcopy(golden_doc["outputs"])
    declared_outputs = list(cb.get("outputs") or ())
    if len(expected) == len(declared_outputs) == 1 and next(iter(expected)) != declared_outputs[0]:
        expected = {declared_outputs[0]: next(iter(expected.values()))}
    if args.case == "negative":
        name = next(iter(expected))
        expected[name] = _perturb(expected[name])

    canonical = CG.canonical_input_values(cap, capsule_dir)
    raws = CG.canonical_input_raws(cap, capsule_dir)
    tensors = cb["tensors"]
    leaves = [name for name in cb["arg_order"] if name not in set(cb["outputs"])]
    values = list(canonical.values())
    if len(leaves) != len(values):
        raise RuntimeError(f"positional input mismatch: {leaves} vs {list(canonical)}")
    cb["canonical_inputs"] = dict(zip(leaves, values))
    raw_values = list(raws.values())
    if len(leaves) != len(raw_values):
        raise RuntimeError("positional raw input mismatch")
    for leaf, raw in zip(leaves, raw_values):
        tensors[leaf]["preload_b64"] = base64.b64encode(raw).decode()

    args.work.mkdir(parents=True, exist_ok=True)
    result = muon_oracles.cyclotron_adapter()(cb, lowered, args.work, args.timeout)
    numeric = CG.compare(expected, result["outputs"], cap["numeric_policy"],
                         golden_source=golden_doc.get("golden_source", "unknown"))

    args.publish.mkdir(parents=True, exist_ok=True)
    public_cb = copy.deepcopy(cb)
    public_cb.pop("canonical_inputs", None)
    for spec in public_cb["tensors"].values():
        spec.pop("preload_b64", None)
    (args.publish / "command_buffer.json").write_text(
        json.dumps(public_cb, indent=2) + "\n", encoding="utf-8")
    (args.publish / "lowered.llvm.mlir").write_text(lowered, encoding="utf-8")
    receipt = {
        "schema": "radiance_application_elementwise_l2_v2",
        "case": args.case,
        "capsule": CAPSULE,
        "expected_control": "pass" if args.case == "positive" else "fail",
        "numeric": numeric,
        "oracle": result["oracle"],
        "timing": result["timing"],
        "cycles": result.get("cycles"),
        "toolchain": result.get("toolchain"),
        "capsule_yaml_sha256": _sha(capsule_dir / "capsule.yaml"),
        "capsule_interface_sha256": _sha(capsule_dir / "capsule.interface.mlir"),
        "lowered_llvm_mlir_sha256": _sha(args.publish / "lowered.llvm.mlir"),
        "oracle_scope": "L2 Cyclotron functional/performance model; not GSIM and not physical RTL",
        "private_expected_output_published": False,
        "pr_kernel_used_as_application_evidence": False,
    }
    (args.publish / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    wanted = "pass" if args.case == "positive" else "fail"
    return 0 if numeric["status"] == wanted else 1


if __name__ == "__main__":
    raise SystemExit(main())
