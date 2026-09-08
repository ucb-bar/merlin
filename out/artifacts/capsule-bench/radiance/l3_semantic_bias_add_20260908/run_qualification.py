#!/usr/bin/env python3
"""Rebuild and L3-qualify the semantic-selector bias-add bridge."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

from merlin.common.paths import repo_root
from merlin.runtime.backends import base as backend_base
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
from merlin.targetgen.linalg_lower import lower_linalg_to_cb


CAPSULE = "RP16_bias_add_fp32_pt"
MAX_CYCLES = 360_000


def _perturb_first(value):
    if isinstance(value, list):
        out = copy.deepcopy(value)
        out[0] = _perturb_first(out[0])
        return out
    return float(value) + 100.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("positive", "negative"), required=True)
    parser.add_argument("--emulator", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--publish", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout must be in [1, 300] seconds")
    if not args.emulator.is_file():
        parser.error(f"emulator does not exist: {args.emulator}")

    root = repo_root()
    capsule_dir = root / "merlin/contract/capsules/radiance/model_slices" / CAPSULE
    selection_path = (
        root / "merlin/experiments/capsule_bench/targets/radiance/contracts"
        / "kernel_library_pr1_v1.yaml"
    )
    hardware_path = root / "out/artifacts/targets/radiance/contracts/target_contract.yaml"
    cap = load_capsule(capsule_dir, contract=root / "merlin/contract")
    parsed = parse_linalg_mlir((capsule_dir / "capsule.interface.mlir").read_text(encoding="utf-8"))
    cb = lower_linalg_to_cb(parsed, target="radiance")

    backend_base._load_oot_backend(
        "muon", root / "merlin/targets/muon/backend", ns="merlin._oot_sim_oracles")
    from merlin._oot_sim_oracles.muon import muon_codegen_mlir, muon_kernel_selection, muon_oracles

    selection_contract = muon_kernel_selection.load_selection_contract(selection_path)
    hardware_contract = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    lowered = muon_codegen_mlir.emit_kernel_mlir(
        cb,
        selection_contract=selection_contract,
        hardware_contract=hardware_contract,
    )

    golden = CG.golden(cap)
    declared_outputs = list(cb.get("outputs") or ())
    if len(golden) == len(declared_outputs) == 1 and next(iter(golden)) != declared_outputs[0]:
        # The linalg reference lowering has a positional ABI and calls its result ``out``; preserve
        # values while rebinding the frontend name (Y0) to that sole declared output, as the production
        # capsule runner does for positional interfaces.
        golden = {declared_outputs[0]: next(iter(golden.values()))}
    if args.case == "negative":
        output_name = next(iter(golden))
        golden = copy.deepcopy(golden)
        golden[output_name] = _perturb_first(golden[output_name])
    canonical = CG.canonical_input_values(cap, capsule_dir)
    raws = CG.canonical_input_raws(cap, capsule_dir)
    if canonical:
        cb["canonical_inputs"] = canonical
        tensors = cb["tensors"]
        if not (set(canonical) & set(tensors)):
            leaves = list(cb["arg_order"][:-1])
            values = list(canonical.values())
            if len(leaves) != len(values):
                raise RuntimeError("positional leaf/canonical operand count mismatch")
            cb["canonical_inputs"] = dict(zip(leaves, values))
            if raws:
                raw_values = list(raws.values())
                if len(leaves) != len(raw_values):
                    raise RuntimeError("positional leaf/raw operand count mismatch")
                for leaf, raw in zip(leaves, raw_values):
                    tensors[leaf]["preload_b64"] = base64.b64encode(raw).decode()
        else:
            for name, spec in tensors.items():
                if name in raws and spec.get("role") in ("input", "weight", "bias"):
                    spec["preload_b64"] = base64.b64encode(raws[name]).decode()
    cb["_oracle_expected_outputs"] = golden
    cb["_oracle_numeric_policy"] = cap["numeric_policy"]

    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    os.environ["MERLIN_MUON_GSIM_EMU"] = str(args.emulator.resolve())
    os.environ["MERLIN_MUON_GSIM_MAXCYCLES"] = str(MAX_CYCLES)
    result = muon_oracles.gsim_muon_adapter("radiance")(
        cb, lowered, args.work, args.timeout)

    args.publish.mkdir(parents=True, exist_ok=True)
    public_cb = copy.deepcopy(cb)
    public_cb.pop("canonical_inputs", None)
    public_cb.pop("_oracle_expected_outputs", None)
    public_cb.pop("_oracle_numeric_policy", None)
    (args.publish / "command_buffer.json").write_text(
        json.dumps(public_cb, indent=2) + "\n", encoding="utf-8")
    (args.publish / "lowered.llvm.mlir").write_text(lowered, encoding="utf-8")
    (args.publish / "selection_report.json").write_text(
        json.dumps(public_cb["params"]["kernel_family_selection"], indent=2) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "radiance_l3_semantic_family_qualification_v1",
        "case": args.case,
        "capsule": CAPSULE,
        "expected_control": "pass" if args.case == "positive" else "fail",
        "numeric_verdict": result["numeric_verdict"],
        "timing": result["timing"],
        "bounded_observation": result["bounded_observation"],
        "oracle": result["oracle"],
        "result_page": result["result_page"],
        "submitted_kernel_elf_sha256": _sha256(args.work / "kernel.radiance.elf"),
        "emulator_sha256": _sha256(args.emulator.resolve()),
        "selection_contract_sha256": _sha256(selection_path),
        "hardware_contract_sha256": _sha256(hardware_path),
        "private_expected_output_published": False,
        "reference_kernel_linked_or_dispatched": False,
    }
    (args.publish / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
