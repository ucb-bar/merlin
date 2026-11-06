"""Run emitted Qwen kernels through npu_speed_of_light.roofline.

Select the desired npu_model checkout with PYTHONPATH. This uses its actual
speed_of_light API and checks outputs/guards before each simulation closes.
"""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from measure_workload import model_instruction, COMPILER_ROOT
from source_roots import MERLIN_ROOT as ROOT
from atlas_virtual_evaluator import (
    RuntimeInputs,
    _tile_bytes,
    _tile_from_bytes,
    evaluate,
    parse_program,
)
from check_program_decode import decode_binary
from npu_model.configs import isa_definition as isa
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram
from npu_speed_of_light import roofline



def analyze(work, variant, memory, inventory):
    blobs = {int(p.stem): p.read_bytes() for p in (work / "inputs").glob("*.bin")}
    tiles = {
        i: _tile_from_bytes(b, "fp8" if len(b) == 1024 else "bf16")
        for i, b in blobs.items()
    }
    reference = evaluate(
        parse_program((work / "native/atlas-virtual.mlir").read_text()),
        RuntimeInputs(tiles),
    )
    plan = json.loads((work / "native/execution_plan.json").read_text())
    initial = bytearray([0x5B] * 0x42000)
    for slot, blob in blobs.items():
        initial[slot * 2048 : slot * 2048 + len(blob)] = blob
    expected = initial.copy()
    for index, address in enumerate(plan["output_addresses"]):
        tile = reference.outputs[index]
        payload = _tile_bytes(tile)
        offset = address - 0x90000000
        expected[offset : offset + len(payload)] = payload
    raw = (work / variant / "program.bin").read_bytes()
    decoded = decode_binary(inventory, raw)
    program = InstantiableProgram(
        [model_instruction(row, inventory, isa) for row in decoded["instructions"]]
    )
    program.dram_base = 0x90000000
    program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
    config = DefaultHardwareConfig()
    config.arch_state_config = dataclasses.replace(
        config.arch_state_config,
        numerics="rtl",
        randomize_init=True,
        init_seed=187,
    )
    if memory == "fast-fixed":
        roofline.configure_memory(config, "fixed", latency=43, cycles_per_beat=1)
    else:
        roofline.configure_memory(config, "curve")
    validation = {}

    class CheckedSimulation(Simulation):
        def close(self):
            try:
                if self.core is not None and self.core.is_finished():
                    stats = self.get_stats()
                    if stats.runtime_errors:
                        raise RuntimeError(
                            f"{stats.runtime_errors} simulator runtime errors"
                        )
                    actual = bytes(
                        self.core.arch_state.read_dram(
                            program.dram_base, len(expected)
                        ).tolist()
                    )
                    if actual != bytes(expected):
                        raise RuntimeError(
                            "virtual-reference output or DRAM guard mismatch"
                        )
                    validation.update(
                        correctness="bit_exact_virtual_reference_and_guards",
                        runtime_errors=stats.runtime_errors,
                        exu_stats=dataclasses.asdict(stats)["exu_stats"],
                    )
            finally:
                super().close()

    original = roofline.Simulation
    roofline.Simulation = CheckedSimulation
    try:
        result = roofline.speed_of_light(program, config, max_cycles=1_000_000)
    finally:
        roofline.Simulation = original
    assert validation and 0 < result.bound.cycles <= result.measured
    bandwidth = result.dram_bytes / result.dram_cycles
    row = dict(
        workload=work.name,
        variant=variant,
        memory=memory,
        memory_description=roofline.describe_memory(config),
        **dataclasses.asdict(result),
        bound=result.bound.name,
        bound_cycles=result.bound.cycles,
        bound_efficiency=result.bound.cycles / result.measured,
        ideal_speedup_ceiling=result.measured / result.bound.cycles,
        arithmetic_intensity=result.flops / result.dram_bytes,
        ridge_flops_per_byte=result.peak_flops_per_cycle / bandwidth,
        attained_flops_per_cycle=result.flops / result.measured,
        dram_bytes_per_cycle=bandwidth,
        program_sha256=hashlib.sha256(raw).hexdigest(),
        **validation,
    )
    row["roofline_side"] = (
        "memory"
        if row["arithmetic_intensity"] < row["ridge_flops_per_byte"]
        else "compute"
    )
    return row, roofline.format_report(
        f"{work.name}/{variant}", row["memory_description"], result
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=ROOT / "out/qwen-final")
    parser.add_argument("-o", "--out", type=Path, default=ROOT / "out/qwen-roofline")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "roofline.json").exists():
        raise SystemExit("choose a fresh output directory")
    inventory = json.loads((COMPILER_ROOT / "docs/selected-variant-inventory.json").read_text())
    model_root = Path(roofline.__file__).resolve().parents[1]
    provenance = dict(
        model_root=str(model_root),
        model_revision=subprocess.check_output(
            ["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        model_tracked_changes=subprocess.check_output(
            ["git", "-C", str(model_root), "diff", "HEAD", "--stat"], text=True
        ),
        roofline_source_sha256=hashlib.sha256(
            Path(roofline.__file__).read_bytes()
        ).hexdigest(),
        curve_sha256=hashlib.sha256(
            (
                model_root / "npu_model/configs/memory_curves/ee290sim_vcs_probe.json"
            ).read_bytes()
        ).hexdigest(),
        torch_version=torch.__version__,
        python=sys.version,
        flop_definition="MXU MACs only; two FLOPs per multiply-accumulate",
    )
    rows = []
    texts = []
    for status in json.loads((args.campaign / "compilation.json").read_text()):
        if status["status"] != "compiled" or (
            args.only and status["name"] not in args.only
        ):
            continue
        for memory in ("curve", "fast-fixed"):
            for variant in ("unscheduled", "boundary_only", "scheduled"):
                row, report = analyze(
                    args.campaign / status["name"], variant, memory, inventory
                )
                rows.append(row)
                texts.append(report)
                print(
                    f"{row['workload']}/{variant}/{memory}: {row['measured']} cycles; "
                    f"{row['bound']} bound {row['bound_cycles']}; "
                    f"{row['bound_efficiency']:.1%} of bound; verified",
                    flush=True,
                )
    (args.out / "roofline.json").write_text(
        json.dumps(dict(provenance=provenance, results=rows), indent=2) + "\n"
    )
    (args.out / "roofline.txt").write_text("\n\n".join(texts) + "\n")


if __name__ == "__main__":
    main()
