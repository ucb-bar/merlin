"""Execute emitted words on the pinned NPU model; verify outputs and guards."""

import collections
import hashlib
import inspect
import dataclasses
import json
import os
from pathlib import Path
import sys
import argparse
import subprocess
import numpy as np
import torch
from safetensors.numpy import load_file

from source_roots import MERLIN_ROOT as ROOT, QWEN_ROOT, bootstrap_compiler

COMPILER_ROOT = bootstrap_compiler()


def model_instruction(row, inventory, isa):
    """Bind a decoded instruction and require an identical re-encoded word.

    npu_model executes instruction objects. Decode uses the existing RTL
    inventory, independently of the emitter and its JSON operand annotations.
    """
    from npu_model.isa import SBType, SType, UJType, UType, VIType

    mode = row["mode_id"]
    variant = next(v for v in inventory["variants"] if v["id"] == mode)
    name = (
        mode.removesuffix("_ANY")
        if mode.startswith("DMA_")
        else variant["model_classes"][0]
    )
    cls, word = getattr(isa, name), row["word_u32"]
    operands = {
        "rd": (word >> 7) & 31,
        "rs1": (word >> 15) & 31,
        "rs2": (word >> 20) & 31,
        "vd": (word >> 7) & 63,
        "vs1": (word >> 13) & 63,
        "vs2": (word >> 19) & 63,
        "ch": (word >> 12) & 7,
        "es1": (word >> 13) & 31,
        "es2": (word >> 19) & 31,
    }
    if issubclass(cls, SBType):
        immediate = row["fields"]["offset_bytes"]
    elif issubclass(cls, UJType):
        immediate = row["fields"]["offset"]
    elif issubclass(cls, SType):
        immediate = ((word >> 25) << 5) | ((word >> 7) & 31)
    elif issubclass(cls, VIType):
        immediate = word >> 16
    elif issubclass(cls, UType):
        immediate = word >> 12
    else:
        immediate = word >> 20
    operands["imm"] = immediate
    instruction = cls(
        **{name: operands[name] for name in inspect.signature(cls).parameters}
    )
    if instruction.to_bytecode() != word:
        raise ValueError(f"npu_model instruction differs from word {row['word_index']}")
    return instruction


from check_program_decode import decode_binary
from atlas_virtual_evaluator import (
    parse_program,
    evaluate,
    RuntimeInputs,
    _tile_from_bytes,
    _tile_bytes,
)
from npu_model.configs import isa_definition as isa
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    compiled = root / "native"
    plan = json.loads((compiled / "execution_plan.json").read_text())
    blobs = {int(p.stem): p.read_bytes() for p in (root / "inputs").glob("*.bin")}
    tiles = {
        i: _tile_from_bytes(blob, "fp8" if len(blob) == 1024 else "bf16")
        for i, blob in blobs.items()
    }
    ref = evaluate(
        parse_program((compiled / "atlas-virtual.mlir").read_text()),
        RuntimeInputs(tiles),
    )
    golden = load_file(str(root / "golden.safetensors"))
    inventory = json.loads((COMPILER_ROOT / "docs/selected-variant-inventory.json").read_text())
    bins = Path(os.environ["ATLAS_OOT_BIN_DIR"])
    reports = []
    for name, pipeline in [
        ("unscheduled", ["--atlas-compile-unscheduled"]),
        (
            "boundary_only",
            ["--lower-atlas-virtual-boundary-to-dma", "--atlas-compile-unscheduled"],
        ),
        ("scheduled", ["--atlas-compile"]),
    ]:
        dest = root / name
        dest.mkdir(exist_ok=True)
        machine = subprocess.check_output(
            [str(bins / "atlas-opt"), *pipeline, str(compiled / "atlas-virtual.mlir")]
        )
        (dest / "atlas-machine.mlir").write_bytes(machine)
        physical = json.loads(
            subprocess.check_output(
                [
                    str(bins / "atlas-emit"),
                    "--program-json",
                    str(dest / "atlas-machine.mlir"),
                ]
            )
        )
        (dest / "physical-program.json").write_text(json.dumps(physical, indent=2))
        raw = b"".join(
            x["word_u32"].to_bytes(4, "little") for x in physical["instructions"]
        )
        (dest / "program.bin").write_bytes(raw)
        decoded = decode_binary(inventory, raw)
        program = InstantiableProgram(
            [model_instruction(row, inventory, isa) for row in decoded["instructions"]]
        )
        program.dram_base = 0x90000000
        initial = bytearray([0x5B] * 0x42000)
        for i, b in blobs.items():
            initial[i * 2048 : i * 2048 + len(b)] = b
        program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
        config = DefaultHardwareConfig()
        config.arch_state_config = dataclasses.replace(
            config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=187
        )
        sim = Simulation(
            config,
            LoggerConfig(filename=str(dest / "trace.json")),
            program,
            verbose=False,
            record_timeline=True,
        )
        state = sim.core.arch_state
        try:
            sim.run(max_cycles=1000000)
            stats = dataclasses.asdict(sim.get_stats())
            assert state.halted and stats["runtime_errors"] == 0, stats
            error = []
            for j, address in enumerate(plan["output_addresses"]):
                data = bytes(state.read_dram(address, 2048).tolist())
                expected = _tile_bytes(ref.outputs[j])
                assert data == expected, (name, j, "output mismatch")
                initial[
                    address - program.dram_base : address - program.dram_base + 2048
                ] = expected
                bits = np.array(ref.outputs[j].bits, dtype=np.uint32) << 16
                actual = bits.view(np.float32).reshape(32, 32)
                wanted = golden[str(j)][:32, :32]
                error.append(
                    {
                        "max_abs": float(np.max(np.abs(actual - wanted))),
                        "relative_l2": float(
                            np.linalg.norm(actual - wanted) / np.linalg.norm(wanted)
                        ),
                    }
                )
            assert bytes(
                state.read_dram(program.dram_base, len(initial)).tolist()
            ) == bytes(initial), "guard/input mutation"
            active = collections.Counter()
            for cycle in sim.timeline:
                units = [u for u, b in cycle.items() if b]
                if len(units) >= 2:
                    active["overlap_cycles"] += 1
                if cycle.get("DMA0") and cycle.get("Matrix0"):
                    active["dma_mxu_overlap"] += 1
            stats.update(
                name=name,
                pipeline=pipeline,
                words=len(raw) // 4,
                precision_error=error,
                correctness="bit_exact_virtual_reference_and_guards",
                overlap=dict(active),
            )
            stats["program_sha256"] = hashlib.sha256(raw).hexdigest()
            reports.append(stats)
            print(f"{root.name}/{name}: {stats['cycles']} cycles; verified", flush=True)
        finally:
            state.close()
    (root / "performance.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
