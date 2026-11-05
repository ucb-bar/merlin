"""Execute physical Atlas assembly with pinned npu_model RTL arithmetic.

The fast path calls the assembled instruction objects' own exec methods on
ArchState, including physical registers, DMA, and branch delay slots. Timing
uses the unmodified Core and a logger that retains no trace events.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).absolute().parent
MODEL_ROOT = Path(os.environ.get("ATLAS_NPU_MODEL_ROOT", ROOT / "unconfigured-model-root")).absolute()
MODEL_PIN = os.environ.get("ATLAS_NPU_MODEL_PIN", "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5")
if any(prefix.is_symlink() for prefix in reversed([MODEL_ROOT, *MODEL_ROOT.parents])):
    raise ValueError("symlink model paths are unsupported")
sys.path.insert(0, str(MODEL_ROOT))

import torch
from npu_model.configs import isa_definition  # Register the physical assembler ISA.
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.hardware import Core
from npu_model.hardware.arch_state import ArchState
from npu_model.hardware.dma import DmaExecutionUnit
from npu_model.isa import SBType, UJType
from npu_model.isa_types import EXU
from npu_model.isa_patterns import TensorComputeBinary, TensorComputeUnary
from npu_model.util.converter import input_to_program

torch.set_num_threads(1)


class QuietLogger:
    """Core logger interface with constant space and no disk traffic."""
    def log_cycle(self, *args, **kwargs): pass
    def log_insn(self, *args, **kwargs): pass
    def log_retire(self, *args, **kwargs): pass
    def log_stage_start(self, *args, **kwargs): pass
    def log_stage_end(self, *args, **kwargs): pass
    def log_dependency(self, *args, **kwargs): pass
    def log_arch_value(self, *args, **kwargs): pass


@dataclass
class Execution:
    arena: bytes
    output: bytes
    cycles: int | None
    elapsed_seconds: float
    instructions: int
    program_words: int
    program_sha256: str
    observed: dict
    halt_pc: int

    def metadata(self):
        return {key: value for key, value in asdict(self).items() if key not in ("arena", "output")}


def safe_path(value):
    path = Path(value).absolute()
    if ".." in path.parts:raise ValueError("parent traversal is unsupported")
    if any(prefix.is_symlink() for prefix in reversed([path, *path.parents])):
        raise ValueError("symlink input and output paths are unsupported")
    return path


def model_binding():
    actual = subprocess.check_output(["git", "-C", str(MODEL_ROOT), "rev-parse", "HEAD"], text=True).strip()
    if actual != MODEL_PIN:
        raise RuntimeError(f"npu_model revision is {actual}, expected {MODEL_PIN}")
    loaded = []
    for name, module in list(sys.modules.items()):
        if name == "npu_model" or name.startswith("npu_model."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = safe_path(filename)
                if not path.is_relative_to(MODEL_ROOT):
                    raise RuntimeError(f"model module imported outside selected root: {name}")
                loaded.append(str(path.relative_to(MODEL_ROOT)))
    subprocess.run(["git", "-C", str(MODEL_ROOT), "ls-files", "--error-unmatch", "--", *loaded],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["git", "-C", str(MODEL_ROOT), "diff", "--quiet", "HEAD", "--", *loaded], check=True)
    return {"commit": actual, "numerics": "rtl", "root": str(MODEL_ROOT)}


def assemble(source):
    program = input_to_program(io.StringIO(source))
    for pc, insn in enumerate(program.instructions):
        if insn.exu == EXU.VECTOR and isinstance(insn, (TensorComputeBinary, TensorComputeUnary)) and insn.mnemonic != "vmov":
            if int(insn.vd) % 2:
                raise ValueError(f"odd VPU pair destination at word {pc}: {insn}")
        if insn.exu == EXU.VECTOR and isinstance(insn, TensorComputeBinary):
            a, b = int(insn.vs1), int(insn.vs2)
            if a != b and a % 32 == b % 32:
                raise ValueError(f"binary VPU sources alias one physical MRF read port at word {pc}: {insn}")
    words = program.assemble()
    if len(words) > 32768:
        raise ValueError("assembled program exceeds physical IMEM")
    return program, hashlib.sha256(b"".join(word.to_bytes(4, "little") for word in words)).hexdigest()


def _branch(insn):
    return isinstance(insn, (SBType, UJType)) or insn.mnemonic == "jalr"


def _functional(program, state, max_instructions):
    pc, pending, executed = 0, None, 0
    events = Counter()
    while 0 <= pc < len(program.instructions):
        if executed >= max_instructions:
            raise RuntimeError(f"functional instruction bound reached at word {pc}")
        insn = program.instructions[pc]
        if pending is not None and _branch(insn):
            raise RuntimeError(f"branch in delay slot at word {pc}")
        state.execute_pc, state.pc, state.npc = pc, pc, pc + 1
        state.redirect_requested = False
        if insn.mnemonic in ("dma.load", "dma.store"):
            events["load_bytes" if insn.mnemonic == "dma.load" else "store_bytes"] += state.read_xrf(insn.rs2)
        insn.exec(state)
        executed += 1
        if state.halted:
            break
        next_pc = pending if pending is not None else pc + 1
        pending = state.npc if state.redirect_requested else None
        pc = next_pc
    return executed, dict(events)


def execute(source: str, arena: bytes, output_offset: int, output_bytes: int, *, timed=False,
            max_cycles=20_000_000, max_instructions=5_000_000, seed=42, dram_base=0x90000000,
            progress=False) -> Execution:
    """Return the actual full staged arena, output bytes, and measured run metadata."""
    if output_offset < 0 or output_bytes < 0 or output_offset + output_bytes > len(arena):
        raise ValueError("output region is outside staged arena")
    if timed and max_cycles <= 0:
        raise ValueError("timed max_cycles must be positive")
    program, digest = assemble(source)
    program.dram_base = dram_base
    config = DefaultHardwareConfig()
    config.arch_state_config = replace(config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=seed)
    data = torch.frombuffer(bytearray(arena), dtype=torch.uint8)
    started = time.perf_counter()
    core = None
    state = None
    try:
        if timed:
            core = Core(config=config, logger=QuietLogger())
            program.memory_regions = [(0, data)]
            core.load_program(program)
            state = core.arch_state
            dma = next(exu for exu in core.exus if isinstance(exu, DmaExecutionUnit))
            events = Counter()
            next_progress = 1_000_000
            while not core.is_finished() and core.cycle_count < max_cycles:
                try:
                    core.tick()
                except Exception as error:
                    raise RuntimeError(f"Core failed at cycle {core.cycle_count}, executing word {state.execute_pc}") from error
                event = dma.last_cycle
                if event.a:
                    events["store_bytes" if event.a[2] else "load_bytes"] += 32
                if core.stall_reason:
                    events["stall_" + core.stall_reason] += 1
                if progress and core.cycle_count >= next_progress:
                    elapsed = time.perf_counter() - started
                    print(f"Atlas timed progress: cycles={core.cycle_count}, elapsed_seconds={elapsed:.3f}, "
                          f"pc_word={state.execute_pc}", file=sys.stderr, flush=True)
                    next_progress += 1_000_000
            if not core.is_finished():
                elapsed = time.perf_counter() - started
                raise TimeoutError(f"timed max_cycles={max_cycles} reached at cycle {core.cycle_count}, "
                                   f"pc_word={state.execute_pc}, elapsed_seconds={elapsed:.3f}")
            cycles, executed = core.cycle_count, core.total_completed
            if any(exu.has_in_flight for exu in core.exus) or any(state.check_flag(ch) for ch in range(8)):
                raise AssertionError("timed run halted before engines and DMA drained")
            events = dict(events)
        else:
            state = ArchState(config.arch_state_config)
            state.write_dram(program.dram_base, data)
            executed, events = _functional(program, state, max_instructions)
            cycles = None
        if not state.halted or state.halt_reason != "ecall":
            raise AssertionError(f"run did not reach ECALL: pc={state.execute_pc}, cycles={cycles}")
        actual = bytes(state.read_dram(program.dram_base, len(arena)).tolist())
        if actual[:output_offset] != arena[:output_offset] or actual[output_offset + output_bytes:] != arena[output_offset + output_bytes:]:
            raise AssertionError("input or output guard bytes changed")
        return Execution(actual, actual[output_offset:output_offset + output_bytes], cycles,
                         time.perf_counter() - started, executed, len(program), digest, events, state.execute_pc)
    finally:
        if core is not None:
            core.close()
        elif state is not None:
            state.close()
