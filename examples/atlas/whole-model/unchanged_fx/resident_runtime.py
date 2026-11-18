"""Qualification-only persistent-IMEM bridge; see the example README recipe."""
from dataclasses import replace
from hashlib import sha256
import json

from npu_model.functional import FunctionalResult
from npu_model.functional.program import FunctionalInstruction, FunctionalProgram
from functional_runtime import FunctionalModelRuntime, PhysicalFunctionalInterpreter, PhysicalFunctionalState
from model_runtime import authenticate_model, check_immutable_memory
from program_residency import NativeProgramResidency


class SelectedDelayInterpreter(PhysicalFunctionalInterpreter):
    """Original instruction.exec with one selected-core control-flow delay slot.

    This separately recorded control adapter agrees with the existing selected
    JAL reference sentinel. It does not establish a new hardware qualification.
    """
    def run(self, program, *, max_instructions=1_000_000, dram=None, vmem=None):
        if type(max_instructions) is not int or max_instructions <= 0:
            raise ValueError("max_instructions must be a positive integer")
        state = PhysicalFunctionalState(self.config)
        if self.math_proxy is not None:
            state.math = self.math_proxy
        if self.memory_factory is not None:
            state.vmem = self.memory_factory(state.cfg.vmem_size)
        for address, data in (dram or {}).items():
            state.load_dram(address, data)
        for address, data in (vmem or {}).items():
            state.load_vmem(address, data)
        pc, executed, pending = 0, 0, None
        while 0 <= pc < len(program.instructions):
            if executed >= max_instructions:
                raise RuntimeError(f"functional instruction limit ({max_instructions}) reached at PC {pc}")
            state.execute_pc, state.npc, state.redirect_requested = pc, pc + 1, False
            program.instructions[pc].instruction.exec(state)
            executed += 1
            if state.halted:
                return FunctionalResult(state, executed, pc + 1, state.halt_reason)
            if pending is not None:
                if state.redirect_requested:
                    raise RuntimeError("redirect in selected-core delay slot")
                pc, pending = pending, None
            elif state.redirect_requested:
                pending, pc = state.npc, pc + 1
            else:
                pc += 1
        raise RuntimeError("execution fell outside physical instruction image")


class ResidentFunctionalModelRuntime(FunctionalModelRuntime):
    """Saved host ABI stages originals; generic policy writes real execution IMEM.

    Each original image is authenticated before generic preparation. Every C
    primitive write is checked, every retained body matches original bytes, and
    the entire persistent IMEM is decoded/re-encoded and executed from PC zero.
    Original plans remain authoritative for memory and immutable-footprint guards.
    """
    def __init__(self, atlas_root, model_root, images, output, **options):
        self._executing = False
        super().__init__(atlas_root, model_root, images, output, **options)
        self.residency = NativeProgramResidency(atlas_root, images, self.output / "residency-native",
                                               busy=lambda: self._executing)
        self.mapped_programs = {}
        # Existing selected-core sentinel: observable JAL link and delay slot.
        words = (0x00000613, 0x0080056f, 0x00100613, 0x00200613,
                 0x00400613, 0x00c505b3, 0xc1059073, 0x00000073)
        sentinel = self._decode(b"".join(word.to_bytes(4, "little") for word in words), allow_csr=True)
        result = SelectedDelayInterpreter(replace(self.functional_config, dram_size=32)).run(sentinel)
        self.delay_sentinel = {key: result.state.read_xrf(index)
                               for key, index in (("x10", 10), ("x12", 12), ("x11", 11))}
        if self.delay_sentinel != dict(x10=2, x12=1, x11=3):
            self.residency.finish()
            raise AssertionError("selected-core JAL delay-slot sentinel differs")
        self.before = authenticate_model(model_root)

    def _decode(self, raw, *, allow_csr=False):
        from tools.check_program_decode import decode_binary
        decoded = decode_binary(self.inventory, raw)
        entries = []
        for index, row in enumerate(decoded["instructions"]):
            instruction = self.model_instruction(row, self.inventory, self.isa)
            if not allow_csr and instruction.mnemonic.startswith("csrr") and instruction.rd != 0:
                raise AssertionError("functional residency refuses CSR timing observations")
            entries.append(FunctionalInstruction(index, instruction, index + 1, str(instruction)))
        if b"".join(entry.instruction.to_bytecode().to_bytes(4, "little") for entry in entries) != raw:
            raise AssertionError("decoded complete physical IMEM differs from actual bytes")
        return FunctionalProgram(entries)

    def write(self, _, address, word):
        # These are source capsule writes, deliberately separate from physical
        # MMIO emitted by the current C policy/target session below.
        if 0x20000 <= address < 0x40000:
            if self.running or (address - 0x20000) % 4:
                self.errors.append("unaligned or active source IMEM write")
                return
            if address == 0x20000:
                self.imem.clear()
            self.imem[(address - 0x20000) // 4] = word
            return
        if address != 0x40018 or word != 1 or self.running:
            self.errors.append(f"unexpected launch CSR {address:#x}:{word}")
            return
        self.starts += 1
        self.polls, self.running = 0, True
        try:
            raw = b"".join(self.imem[i].to_bytes(4, "little") for i in range(len(self.imem)))
            digest = sha256(raw).hexdigest()
            self._current_image = digest
            if digest not in self.images or self.images[digest][0] != raw:
                raise AssertionError("source launch image differs from compiled publication")
            combined, placement = self.residency.prepare(raw)
            mapped_digest = placement["mapped_image_sha256"]
            if mapped_digest not in self.mapped_programs:
                self.mapped_programs[mapped_digest] = self._decode(combined)
            active_bytes = self.active_bytes[digest]
            initial = bytes(memoryview(self.memory)[:active_bytes])
            self._executing = True
            try:
                result = SelectedDelayInterpreter(replace(self.functional_config, dram_size=active_bytes),
                    math_proxy=self.math_proxy, memory_factory=self.memory_factory).run(
                    self.mapped_programs[mapped_digest], max_instructions=self.max_instructions,
                    dram={self.base: initial},
                    vmem=None if self.memory_factory is not None else {0: b'\xcd' * self.functional_config.vmem_size})
            finally:
                self._executing = False
            if not result.state.halted or result.halt_reason not in ("ebreak", "ecall"):
                raise AssertionError("physical mapped program failed to halt normally")
            actual = result.state.read_dram(self.base, active_bytes).numpy().tobytes()
            check_immutable_memory(initial, actual, self.ranges[digest])
            self.memory[:active_bytes] = actual
            hashes = (dict(active_window_base=self.base, active_window_bytes=active_bytes,
                           active_memory_before_sha256=sha256(initial).hexdigest(),
                           active_memory_after_sha256=sha256(actual).hexdigest())
                      if self.compact_dram_prefix else dict(memory_before_sha256=sha256(initial).hexdigest(),
                                                           memory_after_sha256=sha256(actual).hexdigest()))
            self.launches.append(dict(index=self.starts, image_sha256=digest,
                instructions=result.instructions_executed, final_pc=result.final_pc,
                halt_reason=result.halt_reason, residency=placement, **hashes))
            self._instructions += result.instructions_executed
            if self.starts % 128 == 0:
                self._refresh_progress("running")
        except Exception as error:
            self.errors.append(repr(error))
            self._refresh_progress("failed")

    def finish(self):
        residency = self.residency.finish()
        record = super().finish()
        record.update(program_residency=residency, control_flow_adapter="selected-core one delay slot",
                      delay_slot_sentinel=self.delay_sentinel,
                      mapped_execution="full persistent physical IMEM decoded and executed from PC zero",
                      residency_bridge_source_sha256=sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
        (self.output / "runtime.json").write_text(json.dumps(record, indent=2) + "\n")
        return record
