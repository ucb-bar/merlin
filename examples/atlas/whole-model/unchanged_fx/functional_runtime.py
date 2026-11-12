"""Generated physical instructions through the pinned model's RTL semantics.

This backend accounts for functional instructions only. Timing, scheduling,
physical allocation, and binary emission remain separate compiler checks.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import fields, replace
from hashlib import sha256
import json
from pathlib import Path

import torch
from npu_model.functional import FunctionalConfig, FunctionalInterpreter, FunctionalResult, FunctionalState
from npu_model.functional.program import FunctionalInstruction, FunctionalProgram

from model_runtime import (ModelRuntime as _CycleModelRuntime, authenticate_model,
                           output_ranges, active_dram_prefix, check_immutable_memory)


class PhysicalFunctionalState(FunctionalState):
    """Physical register ABI over the model's virtual functional state."""

    def read_xrf(self, register):
        return 0 if register == 0 else super().read_xrf(register)

    def write_xrf(self, register, value):
        if register != 0:
            super().write_xrf(register, value)


class PhysicalFunctionalInterpreter(FunctionalInterpreter):
    """Use original instruction objects with a physical x0 state adapter.

    The model interpreter has no state factory hook. This loop keeps its
    instruction-indexed PC and instruction limit, while creating the physical
    state above. Optional qualified adapters are recorded separately from the
    original arithmetic implementation; decoded instruction execution is shared.
    """

    def __init__(self, config=None, *, math_proxy=None, memory_factory=None):
        super().__init__(config)
        self.math_proxy, self.memory_factory = math_proxy, memory_factory

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
        pc, executed = 0, 0
        while 0 <= pc < len(program.instructions):
            if executed >= max_instructions:
                raise RuntimeError(f"functional instruction limit ({max_instructions}) reached at PC {pc}")
            entry = program.instructions[pc]
            state.execute_pc, state.npc, state.redirect_requested = pc, pc + 1, False
            # The original instruction object executes without rewritten
            # operands. Optional state adapters have separate qualification.
            entry.instruction.exec(state)
            executed += 1
            if state.halted:
                pc += 1
                break
            pc = state.npc if state.redirect_requested else pc + 1
        if pc < 0 or pc > len(program.instructions):
            raise RuntimeError(f"functional control flow reached invalid PC {pc}")
        return FunctionalResult(state, executed, pc, state.halt_reason)




class FunctionalModelRuntime(_CycleModelRuntime):
    """ModelRuntime callback ABI, with original physical instruction execution."""

    def __init__(self, atlas_root: Path, model_root: Path, images: list[bytes], output: Path,
                 *, memory_bytes=262144, image_plans=None, max_instructions=1_000_000, trace_payloads=True, compact_dram_prefix=None,
                 math_proxy=None, memory_factory=None, execution_metadata=None):
        if type(max_instructions) is not int or max_instructions <= 0:
            raise ValueError("max_instructions must be a positive integer")
        if type(trace_payloads) is not bool:
            raise ValueError("trace_payloads must be Boolean")
        if (math_proxy is not None or memory_factory is not None) and not execution_metadata:
            raise ValueError("execution adapters require explicit qualified metadata")
        self.math_proxy, self.memory_factory = math_proxy, memory_factory
        self.execution_metadata = execution_metadata
        super().__init__(atlas_root, model_root, images, output, memory_bytes=memory_bytes, image_plans=image_plans)
        self.trace_payloads = trace_payloads
        self.compact_dram_prefix = not trace_payloads if compact_dram_prefix is None else compact_dram_prefix
        if type(self.compact_dram_prefix) is not bool or (trace_payloads and self.compact_dram_prefix):
            raise ValueError("compact_dram_prefix is available only for compact transfer traces")
        self._instructions = 0
        self._current_image = None
        if set(self.image_plans) != set(self.images):
            raise AssertionError("functional execution requires every compiled image's output footprint")
        self.image_plans = json.loads(json.dumps(self.image_plans))
        self.ranges = {digest: output_ranges(plan, self.base, len(self.memory)) for digest, plan in self.image_plans.items()}
        self.active_bytes = {digest: active_dram_prefix(plan,self.base,len(self.memory)) if self.compact_dram_prefix else len(self.memory)
                             for digest,plan in self.image_plans.items()}
        hardware = self.config_type().arch_state_config
        values = {item.name: getattr(hardware, item.name) for item in fields(FunctionalConfig)}
        values.update(dram_base=self.base, dram_size=len(self.memory), numerics="rtl")
        self.functional_config = FunctionalConfig(**values)
        self.max_instructions = max_instructions
        self.programs = {}
        for digest, (raw, decoded) in self.images.items():
            entries = []
            for index, row in enumerate(decoded['instructions']):
                instruction = self.model_instruction(row, self.inventory, self.isa)
                if instruction.mnemonic.startswith('csrr') and instruction.rd != 0:
                    raise AssertionError("functional backend refuses CSR counters or timing observations")
                entries.append(FunctionalInstruction(index, instruction, index+1, str(instruction)))
            if b''.join(entry.instruction.to_bytecode().to_bytes(4,'little') for entry in entries) != raw:
                raise AssertionError("functional model instructions differ from compiled image bytes")
            self.programs[digest] = FunctionalProgram(entries)
        # Functional modules are imported before this inventory is pinned.
        self.before = authenticate_model(model_root)
        self._refresh_progress("ready")

    def _refresh_progress(self, status):
        record = dict(starts=self.starts,instructions=self._instructions,transfercount=len(self.transfers),
                      errors=list(self.errors),currentimage=self._current_image,status=status)
        temporary = self.output/'.progress.json.tmp'
        temporary.write_text(json.dumps(record,sort_keys=True)+'\n')
        temporary.replace(self.output/'progress.json')

    def _transfer(self, operation, address, value, count):
        if self.trace_payloads:
            super()._transfer(operation,address,value,count)
        else:
            if self.running:
                raise AssertionError("host memory transfer precedes device completion")
            offset = address-self.base
            if offset < 0 or count < 0 or offset+count > len(self.memory):
                raise AssertionError("host transfer exceeds declared memory")
            if operation == "write":
                payload = ct.string_at(value,count)
                self.memory[offset:offset+count] = payload
            else:
                payload = bytes(self.memory[offset:offset+count])
                ct.memmove(value,payload,count)
            self.transfers.append(dict(index=len(self.transfers),operation=operation,address=address,
                bytes=count,after_launch=self.starts,sha256=sha256(payload).hexdigest(),path=None))
        if operation == "read" and self.starts and self.starts % 128 == 0:
            self._refresh_progress("running")

    def write(self, _, address, word):
        if 0x20000 <= address < 0x40000:
            if (address-0x20000)%4:
                self.errors.append("unaligned instruction memory write")
                return
            if address == 0x20000:
                self.imem.clear()
            self.imem[(address-0x20000)//4] = word
            return
        if address != 0x40018 or word != 1 or self.running:
            self.errors.append(f"unexpected launch CSR {address:#x}:{word}")
            return
        self.starts += 1
        self.polls, self.running = 0, True
        try:
            raw = b''.join(self.imem[index].to_bytes(4,'little') for index in range(len(self.imem)))
            digest = sha256(raw).hexdigest()
            self._current_image = digest
            if digest not in self.images or self.images[digest][0] != raw:
                raise AssertionError("launched image differs from compiled publication")
            active_bytes = self.active_bytes[digest]
            initial = bytes(memoryview(self.memory)[:active_bytes])
            config = replace(self.functional_config,dram_size=active_bytes)
            result = PhysicalFunctionalInterpreter(config, math_proxy=self.math_proxy,
                memory_factory=self.memory_factory).run(
                self.programs[digest], max_instructions=self.max_instructions,
                dram={self.base: initial},
                vmem=None if self.memory_factory is not None else {0: b'\xcd'*self.functional_config.vmem_size})
            state = result.state
            if not state.halted or result.halt_reason not in ('ebreak','ecall'):
                raise AssertionError("generated image failed to halt normally")
            actual = state.read_dram(self.base,active_bytes).numpy().tobytes()
            check_immutable_memory(initial,actual,self.ranges[digest])
            self.memory[:active_bytes] = actual
            memory_hashes = (dict(active_window_base=self.base,active_window_bytes=active_bytes,
                active_memory_before_sha256=sha256(initial).hexdigest(),active_memory_after_sha256=sha256(actual).hexdigest())
                if self.compact_dram_prefix else dict(memory_before_sha256=sha256(initial).hexdigest(),
                                                      memory_after_sha256=sha256(actual).hexdigest()))
            self.launches.append(dict(index=self.starts,image_sha256=digest,
                instructions=result.instructions_executed,final_pc=result.final_pc,halt_reason=result.halt_reason,**memory_hashes))
            self._instructions += result.instructions_executed
            if self.starts % 128 == 0:
                self._refresh_progress("running")
        except Exception as error:
            self.errors.append(repr(error))
            try:
                self._refresh_progress("failed")
            except Exception as progress_error:
                self.errors.append("progress update failed: "+repr(progress_error))

    def finish(self):
        after = authenticate_model(self.model_root)
        if any(after.get(key) != value for key,value in self.before.items()):
            self.errors.append("instruction-model source changed during execution")
        record = dict(backend="npu_model.functional.physical",numerics="rtl",model_source_sha256=after,
            model_revision="b97b2f5a97e432141c4ef8504bec396651d77a8f",physical_register_adapter="hardwired scalar x0",
            arithmetic="unchanged original decoded instruction.exec and RTL numerics",
            timing="not modeled; stream, timing, allocation, and emission checks remain separate",
            errors=self.errors,launches=self.launches,transfers=self.transfers,trace_payloads=self.trace_payloads,
            compact_dram_prefix=self.compact_dram_prefix,host_memory_bytes=len(self.memory),
            memory_audit="per-image active DRAM prefix with hard model bounds" if self.compact_dram_prefix else "whole declared DRAM window",
            active_window_bytes=self.active_bytes,
            total_instructions=sum(item['instructions'] for item in self.launches),
            image_plan_sha256={digest:sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                               for digest,plan in self.image_plans.items()})
        if self.execution_metadata is not None:
            record.update(backend="npu_model.functional.physical.qualified_native",
                arithmetic=("original decoded instruction exec with separately qualified native MXU1 arithmetic"
                            if self.math_proxy is not None else "original decoded instruction exec and RTL numerics"),
                memory_representation="qualified uniform sparse VMEM" if self.memory_factory is not None else "original sparse VMEM with explicit fill",
                execution_qualification={**self.execution_metadata,
                    **({"arithmetic": self.math_proxy.execution_metadata()}
                       if self.math_proxy is not None else {}),
                    "native_dot_calls": getattr(self.math_proxy, "calls", 0)})
        self._refresh_progress("failed" if self.errors else "finished")
        (self.output/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
        if self.errors:
            raise AssertionError(self.errors)
        return record


# The callback API also supports changing only the imported module name.
ModelRuntime = FunctionalModelRuntime
