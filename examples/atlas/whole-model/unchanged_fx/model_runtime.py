"""Run generated host C callbacks against the unchanged instruction model."""

from __future__ import annotations

import ctypes as ct
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import torch


READ = ct.CFUNCTYPE(ct.c_uint32, ct.c_void_p, ct.c_size_t)
WRITE = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_size_t, ct.c_uint32)
MEMORY = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t)
MODEL_PIN = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"


def authenticate_model(root):
    result = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not (name == "npu_model" or name.startswith("npu_model.")) or not filename:
            continue
        path = Path(filename)
        relative = path.relative_to(root)
        payload = path.read_bytes()
        pinned = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob",
                                          f"{MODEL_PIN}:{relative}"])
        if payload != pinned:
            raise AssertionError("instruction-model source differs from pin")
        result[str(relative)] = sha256(payload).hexdigest()
    if not result:
        raise AssertionError("model source inventory is empty")
    return result


_ACT_FRAMES = {"atlas.exact_float.v1", "atlas.activation_quantize_vector_plan.v1"}


def _validate_act_frame(plan, base, memory_bytes):
    """Check authored combined result/status or code/exponent publications."""
    def integer(key):
        value = plan.get(key)
        if type(value) is not int or value <= 0:
            raise AssertionError("invalid ACT frame integer: " + key)
        return value

    def aligned(value):
        return (value + 31) // 32 * 32

    schema = plan["schema"]
    forbidden = ("status_bytes", "exponent_addresses", "exponent_buffer_bytes")
    if schema == "atlas.exact_float.v1":
        widths = {"bf16_to_f32": (2, 4), "f32_to_bf16": (4, 2),
                  "bf16_neg": (2, 2), "f32_neg": (4, 4),
                  "bf16_add": (2, 2), "bf16_sub": (2, 2)}
        operation = plan.get("operation")
        if operation not in widths:
            raise AssertionError("unknown exact-float operation")
        width, output_width = widths[operation]
        logical = integer("logical_output_bytes")
        if logical % output_width or not 0 < logical // output_width < 1 << 27:
            raise AssertionError("invalid exact-float logical extent")
        count = logical // output_width
        lanes = min(1024, 2048 // max(width, output_width), aligned(count))
        padded = (count + lanes - 1) // lanes * lanes
        addresses = plan.get("input_addresses")
        if (not isinstance(addresses, list) or not addresses
                or len(addresses) > (2 if operation in {"bf16_add", "bf16_sub"} else 1)):
            raise AssertionError("exact float needs its complete source operands")
        sizes = [padded * width] * len(addresses)
        output = base + sum(sizes)
        result = padded * output_width
        if (integer("result_buffer_bytes") != result
                or plan.get("status_address") != output + result
                or type(plan.get("status_address")) is not int):
            raise AssertionError("exact float status differs from its combined 32-byte publication")
        output_size = result + 32
    else:
        forbidden += ("status_address",)
        width = {"bf16": 2, "f32": 4}.get(plan.get("input_dtype"))
        count = integer("count")
        group, head, row = (integer(key) for key in ("group_elements", "head_elements", "row_elements"))
        if (width is None or count >= 1 << 28 or not row <= group <= head <= count
                or count % head or head % row or group % row):
            raise AssertionError("invalid activation-quantizer geometry")
        sizes = [aligned(count) * width]
        output = base + sizes[0]
        codes = aligned(count)
        # The schema supports both per-head and repeated per-row exponents.
        # It omits the reduce spelling, so admit only their exact padded sizes.
        if plan.get("mode") == "static":
            exponent_sizes = {32}
        elif plan.get("mode") == "dynamic":
            exponent_sizes = {aligned(count // row * 4)}
            if head == count and count % group == 0:
                exponent_sizes.add(aligned(count // group * 4))
        else:
            raise AssertionError("unknown activation-quantizer mode")
        exponent_size = integer("exponent_bytes")
        if (exponent_size not in exponent_sizes or integer("input_address") != base
                or integer("input_bytes") != sizes[0] or integer("code_address") != output
                or integer("code_bytes") != codes or integer("exponent_address") != output + codes):
            raise AssertionError("activation-quantizer code/exponent frame differs")
        output_size = codes + exponent_size
    addresses = plan.get("input_addresses")
    expected = [base + sum(sizes[:index]) for index in range(len(sizes))]
    if (any(key in plan for key in forbidden) or addresses != expected
            or any(type(value) is not int for value in addresses)
            or plan.get("input_payload_bytes") != sizes
            or any(type(value) is not int for value in plan.get("input_payload_bytes", []))
            or plan.get("output_addresses") != [output]
            or any(type(value) is not int for value in plan.get("output_addresses", []))
            or integer("output_bytes") != output_size
            or integer("workspace_bytes") != output + output_size - base
            or not base <= output < output + output_size <= min(base + memory_bytes, 1 << 32)):
        raise AssertionError("ACT combined frame differs from its exact aligned publication")


def restoration_buffers(plan, base, memory_bytes):
    """Read exactly declared readonly exponent pages and mutable status bytes."""
    if plan.get("schema") in _ACT_FRAMES:
        _validate_act_frame(plan, base, memory_bytes)
        # Both families already include their complete mutable tail in output.
        return [], []
    if (plan.get("schema") != "atlas.quantized_loop_execution.v1"
            and any(key in plan for key in ("status_address", "status_bytes", "exponent_addresses", "exponent_buffer_bytes"))):
        raise AssertionError("unknown restoration buffer schema")
    readonly, mutable = [], []
    exponent_keys = ("exponent_addresses", "exponent_buffer_bytes")
    if any(key in plan for key in exponent_keys):
        addresses, sizes = (plan.get(key) for key in exponent_keys)
        if (not isinstance(addresses, list) or not isinstance(sizes, list)
                or len(addresses) != 2 or len(sizes) != 2):
            raise AssertionError("restoration requires two explicit exponent buffers")
        readonly = list(zip(addresses, sizes, strict=True))
    status_keys = ("status_address", "status_bytes")
    if any(key in plan for key in status_keys):
        address, size = (plan.get(key) for key in status_keys)
        if type(size) is not int or size != 2048:
            raise AssertionError("restoration status requires its exact 2048-byte publication")
        mutable = [(address, size)]
    for address, size in readonly + mutable:
        if (type(address) is not int or type(size) is not int or size <= 0
                or address % 32 or size % 32
                or not base <= address < address + size <= base + memory_bytes):
            raise AssertionError("restoration buffer leaves the aligned device memory window")
    all_ranges = sorted((address, address + size) for address, size in readonly + mutable)
    if any(left[1] > right[0] for left, right in zip(all_ranges, all_ranges[1:])):
        raise AssertionError("restoration buffers overlap")
    return readonly, mutable


def output_ranges(plan, base, memory_bytes):
    """Validate the exact publication's mutable DRAM windows."""
    if "output_addresses" in plan:
        size = plan.get("output_bytes")
        addresses = plan["output_addresses"]
        if type(size) is not int or size <= 0 or not isinstance(addresses, list) or not addresses:
            raise AssertionError("invalid published device output footprint")
        ranges = [(address-base, address-base+size) for address in addresses if type(address) is int]
        if len(ranges) != len(addresses):
            raise AssertionError("invalid published device output address")
    elif plan.get("schema") == "atlas.row_loop_execution_plan.v1":
        keys = ("output_base", "groups", "output_stride_bytes")
        if any(type(plan.get(key)) is not int for key in keys) or plan['groups'] <= 0 or plan['output_stride_bytes'] <= 0:
            raise AssertionError("invalid published row-loop footprint")
        first = plan['output_base']-base
        ranges = [(first, first+plan['groups']*plan['output_stride_bytes'])]
        if plan.get("scratch_bytes"):
            keys = ("scratch_address", "scratch_stride_bytes", "scratch_bytes")
            if any(type(plan.get(key)) is not int for key in keys) or plan['scratch_bytes'] <= 0 or plan['scratch_stride_bytes'] < 0:
                raise AssertionError("invalid published row-loop scratch footprint")
            first = plan['scratch_address']-base
            ranges.append((first, first+(plan['groups']-1)*plan['scratch_stride_bytes']+plan['scratch_bytes']))
    else:
        raise AssertionError("unknown generated device output footprint")
    readonly, status = restoration_buffers(plan, base, memory_bytes)
    validate_inputs = bool(readonly or status) or plan.get("schema") in _ACT_FRAMES
    originals = list(ranges)
    input_addresses = plan.get("input_addresses", [])
    input_sizes = plan.get("input_payload_bytes", plan.get("input_buffer_bytes", []))
    if validate_inputs and (not isinstance(input_addresses, list)
            or not isinstance(input_sizes, list) or len(input_addresses) != len(input_sizes)):
        raise AssertionError("restoration needs a complete original input footprint")
    original_inputs = list(zip(input_addresses, input_sizes))
    if validate_inputs:
        for address, size in original_inputs:
            if (type(address) is not int or type(size) is not int or size <= 0
                    or address % 32 or size % 32
                    or not base <= address < address + size <= base + memory_bytes):
                raise AssertionError("original input footprint leaves the aligned device memory window")
            first, last = address - base, address - base + size
            if any(first < end and begin < last for begin, end in originals):
                raise AssertionError("readonly original input aliases a mutable output")
    for address, size in readonly + status:
        first, last = address - base, address - base + size
        others = originals + [(a - base, a - base + n) for a, n in original_inputs]
        if any(first < end and begin < last for begin, end in others):
            raise AssertionError("restoration buffer aliases original operands or outputs")
    ranges += [(address - base, address - base + size) for address, size in status]
    for first, last in ranges:
        if not 0 <= first < last <= memory_bytes:
            raise AssertionError("published output footprint leaves device memory")
    return tuple(sorted(ranges))



def active_dram_prefix(plan, base, memory_bytes):
    """Bound the original interpreter to declared inputs, outputs, and scratch."""
    mutable = output_ranges(plan,base,memory_bytes)
    ends = [last for _,last in mutable]
    if plan.get('schema') == 'atlas.row_loop_execution_plan.v1':
        keys = ('readonly_base','readonly_stride_bytes','groups')
        if any(type(plan.get(key)) is not int for key in keys) or plan['readonly_stride_bytes'] <= 0:
            raise AssertionError('compact execution needs the declared row-loop readonly frame')
        inputs = [(plan['readonly_base'],plan['groups']*plan['readonly_stride_bytes'])]
    else:
        addresses = plan.get('input_addresses')
        sizes = plan.get('input_payload_bytes')
        if sizes is None:
            sizes = plan.get('input_buffer_bytes')
        if (not isinstance(addresses,list) or not isinstance(sizes,list) or len(addresses)!=len(sizes)):
            raise AssertionError('compact execution needs every declared input payload size')
        inputs = list(zip(addresses,sizes,strict=True))
    readonly, _ = restoration_buffers(plan, base, memory_bytes)
    inputs += readonly
    for address,size in inputs:
        if type(address) is not int or type(size) is not int or size <= 0 or not base <= address < address+size <= base+memory_bytes:
            raise AssertionError('declared input footprint leaves device memory')
        ends.append(address-base+size)
    return max(ends)


def check_immutable_memory(before, after, ranges):
    if len(before) != len(after):
        raise AssertionError("functional execution changed the device memory size")
    cursor = 0
    for first, last in ranges:
        if before[cursor:first] != after[cursor:first]:
            raise AssertionError("device changed memory outside its declared outputs and scratch")
        cursor = max(cursor, last)
    if before[cursor:] != after[cursor:]:
        raise AssertionError("device changed memory outside its declared outputs and scratch")


class ModelRuntime:
    def __init__(self, atlas_root: Path, model_root: Path, images: list[bytes], output: Path,
                 *, memory_bytes=262144, image_plans=None, max_cycles=100000):
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction

        self.authenticate = authenticate_model
        self.model_root = model_root
        self.before = self.authenticate(model_root)
        self.isa, self.config_type = isa, DefaultHardwareConfig
        self.logger_type, self.simulation_type = LoggerConfig, Simulation
        self.program_type, self.model_instruction = InstantiableProgram, model_instruction
        self.inventory = json.loads((atlas_root / "docs/selected-variant-inventory.json").read_text())
        self.images = {sha256(raw).hexdigest(): (raw, decode_binary(self.inventory, raw)) for raw in images}
        self.image_plans = image_plans or {}
        self.max_cycles = max_cycles
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.memory = bytearray([0xa5]) * memory_bytes
        self.imem = {}
        self.starts, self.polls, self.running = 0, 0, False
        self.errors, self.launches, self.transfers = [], [], []
        self.base = 0x90000000
        self.callbacks = READ(self.read), WRITE(self.write), MEMORY(self.put), MEMORY(self.get)

    def read(self, _, address):
        if address == 0x40004:
            return self.starts
        if address != 0x40008:
            self.errors.append(f"unexpected CSR read {address:#x}")
            return 3
        if not self.running:
            return 5
        self.polls += 1
        if self.polls == 1:
            return 0
        self.running = False
        return 3 if self.errors else 5

    def write(self, _, address, word):
        if 0x20000 <= address < 0x40000:
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
            raw = b"".join(self.imem[index].to_bytes(4, "little") for index in range(len(self.imem)))
            digest = sha256(raw).hexdigest()
            if digest not in self.images or self.images[digest][0] != raw:
                raise AssertionError("launched image differs from compiled publication")
            decoded = self.images[digest][1]
            program = self.program_type([self.model_instruction(row, self.inventory, self.isa)
                                         for row in decoded["instructions"]])
            program.dram_base = self.base
            initial = bytes(self.memory)
            program.memory_regions = [(0, torch.frombuffer(bytearray(initial), dtype=torch.uint8))]
            config = self.config_type()
            config.arch_state_config = replace(config.arch_state_config, numerics="rtl",
                                               randomize_init=True, init_seed=217)
            simulation = self.simulation_type(config, self.logger_type(
                filename=str(self.output / f"launch-{self.starts}.json")), program, verbose=False)
            state = simulation.core.arch_state
            try:
                state.vmem.fill_(0xcd)
                simulation.run(max_cycles=self.max_cycles)
                stats = simulation.get_stats()
                if not state.halted or stats.runtime_errors:
                    raise AssertionError("generated image failed to halt without runtime errors")
                actual = bytes(state.read_dram(self.base, len(self.memory)).tolist())
                if digest in self.image_plans:
                    plan = self.image_plans[digest]
                    ranges = output_ranges(plan, self.base, len(initial))
                    check_immutable_memory(initial, actual, ranges)
                self.memory[:] = actual
                self.launches.append({"index": self.starts, "image_sha256": digest,
                                      "cycles": stats.cycles,
                                      "memory_before_sha256": sha256(initial).hexdigest(),
                                      "memory_after_sha256": sha256(self.memory).hexdigest()})
            finally:
                state.close()
        except Exception as error:
            self.errors.append(repr(error))

    def _transfer(self, operation, address, value, count):
        if self.running:
            raise AssertionError("host memory transfer precedes device completion")
        offset = address - self.base
        if offset < 0 or count < 0 or offset + count > len(self.memory):
            raise AssertionError("host transfer exceeds declared memory")
        if operation == "write":
            payload = ct.string_at(value, count)
            self.memory[offset:offset + count] = payload
        else:
            payload = bytes(self.memory[offset:offset + count])
            ct.memmove(value, payload, count)
        index = len(self.transfers)
        member = f"transfer-{index}.bin"
        (self.output / member).write_bytes(payload)
        self.transfers.append({"index": index, "operation": operation, "address": address,
                               "bytes": count, "after_launch": self.starts,
                               "sha256": sha256(payload).hexdigest(), "path": member})

    def put(self, _, address, value, count):
        try:
            self._transfer("write", address, value, count)
            return 0
        except Exception as error:
            self.errors.append(repr(error))
            return 1

    def get(self, _, address, value, count):
        try:
            self._transfer("read", address, value, count)
            return 0
        except Exception as error:
            self.errors.append(repr(error))
            return 1

    def finish(self):
        after = self.authenticate(self.model_root)
        if any(after.get(key) != value for key, value in self.before.items()):
            self.errors.append("instruction-model source changed during execution")
        record = {"numerics": "rtl", "model_source_sha256": after, "errors": self.errors,
                  "launches": self.launches, "transfers": self.transfers,
                  "total_cycles": sum(item["cycles"] for item in self.launches)}
        (self.output / "runtime.json").write_text(json.dumps(record, indent=2) + "\n")
        if self.errors:
            raise AssertionError(self.errors)
        return record
