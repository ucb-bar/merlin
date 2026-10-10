"""Values-free prepared files for an explicitly selected native process.

The fixed original harness plans determine names, extents and the complete byte
budget. The selected external process owns ELF symbol resolution, loading and
readback semantics. This transport grants none of those semantics or phase roles.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common.quant_formats import get
from merlin.common.strict_json import loads
from merlin.runtime.direct_kernel_counter import DirectKernelCounterPlan
from merlin.runtime.direct_kernel_harness import DirectKernelAbi
from merlin.runtime.direct_kernel_invocation import DirectKernelInvocationPlan
from merlin.runtime.direct_kernel_phases import DirectKernelPhasePlan

from .build_service import file_digest

SCHEMA = "merlin.prepared_process_readback.v1"
OPERANDS = ("{request}", "{output}")


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _plain(path):
    if type(path) is not Path:
        path = Path(path)
    if not path.is_absolute() or path.resolve() != path or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("prepared process requires canonical unlinked members")
    return path


def _read(path, limit):
    path = _plain(path)
    if not path.is_file():
        raise ValueError("prepared process member is not an ordinary file")
    with path.open("rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
            raise ValueError("prepared process requires individually owned ordinary members")
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("prepared process member exceeds its original byte budget")
    return raw


def _metadata(value, limit):
    pending, nodes, scalar_bytes = [(value, 0)], 0, 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > 16384 or depth > 32:
            raise ValueError("prepared process original metadata exceeds its node/depth budget")
        if type(value) in (dict, list, tuple):
            if len(value) > 1024:
                raise ValueError("prepared process original metadata exceeds its member budget")
            if type(value) is dict:
                if any(type(key) is not str or len(key) > 1024 for key in value):
                    raise ValueError("prepared process original metadata has unsupported keys")
                scalar_bytes += sum(len(key.encode("utf-8")) for key in value)
                value = value.values()
            pending.extend((member, depth + 1) for member in value)
        elif type(value) is int:
            if value.bit_length() > 64:
                raise ValueError("prepared process original metadata exceeds its scalar budget")
            scalar_bytes += 24
        elif type(value) is str:
            if len(value) > 65536:
                raise ValueError("prepared process original metadata exceeds its scalar budget")
            scalar_bytes += len(value.encode("utf-8"))
        elif type(value) is float:
            if not math.isfinite(value):
                raise ValueError("prepared process original metadata has a non-finite scalar")
            scalar_bytes += 32
        elif type(value) not in (bool, type(None)):
            raise ValueError("prepared process original metadata has unsupported values")
        if scalar_bytes > limit:
            raise ValueError("prepared process original metadata exceeds its byte budget")


@dataclass(frozen=True)
class PreparedProcessReadbackPlan:
    """Explicit original typed plans; no expected values or target addresses.

    ``frame_bytes`` is an explicit source-owned wire selection, not inferred
    packet semantics. V1 transports one exact-size file; its contents still
    require the independently selected original complete decoder.
    """

    abi: DirectKernelAbi
    invocation_plan: DirectKernelInvocationPlan
    counter_plan: DirectKernelCounterPlan | None
    phase_plan: DirectKernelPhasePlan | None
    frame_bytes: int
    max_payload_bytes: int
    max_request_bytes: int

    def record(self):
        if (
            type(self) is not PreparedProcessReadbackPlan
            or type(self.abi) is not DirectKernelAbi
            or type(self.invocation_plan) is not DirectKernelInvocationPlan
            or self.counter_plan is not None
            and type(self.counter_plan) is not DirectKernelCounterPlan
            or self.phase_plan is not None
            and type(self.phase_plan) is not DirectKernelPhasePlan
            or self.phase_plan is not None
            and self.phase_plan.counter_plan != self.counter_plan
            or type(self.frame_bytes) is not int
            or not 0 <= self.frame_bytes <= 65536
            or type(self.max_payload_bytes) is not int
            or not 1 <= self.max_payload_bytes <= 4 * 1024 * 1024
            or type(self.max_request_bytes) is not int
            or not 1 <= self.max_request_bytes <= 1024 * 1024
        ):
            raise ValueError("prepared process requires exact original typed plans and explicit bounded budgets")
        original = self.invocation_plan.original_abi
        from .compile_only import CompileOnlySourceAbi, CompileOnlyTensor

        if type(original) is not CompileOnlySourceAbi or any(
            type(slots) is not tuple or len(slots) > 512 for slots in (original.inputs, original.outputs)
        ):
            raise ValueError("prepared process original ABI exceeds its member budget")
        for slot in (*original.inputs, *original.outputs):
            if type(slot) is not CompileOnlyTensor or type(slot.shape) is not tuple or len(slot.shape) > 64:
                raise ValueError("prepared process original ABI has unsupported shaped metadata")
            _metadata((slot.name, slot.shape, slot.dtype), self.max_request_bytes)
        self.abi.verify()
        record = {
            "schema": SCHEMA,
            "abi": asdict(self.abi),
            "invocations": self.invocation_plan.record(),
            "counters": None if self.counter_plan is None else self.counter_plan.record(),
            "phases": None if self.phase_plan is None else self.phase_plan.record(),
            "frame_bytes": self.frame_bytes,
            "max_payload_bytes": self.max_payload_bytes,
            "max_request_bytes": self.max_request_bytes,
            "scope": (
                "original object/file transport only; ELF resolution, memory, completion, timing and roles unqualified"
            ),
        }
        _metadata(record, self.max_request_bytes)
        return record

    def source_paths(self):
        # Known fixed derivation implementations only, not an import closure.
        from merlin.common.paths import module_source_path

        return tuple(
            module_source_path(name)
            for name in (
                __name__,
                "merlin.targetgen.contract.compile_only",
                "merlin.targetgen.contract.tensor_types",
                "merlin.common.strict_json",
                "merlin.common.quant_formats",
                "merlin.runtime.commandbuffer",
                "merlin.runtime.direct_kernel_harness",
                "merlin.runtime.direct_kernel_invocation",
                "merlin.runtime.direct_kernel_counter",
                "merlin.runtime.direct_kernel_phases",
            )
        )

    def bind(self, cb):
        record = self.record()
        # Bound original metadata before products, rows or shaped data. Scalar
        # integers have a finite width; no tensor payload is allocated here.
        _metadata(cb, self.max_request_bytes)
        if len(_json(cb)) > self.max_request_bytes or len(_json(record)) > self.max_request_bytes:
            raise ValueError("prepared process original metadata exceeds its byte budget")
        original = self.invocation_plan.original_abi
        original.bind(cb)
        histories = self.invocation_plan.bind(
            cb, entry_symbol=self.abi.entry_symbol, completion_symbol=self.abi.completion_symbol
        )
        objects = {}
        for index, argument in enumerate(cb["kernel_abi"]["args"]):
            spec = cb["tensors"][argument["tensor"]]
            dtype = get(spec["dtype"])
            if dtype.is_block_scaled or dtype.element_bits not in (8, 16, 32, 64):
                raise ValueError("prepared process requires byte-aligned original storage")
            objects["tensor_" + str(index)] = math.prod(spec["shape"]) * (dtype.element_bits // 8)
        objects |= {row.symbol: row.byte_extent for row in histories}
        objects[self.invocation_plan.count_symbol] = 8
        if self.counter_plan is not None:
            objects |= self.counter_plan.bind(cb, abi=self.abi, invocation_plan=self.invocation_plan)
        if self.phase_plan is not None:
            objects |= self.phase_plan.bind(cb, abi=self.abi, invocation_plan=self.invocation_plan)
        total = sum(objects.values())
        if (
            not objects
            or len(objects) > 1024
            or any(extent <= 0 for extent in objects.values())
            or total > self.max_payload_bytes
        ):
            raise ValueError("prepared process complete original roster exceeds its byte/object budget")
        return tuple(objects.items())

    def wire(self, cb, elf_sha256):
        objects = self.bind(cb)
        wire = {
            "schema": SCHEMA,
            "elf_sha256": elf_sha256,
            "command_buffer_sha256": hashlib.sha256(_json(cb)).hexdigest(),
            "plan": self.record(),
            "objects": [{"symbol": name, "bytes": extent} for name, extent in objects],
            "payload_bytes": sum(extent for _, extent in objects),
            "product_bytes": sum(extent for _, extent in objects) + self.frame_bytes,
        }
        if len(_json(wire)) > self.max_request_bytes:
            raise ValueError("prepared process request exceeds its original metadata budget")
        return wire

    def prepare(self, *, cb, elf_path, workdir):
        elf_path, workdir = _plain(elf_path), _plain(workdir)
        if not workdir.is_dir() or not elf_path.is_file():
            raise ValueError("prepared process requires an ordinary ELF and preparation directory")
        wire = self.wire(cb, file_digest(elf_path))
        owner = Path(tempfile.mkdtemp(prefix="prepared_readback_", dir=workdir))
        cb_path, request_path = owner / "original_cb.json", owner / "request.json"
        for path, raw in ((cb_path, _json(cb)), (request_path, _json(wire))):
            with path.open("xb") as output:
                output.write(raw)
        request = {
            "schema": SCHEMA,
            "elf_sha256": wire["elf_sha256"],
            "cb_path": str(cb_path),
            "request_path": str(request_path),
            "output_path": str(owner / "readback.bin"),
            "request_sha256": hashlib.sha256(_json(wire)).hexdigest(),
        }
        validate_request(self, request, elf_path, completed=False)
        return {"memory_readback": request}


def validate_request(plan, request, elf, *, completed):
    """Recompute the original roster; supplied sizes/rows never grant admission."""
    if (
        type(plan) is not PreparedProcessReadbackPlan
        or type(request) is not dict
        or set(request) != {"schema", "elf_sha256", "cb_path", "request_path", "output_path", "request_sha256"}
        or request["schema"] != SCHEMA
    ):
        raise ValueError("prepared process received an unsupported closed request")
    plan.record()
    paths = tuple(_plain(request[name]) for name in ("cb_path", "request_path", "output_path"))
    owner = paths[0].parent
    if len(set(paths)) != 3 or any(path.parent != owner for path in paths) or not owner.is_dir():
        raise ValueError("prepared process request members do not have one distinct private owner")
    if owner.stat().st_uid != os.getuid() or owner.stat().st_mode & 0o077:
        raise ValueError("prepared process request owner is not private and current-user owned")
    digest = file_digest(_plain(elf))
    if request["elf_sha256"] != digest:
        raise ValueError("prepared process request differs from its actual selected ELF")
    cb = loads(_read(paths[0], plan.max_request_bytes).decode("utf-8"))
    wire = plan.wire(cb, digest)
    expected = _json(wire)
    if (
        _read(paths[1], plan.max_request_bytes) != expected
        or request["request_sha256"] != hashlib.sha256(expected).hexdigest()
    ):
        raise ValueError("prepared process request differs from its original typed roster")
    if completed:
        raw = _read(paths[2], wire["product_bytes"])
        if len(raw) != wire["product_bytes"]:
            raise ValueError("prepared process omitted complete original readback bytes")
    elif paths[2].exists() or paths[2].is_symlink():
        raise ValueError("prepared process output must be absent before native execution")
    return wire, paths
