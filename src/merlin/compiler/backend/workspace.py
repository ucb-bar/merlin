"""Stateless C calls using bounded, source-backed caller tensor workspace.

The v1/v2 contracts stay unchanged. This third contract gives the ordinary
pointer call a workspace pointer as context, without persistent state or loaders.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from merlin.compiler.backend.cabi import CRegionABI, TensorPort

_WIDTH = {"i8": 1, "i16": 2, "f16": 2, "bf16": 2, "i32": 4, "f32": 4, "i64": 8, "f64": 8}


@dataclass(frozen=True)
class TensorWorkspace:
    sources: tuple[tuple[int, int], ...]
    ports: tuple[TensorPort, ...]
    offsets: tuple[int, ...]
    bytes: int
    alignment: int

    def __post_init__(self):
        if (type(self.sources) is not tuple or not self.sources or len(set(self.sources)) != len(self.sources)
                or any(type(pair) is not tuple or len(pair) != 2
                       or any(type(n) is not int or n < 0 for n in pair) for pair in self.sources)
                or type(self.ports) is not tuple or len(self.ports) != len(self.sources)
                or any(type(port) is not TensorPort for port in self.ports)
                or type(self.offsets) is not tuple or len(self.offsets) != len(self.ports)):
            raise ValueError("tensor workspace needs exact source results and ordered static ports")
        if (type(self.alignment) is not int or self.alignment <= 0 or self.alignment & (self.alignment - 1)
                or type(self.bytes) is not int or self.bytes <= 0 or self.bytes % self.alignment):
            raise ValueError("tensor workspace needs positive bounded aligned storage")
        end = 0
        for port, offset in zip(self.ports, self.offsets, strict=True):
            port.__post_init__()
            if (type(offset) is not int or offset < end or offset % self.alignment
                    or self.alignment < _WIDTH[port.dtype]):
                raise ValueError("tensor workspace ports overlap or lack dense storage alignment")
            end = offset + prod(port.shape) * _WIDTH[port.dtype]
        if end > self.bytes:
            raise ValueError("tensor workspace ports exceed their reserved storage")

    def record(self):
        self.__post_init__()
        return {"sources": [list(pair) for pair in self.sources],
                "ports": [{"shape": list(port.shape), "dtype": port.dtype} for port in self.ports],
                "offsets": list(self.offsets), "bytes": self.bytes, "alignment": self.alignment}

    @classmethod
    def pack(cls, sources, ports):
        """One canonical dense layout; no lifetime alias reuse or hardware estimate."""
        sources, ports = tuple(sources), tuple(ports)
        alignment = max(_WIDTH[port.dtype] for port in ports)
        offsets, cursor = [], 0
        for port in ports:
            cursor = (cursor + alignment - 1) // alignment * alignment
            offsets.append(cursor)
            cursor += prod(port.shape) * _WIDTH[port.dtype]
        size = (cursor + alignment - 1) // alignment * alignment
        return cls(sources, ports, tuple(offsets), size, alignment)

    @classmethod
    def from_record(cls, raw):
        if type(raw) is not dict or set(raw) != {"sources", "ports", "offsets", "bytes", "alignment"}:
            raise ValueError("tensor workspace requires its exact versioned fields")
        if (type(raw["sources"]) not in (list, tuple) or type(raw["ports"]) not in (list, tuple)
                or type(raw["offsets"]) not in (list, tuple)
                or any(type(row) is not dict or set(row) != {"shape", "dtype"}
                       or type(row["shape"]) not in (list, tuple) for row in raw["ports"])):
            raise ValueError("tensor workspace requires exact dense port and source rosters")
        return cls(tuple(tuple(pair) for pair in raw["sources"]),
                   tuple(TensorPort(tuple(row["shape"]), row["dtype"]) for row in raw["ports"]),
                   tuple(raw["offsets"]), raw["bytes"], raw["alignment"])


@dataclass(frozen=True)
class CWorkspaceABI:
    call: CRegionABI
    workspace: TensorWorkspace

    def __post_init__(self):
        if type(self.call) is not CRegionABI or type(self.workspace) is not TensorWorkspace:
            raise ValueError("workspace ABI requires original call and source tensor layout")
        self.call.__post_init__()
        self.workspace.__post_init__()
        if self.call.payloads or self.call.resource_kind != "cpu":
            raise ValueError("stateless source workspace currently requires CPU calls without binary payloads")

    def __getattr__(self, name):
        if name in CRegionABI.__dataclass_fields__:
            return getattr(self.call, name)
        raise AttributeError(name)

    def record(self):
        self.__post_init__()
        return {**self.call.record(), "schema": "merlin.c_call.v3", "workspace": self.workspace.record()}

    @classmethod
    def from_compilation(cls, compilation):
        compilation.verify()
        raw = compilation.metadata.get("c_call")
        if (type(raw) is not dict or set(raw) != set(CRegionABI.__dataclass_fields__) | {"schema", "workspace"}
                or raw.get("schema") != "merlin.c_call.v3"):
            raise ValueError("compiled region has no supported source-workspace C call contract")
        base = {key: value for key, value in raw.items() if key != "workspace"}
        base["schema"] = "merlin.c_call.v1"
        call = CRegionABI.from_compilation(replace(compilation, metadata={**compilation.metadata, "c_call": base}))
        return cls(call, TensorWorkspace.from_record(raw["workspace"]))
