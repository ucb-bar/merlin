"""Explicit caller-owned contexts for synchronous C calls with device payloads.

These are custody and reservation records, not arithmetic, loader-execution or
hardware certificates. The stateless v1 reader and emitted records are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import PurePosixPath

from merlin.common.digest import is_sha256
from merlin.compiler.backend.cabi import CRegionABI


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identifier(value):
    letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    return (
        type(value) is str and bool(value) and value[0] in letters and all(c in letters + "0123456789" for c in value)
    )


def _member(value):
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or ":" in value
        or PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
        or PurePosixPath(value).as_posix() != value
    ):
        raise ValueError("context and payload members require portable relative paths")


def _alignment(value):
    return type(value) is int and value > 0 and value & (value - 1) == 0


@dataclass(frozen=True)
class CContext:
    """An explicit bound, discharged by the selected initializer/runtime.

    ``identity`` admits sharing only an identical selected binding. State and
    workspace belong to the named host resource/domain, never implicitly to a
    device-local domain. On error their owner retains them until quiescence.
    Sizes are reservation limits, not guessed native struct layouts.
    """

    identity: str
    contract_sha256: str
    resource: str
    memory_domain: str
    state_bytes: int
    state_alignment: int
    workspace_bytes: int
    workspace_alignment: int
    declaration: str
    ctype: str
    initialize: str
    ownership: str = "caller"
    lifetime: str = "until_quiescent"
    workspace_reuse: str = "serial_after_success"
    error: str = "retain_until_drained_or_reset"

    def __post_init__(self):
        for value in (self.identity, self.resource, self.memory_domain):
            if type(value) is not str or not value.strip():
                raise ValueError("C context needs explicit identity, host resource and memory domain")
        if not is_sha256(self.contract_sha256):
            raise ValueError("C context needs a selected implementation contract SHA-256")
        if (
            type(self.state_bytes) is not int
            or self.state_bytes <= 0
            or type(self.workspace_bytes) is not int
            or self.workspace_bytes < 0
            or not _alignment(self.state_alignment)
            or not _alignment(self.workspace_alignment)
        ):
            raise ValueError("C context needs bounded state/workspace bytes and power-of-two alignments")
        _member(self.declaration)
        if not _identifier(self.ctype) or not _identifier(self.initialize):
            raise ValueError("C context declaration type and initializer must be C identifiers")
        if (self.ownership, self.lifetime, self.workspace_reuse, self.error) != (
            "caller",
            "until_quiescent",
            "serial_after_success",
            "retain_until_drained_or_reset",
        ):
            raise ValueError("C context has no admitted ownership, serial reuse or failure lifetime")

    def record(self):
        self.__post_init__()
        return asdict(self)

    @classmethod
    def from_record(cls, value):
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("C context requires its exact versioned fields")
        return cls(**value)

    def selection_metadata(self):
        return "c.context", _json(self.record())


def selected_context(alternative):
    rows = tuple(value for key, value in alternative.metadata if key == "c.context")
    if not rows:
        return None
    if len(rows) != 1 or type(rows[0]) is not str:
        raise ValueError("selected context needs one canonical original requirement record")
    context = CContext.from_record(json.loads(rows[0]))
    if rows[0] != _json(context.record()):
        raise ValueError("selected context requirement must be canonical JSON without duplicate fields")
    if alternative.scratch_bytes != 0:
        raise ValueError("stateful host workspace must not also be charged as region scratch")
    return context


def context_reservations(selected, topology):
    """Unique persistent contexts plus maximum serial workspace per domain.

    Serial reuse is admitted explicitly by every original context requirement;
    the dispatcher stops with active state on error, preventing subsequent reuse.
    Actual initialization/loading/completion is still a runtime obligation.
    """
    contexts, domains = {}, {}
    for item in selected:
        context = selected_context(item.alternative)
        if context is None:
            continue
        if topology.resource(context.resource).kind != "cpu":
            raise ValueError("host C context requires an explicitly selected CPU execution resource")
        topology.require_access(context.resource, context.memory_domain, read=True, write=True)
        memory = topology.memory(context.memory_domain)
        if max(context.state_alignment, context.workspace_alignment) > memory.alignment_bytes:
            raise ValueError("context alignment exceeds its explicit host memory domain contract")
        previous = contexts.get(context.identity)
        if previous is not None and previous != context:
            raise ValueError("shared context identity has conflicting original requirements")
        if previous is not None:
            continue
        contexts[context.identity] = context
        row = domains.setdefault(context.memory_domain, {"state_bytes": 0, "workspace_bytes": 0})
        alignment = memory.alignment_bytes

        def rounded(n):
            return ((n + alignment - 1) // alignment) * alignment

        row["state_bytes"] += rounded(context.state_bytes)
        row["workspace_bytes"] = max(row["workspace_bytes"], rounded(context.workspace_bytes))
    for name, row in domains.items():
        row["total_bytes"] = row["state_bytes"] + row["workspace_bytes"]
        if row["total_bytes"] > topology.memory(name).capacity_bytes:
            raise ValueError("selected host contexts/workspace exceed their explicit memory domain capacity")
    return {"contexts": [context.record() for context in contexts.values()], "domains": domains}


def render_u32_payload(symbol, payload):
    """Canonical C data embedding of explicitly selected little-endian words.

    This transports bytes; no word is decoded or claimed to be an instruction.
    """
    if not _identifier(symbol) or type(payload) is not bytes or not payload or len(payload) % 4:
        raise ValueError("u32le payload requires a C identifier and nonempty whole words")
    words = tuple(int.from_bytes(payload[n : n + 4], "little") for n in range(0, len(payload), 4))
    lines = ["#include <stdint.h>", f"static const uint32_t {symbol}[{len(words)}] = {{"]
    lines.extend("  " + ", ".join(f"0x{word:08x}u" for word in words[n : n + 8]) + "," for n in range(0, len(words), 8))
    return "\n".join([*lines, "};", ""])


@dataclass(frozen=True)
class EmbeddedPayload:
    binary: str
    header: str
    symbol: str
    encoding: str = "u32le"

    def __post_init__(self):
        _member(self.binary)
        _member(self.header)
        if self.binary == self.header or not _identifier(self.symbol) or self.encoding != "u32le":
            raise ValueError("embedded payload requires distinct members, a C symbol and selected u32le encoding")

    @classmethod
    def from_record(cls, raw):
        if type(raw) is not dict or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("embedded payload requires its exact fields")
        return cls(**raw)


@dataclass(frozen=True)
class CRegionABI2:
    """Same pointer call as v1, with explicit context and embedded payload custody.

    Zero means synchronous completion/visibility and complete output values.
    Nonzero leaves the context, its workspace and device buffers retained. The
    caller must drain/reset through the selected platform before reuse. This
    receipt does not establish that initialization or a device launch occurred.
    """

    call: CRegionABI
    context: CContext
    embedded: tuple[EmbeddedPayload, ...]

    def __post_init__(self):
        if type(self.call) is not CRegionABI or type(self.context) is not CContext:
            raise ValueError("stateful C call needs original typed call/context records")
        self.call.__post_init__()
        self.context.__post_init__()
        if (
            type(self.embedded) is not tuple
            or not self.embedded
            or any(type(item) is not EmbeddedPayload for item in self.embedded)
        ):
            raise ValueError("stateful binary call requires explicit immutable embedded payloads")
        for item in self.embedded:
            item.__post_init__()
        if (
            tuple(item.binary for item in self.embedded) != self.call.payloads
            or len({item.header for item in self.embedded}) != len(self.embedded)
            or len({item.symbol for item in self.embedded}) != len(self.embedded)
        ):
            raise ValueError("stateful call payload list differs from exact unique embeddings")

    def __getattr__(self, name):
        # Explicit composition preserves the existing call fields without a
        # second symbol/port/artifact validator or inheritance-based ABI override.
        if name in CRegionABI.__dataclass_fields__:
            return getattr(self.call, name)
        raise AttributeError(name)

    def record(self):
        self.__post_init__()
        return {
            **self.call.record(),
            "schema": "merlin.c_call.v2",
            "context": self.context.record(),
            "embedded": [asdict(item) for item in self.embedded],
        }

    @classmethod
    def from_compilation(cls, compilation):
        compilation.verify()
        raw = compilation.metadata.get("c_call")
        fields = set(CRegionABI.__dataclass_fields__) | {"schema", "context", "embedded"}
        if type(raw) is not dict or set(raw) != fields or raw.get("schema") != "merlin.c_call.v2":
            raise ValueError("compiled region has no supported stateful C call contract")
        if type(raw["embedded"]) not in (tuple, list):
            raise ValueError("stateful embeddings must be an ordered sequence")
        v1 = {key: value for key, value in raw.items() if key not in {"context", "embedded"}}
        v1["schema"] = "merlin.c_call.v1"
        call = CRegionABI.from_compilation(replace(compilation, metadata={**compilation.metadata, "c_call": v1}))
        result = cls(
            call,
            CContext.from_record(raw["context"]),
            tuple(EmbeddedPayload.from_record(row) for row in raw["embedded"]),
        )
        result.__post_init__()
        artifacts = {item.path.relative_to(compilation.directory).as_posix(): item for item in compilation.artifacts}
        declaration = artifacts.get(result.context.declaration)
        if declaration is None or declaration.kind != "c-header":
            raise ValueError("stateful context declaration is not an actual C header artifact")
        for item in result.embedded:
            binary, header = artifacts.get(item.binary), artifacts.get(item.header)
            if binary is None or binary.kind != "device-binary" or header is None or header.kind != "c-header":
                raise ValueError("embedded payload needs actual device-binary and C header artifacts")
            if header.path.read_bytes() != render_u32_payload(item.symbol, binary.path.read_bytes()).encode():
                raise ValueError("embedded header differs from the exact selected binary bytes")
        return result


def read_c_abi(compilation):
    raw = compilation.metadata.get("c_call")
    if type(raw) is dict and raw.get("schema") == "merlin.c_call.v3":
        from merlin.compiler.backend.workspace import CWorkspaceABI

        return CWorkspaceABI.from_compilation(compilation)
    if type(raw) is dict and raw.get("schema") == "merlin.c_call.v2":
        return CRegionABI2.from_compilation(compilation)
    return CRegionABI.from_compilation(compilation)
