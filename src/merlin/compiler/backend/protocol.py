"""Source-independent implementation proposal records for backend authors.

These are capability and lowering claims. They carry no source planner,
selected-source admission, target facts or independent verification authority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath

from merlin.common.digest import is_sha256


def validator_identity(callback):
    """Bind a selected Python callback without importing a compiler driver."""
    function = getattr(callback, "__func__", callback)
    code = getattr(function, "__code__", None)
    if code is None:
        raise ValueError("implementation validator must be an original Python function or bound method")
    return callback, function, code, getattr(callback, "__self__", None)


@dataclass(frozen=True)
class LoweringRecipe:
    """An exact provider implementation claim, not numerical verification."""

    implementation_sha256: str
    pipeline: str
    guards: tuple[tuple[str, str], ...]
    artifacts: tuple[tuple[str, str], ...]
    abi: str

    def __post_init__(self):
        if (
            type(self.implementation_sha256) is not str
            or len(self.implementation_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.implementation_sha256)
        ):
            raise ValueError("lowering recipe requires an exact implementation SHA-256")
        if any(type(v) is not str or not v for v in (self.pipeline, self.abi)):
            raise ValueError("lowering recipe requires explicit pipeline and ABI identities")
        if (
            type(self.guards) is not tuple
            or not self.guards
            or any(
                type(row) is not tuple or len(row) != 2 or any(type(v) is not str or not v for v in row)
                for row in self.guards
            )
            or len(dict(self.guards)) != len(self.guards)
        ):
            raise ValueError("lowering recipe requires immutable named guard identities")
        if (
            type(self.artifacts) is not tuple
            or not self.artifacts
            or any(
                type(row) is not tuple or len(row) != 2 or any(type(v) is not str or not v for v in row)
                for row in self.artifacts
            )
            or len(set(self.artifacts)) != len(self.artifacts)
        ):
            raise ValueError("lowering recipe requires explicit unique artifact kind/role pairs")


@dataclass(frozen=True)
class ProposalBudget:
    """Independent output and work-session bounds for trusted in-process hooks.

    Sessions bound matcher/checker invocations, not their instruction count or
    wall time. Process-backed untrusted compilers still need runner limits.
    """

    max_candidates: int = 256
    max_match_visits: int = 128
    max_proof_sessions: int = 384
    max_region_operations: int = 16
    max_source_operations: int = 10000
    max_tensor_elements: int = 1_000_000

    def __post_init__(self):
        limits = {
            "max_candidates": (1, 1024),
            "max_match_visits": (1, 512),
            "max_proof_sessions": (1, 1536),
            "max_region_operations": (1, 64),
            "max_source_operations": (1, 10000),
            "max_tensor_elements": (1, (1 << 63) - 1),
        }
        for name, (low, high) in limits.items():
            if type(getattr(self, name)) is not int or not low <= getattr(self, name) <= high:
                raise ValueError(f"proposal discovery requires bounded {name}")


@dataclass(frozen=True)
class SourcePattern:
    id: str
    root_operations: tuple[str, ...]
    pipeline: str
    alternatives_per_anchor: int

    def __post_init__(self):
        if (
            any(type(v) is not str or not v for v in (self.id, self.pipeline))
            or type(self.root_operations) is not tuple
            or not self.root_operations
            or len(set(self.root_operations)) != len(self.root_operations)
            or any(type(v) is not str or not v for v in self.root_operations)
            or type(self.alternatives_per_anchor) is not int
            or not 1 <= self.alternatives_per_anchor <= 16
        ):
            raise ValueError("source pattern requires explicit identities and finite anchor alternatives")


@dataclass(frozen=True)
class ProposalContext:
    source_sha256: str
    selected_contract_sha256: str
    numerical_policy_sha256: str
    preparation_sha256: str
    topology_sha256: str
    provider: str
    implementation_sha256: str
    patterns: tuple[SourcePattern, ...]
    resource: str
    memory_domain: str
    budget: ProposalBudget

    def __post_init__(self):
        for name in (
            "source_sha256",
            "selected_contract_sha256",
            "numerical_policy_sha256",
            "preparation_sha256",
            "topology_sha256",
            "implementation_sha256",
        ):
            if not is_sha256(getattr(self, name)):
                raise ValueError(f"proposal discovery requires pinned {name}")
        if (
            any(type(v) is not str or not v for v in (self.provider, self.resource, self.memory_domain))
            or type(self.budget) is not ProposalBudget
            or type(self.patterns) is not tuple
            or not self.patterns
            or len(self.patterns) > 64
            or any(type(p) is not SourcePattern for p in self.patterns)
            or len({p.id for p in self.patterns}) != len(self.patterns)
        ):
            raise ValueError("proposal discovery requires an immutable bounded provider context")

    def record(self):
        return asdict(self)


# Synchronous dense pointer-call declarations; host context admission stays outside this module.


@dataclass(frozen=True)
class TensorPort:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        if type(self.shape) is not tuple or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("C call ports require static positive tensor extents")
        if type(self.dtype) is not str or self.dtype not in {"i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64"}:
            raise ValueError("C call port has no admitted dense whole-byte representation")

    @classmethod
    def from_type(cls, typ):
        from xdsl.dialects.builtin import NoneAttr, TensorType

        if type(typ) is not TensorType or type(typ.encoding) is not NoneAttr:
            raise ValueError("C call ABI requires unencoded dense tensors")
        return cls(tuple(typ.get_shape()), str(typ.element_type))


@dataclass(frozen=True)
class CRegionABI:
    """``int symbol(void *, const void *const *, void *const *)``.

    Context is unused. Inputs are read-only, disjoint outputs are fully written,
    and zero means all work and required visibility operations have completed.
    Ports are dense row-major values in the selected host's byte order; packing
    and launch/readout adaptation belong inside the selected backend wrapper.
    Link inputs and binary payloads must be actual compilation artifacts.
    """

    symbol: str
    inputs: tuple[TensorPort, ...]
    outputs: tuple[TensorPort, ...]
    buffer_alignment: int
    link_inputs: tuple[str, ...]
    payloads: tuple[str, ...] = ()
    completion: str = "synchronous"
    scratch_bytes: int = 0
    resource_kind: str = field(kw_only=True)

    def __post_init__(self):
        if type(self.resource_kind) is not str or not self.resource_kind.strip():
            raise ValueError("C call must declare the implementation resource kind")
        letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"
        if (
            type(self.symbol) is not str
            or not self.symbol
            or self.symbol[0] not in letters
            or any(c not in letters + "0123456789" for c in self.symbol)
        ):
            raise ValueError("C call requires an explicit C symbol")
        for ports in (self.inputs, self.outputs):
            if type(ports) is not tuple or any(type(port) is not TensorPort for port in ports):
                raise ValueError("C call requires ordered immutable tensor ports")
            for port in ports:
                port.__post_init__()
        alignment = self.buffer_alignment
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("C call requires a positive power-of-two alignment")
        if self.completion != "synchronous" or type(self.scratch_bytes) is not int or self.scratch_bytes != 0:
            raise ValueError("C call route requires synchronous completion and zero additional host scratch")
        if not self.link_inputs:
            raise ValueError("C call requires a concrete wrapper link input")
        for paths in (self.link_inputs, self.payloads):
            if type(paths) is not tuple or len(paths) != len(set(paths)):
                raise ValueError("C call artifact paths must be immutable and unique")
            for path in paths:
                if (
                    type(path) is not str
                    or not path
                    or "\\" in path
                    or ":" in path
                    or PurePosixPath(path).is_absolute()
                    or ".." in PurePosixPath(path).parts
                    or PurePosixPath(path).as_posix() != path
                ):
                    raise ValueError("C call artifacts require portable relative paths")
        if set(self.link_inputs) & set(self.payloads):
            raise ValueError("C call payloads and link inputs must be distinct")

    def record(self):
        self.__post_init__()
        return {"schema": "merlin.c_call.v1", **asdict(self)}

    @classmethod
    def from_compilation(cls, compilation):
        compilation.verify()
        raw = compilation.metadata.get("c_call")
        fields = {
            "schema",
            "symbol",
            "inputs",
            "outputs",
            "buffer_alignment",
            "link_inputs",
            "payloads",
            "completion",
            "scratch_bytes",
            "resource_kind",
        }
        if type(raw) is not dict or set(raw) != fields or raw["schema"] != "merlin.c_call.v1":
            raise ValueError("compiled region has no supported explicit C call contract")

        def ports(values):
            if not isinstance(values, (list, tuple)):
                raise ValueError("C call ports must be an ordered sequence")
            result = []
            for value in values:
                if type(value) is not dict or set(value) != {"shape", "dtype"}:
                    raise ValueError("malformed C call tensor port")
                if not isinstance(value["shape"], (list, tuple)):
                    raise ValueError("C call tensor shape must be an ordered sequence")
                result.append(TensorPort(tuple(value["shape"]), value["dtype"]))
            return tuple(result)

        for key in ("link_inputs", "payloads"):
            if not isinstance(raw[key], (list, tuple)):
                raise ValueError("C call artifact paths must be an ordered sequence")
        result = cls(
            raw["symbol"],
            ports(raw["inputs"]),
            ports(raw["outputs"]),
            raw["buffer_alignment"],
            tuple(raw["link_inputs"]),
            tuple(raw["payloads"]),
            raw["completion"],
            raw["scratch_bytes"],
            resource_kind=raw["resource_kind"],
        )
        products = {a.path.relative_to(compilation.directory).as_posix(): a for a in compilation.artifacts}
        for path in (*result.link_inputs, *result.payloads):
            if path not in products:
                raise ValueError("C call refers to a product outside its compilation artifacts")
        for path in result.link_inputs:
            if products[path].kind not in {"c-source", "object", "archive"}:
                raise ValueError("C call link input needs C source or an admitted object/archive; lower IR first")
        return result
