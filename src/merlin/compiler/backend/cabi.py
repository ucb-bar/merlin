"""Versioned backend claims for the shared synchronous C pointer-array ABI.

This record binds products and ordered dense tensor ports. It does not verify
the implementation's arithmetic, target instructions, or execution. The initial
ABI is stateless and requires no additional host scratch or retained pointers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath


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
