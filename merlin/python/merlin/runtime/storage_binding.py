"""Data-format-only caller bindings for explicit logical/physical storage contracts.

No operation interpretation, casting, quantization, or tensor algebra occurs here. The
caller supplies its dtype's storage words after resolving logical input values; the
encoding only relocates those words. An explicit map is complete, never a sparse hint.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any

from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.perf.structural_transitions import _element_bytes
from merlin.runtime.tensor import Tensor


class StoragePrepackRequired(ValueError):
    """An exact, inspectable obligation; a candidate's weight role is not authorization."""
    def __init__(self, obligations: Mapping[str, Any]):
        self.storage_obligations = dict(obligations)
        super().__init__("explicit caller input requires_prepack without host immutable-input authorization: "
                         + ", ".join(obligations))


@dataclass(frozen=True)
class StorageBinding:
    encoding: GroupedAxesStorage
    access: str
    logical_values: tuple[Any, ...] | None
    prepack_authorization: Any = None

    @property
    def requires_prepack(self) -> bool:
        """Whether non-unit logical axes change order, rather than merely adding padding."""
        active = [(axis, stride) for axis, (extent, stride) in enumerate(zip(
            self.encoding.logical_shape, self.encoding.logical_strides_elements, strict=True))
            if extent > 1]
        return [axis for axis, _ in active] != [axis for axis, _ in sorted(active, key=lambda pair: -pair[1])]

    def setup_evidence(self) -> dict[str, Any]:
        return {"schema": "caller_storage_setup_v1", "encoding": self.encoding.to_dict(),
            "logical_copy_elements": prod(self.encoding.logical_shape),
            "requires_prepack": self.requires_prepack and self.access != "write",
            "packing_inside_compute_roi": False,
            "prepack_authorization": (self.prepack_authorization.to_evidence()
                if self.prepack_authorization is not None else
                "UNPROVEN" if self.requires_prepack and self.access != "write" else "not_required"),
            "scope": "format setup/readback only; no model arithmetic or timing qualification"}

    def pack_words(self, words: Sequence[int]) -> list[int]:
        """Copy already encoded scalar storage words; retain each word unchanged."""
        encoding = self.encoding
        if len(words) != prod(encoding.logical_shape):
            raise ValueError("logical storage-word count disagrees with the encoding")
        words = tuple(words)
        if self.prepack_authorization is not None:
            width = _element_bytes(encoding.dtype)
            if width not in (1, 2, 4, 8) or any(type(word) is not int for word in words):
                raise ValueError("authorized prepack requires exact whole-byte integer storage words")
            modulus = 1 << (8 * width)
            if any(not -(modulus // 2) <= word < modulus for word in words):
                raise ValueError("authorized prepack storage word exceeds declared dtype width")
            payload = b"".join((word % modulus).to_bytes(width, "little") for word in words)
            self.prepack_authorization.validate_payload(payload)
        result = [0] * encoding.storage_elements
        strides = encoding.logical_strides_elements
        for word, index in zip(words, product(*(range(d) for d in encoding.logical_shape)), strict=True):
            offset = encoding.offset_elements + sum(i*s for i, s in zip(index, strides, strict=True))
            result[offset] = word
        return result

    def logical_offset_terms(self) -> tuple[tuple[int, int, int], ...]:
        """(row-major linear divisor, logical extent, physical stride) for readback."""
        shape = self.encoding.logical_shape
        return tuple((prod(shape[axis+1:]), extent, stride) for axis, (extent, stride) in
                     enumerate(zip(shape, self.encoding.logical_strides_elements, strict=True)))


def _logical_values(value: Any, shape: tuple[int, ...], dtype: str, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, Tensor):
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"logical input {name!r} Tensor shape/dtype disagrees with storage encoding")
        return tuple(value.data)
    def visit(node, dims):
        if not dims:
            if isinstance(node, (list, tuple, Mapping)):
                raise ValueError(f"logical input {name!r} has a non-scalar leaf")
            return [node]
        if not isinstance(node, (list, tuple)) or len(node) != dims[0]:
            raise ValueError(f"logical input {name!r} shape disagrees with storage encoding")
        return [item for child in node for item in visit(child, dims[1:])]
    return tuple(visit(value, shape))


def resolve_storage_bindings(cb: Mapping[str, Any], inputs: Mapping[str, Any] | None = None,
                             *, max_storage_bytes: int,
                             reference_only_allow_prepack: bool = False,
                             prepack_authorizations: Mapping[str, Any] | None = None) -> dict[str, StorageBinding] | None:
    """Validate all storage before materializing inputs, or return None for the legacy ABI.

    ``max_storage_bytes`` is a host resource bound, not a claim about target capacity.
    Physical tensor descriptors retain unpadded extents; allocated strides/offset/padding
    come exclusively from the checked explicit encoding.

    ``reference_only_allow_prepack`` permits testing the data-format copy itself; it is
    NOT an authorization to exclude candidate-selected input work from a timed invocation.
    ``prepack_authorizations`` is an out-of-band host grant, never command-buffer
    metadata. Binding is checked here; the exact initializer words are checked by
    ``pack_words`` before relocation. Missing authorized inputs never materialize
    from deterministic names.
    """
    params = cb.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("whole-program params must be a mapping")
    if "storage_encodings" not in params:
        if prepack_authorizations is not None:
            raise ValueError("host prepack authorization requires explicit storage encodings")
        return None
    if type(max_storage_bytes) is not int or max_storage_bytes <= 0:
        raise ValueError("explicit storage requires a positive host allocation budget")
    # Derived arithmetic belongs in the submitted compiler, never in this format-only path.
    from merlin.runtime.commandbuffer import DERIVATION_RECIPE_KEYS
    if any(params.get(key) for key in DERIVATION_RECIPE_KEYS):
        raise ValueError("explicit storage cannot move tensor derivation into the caller")
    records, tensors, abi = params["storage_encodings"], cb.get("tensors"), cb.get("kernel_abi")
    if not isinstance(records, Mapping) or not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("explicit storage requires a nonempty complete tensor map")
    if not isinstance(abi, Mapping) or abi.get("kind") != "whole_program":
        raise ValueError("explicit storage caller requires the whole_program pointer ABI")
    args, outputs = abi.get("args"), abi.get("outputs")
    if not isinstance(args, list) or not isinstance(outputs, list) or not outputs:
        raise ValueError("explicit storage requires declared arguments and outputs")
    access = {}
    for arg in args:
        if (not isinstance(arg, Mapping) or not isinstance(arg.get("tensor"), str)
                or arg.get("access") not in ("read", "write", "readwrite")
                or arg["tensor"] in access):
            raise ValueError("explicit storage has malformed or duplicate pointer arguments")
        access[arg["tensor"]] = arg["access"]
    if set(records) != set(tensors) or set(access) != set(tensors):
        raise ValueError("storage encodings must cover exactly every tensor and ABI argument")
    if (any(not isinstance(name, str) or name not in access or access[name] == "read" for name in outputs)
            or len(outputs) != len(set(outputs))):
        raise ValueError("explicit storage outputs must name distinct writable arguments")
    encodings, allocated = {}, 0
    for name, spec in tensors.items():
        if not isinstance(spec, Mapping):
            raise ValueError("explicit storage requires typed tensor descriptors")
        encoding = GroupedAxesStorage.from_dict(records[name])
        if (not isinstance(spec.get("shape"), list)
                or any(type(dim) is not int or dim <= 0 for dim in spec["shape"])
                or spec["shape"] != list(encoding.physical_shape) or spec.get("dtype") != encoding.dtype):
            raise ValueError(f"tensor {name!r} physical shape/dtype disagrees with storage encoding")
        if (spec.get("role") in ("input", "weight", "bias", "scale") and access[name] == "write"
                or spec.get("role") in ("output", "intermediate") and access[name] == "read"):
            raise ValueError("tensor role contradicts storage argument access")
        allocated += encoding.storage_elements * _element_bytes(encoding.dtype)
        if allocated > max_storage_bytes:
            raise ValueError("explicit storage exceeds the host aggregate allocation budget")
        encodings[name] = encoding
    prepack = {name: StorageBinding(encoding, access[name], None).setup_evidence()
               for name, encoding in encodings.items()
               if access[name] != "write" and StorageBinding(encoding, access[name], None).requires_prepack}
    provided = {} if inputs is None else inputs
    if not isinstance(provided, Mapping) or set(provided) - set(tensors):
        raise ValueError("explicit logical inputs must name declared tensors")
    authorizations = {} if prepack_authorizations is None else prepack_authorizations
    if not isinstance(authorizations, Mapping) or set(authorizations) - set(tensors):
        raise ValueError("host prepack authorizations must name declared tensors")
    if authorizations:
        from merlin.runtime.prepack_authority import HostPrepackAuthorization
        for name, authorization in authorizations.items():
            if type(authorization) is not HostPrepackAuthorization:
                raise ValueError("prepack requires an exact host authorization object, not candidate metadata")
            if access[name] != "read" or name not in provided:
                raise ValueError("authorized prepack requires an explicit read-only logical input")
            authorization.validate_binding(cb, name, encodings[name])
    missing = {name: obligation for name, obligation in prepack.items() if name not in authorizations}
    if missing and not reference_only_allow_prepack:
        raise StoragePrepackRequired(missing)
    result = {}
    for name, encoding in encodings.items():
        values = None
        if access[name] == "write":
            if name in provided:
                raise ValueError("write-only output cannot be initialized from supplied inputs")
        elif name in provided:
            values = _logical_values(provided[name], encoding.logical_shape, encoding.dtype, name=name)
        else:
            values = tuple(Tensor.deterministic(name, encoding.logical_shape, encoding.dtype).data)
        result[name] = StorageBinding(encoding, access[name], values, authorizations.get(name))
    return result
