"""Numerics contracts: the declared arithmetic a program's golden is computed under.

A contract is a target-neutral definition of how an integer accumulator becomes the stored output:
the scale granularity, the rounding, the saturation bounds and the activation. Goldens are generated
UNDER a contract, and a program is graded against the golden of the contract it declares. Grading a
per-tensor program against a per-channel golden fails for reasons that have nothing to do with the
program, and grading a device epilogue against a host integer chain is impossible in principle (the
device rounds an f32 product; the host chain does not).

Contracts are built from a target's DERIVED readout facts (:func:`from_readout_facts`), i.e. the dict
a target's own header-verified readout contract produces. The clamp bounds, dtypes and rounding come
from there, never from a literal here.

Readout arithmetic, exactly as a scalar C readout macro computes it on an FLT_EVAL_METHOD=0 host:

    prod = f32(acc) * f32(scale)        # one IEEE-754 single multiply
    y    = roundeven(prod)              # round half to even
    out  = clamp(y, clamp_min, clamp_max)
    out  = max(out, 0) if activation == "relu"

``numpy`` float32 arithmetic and ``numpy.rint`` reproduce each step bit-for-bit for |prod| < 2**63,
which covers every accumulator of the declared width times a finite scale of magnitude < 2**31.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

#: How the scale varies over the accumulator tile. ``row``/``column`` are one scale per accumulator
#: row / column of a 2-D tile; ``rank1`` is an outer product of a row vector and a column vector.
GRANULARITIES = ("tensor", "row", "column", "rank1")
#: Roundings a contract may declare. Only the one a readout contract has been verified to implement.
ROUNDINGS = ("half_even",)
#: The readout-facts schema this module knows how to consume.
READOUT_FACTS_SCHEMA = "scalar_narrow_readout_contract_v1"


class ContractError(ValueError):
    """A contract cannot be built, or a readout cannot be evaluated under it."""


def _int_dtype(spelling: str) -> np.dtype:
    """``i8``/``i16``/``i32``/``i64`` -> the numpy signed integer dtype. Fail closed on anything else."""
    if not isinstance(spelling, str) or not spelling.startswith("i"):
        raise ContractError(f"not a signed integer dtype spelling: {spelling!r}")
    bits = spelling[1:]
    if not bits.isdigit() or int(bits) not in (8, 16, 32, 64):
        raise ContractError(f"unsupported integer width in {spelling!r}")
    return np.dtype(f"int{int(bits)}")


@dataclass(frozen=True)
class NumericsContract:
    """One declared readout arithmetic. Two contracts with equal :meth:`digest` compute identical bits."""

    contract_id: str
    accumulator_dtype: str
    output_dtype: str
    scale_dtype: str
    scale_granularity: str
    rounding: str
    clamp_min: int
    clamp_max: int
    activations: tuple[str, ...] = ("none", "relu")
    #: Where the facts came from (header digests, macro names). Recorded, but NOT part of the
    #: semantics: two sources that derive the same arithmetic produce the same digest.
    provenance: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.scale_granularity not in GRANULARITIES:
            raise ContractError(f"unknown scale granularity {self.scale_granularity!r}")
        if self.rounding not in ROUNDINGS:
            raise ContractError(f"unknown rounding {self.rounding!r}")
        if self.scale_dtype != "f32":
            raise ContractError(f"only an f32 scale is declared, got {self.scale_dtype!r}")
        acc = _int_dtype(self.accumulator_dtype)
        out = _int_dtype(self.output_dtype)
        if not self.clamp_min < self.clamp_max:
            raise ContractError("clamp_min must be below clamp_max")
        info = np.iinfo(out)
        if self.clamp_min < info.min or self.clamp_max > info.max:
            raise ContractError(f"clamp [{self.clamp_min}, {self.clamp_max}] exceeds {self.output_dtype}")
        if np.iinfo(acc).bits < info.bits:
            raise ContractError("accumulator narrower than output")
        for act in self.activations:
            if act not in ("none", "relu"):
                raise ContractError(f"unknown activation {act!r}")

    def semantic_fields(self) -> dict[str, Any]:
        """Every field that changes the computed bits (i.e. all but provenance)."""
        return {
            "contract_id": self.contract_id,
            "accumulator_dtype": self.accumulator_dtype,
            "output_dtype": self.output_dtype,
            "scale_dtype": self.scale_dtype,
            "scale_granularity": self.scale_granularity,
            "rounding": self.rounding,
            "clamp_min": self.clamp_min,
            "clamp_max": self.clamp_max,
            "activations": list(self.activations),
        }

    def digest(self) -> str:
        """sha256 of the canonical semantic fields. A golden records this; the grader compares it."""
        blob = json.dumps(self.semantic_fields(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _scale_operand(self, acc: np.ndarray, scale) -> np.ndarray:
        s = np.asarray(scale, dtype=np.float32)
        if self.scale_granularity == "tensor":
            if s.size != 1:
                raise ContractError(f"a per-tensor contract takes one scale, got shape {s.shape}")
            return s.reshape(())
        if self.scale_granularity == "rank1":
            raise ContractError(
                "rank1 readout arithmetic is declared but its product order is not yet fixed; refuse rather than guess"
            )
        if acc.ndim != 2:
            raise ContractError(f"a per-{self.scale_granularity} readout needs a 2-D tile, got {acc.shape}")
        if self.scale_granularity == "row":
            if s.shape != (acc.shape[0],):
                raise ContractError(f"row scales must have shape ({acc.shape[0]},), got {s.shape}")
            return s[:, None]
        if s.shape != (acc.shape[1],):
            raise ContractError(f"column scales must have shape ({acc.shape[1]},), got {s.shape}")
        return s[None, :]

    def readout(self, acc, scale, *, activation: str = "none") -> np.ndarray:
        """Evaluate the readout on an integer accumulator tile. Returns ``output_dtype`` values."""
        if activation not in self.activations:
            raise ContractError(f"activation {activation!r} not admitted by {self.contract_id}")
        a = np.asarray(acc)
        if a.dtype.kind not in "iu":
            raise ContractError(f"accumulator must be an integer array, got {a.dtype}")
        info = np.iinfo(_int_dtype(self.accumulator_dtype))
        if a.size and (a.min() < info.min or a.max() > info.max):
            raise ContractError(f"accumulator value outside {self.accumulator_dtype}")
        s = self._scale_operand(a, scale)
        if not np.all(np.isfinite(s)):
            raise ContractError("non-finite scale")
        prod = a.astype(np.float32) * s  # one float32 multiply, IEEE round-to-nearest
        y = np.rint(prod)  # round half to even, still float32
        y = np.clip(y, self.clamp_min, self.clamp_max)
        if activation == "relu":
            y = np.maximum(y, 0)
        return y.astype(_int_dtype(self.output_dtype))


def from_readout_facts(readout: Mapping[str, Any], *, contract_id: str, granularity: str) -> NumericsContract:
    """Build a contract from a target's derived readout facts.

    ``readout`` is the dict a target's header-verified readout contract returns (schema
    ``scalar_narrow_readout_contract_v1``): accumulator/output/scale dtypes and clamp bounds. That
    schema is only produced after the target's scale and rounding macros were matched against the
    round-half-to-even construction, which is what licenses ``rounding="half_even"`` here.
    """
    if readout.get("schema") != READOUT_FACTS_SCHEMA:
        raise ContractError(f"unsupported readout facts schema {readout.get('schema')!r}")
    try:
        return NumericsContract(
            contract_id=contract_id,
            accumulator_dtype=str(readout["accumulator_dtype"]),
            output_dtype=str(readout["output_dtype"]),
            scale_dtype=str(readout["scale_dtype"]),
            scale_granularity=granularity,
            rounding="half_even",
            clamp_min=int(readout["clamp_min"]),
            clamp_max=int(readout["clamp_max"]),
            provenance=dict(readout.get("provenance") or {}),
        )
    except KeyError as missing:
        raise ContractError(f"readout facts lack {missing}") from None


#: Readout-level contracts, keyed by id. Model-level contracts (bias seeding, residual add, pooling)
#: compose one of these with further stages and are declared separately.
_BUILDERS: dict[str, Callable[[Mapping[str, Any]], NumericsContract]] = {
    "per_tensor_readout_v1": lambda r: from_readout_facts(r, contract_id="per_tensor_readout_v1", granularity="tensor"),
    "per_row_readout_v1": lambda r: from_readout_facts(r, contract_id="per_row_readout_v1", granularity="row"),
    "per_column_readout_v1": lambda r: from_readout_facts(r, contract_id="per_column_readout_v1", granularity="column"),
}


def known_contracts() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def contract(contract_id: str, readout: Mapping[str, Any]) -> NumericsContract:
    """The named contract, instantiated from one target's readout facts."""
    try:
        build = _BUILDERS[contract_id]
    except KeyError:
        raise ContractError(f"unknown contract {contract_id!r}; known: {known_contracts()}") from None
    return build(readout)
