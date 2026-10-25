"""Exact rounded readout composition for spatial maximum proposals.

The two source quantizers remain distinct. A replacement is admitted only when
the complete signed-i32 transition partition agrees, including saturation and
both rounding boundaries. This is numerical evidence, never target admission.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from merlin.llvmlower.integer_readout import derive, evaluate
from merlin.llvmlower.requantization import quantized


@dataclass(frozen=True)
class RoundedReadout:
    scales: tuple[float, ...]
    relu: bool = False

    def __post_init__(self):
        if type(self.scales) is not tuple or type(self.relu) is not bool:
            raise ValueError("readout requires ordered scale constants and explicit activation")
        proof = derive(self.scales, relu=self.relu)
        object.__setattr__(self, "scales", tuple(proof["source_scales"]))


class ReadoutCompositionUnavailable(ValueError):
    """The proposed replacement changes an exact source value."""

    def __init__(self, witness):
        self.witness = witness
        super().__init__(
            f"rounded readout composition differs at accumulator {witness['accumulator']}: "
            f"source={witness['source']}, replacement={witness['replacement']}"
        )


@dataclass(frozen=True)
class SpatialReadoutChain:
    # None means the source producer really exposes raw i32 accumulators.
    producer: RoundedReadout | None
    pool: RoundedReadout
    replacement: RoundedReadout | None

    def __post_init__(self):
        if (
            self.producer is not None
            and type(self.producer) is not RoundedReadout
            or type(self.pool) is not RoundedReadout
            or self.replacement is not None
            and type(self.replacement) is not RoundedReadout
        ):
            raise ValueError("spatial readout requires exact typed source and proposed replacement stages")
        for stage in (self.producer, self.pool, self.replacement):
            if stage is not None:
                stage.__post_init__()

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def prove_spatial_readout(chain: SpatialReadoutChain):
    """Prove both quantizers against one proposed hardware readout on all i32.

    The first quantizer and replacement are constant between their complete
    monotone i8 transition points. Applying the second quantizer to that first
    i8 cannot add a transition elsewhere. Checking the union and immediate
    predecessors therefore covers every signed-i32 input, without samples.
    A positive readout is monotone, so exact maximum can commute with it; the
    caller still owns padding, source matching, observers and target semantics.
    """
    if type(chain) is not SpatialReadoutChain:
        raise ValueError("exact spatial readout chain required")
    chain.__post_init__()
    if chain.replacement is None:
        raise ValueError("single-readout proof requires an explicit proposed replacement")
    source_stage = chain.producer or chain.pool
    source = derive(source_stage.scales, relu=source_stage.relu)
    replacement = derive(chain.replacement.scales, relu=chain.replacement.relu)
    low, high = -(1 << 31), (1 << 31) - 1
    points = {low, high}
    for proof in (source, replacement):
        for transition in proof["thresholds"]:
            points.update(x for x in (transition - 1, transition) if low <= x <= high)
    for value in sorted(points):
        first = evaluate(value, source)
        expected = quantized(first, chain.pool.scales, chain.pool.relu) if chain.producer is not None else first
        actual = evaluate(value, replacement)
        if expected != actual:
            raise ReadoutCompositionUnavailable(
                dict(
                    accumulator=value,
                    intermediate=first,
                    source=expected,
                    replacement=actual,
                    chain_sha256=chain.identity,
                )
            )
    second_identity = all(quantized(value, chain.pool.scales, chain.pool.relu) == value for value in range(-128, 128))
    return dict(
        schema="merlin.spatial_readout_composition.v1",
        chain_sha256=chain.identity,
        domain=[low, high],
        checked_partition_endpoints=len(points),
        exact=True,
        producer_raw=chain.producer is None,
        second_readout_identity=second_identity,
        source_scales=None if chain.producer is None else list(chain.producer.scales),
        pool_scales=list(chain.pool.scales),
        replacement_scales=list(chain.replacement.scales),
        proof="complete union of source and replacement monotone transition partitions",
        scope="numerical equivalence only; no source, placement, ABI or execution authority",
    )
