"""What a mixed-lane program costs: bytes moved, and which lane moves them.

The compute-axis sibling of this module is :mod:`merlin.perf.work_volume` (MACs) and the
descriptor-axis sibling is :mod:`merlin.perf.dma_volume` (DMA bytes actually described). Neither
answers the question this one does: for a program the compiler split across an accelerator lane and
a host lane, **how many bytes does the program have to move through DRAM, and how much of that exists
only because work landed on the host lane?**

WHY IT EXISTS. On a target whose accelerator reduces MACs but not bytes, bytes decide runtime: this
tree has measured streaming/movement ops punished 89-202x against a contraction's 2.9x on real
silicon. A compiler change that folds a per-output epilogue into the accelerator's own readout does
not change a single MAC, so every compute-axis metric reports it as a no-op, while it can delete an
entire accumulator round trip. That delta is what this module makes visible -- and it is derivable
statically from the emitted command buffer in milliseconds, with no simulator, which is what makes it
usable as an inner-loop objective rather than a post-hoc report.

DERIVED, NEVER ASSUMED. Element widths come from parsing the dtype token the compiler itself wrote;
roles and lanes come from the command buffer's own vocabulary. No target name, no default width, no
assumed dtype set appears here. An unparseable dtype or an absent lane record makes the affected
quantity UNKNOWN and the totals an explicit LOWER BOUND -- it never substitutes a guess, because a
byte total silently completed by a default is worse than no total: it reads as a measurement.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["dtype_bits", "TensorBytes", "LaneCost", "lane_cost", "format_report"]

#: Roles whose tensors are live only inside one invocation, so each one is written once and read
#: back at least once. Everything else (a parameter, an entry input, the output) is touched once.
#: These are command-buffer ABI role names, not target facts.
_ROUND_TRIP_ROLES = frozenset({"intermediate", "temporary", "scratch"})


def dtype_bits(token: Any) -> int | None:
    """Element width in bits for a dtype token, or ``None`` when it cannot be derived.

    Parses STRUCTURALLY (a leading kind letter/word then a bit count), so a spelling this tree has
    not seen still resolves if it carries its width, and one that does not is honestly UNKNOWN.
    Deliberately not a lookup table of the dtypes we happen to use today: a table silently maps an
    unlisted dtype to nothing, and a caller that defaults the miss to 4 bytes mis-sizes every f64 and
    every fp8 tensor in the same direction -- flatteringly.
    """
    if not isinstance(token, str) or not token:
        return None
    name = token.strip().lower()
    # Strip a trailing float-format suffix (e.g. "f8e4m3fn" -> the leading "f8"): the width is the
    # first run of digits after the kind, and the rest names the exponent/mantissa split.
    for kind in ("bf", "fp", "f", "i", "si", "ui", "u", "int", "uint", "float", "bfloat"):
        if not name.startswith(kind):
            continue
        rest = name[len(kind):]
        digits = ""
        for ch in rest:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            bits = int(digits)
            return bits if bits > 0 else None
    return None


@dataclass(frozen=True)
class TensorBytes:
    """One tensor's derived footprint, or an honest refusal for it."""

    name: str
    role: str
    dtype: str
    elements: int | None
    bytes: int | None
    refusal: str | None = None


@dataclass(frozen=True)
class LaneCost:
    """Bytes and lane accounting for one program. ``is_lower_bound`` is load-bearing."""

    tensors: tuple[TensorBytes, ...]
    #: (role, dtype) -> bytes, for every tensor whose footprint was derivable.
    bytes_by_role_dtype: Mapping[tuple[str, str], int]
    footprint_bytes: int
    #: DRAM traffic per invocation, counting round-trip roles twice and others once.
    #:
    #: ⚠️ This is a lower bound ONLY for tensors that are streamed in full. It is an UPPER bound for
    #: one that is randomly accessed -- an embedding table read by token lookup touches `seq_len`
    #: rows, not all of them, and nothing in a command buffer says which access pattern a host-side
    #: input gets. MEASURED consequence: a 2.6B-parameter model's f32 embedding table is
    #: 2,250 MiB here and is referenced by no accelerator command at all, so treating that figure as
    #: traffic overstates it by however much of the table a single inference never touches. Compare
    #: traffic across two compiles of the SAME graph (where the access patterns are identical) and
    #: it is sound; read one number as an absolute and it is not.
    traffic_bytes: int
    #: (lane, family) -> region count, from the program's own lane record; empty when absent.
    regions_by_lane_family: Mapping[tuple[str, str], int]
    is_lower_bound: bool
    refusals: tuple[str, ...] = ()

    @property
    def host_lane_region_count(self) -> int:
        """Regions on any lane the program did not mark as its accelerator lane."""
        return sum(n for (lane, _), n in self.regions_by_lane_family.items()
                   if lane not in self._accelerator_lanes())

    def _accelerator_lanes(self) -> frozenset[str]:
        """Lanes the program's own records call accelerator lanes. Derived from the record, not named
        here: a lane vocabulary is the compiler's, and hardcoding one target's spelling would make
        every other target read as 100% host."""
        return frozenset(lane for (lane, _) in self.regions_by_lane_family
                         if "mesh" in lane or "accel" in lane or "unit" in lane)

    def to_dict(self) -> dict[str, Any]:
        return {
            "footprint_bytes": self.footprint_bytes,
            "traffic_bytes": self.traffic_bytes,
            "is_lower_bound": self.is_lower_bound,
            "bytes_by_role_dtype": {f"{r}/{d}": b for (r, d), b in
                                    sorted(self.bytes_by_role_dtype.items(), key=lambda kv: -kv[1])},
            "regions_by_lane_family": {f"{l}/{f}": n for (l, f), n in
                                       sorted(self.regions_by_lane_family.items(),
                                              key=lambda kv: -kv[1])},
            "host_lane_region_count": self.host_lane_region_count,
            "refusals": list(self.refusals),
        }


def _elements(shape: Any) -> int | None:
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        return None
    total = 1
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            return None
        total *= dim
    return total


def lane_cost(command_buffer: Mapping[str, Any]) -> LaneCost:
    """Derive the byte and lane accounting of ``command_buffer``.

    Every quantity comes from the buffer itself. A tensor with an unparseable dtype or shape is
    recorded with a refusal and excluded from the totals, which then declare themselves a lower
    bound.
    """
    tensors_in = command_buffer.get("tensors")
    if not isinstance(tensors_in, Mapping):
        return LaneCost((), {}, 0, 0, {}, True,
                        ("command buffer declares no tensors mapping",))

    rows: list[TensorBytes] = []
    by_key: dict[tuple[str, str], int] = {}
    footprint = 0
    traffic = 0
    refusals: list[str] = []

    for name, spec in tensors_in.items():
        if not isinstance(spec, Mapping):
            rows.append(TensorBytes(str(name), "", "", None, None, "tensor spec is not a mapping"))
            refusals.append(f"{name}: tensor spec is not a mapping")
            continue
        role = str(spec.get("role") or "")
        dtype = str(spec.get("dtype") or "")
        elems = _elements(spec.get("shape"))
        bits = dtype_bits(dtype)
        if elems is None or bits is None:
            why = ("shape is not a sequence of non-negative ints" if elems is None
                   else f"dtype {dtype!r} carries no derivable element width")
            rows.append(TensorBytes(str(name), role, dtype, elems, None, why))
            refusals.append(f"{name}: {why}")
            continue
        nbytes = elems * ((bits + 7) // 8)
        rows.append(TensorBytes(str(name), role, dtype, elems, nbytes))
        by_key[(role, dtype)] = by_key.get((role, dtype), 0) + nbytes
        footprint += nbytes
        traffic += nbytes * (2 if role in _ROUND_TRIP_ROLES else 1)

    params = command_buffer.get("params")
    placement = params.get("lane_placement") if isinstance(params, Mapping) else None
    by_lane: dict[tuple[str, str], int] = {}
    if isinstance(placement, Sequence) and not isinstance(placement, (str, bytes)):
        for entry in placement:
            if not isinstance(entry, Mapping):
                continue
            key = (str(entry.get("lane") or ""), str(entry.get("family") or ""))
            by_lane[key] = by_lane.get(key, 0) + 1
    elif placement is not None:
        refusals.append("params.lane_placement is present but not a sequence of mappings")

    return LaneCost(tuple(rows), by_key, footprint, traffic, by_lane,
                    bool(refusals), tuple(refusals))


def format_report(cost: LaneCost, *, label: str = "") -> str:
    """A short human-readable table. Reporting only -- callers compare ``to_dict()``."""
    mib = 1024 * 1024
    out = [f"== lane cost {label}".rstrip()]
    out.append(f"{'role/dtype':24} {'MiB':>10}")
    for (role, dtype), nbytes in sorted(cost.bytes_by_role_dtype.items(), key=lambda kv: -kv[1]):
        out.append(f"{role + '/' + dtype:24} {nbytes / mib:10.2f}")
    bound = " (LOWER BOUND)" if cost.is_lower_bound else ""
    out.append(f"{'footprint':24} {cost.footprint_bytes / mib:10.2f}{bound}")
    out.append(f"{'DRAM traffic':24} {cost.traffic_bytes / mib:10.2f}{bound}")
    if cost.regions_by_lane_family:
        out.append(f"{'lane/family':24} {'regions':>10}")
        for (lane, fam), n in sorted(cost.regions_by_lane_family.items(), key=lambda kv: -kv[1]):
            out.append(f"{lane + '/' + fam:24} {n:10d}")
        out.append(f"{'host-lane regions':24} {cost.host_lane_region_count:10d}")
    for why in cost.refusals[:8]:
        out.append(f"  REFUSAL {why}")
    if len(cost.refusals) > 8:
        out.append(f"  ... and {len(cost.refusals) - 8} more refusals")
    return "\n".join(out)
