"""Derived Phase 0 stage: capsules for the op-forms an ITERATION workload asks of a target.

A corpus written by hand samples the parameters its author thought of. A package that passes all of
it can still be wrong on a real model: one passed every ``residual_add`` capsule in a corpus and was
wrong on every residual add of a real network, because every capsule drew its stimulus from a few
steps around zero and used multipliers at or below one -- a multiplier above one on a saturating
operand load clips before the add, and nothing in the corpus reached that clip.

This stage derives those capsules instead. It reads each declared iteration capture, forms the
target's compute groups through the ONE grouping (``compute_groups`` + ``group_command``, stated by
:func:`merlin.targetgen.group_capsule_entries.entries`), and emits one entry per distinct OP-FORM in
the corpus vocabulary. The entries go into the digest-bound synthesis plan with every other derived
member; the Phase 0 writer builds them under the one corpus binding.

A FORM is what a backend must lower the same way: the operation, its epilogue stages in readout
order, the operand dtype, the SCALE CLASS (every multiplier below one, any above one, all exactly one,
or none), the bias (per output or absent) and the geometry class (a convolution's taps, stride, padding
and fused pool; which contraction axes are degenerate). Extents and the exact multiplier value are not
part of a form -- the representatives are chosen over them: the groups at the two EXTREMES of the
multiplier (a clip, rounding or sign error shows first at the edge), or the most frequent group when
the form has none.

Only the POSITIONS are reduced (the streamed rows, or a convolution's image), to a size that still
spans ``TILES_PER_AXIS`` tiles of the target's own edge, keeps the model's residue modulo that edge and
sits in the same store-capacity regime as the model's layer. Reduction depth and feature count are the
model's. A member whose written output exceeds the target's derived certification ceiling is graded at
its loop tier and ``extends`` a certified sibling of the same form.

HELD-OUT MODELS ARE REFUSED BY NAME. Claim models and evaluation-only models measure the
generalization claim; a form derived from one would build the corpus from the model it is later said
to generalize to. Only the declared iteration roster reaches this stage, and every entry carries its
source ``model`` so :func:`~.claim_boundary.assert_no_claim_capsules` can check it.

Nothing here names a target, a model or an ISA constant: the tile edge comes from the binding, element
ranges from the format registry, store capacities from the RTL facts' address space, the certification
ceiling from the derived requirement.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "model_form_capsules_v1"
#: Every capsule minted here carries this prefix, so its origin is readable from its name.
PREFIX = "MF"
#: The provenance block each entry carries onto its capsule.
BLOCK = "model_form"
#: How many tiles the reduced positions span when the model has more than that.
TILES_PER_AXIS = 2
#: The output spread a requantizing readout's stimulus is sized for, as a fraction of the output
#: format's positive range: one standard deviation at half the range reaches both clip edges.
OUTPUT_SPREAD = 0.5
#: Suffix of the certified sibling a model-scale member rests on when it is too large to certify.
CERT_SUFFIX = "cert"

UNSCALED = "unscaled"
UNITY = "unity"
GAIN_BELOW_ONE = "gain_lt_1"
GAIN_ABOVE_ONE = "gain_gt_1"
UNDERIVABLE = "UNDERIVABLE"

_EXTENT_KEYS = ("M", "K", "N", "ci", "Himg", "Wimg")


class ModelFormRefusal(ValueError):
    """A model this stage cannot, or may not, derive from. The message names the model and why."""


# ------------------------------------------------------------------------------------------------
# forms
# ------------------------------------------------------------------------------------------------
def multipliers(entry: Mapping[str, Any]) -> list[float]:
    """The multipliers an entry applies: its readout scale, or each addend's."""
    out = [float(entry[k]) for k in ("lhs_scale", "rhs_scale") if entry.get(k) is not None]
    if entry.get("acc_scale") is not None and "acc_scale" in (entry.get("epilogue") or ()):
        out.append(float(entry["acc_scale"]))
    return out


def scale_class(entry: Mapping[str, Any]) -> str:
    """Which side of one the entry's multipliers sit on. Any multiplier above one makes the class."""
    values = multipliers(entry)
    if not values:
        return UNSCALED
    if max(values) > 1.0:
        return GAIN_ABOVE_ONE
    if all(v == 1.0 for v in values):
        return UNITY
    return GAIN_BELOW_ONE


def geometry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The part of an entry's shape a lowering must handle differently -- never its extents."""
    if str(entry.get("op")) == "conv2d":
        geo = {k: entry.get(k) for k in ("kh", "kw", "stride", "padding")}
        if "maxpool" in (entry.get("epilogue") or ()):
            geo.update({k: entry.get(k) for k in ("pool_size", "pool_stride", "pool_padding", "pool_pad_value")})
        return geo
    degenerate = sorted(axis for axis in ("M", "K", "N") if entry.get(axis) is not None and int(entry[axis]) == 1)
    return {"degenerate_axes": degenerate}


def form_key(entry: Mapping[str, Any]) -> dict[str, Any]:
    """What makes two groups the same form (see the module docstring)."""
    from merlin.xdsl_dialects.lowering.group_command import STATIONARY_KEY

    epilogue = [str(s) for s in (entry.get("epilogue") or ())]
    stationary = {"stationary": str(entry[STATIONARY_KEY])} if entry.get(STATIONARY_KEY) else {}
    return {
        **stationary,
        "op": str(entry.get("op")),
        "epilogue": epilogue,
        "operand_dtype": str(entry.get("operand_dtype") or ""),
        "scale_class": scale_class(entry),
        "bias": "per_output" if any(s.startswith("bias") for s in epilogue) else "none",
        "geometry": geometry(entry),
    }


def key_text(key: Mapping[str, Any]) -> str:
    return json.dumps(key, sort_keys=True)


def representatives(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    """``[(extreme, row)]`` for one form: the largest- and the smallest-multiplier group.

    Deterministic: ties break on the group count (more first) and then the name. One row is returned
    once, under the first extreme it is chosen for.
    """
    ordered = sorted(rows, key=lambda r: (-int(r["count"]), str(r["name"])))
    scaled = [r for r in ordered if multipliers(r["entry"])]
    if not scaled:
        return [("most_frequent", ordered[0])]
    hi = max(scaled, key=lambda r: max(multipliers(r["entry"])))
    lo = min(scaled, key=lambda r: min(multipliers(r["entry"])))
    picks = [("max_multiplier", hi)]
    if lo is not hi:
        picks.append(("min_multiplier", lo))
    return picks


# ------------------------------------------------------------------------------------------------
# reduction of the positions
# ------------------------------------------------------------------------------------------------
def reduce_extent(extent: int, tile: int, tiles: int = TILES_PER_AXIS) -> int:
    """``extent`` cut to ``tiles`` whole tiles plus its own residue modulo ``tile``; smaller is kept."""
    extent, tile = int(extent), int(tile)
    if tile <= 0:
        raise ValueError(f"a tile edge of {tile} reduces nothing")
    keep = tiles * tile + extent % tile
    return extent if extent <= keep else keep


def _conv_out(size: int, pad_before: int, pad_after: int, tap: int, stride: int) -> int:
    return (size + pad_before + pad_after - tap) // stride + 1


def _conv_image_sizes(entry: Mapping[str, Any], tile: int) -> list[int]:
    """Candidate square images, smallest first, each giving at least ``TILES_PER_AXIS`` tiles of
    output positions (and two pooled positions per side under a fused pool), up to the model's own."""
    kh, stride = int(entry["kh"]), int(entry["stride"][0])
    pad = [int(p) for p in (entry.get("padding") or [0, 0, 0, 0])]
    pooled = "maxpool" in (entry.get("epilogue") or ())
    real = int(entry["Himg"])
    sizes = []
    for size in range(kh, real + 1):
        positions = _conv_out(size, pad[0], pad[2], kh, stride)
        if positions < 1 or positions * positions < TILES_PER_AXIS * tile:
            continue
        if pooled:
            ps, pst = int(entry["pool_size"][0]), int(entry["pool_stride"][0])
            pp = [int(p) for p in (entry.get("pool_padding") or [0, 0, 0, 0])]
            if _conv_out(positions, pp[0], pp[2], ps, pst) < 2:
                continue
        sizes.append(size)
    return sizes or [real]


def reduce_entry(
    entry: Mapping[str, Any], tile: int, *, same_regime: Callable[[Mapping[str, Any]], bool] | None = None
) -> dict[str, Any]:
    """The entry at a reduced, multi-tile size with every non-extent parameter unchanged.

    Only the positions are reduced, and never across a store-capacity boundary: ``same_regime`` says
    whether a candidate sits in the model layer's regime, and the positions grow from the smallest
    multi-tile size until one does. A capsule that fits where the model spills tests another program.
    """
    out = dict(entry)
    candidates: list[dict[str, Any]] = []
    if str(entry.get("op")) == "conv2d":
        for size in _conv_image_sizes(entry, tile):
            candidates.append({**out, "Himg": size, "Wimg": min(size, int(entry["Wimg"]))})
    elif entry.get("M") is not None:
        real = int(entry["M"])
        start = reduce_extent(real, tile)
        candidates = [{**out, "M": value} for value in [*range(start, real, tile), real]]
    candidates = candidates or [out]
    if same_regime is None:
        return candidates[0]
    for candidate in candidates:
        if same_regime(candidate):
            return candidate
    return candidates[-1]


# ------------------------------------------------------------------------------------------------
# store-capacity regimes, from the RTL facts' address space
# ------------------------------------------------------------------------------------------------
def working_sets(entry: Mapping[str, Any], operand_dtype: str, accum_dtype: str) -> dict[str, list]:
    """The tensors each store holds for one entry: operands in the operand store, the result in the
    accumulator. Shapes are row-major with the axis laid along a row last."""
    op = str(entry.get("op"))
    if op == "conv2d":
        pad = [int(p) for p in (entry.get("padding") or [0, 0, 0, 0])]
        stride = [int(s) for s in entry["stride"]]
        ho = _conv_out(int(entry["Himg"]), pad[0], pad[2], int(entry["kh"]), stride[0])
        wo = _conv_out(int(entry["Wimg"]), pad[1], pad[3], int(entry["kw"]), stride[1])
        taps = int(entry["kh"]) * int(entry["kw"]) * int(entry["ci"])
        operands = [
            ([int(entry["Himg"]) * int(entry["Wimg"]), int(entry["ci"])], operand_dtype),
            ([taps, int(entry["N"])], operand_dtype),
        ]
        result = [([ho * wo, int(entry["N"])], accum_dtype)]
    elif op == "residual_add":
        shape = [int(entry["M"]), int(entry["N"])]
        operands, result = [(shape, operand_dtype), (shape, operand_dtype)], [(shape, accum_dtype)]
    else:
        m, k, n = int(entry["M"]), int(entry.get("K") or 1), int(entry["N"])
        operands = [([m, k], operand_dtype), ([k, n], operand_dtype)]
        result = [([m, n], accum_dtype)]
    return {"operand_capacity": operands, "accumulator_capacity": result}


class CapacityModel:
    """The target's operand store and accumulator, derived once from its RTL facts."""

    def __init__(self, target: str, facts: Mapping[str, Any] | None):
        from merlin.targetgen import address_space as AS

        self._as = AS
        self.stores: dict[str, Any] = {"operand_capacity": None, "accumulator_capacity": None}
        try:
            space = AS.derive_address_space(target, facts=dict(facts) if facts is not None else None)
            self.stores["operand_capacity"] = getattr(AS.operand_store(space), "store", None)
            self.stores["accumulator_capacity"] = getattr(AS.accumulator_store(space), "store", None)
        except Exception:  # noqa: BLE001 -- an underivable address space leaves every regime UNDERIVABLE
            pass

    def regimes(self, entry: Mapping[str, Any], *, operand_dtype: str, accum_dtype: str) -> dict[str, str]:
        """``{axis: fits | spills | UNDERIVABLE}`` for the operand store and the accumulator."""
        out: dict[str, str] = {}
        for axis, tensors in working_sets(entry, operand_dtype, accum_dtype).items():
            store = self.stores.get(axis)
            capacity = getattr(store, "total_rows", None) if store is not None else None
            if not capacity:
                out[axis] = UNDERIVABLE
                continue
            rows = [self._as.working_set_rows(store, shape, dtype) for shape, dtype in tensors]
            if any(r is None for r in rows):
                out[axis] = UNDERIVABLE
                continue
            out[axis] = "fits" if sum(rows) <= int(capacity) else "spills"
        return out


# ------------------------------------------------------------------------------------------------
# stimulus
# ------------------------------------------------------------------------------------------------
def element_range(dtype: str) -> tuple[int, int]:
    """The whole range of an integer element format, from the format registry or the dtype table."""
    from merlin.common import quant_formats as qf
    from merlin.targetgen import corpus_spec as CS

    token = str(dtype)
    try:
        fmt = qf.get(token)
    except (KeyError, ValueError):
        fmt = None
    if fmt is not None:
        if fmt.kind != "int_affine":
            raise ModelFormRefusal(f"element format {dtype!r} is not an integer format; no element range to span")
        half = 1 << (int(fmt.element_bits) - 1)
        return -half, half - 1
    try:
        spelling, _mlir, width, integer = CS.dtype_info(token)
    except KeyError as error:
        raise ModelFormRefusal(f"element format {token!r} is neither a registered format nor a dtype token") from error
    if not integer or not width:
        raise ModelFormRefusal(f"element format {token!r} is not an integer format; no element range to span")
    bits = 8 * int(width)
    if spelling[:1] == "u":
        return 0, (1 << bits) - 1
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _reduction(entry: Mapping[str, Any]) -> int:
    if str(entry.get("op")) == "conv2d":
        return int(entry["ci"]) * int(entry["kh"]) * int(entry["kw"])
    return int(entry.get("K") or 1)


def mac_limit(numerical_semantics: Mapping[str, Any] | None) -> int | None:
    """The largest partial sum the declared internal MAC keeps exact, or ``None`` when undeclared.

    Read from the selected software spec's ``internal_arithmetic``: only a declared bounded-exact
    policy with a declared MAC width bounds a stimulus; anything else leaves it unbounded here and the
    writer's own partial-sum screen still decides.
    """
    internal = (numerical_semantics or {}).get("internal_arithmetic") or {}
    bits = internal.get("mac_result_bits")
    if internal.get("full_operation_overflow_policy") != "bounded_exact_requires_each_partial_sum":
        return None
    if isinstance(bits, bool) or not isinstance(bits, int) or bits < 2:
        return None
    return (1 << (bits - 1)) - 1


def stimulus_range(
    entry: Mapping[str, Any],
    *,
    operand_range: tuple[int, int],
    output_range: tuple[int, int],
    partial_sum_limit: int | None = None,
) -> list[int]:
    """The stimulus range for one reduced entry.

    * per-operand multipliers (a declared ``bound_lsb``): the operand format's whole range;
    * a contraction: the widest SYMMETRIC range whose accumulator, under the model's own readout
      multiplier (one when none), spreads to ``OUTPUT_SPREAD`` of the committed output's range --
      never narrower than the signed default, never wider than the operand format. Symmetric on
      purpose: a two's-complement ``[-m, m-1]`` has a nonzero product mean that, over a deep
      reduction, saturates a narrow commit on almost every element.
    """
    from merlin.runtime.commandbuffer import SIGNED_STIMULUS_RANGE

    lo, hi = int(operand_range[0]), int(operand_range[1])
    if "bound_lsb" in entry:
        return [lo, hi]
    scale = float(entry["acc_scale"]) if "acc_scale" in (entry.get("epilogue") or ()) else 1.0
    if scale <= 0.0:
        return [lo, hi]
    target_sigma = OUTPUT_SPREAD * float(output_range[1])
    magnitude = math.sqrt(3.0 * target_sigma / (scale * math.sqrt(max(1, _reduction(entry)))))
    floor_mag = int(SIGNED_STIMULUS_RANGE[1])
    magnitude = int(min(float(min(-lo, hi)), max(float(floor_mag), round(magnitude))))
    if partial_sum_limit is not None and str(entry.get("op")) != "residual_add":
        # THE DECLARED MAC WIDTH BOUNDS EVERY PARTIAL SUM, so the stimulus may not exceed it: a
        # reduction of K products of magnitude m^2, plus a bias of magnitude m, must stay exact.
        depth = max(1, _reduction(entry))
        exact = int(math.isqrt(max(0, int(partial_sum_limit)) // depth))
        while exact > 0 and depth * exact * exact + exact > int(partial_sum_limit):
            exact -= 1
        if exact < 1:
            raise ModelFormRefusal(
                f"a reduction of {depth} exceeds the declared internal MAC width for any nonzero stimulus"
            )
        magnitude = min(magnitude, exact)
    return [-magnitude, magnitude]


# ------------------------------------------------------------------------------------------------
# certification
# ------------------------------------------------------------------------------------------------
def written_elements(entry: Mapping[str, Any]) -> int:
    """How many output elements the device writes for ``entry`` -- what certification cost tracks."""
    from merlin.xdsl_dialects.lowering import group_command as GC

    rows, cols = GC.device_output_shape(entry)
    return int(rows) * int(cols)


def loop_tier(tiers: Sequence[str]) -> str | None:
    """The deepest declared tier that is NOT cycle-accurate, or ``None`` when there is none."""
    from merlin.targetgen.phase_policy import _CYCLE_ACCURATE_TIERS  # noqa: PLC2701

    declared = [str(t) for t in tiers]
    cheap = [t for t in declared if t not in _CYCLE_ACCURATE_TIERS]
    return cheap[-1] if cheap and len(cheap) < len(declared) else None


def _fit_features(features: int, tile: int, budget: int) -> int:
    """The most output features inside ``budget`` elements per position row, keeping the model's
    residue modulo the tile where it still fits."""
    if features <= budget:
        return features
    whole = max(1, budget // tile) * tile
    residue = features % tile
    return whole + residue if residue and whole + residue <= budget else min(whole, features)


def _position_candidates(entry: Mapping[str, Any], tile: int) -> list[dict[str, Any]]:
    """The entry at every smaller position count, smallest first: a convolution's image, a streamed M."""
    out = []
    if str(entry.get("op")) == "conv2d":
        kh, stride = int(entry["kh"]), int(entry["stride"][0])
        pad = [int(p) for p in (entry.get("padding") or [0, 0, 0, 0])]
        pooled = "maxpool" in (entry.get("epilogue") or ())
        for size in range(1, int(entry["Himg"]) + 1):
            positions = _conv_out(size, pad[0], pad[2], kh, stride)
            if positions < 1:
                continue
            if pooled:
                ps, pst = int(entry["pool_size"][0]), int(entry["pool_stride"][0])
                pp = [int(p) for p in (entry.get("pool_padding") or [0, 0, 0, 0])]
                if _conv_out(positions, pp[0], pp[2], ps, pst) < 1:
                    continue
            out.append(dict(entry, Himg=size, Wimg=min(size, int(entry["Wimg"]))))
        return out
    real = int(entry["M"])
    values = sorted({*range(1, min(real, tile) + 1), *range(tile, real + 1, tile), real})
    return [dict(entry, M=m) for m in values]


def _cert_sibling(entry, *, tile, ceiling, regimes) -> tuple[dict[str, Any] | None, list[str]]:
    """``(sibling, properties it lost)``: keep the member's output features and capacity regime and
    shrink its positions; only when that cannot fit the ceiling are the features cut, and said so."""
    member_regime = regimes(entry) if regimes else None
    for candidate in _position_candidates(entry, tile):
        if member_regime is not None and regimes(candidate) != member_regime:
            continue
        if written_elements(candidate) <= ceiling:
            return candidate, []
        break  # a larger image or row count only writes more
    sibling = dict(entry)
    if str(entry.get("op")) == "conv2d":
        sibling["Himg"] = _conv_image_sizes(entry, tile)[0]
        sibling["Wimg"] = min(sibling["Himg"], int(entry["Wimg"]))
    else:
        sibling["M"] = reduce_extent(int(entry["M"]), tile)
    rows = written_elements(dict(sibling, N=1))
    if rows <= 0 or rows > ceiling:
        return None, ["output_features", "accumulator_capacity", "operand_capacity"]
    sibling["N"] = _fit_features(int(entry["N"]), tile, ceiling // rows)
    if written_elements(sibling) > ceiling:
        return None, ["output_features", "accumulator_capacity", "operand_capacity"]
    lost = [f"output_features:{entry['N']}->{sibling['N']}"] if int(sibling["N"]) != int(entry["N"]) else []
    if member_regime is not None:
        now = regimes(sibling)
        lost += [f"{a}:{member_regime[a]}->{now.get(a)}" for a in member_regime if now.get(a) != member_regime[a]]
    return sibling, lost


def split_for_certification(
    entry: dict[str, Any],
    *,
    tile: int,
    ceiling: int | None,
    cheap_tier: str | None,
    regimes,
    operand_range: tuple[int, int],
    output_range: tuple[int, int],
    partial_sum_limit: int | None = None,
) -> list[dict[str, Any]]:
    """``[entry]`` when certifiable, else ``[certified sibling, entry capped at the loop tier]``.

    The model-scale member is the one that can see a capacity-driven defect; the sibling certifies
    the form's arithmetic at the model's multipliers and reduction depth. Neither replaces the other.
    """
    block = entry[BLOCK]
    written = written_elements(entry)
    block["written_elements"] = written
    if ceiling is None or written <= ceiling:
        block["certification"] = "certified" if ceiling is not None else "unpriced"
        return [entry]
    if cheap_tier is None:
        block["certification"] = "unaffordable_no_cheaper_tier"
        return [entry]
    sibling, lost = _cert_sibling(entry, tile=tile, ceiling=ceiling, regimes=regimes)
    entry["max_oracle_tier"] = cheap_tier
    if sibling is None:
        block["certification"] = f"loop_tier_only: no sibling of this form fits {ceiling} written elements"
        return [entry]
    name = f"{entry['name']}_{CERT_SUFFIX}"
    entry["extends"] = name
    block["certification"] = f"loop_tier: {written} written elements exceed the {ceiling} a certification affords"
    sibling["name"] = name
    sibling.pop("max_oracle_tier", None)
    sibling.pop("extends", None)
    sibling["stimulus_range"] = stimulus_range(
        sibling, operand_range=operand_range, output_range=output_range, partial_sum_limit=partial_sum_limit
    )
    sibling["source_reference"] = (
        f"{entry['source_reference']}; the certified sibling of {entry['name']}, keeping its output "
        f"features and store-capacity regime where the {ceiling} written elements a certification affords allow"
    )
    sibling[BLOCK] = {
        **block,
        "representative": f"{block['representative']}+{CERT_SUFFIX}",
        "reduced_extents": {k: sibling[k] for k in _EXTENT_KEYS if sibling.get(k) is not None},
        "written_elements": written_elements(sibling),
        "certification": "certified",
        "capacity_regimes": regimes(sibling) if regimes else None,
        "capacity_regimes_kept": not lost,
    }
    if lost:
        sibling[BLOCK]["cert_regime_lost"] = list(lost)
    return [sibling, entry]


# ------------------------------------------------------------------------------------------------
# naming
# ------------------------------------------------------------------------------------------------
def _slug(text: str) -> str:
    out = "".join(c if c.isalnum() else "_" for c in str(text).lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _geometry_tag(entry: Mapping[str, Any]) -> str:
    if str(entry.get("op")) == "conv2d":
        pad = entry.get("padding") or [0, 0, 0, 0]
        tag = f"k{entry['kh']}x{entry['kw']}s{entry['stride'][0]}p{pad[0]}"
        if "maxpool" in (entry.get("epilogue") or ()):
            tag += f"_pool{entry['pool_size'][0]}s{entry['pool_stride'][0]}"
        return tag
    return "_".join(f"{axis.lower()}1" for axis in geometry(entry)["degenerate_axes"])


def capsule_name(model: str, entry: Mapping[str, Any], extreme: str) -> str:
    """``MF_<model>_<op>[_<geometry>]_<stages|raw>_<scale class>[_<extreme>]``."""
    from merlin.xdsl_dialects.lowering.group_command import STATIONARY_KEY

    stages = "_".join(str(s) for s in (entry.get("epilogue") or ())) or "raw"
    # The stationary operand is part of the form (form_key), so it is part of the name: two forms that
    # differ only in which operand stays resident would otherwise mint one capsule name twice.
    stationary = f"{entry[STATIONARY_KEY]}_stationary" if entry.get(STATIONARY_KEY) else ""
    parts = [PREFIX, _slug(model), str(entry["op"]), stationary, _geometry_tag(entry), stages, scale_class(entry)]
    if extreme in ("max_multiplier", "min_multiplier"):
        parts.append("max" if extreme == "max_multiplier" else "min")
    return "_".join(p for p in parts if p)


# ------------------------------------------------------------------------------------------------
# derivation
# ------------------------------------------------------------------------------------------------
def derive_application(
    label: str,
    stated: Mapping[str, Any],
    binding,
    *,
    capture_sha256: str,
    capacity: CapacityModel | None = None,
    ceiling: int | None = None,
    partial_sum_limit: int | None = None,
) -> dict[str, Any]:
    """Model-form entries for one iteration application's stated groups (``group_capsule_entries.entries``)."""
    from merlin.targetgen import corpus_spec as CS
    from merlin.targetgen.group_capsule_entries import SOURCE_ROLE

    tile = int(binding.tile_dim)
    cheap = loop_tier(binding.tiers)
    operand_token = binding.cap_dtype(binding.operand_dtype)
    accum_token = binding.cap_dtype(binding.accum_dtype)

    def regimes(e):
        if capacity is None:
            return {}
        return capacity.regimes(e, operand_dtype=operand_token, accum_dtype=accum_token)

    from merlin.xdsl_dialects.lowering.group_command import device_orientation

    forms: dict[str, dict[str, Any]] = {}
    for row in stated.get("entries") or []:
        if row.get("raw_of"):
            continue
        # The capsule certifies the orientation a whole-model program HOLDS the operands in, which for a
        # window mean is not the one its group statement uses (one definition: group_command).
        row = {**row, "entry": device_orientation(row["entry"], row.get("program") or {"stored_operand": None})}
        key = form_key(row["entry"])
        forms.setdefault(key_text(key), {"key": key, "rows": []})["rows"].append(row)
    entries: list[dict[str, Any]] = []
    refused: dict[str, str] = {}
    for text in sorted(forms):
        form = forms[text]
        key = form["key"]
        form_groups = sorted(g for row in form["rows"] for g in row["groups"])
        for extreme, row in representatives(form["rows"]):
            real = dict(row["entry"])
            real_regimes = regimes(real)
            reduced = reduce_entry(real, tile, same_regime=lambda e, want=real_regimes: regimes(e) == want)
            name = capsule_name(label, reduced, extreme)
            stages = list(reduced.get("epilogue") or ())
            # A bias stage brings its own operand, which is what licenses a bias route (the same roles
            # the builders pass): without it a seeded bias would be refused as if no route existed.
            roles = frozenset({"bias"}) if any(stage in CS._BIAS_STAGES for stage in stages) else frozenset()  # noqa: SLF001
            output_dtype = CS._resolve_output_dtype(binding, stages, reduced, available_operand_roles=roles)  # noqa: SLF001
            try:
                operand_range = element_range(str(reduced.get("operand_dtype") or binding.operand_dtype))
                output_range = element_range(output_dtype) if binding.integer else operand_range
            except ModelFormRefusal as error:
                refused[name] = str(error)
                continue
            for k in ("stimulus_range", "source_reference", "comment", "name", "cat", "kind", "label"):
                reduced.pop(k, None)
            try:
                reduced["stimulus_range"] = stimulus_range(
                    reduced,
                    operand_range=operand_range,
                    output_range=output_range,
                    partial_sum_limit=partial_sum_limit,
                )
            except ModelFormRefusal as error:
                refused[name] = str(error)
                continue
            program = row.get("program") or {}
            entry = {
                **reduced,
                "name": name,
                "cat": "layers",
                "kind": "layer",
                "label": "public",
                "model": label,
                "source_role": SOURCE_ROLE,
                "source_reference": (
                    f"derived from {row['count']} compute group(s) of iteration workload {label}: the "
                    f"{extreme.replace('_', ' ')} representative of its {key['op']} form with stages "
                    f"{key['epilogue'] or 'none'}, scale class {key['scale_class']}, at the model's own "
                    f"multipliers and a {TILES_PER_AXIS}-tile reduction of its positions"
                ),
                "generalization": {"generalization_axis": "application"},
                BLOCK: {
                    "schema": SCHEMA,
                    "model": label,
                    "workload_role": "iteration",
                    "capture_sha256": capture_sha256,
                    "form": key,
                    "representative": extreme,
                    "groups": list(row["groups"]),
                    "form_groups": form_groups,
                    "multipliers": multipliers(real),
                    "real_extents": {k: real[k] for k in _EXTENT_KEYS if real.get(k) is not None},
                    "reduced_extents": {k: reduced[k] for k in _EXTENT_KEYS if reduced.get(k) is not None},
                    "tile_edge": tile,
                    "capacity_regimes": dict(real_regimes),
                    "capacity_regimes_kept": regimes(reduced) == real_regimes,
                    "output_dtype": output_dtype,
                    "stimulus_rule": "operand_full_range" if "bound_lsb" in reduced else "readout_spread",
                    "operand_order": {"transposed": bool(program.get("transposed"))},
                },
            }
            if row.get("host_region"):
                entry[BLOCK]["host_region"] = {
                    k: row["host_region"][k] for k in ("kind", "rows", "window", "divisor", "region_dtype")
                }
            entries.extend(
                split_for_certification(
                    entry,
                    tile=tile,
                    ceiling=ceiling,
                    cheap_tier=cheap,
                    regimes=regimes if capacity is not None else None,
                    operand_range=operand_range,
                    output_range=output_range,
                    partial_sum_limit=partial_sum_limit,
                )
            )
    return {
        "status": "derived" if entries else "empty",
        "capture_sha256": capture_sha256,
        "accelerator_groups": stated.get("accelerator_groups"),
        "stated_groups": stated.get("stated"),
        "forms": len(forms),
        "unstated": dict(stated.get("unstated") or {}),
        "refused_by_format": refused,
        "entries": entries,
    }


def host_reduction_rows(target: str, module, binding, *, oracle=None) -> list[dict[str, Any]]:
    """Host-region trailing-window reductions (row sums, window means) as stated rows in device form.

    A softmax denominator, a normalization's mean or a global pool spends its time on the host while
    its core is a sum a unit can take; these rows offer that device form at the workload's own window
    and row count, deduplicated like any other stated group.
    """
    from merlin.targetgen import group_capsule_entries as G
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    found: dict[str, dict[str, Any]] = {}
    for form in G.host_reduction_forms(CG.form_groups(module, target, oracle=oracle)):
        identity = key_text({k: form[k] for k in ("kind", "rows", "window", "divisor")})
        entry = G.reduction_entry(form, operand_dtype=str(binding.operand_dtype))
        row = found.setdefault(
            identity,
            {
                "name": f"H_{form['kind']}_{form['rows']}x{form['window']}",
                "count": 0,
                "groups": [],
                "entry": entry,
                "host_region": dict(form),
            },
        )
        row["count"] += 1
        row["groups"].append(form["group"])
    return list(found.values())


def derive_model_forms(
    target: str,
    captures: Mapping[str, Path],
    binding,
    *,
    iteration_roster: Sequence[str],
    held_out: Sequence[str],
    facts: Mapping[str, Any] | None = None,
    ceiling: int | None = None,
    numerical_semantics: Mapping[str, Any] | None = None,
    oracle=None,
) -> dict[str, Any]:
    """The model-form stage over the declared ITERATION captures; refuses any held-out model.

    ``captures`` must be exactly the iteration roster (the derivation already requires that), and no
    label may be a claim or evaluation-only model. Returns ``{"provenance": ..., "entries": [...]}``.
    """
    from merlin.common import mlir_query as mq
    from merlin.targetgen.group_capsule_entries import entries as stated_entries
    from merlin.xdsl_dialects.lowering import stream_plan

    from .claim_boundary import assert_no_claim_capsules, is_held_out

    if set(captures) != set(iteration_roster):
        raise ModelFormRefusal(
            f"model forms derive from the iteration roster only: got {sorted(captures)}, "
            f"declared {sorted(iteration_roster)}"
        )
    for label in captures:
        held = is_held_out(label, held_out)
        if held is not None:
            raise ModelFormRefusal(f"{label!r} belongs to held-out model {held!r}; it may be evaluated, never derived")
    capacity = CapacityModel(target, facts)
    models: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    for label in sorted(captures):
        path = Path(captures[label])
        raw = path.read_bytes()
        manifest = path.with_name("weights.safetensors.manifest.json")
        weights = (
            stream_plan.weight_args_of(json.loads(manifest.read_text(encoding="utf-8"))) if manifest.is_file() else None
        )
        module = mq.parse(str(path))
        stated = dict(stated_entries(target, module, weight_args=weights, model=label, with_raw=False, oracle=oracle))
        stated["entries"] = [*stated["entries"], *host_reduction_rows(target, module, binding, oracle=oracle)]
        got = derive_application(
            label,
            stated,
            binding,
            capture_sha256=hashlib.sha256(raw).hexdigest(),
            capacity=capacity,
            ceiling=ceiling,
            partial_sum_limit=mac_limit(numerical_semantics),
        )
        models[label] = {k: v for k, v in got.items() if k != "entries"} | {
            "capsules": [e["name"] for e in got["entries"]]
        }
        entries.extend(got["entries"])
    names = [e["name"] for e in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"two representatives minted the same capsule name: {sorted(names)}")
    assert_no_claim_capsules(entries, list(held_out))
    return {
        "provenance": {
            "schema": SCHEMA,
            "generated_by": f"{__name__}.derive_model_forms",
            "target": target,
            "tiles_per_axis": TILES_PER_AXIS,
            "output_spread": OUTPUT_SPREAD,
            "certification_ceiling": ceiling,
            "partial_sum_limit": mac_limit(numerical_semantics),
            "grouping": "merlin.targetgen.group_capsule_entries.entries",
            "held_out_refused": sorted(held_out),
            "models": models,
            "n_capsules": len(entries),
        },
        "entries": entries,
    }
