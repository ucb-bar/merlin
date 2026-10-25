"""A contraction unit's output path as an ordered sequence of target-neutral stages.

The capability record (:mod:`merlin.targetgen.rtl.output_epilogue`) says WHAT a design's output path
can do; this states it in the one vocabulary the fusion pass
(:mod:`merlin.xdsl_dialects.lowering.output_fusion`) matches a source chain against. Two parts:

``accumulator``
    Integer stages that change the accumulator before it is read out, in no particular order
    (integer additions commute): ``preload`` (a value loaded into the accumulator that the
    contraction adds onto) and ``accumulate`` (an operand added by loading into the accumulator).
``readout``
    The stages a narrowing readout applies, IN ORDER: ``activation`` (in the ``accumulator`` or the
    ``scaled`` domain), ``convert`` (integer to the scale's float format), ``scale``, ``requant``
    (round, clamp and narrow) and ``pool`` (a window reduction of requantized values).

``exits`` says where a value can leave the unit: ``accumulator`` (the raw accumulator at full width)
and ``requant`` (the narrowed readout). Every stage carries the dtype, granularity and rounding
the fusion proof models; a field a deriver could not establish is absent, and the pass refuses
whatever needs it. Nothing here names a target.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA = "output_path_v1"
ACCUMULATOR_KINDS = ("preload", "accumulate")
READOUT_KINDS = ("activation", "convert", "scale", "requant", "pool")
ACTIVATION_DOMAINS = ("accumulator", "scaled")
GRANULARITIES = ("per_tensor", "per_command", "per_channel")
EXITS = ("accumulator", "requant")


def from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """The ordered output path a derived ``output_epilogue_v1`` record establishes."""
    derived = "derived"
    scale, activation = record.get("scale") or {}, record.get("activation") or {}
    accumulator_dtype = record.get("accumulator_dtype")
    path: dict[str, Any] = {"schema": SCHEMA, "accumulator": {"dtype": accumulator_dtype}, "accumulator_stages": []}
    preload = record.get("bias_preload") or {}
    if preload.get("status") == derived:
        path["accumulator_stages"].append(
            {"kind": "preload", "dtype": preload.get("dtype"), "range": preload.get("range"), "rounding": "none"}
        )
    if (record.get("accumulate") or {}).get("status") == derived and preload.get("status") == derived:
        # An operand the load path delivers unchanged: the same exact lanes a preload uses.
        path["accumulator_stages"].append({"kind": "accumulate", "dtype": accumulator_dtype, "operand_scale": "none"})
    readout: list[dict[str, Any]] = []
    rectifier = activation.get("status") == derived and "relu" in (activation.get("functions") or {})
    if rectifier and activation.get("before_scale") is True:
        readout.append({"kind": "activation", "function": "relu", "domain": "accumulator"})
    scaled = scale.get("status") == derived and scale.get("rounding")
    if scaled:
        readout.append({"kind": "convert", "to": scale.get("dtype"), "rounding": scale["rounding"]})
        readout.append(
            {
                "kind": "scale",
                "dtype": scale.get("dtype"),
                "rounding": scale["rounding"],
                "granularity": "per_command",
                "columns_per_scale": scale.get("columns_per_scale"),
            }
        )
        if rectifier and activation.get("before_scale") is False:
            readout.append({"kind": "activation", "function": "relu", "domain": "scaled"})
        if scale.get("saturation") and scale.get("output_dtypes"):
            readout.append(
                {
                    "kind": "requant",
                    "to": scale["output_dtypes"][0],
                    "rounding": scale["rounding"],
                    "clamp": list(scale["saturation"]),
                }
            )
    pool = record.get("pool") or {}
    if pool.get("status") == derived and readout and readout[-1]["kind"] == "requant":
        # No padding value has been derived by the epilogue reader. Keep the
        # output-path refusal spelling even for historical records omitting it.
        readout.append({"kind": "pool", "reductions": list(pool.get("reductions") or ()), "padding": "unknown"})
    path["readout"] = readout
    full = (record.get("full_width_readout") or {}).get("status") == derived
    path["exits"] = {"accumulator": full, "requant": bool(readout) and any(s["kind"] == "requant" for s in readout)}
    return validate(path)


def validate(path: Any) -> dict[str, Any]:
    """Raise unless ``path`` is a well-formed ``output_path_v1``; return it."""
    if not isinstance(path, Mapping) or path.get("schema") != SCHEMA:
        raise ValueError(f"an output path must be a {SCHEMA} mapping")
    for stage in path.get("accumulator_stages") or ():
        if not isinstance(stage, Mapping) or stage.get("kind") not in ACCUMULATOR_KINDS:
            raise ValueError(f"accumulator stages are one of {ACCUMULATOR_KINDS}")
    order = [stage.get("kind") for stage in path.get("readout") or () if isinstance(stage, Mapping)]
    if len(order) != len(path.get("readout") or ()) or any(kind not in READOUT_KINDS for kind in order):
        raise ValueError(f"readout stages are one of {READOUT_KINDS}")
    if "scale" in order and ("convert" not in order or order.index("convert") > order.index("scale")):
        raise ValueError("a scale is applied to a converted value, so a convert precedes it")
    if "requant" in order and order.index("requant") != len([k for k in order if k != "pool"]) - 1:
        raise ValueError("requant ends the readout; only a pool may follow it")
    for stage in path.get("readout") or ():
        if stage["kind"] == "activation" and stage.get("domain") not in ACTIVATION_DOMAINS:
            raise ValueError(f"an activation states its domain, one of {ACTIVATION_DOMAINS}")
        if stage["kind"] == "scale" and stage.get("granularity") not in GRANULARITIES:
            raise ValueError(f"a scale states its granularity, one of {GRANULARITIES}")
    exits = path.get("exits")
    if not isinstance(exits, Mapping) or set(exits) != set(EXITS):
        raise ValueError(f"an output path states each of its exits {EXITS}")
    return dict(path)
