"""A target's quantization METHODOLOGY: one declared rule set, applied to every model's recipe.

A recipe (:mod:`merlin.targetgen.quant_recipe`) says which number formats a unit computes. That is
not enough to quantize a model the way the unit computes it. A contraction unit's output path
(:mod:`merlin.targetgen.output_path`) fixes much more: which value is preloaded into the
accumulator (a bias), which rectifier runs in which domain, how many columns share ONE scale, the
rounding, the clamp, the window maximum. A model quantized without those facts computes a
different function: a float bias off the accumulator grid, or three float roundings where the unit
has one. Explicit per-channel source scales leave independent output partitions to the compiler,
while retaining the physical readout vector's one shared scale. Every one of
them makes the compiler's exactness proof fail, and each one was chosen by a default, not by the
target.

So a target DECLARES a methodology in its software spec (``quantization.methodology``), every
value in it either a choice that is not a hardware fact (calibration, accuracy recovery, the float
island) or the word ``from_output_path``. :func:`derive` combines it with the derived output path
and a hardware-derived base recipe into a recipe whose ``epilogue`` block the generic capture-side
quantizer (``_integer_epilogue``) realizes with TorchAO's own extension points. A per-model change
to that recipe exists only as a reviewed ``model_variants`` row carrying its reason and measured
accuracy; nothing else can alter a model's recipe.

Nothing here imports a framework or names a target. A fact the path does not establish makes the
recipe ``underivable`` with the reason; it is never filled with a common default.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from merlin.targetgen import output_path as OP
from merlin.targetgen import quant_recipe as QR

SCHEMA = "merlin.quantization_methodology.v1"
EPILOGUE_SCHEMA = "integer_epilogue_v1"
FROM_OUTPUT_PATH = "from_output_path"

#: Authored vocabulary. Closed: a new rule adds a word here and the code that realizes it.
SCALE_GRANULARITIES = (FROM_OUTPUT_PATH, "per_tensor", "per_channel")
BIASES = ("accumulator_preload_int32", "host_float")
EPILOGUES = ("integer_domain",)
CALIBRATIONS = ("mse", "percentile", "minmax")
WEIGHT_CALIBRATIONS = ("minmax",)
RECOVERY = ("cross_layer_equalization", "bias_correction")
#: Framework-neutral names of operations a float island keeps (the capture side maps them to ATen).
ISLAND_OPS = ("softmax", "layer_norm", "rms_norm", "group_norm", "gelu", "silu", "rotary")
#: The only island format: Merlin's host computes these in IEEE single precision, never bf16/f16.
ISLAND_DTYPE = "fp32"
#: What a reviewed model variant may change in a derived recipe, and nothing else.
VARIANT_CHANGES = ("families", "dynamic", "calibration_count", "scale_granularity", "bias", "static_planes")
_FIELDS = {
    "schema",
    "scale_granularity",
    "bias",
    "epilogue",
    "rounding",
    "clamp",
    "calibration",
    "recovery",
    "float_island",
    "model_variants",
    "description",
}
_VARIANT_FIELDS = {"model", "name", "status", "reason", "reviewed_by", "measured_accuracy", "changes"}
#: Roundings the capture graph spells exactly (``quantize_per_tensor`` and IEEE arithmetic).
_EXACT_ROUNDINGS = ("half_even",)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"quantization.methodology: {message}")


def _measured(value: Any, where: str) -> None:
    if isinstance(value, Mapping):
        _require(bool(value), f"{where} is empty")
        for key, member in value.items():
            _require(isinstance(key, str) and bool(key.strip()), f"{where} keys are metric names")
            _measured(member, f"{where}.{key}")
    elif isinstance(value, bool) or not isinstance(value, (int, float, str)):
        _require(False, f"{where} holds numbers (or a basis string)")
    elif isinstance(value, float):
        _require(math.isfinite(value), f"{where} is finite")


def validate(block: Any) -> dict[str, Any]:
    """The validated methodology, or ``ValueError`` naming the field. The authored bytes are not changed."""
    _require(isinstance(block, Mapping), "must be a mapping")
    unknown = set(block) - _FIELDS
    _require(not unknown, f"unknown fields {sorted(unknown)}")
    _require(block.get("schema", SCHEMA) == SCHEMA, f"schema is {SCHEMA}")
    body = copy.deepcopy(dict(block))
    body["schema"] = SCHEMA
    for key, allowed in (
        ("scale_granularity", SCALE_GRANULARITIES),
        ("bias", BIASES),
        ("epilogue", EPILOGUES),
        ("rounding", (FROM_OUTPUT_PATH,)),
        ("clamp", (FROM_OUTPUT_PATH,)),
    ):
        _require(body.get(key) in allowed, f"{key} is one of {list(allowed)}")
    calibration = body.get("calibration")
    _require(isinstance(calibration, Mapping), "calibration states the activation and weight methods")
    _require(set(calibration) <= {"activation", "weight", "percentile"}, "calibration has unknown fields")
    _require(calibration.get("activation") in CALIBRATIONS, f"calibration.activation is one of {list(CALIBRATIONS)}")
    _require(
        calibration.get("weight") in WEIGHT_CALIBRATIONS,
        f"calibration.weight is one of {list(WEIGHT_CALIBRATIONS)}",
    )
    if calibration["activation"] == "percentile":
        percentile = calibration.get("percentile")
        _require(
            type(percentile) in (int, float) and 0 < percentile <= 100,
            "a percentile calibration states its percentile in (0, 100]",
        )
    else:
        _require("percentile" not in calibration, "a percentile is stated only for a percentile calibration")
    recovery = body.get("recovery")
    _require(
        isinstance(recovery, Mapping)
        and set(recovery) == set(RECOVERY)
        and all(type(v) is bool for v in recovery.values()),
        f"recovery states each of {list(RECOVERY)} as true or false",
    )
    island = body.get("float_island")
    _require(isinstance(island, Mapping), "float_island states the host format and its operations")
    _require(set(island) <= {"dtype", "ops", "host"}, "float_island has unknown fields")
    _require(island.get("dtype") == ISLAND_DTYPE, f"float_island.dtype is {ISLAND_DTYPE}: never bf16 or f16")
    ops = island.get("ops")
    _require(
        isinstance(ops, list)
        and bool(ops)
        and all(isinstance(o, str) and o in ISLAND_OPS for o in ops)
        and len(set(ops)) == len(ops),
        f"float_island.ops are distinct names from {list(ISLAND_OPS)}",
    )
    if "host" in island:
        _require(island["host"] == "merlin", "float_island.host is merlin")
    variants = body.get("model_variants", [])
    _require(isinstance(variants, list), "model_variants is a list")
    seen = set()
    for index, row in enumerate(variants):
        where = f"model_variants[{index}]"
        _require(isinstance(row, Mapping), f"{where} is a mapping")
        _require(not (set(row) - _VARIANT_FIELDS), f"{where} has unknown fields {sorted(set(row) - _VARIANT_FIELDS)}")
        _require(row.get("status") == "reviewed", f"{where}: a model variant exists only as a reviewed row")
        for key in ("model", "name", "reason", "reviewed_by"):
            _require(isinstance(row.get(key), str) and bool(row[key].strip()), f"{where}.{key} is stated")
        _measured(row.get("measured_accuracy"), f"{where}.measured_accuracy")
        changes = row.get("changes")
        _require(isinstance(changes, Mapping) and bool(changes), f"{where}.changes names what the variant changes")
        _require(set(changes) <= set(VARIANT_CHANGES), f"{where}.changes are among {list(VARIANT_CHANGES)}")
        if "families" in changes:
            families = changes["families"]
            _require(
                isinstance(families, list) and bool(families) and all(isinstance(f, str) for f in families),
                f"{where}.changes.families is a family list",
            )
        if "dynamic" in changes:
            _require(isinstance(changes["dynamic"], Mapping), f"{where}.changes.dynamic is a recipe dynamic block")
        if "calibration_count" in changes:
            count = changes["calibration_count"]
            _require(type(count) is int and count > 0, f"{where}.changes.calibration_count is a positive integer")
        if "scale_granularity" in changes:
            _require(
                changes["scale_granularity"] in SCALE_GRANULARITIES,
                f"{where}.changes.scale_granularity is one of {list(SCALE_GRANULARITIES)}",
            )
        if "bias" in changes:
            _require(changes["bias"] in BIASES, f"{where}.changes.bias is one of {list(BIASES)}")
        if "static_planes" in changes:
            planes = changes["static_planes"]
            template = QR.static_planes_block()
            _require(
                isinstance(planes, Mapping) and set(planes) == set(template),
                f"{where}.changes.static_planes states exactly {sorted(template)}",
            )
            _require(
                planes["schema"] == template["schema"] and planes["families"] == template["families"],
                f"{where}.changes.static_planes has the static contraction-plane schema",
            )
            _require(
                planes["owners"] is None or isinstance(planes["owners"], list),
                f"{where}.changes.static_planes.owners is a list or null",
            )
            try:
                QR.static_planes_block(planes=planes["planes"], observer=planes["observer"], owners=planes["owners"])
            except ValueError as error:
                _require(False, f"{where}.changes.static_planes: {error}")
        key = (row["model"], row["name"])
        _require(key not in seen, f"{where} repeats model variant {key}")
        seen.add(key)
    return body


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def variants_for(methodology: Mapping[str, Any], model: str | None) -> list[dict[str, Any]]:
    """The reviewed variant rows of ``model`` (none for a model-independent recipe)."""
    if model is None:
        return []
    return [dict(row) for row in methodology.get("model_variants") or () if row.get("model") == model]


def _format(token: Any) -> str | None:
    from merlin.common import quant_formats as qf

    text = str(token or "")
    for candidate in (text, "int" + text[1:] if text[:1] == "i" and text[1:].isdigit() else text):
        if qf.has(candidate):
            return qf.get(candidate).name
    return None


def _stage(path: Mapping[str, Any], kind: str) -> dict[str, Any] | None:
    found = [dict(s) for s in path.get("readout") or () if s.get("kind") == kind]
    return found[0] if len(found) == 1 else None


def _preload(path: Mapping[str, Any]) -> dict[str, Any] | None:
    found = [dict(s) for s in path.get("accumulator_stages") or () if s.get("kind") == "preload"]
    return found[0] if len(found) == 1 else None


#: Variant blocks a base recipe may carry that this derivation does not take over silently.
_VARIANT_BLOCKS = ("dynamic", "static_planes")


def derive(
    base: Mapping[str, Any],
    methodology: Mapping[str, Any],
    output_path: Mapping[str, Any] | None,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """The recipe ``methodology`` derives from ``base`` and the target's ``output_path``.

    ``base`` is a derived ``quant_recipe_v1`` (formats, families, software admission). Its variant
    blocks are dropped: a model-specific change comes back only through ``model``'s reviewed
    ``model_variants`` rows. The result is a new ``quant_recipe_v1`` with an ``epilogue`` block, or the
    same document with ``status: underivable`` and the reasons.
    """
    method = validate(methodology)
    selected_granularity = method["scale_granularity"]
    selected_bias = method["bias"]
    for variant in variants_for(method, model):
        selected_granularity = variant["changes"].get("scale_granularity", selected_granularity)
        selected_bias = variant["changes"].get("bias", selected_bias)
    if base.get("schema") != QR.SCHEMA or base.get("status") != QR.DERIVED:
        raise ValueError("a methodology derives from a derived quant_recipe_v1")
    body = json.loads(json.dumps(dict(base)))
    for block in _VARIANT_BLOCKS:
        body.pop(block, None)
    for key in ("rounding", "gptq"):
        (body.get("weight") or {}).pop(key, None)
    body["families"] = [f for f in body.get("families") or () if f != QR.ACTIVATION_CONTRACTION]
    body["why"] = {
        key: value
        for key, value in (body.get("why") or {}).items()
        if not key.startswith("variant") and key not in _VARIANT_BLOCKS and not str(value).startswith("VARIANT")
    }
    underivable: dict[str, str] = {}
    weight, activation = body["weight"], body["activation"]
    if output_path is None:
        underivable["output_path"] = "the selected contract declares no derived output path"
        return _finish(body, method, None, underivable, model)
    try:
        path = OP.validate(output_path)
    except ValueError as error:
        underivable["output_path"] = str(error)
        return _finish(body, method, None, underivable, model)
    if _format((path.get("accumulator") or {}).get("dtype")) != "int32":
        underivable["accumulator.dtype"] = "the integer epilogue requires a declared int32 accumulator"
    order = [s["kind"] for s in path.get("readout") or ()]
    if len(order) != len(set(order)):
        underivable["output_path.order"] = "the integer epilogue cannot realize repeated readout stages"
    convert, scale, requant = _stage(path, "convert"), _stage(path, "scale"), _stage(path, "requant")
    activation_stage, pool = _stage(path, "activation"), _stage(path, "pool")
    exits = dict(path.get("exits") or {})
    if scale is None or convert is None:
        underivable["scale"] = "the output path applies no converted scale"
        return _finish(body, method, path, underivable, model)
    if activation_stage is not None:
        a = order.index("activation")
        if (
            activation_stage["domain"] == "accumulator"
            and a > order.index("convert")
            or activation_stage["domain"] == "scaled"
            and a < order.index("scale")
        ):
            underivable["output_path.order"] = "the activation's position contradicts its declared domain"
    # Rounding: the path's own, realized only where the graph spells it exactly.
    roundings = {s["kind"]: s.get("rounding") for s in (convert, scale, requant) if s is not None}
    inexact = {kind: r for kind, r in roundings.items() if r not in _EXACT_ROUNDINGS}
    if inexact:
        underivable["rounding"] = (
            f"the output path rounds {inexact}; the capture graph spells round-half-to-even only, "
            "so another rounding has no exact quantized form"
        )
    if _format(scale.get("dtype")) != "fp32" or _format(convert.get("to")) != "fp32":
        underivable["scale.dtype"] = (
            f"the output path scales in {scale.get('dtype')!r} after converting to {convert.get('to')!r}; "
            "the capture graph computes the scale in fp32 only"
        )
    # Scale granularity: the columns ONE scale covers.
    if selected_granularity == "per_tensor":
        weight.update(granularity="tensor")
        weight.pop("channels_per_scale", None)
        body["why"]["weight.granularity"] = "the methodology asks for one weight scale per tensor"
    elif selected_granularity == "per_channel":
        weight.update(granularity="channel")
        weight.pop("channels_per_scale", None)
        if scale.get("granularity") not in ("per_channel", "per_command"):
            underivable["scale.granularity"] = "per-channel source scales require a configurable readout gain"
        elif scale.get("granularity") == "per_command" and (
            type(scale.get("columns_per_scale")) is not int or scale["columns_per_scale"] < 1
        ):
            underivable["scale.columns_per_scale"] = "the physical readout scale-vector width is not derived"
        body["why"]["weight.granularity"] = (
            "explicit per-channel source weight scales; independent output partitions remain a compiler "
            "obligation under the unchanged physical readout width"
        )
    elif scale.get("granularity") == "per_channel":
        weight.update(granularity="channel")
        weight.pop("channels_per_scale", None)
        body["why"]["weight.granularity"] = "the output path holds one scale per output column"
    elif scale.get("granularity") == "per_command":
        columns = scale.get("columns_per_scale")
        if type(columns) is not int or columns < 1:
            underivable["scale.columns_per_scale"] = (
                "the output path does not establish how many columns one scale covers"
            )
        else:
            weight.update(granularity="channel", channels_per_scale=columns)
            body["why"]["weight.granularity"] = (
                f"the output path applies one scale per readout command of {columns} output columns; "
                "each run of that many output channels shares the largest range in it"
            )
    elif scale.get("granularity") == "per_tensor":
        weight.update(granularity="tensor")
        weight.pop("channels_per_scale", None)
        body["why"]["weight.granularity"] = "the output path holds one scale per tensor"
    # Clamp: the requantizing exit's integer range is every activation's grid.
    if requant is not None:
        clamp = requant.get("clamp")
        if _format(requant.get("to")) != _format(activation.get("dtype")):
            underivable["requant.to"] = (
                f"the output path narrows to {requant.get('to')!r}, "
                f"the recipe's activations are {activation.get('dtype')!r}"
            )
        elif not (isinstance(clamp, list) and len(clamp) == 2 and all(type(v) is int for v in clamp)):
            underivable["requant.clamp"] = "the output path's clamp is not an integer range"
        else:
            activation.update(quant_min=int(clamp[0]), quant_max=int(clamp[1]))
            body["why"]["activation.range"] = f"the output path clamps its requantized readout to {clamp}"
    elif exits.get("requant"):
        underivable["requant"] = "the output path declares a requantizing exit without its requant stage"
    # Calibration: methods are choices, not hardware facts.
    calibration = method["calibration"]
    activation["observer"] = {"mse": "histogram", "minmax": "minmax"}.get(
        calibration["activation"], f"p{calibration.get('percentile')}"
    )
    weight["observer"] = calibration["weight"]
    body["why"]["activation.observer"] = f"methodology calibration {calibration['activation']!r}" + (
        " (TorchAO's histogram observer minimizes the L2 quantization error)"
        if calibration["activation"] == "mse"
        else ""
    )
    # Bias: preloaded into the accumulator as an i32 code, where the path loads one.
    preload = _preload(path)
    if (
        selected_bias == "accumulator_preload_int32"
        and preload is not None
        and _format(preload.get("dtype")) == "int32"
    ):
        bias = {
            "mode": "accumulator_preload",
            "dtype": "int32",
            "scale": "s_x * s_w",
            "rounding": "half_even",
            "range": list(preload.get("range") or ()),
        }
        if bias["range"] != [-(2**31), 2**31 - 1]:
            underivable["bias.range"] = "the preload must establish the complete int32 code range"
        body["bias_domain"] = "accumulator"
        body["why"]["bias_domain"] = (
            "the output path preloads an i32 value into the accumulator: each bias is the code "
            "round_half_even(b / (s_x * s_w)), refused outside i32"
        )
    else:
        reason = (
            "the methodology keeps biases as host float adds"
            if selected_bias == "host_float"
            else "the output path preloads no i32 value, so a bias cannot ride in the accumulator"
        )
        bias = {"mode": "host_float", "why": reason}
        body["bias_domain"] = None
        body["why"]["bias_domain"] = reason
    epilogue = {
        "schema": EPILOGUE_SCHEMA,
        "order": [s["kind"] for s in path.get("readout") or ()],
        "activation": (
            {"function": activation_stage.get("function"), "domain": activation_stage.get("domain")}
            if activation_stage is not None
            else None
        ),
        "convert": {"to": _format(convert.get("to")), "rounding": convert.get("rounding")},
        "scale": {
            "dtype": _format(scale.get("dtype")),
            "rounding": scale.get("rounding"),
            "granularity": scale.get("granularity"),
            "columns_per_scale": scale.get("columns_per_scale"),
        },
        "requant": (
            {
                "to": _format(requant.get("to")),
                "rounding": requant.get("rounding"),
                "clamp": list(requant.get("clamp") or ()),
            }
            if requant is not None
            else None
        ),
        "pool": (
            {"reductions": list(pool.get("reductions") or ()), "padding": pool.get("padding")}
            if pool is not None
            else None
        ),
        "bias": bias,
        "exits": {"requant": bool(exits.get("requant")), "accumulator": bool(exits.get("accumulator"))},
    }
    if epilogue["activation"] is not None and epilogue["activation"]["function"] != "relu":
        epilogue["activation"] = None  # only a rectifier is a stage the graph spells; others stay on the host
    body["epilogue"] = epilogue
    if selected_granularity == "per_channel" and scale.get("granularity") == "per_command":
        epilogue["scale"]["logical_granularity"] = "per_channel"
    body["why"]["epilogue"] = (
        "integer-domain epilogue in the output path's order: the contraction and the stages the path performs "
        "after it are one pattern, computed as the unit does (integer operands, accumulator plus preloaded bias, "
        "rectifier, one f32 multiplier per readout command, round half to even, clamp)"
    )
    body["recovery"] = dict(method["recovery"])
    body["float_island"] = {"dtype": ISLAND_DTYPE, "ops": list(method["float_island"]["ops"])}
    return _finish(body, method, path, underivable, model)


def _finish(body, method, path, underivable, model) -> dict[str, Any]:
    body["methodology"] = {
        "schema": SCHEMA,
        "methodology_sha256": digest({k: v for k, v in method.items() if k != "model_variants"}),
        "output_path_sha256": digest(path) if path is not None else None,
        "model": model,
    }
    selected_planes = None
    for row in variants_for(method, model):
        changes = row["changes"]
        if "families" in changes:
            dropped = [f for f in body.get("families") or () if f not in changes["families"]]
            body["families"] = list(dict.fromkeys(changes["families"]))
            unquantized = body.setdefault("unquantized", {})
            for family in body["families"]:
                unquantized.pop(family, None)
            for family in dropped:
                unquantized[family] = f"reviewed model variant {row['name']}: left in floating point on the host"
        if "dynamic" in changes:
            body["dynamic"] = json.loads(json.dumps(changes["dynamic"]))
        if "calibration_count" in changes:
            body.setdefault("framework_capture_policy", {})["calibration_count"] = changes["calibration_count"]
        if "static_planes" in changes:
            selected_planes = changes["static_planes"]
        body["methodology"].setdefault("variants", []).append(
            {
                "name": row["name"],
                "reason": row["reason"],
                "reviewed_by": row["reviewed_by"],
                "changes": sorted(changes),
            }
        )
        body["why"][f"variant.{row['name']}"] = f"reviewed model variant: {row['reason']}"
    if selected_planes is not None:
        # Validate the final effective policy: later rows can change families or
        # bias placement. Digit planes combine accumulators before a host bias add.
        try:
            if (body.get("epilogue") or {}).get("bias", {}).get("mode") != "host_float":
                raise ValueError("static planes require host_float bias placement after combining accumulators")
            body = QR.with_static_planes(
                body,
                planes=selected_planes["planes"],
                observer=selected_planes["observer"],
                owners=selected_planes["owners"],
            )
        except ValueError as error:
            body[QR.STATIC_PLANES] = json.loads(json.dumps(selected_planes))
            underivable[QR.STATIC_PLANES] = str(error)
    if underivable:
        body["status"] = QR.UNDERIVABLE
        body["underivable"] = dict(sorted({**(body.get("underivable") or {}), **underivable}.items()))
    body["why"] = dict(sorted(body["why"].items()))
    body["recipe_sha256"] = QR.digest(body)
    return body


def output_path_of(contract: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The contract's derived output path, or None when it declares none."""
    record = (contract or {}).get("output_epilogue") or {}
    path = record.get("output_path") if isinstance(record, Mapping) else None
    return dict(path) if isinstance(path, Mapping) else None


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m merlin.targetgen.quant_methodology --spec S --contract C --base B [--model M] --out NEW``."""
    import argparse
    from pathlib import Path

    from merlin.targetgen.software_spec import load_software_spec

    parser = argparse.ArgumentParser(prog="python -m merlin.targetgen.quant_methodology", description=derive.__doc__)
    parser.add_argument(
        "--spec", type=Path, required=True, help="the target's software spec (declares the methodology)"
    )
    parser.add_argument(
        "--contract", type=Path, required=True, help="selected contract carrying the derived output path"
    )
    parser.add_argument("--base", type=Path, required=True, help="the hardware-derived base recipe")
    parser.add_argument("--model", default=None, help="apply this model's reviewed variant rows")
    parser.add_argument("--out", type=Path, required=True, help="a NEW file; an existing one is never overwritten")
    args = parser.parse_args(argv)
    spec = load_software_spec(args.spec)
    methodology = (spec.get("quantization") or {}).get("methodology")
    if methodology is None:
        parser.error(f"{args.spec} declares no quantization.methodology")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    base = json.loads(args.base.read_text(encoding="utf-8"))
    body = derive(base, methodology, output_path_of(contract), model=args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(f"{args.out}: {body['status']} recipe_sha256 {body['recipe_sha256']}")
    return 0 if body["status"] == QR.DERIVED else 1


if __name__ == "__main__":
    raise SystemExit(_main())
