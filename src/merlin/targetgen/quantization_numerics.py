"""Explicit per-scheme quantization numerics, with evidence and distinguishing boundary witnesses.

A declared scheme (``quantization.schemes``) names axis points; it does not say how an int8 code
becomes an int32 accumulator and back. This block states that for every scheme, property by
property -- scale value and granularity, zero points, signedness, operand and accumulator types,
internal accumulation, bias width, the order of conversions, rounding, saturation and host/device
transfer -- and records the evidence each value rests on. Internal arithmetic is never inferred
from int8 storage and an int32 result: it is declared, with its own evidence.

Each audited boundary carries a small input that distinguishes the declared regime from a named
mismatched one. :func:`evaluate` is an independent exact model of the conversion steps (rational
arithmetic, correctly rounded binary32), so a witness is checked, not asserted: the declared order
must produce the declared result, and the rejected regime a different, declared one.

Nothing here names a target. Values are the spec's declarations; ``rtl_derived`` claims are checked
against the selected RTL facts by :func:`audit`, and an unconfirmable claim fails closed.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction

SCHEMA = "merlin.quantization_numerics.v1"
#: How a declared value is known. ``rtl_derived`` must name a fact path that :func:`audit` checks;
#: ``host_definition`` is arithmetic the host program itself defines (no hardware claim).
EVIDENCE = frozenset(
    {"rtl_derived", "rtl_execution_verified", "spike_verified", "host_definition", "declared_unverified"}
)
UNCONFIRMED = "rtl_claim_unconfirmed"
#: Every scheme declares each of these properties.
PROPERTIES = (
    "scale_value",
    "scale_granularity",
    "zero_point",
    "signedness",
    "operand_type",
    "accumulator_type",
    "internal_accumulation",
    "bias",
    "conversion_order",
    "rounding",
    "saturation",
    "transfer",
)
# Policies describe where scales come from, independently of arithmetic scheme rows.
POLICIES = {"scale_source": ("constant", "runtime_input"), "block_scale_reduction": ("max_abs", "mean_abs")}
_TIES = frozenset({"half_even", "half_away_from_zero", "toward_zero"})
_INTEGER_RANGES = {"int8": (-128, 127), "uint8": (0, 255), "int32": (-(2**31), 2**31 - 1)}
_STEPS = frozenset(
    {
        "quantize",
        "transfer",
        "seed",
        "multiply_accumulate",
        "operand_sum",
        "accumulate",
        "activation",
        "convert",
        "multiply",
        "round",
        "saturate",
    }
)
REFUSED = "refused"
#: The conversion paths a scheme orders: a contraction readout and a scaled operand sum.
PATHS = ("contraction", "operand_sum")


class Refused(Exception):
    """The declared regime refuses this input (it lies outside the admitted domain)."""


def _dtype_name(value: str) -> str:
    from merlin.common import quant_formats

    try:
        return quant_formats.get(value).name
    except (KeyError, ValueError, TypeError):
        return {"ui8": "uint8", "u8": "uint8"}.get(value, value)


def _range(dtype: str) -> tuple[int, int]:
    name = _dtype_name(dtype)
    if name not in _INTEGER_RANGES:
        raise ValueError(f"no integer range for {dtype!r}")
    return _INTEGER_RANGES[name]


def _rational(value) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"witness values are integers or exact rational strings, not {value!r}")
    return Fraction(value)


def _round(value: Fraction, ties: str) -> int:
    floor = value.numerator // value.denominator
    remainder = value - floor
    if remainder != Fraction(1, 2):
        return round(value) if ties != "toward_zero" else int(value)
    if ties == "half_even":
        return floor if floor % 2 == 0 else floor + 1
    if ties == "half_away_from_zero":
        return floor + 1 if value > 0 else floor
    return int(value)


def _f32(value: Fraction) -> Fraction:
    from merlin.targetgen.exact_arithmetic import round_binary

    return Fraction(round_binary(value))


def _fit(value: int, dtype: str, mode: str) -> int:
    low, high = _range(dtype)
    if low <= value <= high:
        return value
    if mode == "saturate":
        return min(high, max(low, value))
    if mode == "wrap":
        span = high - low + 1
        return (value - low) % span + low
    raise Refused(f"{value} exceeds {dtype}")


def _step(state: dict, step: Mapping, witness: Mapping) -> None:
    kind = step["step"]
    if kind == "quantize":
        scale = _rational(witness["scale"])
        zero = int(step.get("zero_point", 0))
        state["values"] = [
            _fit(_round(_rational(x) / scale, step["rounding"]) + zero, step["dtype"], "saturate")
            for x in witness["real"]
        ]
    elif kind == "transfer":
        for value in state["values"]:
            if value.denominator != 1:
                raise Refused("a transfer carries integer codes")
            _fit(int(value), step["dtype"], "refuse")
    elif kind == "seed":
        state["seed"] = _fit(int(witness.get("bias", 0)), step["dtype"], "saturate")
    elif kind == "multiply_accumulate":
        bits = int(step["internal_bits"])
        low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        total = 0
        for a, b in zip(witness["lhs"], witness["rhs"], strict=True):
            _fit(int(a), step["operand"], "refuse")
            _fit(int(b), step["operand"], "refuse")
            total += int(a) * int(b)
            if not low <= total <= high:
                if step["admission"] == "bounded_partial_sums":
                    raise Refused(f"partial sum {total} exceeds {bits} signed bits")
                if step["admission"] != "none":
                    raise ValueError(f"unknown accumulation admission {step['admission']!r}")
        state["values"] = [Fraction(total)]
    elif kind == "operand_sum":
        terms = [(_rational(x), _rational(m)) for x, m in witness["operands"]]
        if step["rounding"] == "per_operand":
            # Each load scales, rounds and saturates on its own before the integer sum.
            if step["load_multiplier"] == "rne_f32_of_exact_rational":
                terms = [(x, _f32(m)) for x, m in terms]
            products = [x * m for x, m in terms]
            if step["load_product"] == "f32_half_even":
                products = [_f32(p) for p in products]
            loads = [_fit(_round(p, step["ties"]), step["load_dtype"], "saturate") for p in products]
            state["values"] = [Fraction(sum(loads))]
        elif step["rounding"] == "single_rounding":
            state["values"] = [sum(x * m for x, m in terms)]
        else:
            raise ValueError(f"unknown operand sum rounding {step['rounding']!r}")
    elif kind == "accumulate":
        seed = state.pop("seed", 0)
        if any(v.denominator != 1 for v in state["values"]):
            raise Refused("an accumulator holds integers; an unrounded operand sum cannot be stored")
        state["values"] = [Fraction(_fit(int(v) + seed, step["dtype"], "refuse")) for v in state["values"]]
    elif kind == "activation":
        if witness.get("relu"):
            state["values"] = [max(v, Fraction(0)) for v in state["values"]]
    elif kind == "convert":
        if step["to"] != "f32" or step["rounding"] != "half_even":
            raise ValueError("convert models the f32 half-even boundary only")
        state["values"] = [_f32(v) for v in state["values"]]
    elif kind == "multiply":
        multipliers = [_rational(m) for m in witness["multipliers"]]
        if step["multiplier"] == "rne_f32_of_exact_rational":
            multipliers = [_f32(m) for m in multipliers]
        elif step["multiplier"] != "exact_rational":
            raise ValueError(f"unknown multiplier formation {step['multiplier']!r}")
        blocks = witness.get("column_blocks") or [0] * len(state["values"])
        if step["granularity"] == "tensor":
            blocks = [0] * len(state["values"])
        elif step["granularity"] != "column_block":
            raise ValueError(f"unknown scale granularity {step['granularity']!r}")
        products = [v * multipliers[block] for v, block in zip(state["values"], blocks, strict=True)]
        if step["product"] == "f32_half_even":
            products = [_f32(p) for p in products]
        elif step["product"] != "exact":
            raise ValueError(f"unknown product rounding {step['product']!r}")
        state["values"] = products
    elif kind == "round":
        state["values"] = [Fraction(_round(v, step["ties"])) for v in state["values"]]
    elif kind == "saturate":
        state["values"] = [Fraction(_fit(int(v), step["dtype"], step.get("mode", "saturate"))) for v in state["values"]]
    else:
        raise ValueError(f"unknown conversion step {kind!r}")


def evaluate(steps: Sequence[Mapping], witness: Mapping) -> list[int] | str:
    """Run conversion steps on one witness; integer results, or ``refused`` outside the domain."""
    state = {"values": [_rational(v) for v in witness.get("accumulators", [])]}
    try:
        for step in steps:
            _step(state, step, witness)
    except Refused:
        return REFUSED
    if any(value.denominator != 1 for value in state["values"]):
        raise ValueError("a witness segment must end on integer codes")
    return [int(value) for value in state["values"]]


def _segment(order: Sequence[Mapping], start: str, stop: str) -> list[Mapping]:
    names = [step["step"] for step in order]
    if start not in names or stop not in names:
        raise ValueError(f"conversion order lacks the witness segment {start}..{stop}")
    first = names.index(start)
    last = len(names) - 1 - names[::-1].index(stop)
    if last < first:
        raise ValueError(f"witness segment {start}..{stop} is out of order")
    return list(order[first : last + 1])


def _validate_steps(order: object, label: str) -> None:
    if not isinstance(order, list) or not order:
        raise ValueError(f"{label} must be a nonempty step list")
    for step in order:
        if not isinstance(step, Mapping) or step.get("step") not in _STEPS:
            raise ValueError(f"{label} steps must name one of {sorted(_STEPS)}")
        # An operand sum's ``rounding`` names where it rounds (per operand or once), not its ties.
        for key in ("ties",) if step["step"] == "operand_sum" else ("rounding", "ties"):
            if key in step and step[key] not in _TIES:
                raise ValueError(f"{label} {step['step']}.{key} must be one of {sorted(_TIES)}")


def _validate_property(record: object, label: str) -> None:
    if not isinstance(record, Mapping) or "value" not in record:
        raise ValueError(f"{label} must be a mapping with value, evidence and basis")
    if record.get("evidence") not in EVIDENCE:
        raise ValueError(f"{label}.evidence must be one of {sorted(EVIDENCE)}")
    if not isinstance(record.get("basis"), str) or not record["basis"].strip():
        raise ValueError(f"{label}.basis must state what the evidence is")
    fact = record.get("fact")
    if record["evidence"] == "rtl_derived":
        if not isinstance(fact, list) or not fact or any(not isinstance(p, str) or not p for p in fact):
            raise ValueError(f"{label} is rtl_derived and must name the fact path it is checked against")
    elif fact is not None or "fact_value" in record:
        raise ValueError(f"{label}.fact applies only to rtl_derived evidence")


def _consistent_types(order: Sequence[Mapping], declared: Mapping, label: str, strategy: str) -> None:
    """Bind every typed conversion to its property and preserve transfer code types."""
    values = {name: record["value"] for name, record in declared.items()}
    operand = values["operand_type"]["dtype"]
    accumulator = values["accumulator_type"]["dtype"]
    typed = {
        "quantize": ("dtype", operand),
        "seed": ("dtype", values["bias"]["dtype"]),
        "multiply_accumulate": ("operand", operand),
        "accumulate": ("dtype", accumulator),
        "saturate": ("dtype", values["saturation"]["dtype"]),
    }

    def require_type(actual, expected, where):
        if actual is None or expected is None or _dtype_name(actual) != _dtype_name(expected):
            raise ValueError(f"{label}.{where} type {actual!r} differs from declared {expected!r}")

    current = None
    for step in order:
        kind = step["step"]
        if kind in typed:
            field, expected = typed[kind]
            require_type(step.get(field), expected, f"{kind}.{field}")
            if kind in {"quantize", "accumulate", "saturate"}:
                current = expected
        if kind == "transfer":
            direction = f"{step.get('from')}_to_{step.get('to')}"
            transfer = values["transfer"].get(direction, {})
            require_type(step.get("dtype"), transfer.get("dtype"), f"transfer.{direction}")
            require_type(step.get("dtype"), current, "transfer.source")
        elif kind in {"multiply_accumulate", "operand_sum"}:
            require_type(current, operand, f"{kind}.input")
            if kind == "operand_sum" and step.get("rounding") == "per_operand":
                require_type(step.get("load_dtype"), operand, "operand_sum.load_dtype")
            current = None  # Internal partial sums acquire their storage type at accumulate.
        elif kind == "convert":
            if strategy != "f32_product_rne":
                raise ValueError(f"{label}.convert differs from readout {strategy!r}")
            require_type(current, accumulator, "convert.input")
            require_type(step.get("to"), "f32", "convert.to")
            current = step["to"]
        elif kind in {"multiply", "round"}:
            current = None  # Products and rounded integers need saturation before typed transfer.


def _consistent(scheme_id: str, row: Mapping, declared: Mapping, scheme: Mapping, semantics: Mapping) -> None:
    """The numerics restate the scheme's axis points and the selected readout; they cannot drift."""
    label = f"quantization.numerics.schemes.{scheme_id}"
    value = lambda name: declared[name]["value"]  # noqa: E731
    operand_range = _range(value("operand_type")["dtype"])
    if value("signedness") != "signed_twos_complement" or operand_range[0] >= 0:
        raise ValueError(f"{label}.signedness differs from the modeled signed operand type")
    for name in ("operand_type", "accumulator_type", "bias", "saturation"):
        record = value(name)
        low, high = _range(record["dtype"])
        if "range" in record and record["range"] != [low, high]:
            raise ValueError(f"{label}.{name}.range differs from its dtype")
        if "width" in record and record["width"] != high.bit_length() + int(low < 0):
            raise ValueError(f"{label}.{name}.width differs from its dtype")
    if _dtype_name(value("bias")["dtype"]) != _dtype_name(value("accumulator_type")["dtype"]):
        raise ValueError(f"{label}.bias differs from the accumulator type")
    if value("scale_value") != scheme["scale_value"]:
        raise ValueError(f"{label}.scale_value differs from the scheme axis point")
    granularity = value("scale_granularity")
    if not isinstance(granularity, Mapping) or granularity.get("weight") != scheme["weight_scale_granularity"]:
        raise ValueError(f"{label}.scale_granularity.weight differs from the scheme axis point")
    if (
        "activation_scale_granularity" in scheme
        and granularity.get("activation") != scheme["activation_scale_granularity"]
    ):
        raise ValueError(f"{label}.scale_granularity.activation differs from the scheme axis point")
    for name, field in (("operand_type", "operand_dtype"), ("accumulator_type", "accumulator_dtype")):
        if field in semantics and _dtype_name(value(name)["dtype"]) != _dtype_name(semantics[field]):
            raise ValueError(f"{label}.{name} differs from numerical_semantics.{field}")
    orders = value("conversion_order")
    if not isinstance(orders, Mapping) or set(orders) != set(PATHS):
        raise ValueError(f"{label}.conversion_order must declare the paths {list(PATHS)}")
    strategy = row.get("readout_strategy") or (semantics.get("readout") or {}).get("strategy")
    formation = {"exact_rational_host": "exact_rational", "f32_product_rne": "rne_f32_of_exact_rational"}.get(strategy)
    multiply_granularity = "column_block" if scheme["weight_scale_granularity"] != "tensor" else "tensor"
    for path, order in orders.items():
        _validate_steps(order, f"{label}.conversion_order.{path}")
        _consistent_types(order, declared, f"{label}.conversion_order.{path}", strategy)
        steps = {step["step"]: step for step in order}
        where = f"{label}.conversion_order.{path}"
        multiply = steps.get("multiply") or {}
        if formation is None or multiply.get("multiplier") != formation:
            raise ValueError(f"{where} multiplier formation differs from readout {strategy!r}")
        if strategy == "f32_product_rne" and ("convert" not in steps or multiply.get("product") != "f32_half_even"):
            raise ValueError(f"{where} lacks the f32 conversion and product of {strategy}")
        if steps.get("round", {}).get("ties") != value("rounding").get("ties"):
            raise ValueError(f"{label}.rounding differs from {where}")
        if steps.get("saturate", {}).get("dtype") != value("saturation").get("dtype"):
            raise ValueError(f"{label}.saturation differs from {where}")
        if path == "contraction":
            if multiply.get("granularity") != multiply_granularity:
                raise ValueError(f"{where} scale granularity differs from the scheme")
            mac = steps.get("multiply_accumulate") or {}
            internal = value("internal_accumulation")
            if mac.get("internal_bits") != internal.get("bits") or mac.get("admission") != internal.get("admission"):
                raise ValueError(f"{label}.internal_accumulation differs from {where}")
            if steps.get("seed", {}).get("dtype") != value("bias").get("dtype"):
                raise ValueError(f"{label}.bias differs from {where}")
        elif (steps.get("operand_sum") or {}).get("rounding") != scheme["operand_sum_rounding"]:
            raise ValueError(f"{where} operand_sum rounding differs from the scheme")


def validate(spec: Mapping) -> dict:
    """Validate the block against the declared schemes and check every boundary witness."""
    quantization = spec.get("quantization") or {}
    block = quantization.get("numerics")
    if not isinstance(block, Mapping) or block.get("schema") != SCHEMA:
        raise ValueError(f"quantization.numerics must declare schema {SCHEMA}")
    from merlin.targetgen.operator_review import STATUSES

    if block.get("status") not in STATUSES:
        raise ValueError("quantization.numerics has an invalid review status")
    unknown = set(block) - {"schema", "status", "schemes", "boundaries", "unmodeled", "policies"}
    if unknown:
        raise ValueError(f"quantization.numerics has unknown fields {sorted(unknown)}")
    schemes = {row["id"]: row for row in (quantization.get("schemes") or {}).get("schemes") or []}
    declared = block.get("schemes")
    if not isinstance(declared, Mapping) or set(declared) != set(schemes):
        raise ValueError("quantization.numerics.schemes must declare exactly the declared schemes")
    semantics = spec.get("numerical_semantics") or {}
    axes = (quantization.get("schemes") or {}).get("axes") or {}
    _validate_policies(block.get("policies", {}), axes, schemes)
    for scheme_id, properties in declared.items():
        if not isinstance(properties, Mapping) or set(properties) != set(PROPERTIES):
            raise ValueError(f"quantization.numerics.schemes.{scheme_id} must declare exactly {list(PROPERTIES)}")
        for name in PROPERTIES:
            _validate_property(properties[name], f"quantization.numerics.schemes.{scheme_id}.{name}")
        scheme = dict(schemes[scheme_id])
        for axis, values in axes.items():
            if axis not in scheme and len(values) == 1:
                scheme[axis] = values[0]
        _consistent(scheme_id, schemes[scheme_id], properties, scheme, semantics)
    boundaries = block.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("quantization.numerics.boundaries must list audited boundaries")
    seen = set()
    for boundary in boundaries:
        _check_boundary(boundary, declared, seen)
    covered = {scheme_id for boundary in boundaries for scheme_id in boundary["schemes"]}
    if covered != set(declared):
        raise ValueError(f"schemes without an audited boundary: {sorted(set(declared) - covered)}")
    unmodeled = block.get("unmodeled")
    if (
        not isinstance(unmodeled, list)
        or not unmodeled
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"subject", "reason"}
            or not all(isinstance(row[key], str) and row[key].strip() for key in row)
            for row in unmodeled
        )
    ):
        raise ValueError("quantization.numerics.unmodeled must list {subject, reason} entries")
    return copy.deepcopy(dict(block))


def _validate_policies(policies: Mapping, axes: Mapping, schemes: Mapping) -> None:
    if not isinstance(policies, Mapping) or set(policies) - set(POLICIES):
        raise ValueError("quantization.numerics.policies names unknown scale policies")
    for name, values in POLICIES.items():
        selected = name in axes or any(name in scheme for scheme in schemes.values())
        if selected and name not in policies:
            raise ValueError(f"quantization.numerics.policies requires {name} evidence")
        if name not in policies:
            continue
        records = policies[name]
        if not isinstance(records, Mapping) or set(records) != set(values):
            raise ValueError(f"quantization.numerics.policies.{name} requires {values}")
        for value, record in records.items():
            _validate_property(record, f"quantization.numerics.policies.{name}.{value}")
            if record["value"] != value:
                raise ValueError(f"quantization.numerics.policies.{name}.{value} differs from its policy")
            if "domain" in record:
                _validate_scale_domain(record["domain"], f"quantization.numerics.policies.{name}.{value}.domain")


def _validate_scale_domain(domain: object, label: str) -> None:
    """A declared value domain of a runtime scale input: a finite minimum and a nonfinite policy."""
    from merlin.targetgen.operator_review import STATUSES

    if not isinstance(domain, Mapping) or set(domain) != {"status", "minimum", "nonfinite", "basis"}:
        raise ValueError(f"{label} requires exactly status, minimum, nonfinite and basis")
    if domain["status"] not in STATUSES:
        raise ValueError(f"{label}.status must be a review status")
    if type(domain["minimum"]) not in (int, float) or not math.isfinite(domain["minimum"]):
        raise ValueError(f"{label}.minimum must be a finite number")
    if domain["nonfinite"] not in {"allow", "forbid"}:
        raise ValueError(f"{label}.nonfinite must be allow or forbid")
    if not isinstance(domain["basis"], str) or not domain["basis"].strip():
        raise ValueError(f"{label}.basis must state why the domain holds")


def _check_boundary(boundary: object, declared: Mapping, seen: set) -> None:
    if not isinstance(boundary, Mapping):
        raise ValueError("a numerics boundary must be a mapping")
    required = {"id", "property", "schemes", "path", "segment", "input", "expected", "rejected"}
    if set(boundary) != required:
        raise ValueError(f"numerics boundary requires exactly {sorted(required)}")
    label = f"quantization.numerics.boundaries.{boundary['id']}"
    if boundary["id"] in seen:
        raise ValueError(f"duplicate numerics boundary {boundary['id']!r}")
    seen.add(boundary["id"])
    if boundary["property"] not in PROPERTIES:
        raise ValueError(f"{label}.property must be one of {list(PROPERTIES)}")
    if not boundary["schemes"] or set(boundary["schemes"]) - set(declared):
        raise ValueError(f"{label}.schemes must name declared schemes")
    if boundary["path"] not in PATHS:
        raise ValueError(f"{label}.path must be one of {list(PATHS)}")
    segment = boundary["segment"]
    if not isinstance(segment, list) or len(segment) != 2:
        raise ValueError(f"{label}.segment must be [first step, last step]")
    rejected = boundary["rejected"]
    if not isinstance(rejected, Mapping) or set(rejected) != {"regime", "result"}:
        raise ValueError(f"{label}.rejected requires regime and result")
    _validate_steps(rejected["regime"], f"{label}.rejected.regime")
    refusal = evaluate(rejected["regime"], boundary["input"])
    if refusal != rejected["result"]:
        raise ValueError(f"{label}: the rejected regime gives {refusal}, not the declared {rejected['result']}")
    for scheme_id in boundary["schemes"]:
        order = declared[scheme_id]["conversion_order"]["value"][boundary["path"]]
        result = evaluate(_segment(order, *segment), boundary["input"])
        if result != boundary["expected"]:
            raise ValueError(f"{label}: scheme {scheme_id} gives {result}, not the declared {boundary['expected']}")
        if result == refusal:
            raise ValueError(f"{label}: the input does not distinguish {scheme_id} from the rejected regime")


def _fact(facts: Mapping, path: Sequence[str]):
    """Follow a fact path; a list step selects the member whose ``name`` is that step."""
    node = facts.get("facts", facts)
    for part in path:
        if isinstance(node, list):
            node = next((item for item in node if isinstance(item, Mapping) and item.get("name") == part), None)
        elif isinstance(node, Mapping):
            node = node.get(part)
        else:
            node = None
        if node is None:
            return None
    return node


def _claimed(record: Mapping):
    """The value the fact must equal: ``fact_value`` when the declared value is composite."""
    if "fact_value" in record:
        return record["fact_value"]
    value = record["value"]
    return value.get("dtype") if isinstance(value, Mapping) else value


def audit(block: Mapping, facts: Mapping | None) -> dict:
    """Per scheme and property, the evidence status after checking rtl_derived claims against facts."""
    table = {}
    records = dict(block["schemes"])
    records.update({f"policy.{name}": values for name, values in block.get("policies", {}).items()})
    for scheme_id, properties in records.items():
        row = {}
        for name, record in properties.items():
            status = record["evidence"]
            if status == "rtl_derived":
                observed = _fact(facts or {}, record["fact"])
                if observed is None or _dtype_name(str(observed)) != _dtype_name(str(_claimed(record))):
                    status = UNCONFIRMED
            row[name] = status
        table[scheme_id] = row
    return table
