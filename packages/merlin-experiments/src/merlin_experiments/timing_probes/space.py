"""The timing model's parameter space for one target, resolved against that target's RTL facts.

Three inputs, all data:

* the selected timing provider's **profile** (JSON): its price classes, each class's parameters, its
  rate groups and its resources;
* the target's **timing space** (YAML, ``merlin.timing_probe_space.v1``, in the target's data
  directory): how the provider decodes an instruction stream into those classes -- which RTL funct
  names a class covers, which configuration-state fields select it, and which decoded quantity each
  parameter multiplies;
* the selected **RTL facts**: funct codes by name, register-bundle field positions, geometry,
  datapath widths and elaborated features.

Every bit position, funct code and capacity comes from the facts. The timing space names facts
fields; it never states an offset. A name the facts do not carry leaves its class or quantity
UNKNOWN with the reason, never decoded at an assumed position.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from merlin.common.facts_view import interface as _interface
from merlin.targetgen.rocc.semantics import bundle_layouts, field_slot

SCHEMA = "merlin.timing_probe_space.v1"
STATUSES = ("required", "support_unknown", "unsupported", "undecodable", "out_of_domain")
#: Quantities a decoded command contributes. ``count`` is implicit for every class.
COUNT = "count"
FIXED = "fixed"


class SpaceError(ValueError):
    """The timing space, profile or facts cannot ground a parameter space."""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sha(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _walk(document: Any, path: list[str], what: str) -> Any:
    value = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise SpaceError(f"{what}: profile has no {'.'.join(path)}")
        value = value[key]
    return value


@dataclass(frozen=True)
class Slot:
    """One decoded field: which operand register, bit offset and width (all from the facts)."""

    operand: str
    offset: int
    width: int

    def extract(self, raw: int | None) -> int | None:
        if raw is None:
            return None
        return (raw >> self.offset) & ((1 << self.width) - 1)


@dataclass(frozen=True)
class PriceClass:
    name: str
    instructions: tuple[str, ...]
    state: dict[str, Any]
    quantities: dict[str, tuple[str, ...]]
    rates: dict[str, str]  # rate quantity -> this class's quantity that contributes to it
    status: str
    reason: str | None
    parameters: dict[str, str]  # profile parameter -> quantity it multiplies


@dataclass(frozen=True)
class Column:
    """One feature column of the whole-call linear design and the profile parameters it informs."""

    name: str
    kind: str  # fixed | class | rate
    price_class: str | None
    quantity: str
    parameters: tuple[str, ...]
    sources: tuple[str, ...] = ()


@dataclass
class TimingSpace:
    target: str
    classes: list[PriceClass]
    columns: list[Column]
    funct_names: dict[int, str]
    pseudo_classes: tuple[str, ...]
    config_instruction: str | None
    selector: Slot | None
    states: dict[str, dict[str, Any]]
    reset: dict[str, int]
    operand_slots: dict[str, dict[str, Slot]]
    constants: dict[str, int]
    resources: list[str]
    explanations: list[dict]
    identity: dict[str, str]
    data_dir: Path
    document: dict = field(repr=False)
    layouts: dict = field(default_factory=dict, repr=False)

    def by_name(self) -> dict[str, PriceClass]:
        return {c.name: c for c in self.classes}

    def pack(self, bundle: str, values: dict[str, int]) -> int:
        """Encode ``values`` into one operand word at the facts' positions; refuse unknown or oversized fields."""
        word = 0
        for name, value in values.items():
            slot = field_slot(self.layouts.get(bundle), name)
            if slot is None:
                raise SpaceError(f"facts carry no layout for {bundle}.{name}")
            if type(value) is not int or not 0 <= value < (1 << slot[1]):
                raise SpaceError(f"{bundle}.{name}={value!r} does not fit its {slot[1]}-bit field")
            word |= value << slot[0]
        return word

    def state_slot(self, key: str) -> Slot | None:
        state, _, name = key.partition(".")
        return (self.states.get(state) or {}).get("slots", {}).get(name)


def _funct_table(body: dict) -> tuple[dict[int, str], dict]:
    table = _interface(body, "funct_decode_table")
    if not table:
        raise SpaceError("the RTL facts carry no funct_decode_table")
    legal = table.get("legal_funct")
    names = table.get("names") or {}
    if not isinstance(legal, list) or not legal or not isinstance(names, dict):
        raise SpaceError("funct_decode_table needs legal_funct and names")
    out = {}
    for code in legal:
        name = names.get(str(code))
        if type(code) is int and isinstance(name, str) and name:
            out[code] = name
    return out, table


def _bytes_of(dtype: Any) -> int | None:
    text = str(dtype or "")
    if len(text) < 2 or text[0] not in "iuf" or not text[1:].isdigit() or int(text[1:]) % 8:
        return None
    return int(text[1:]) // 8


def resolve_fact(body: dict, path: str) -> int | None:
    """``arrays.<name>.<key>``, ``memories.<name>.<key>`` or ``datapaths.<name>.bytes`` from the facts."""
    parts = path.split(".")
    if len(parts) != 3:
        return None
    group, name, key = parts
    rows = body.get(group) if isinstance(body.get(group), list) else []
    row = next((r for r in rows if isinstance(r, dict) and r.get("name") == name), None)
    if row is None:
        return None
    if group == "datapaths" and key == "bytes":
        return _bytes_of(row.get("dtype"))
    value = row.get(key)
    return value if type(value) is int and value >= 0 else None


def _slot(layouts: dict, operand: str, bundle: str, name: str) -> Slot | None:
    found = field_slot(layouts.get(bundle), name)
    return None if found is None else Slot(operand, found[0], found[1])


def _predicate_ok(value: Any) -> bool:
    if type(value) is int:
        return value >= 0
    if isinstance(value, list):
        return bool(value) and all(type(v) is int and v >= 0 for v in value)
    if isinstance(value, dict):
        return set(value) == {"not"} and type(value["not"]) is int
    return False


def load_space(path: str | Path, *, facts: dict, profile: dict, target: str | None = None) -> TimingSpace:
    """Resolve the timing space at ``path`` against ``facts`` and ``profile``; refuse inconsistent data."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = yaml.safe_load(raw)
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise SpaceError(f"{path}: expected a {SCHEMA} document")
    name = document.get("target")
    if not isinstance(name, str) or not name or (target is not None and name != target):
        raise SpaceError(f"{path}: timing space names target {name!r}, not {target!r}")
    body = facts.get("facts", facts) if isinstance(facts, dict) else {}
    declared = body.get("target")
    if declared not in (None, name):
        raise SpaceError(f"RTL facts describe {declared!r}, not {name!r}")
    functs, _ = _funct_table(body)
    by_funct_name = {v: k for k, v in functs.items()}
    layouts = bundle_layouts(body)
    explanations: list[dict] = []

    spec_profile = document.get("profile") or {}
    prices = _walk(profile, list(spec_profile.get("prices") or []), "prices")
    if not isinstance(prices, dict) or not prices:
        raise SpaceError("profile prices must be a nonempty mapping")
    parameter_quantities = spec_profile.get("parameter_quantities") or {}
    rate_groups = spec_profile.get("rate_groups") or {}
    rates: dict[str, tuple[str, str]] = {}  # rate quantity -> (group, parameter)
    for group, members in rate_groups.items():
        if group not in prices or not isinstance(members, dict):
            raise SpaceError(f"profile has no rate group {group!r}")
        for parameter, quantity in members.items():
            if parameter not in prices[group]:
                raise SpaceError(f"profile rate group {group!r} has no {parameter!r}")
            rates[str(quantity)] = (group, parameter)
    resources_path = spec_profile.get("resources")
    resources = sorted(_walk(profile, list(resources_path), "resources")) if resources_path else []

    decode = document.get("decode") or {}
    pseudo = tuple(str(x) for x in decode.get("pseudo_classes") or ())
    config = decode.get("config") or {}
    config_instruction = config.get("instruction")
    selector = None
    states: dict[str, dict[str, Any]] = {}
    if config_instruction is not None:
        if config_instruction not in by_funct_name:
            raise SpaceError(f"facts declare no funct named {config_instruction!r}")
        sel = config.get("selector") or {}
        selector = _slot(layouts, str(sel.get("operand", "rs1")), str(sel.get("bundle")), str(sel.get("field")))
        if selector is None:
            explanations.append(
                {
                    "subject": "config_state",
                    "status": "undecodable",
                    "reason": f"facts carry no {sel.get('bundle')}.{sel.get('field')} selector layout",
                }
            )
        for state, entry in (config.get("states") or {}).items():
            code = entry.get("code")
            if type(code) is not int or (selector is not None and not 0 <= code < (1 << selector.width)):
                raise SpaceError(f"config state {state!r} needs a code inside the selector field")
            slots: dict[str, Slot] = {}
            for operand in ("rs1", "rs2"):
                bundle = entry.get(operand)
                if bundle is None:
                    continue
                for fname in (layouts.get(bundle) or {}).get("fields") or {}:
                    slot = _slot(layouts, operand, bundle, fname)
                    if slot is not None and fname not in slots:
                        slots[fname] = slot
            states[str(state)] = {"code": code, "slots": slots, "evidence": entry.get("evidence")}
    reset = {str(k): int(v) for k, v in (config.get("reset") or {}).items()}
    operand_slots: dict[str, dict[str, Slot]] = {}
    for instruction, operands in (decode.get("operands") or {}).items():
        if instruction not in by_funct_name:
            continue  # every class issuing it is reported undecodable below
        slots = {}
        for operand, bundle in (operands or {}).items():
            for fname in (layouts.get(bundle) or {}).get("fields") or {}:
                slot = _slot(layouts, str(operand), str(bundle), fname)
                if slot is not None:
                    slots[f"{operand}.{fname}"] = slot
        operand_slots[str(instruction)] = slots

    features = (_interface(body, "elaborated_rtl_features") or {}).get("features") or {}
    constants: dict[str, int] = {}
    classes: list[PriceClass] = []
    seen: set[str] = set()
    for entry in document.get("classes") or ():
        cname = str(entry.get("name"))
        if cname in seen:
            raise SpaceError(f"duplicate class {cname!r}")
        seen.add(cname)
        if cname not in prices or cname in rate_groups:
            raise SpaceError(f"class {cname!r} is not a price class of the selected profile")
        params = {p: str(parameter_quantities.get(p)) for p in prices[cname] if p in parameter_quantities}
        instructions = tuple(str(x) for x in entry.get("instructions") or ())
        state = dict(entry.get("state") or {})
        quantities = {str(q): tuple(str(t) for t in terms) for q, terms in (entry.get("quantities") or {}).items()}
        class_rates = {str(r): str(q) for r, q in (entry.get("rates") or {}).items()}
        if any(q not in quantities for q in class_rates.values()):
            raise SpaceError(f"class {cname!r} feeds a rate from a quantity it does not declare")
        status, reason = "required", None
        if entry.get("undecodable"):
            status, reason = "undecodable", str(entry["undecodable"])
        elif entry.get("role") == "instrumentation":
            status, reason = "out_of_domain", "instrumentation command; compiled kernels in the domain do not issue it"
        else:
            missing = [i for i in instructions if i not in by_funct_name and i not in pseudo]
            bad_state = [k for k, v in state.items() if not _predicate_ok(v) or states.get(k.partition(".")[0]) is None]
            unknown_state = [
                k
                for k in state
                if k not in bad_state
                and (states.get(k.partition(".")[0]) or {}).get("slots", {}).get(k.partition(".")[2]) is None
            ]
            if not instructions:
                raise SpaceError(f"class {cname!r} lists no instructions")
            if missing:
                status, reason = "undecodable", f"facts declare no funct named {missing}"
            elif bad_state:
                raise SpaceError(f"class {cname!r} has malformed state predicates {bad_state}")
            elif unknown_state:
                status, reason = "undecodable", f"facts carry no layout for state fields {unknown_state}"
            elif state and selector is None:
                status, reason = "undecodable", "configuration state cannot be decoded without a selector layout"
            for feature in entry.get("requires_features") or ():
                if status != "required":
                    break
                value = features.get(feature)
                if value is False:
                    status, reason = "unsupported", f"elaborated_rtl_features.{feature} is false"
                elif value is not True:
                    status, reason = (
                        "support_unknown",
                        f"elaborated_rtl_features does not state {feature!r}; a probe establishes support",
                    )
        for q, terms in quantities.items():
            for term in terms:
                head, _, rest = term.partition(".")
                if head == "fact":
                    value = resolve_fact(body, rest)
                    if value is None:
                        explanations.append(
                            {"subject": f"{cname}.{q}", "status": "unknown", "reason": f"facts carry no {rest}"}
                        )
                    else:
                        constants[term] = value
                elif head == "state":
                    state_name, _, fname = rest.partition(".")
                    if (states.get(state_name) or {}).get("slots", {}).get(fname) is None:
                        explanations.append(
                            {
                                "subject": f"{cname}.{q}",
                                "status": "unknown",
                                "reason": f"facts carry no layout for state field {rest}",
                            }
                        )
                elif head not in ("rs1", "rs2"):
                    raise SpaceError(f"class {cname!r} quantity term {term!r} is not fact.*, state.* or rs1/rs2.*")
        classes.append(PriceClass(cname, instructions, state, quantities, class_rates, status, reason, params))
    for cname in prices:
        if cname not in seen and cname not in rate_groups:
            classes.append(
                PriceClass(
                    cname,
                    (),
                    {},
                    {},
                    {},
                    "undecodable",
                    "the timing space declares no decode for this profile class",
                    {},
                )
            )

    columns = [Column(FIXED, "fixed", None, FIXED, ("fixed invocation cycles",))]
    for cls in classes:
        if cls.status != "required":
            continue
        for quantity in (COUNT, *cls.quantities):
            aliased = tuple(f"{cls.name}.{p}" for p, q in cls.parameters.items() if q == quantity)
            columns.append(Column(f"{cls.name}.{quantity}", "class", cls.name, quantity, aliased))
    for quantity, (group, parameter) in sorted(rates.items()):
        sources = tuple(
            f"{c.name}.{c.rates[quantity]}" for c in classes if quantity in c.rates and c.status == "required"
        )
        columns.append(Column(f"rate.{quantity}", "rate", None, quantity, (f"{group}.{parameter}",), sources))
    identity = {
        "timing_space_sha256": _sha(raw),
        "profile_sha256": canonical_sha(profile),
        "facts_sha256": canonical_sha(facts),
    }
    return TimingSpace(
        target=name,
        classes=classes,
        columns=columns,
        funct_names=functs,
        pseudo_classes=pseudo,
        config_instruction=config_instruction,
        selector=selector,
        states=states,
        reset=reset,
        operand_slots=operand_slots,
        constants=constants,
        resources=resources,
        explanations=explanations,
        identity=identity,
        data_dir=path.parent,
        document=document,
        layouts=layouts,
    )


def parameter_explanations(space: TimingSpace) -> list[dict]:
    """Parameters no whole-call label can identify, and why -- independent of any measurement."""
    rows = list(space.explanations)
    for cls in space.classes:
        if cls.status != "required":
            rows.append({"subject": cls.name, "status": cls.status, "reason": cls.reason})
            continue
        for parameter, quantity in cls.parameters.items():
            if quantity != COUNT and quantity not in cls.quantities:
                rows.append(
                    {
                        "subject": f"{cls.name}.{parameter}",
                        "status": "not_applicable",
                        "reason": f"the class's decoded commands carry no {quantity!r} quantity",
                    }
                )
    for column in space.columns:
        if len(column.parameters) > 1:
            rows.append(
                {
                    "subject": column.name,
                    "status": "aliased",
                    "parameters": list(column.parameters),
                    "reason": "these parameters multiply the same decoded quantity; whole-call labels "
                    "identify only their sum (separating them needs the event model)",
                }
            )
        if column.kind == "rate":
            if len(column.sources) == 1:
                reason = (
                    f"equals {column.sources[0]} by construction: whole-call labels identify only the "
                    "combined per-unit price"
                )
            else:
                reason = (
                    f"is the sum of the per-class quantities of {list(column.sources)} by construction: "
                    "structurally dependent on those columns"
                )
            rows.append(
                {
                    "subject": column.name,
                    "status": "structurally_dependent",
                    "parameters": list(column.parameters),
                    "reason": reason,
                }
            )
    for resource in space.resources:
        rows.append(
            {
                "subject": f"resource.{resource}",
                "status": "requires_event_model",
                "reason": "queue/lane capacities shape overlap, not a linear whole-call price; take them "
                "from structural RTL facts or fit them with the event model",
            }
        )
    return rows
