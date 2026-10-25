"""What a contraction unit's OUTPUT path does to an accumulator on its way to memory, derived.

A quantized model leaves every contraction with the same chain: convert the accumulator, scale it,
add a bias, rectify, requantize, sometimes add a residual or pool a window. Whether the unit performs
those stages as it moves the accumulator out, or the host does them afterwards, decides where most
of a model's per-element work runs. The answer is written in the design, and this module reads it
there instead of taking it from a declaration somebody typed.

FOUR CAPABILITIES, each a generic property any accelerator's output path can have:

``scale``
    The accumulator is multiplied by a per-command scale on the way out. Derived: the command-register
    field that carries the ``scale`` role and its width (register layouts), the scale's NUMBER FORMAT
    read from how the elaborated readout splits it into sign, exponent and fraction fields
    (FIRRTL structure, looked up in the format registry by its field widths), the accumulator format
    (extracted datapath) and the output formats.
``activation``
    A function applied on move-out, selected by a command field. Derived only from the elaborated
    readout: a value selected by ``mux(eq(<activation field>, c), max(x, 0), x)`` is a rectifier at
    code ``c``. Codes the readout compares and whose function is not recognized are listed and never
    claimed.
``accumulate``
    An operand added into the accumulator (a residual summed by accumulating rather than on the host).
    Derived from the ``accumulate_on_load`` behavioural fact (:mod:`.accumulate_fact`) when a semantic
    facts document carrying it is supplied; otherwise UNKNOWN with the reason.
``pool``
    A window reduction applied on move-out. Derived from the elaborated build gate the facts record
    (``elaborated_rtl_features``) and the command fields carrying the ``pool`` role; the reduction is
    read from the tokens of the target's own feature name against a closed, target-neutral vocabulary.

NOTHING HERE NAMES A TARGET. The module and signal names a FIRRTL reading needs are the target's own
data, declared in its contract under ``rtl_extraction.output_epilogue`` (``module``, ``value``,
``scale``, ``activation``); field roles come from the target's own register field names. A rung that
cannot run leaves its fields ``None`` with the reason under ``unknown``: never a default, never a
literal, and a stage is only ever reported ``derived`` from structural evidence.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "output_epilogue_v1"
#: The contract key the derived record is declared under, and the ``rtl_extraction`` probe key.
FIELD = "output_epilogue"
DERIVED = "derived"
UNKNOWN = "UNKNOWN"
CAPABILITIES = ("scale", "activation", "accumulate", "pool")
#: Capabilities a record states when its deriver knows them; a record written before one existed
#: omits it, which reads as UNKNOWN.
OPTIONAL_CAPABILITIES = ("bias_preload", "full_width_readout")
#: Command-buffer ABI stage -> the capability that performs it on move-out. ``bias`` is absent on
#: purpose: an output path with no bias operand applies none, and a bias seeded into the accumulator
#: before the contraction is a separately declared stage route, not an output-path fact.
STAGE_CAPABILITY: dict[str, str] = {"acc_scale": "scale", "relu": "activation", "maxpool": "pool"}
#: Rectifier ``max(x, 0)``: the activation function this module can recognise structurally.
RECTIFIER = "relu"
#: Reduction kinds, read from the tokens of a target's own feature name. Closed and target-neutral.
_REDUCTION_TOKENS: dict[str, str] = {"max": "max", "min": "min", "avg": "mean", "mean": "mean", "sum": "sum"}
#: Comparison primitives of FIRRTL, as ``op -> (holds when lhs > rhs, holds when equal)``.
_COMPARE = {"gt": (True, False), "geq": (True, True), "lt": (False, False), "leq": (False, True)}
#: Primitives that only reinterpret a literal's type.
_RECAST = ("asSInt", "asUInt", "cvt")


def _body(facts: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(facts, Mapping):
        return {}
    inner = facts.get("facts")
    return inner if isinstance(inner, Mapping) else facts


def _interface(body: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for item in body.get("interfaces") or ():
        if isinstance(item, Mapping) and item.get("name") == name:
            return item
    return None


def _datapath(body: Mapping[str, Any], name: str) -> str | None:
    for key in ("storage_datapaths", "datapaths"):
        for row in body.get(key) or ():
            if isinstance(row, Mapping) and row.get("name") == name and isinstance(row.get("dtype"), str):
                return row["dtype"]
    return None


def _note(record: dict, capability: str, rung: str, observed: str) -> None:
    record[capability].setdefault("evidence", []).append({"rung": rung, "observed": observed})


# ------------------------------------------------------------------------------- register layouts


def _carriers(body: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    """``{role: [{bundle, field, bits}]}`` for the output-path roles, and why none could be read."""
    from merlin.targetgen.readout_facet import field_roles

    layouts = _interface(body, "register_bundle_layouts")
    if layouts is None:
        return {}, "the facts carry no command-register layouts"
    out: dict[str, list[dict[str, Any]]] = {}
    for bundle, layout in sorted((layouts.get("bundles") or {}).items()):
        for name, placed in sorted(((layout or {}).get("fields") or {}).items()):
            for role in field_roles(name):
                if role in ("scale", "activation", "pool"):
                    bits = (placed or {}).get("width") or (placed or {}).get("slot_width")
                    out.setdefault(role, []).append({"bundle": bundle, "field": name, "bits": bits})
    unresolved = layouts.get("unresolved")
    return out, (f"layouts not derived for {sorted(unresolved)}" if unresolved else None)


# ----------------------------------------------------------------------------- FIRRTL structure


def _zero(expr: Any) -> bool:
    from . import firrtl_struct as F

    if isinstance(expr, F.Lit):
        return expr.value == 0
    if isinstance(expr, F.Prim) and expr.op in _RECAST and len(expr.args) == 1:
        return _zero(expr.args[0])
    return False


def _rectified(module, expr: Any, value: str) -> str | None:
    """The reference ``expr`` computes ``max(x, 0)`` of, when ``x`` matches ``value``; else ``None``."""
    from . import firrtl_struct as F

    expr = _resolve(module, expr)
    if not (isinstance(expr, F.Prim) and expr.op == "mux" and len(expr.args) == 3):
        return None
    cond, when_true, when_false = (_resolve(module, arg) if index == 0 else arg for index, arg in enumerate(expr.args))
    if not (isinstance(cond, F.Prim) and cond.op in _COMPARE and len(cond.args) == 2):
        return None
    lhs, rhs = cond.args
    greater, _ = _COMPARE[cond.op]
    if isinstance(lhs, F.Ref) and _zero(rhs):
        x, keeps_x_when = lhs, greater  # cmp(x, 0): x kept when the comparison says x > 0
    elif isinstance(rhs, F.Ref) and _zero(lhs):
        x, keeps_x_when = rhs, not greater  # cmp(0, x): the order is reversed
    else:
        return None
    if not F.ref_matches(x.text(), value):
        return None
    kept, cleared = (when_true, when_false) if keeps_x_when else (when_false, when_true)
    if isinstance(kept, F.Ref) and kept.text() == x.text() and _zero(cleared):
        return x.text()
    return None


def _resolve(module, expr: Any) -> Any:
    """Follow a reference to a node's own expression, so a mux arm named by a node can be read."""
    from . import firrtl_struct as F

    seen = 0
    while isinstance(expr, F.Ref) and not expr.segments and expr.root in module.nodes and seen < 64:
        expr = module.nodes[expr.root][0]
        seen += 1
    return expr


def _activation(module, probe: Mapping[str, Any]) -> dict[str, Any]:
    """Rectifier codes selected by the activation field, and the codes compared but not recognized."""
    from . import firrtl_struct as F

    value, field = str(probe["value"]), str(probe["activation"])
    cone = F.Cone(module)
    found: dict[int, dict[str, Any]] = {}
    compared: set[int] = set()
    for name, (expr, line, loc) in module.nodes.items():
        compared |= F.eq_constants(F._prims_in(expr), field)  # noqa: SLF001 -- the module's own walker
        if not (isinstance(expr, F.Prim) and expr.op == "mux" and len(expr.args) == 3):
            continue
        cond, when_true, when_false = expr.args
        rectifier = _rectified(module, when_true, value)
        if rectifier is None or not (isinstance(when_false, F.Ref) and when_false.text() == rectifier):
            continue
        _leaves, prims, _lines = cone.walk(cond)
        for code in F.eq_constants(prims, field):
            cite = {"module": module.name, "node": name, "firrtl_line": line}
            citation = F.locator_citation(loc)
            if citation:
                cite.update({"scala_file": citation["file"], "scala_line": citation["line"]})
            found.setdefault(code, cite)
    if not found:
        raise _Unreadable(f"no node of {module.name} selects max({value}, 0) by an equality on {field}")
    return {
        "functions": {RECTIFIER: {"codes": sorted(found), "applied_to": value, "cite": found[min(found)]}},
        "unrecognized_codes": sorted(compared - set(found)),
    }


def _scale_format(module, probe: Mapping[str, Any]) -> dict[str, Any]:
    """The scale's number format, from the sign / exponent / fraction fields the readout cuts it into."""
    from merlin.common import quant_formats as qf

    from . import firrtl_struct as F

    scale = str(probe["scale"])
    cone = F.Cone(module)
    by_base: dict[str, set[tuple[int, int]]] = {}
    for _name, (expr, _line, _loc) in module.nodes.items():
        if not (isinstance(expr, F.Prim) and expr.op == "bits" and len(expr.args) == 3):
            continue
        base, hi, lo = expr.args
        if not (isinstance(base, F.Ref) and isinstance(hi, F.IntArg) and isinstance(lo, F.IntArg)):
            continue
        if any(F.ref_matches(leaf, scale) for leaf in cone.leaves(base)):
            by_base.setdefault(base.text(), set()).add((hi.value, lo.value))
    for base, ranges in sorted(by_base.items()):
        # sign | exponent | fraction: the top bit alone, then two fields that meet at one boundary.
        width = max(hi for hi, _ in ranges) + 1
        if (width - 1, width - 1) not in ranges:
            continue
        boundaries = sorted(f for f in range(1, width - 1) if (f - 1, 0) in ranges and (width - 2, f) in ranges)
        if len(boundaries) != 1:
            continue
        fraction = boundaries[0]
        exponent = width - 1 - fraction
        matches = [
            f
            for f in qf.registry().values()
            if f.kind == "float_ieee" and f.element_bits == width and f.exp_bits == exponent and f.mant_bits == fraction
        ]
        if len(matches) != 1:
            raise _Unreadable(
                f"{scale} is split as sign|e{exponent}|m{fraction} ({width} bits), which names "
                f"{len(matches)} registered IEEE format(s)"
            )
        name = matches[0].name
        token = f"f{width}"
        mlir = token if qf.has(token) and qf.get(token).name == name else name
        return {
            "format": name,
            "dtype": mlir,
            "fields": {
                "sign": [width - 1, width - 1],
                "exponent": [width - 2, fraction],
                "fraction": [fraction - 1, 0],
            },
            "split_of": base,
        }
    raise _Unreadable(f"no node of {module.name} cuts {scale} into sign, exponent and fraction fields")


def _combined(module, probe: Mapping[str, Any]) -> dict[str, Any]:
    """The instance that takes BOTH the scale and (through another instance) the accumulator value."""
    from . import firrtl_struct as F

    value, scale = str(probe["value"]), str(probe["scale"])
    cone = F.Cone(module)
    inputs: dict[str, list[set[str]]] = {}
    for connect in module.connects:
        decl = module.decls.get(connect.lhs.root)
        if decl is None or decl[0] != "inst":
            continue
        inputs.setdefault(connect.lhs.root, []).append(cone.leaves(connect.rhs))
    reads_value = {inst for inst, pins in inputs.items() if any(F.ref_matches(x, value) for p in pins for x in p)}
    for inst, pins in sorted(inputs.items()):
        takes_scale = any(F.ref_matches(x, scale) for p in pins for x in p)
        fed = sorted({x.split(".", 1)[0] for p in pins for x in p} & reads_value)
        if takes_scale and fed:
            return {"instance": inst, "of": module.decls[inst][1], "value_through": fed}
    raise _Unreadable(f"no instance of {module.name} combines {scale} with {value}")


def _feeds(module, node: str, inst: str) -> bool:
    """Whether node ``node`` is in the fan-in of any input pin of instance ``inst``."""
    from . import firrtl_struct as F

    line = (module.nodes.get(node) or (None, None, None))[1]
    cone = F.Cone(module)
    return line is not None and any(
        line in cone.walk(connect.rhs)[2] for connect in module.connects if connect.lhs.root == inst
    )


def _declared_type(module, root: str):
    from . import firrtl_struct as F

    decl = module.decls.get(root)
    if decl is None or decl[0] not in ("input", "output", "reg", "regreset", "wire"):
        return None
    try:
        return F.parse_type(decl[1])
    except F.FirrtlParseError:
        return None


def _path(ref_text: str) -> tuple[str, list[str]]:
    root, _, rest = ref_text.partition(".")
    return root, [segment for segment in rest.split(".") if segment]


def _columns_per_scale(module, probe: Mapping[str, Any]) -> dict[str, Any]:
    """How many accumulator elements ONE scale operand covers in the readout, read from its port types.

    The readout carries the scale beside a vector of accumulator elements; every element of that
    vector is scaled by the one value, so the vector's element count is the column group a single
    scale can serve.
    """
    from . import firrtl_struct as F

    root, path = _path(str(probe["value"]))
    value_type = F.field_type(_declared_type(module, root), path)
    scale_root, scale_path = _path(str(probe["scale"]))
    scale_type = F.field_type(_declared_type(module, scale_root), scale_path)
    count, leaf = 1, value_type
    while isinstance(leaf, tuple) and leaf[0] == "vec" and isinstance(leaf[2], int):
        count, leaf = count * leaf[2], leaf[1]
    if not (isinstance(leaf, tuple) and leaf[0] in ("sint", "uint")) or scale_type is None:
        raise _Unreadable(f"the types of {probe['value']} and {probe['scale']} are not readable in {module.name}")
    if scale_type[0] not in ("uint", "sint", "bundle") or (scale_type[0] == "bundle" and len(scale_type[1]) != 1):
        raise _Unreadable(f"{probe['scale']} is not one value beside the vector it scales")
    return {"columns_per_scale": count, "element_bits": leaf[1]}


def _passthrough_lanes(module, bits: int) -> dict[str, Any] | None:
    """A vector of ``bits``-wide integer lanes the module copies out lane for lane, with no arithmetic."""
    from . import firrtl_struct as F

    cone = F.Cone(module)
    impure: set[str] = set()
    for expr, _line, _loc in module.nodes.values():
        if isinstance(expr, F.Prim):
            impure |= {ref.text() for ref in F._refs_in(expr)}  # noqa: SLF001 -- the module's own walker
    copies: dict[str, dict[int, int]] = {}
    for connect in module.connects:
        if isinstance(connect.rhs, F.Prim):
            impure |= {ref.text() for ref in F._refs_in(connect.rhs)}  # noqa: SLF001
            continue
        leaves, prims, _lines = cone.walk(connect.rhs)
        if prims or len(leaves) != 1:
            continue
        (leaf,) = leaves
        base, _, index = leaf.rpartition("[")
        if base and index.rstrip("]").isdigit():
            copies.setdefault(base, {})[int(index.rstrip("]"))] = connect.line
    for base, lanes in sorted(copies.items()):
        root, path = _path(base)
        typ = F.field_type(_declared_type(module, root), path)
        if not (isinstance(typ, tuple) and typ[0] == "vec" and isinstance(typ[1], tuple)):
            continue
        element, count = typ[1], typ[2]
        if element[0] not in ("sint", "uint") or element[1] != bits or set(lanes) != set(range(count or 0)):
            continue
        if any(f"{base}[{i}]" in impure for i in lanes):
            continue
        return {"module": module.name, "vector": base, "lanes": count, "bits": bits, "firrtl_line": min(lanes.values())}
    return None


def _output_fields(typ: Any, prefix: str, flipped: bool = False):
    """``(path text, type)`` of every field of a port type that leaves the module."""
    if isinstance(typ, tuple) and typ[0] == "bundle":
        for name, flip, sub in typ[1]:
            yield from _output_fields(sub, f"{prefix}.{name}", flipped != flip)
    elif not flipped:
        yield prefix, typ


def _full_width_exit(module, probe: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the readout ALSO hands the accumulator out unmodified, at its own width.

    Two structural readings: the accumulator value is copied whole, with no operation, into a field
    of its own type; and an output field of that same type is driven with no arithmetic anywhere in
    its fan-in. Together they say the raw accumulator leaves the readout as it entered.
    """
    from . import firrtl_struct as F

    value = str(probe["value"])
    root, path = _path(value)
    value_type = F.field_type(_declared_type(module, root), path)
    if value_type is None:
        raise _Unreadable(f"the type of {value} is not readable in {module.name}")
    copied = [
        c.lhs.text()
        for c in module.connects
        if isinstance(c.rhs, F.Ref)
        and c.rhs.text() == value
        and F.field_type(_declared_type(module, c.lhs.root), _path(c.lhs.text())[1]) == value_type
    ]
    if not copied:
        raise _Unreadable(f"{module.name} copies {value} into no field of its own type")
    cone = F.Cone(module)
    for name, (keyword, _type_text, _line) in sorted(module.decls.items()):
        if keyword != "output":
            continue
        for field, typ in _output_fields(_declared_type(module, name), name):
            if typ != value_type or field == value:
                continue
            drivers = [c for c in module.connects if c.lhs.text() == field]
            if drivers and all(not cone.walk(c.rhs)[1] for c in drivers):
                return {"copied_into": copied[0], "leaves_as": field, "lanes": _lane_count(value_type)}
    raise _Unreadable(f"no output of {module.name} carries a {value_type} field without arithmetic")


def _lane_count(typ: Any) -> int:
    count = 1
    while isinstance(typ, tuple) and typ[0] == "vec" and isinstance(typ[2], int):
        count, typ = count * typ[2], typ[1]
    return count


def _store_writes_data(module, select: str, summed: str) -> str:
    """The arm the accumulator store writes when its accumulate bit is CLEAR: the write's own data.

    ``accumulate_on_load`` establishes ``mux(select, sum, other)``; a preload needs ``other`` to be a
    field of the same write record as the select (its arriving data), not a constant or the sum.
    """
    from . import firrtl_struct as F

    for expr, _line, _loc in module.nodes.values():
        for prim in F._prims_in(expr):  # noqa: SLF001 -- the module's own walker
            if prim.op != "mux" or len(prim.args) != 3 or not isinstance(prim.args[0], F.Ref):
                continue
            chooser, when_set, when_clear = prim.args
            if not (F.ref_matches(chooser.text(), select) and isinstance(when_set, F.Ref)):
                continue
            if not F.ref_matches(when_set.text(), summed) or not isinstance(when_clear, F.Ref):
                continue
            same_write = when_clear.root == chooser.root and when_clear.segments[:-1] == chooser.segments[:-1]
            # The arriving data has the sum's own type: a flag of the same record does not.
            types = [
                F.field_type(_declared_type(module, ref.root), [seg for seg in ref.segments if seg is not None])
                for ref in (when_set, when_clear)
            ]
            if same_write and when_clear != chooser and types[0] is not None and types[0] == types[1]:
                return when_clear.text()
    raise _Unreadable(f"no store in {module.name} writes its own data when {select} is clear")


def _bias_preload(body: Mapping[str, Any], accumulate: Mapping[str, Any], accumulator: str | None) -> dict[str, Any]:
    """Whether a value can be LOADED into the accumulator unchanged, so a contraction adds onto it.

    Read from three structural facts. The accumulator store writes the arriving data when a write's
    accumulate bit is clear (``accumulate_on_load``'s store reading: the arm beside the adder's sum);
    a load selects that bit itself (the same fact's load path); and some instance on that load path
    copies accumulator-width lanes out with no arithmetic, so a full-width value arrives as it was.
    """
    from merlin.common import quant_formats as qf

    value = accumulate.get("value") or {}
    load = value.get("load_path") or {}
    owner = (load.get("cite") or {}).get("module")
    roots = sorted({str(item).split(".", 1)[0] for item in load.get("carried_by") or ()})
    if not owner or not roots:
        raise _Unreadable("the accumulate fact names no load-path module and instances")
    integer = bool(accumulator) and qf.has(str(accumulator)) and qf.get(str(accumulator)).exp_bits is None
    bits = qf.machine_bits(str(accumulator)) if integer else None
    if not bits:
        raise _Unreadable(f"the accumulator format {accumulator!r} has no integer width")
    store = value.get("store") or {}
    store_module = (store.get("cite") or {}).get("module")
    if not store_module or not store.get("select") or not store.get("sum"):
        raise _Unreadable("the accumulate fact names no store select and sum")
    modules, _provenance, why = _load(body, sorted({owner, store_module}))
    if why:
        raise _Unreadable(why)
    written = _store_writes_data(modules[store_module], str(store["select"]), str(store["sum"]))
    instances = {root: modules[owner].decls[root][1] for root in roots if root in modules[owner].decls}
    loaded, _provenance, why = _load(body, sorted(set(instances.values())))
    if why:
        raise _Unreadable(why)
    for root, kind in sorted(instances.items()):
        if kind in loaded:
            lanes = _passthrough_lanes(loaded[kind], int(bits))
            if lanes is not None:
                return {"through": root, "store_writes": written, **lanes}
    raise _Unreadable(
        f"no instance on the load path ({sorted(instances)}) copies {bits}-bit accumulator lanes unchanged"
    )


def _load(body: Mapping[str, Any], names: list[str]) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Named modules of the FIRRTL the facts pin, the provenance of the bytes, and why they are absent."""
    from . import firrtl_struct as F

    source = body.get("source") or {}
    path = Path(str(source.get("fir_path") or ""))
    if not source.get("fir_path") or not path.is_file():
        return {}, {}, "the facts name no elaborated FIRRTL that is present on this machine"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {"fir_path": str(path), "fir_sha256": digest}
    if source.get("fir_sha256") and source["fir_sha256"] != digest:
        return {}, provenance, f"the FIRRTL at {path} is not the one the facts pin"
    modules = F.load_modules(path, names)
    missing = sorted(set(names) - set(modules))
    return modules, provenance, (f"the elaboration defines no module {missing}" if missing else None)


class _Unreadable(Exception):
    """A rung could not read its fact; the message is the reason."""


def _firrtl_modules(body: Mapping[str, Any], module: str) -> tuple[Any | None, dict[str, Any], str | None]:
    """The probed module, the provenance of the bytes read, and why it could not be read."""
    from . import firrtl_struct as F

    source = body.get("source") or {}
    path = Path(str(source.get("fir_path") or ""))
    if not source.get("fir_path") or not path.is_file():
        return None, {}, "the facts name no elaborated FIRRTL that is present on this machine"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pinned = source.get("fir_sha256")
    provenance = {"fir_path": str(path), "fir_sha256": digest}
    if pinned and pinned != digest:
        return None, provenance, f"the FIRRTL at {path} is not the one the facts pin ({str(pinned)[:12]})"
    modules = F.load_modules(path, [module])
    if module not in modules:
        return None, provenance, f"the elaboration defines no module {module!r}"
    return modules[module], provenance, None


# ------------------------------------------------------------------------------------- derivation


def _probe(contract: Mapping[str, Any] | None) -> tuple[Mapping[str, Any] | None, str | None]:
    raw = ((contract or {}).get("rtl_extraction") or {}).get(FIELD)
    if raw is None:
        return None, f"the contract declares no rtl_extraction.{FIELD} probe naming the readout module"
    if not isinstance(raw, Mapping):
        raise ValueError(f"rtl_extraction.{FIELD} must be a mapping")
    missing = [key for key in ("module", "value", "scale", "activation") if not isinstance(raw.get(key), str)]
    if missing:
        raise ValueError(f"rtl_extraction.{FIELD} must name module, value, scale and activation; missing {missing}")
    return raw, None


def _scalar_abi(contract: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    readout = (((contract or {}).get("runner") or {}).get("chipyard_rocc") or {}).get("readout")
    abi = (readout or {}).get("scalar_abi") if isinstance(readout, Mapping) else None
    return abi if isinstance(abi, Mapping) else None


def derive(
    facts: Mapping[str, Any] | None,
    *,
    contract: Mapping[str, Any] | None = None,
    semantic_facts: Mapping[str, Any] | None = None,
    module: Any | None = None,
) -> dict[str, Any]:
    """The output-epilogue record for a target, from its facts, its contract's probe and its sources.

    ``module`` substitutes an already-parsed readout module for the FIRRTL the facts name (a test, or a
    caller that has the bytes in hand). Pure apart from reading the FIRRTL the facts pin.
    """
    body = _body(facts)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        **{name: {"status": UNKNOWN} for name in (*CAPABILITIES, *OPTIONAL_CAPABILITIES)},
    }
    unknown: dict[str, str] = {}
    accumulator, element = _datapath(body, "accumulator"), _datapath(body, "input")
    record["accumulator_dtype"] = accumulator
    carriers, layouts_why = _carriers(body)
    for role, capability in (("scale", "scale"), ("activation", "activation"), ("pool", "pool")):
        if carriers.get(role):
            record[capability]["carriers"] = carriers[role]
            _note(
                record,
                capability,
                "register_layout",
                "command-register field(s) carry the "
                + repr(role)
                + " role: "
                + ", ".join(f"{c['bundle']}.{c['field']} ({c['bits']} bits)" for c in carriers[role]),
            )
    probe, probe_why = _probe(contract)
    readout, provenance, firrtl_why = (None, {}, probe_why)
    if probe is not None:
        if module is not None:
            readout, firrtl_why = module, None
        else:
            readout, provenance, firrtl_why = _firrtl_modules(body, str(probe["module"]))
    if provenance:
        record["firrtl"] = provenance

    # --- scale -----------------------------------------------------------------------------------
    scale = record["scale"]
    scale.update({"accumulator_dtype": accumulator, "output_dtypes": [element] if element else []})
    if readout is not None:
        try:
            scale["format"] = _scale_format(readout, probe)
            scale["dtype"] = scale["format"]["dtype"]
            _note(
                record,
                "scale",
                "firrtl_structural",
                f"{probe['scale']} is decoded as {scale['format']['format']} "
                f"(sign|exponent|fraction at {scale['format']['fields']})",
            )
            scale["combined_by"] = _combined(readout, probe)
            scale.update(_columns_per_scale(readout, probe))
            _note(
                record,
                "scale",
                "firrtl_structural",
                f"instance {scale['combined_by']['instance']} "
                f"({scale['combined_by']['of']}) takes the decoded scale and the accumulator value",
            )
        except _Unreadable as exc:
            unknown["scale.format"] = str(exc)
    else:
        unknown["scale.format"] = firrtl_why or "no readout module was read"
    full_width = record["full_width_readout"]
    if readout is not None:
        try:
            full_width.update({"status": DERIVED, "dtype": accumulator, **_full_width_exit(readout, probe)})
            _note(
                record,
                "full_width_readout",
                "firrtl_structural",
                f"{probe['value']} is copied unmodified into {full_width['copied_into']} and "
                f"{full_width['leaves_as']} leaves the readout with no arithmetic in its fan-in",
            )
        except _Unreadable as exc:
            unknown["full_width_readout"] = str(exc)
    else:
        unknown["full_width_readout"] = firrtl_why or "no readout module was read"
    abi = _scalar_abi(contract)
    if abi is not None and abi.get("schema") == "scalar_narrow_readout_contract_v1":
        scale["rounding"] = "half_even"
        if abi.get("clamp_min") is not None and abi.get("clamp_max") is not None:
            scale["saturation"] = [int(abi["clamp_min"]), int(abi["clamp_max"])]
        if abi.get("scale_dtype") and scale.get("dtype") and abi["scale_dtype"] != scale["dtype"]:
            unknown["scale.dtype"] = f"the RTL decodes {scale['dtype']} and the scalar ABI states {abi['scale_dtype']}"
            scale.pop("dtype")
        _note(record, "scale", "scalar_abi", "the scalar readout contract states round-half-to-even and its clamp")
    else:
        unknown["scale.rounding"] = "no scalar readout contract states the rounding"
    if scale.get("carriers") and scale.get("dtype") and accumulator and "combined_by" in scale:
        scale["status"] = DERIVED
    elif not scale.get("carriers"):
        unknown["scale"] = layouts_why or "no command-register field carries the scale role"

    # --- activation ------------------------------------------------------------------------------
    activation = record["activation"]
    if readout is not None:
        try:
            activation.update(_activation(readout, probe))
            activation["status"] = DERIVED
            codes = activation["functions"][RECTIFIER]["codes"]
            through = (scale.get("combined_by") or {}).get("value_through") or ()
            node = activation["functions"][RECTIFIER]["cite"]["node"]
            # Where the rectifier sits relative to the scale is read, not assumed: it precedes the scale
            # when its node is in the fan-in of the instance that carries the value into the multiply.
            activation["before_scale"] = bool(through) and any(_feeds(readout, node, inst) for inst in through)
            _note(
                record,
                "activation",
                "firrtl_structural",
                f"max({probe['value']}, 0) is selected when "
                f"{probe['activation']} equals {codes}"
                + ("; it is applied to the accumulator before the scale" if activation["before_scale"] else ""),
            )
        except _Unreadable as exc:
            unknown["activation"] = str(exc)
    else:
        unknown["activation"] = (
            "an activation field is encodable, but which function each code selects is not derived: "
            + (firrtl_why or "no readout module was read")
            if activation.get("carriers")
            else firrtl_why or "no readout module was read"
        )

    # --- accumulate ------------------------------------------------------------------------------
    accumulate = record["accumulate"]
    fact = ((semantic_facts or {}).get("facts") or {}).get("accumulate_on_load")
    if isinstance(fact, Mapping) and fact.get("status") == DERIVED:
        accumulate.update({"status": DERIVED, "fact": "accumulate_on_load", "value": copy.deepcopy(fact.get("value"))})
        _note(
            record,
            "accumulate",
            "firrtl_structural",
            "accumulate_on_load is derived: a load adds into the accumulator, selected per command",
        )
    else:
        unknown["accumulate"] = (
            "no semantic facts document reports accumulate_on_load as derived"
            if fact is None
            else f"accumulate_on_load is {fact.get('status')!r}"
        )

    # --- bias preload ---------------------------------------------------------------------------
    bias = record["bias_preload"]
    if accumulate["status"] == DERIVED:
        try:
            lanes = _bias_preload(body, accumulate, accumulator)
            span = 1 << (int(lanes["bits"]) - 1)
            bias.update(
                {
                    "status": DERIVED,
                    "dtype": accumulator,
                    "bits": lanes["bits"],
                    "rounding": "none",
                    "range": [-span, span - 1],
                    "lanes": lanes,
                }
            )
            _note(
                record,
                "bias_preload",
                "firrtl_structural",
                f"a load writes the accumulator when its accumulate bit is clear, and {lanes['through']} "
                f"({lanes['module']}) copies {lanes['lanes']} {lanes['bits']}-bit lanes of {lanes['vector']} "
                "with no arithmetic: a full-width value is preloaded exactly",
            )
        except _Unreadable as exc:
            unknown["bias_preload"] = str(exc)
    else:
        unknown["bias_preload"] = "the accumulator is not derived to take a load, so nothing can be preloaded"

    # --- pool ------------------------------------------------------------------------------------
    pool = record["pool"]
    # A build gate and window geometry do not establish the value contributed by
    # an out-of-bounds lane. That requires the complete store-to-reduction data
    # cone, including sentinel handling, which this reader does not observe.
    pool["padding"] = UNKNOWN
    unknown["pool.padding"] = (
        "the padded store-to-reduction value is not derived; pooling gates and "
        "command fields do not establish padding semantics"
    )
    features = _interface(body, "elaborated_rtl_features")
    if features is not None and features.get("status") == DERIVED:
        kinds = sorted(
            {
                _REDUCTION_TOKENS[token]
                for name, on in (features.get("features") or {}).items()
                if on is True and "pool" in str(name).lower().split("_")
                for token in str(name).lower().split("_")
                if token in _REDUCTION_TOKENS
            }
        )
        if kinds:
            pool.update({"status": DERIVED, "reductions": kinds})
            _note(
                record, "pool", "firrtl_build_gate", f"the elaborated pooling gate is on ({features.get('evidence')})"
            )
        else:
            unknown["pool"] = "no enabled elaborated feature names a pooling reduction"
    else:
        unknown["pool"] = "the facts record no derived elaborated feature gates"
    if pool["status"] == DERIVED and not pool.get("carriers"):
        pool["status"] = UNKNOWN
        unknown["pool"] = "pooling is built, but no command-register field carries the pool role"

    record["stages"] = {
        stage: record[capability]["status"] == DERIVED for stage, capability in STAGE_CAPABILITY.items()
    }
    record["unknown"] = dict(sorted(unknown.items()))
    from merlin.targetgen import output_path as OP

    record["output_path"] = OP.from_record(record)
    return validate(record)


def validate(record: Any) -> dict[str, Any]:
    """Raise unless ``record`` is a well-formed ``output_epilogue_v1`` declaration; return it."""
    if not isinstance(record, Mapping) or record.get("schema") != SCHEMA:
        raise ValueError(f"{FIELD} must be a {SCHEMA} mapping")
    for name in (*CAPABILITIES, *(name for name in OPTIONAL_CAPABILITIES if name in record)):
        row = record.get(name)
        if not isinstance(row, Mapping) or row.get("status") not in (DERIVED, UNKNOWN):
            raise ValueError(f"{FIELD}.{name} needs a status of {DERIVED!r} or {UNKNOWN!r}")
        for item in row.get("evidence") or ():
            if not isinstance(item, Mapping) or not item.get("rung") or not item.get("observed"):
                raise ValueError(f"{FIELD}.{name} evidence rows need a rung and an observation")
    if record["scale"]["status"] == DERIVED and not (record["scale"].get("dtype") and record["scale"].get("carriers")):
        raise ValueError(f"a derived {FIELD}.scale needs its format and its carrying field")
    if record["activation"]["status"] == DERIVED and not (record["activation"].get("functions") or {}):
        raise ValueError(f"a derived {FIELD}.activation needs at least one recognized function")
    stages = record.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(STAGE_CAPABILITY):
        raise ValueError(f"{FIELD}.stages must state every one of {sorted(STAGE_CAPABILITY)}")
    for stage, capability in STAGE_CAPABILITY.items():
        if stages[stage] is not (record[capability]["status"] == DERIVED):
            raise ValueError(f"{FIELD}.stages.{stage} disagrees with {capability}.status")
    if "output_path" in record:
        from merlin.targetgen import output_path as OP

        OP.validate(record["output_path"])
    return dict(record)


def underived(reason: str) -> dict[str, Any]:
    """A well-formed record in which nothing is derived, carrying why: the fail-closed declaration."""
    record: dict[str, Any] = {
        "schema": SCHEMA,
        **{name: {"status": UNKNOWN} for name in (*CAPABILITIES, *OPTIONAL_CAPABILITIES)},
    }
    record["stages"] = dict.fromkeys(STAGE_CAPABILITY, False)
    record["pool"]["padding"] = UNKNOWN
    record["unknown"] = {"derivation": str(reason), "pool.padding": str(reason)}
    return validate(record)


def declare(
    contract: Mapping[str, Any],
    facts: Mapping[str, Any] | None,
    *,
    probe: Mapping[str, Any] | None = None,
    semantic_facts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A copy of ``contract`` carrying its derived ``output_epilogue`` (and ``probe``, when given).

    For a selected contract that is not produced by contract derivation
    (:func:`merlin.targetgen.capability_manifests.derive_manifest`): the record is derived from the
    facts the contract is selected with, so the declaration and the facts describe one design.
    """
    out = copy.deepcopy(dict(contract))
    if probe is not None:
        out.setdefault("rtl_extraction", {})[FIELD] = dict(probe)
    out[FIELD] = derive(facts, contract=out, semantic_facts=semantic_facts)
    return out


def main(argv: list[str] | None = None) -> int:
    """``python -m merlin.targetgen.rtl.output_epilogue --contract C --facts F [--probe JSON] --out O``."""
    import argparse
    import json

    import yaml

    parser = argparse.ArgumentParser(description="Declare a contract's derived output epilogue.")
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--facts", required=True, type=Path)
    parser.add_argument("--probe", help="JSON {module, value, scale, activation} naming the readout")
    parser.add_argument("--semantic-facts", type=Path, help="a semantic_facts_v1 document")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"{args.out} exists; a declared contract is written once, never overwritten")
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    facts = json.loads(args.facts.read_text(encoding="utf-8"))
    semantic = json.loads(args.semantic_facts.read_text(encoding="utf-8")) if args.semantic_facts else None
    declared = declare(contract, facts, probe=json.loads(args.probe) if args.probe else None, semantic_facts=semantic)
    args.out.write_text(json.dumps(declared, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"out": str(args.out), "stages": declared[FIELD]["stages"], "unknown": declared[FIELD]["unknown"]}
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
