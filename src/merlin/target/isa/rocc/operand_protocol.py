"""Detached, optional operand protocol projections over selected source RTL facts.

Raw command identities remain owned by BoundSemantics. These projections do not
execute code, discover a provider, select a target, or qualify numerical results.
A reserved packing slot is never substituted for an unresolved actual RTL width.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

from merlin.target.isa.rocc import semantics as S
from merlin.target.isa.rocc.semantics import BoundSemantics
from merlin.target.rtl.extraction.source_custody import relocate_operand_sources
from merlin.target.rtl.firrtl.operand_slices import apply_selected_layouts, verify_selected_layout_source

# ---- normalized_operands -----------------------------------------------------------------------


def _keys(value, required, optional=()):
    if type(value) is not dict or set(value) - set(required) - set(optional) or not set(required) <= set(value):
        raise ValueError("operand protocol has missing or unsupported fields")


def _strict_layouts(facts, word_bits):
    layouts = copy.deepcopy(S.bundle_layouts(facts))
    for bundle in layouts.values():
        if type(bundle) is not dict or type(bundle.get("fields")) is not dict:
            raise ValueError("operand protocol requires source bundle field mappings")
        occupied = 0
        projections = {}
        for name, field in bundle["fields"].items():
            if type(name) is not str or not name.isidentifier():
                raise ValueError("operand field has no unique original source name")
            if type(field) is not dict:
                raise ValueError("operand protocol field layout is malformed")
            width, offset = field.get("width"), field.get("offset")
            reserved = field.pop("slot_width", None)
            if width is None:
                continue
            if (
                type(width) is not int
                or width <= 0
                or type(offset) is not int
                or offset < 0
                or offset + width > word_bits
                or (type(reserved) is int and width > reserved)
            ):
                raise ValueError("actual operand field width does not fit its source register layout")
            mask = ((1 << width) - 1) << offset
            if occupied & mask:
                raise ValueError("actual operand fields overlap")
            occupied |= mask
            if "source_subfields" in field:
                children = field["source_subfields"]
                if type(children) is not list or not children or len(children) > word_bits:
                    raise ValueError("operand subfield roster is not an actual bounded source projection")
                end = offset
                for child in children:
                    _keys(child, {"field", "offset", "width"})
                    path, start, child_width = child["field"], child["offset"], child["width"]
                    if (
                        type(path) is not str
                        or not path
                        or not all(part.isidentifier() for part in path.split("."))
                        or type(start) is not int
                        or start != end
                        or type(child_width) is not int
                        or child_width <= 0
                        or start + child_width > offset + width
                    ):
                        raise ValueError("operand subfields have gaps, overlaps or unsupported source paths")
                    key = name + "." + path
                    if key in projections or key in bundle["fields"]:
                        raise ValueError("operand subfield has ambiguous original ownership")
                    projections[key] = {"offset": start, "width": child_width}
                    end += child_width
                if end != offset + width:
                    raise ValueError("operand subfields do not preserve the complete original packed field")
        # These are views of the original field, not new disjoint register fields.
        bundle["fields"].update(projections)
    return layouts


class BoundOperandObserver:
    """Immutable selected normalized decode program; no ambient manifest/facts fallback."""

    def __init__(self, *, target, protocol, facts, raw_isa, word_bits, load_states=None):
        from merlin.target.rtl.firrtl.load_states import LoadStateMapping

        if load_states is not None and type(load_states) is not LoadStateMapping:
            raise ValueError("operand observer requires typed original load-state evidence")
        self._load_states = load_states
        _keys(protocol, {"version", "encoding", "roles"}, {"address_field"})
        if type(protocol["version"]) is not int or protocol["version"] != 1:
            raise ValueError("unsupported operand protocol version")
        if type(word_bits) is not int or word_bits not in {32, 64}:
            raise ValueError("operand observer requires the selected register width")
        self._target, self._word_bits = target, word_bits
        protocol = copy.deepcopy(protocol)
        encoding, roles = protocol["encoding"], protocol["roles"]
        _keys(encoding, {"semantic_class"}, {"config_subtype"})
        _keys(roles, {"classes"}, {"config", "local_address_flags"})
        classes = S._numeric_code_map(encoding["semantic_class"], field="protocol.semantic_class", width=7)
        if set(classes) != set(raw_isa["FUNCT_CLASS"]):
            raise ValueError("normalized protocol must bind the exact selected raw instruction roster")
        if any(not label.isidentifier() or label == "UNKNOWN" for label in classes.values()):
            raise ValueError("normalized protocol requires explicit unique class labels")
        layouts = _strict_layouts(facts, word_bits)
        config, selector, selector_slot, subtypes = roles.get("config"), {}, None, {}
        if config is not None:
            _keys(config, {"class", "selector"})
            selector = config["selector"]
            _keys(selector, {"operand", "bundle", "field"})
            if selector["operand"] not in {"rs1", "rs2"} or config["class"] not in classes.values():
                raise ValueError("configuration selector does not bind a selected class and register")
            selector_slot = S.field_slot(layouts.get(selector["bundle"]), selector["field"])
            # An unresolved selector stays UNKNOWN at decode, rather than reading reserved bits.
            subtypes = S._numeric_code_map(
                encoding.get("config_subtype"),
                field="protocol.config_subtype",
                width=selector_slot[1] if selector_slot else word_bits,
            )
            if set(subtypes.values()) & set(classes.values()):
                raise ValueError("configuration subtypes collide with base instruction classes")
        elif "config_subtype" in encoding:
            raise ValueError("configuration subtypes require a source-bound selector")
        known = set(classes.values()) | set(subtypes.values())
        if type(roles["classes"]) is not dict or set(roles["classes"]) - known:
            raise ValueError("operand fields name an unselected normalized class")
        for spec in roles["classes"].values():
            _keys(spec, {"fields"}, {"unresolved"})
            if type(spec["fields"]) is not list:
                raise ValueError("operand protocol fields must be a list")
            spec["unresolved"] = "null"
            names = []
            for field in spec["fields"]:
                _keys(
                    field,
                    set(),
                    {
                        "name",
                        "names",
                        "operand",
                        "bundle",
                        "field",
                        "as",
                        "const",
                        "from",
                        "equals",
                        "flag",
                        "set",
                        "clear",
                        "all",
                    },
                )
                if "const" in field and type(field["const"]) is not str:
                    raise ValueError("protocol constants are symbolic labels, not invented numerical fields")
                if field.get("as") in {"const", "equals"}:
                    raise ValueError("constant/comparison projections require explicit const/from declarations")
                if "all" in field:
                    _keys(field, {"name", "all"}, {"set", "clear"})
                    if (
                        type(field["all"]) is not list
                        or not 1 <= len(field["all"]) <= 32
                        or ("set" in field) != ("clear" in field)
                    ):
                        raise ValueError("operand conjunction requires a bounded original predicate roster")
                    for predicate in field["all"]:
                        _keys(predicate, {"from", "equals"})
                        if predicate["from"] not in names or type(predicate["equals"]) not in {bool, int, str}:
                            raise ValueError("operand conjunction must reference previously decoded scalar fields")
                    if "set" in field and any(
                        type(field[key]) is not str or not field[key] for key in ("set", "clear")
                    ):
                        raise ValueError("operand conjunction choices must be explicit symbolic labels")
                if "names" in field:
                    _keys(field, {"names", "operand", "bundle"})
                    if field["operand"] not in {"rs1", "rs2"}:
                        raise ValueError("operand field projection has no selected register")
                    if type(field["names"]) is not list or not all(type(n) is str for n in field["names"]):
                        raise ValueError("operand field names must be explicit strings")
                    names.extend(field["names"])
                else:
                    names.append(field.get("name"))
            if any(type(n) is not str or not n.isidentifier() for n in names) or len(names) != len(set(names)):
                raise ValueError("operand protocol field names are missing or duplicated")
        address_width = None
        if "address_field" in protocol:
            address = protocol["address_field"]
            _keys(address, {"bundle", "field"})
            slot = S.field_slot(layouts.get(address["bundle"]), address["field"])
            address_width = slot[1] if slot else None
        flag_specs = roles.get("local_address_flags", {})
        if type(flag_specs) is not dict:
            raise ValueError("local address flag roles must be a mapping")
        for spec in flag_specs.values():
            _keys(spec, {"msb_offset"})
            if type(spec["msb_offset"]) is not int or not 0 <= spec["msb_offset"] < word_bits:
                raise ValueError("local address flag offset is not bounded")
        if flag_specs and "address_field" not in protocol:
            raise ValueError("local address flags require a source-bound address field")
        flags = S.local_address_flags(roles, address_width) if address_width else {name: 0 for name in flag_specs}
        programs = S._compile_classes(roles, layouts, flags)
        if address_width is None:
            for program in programs.values():
                for step in program["fields"]:
                    if step["as"] in {"flag", "flag_choice"}:
                        step.update(offset=None, width=None)
        self._isa = {
            "CUSTOM_OPCODE": raw_isa["CUSTOM_OPCODE"],
            "FUNCT3": raw_isa["FUNCT3"],
            "FUNCT_CLASS": classes,
            "CONFIG_CLASS": config["class"] if config else None,
            "CONFIG_SELECTOR": {"operand": selector["operand"], "offset": selector_slot[0], "width": selector_slot[1]}
            if selector_slot
            else None,
            "CONFIG_SUBTYPE": subtypes,
            "OPERAND_FIELDS": programs,
        }

    @property
    def word_bits(self):
        return self._word_bits

    @property
    def load_states(self):
        return self._load_states

    def isa_constants(self, target):
        if target != self._target:
            raise ValueError("operand observer belongs to a different selected target")
        return copy.deepcopy(self._isa)

    def decode_instruction(self, funct, rs1, rs2, isa):
        if isa != self._isa:
            raise ValueError("operand observer requires its own selected decode program")
        observations = []
        for operand in (rs1, rs2):
            if type(operand) is not dict:
                raise ValueError("operand observation must retain its original identity")
            observation = dict(operand)
            S._known_bits(observation, word_bits=self.word_bits)
            raw = observation.get("raw")
            if raw is not None:
                if type(raw) is not int or not -(1 << (self.word_bits - 1)) <= raw < (1 << self.word_bits):
                    raise ValueError("observed register value exceeds selected word width")
                if raw < 0:
                    observation["raw"] = raw + (1 << self.word_bits)
            observations.append(observation)
        return S.decode_instruction(funct, *observations, self._isa)


# ---- selected_operand_protocol -----------------------------------------------------------------
# Explicit source-pinned operand observations over unchanged original facts.
#
# The caller must select the sidecar byte pin through its original-owned profile.
# This loader rechecks that data selection; it neither grants an original owner nor
# discovers a provider, substitutes original facts, or qualifies execution.

SCHEMA = "merlin.selected_operand_protocol.v1"


# Cache detached structural observations, never owners, observers or acceptance.
# Every use still reopens all selected source pins before and after publication.
_DERIVATION_LIMIT = 4
_DERIVATION_BYTES = 16 * 1024 * 1024
_derivations = OrderedDict()
_derivation_lock = threading.RLock()


def _reset_derivations():
    global _derivations, _derivation_lock
    _derivations = OrderedDict()
    _derivation_lock = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_derivations)


def _structural_observations(original, selected, observed_source_facts, raw_isa, key):
    # The large source is checked even on a cache hit. A changed path, symlink,
    # same-size write or restored mtime must never reuse a structural result.
    verify_selected_layout_source(selected["layouts"])
    with _derivation_lock:
        cached = _derivations.get(key)
        if cached is not None:
            _derivations.move_to_end(key)
            return copy.deepcopy(cached[0]), cached[1]
        observed = apply_selected_layouts(observed_source_facts, selected["layouts"])
        mapping = None
        if "load_state_selection" in selected:
            from merlin.target.rtl.firrtl.load_states import derive_load_state_mapping

            temporary = BoundOperandObserver(
                target=original._target,
                protocol=selected["protocol"],
                facts=observed,
                raw_isa=raw_isa,
                word_bits=original._width,
            )
            mapping = derive_load_state_mapping(
                selected["layouts"], selected["load_state_selection"], temporary.isa_constants(original._target)
            )
        verify_selected_layout_source(selected["layouts"])
        # Bound retained bytes, including the copy's serialized representation;
        # oversized observations remain usable but are not retained.
        size = len(json.dumps(observed).encode()) + len(repr(mapping).encode())
        if size <= _DERIVATION_BYTES:
            _derivations[key] = (copy.deepcopy(observed), mapping, size)
            while (
                len(_derivations) > _DERIVATION_LIMIT
                or sum(row[2] for row in _derivations.values()) > _DERIVATION_BYTES
            ):
                _derivations.popitem(last=False)
        return observed, mapping


def _read_pin(pin, limit):
    if type(pin) is not dict or set(pin) != {"path", "sha256"} or type(pin["path"]) is not str:
        raise ValueError("operand selection file pin is malformed")
    path = Path(pin["path"])
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError("operand selection requires an ordinary canonical selected file")
    if path.stat().st_size > limit:
        raise ValueError("operand selection exceeds its bounded byte document")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != _sha(pin["sha256"]):
        raise ValueError("operand selection file differs from its selected byte pin")
    return raw


# Optional sidecar keys a decoder structure copies verbatim; see decoder_structure.
_DECODER_KEYS = ("operand_field_kinds", "operand_field_kind_sources", "operand_layout_roster")


def decoder_structure(selected):
    """The two decoder-structure keys a sidecar exposes, checked for shape.

    ``operand_field_kinds`` maps a decoded ``CLASS.field`` label to one declared
    kind (absent kinds stay unknown); ``operand_layout_roster`` reports whether
    every derived RTL layout field is projected. A sidecar without them yields
    an empty kind map and no roster, never a guessed one.
    """
    kinds = selected.get("operand_field_kinds", {})
    if type(kinds) is not dict or any(
        type(label) is not str or label.count(".") != 1 or type(kind) is not str for label, kind in kinds.items()
    ):
        raise ValueError("operand sidecar field kinds must map CLASS.field labels to kinds")
    sources = selected.get("operand_field_kind_sources", [])
    if type(sources) is not list or sorted(row.get("label") for row in sources if type(row) is dict) != sorted(kinds):
        raise ValueError("operand sidecar field kinds lack exactly one RTL consumer citation each")
    roster = selected.get("operand_layout_roster")
    if roster is not None and (
        type(roster) is not dict
        or set(roster) != {"status", "derived_fields", "unprojected_fields"}
        or roster["status"] not in ("complete", "declared_subset")
        or (roster["status"] == "complete") != (roster["unprojected_fields"] == [])
    ):
        raise ValueError("operand sidecar layout roster is malformed")
    return dict(operand_field_kinds=copy.deepcopy(kinds), operand_layout_roster=copy.deepcopy(roster))


def selected_decoder_structure(selection):
    """``decoder_structure`` of an explicitly selected, byte-pinned sidecar."""
    if type(selection) is not dict or "sidecar" not in selection:
        raise ValueError("operand selection requires a pinned sidecar")
    return decoder_structure(_json(_read_pin(selection["sidecar"], 2 * 1024 * 1024), 2 * 1024 * 1024))


def selected_inputs(selection):
    """Explicit checker-only files that a fresh original profile must freeze."""
    if type(selection) is not dict or set(selection) != {"original_facts", "sidecar"}:
        raise ValueError("operand selection requires only original facts and a pinned sidecar")
    original_facts = _json(_read_pin(selection["original_facts"], 16 * 1024 * 1024), 16 * 1024 * 1024)
    selected = _json(_read_pin(selection["sidecar"], 2 * 1024 * 1024), 2 * 1024 * 1024)
    if type(selected) is not dict or selected.get("schema") != SCHEMA:
        raise ValueError("operand selection sidecar has an unsupported schema")
    _pins(selected.get("source_pins"))
    relocate_operand_sources(original_facts, selected.get("source_relocations"))
    layouts = selected.get("layouts")
    if type(layouts) is not dict or type(layouts.get("source")) is not str:
        raise ValueError("operand selection omits its exact FIRRTL source pin")
    # The layout reader supplies the separate large-FIRRTL bounds and byte check.
    verify_selected_layout_source(layouts)
    source = {"path": layouts["source"], "sha256": _sha(layouts["source_sha256"])}
    rows = [selection["original_facts"], selection["sidecar"], *selected["source_pins"], source]
    return tuple(dict.fromkeys(row["path"] for row in rows))


def selected_source_relocations(original, selection):
    """Return only the explicit relocation roster of a currently checked observer."""
    before = copy.deepcopy(selection)
    bind_selected_observer(original, selection)
    selected = _json(_read_pin(selection["sidecar"], 2 * 1024 * 1024), 2 * 1024 * 1024)
    if selection != before:
        raise ValueError("operand selection changed before source relocation publication")
    return copy.deepcopy(selected.get("source_relocations"))


def bind_selected_observer(original, selection):
    """Consume only explicit selected contract data, with final checks before return."""
    if type(selection) is not dict or set(selection) != {"original_facts", "sidecar"}:
        raise ValueError("operand selection requires only original facts and a pinned sidecar")
    facts = _read_pin(selection["original_facts"], 16 * 1024 * 1024)
    sidecar = _read_pin(selection["sidecar"], 2 * 1024 * 1024)
    observer = bind_original_operand_selection(
        original,
        original_facts_bytes=facts,
        selection_bytes=sidecar,
        expected_selection_sha256=selection["sidecar"]["sha256"],
    )
    if (
        _read_pin(selection["original_facts"], 16 * 1024 * 1024) != facts
        or _read_pin(selection["sidecar"], 2 * 1024 * 1024) != sidecar
    ):
        raise ValueError("operand selection changed before observer publication")
    return observer


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("operand selection needs an exact SHA-256 byte pin")
    return value


def _json(data, limit):
    if type(data) is not bytes or not 0 < len(data) <= limit:
        raise ValueError("operand selection exceeds its bounded byte document")

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("operand selection has duplicate JSON ownership")
            result[key] = value
        return result

    def constant(_value):
        raise ValueError("operand selection is not finite JSON data")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _pins(rows):
    if type(rows) is not list or not 1 <= len(rows) <= 64:
        raise ValueError("operand selection needs a bounded original source roster")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256"} or type(row["path"]) is not str:
            raise ValueError("operand source pin is malformed")
        path = Path(row["path"])
        if (
            not path.is_absolute()
            or path.resolve() != path
            or path.is_symlink()
            or not path.is_file()
            or path in seen
            or path.stat().st_size > 8 * 1024 * 1024
        ):
            raise ValueError("operand source is not a unique bounded ordinary selected file")
        seen.add(path)
        expected = _sha(row["sha256"])
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("selected operand semantic source bytes changed")


def bind_original_operand_selection(original, *, original_facts_bytes, selection_bytes, expected_selection_sha256):
    """Return an optional observer; original facts/contract/raw decode stay exact.

    Both byte documents are explicitly resolved by the caller's sealed original
    selection. Controller/ISA pins and selected FIRRTL packing are re-opened
    before publication. Selector validity, guard truth, queue acceptance and
    geometry/schedule proofs remain separate obligations.
    """
    if type(original) is not BoundSemantics or original.operand_observer is not None:
        raise ValueError("operand sidecar requires the exact original unextended semantics selection")
    expected_selection_sha256 = _sha(expected_selection_sha256)
    if type(selection_bytes) is not bytes or hashlib.sha256(selection_bytes).hexdigest() != expected_selection_sha256:
        raise ValueError("operand sidecar differs from its explicitly selected byte pin")
    selected = _json(selection_bytes, 2 * 1024 * 1024)
    required = {"schema", "target", "original_facts_sha256", "source_pins", "layouts", "protocol"}
    if (
        type(selected) is not dict
        or not required <= set(selected)
        or set(selected) - required - {"load_state_selection", "source_relocations", *_DECODER_KEYS}
        or selected["schema"] != SCHEMA
        or selected["target"] != original._target
    ):
        raise ValueError("operand sidecar is not a closed selection for the original target")
    original_facts = _json(original_facts_bytes, 16 * 1024 * 1024)
    if (
        hashlib.sha256(original_facts_bytes).hexdigest() != _sha(selected["original_facts_sha256"])
        or original_facts != original._facts
    ):
        raise ValueError("operand sidecar does not bind the unchanged original facts bytes and owner")
    decoder_structure(selected)
    observed_source_facts = relocate_operand_sources(original_facts, selected.get("source_relocations"))
    inputs = observed_source_facts.get("inputs") or {}
    isa = {"path": inputs.get("isa_path"), "sha256": inputs.get("isa_sha256")}
    _pins(selected["source_pins"])
    if isa not in selected["source_pins"]:
        raise ValueError("operand sidecar omits the original selected ISA source pin")
    raw_isa, width = original.isa_constants(original._target), original._width
    key = (
        hashlib.sha256(original_facts_bytes).digest(),
        expected_selection_sha256,
        hashlib.sha256(repr(raw_isa).encode()).digest(),
        original._target,
        width,
    )
    observed_facts, mapping = _structural_observations(original, selected, observed_source_facts, raw_isa, key)
    observer = BoundOperandObserver(
        target=original._target,
        protocol=selected["protocol"],
        facts=observed_facts,
        raw_isa=raw_isa,
        word_bits=width,
        **({"load_states": mapping} if mapping is not None else {}),
    )
    _pins(selected["source_pins"])
    verify_selected_layout_source(selected["layouts"])
    if relocate_operand_sources(original_facts, selected.get("source_relocations")) != observed_source_facts:
        raise ValueError("operand source relocations changed before observer publication")
    if (
        original._facts != original_facts
        or original._target != selected["target"]
        or original._width != width
        or original.isa_constants(original._target) != raw_isa
        or original.operand_observer is not None
    ):
        raise ValueError("original semantics changed before operand sidecar publication")
    # Do not mutate original._facts or its selected observer while returning this
    # detached view. All selected sources have final checks before publication.
    return observer
