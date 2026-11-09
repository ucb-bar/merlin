"""Explicit operator-reviewed additive observer revision; never a relocation."""
from __future__ import annotations

import copy

from merlin.runtime.backends.rocc_selection import select_operand_protocol
from merlin.targetgen.rocc.selected_operand_protocol import _json, _read_pin

SCHEMA = "merlin.operand_observer_revision.v1"


def select(contract, selection, review_pin):
    """Preserve facts and all existing observations; admit exactly reviewed additions.

    The release operator supplies this pin explicitly. It is not discovered from
    candidate outputs or an ambient path. Normal relocation remains byte-exact.
    Native operand binding still derives and checks the complete source closure.
    """
    review_bytes = _read_pin(review_pin, 2 * 1024 * 1024)
    review = _json(review_bytes, 2 * 1024 * 1024)
    required = {"schema", "previous_sidecar_sha256", "selected_sidecar_sha256", "original_facts_sha256",
                "field_additions", "source_pin_replacements"}
    if type(review) is not dict or set(review) != required or review["schema"] != SCHEMA:
        raise ValueError("observer revision requires a closed explicit review")
    previous = contract.get("rocc_operand_selection")
    if type(previous) is not dict or set(previous) != {"original_facts", "sidecar"}:
        raise ValueError("observer revision requires an existing selected observer")
    if type(selection) is not dict or set(selection) != set(previous):
        raise ValueError("observer revision requires complete selected facts and sidecar")
    if (previous["sidecar"]["sha256"] != review["previous_sidecar_sha256"]
            or selection["sidecar"]["sha256"] != review["selected_sidecar_sha256"]
            or previous["original_facts"]["sha256"] != review["original_facts_sha256"]
            or selection["original_facts"]["sha256"] != review["original_facts_sha256"]):
        raise ValueError("observer revision identities differ from the explicit review")
    originals = {k: _read_pin(v, 16 * 1024 * 1024) for k, v in previous.items()}
    revised = {k: _read_pin(v, 16 * 1024 * 1024) for k, v in selection.items()}
    if originals["original_facts"] != revised["original_facts"]:
        raise ValueError("observer revision changed original facts")
    before = _json(originals["sidecar"], 2 * 1024 * 1024)
    after = _json(revised["sidecar"], 2 * 1024 * 1024)
    stripped = copy.deepcopy(after)
    additions = review["field_additions"]
    if type(additions) is not dict or not additions or len(additions) > 64:
        raise ValueError("observer revision requires bounded additive fields")
    for name, fields in additions.items():
        if type(fields) is not list or not fields or len(fields) > 32:
            raise ValueError("observer field addition is not a bounded roster")
        if any(type(f) is not dict or type(f.get("name")) is not str for f in fields):
            raise ValueError("observer addition omits field identity")
        names = [f["name"] for f in fields]
        old_fields = before.get("protocol", {}).get("roles", {}).get("classes", {}).get(name, {}).get("fields")
        new_fields = stripped.get("protocol", {}).get("roles", {}).get("classes", {}).get(name, {}).get("fields")
        if (type(old_fields) is not list or type(new_fields) is not list or len(set(names)) != len(names)
                or set(names) & {f["name"] for f in old_fields}
                or [f for f in new_fields if f.get("name") in names] != fields):
            raise ValueError("observer addition replaced existing fields or differs from review")
        stripped["protocol"]["roles"]["classes"][name]["fields"] = [f for f in new_fields if f.get("name") not in names]
    replacements = review["source_pin_replacements"]
    if type(replacements) is not list or len(replacements) > 64:
        raise ValueError("observer source replacements exceed bounded review")
    for row in replacements:
        if type(row) is not dict or set(row) != {"previous", "selected"}:
            raise ValueError("observer source replacement requires both exact pins")
        _read_pin(row["previous"], 8 * 1024 * 1024)
        _read_pin(row["selected"], 8 * 1024 * 1024)
        if before["source_pins"].count(row["previous"]) != 1 or stripped["source_pins"].count(row["selected"]) != 1:
            raise ValueError("observer source replacement has ambiguous ownership")
        stripped["source_pins"][stripped["source_pins"].index(row["selected"])] = row["previous"]
    if stripped != before:
        raise ValueError("observer revision contains unreviewed or nonadditive changes")
    base = copy.deepcopy(contract)
    del base["rocc_operand_selection"]
    result = select_operand_protocol(base, selection)
    if (_read_pin(review_pin, 2 * 1024 * 1024) != review_bytes
            or any(_read_pin(previous[k], 16 * 1024 * 1024) != v for k, v in originals.items())
            or any(_read_pin(selection[k], 16 * 1024 * 1024) != v for k, v in revised.items())):
        raise ValueError("observer revision inputs changed before publication")
    return result
