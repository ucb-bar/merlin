"""Explicit reviewed lane-crossing contracts, not inferred runtime dispatch."""

from __future__ import annotations

import copy


def validate_transfer_contracts(document: dict | None) -> dict | None:
    if document is None:
        return None
    if not isinstance(document, dict):
        raise ValueError("transfer_contracts must be a mapping")
    document = copy.deepcopy(document)
    document.setdefault("status", "unreviewed")
    if document["status"] not in {"reviewed", "unreviewed"}:
        raise ValueError("transfer_contracts has an invalid review status")
    declarations = document.get("declarations")
    if declarations is None:
        # Directly named transfers are an authoring form, not another consumer
        # contract. Keep review status separate from transfer identities.
        declarations = {key: value for key, value in document.items() if key != "status"}
        if not declarations:
            raise ValueError("transfer_contracts requires explicit declarations")
        document = {"status": document["status"], "declarations": declarations}
    if isinstance(declarations, dict):
        normalized = []
        for identity, declaration in declarations.items():
            if not isinstance(identity, str) or not identity.strip() or not isinstance(declaration, dict):
                raise ValueError("named transfer declarations require nonempty names and mappings")
            row = copy.deepcopy(declaration)
            if "id" in row and row["id"] != identity:
                raise ValueError("named transfer id conflicts with its name")
            row["id"] = identity
            normalized.append(row)
        declarations = normalized
        document["declarations"] = declarations
    if not isinstance(declarations, list):
        raise ValueError("transfer_contracts.declarations must be a mapping or list")
    seen = set()
    for row in declarations:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("transfer declarations require unique IDs")
        seen.add(row["id"])
        if "copy" in row:
            if "signature" in row or "semantics" in row:
                raise ValueError("copy transfer cannot mix compact and expanded constraints")
            if set(row) - {"id", "from", "to", "copy", "valid_window", "evidence"}:
                raise ValueError("copy transfer contains unknown fields")
            copied = row.pop("copy")
            if (
                not isinstance(copied, dict)
                or set(copied) != {"dtype", "layout"}
                or any(not isinstance(value, str) or not value.strip() for value in copied.values())
            ):
                raise ValueError("copy transfer requires explicit dtype and layout")
            if "valid_window" in row and row["valid_window"] != "required":
                raise ValueError("compact copy valid_window must explicitly be required")
            row["signature"] = {
                "operand_dtype": copied["dtype"],
                "result_dtype": copied["dtype"],
                "operand_layout": copied["layout"],
                "result_layout": copied["layout"],
            }
            row["semantics"] = {"conversion": "bit_preserving"}
            if "valid_window" in row:
                row["semantics"]["valid_window"] = row.pop("valid_window")
        if row.get("from") not in {"host", "accelerator"} or row.get("to") not in {"host", "accelerator"}:
            raise ValueError("transfer declarations must name explicit host/accelerator endpoints")
        signature = row.get("signature")
        if not isinstance(signature, dict) or any(
            not isinstance(signature.get(key), str) or not signature[key] for key in ("operand_dtype", "result_dtype")
        ):
            raise ValueError("transfer declarations require explicit operand/result dtypes")
        if set(signature) - {"operand_dtype", "result_dtype", "operand_layout", "result_layout"}:
            raise ValueError("transfer declarations contain unknown typed constraints")
        if "evidence" not in row and document["status"] == "unreviewed":
            row["evidence"] = {}
        if not isinstance(row.get("semantics"), dict) or not isinstance(row.get("evidence"), dict):
            raise ValueError("transfer declarations require structured semantics and review evidence")
    return document


def _canonical(key: str, value):
    """One spelling per dtype, from the quant-format registry, exactly as operation admission compares.

    An MLIR element type (``i8``) and the registry name an author writes (``int8``) are the same
    format; comparing their spellings refused every observed transfer. Layouts compare as written.
    """
    if not key.endswith("_dtype"):
        return value
    from merlin.common.quant_formats import get as quant_format

    try:
        return quant_format(value).name
    except (KeyError, ValueError, TypeError):
        return value


def screen_transfer_contract(
    software_spec: dict | None,
    *,
    source_placement: str,
    destination_placement: str,
    operand_dtype: str | None,
    result_dtype: str | None,
    operand_layout: str | None = None,
    result_layout: str | None = None,
) -> dict:
    """Screen an actually selected lane crossing; missing observations stay unknown."""
    if source_placement not in {"host", "accelerator"} or destination_placement not in {"host", "accelerator"}:
        return {
            "status": "unknown",
            "reviewed": False,
            "matching_declarations": [],
            "reason": "transfer endpoints are not selected",
        }
    if source_placement == destination_placement:
        return {
            "status": "not_applicable",
            "reviewed": False,
            "matching_declarations": [],
            "reason": "same-lane SSA use is not itself a cross-lane transfer",
        }
    selected = validate_transfer_contracts((software_spec or {}).get("transfer_contracts"))
    if selected is None:
        return {
            "status": "unknown",
            "reviewed": False,
            "matching_declarations": [],
            "reason": "no explicit software transfer contract",
        }
    observations = {
        "operand_dtype": operand_dtype,
        "result_dtype": result_dtype,
        "operand_layout": operand_layout,
        "result_layout": result_layout,
    }
    decisions = []
    for row in selected["declarations"]:
        if (row["from"], row["to"]) != (source_placement, destination_placement):
            continue
        constraints = row["signature"]
        missing = [key for key in constraints if observations[key] is None or constraints[key] == "unknown"]
        refused = [
            key
            for key in constraints
            if observations[key] is not None
            and constraints[key] != "unknown"
            and _canonical(key, constraints[key]) != _canonical(key, observations[key])
        ]
        semantics_known = (
            bool(row["semantics"])
            and bool(row["evidence"])
            and row["evidence"].get("status")
            not in {
                "unknown",
                "unqualified",
                "unreviewed",
            }
        )
        status = (
            "unsupported"
            if refused
            else "unknown"
            if missing or selected["status"] != "reviewed" or not semantics_known
            else "admitted"
        )
        decisions.append({"id": row["id"], "status": status, "missing": missing, "refused": refused})
    verdict = next(
        (row for row in decisions if row["status"] == "admitted"),
        next((row for row in decisions if row["status"] == "unknown"), None),
    )
    return {
        "status": verdict["status"] if verdict else "unsupported",
        "reviewed": selected["status"] == "reviewed",
        "matching_declarations": [row["id"] for row in decisions],
        "decisions": decisions,
        "reason": "typed selected transfer declaration screen; execution remains unverified",
    }
