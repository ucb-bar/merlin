"""Project typed storage probes from the selected complete application inventory."""

import copy
import hashlib
import json

from merlin.common.digest import is_sha256


def host_source_declaration(observed, family, dtype, inventory, inventory_sha256):
    if observed.get("op") not in {"permute", "dtype_cast"} or not inventory or not is_sha256(inventory_sha256):
        return None
    matches = []
    for app in (inventory.get("applications") or {}).values():
        for row in app.get("signatures") or ():
            operands, results = row.get("ordered_operand_types") or [], row.get("ordered_result_types") or []
            if (row.get("provenance_op") != observed["op"]
                    or row.get("frontend_op") != observed.get("frontend_op")
                    or row.get("semantic_family") != family
                    or row.get("disposition") not in {"host_required", "support_required"}
                    or not operands or operands[0].get("dtype") != dtype
                    or len(results) != 1 or not results[0].get("dtype")):
                continue
            signature = {key: value for key, value in row.items() if key not in {"count", "ordinals"}}
            digest = hashlib.sha256(json.dumps(
                signature, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode()).hexdigest()
            matches.append((results[0]["dtype"], app["capture_sha256"], digest, signature))
    if len({match[0] for match in matches}) != 1:
        return None
    destination, capture, digest, signature = min(matches, key=lambda match: (match[1], match[2]))
    declaration = {
        "capture_dtype": "fp32", "native_input_dtype": dtype, "native_output_dtype": destination,
        "native_source_abi": {"full_inventory_sha256": inventory_sha256, "capture_sha256": capture,
                              "signature_sha256": digest, "signature": signature},
    }
    from .native_storage_source import native_storage_source_spec

    try:
        native_storage_source_spec({**observed, **declaration,
            "generalization": {"must_accelerate": False, "eligible": False}})
    except ValueError:
        return None
    return declaration


def apply_probe_declaration(entry, observed):
    for key in ("capture_dtype", "native_input_dtype", "native_output_dtype", "native_source_abi"):
        if key in observed:
            entry[key] = copy.deepcopy(observed[key])
