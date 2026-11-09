"""Byte-bound declared atomic effects; producer/execution proofs stay separate."""

EFFECT_PROFILE = "private-invocation-integer-atomics-v1"
_KINDS = {"and": 2, "or": 14, "xor": 15, "add": 1, "max": 6, "min": 10, "umax": 7, "umin": 11, "exch": 3}


def checked_atomic_effects(admission, memory, records, buffers):
    if (admission.get("effect_profile") != EFFECT_PROFILE or memory.get("effect_profile") != EFFECT_PROFILE
            or admission.get("address_policy") != "source-words-offset64"
            or not isinstance(admission.get("atomic_adaptations"), list)
            or admission["atomic_adaptations"] != memory.get("atomic_payload")):
        raise ValueError("atomic memory evidence lacks its explicit private-invocation effect contract")
    source = {r["id"]: r.get("atomic") for r in records if r.get("opcode") == "tt.atomic_rmw"}
    accesses = [a for a in admission["accesses"] if a["effect"] == "atomic-readwrite"]
    if len(accesses) != len(admission["atomic_adaptations"]):
        raise ValueError("atomic memory evidence does not account for every payload event")
    for access, actual in zip(accesses, admission["atomic_adaptations"], strict=True):
        info = access.get("atomic")
        buffer = buffers[access["buffer"]]
        candidates = [source[s] for s in access["source_ids"] if s in source]
        if (not isinstance(info, dict) or not candidates or any(c != info for c in candidates)
                or buffer.get("dtype") != "i" + str(info.get("word_bits"))
                or info.get("word_bits") not in (32, 64)
                or info.get("operation") not in _KINDS or info.get("standard_kind") != _KINDS[info["operation"]]
                or info.get("source_order") not in ("relaxed", "acquire", "release", "acq_rel")
                or info.get("source_scope") not in ("gpu", "cta", "sys")
                or info.get("standard_order") != "acq_rel"
                or info.get("logical_scope") != ("one-logical-program" if info["source_scope"] == "cta"
                    else "all-logical-programs-in-private-invocation")
                or info.get("external_participants") != "excluded-by-private-buffer-ABI"
                or not isinstance(actual, dict) or any(actual.get(key) != value for key, value in info.items())
                or actual.get("buffer") != buffer["name"] or not actual.get("source_ids")
                or not set(actual["source_ids"]) & set(access["source_ids"])
                or actual.get("payload") != "standard-memref-atomic-rmw-with-original-returned-old-value"
                or actual.get("effect_boundary") != "standard-execute-region-no-inline"
                or actual.get("inactive_old_value") != ("zero-refinement-of-source-undefined-old-value"
                    if access["mask"] == "source-per-lane" else "no-inactive-lanes")):
            raise ValueError("atomic memory payload differs from its typed source operation/order/scope inventory")
