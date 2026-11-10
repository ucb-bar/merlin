"""Byte-bound declared atomic effects; producer/execution proofs stay separate."""

EFFECT_PROFILE = "private-invocation-integer-atomics-v1"
TYPED_EFFECT_PROFILE = "private-invocation-typed-atomics-v2"
_KINDS = {"and": 2, "or": 14, "xor": 15, "add": 1, "max": 6, "min": 10, "umax": 7, "umin": 11, "exch": 3,
          "cas": "compare-exchange", "fadd": 0}
_FLOAT_CONTRACTS = {("exch", "f32"): ("bitwise-exchange",), ("exch", "f64"): ("bitwise-exchange",),
                    ("fadd", "f64"): ("ieee-f64-rne-add", "ieee-f64-rne-add-quiet-nans"),
                    ("fadd", "f32"): ("ieee-f32-rne-add", "f32-rne-sign-preserving-ftz-add")}


def _payload(info):
    if info.get("numeric_contract") == "ieee-f64-rne-add-quiet-nans":
        return "standard-memref-generic-f64-quiet-add-with-original-returned-old-value"
    if info.get("numeric_contract") == "ieee-f32-rne-add":
        return "standard-memref-generic-f32-quiet-add-with-original-returned-old-value"
    if info.get("numeric_contract") == "f32-rne-sign-preserving-ftz-add":
        return "standard-memref-generic-f32-ftz-add-with-original-returned-old-value"
    if "word_view" in info:
        return "standard-memref-generic-integer-word-update-with-original-returned-old-value"
    return ("standard-memref-generic-compare-exchange-with-original-returned-old-value" if info["operation"] == "cas"
            else "standard-memref-atomic-rmw-with-original-returned-old-value")


def checked_atomic_effects(admission, memory, records, buffers):
    profile = TYPED_EFFECT_PROFILE if admission.get("schema") == "t2m.memory-admission.v4" else EFFECT_PROFILE
    if (admission.get("effect_profile") != profile or memory.get("effect_profile") != profile
            or admission.get("f32_atomic_add") not in (None, "ieee", "ftz")
            or memory.get("f32_atomic_add") not in (None, "ieee", "ftz")
            or admission.get("address_policy") != "source-words-offset64"
            or not isinstance(admission.get("atomic_adaptations"), list)
            or admission["atomic_adaptations"] != memory.get("atomic_payload")):
        raise ValueError("atomic memory evidence lacks its explicit private-invocation effect contract")
    source = {r["id"]: r.get("atomic") for r in records if r.get("opcode") in ("tt.atomic_rmw", "tt.atomic_cas")}
    opcodes = {r["id"]: r.get("opcode") for r in records}
    accesses = [a for a in admission["accesses"] if a["effect"] == "atomic-readwrite"]
    if len(accesses) != len(admission["atomic_adaptations"]):
        raise ValueError("atomic memory evidence does not account for every payload event")
    for access, actual in zip(accesses, admission["atomic_adaptations"], strict=True):
        info = access.get("atomic")
        buffer = buffers[access["buffer"]]
        candidates = [source[s] for s in access["source_ids"] if s in source]
        floating = isinstance(info, dict) and info.get("value_dtype") in ("f32", "f64")
        dtype = info.get("value_dtype") if floating else "i" + str(info.get("word_bits")) if isinstance(info, dict) else None
        word = isinstance(info, dict) and "word_view" in info
        # Accept the original primitive or the exact legacy enriched record.
        primitive = {key: value for key, value in info.items() if key != "word_view"} if word else info
        expected_view = {"storage_dtype": "f" + str(info.get("word_bits")), "value_dtype": "i" + str(info.get("word_bits")),
            "element_bits": info.get("word_bits"), "address_units": "equal-width-elements"} if word else None
        if word:
            dtype = expected_view["storage_dtype"]
        if (not isinstance(info, dict) or not candidates or any(c != info and c != primitive for c in candidates)
                or buffer.get("dtype") != dtype
                or word and (profile != TYPED_EFFECT_PROFILE or floating or info.get("operation") == "cas"
                    or info.get("word_bits") not in (32, 64) or info["word_view"] != expected_view)
                or floating and (profile != TYPED_EFFECT_PROFILE
                    or (info.get("operation"), dtype) not in _FLOAT_CONTRACTS
                    or dtype != "f" + str(info.get("word_bits"))
                    or info.get("numeric_contract") not in _FLOAT_CONTRACTS.get((info.get("operation"), dtype), ()))
                or not floating and info.get("operation") == "fadd"
                or floating and info.get("operation") == "fadd" and dtype == "f32"
                    and (admission.get("f32_atomic_add") != memory.get("f32_atomic_add")
                         or info.get("numeric_contract") != {"ieee": "ieee-f32-rne-add", "ftz": "f32-rne-sign-preserving-ftz-add"}.get(admission.get("f32_atomic_add")))
                or not floating and ("value_dtype" in info or "numeric_contract" in info)
                or info.get("word_bits") not in ((16, 32, 64) if info.get("operation") == "cas" else (32, 64))
                or info.get("operation") not in _KINDS or info.get("standard_kind") != _KINDS[info["operation"]]
                or access.get("opcode") != ("tt.atomic_cas" if info["operation"] == "cas" else "tts.unstructured_atomic_rmw")
                or any(opcodes[s] != ("tt.atomic_cas" if info["operation"] == "cas" else "tt.atomic_rmw") for s in access["source_ids"] if s in source)
                or info["operation"] == "cas" and access.get("mask") != "unmasked"
                or info.get("source_order") not in ("relaxed", "acquire", "release", "acq_rel")
                or info.get("source_scope") not in ("gpu", "cta", "sys")
                or info.get("standard_order") != "acq_rel"
                or info.get("logical_scope") != ("one-logical-program" if info["source_scope"] == "cta"
                    else "all-logical-programs-in-private-invocation")
                or info.get("external_participants") != "excluded-by-private-buffer-ABI"
                or not isinstance(actual, dict) or any(actual.get(key) != value for key, value in info.items())
                or actual.get("buffer") != buffer["name"] or not actual.get("source_ids")
                or not set(actual["source_ids"]) & set(access["source_ids"])
                or actual.get("payload") != _payload(info)
                or actual.get("effect_boundary") != "standard-execute-region-no-inline"
                or actual.get("inactive_old_value") != ("zero-refinement-of-source-undefined-old-value"
                    if access["mask"] == "source-per-lane" else "no-inactive-lanes")):
            raise ValueError("atomic memory payload differs from its typed source operation/order/scope inventory")
