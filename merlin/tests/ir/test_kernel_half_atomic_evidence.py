"""Consumer half-atomic declarations independently bind storage and arithmetic."""

from copy import deepcopy

import pytest

from merlin.frontends.kernel_atomic_evidence import checked_atomic_effects


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(
    "change", [None, "width", "fractional-width", "format", "rounding", "payload", "storage", "kind", "profile"]
)
def test_half_atomic_evidence_is_typed_and_complete(dtype, change):
    info = {
        "operation": "fadd",
        "standard_kind": 0,
        "word_bits": 16,
        "value_dtype": dtype,
        "numeric_contract": f"ieee-{dtype}-rne-add-quiet-nans",
        "source_order": "acq_rel",
        "source_scope": "sys",
        "logical_scope": "all-logical-programs-in-private-invocation",
        "standard_order": "acq_rel",
        "external_participants": "excluded-by-private-buffer-ABI",
    }
    access = {
        "source_ids": ["source"],
        "opcode": "tts.unstructured_atomic_rmw",
        "buffer": "state",
        "effect": "atomic-readwrite",
        "mask": "source-per-lane",
        "atomic": info,
    }
    actual = {
        **info,
        "source_ids": ["source"],
        "buffer": "state",
        "inactive_old_value": "zero-refinement-of-source-undefined-old-value",
        "effect_boundary": "standard-execute-region-no-inline",
        "payload": f"standard-memref-generic-{dtype}-quiet-add-with-original-returned-old-value",
    }
    profile = "private-invocation-typed-atomics-v2"
    admission = {
        "schema": "t2m.memory-admission.v4",
        "effect_profile": profile,
        "address_policy": "source-words-offset64",
        "accesses": [access],
        "atomic_adaptations": [actual],
    }
    memory = {"effect_profile": profile}
    buffers = {"state": {"name": "state", "dtype": dtype}}
    if change == "width":
        info["word_bits"] = 32
    elif change == "fractional-width":
        info["word_bits"] = 16.0
    elif change == "format":
        info["numeric_contract"] = f"ieee-{'bf16' if dtype == 'f16' else 'f16'}-rne-add-quiet-nans"
    elif change == "rounding":
        info["numeric_contract"] = "round-toward-zero"
    elif change == "payload":
        actual["payload"] = "unqualified-add"
    elif change == "storage":
        buffers["state"]["dtype"] = "i16"
    elif change == "kind":
        info.update(operation="exch", standard_kind=3)
    elif change == "profile":
        admission.update(schema="t2m.memory-admission.v3", effect_profile="private-invocation-integer-atomics-v1")
        memory["effect_profile"] = admission["effect_profile"]
    actual.update(info)
    memory["atomic_payload"] = deepcopy([actual])
    records = [{"id": "source", "opcode": "tt.atomic_rmw", "atomic": deepcopy(info)}]
    if change is None:
        checked_atomic_effects(admission, memory, records, buffers)
    else:
        with pytest.raises(ValueError, match="atomic memory"):
            checked_atomic_effects(admission, memory, records, buffers)
