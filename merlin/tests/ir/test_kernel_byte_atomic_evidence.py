"""Independent declared byte/word contracts; executable proofs stay separate."""

import json
from copy import deepcopy

import pytest
from test_kernel_compile_inputs import word_fixture

from merlin.frontends.kernel_atomic_evidence import checked_atomic_effects
from merlin.frontends.kernel_word_evidence import checked_word_stages


@pytest.mark.parametrize("width,cas", ((16, True), (32, True), (64, True), (32, False), (64, False)))
@pytest.mark.parametrize("order", ("little", "big"))
@pytest.mark.parametrize(
    "change",
    (
        None,
        "order",
        "capture",
        "formal",
        "lost-template",
        "lost-view",
        "units",
        "guard",
        "primitive",
        "width",
        "dtype",
        "mask",
        "fractional-width",
        "fractional-alignment",
        "fractional-tile",
    ),
)
def test_byte_word_bindings_require_source_storage_units_and_complete_guards(tmp_path, width, cas, order, change):
    root = tmp_path / "capture"
    word_fixture(root, isolated=True)
    stage = json.loads((root / "word-stage.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    memory = provenance["memory"]
    profile = "private-invocation-typed-atomics-v2"
    view = {
        "storage_dtype": "i8",
        "value_dtype": f"i{width}",
        "word_bits": width,
        "byte_order": order,
        "address_units": "signed-i64-bytes",
        "template_address_units": "word-elements-after-active-alignment",
        "alignment_bytes": width // 8,
    }
    binding = stage["events"]["bindings"][0]
    opcode = "tt.atomic_cas" if cas else "tt.atomic_rmw"
    types = ["!tt.ptr<i8>", "tensor<2xi64>", *[f"tensor<2xi{width}>"] * (2 if cas else 1)]
    binding.update(
        opcode=opcode,
        fill="unchanged",
        input_types=types,
        result_types=[f"tensor<2xi{width}>"],
        template_input_types=[f"!tt.ptr<i{width}>", *types[1:]],
        byte_view=deepcopy(view),
    )
    stage["events"].update(byte_order=order, bindings=[binding])
    primitive = {
        "operation": "cas" if cas else "add",
        "standard_kind": "compare-exchange" if cas else 1,
        "word_bits": width,
        "source_order": "acq_rel",
        "source_scope": "gpu",
        "logical_scope": "all-logical-programs-in-private-invocation",
        "standard_order": "acq_rel",
        "external_participants": "excluded-by-private-buffer-ABI",
    }
    info = {**primitive, "byte_view": deepcopy(view)}
    access = {
        "source_ids": binding["source_ids"],
        "opcode": "tt.atomic_cas" if cas else "tts.unstructured_atomic_rmw",
        "buffer": "state",
        "effect": "atomic-readwrite",
        "tile": [2],
        "guard_tile": [2, width // 8],
        "mask": "unmasked",
        "atomic": info,
    }
    actual = {
        **info,
        "source_ids": binding["source_ids"],
        "buffer": "state",
        "inactive_old_value": "no-inactive-lanes",
        "effect_boundary": "standard-execute-region-no-inline",
        "payload": "standard-byte-backed-word-atomic-with-original-returned-old-value",
    }
    admission = {
        "schema": "t2m.memory-admission.v4",
        "effect_profile": profile,
        "byte_order": order,
        "address_policy": "source-words-offset64",
        "accesses": [access],
        "atomic_adaptations": [actual],
    }
    memory.update(effect_profile=profile, byte_order=order, atomic_payload=[actual])
    records = [{"id": s, "opcode": opcode, "atomic": deepcopy(primitive)} for s in binding["source_ids"]]
    buffers = {"state": {"name": "state", "dtype": "i8"}}
    if change == "order":
        stage["events"]["byte_order"] = "big" if order == "little" else "little"
    elif change == "capture":
        binding["input_types"][0] = f"!tt.ptr<i{width}>"
    elif change == "formal":
        binding["template_input_types"][0] = "!tt.ptr<i8>"
    elif change == "lost-template":
        binding.pop("template_input_types")
    elif change == "lost-view":
        binding.pop("byte_view")
    elif change == "units":
        info["byte_view"]["address_units"] = "word-elements"
    elif change == "guard":
        access["guard_tile"] = [2]
    elif change == "primitive":
        records[0]["atomic"]["operation"] = "xor"
    elif change == "width":
        info["word_bits"] *= 2
    elif change == "dtype":
        buffers["state"]["dtype"] = f"i{width}"
    elif change == "mask":
        access["mask"] = "source-prefix"
    elif change == "fractional-width":
        info["word_bits"] = float(width)
    elif change == "fractional-alignment":
        info["byte_view"]["alignment_bytes"] = float(width // 8)
    elif change == "fractional-tile":
        access["guard_tile"][-1] = float(width // 8)

    def check():
        checked_atomic_effects(admission, memory, records, buffers)
        checked_word_stages(
            stage,
            (root / "word-input.mlir").read_bytes(),
            (root / "word-tensor.mlir").read_bytes(),
            memory,
            {r["id"] for r in provenance["normalized_operations"]},
        )

    if change is None:
        check()
    else:
        with pytest.raises(ValueError):
            check()
