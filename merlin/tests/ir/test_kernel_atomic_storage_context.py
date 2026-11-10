"""Declared source context and actual capture dtype bind each atomic view."""

from copy import deepcopy
import json

import pytest

from merlin.frontends.kernel_word_evidence import checked_word_stages
from test_kernel_compile_inputs import word_fixture


def fixture(root, width, cas, reverse):
    # Byte custody only. Producer composition tests own executable semantics.
    word_fixture(root, isolated=True)
    stage = json.loads((root / "word-stage.json").read_text())
    provenance = json.loads((root / "provenance.json").read_text())
    memory = provenance["memory"]
    memory["effect_profile"] = "private-invocation-typed-atomics-v2"
    binding = stage["events"]["bindings"][0]
    binding.update(opcode="tt.atomic_cas" if cas else "tt.atomic_rmw", fill="unchanged",
        input_types=[f"!tt.ptr<i{width}>", "tensor<2xi64>", *[f"tensor<2xi{width}>"] * (2 if cas else 1)],
        result_types=[f"tensor<2xi{width}>"])
    integer = deepcopy(binding)
    integer["symbol"] = "integer"
    floating = deepcopy(binding)
    floating.update(symbol="floating", word_view={"storage_dtype": f"f{width}", "value_dtype": f"i{width}",
        "element_bits": width, "address_units": "equal-width-elements"})
    floating["input_types"][0] = f"!tt.ptr<f{width}>"
    if cas:
        floating["template_input_types"] = deepcopy(integer["input_types"])
    primitive = {"source_ids": deepcopy(binding["source_ids"]), "operation": "cas" if cas else "xor", "word_bits": width}
    plain = {**deepcopy(primitive), "buffer": "integer"}
    viewed = {**deepcopy(primitive), "buffer": "floating", "word_view": deepcopy(floating["word_view"])}
    stage["events"]["bindings"] = [floating, integer] if reverse else [integer, floating]
    memory["atomic_payload"] = [viewed, plain] if reverse else [plain, viewed]
    inventory = {r["id"] for r in provenance["normalized_operations"]}
    args = (stage, (root / "word-input.mlir").read_bytes(), (root / "word-tensor.mlir").read_bytes(), memory, inventory)
    return args, integer, floating, plain, viewed


@pytest.mark.parametrize("width,cas", [(16, True), (32, True), (64, True), (32, False), (64, False)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("change", [None, "lost-integer", "lost-floating", "context", "view", "lost-binding-view", "capture", "profile", "address-space"])
def test_selected_atomic_views_bind_context_and_actual_capture_storage(tmp_path, width, cas, reverse, change):
    args, integer, floating, plain, viewed = fixture(tmp_path / "capture", width, cas, reverse)
    memory = args[3]
    if change == "lost-integer": memory["atomic_payload"].remove(plain)
    elif change == "lost-floating": memory["atomic_payload"].remove(viewed)
    elif change == "context": viewed["source_ids"] = ["unrelated-call-context"]
    elif change == "view": viewed["word_view"]["value_dtype"] = f"f{width}"
    elif change == "lost-binding-view": floating.pop("word_view")
    elif change == "capture": integer["input_types"][0] = floating["input_types"][0]
    elif change == "profile": memory["effect_profile"] = "private-invocation-integer-atomics-v1"
    elif change == "address-space": floating["input_types"][0] = f"!tt.ptr<f{width},3>"
    if change is None:
        checked_word_stages(*args)
    else:
        with pytest.raises(ValueError, match="storage|typed atomic"):
            checked_word_stages(*args)
