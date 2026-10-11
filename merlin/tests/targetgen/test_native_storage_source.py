"""Typed storage probes retain real ABI types and reject substituted captures."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_source as source
from merlin.targetgen import corpus_spec, corpus_synth
from merlin.targetgen.native_storage_probe import host_source_declaration
from merlin.targetgen.native_storage_source import native_storage_source_spec, publish_native_storage


def _spec(op, dtype, result):
    family = "movement" if op == "permute" else "elementwise_map"
    frontend = "aten.permute.default" if op == "permute" else "aten._to_copy.default"
    signature = {
        "provenance_op": op, "frontend_op": frontend, "semantic_family": family,
        "disposition": "support_required" if op == "permute" else "host_required",
        "ordered_operand_types": [{"dtype": dtype, "shape": [17, 19]},
                                   {"dtype": result, "shape": [17, 19]}],
        "ordered_result_types": [{"dtype": result, "shape": [19, 17] if op == "permute" else [17, 19]}],
    }
    return {
        "op": op, "frontend_op": frontend, "operand_dtype": dtype, "dtype": "fp32", "capture_dtype": "fp32",
        "M": 2, "K": 3, "N": 3, "shape": [2, 3], "seed": 7,
        "native_input_dtype": dtype, "native_output_dtype": result,
        "generalization": {"must_accelerate": False, "eligible": False},
        "native_source_abi": {
            "full_inventory_sha256": "a" * 64, "capture_sha256": "b" * 64,
            "signature_sha256": hashlib.sha256(json.dumps(
                signature, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(), "signature": signature,
        },
    }


@pytest.mark.parametrize("op,dtype,result", [("permute", "i8", "i8"), ("permute", "f32", "f32"),
                                            ("dtype_cast", "i32", "f32")])
def test_native_loaders_use_real_storage_and_publish_actual_result_policy(op, dtype, result):
    torch = pytest.importorskip("torch")
    spec = _spec(op, dtype, result)
    assert corpus_synth._writer_for(spec) == "pytorch"
    scope = {}
    exec(source.build_loader_src(spec), scope)
    model, inputs = scope["get_model_and_inputs"]()
    _, repeated = scope["get_model_and_inputs"]()
    assert torch.equal(inputs[0], repeated[0])
    assert str(inputs[0].dtype).removeprefix("torch.") == {"i8": "int8", "i32": "int32", "f32": "float32"}[dtype]
    output = model(*inputs)
    expected = inputs[0].permute(1, 0) if op == "permute" else inputs[0].to(torch.float32)
    assert torch.equal(output, expected)
    shape, outshape = [2, 3], list(output.shape)
    tensor = lambda dims, dt: "tensor<" + "x".join(map(str, dims)) + "x" + dt + ">"
    argtype, outtype = tensor(shape, dtype), tensor(outshape, result)
    art = SimpleNamespace(op=op, dtype="fp32", weights_path=None, inputs=[inputs[0].tolist()],
        golden=output.tolist(), meta={"input_abi": [{"shape": shape, "dtype": dtype}],
                                     "output_abi": [{"shape": outshape, "dtype": result}]},
        linalg_mlir=f"module {{ func.func @forward(%x: {argtype}) -> {outtype} {{\n"
                    f"%y = tensor.empty() : {outtype}\nreturn %y : {outtype}\n}} }}")
    binding = corpus_spec.CorpusBinding("fixture", 4, dtype, "i32", True, ["L0", "L3"], "exact_int")
    cap = source._fused_capsule_yaml({**spec, "name": "probe"}, binding, art, ["X"])
    golden = source._host_eager_golden(art, ["X"], "Y0", binding,
                                      interface="linalg_positional", arg_order=["X", "Y0"])
    publish_native_storage(spec, art, cap, golden, binding)
    assert cap["inputs"][0]["dtype"] == dtype
    assert cap["numeric_policy"]["dtype"] == result
    assert cap["numeric_policy"]["compare"] == ("exact_int" if result == "i8" else "tolerance_float")
    assert cap["required_oracle_tiers"] == ["L0", "L3"]
    for mutation in ("dtype", "ragged", "out_of_range", "extra_result"):
        changed = copy.deepcopy(art)
        if mutation == "dtype":
            changed.meta["input_abi"][0]["dtype"] = "f64"
        elif mutation == "ragged":
            changed.inputs[0][0].pop()
        elif mutation == "out_of_range":
            changed.inputs[0][0][0] = float("inf")
        else:
            changed.meta["output_abi"].append(copy.deepcopy(changed.meta["output_abi"][0]))
        with pytest.raises(ValueError):
            publish_native_storage(spec, changed, cap, golden, binding)


def test_cast_destination_is_projected_from_source_and_ambiguity_refuses():
    spec = _spec("dtype_cast", "i32", "f32")
    signature = spec["native_source_abi"]["signature"]
    inventory = {"applications": {"public": {"capture_sha256": "b" * 64, "signatures": [signature]}}}
    observed = {"op": spec["op"], "frontend_op": spec["frontend_op"]}
    declaration = host_source_declaration(observed, "elementwise_map", "i32", inventory, "a" * 64)
    assert declaration["native_output_dtype"] == "f32"
    conflicting = copy.deepcopy(signature)
    conflicting["ordered_result_types"][0]["dtype"] = "f16"
    inventory["applications"]["public"]["signatures"].append(conflicting)
    assert host_source_declaration(observed, "elementwise_map", "i32", inventory, "a" * 64) is None


@pytest.mark.parametrize("change", ["partial", "hash", "quant", "placement", "arithmetic", "frontend"])
def test_native_source_cannot_relax_ordinary_integer_arithmetic(change):
    spec = _spec("dtype_cast", "i32", "f32")
    if change == "partial":
        del spec["native_output_dtype"]
    elif change == "hash":
        spec["native_source_abi"]["signature_sha256"] = "c" * 64
    elif change == "quant":
        spec["quant_scheme"] = "selected"
    elif change == "placement":
        spec["generalization"]["eligible"] = True
    elif change == "arithmetic":
        spec["op"] = "add"
    else:
        spec["frontend_op"] = "different"
    with pytest.raises(ValueError):
        native_storage_source_spec(spec)


def test_cast_requires_destination_even_in_the_float_regime():
    spec = {"op": "dtype_cast", "operand_dtype": "f32", "dtype": "fp32"}
    assert corpus_synth._writer_for(spec) is None
    with pytest.raises(ValueError, match="explicit native source/result declaration"):
        source.build_loader_src(spec)
    assert corpus_synth._writer_for({"op": "mul", "operand_dtype": "i32"}) is None
