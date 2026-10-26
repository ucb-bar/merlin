"""Runtime f32 readout premises are source structural and never sampled."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import yaml

from merlin.common import mlir_query as mq
from merlin.common.paths import repo_root
from merlin.targetgen import model_coverage
from merlin.targetgen.composition_capsules import _Source
from merlin.targetgen.composition_facts import derive, derive_runtime_readout_scale
from merlin.verify.lean_contract import document_digest
from merlin.xdsl_dialects.lowering import output_fusion as OF
from merlin.xdsl_dialects.lowering.runtime_readout import match_closed_readout

FACTS_FILE = Path(os.environ.get("RT_FACTS") or os.environ.get("MERLIN_RTL_FACTS") or "")
ROOT = repo_root()
TRANSPORT = dict(
    schema="merlin.runtime_readout_scale.v1",
    dtype="f32",
    granularity="per_command",
    operand="rs2",
    bundle="ConfigMvoutRs2",
    field="acc_scale",
    offset=32,
    width=32,
    encoding="ieee754_binary32_bits",
)
PATH = {
    "schema": "output_path_v1",
    "accumulator": {"dtype": "i32"},
    "accumulator_stages": [{"kind": "preload", "dtype": "i32"}],
    "readout": [
        {"kind": "activation", "function": "relu", "domain": "accumulator"},
        {"kind": "convert", "to": "f32", "rounding": "half_even"},
        {
            "kind": "scale",
            "dtype": "f32",
            "rounding": "half_even",
            "granularity": "per_command",
            "columns_per_scale": 16,
        },
        {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
    ],
    "exits": {"accumulator": True, "requant": True},
}


def source(*, relu=False, split=False, constant=False):
    s = _Source()
    a = s.input("lhs", [3, 7], "i8")
    w = s.input("w", [7, 17], "i8")
    s.input("unused", [1], "f32")
    scale = s.input("scale", [1], "f32")
    acc = s.matmul("acc", a, w, 3, 7, 17)
    if not split:
        out = s.readout("q", acc, 3, 17, [0.25], 16, relu=relu, runtime_scale=None if constant else scale)
    else:
        out = s.generic("f", [acc], [3, 17], "f32", ["%v = arith.sitofp %x0 : i32 to f32", "linalg.yield %v : f32"])
        if relu:
            out = s.generic(
                "act",
                [out],
                [3, 17],
                "f32",
                ["%z = arith.constant 0.0 : f32", "%v = arith.maximumf %x0, %z : f32", "linalg.yield %v : f32"],
            )
        out = s.generic(
            "product",
            [out, scale],
            [3, 17],
            "f32",
            ["%v = arith.mulf %x0, %x1 : f32", "linalg.yield %v : f32"],
            maps=["affine_map<(d0,d1)->(d0,d1)>", "affine_map<(d0,d1)->(0)>", "affine_map<(d0,d1)->(d0,d1)>"],
        )
        s.lines += [
            "%qs = arith.constant dense<1.0> : tensor<f32>",
            "%qz = arith.constant dense<0> : tensor<i64>",
            f'%q = "quant_ext.quantize_per_tensor"(%{out}, %qs, %qz) '
            '<{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> '
            ": (tensor<3x17xf32>, tensor<f32>, tensor<i64>) -> tensor<3x17xi8>",
        ]
        s.types["q"] = "tensor<3x17xi8>"
        out = "q"
    args = ", ".join(f"%{row['name']}: {s.types[row['name']]}" for row in s.inputs)
    return "\n".join(
        [
            f"builtin.module {{ func.func @run({args}) -> tensor<3x17xi8> {{",
            *s.lines,
            f"func.return %{out} : tensor<3x17xi8>",
            "}}",
        ]
    )


def read(text, values=None, path=None, transport=TRANSPORT):
    module = mq.parse(text)
    root = next(
        op
        for op in module.walk()
        if op.attributes.get("prov.composition_stage") and str(op.attributes["prov.composition_stage"]) == '"acc"'
    )
    values = values or OF.Values()
    chain = OF.source_chain(root, values)
    result = OF.decide(
        chain,
        path or PATH,
        values,
        model_coverage._contraction_extents(module),
        readout_strategy="f32_product_rne",
        runtime_scale_facts=transport,
    )
    return module, chain, result


class NoScaleReads(OF.Values):
    def array(self, value):
        from xdsl.ir import BlockArgument

        if isinstance(value, BlockArgument) and str(value.type) == "tensor<1xf32>":
            raise AssertionError("runtime source value was sampled")
        return super().array(value)


@pytest.mark.parametrize("relu", [False, True])
def test_closed_generic_runtime_scale_is_source_bound(relu):
    module, chain, decision = read(source(relu=relu), NoScaleReads())
    assert decision.cut == "requant", decision.refusals
    assert match_closed_readout(chain.root).ops == chain.ops
    assert len(chain.ops) == len(set(chain.ops)) == 1
    assert decision.members == chain.ops
    assert chain.stages[0].kind == "runtime_readout" and chain.stages[0].numbers is None
    proof = decision.record()["proof"]
    assert proof["scale_operand"]["source"] == {"func_arg": 3, "element_index": 0}
    assert proof["runtime_scale_facts_sha256"] == document_digest(TRANSPORT)
    assert proof["readout_stages"] == PATH["readout"]
    assert proof["preload"] == PATH["accumulator_stages"][0]
    assert proof["accumulator_bound"] == [-114688, 114688]
    assert not {"multiplier_bits", "device_scale", "columns_per_scale"} & proof.keys()
    json.dumps(decision.record())
    assert decision.stages == ["convert"] + (["relu"] if relu else []) + ["scale", "quantize"]


@pytest.mark.parametrize("relu", [False, True])
def test_split_runtime_scale_chain_has_identical_numeric_boundary(relu):
    _, composite, expected = read(source(relu=relu))
    _, split, actual = read(source(relu=relu, split=True), NoScaleReads())
    assert actual.cut == "requant", actual.refusals
    assert actual.record() == expected.record()
    assert len(split.ops) == 3 + int(relu)
    assert actual.members == split.ops and len(set(split.ops)) == len(split.ops)
    assert next(s for s in split.stages if s.kind == "scale").numbers is None


def mutate(text, mutation):
    if mutation in ("f16", "f64", "shape2", "rank0"):
        return text.replace(
            "tensor<1xf32>",
            {"f16": "tensor<1xf16>", "f64": "tensor<1xf64>", "shape2": "tensor<2xf32>", "rank0": "tensor<f32>"}[
                mutation
            ],
        )
    if mutation in ("row", "column"):
        return text.replace("(d0,d1)->(0)", f"(d0,d1)->({'d0' if mutation == 'row' else 'd1'})")
    if mutation == "extra_mul":
        return text.replace("%r = math.roundeven %p", "%extra = arith.mulf %p, %x1 : f32\n%r = math.roundeven %extra")
    if mutation == "division":
        return text.replace("arith.mulf", "arith.divf")
    if mutation == "post_relu":
        return (
            text.replace("%act = arith.maximumf %f, %zero : f32", "%act = arith.maximumf %p, %zero : f32")
            .replace("arith.mulf %act,", "arith.mulf %f,")
            .replace("math.roundeven %p", "math.roundeven %act")
        )
    if mutation == "nonunit":
        return text.replace("dense<1.0>", "dense<2.0>")
    if mutation == "runtime_denominator":
        return text.replace("%product, %qs, %qz", "%product, %scale, %qz").replace(
            "tensor<3x17xf32>, tensor<f32>, tensor<i64>", "tensor<3x17xf32>, tensor<1xf32>, tensor<i64>"
        )
    if mutation == "observer":
        return text.replace("func.return %q", '"test.observe"(%f) : (tensor<3x17xf32>) -> ()\nfunc.return %q')
    if mutation == "clamp":
        return text.replace("-128.0", "-127.0")
    if mutation == "rounding":
        return text.replace("math.roundeven", "math.floor")
    if mutation == "fastmath":
        return text.replace("arith.mulf %act, %x1 : f32", "arith.mulf %act, %x1 fastmath<fast> : f32")
    if mutation == "initializer_use":
        return text.replace("arith.fptosi %h", "arith.fptosi %h").replace(
            "linalg.yield %v : i8", "linalg.yield %dest : i8"
        )
    raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "f16",
        "f64",
        "shape2",
        "rank0",
        "row",
        "column",
        "extra_mul",
        "division",
        "post_relu",
        "nonunit",
        "runtime_denominator",
        "observer",
        "clamp",
        "rounding",
        "fastmath",
        "initializer_use",
    ],
)
def test_runtime_scale_rejects_wrong_shape_dtype_map_or_extra_math(mutation):
    split = mutation in ("nonunit", "runtime_denominator", "observer")
    _, chain, decision = read(mutate(source(relu=not split, split=split), mutation), NoScaleReads())
    assert decision.cut != "requant", (mutation, chain, decision)


@pytest.mark.parametrize("mutation", ["missing", "bool", "extra", "width", "offset", "operand"])
def test_runtime_proof_requires_exact_transport_record(mutation):
    facts = copy.deepcopy(TRANSPORT)
    if mutation == "missing":
        facts = None
    elif mutation == "bool":
        facts["width"] = True
    elif mutation == "extra":
        facts["trusted"] = True
    else:
        facts[mutation] = {"width": 16, "offset": -1, "operand": "rs1"}[mutation]
    assert read(source(), transport=facts)[2].cut == "bare"


def test_runtime_proof_takes_the_slot_position_from_the_derived_record():
    # The record's offset is the layout's; a relocated whole f32 field still proves.
    for offset in (0, 32):
        _, _, decision = read(source(), transport=dict(TRANSPORT, offset=offset))
        assert decision.cut == "requant", (offset, decision.refusals)


def test_runtime_transport_uses_selected_role_layout_and_abi():
    if not FACTS_FILE.is_file():
        pytest.skip("select actual target facts with RT_FACTS or MERLIN_RTL_FACTS")
    facts = json.loads(FACTS_FILE.read_text())
    contract = yaml.safe_load((ROOT / "examples/gemmini/support/contracts/target_contract.yaml").read_text())
    record = derive_runtime_readout_scale(facts=facts, contract=contract, output_path=PATH)
    assert record["status"] == "derived", record
    assert record["value"] == TRANSPORT
    assert len(record["evidence"]) == 4
    published = derive(facts=facts, contract=contract, software_spec={}, output_path=PATH)
    assert published["facts"]["runtime_readout_scale"] == TRANSPORT
    assert published["evidence"]["runtime_readout_scale"] == record["evidence"]
    for mutation in ("role", "layout", "abi", "evidence", "path", "bool", "slot", "outside", "register"):
        f, c, p = copy.deepcopy((facts, contract, PATH))
        layouts = next(i for i in f["facts"]["interfaces"] if i["name"] == "register_bundle_layouts")
        if mutation == "role":
            c["rocc_operand_roles"]["classes"]["CONFIG_ST"]["fields"].append(
                dict(name="acc_scale", operand="rs1", bundle="ConfigMvoutRs2", field="acc_scale")
            )
        elif mutation == "layout":
            layouts["bundles"]["ConfigMvoutRs2"]["fields"]["acc_scale"]["offset"] = 16
        elif mutation == "abi":
            c["readout_semantics"]["scalar_abi"]["scale_dtype"] = "f16"
        elif mutation == "evidence":
            del layouts["evidence"]
        elif mutation == "path":
            p["readout"] = []
        elif mutation == "outside":
            layouts["bundles"]["ConfigMvoutRs2"]["fields"]["acc_scale"]["offset"] = 48
        elif mutation == "register":
            del layouts["bundles"]["ConfigMvoutRs2"]["width"]
        else:
            layouts["bundles"]["ConfigMvoutRs2"]["fields"]["acc_scale"]["slot_width"] = (
                True if mutation == "bool" else 64
            )
        assert derive_runtime_readout_scale(facts=f, contract=c, output_path=p)["status"] == "unknown", mutation
    # The slot position is read from the layout, not assumed: swap the two fields.
    f = copy.deepcopy(facts)
    fields = next(i for i in f["facts"]["interfaces"] if i["name"] == "register_bundle_layouts")["bundles"][
        "ConfigMvoutRs2"
    ]["fields"]
    fields["acc_scale"]["offset"], fields["stride"]["offset"] = 0, 32
    moved = derive_runtime_readout_scale(facts=f, contract=contract, output_path=PATH)
    assert moved["status"] == "derived" and moved["value"] == dict(TRANSPORT, offset=0)


def test_constant_composite_keeps_multiplier_bits():
    _, chain, decision = read(source(constant=True))
    assert decision.cut == "requant", decision.refusals
    assert decision.proof["multiplier_bits"] == [0x3E800000] * 17
    assert "scale_operand" not in decision.proof
    assert len(decision.members) == 1


def test_runtime_bound_and_zero_seed_are_still_required():
    assert read(source().replace("constant 0 : i32", "constant 1 : i32"))[2].cut == "bare"
    _, chain, _ = read(source())
    assert (
        OF.decide(chain, PATH, OF.Values(), {}, readout_strategy="f32_product_rne", runtime_scale_facts=TRANSPORT).cut
        == "bare"
    )


def test_integer_relu_and_float_relu_have_identical_runtime_proof():
    text = source(relu=True, split=True)
    start = text.index("    %f_empty")
    stop = text.index("    %product_empty", start)
    prefix = text[start:stop]
    integer = prefix[prefix.index("    %act_empty") :].replace("xf32", "xi32").replace("%f :", "%acc :")
    integer = integer.replace("%x0: f32", "%x0: i32").replace("%dest: f32", "%dest: i32")
    integer = integer.replace("0.0 : f32", "0 : i32").replace("arith.maximumf", "arith.maxsi")
    integer = integer.replace("%x0, %z : f32", "%x0, %z : i32").replace(
        "linalg.yield %v : f32", "linalg.yield %v : i32"
    )
    convert = prefix[: prefix.index("    %act_empty")].replace("ins(%acc :", "ins(%act :")
    text = text[:start] + integer + convert + text[stop:]
    text = text.replace("ins(%act, %scale", "ins(%f, %scale")
    _, _, expected = read(source(relu=True, split=True))
    _, chain, actual = read(text)
    assert actual.cut == "requant", actual.refusals
    assert actual.record() == expected.record()
    assert [s.kind for s in chain.stages] == ["relu", "convert", "scale", "quantize"]


@pytest.mark.parametrize("commute", ["product", "relu", "low", "high"])
def test_closed_readout_accepts_commuted_equal_operations(commute):
    replacements = {
        "product": ("%act, %x1", "%x1, %act"),
        "relu": ("%f, %zero", "%zero, %f"),
        "low": ("%r, %lo", "%lo, %r"),
        "high": ("%l, %hi", "%hi, %l"),
    }
    a, b = replacements[commute]
    assert read(source(relu=True).replace(a, b))[2].record() == read(source(relu=True))[2].record()


@pytest.mark.parametrize("mutation", ["dtype", "rounding", "clamp", "denominator_dtype"])
def test_split_quantizer_requires_exact_f32_rne_contract(mutation):
    text = source(split=True)
    if mutation == "dtype":
        text = text.replace('output_dtype = "int8"', 'output_dtype = "int16"')
    elif mutation == "rounding":
        text = text.replace('output_dtype = "int8"', 'output_dtype = "int8", rounding = "floor"')
    elif mutation == "clamp":
        text = text.replace("quant_min = -128", "quant_min = -127")
    else:
        text = text.replace("tensor<f32>", "tensor<f16>")
    assert read(text, NoScaleReads())[2].cut != "requant"
