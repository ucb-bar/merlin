"""A prequantized contraction keeps its declared FP8 decode, numerical route, and bias order."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from corefixtures.row_table import run_rows

from merlin.compiler.quantization.contraction.scaled_fp8 import (
    ContractionNumerics,
    ScaledFP8Spec,
    require_same_numerics,
)
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType


def _source(*, dtype="bf16", numerics=True, bias_shape=(3,)):
    raw = lambda shape, kind: TensorType(shape, kind, "raw-bit-pattern")  # noqa: E731
    types = (raw((2, 64), "i8"), raw((2, 1), "i32"), raw((3, 64), "i8"), raw((), "i32"), raw(bias_shape, dtype))
    inputs = tuple(SemanticNode(key, "input", (), typ, effect="input") for key, typ in zip(("a", "ae", "w", "we", "b"), types))
    attrs = (("semantic_kind", "prequantized_contraction"), ("bias_order", "after_output_cast"),
             ("prov.region_id", "original"))
    if numerics:
        contract = ContractionNumerics("ordered-block32", "bf16-rne-per-instruction",
            ("contract", "restore_scale", "cast_bf16", "source-bias-after-output-cast"), "bf16-rne")
        attrs += (("numerical.contraction", json.dumps(contract.record(), sort_keys=True)),)
    result = SemanticNode("result", "quantized_nt", tuple(n.id for n in inputs), raw((2, 3), dtype), attrs)
    return KernelRequest((*inputs, result), ("result",), ("external",),
                         tuple((n.id, "external") for n in inputs), "test-target", "diagnostic")


def _fp8_decode(directory, patch):
    codes = np.array([[0x38, 0xb8, 0x7e], [8, 1, 0x80]], dtype=np.uint8).view(np.int8)
    got = ScaledFP8Spec((0,)).decode(codes, np.array([2, -1], dtype=np.int32))
    np.testing.assert_array_equal(got, [[4, -4, 1792], [2**-7, 2**-10, -0.]])
    assert np.signbit(got[1, 2])
    np.testing.assert_array_equal(ScaledFP8Spec(()).decode(codes, np.array(0, dtype=np.int32)),
                                  [[1, -1, 448], [2**-6, 2**-9, -0.]])
    with pytest.raises(ValueError):
        ScaledFP8Spec((1,)).decode(codes, np.array([2, -1], dtype=np.int32))
    with pytest.raises(ValueError):
        ScaledFP8Spec((0,)).decode(codes, np.array([2, -1], dtype=np.int8))
    with pytest.raises(ValueError):
        ScaledFP8Spec(()).decode(np.array([127], dtype=np.int8), np.array(0, dtype=np.int32))
    # The zero-flush decoding gives exponent field 0 and the reserved codes the code's signed zero.
    flush = ScaledFP8Spec((), "e4m3fn-zero-flush")
    reserved = np.array([1, 0x81, 0x7f, 0xff, 0x38], dtype=np.uint8).view(np.int8)
    got = flush.decode(reserved, np.array(0, dtype=np.int32))
    np.testing.assert_array_equal(got, [0, 0, 0, 0, 1])
    np.testing.assert_array_equal(np.signbit(got), [False, True, False, True, False])
    assert flush.record()["encoding"] == "e4m3fn-zero-flush" and ScaledFP8Spec(()).record()["encoding"] == "e4m3fn"
    with pytest.raises(ValueError, match="named code decoding"):
        ScaledFP8Spec((), "e4m3")
    # Extreme scales: an exact finite binary64 value or a refusal, never a lossy inspection.
    spec = ScaledFP8Spec(())
    code = np.array([3], dtype=np.int8)  # 3 * 2**-9
    # 3 * 2**-1075 rounds to a nonzero binary64 value: still lossy.
    with pytest.raises(ValueError, match="exact finite binary64"):
        spec.decode(code, np.array(-1066, dtype=np.int32))
    np.testing.assert_array_equal(spec.decode(code, np.array(-1065, dtype=np.int32)), [3 * 2.0**-1074])
    for exponent in (-2**31, 2**31 - 1):
        with pytest.raises(ValueError):
            spec.decode(code, np.array(exponent, dtype=np.int32))
        zero = spec.decode(np.array([-128], dtype=np.int8), np.array(exponent, dtype=np.int32))
        assert np.signbit(zero[0]) and zero[0] == 0


def _numeric_route(directory, patch):
    source = ContractionNumerics('block32-anchor-truncate-wrap-i32', 'bf16-rne-per-instruction',
        ('contract', 'restore_scale', 'cast_bf16', 'bias', 'activation'), 'bf16-rne', 'ordered-select')
    require_same_numerics(source, replace(source))
    for changed in (replace(source, bias_rounding='bf16-chop'), replace(source, activation='sign-clamp'),
                    replace(source, accumulation='exact-dot'), replace(source, stages=('bias', 'contract', 'restore_scale'))):
        with pytest.raises(ValueError):
            require_same_numerics(source, changed)


def _bias_order(directory, patch):
    from merlin.compiler.semantic.prequantized_bias import split_prequantized_bias
    from merlin.compiler.semantic.source_preparation import (
        is_prequantized_bias_add,
        preparation_counts,
        preparation_filename,
        prepare_source,
        verify_source_preparation,
    )

    source = _source()
    prepared, receipt = prepare_source(source)
    assert verify_source_preparation(prepared, receipt).digest() == source.digest()
    assert prepared.outputs == source.outputs and prepared.source_identity == source.source_identity
    assert preparation_counts(receipt)["source_prequantized_bias_splits"] == 1
    assert preparation_filename(receipt) == "source-preparation.json"
    add = prepared.node("result")
    assert is_prequantized_bias_add(prepared, add)
    readout = prepared.node(add.inputs[0])
    assert readout.inputs == ("a", "ae", "w", "we")
    numerical = ContractionNumerics.from_record(json.loads(dict(readout.attrs)["numerical.contraction"]))
    assert numerical.stages == ("contract", "restore_scale", "cast_bf16") and numerical.bias_rounding == "none"
    broadcast = prepared.node(add.inputs[1])
    assert broadcast.index_maps[0].apply((1, 2)) == (2,)
    assert broadcast.index_maps[-1].apply((1, 2)) == (1, 2)
    # The split feeds original BF16 bias into one source RNE addition.
    summed = (np.ones((2, 3), dtype=np.float32) + np.array([3/512, -3/512, 0], dtype=np.float32)).view(np.uint32)
    np.testing.assert_array_equal((summed + 0x7fff + ((summed >> 16) & 1)) >> 16, [[0x3f81, 0x3f7e, 0x3f80]] * 2)
    legacy, _ = split_prequantized_bias(_source(numerics=False))
    assert is_prequantized_bias_add(legacy, legacy.node("result"))
    f32 = _source(dtype="f32", numerics=False)
    assert split_prequantized_bias(f32)[0] == f32


def _bias_forgery(directory, patch):
    from merlin.compiler.semantic.prequantized_bias import (
        is_prequantized_bias_add,
        split_prequantized_bias,
        verify_prequantized_bias,
    )
    from merlin.compiler.semantic.source_preparation import prepare_source

    source = _source()
    prepared, receipt = split_prequantized_bias(source)
    add, broadcast = prepared.node("result"), prepared.node("result.bias")

    def changed(node):
        return replace(prepared, nodes=tuple(node if n.id == node.id else n for n in prepared.nodes))

    chopped = replace(add, attrs=(("numeric_semantics", "bf16-chop"),))
    assert not is_prequantized_bias_add(changed(chopped), chopped)
    shifted = replace(broadcast, index_maps=(IndexMap(2, ((1, 0),), (0,)), broadcast.index_maps[-1]))
    assert not is_prequantized_bias_add(changed(shifted), add)
    with pytest.raises(ValueError):
        verify_prequantized_bias(changed(shifted), receipt)
    forged = copy.deepcopy(receipt)
    forged["changes"][0]["original"]["attrs"]["bias_order"] = "before_output_cast"
    with pytest.raises(ValueError):
        verify_prequantized_bias(prepared, forged)
    original = source.node("result")
    numerical = ContractionNumerics.from_record(json.loads(dict(original.attrs)["numerical.contraction"]))
    for contract in (replace(numerical, bias_rounding="bf16-chop"), replace(numerical, stages=tuple(reversed(numerical.stages))),
                     replace(numerical, activation="ordered-select")):
        attrs = dict(original.attrs)
        attrs["numerical.contraction"] = json.dumps(contract.record())
        with pytest.raises(ValueError):
            split_prequantized_bias(replace(source, nodes=(*source.nodes[:-1], replace(original, attrs=tuple(attrs.items())))))
    with pytest.raises(ValueError):
        ContractionNumerics.from_record({**numerical.record(), "extra": "untrusted"})
    with pytest.raises(ValueError):
        ContractionNumerics.from_record(None)
    stale = replace(prepared.node("result.readout"), attrs=original.attrs)
    assert not is_prequantized_bias_add(changed(stale), add)
    collision = replace(source.nodes[0], id="result.readout")
    with pytest.raises(ValueError, match="collision"):
        split_prequantized_bias(replace(source, nodes=(*source.nodes[:-1], collision,
            replace(original, inputs=("a", "ae", "w", "we", "b"))),
            input_storages=(*source.input_storages, (collision.id, "external"))))
    # A bias that is not one value per output column refuses rather than broadcasting another way.
    with pytest.raises(ValueError, match="column vector after the output cast"):
        prepare_source(_source(numerics=False, bias_shape=(2,)))


def _folded_bias(directory, patch):
    from merlin.compiler.semantic.source_preparation import is_prequantized_bias_add, prepare_source, verify_source_preparation

    type_ = TensorType((1, 32), "bf16", "raw-bit-pattern")
    inputs = tuple(SemanticNode(name, "input", (), TensorType((32,), "bf16", "raw-bit-pattern") if name == "bias" else type_,
                                effect="input") for name in ("x", "xe", "w", "we", "bias"))
    result = SemanticNode("linear", "contraction", tuple(node.id for node in inputs), type_,
                          (("semantic_kind", "prequantized_contraction"), ("bias_order", "after_output_cast")))
    source = KernelRequest(inputs + (result,), ("linear",), ("external",),
                           tuple((node.id, "external") for node in inputs), "test", "diagnostic")
    bias = replace(source.node("bias"), op="constant", effect="constant")
    source = replace(source, nodes=tuple(bias if node.id == bias.id else node for node in source.nodes),
                     input_storages=tuple(row for row in source.input_storages if row[0] != bias.id),
                     constants=(ConstantBinding(bias.id, "bf16-le-raw", "803f" * 32),))
    prepared, receipt = prepare_source(source)
    folded = prepared.node("linear.bias")
    assert folded.effect == "constant"
    assert is_prequantized_bias_add(prepared, prepared.node("linear"))
    assert verify_source_preparation(prepared, receipt).digest() == source.digest()
    binding = next(row for row in prepared.constants if row.node_id == folded.id)
    # KernelRequest authenticates complete, type-correct byte bindings before
    # the structural placement recognizer can consume them.
    for mutate in (lambda: replace(prepared, constants=()),
                   lambda: replace(prepared, constants=(replace(binding, encoding="f32-le-raw"),)),
                   lambda: replace(prepared, constants=(replace(binding, data_hex="803f"),)),
                   lambda: replace(prepared, constants=(replace(binding, data_hex="invalid"),)),
                   lambda: replace(prepared, constants=(binding, binding))):
        with pytest.raises(ValueError):
            mutate()
    bad = replace(prepared, nodes=tuple(replace(node, attrs=(("unchecked", True),)) if node.id == folded.id else node
                                        for node in prepared.nodes))
    assert not is_prequantized_bias_add(bad, bad.node("linear"))


def test_prequantized_contraction_keeps_declared_numerics(tmp_path, monkeypatch):
    run_rows(tmp_path, monkeypatch, {
        "fp8_decode": _fp8_decode,
        "numeric_route": _numeric_route,
        "bias_order": _bias_order,
        "bias_forgery": _bias_forgery,
        "folded_bias": _folded_bias,
    })
