"""Check the exact bias boundary carried across the package refactor."""

from copy import deepcopy
from dataclasses import replace

import unittest

from merlin.compiler.lower.c.float_arithmetic import emit_float_arithmetic_c
from merlin.compiler.lower.c.pointwise_host import emit_host_pointwise_c
from merlin.compiler.partition.semantic_regions import extract_semantic_region
from merlin.compiler.semantic.model import ConstantBinding, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.compiler.semantic.source_preparation import (
    is_prequantized_bias_add, prepare_source, verify_source_preparation,
)


def _biased_contraction():
    type_ = TensorType((1, 32), "bf16", "raw-bit-pattern")
    inputs = tuple(SemanticNode(
        name, "input", (),
        TensorType((32,), "bf16", "raw-bit-pattern") if name == "bias" else type_,
        effect="input",
    ) for name in ("x", "xe", "w", "we", "bias"))
    result = SemanticNode(
        "linear", "contraction", tuple(node.id for node in inputs), type_,
        (("semantic_kind", "prequantized_contraction"), ("bias_order", "after_output_cast")),
    )
    return KernelRequest(
        inputs + (result,), ("linear",), ("external",),
        tuple((node.id, "external") for node in inputs), "test", "diagnostic",
    )


class PrequantizedBiasPortTests(unittest.TestCase):
    def test_bias_preparation_and_host_epilogue(self):
        for check in (
            self._check_bias_is_added_after_readout_and_replays,
            self._check_bias_replay_rejects_changed_original,
            self._check_folded_bias_requires_complete_constant_binding,
            self._check_host_relu_refuses_sign_clamp_declaration,
        ):
            with self.subTest(check=check.__name__):
                check()

    def _check_bias_is_added_after_readout_and_replays(self):
        source = _biased_contraction()
        prepared, receipt = prepare_source(source)
        assert prepared.outputs == source.outputs
        assert prepared.node("linear.readout").inputs == ("x", "xe", "w", "we")
        assert is_prequantized_bias_add(prepared, prepared.node("linear"))
        assert verify_source_preparation(prepared, receipt).digest() == source.digest()
        region = extract_semantic_region(prepared, ("linear",))
        record, files = emit_float_arithmetic_c(region, "bias_add")
        assert record["numeric_semantics"] == "ieee754-bf16-add-rne-v1"
        assert record["inputs"] == ["linear.readout", "linear.bias"]
        assert files


    def _check_bias_replay_rejects_changed_original(self):
        prepared, receipt = prepare_source(_biased_contraction())
        changed = deepcopy(receipt)
        changed["bias"]["changes"][0]["original"]["inputs"][-1] = "w"
        with self.assertRaises(ValueError):
            verify_source_preparation(prepared, changed)


    def _check_folded_bias_requires_complete_constant_binding(self):
        source = _biased_contraction()
        bias = replace(source.node("bias"), op="constant", effect="constant")
        source = replace(
            source, nodes=tuple(bias if node.id == bias.id else node for node in source.nodes),
            input_storages=tuple(row for row in source.input_storages if row[0] != bias.id),
            constants=(ConstantBinding(bias.id, "bf16-le-raw", "803f" * 32),),
        )
        prepared, receipt = prepare_source(source)
        folded = prepared.node("linear.bias")
        assert folded.effect == "constant"
        assert is_prequantized_bias_add(prepared, prepared.node("linear"))
        assert verify_source_preparation(prepared, receipt).digest() == source.digest()
        binding = next(row for row in prepared.constants if row.node_id == folded.id)
        for label, mutate in (
            ("missing binding", lambda: replace(prepared, constants=())),
            ("wrong encoding", lambda: replace(prepared, constants=(replace(binding, encoding="f32-le-raw"),))),
            ("short payload", lambda: replace(prepared, constants=(replace(binding, data_hex="803f"),))),
            ("invalid hex", lambda: replace(prepared, constants=(replace(binding, data_hex="invalid"),))),
            ("duplicate binding", lambda: replace(prepared, constants=(binding, binding))),
        ):
            # KernelRequest authenticates complete, type-correct byte bindings
            # before the structural placement recognizer can consume them.
            with self.subTest(mutation=label), self.assertRaises(ValueError):
                mutate()
        bad = replace(prepared, nodes=tuple(
            replace(node, attrs=(("unchecked", True),)) if node.id == folded.id else node
            for node in prepared.nodes))
        assert not is_prequantized_bias_add(bad, bad.node("linear"))


    def _check_host_relu_refuses_sign_clamp_declaration(self):
        type_ = TensorType((1, 32), "bf16", "raw-bit-pattern")
        operand = SemanticNode("x", "input", (), type_, effect="input")
        result = SemanticNode(
            "y", "relu", ("x",), type_,
            (("input_domain", "finite-normal-or-zero-bf16"), ("numeric_semantics", "bf16-sign-clamp-v1")),
            index_maps=pointwise_identity_maps(2, 1),
        )
        source = KernelRequest((operand, result), ("y",), ("external",), (("x", "external"),), "test", "diagnostic")
        with self.assertRaisesRegex(ValueError, "indexing, or policy differs"):
            emit_host_pointwise_c(source, "relu")


if __name__ == "__main__":
    unittest.main()
