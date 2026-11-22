"""Standalone first-milestone admission tests; no native compiler required."""
import unittest
from dataclasses import replace
import numpy as np
from merlin.compiler.quantization.contraction.scaled_fp8 import (
    ScaledFP8Spec, ContractionNumerics, require_same_numerics,
)

class ContractAdmission(unittest.TestCase):
    def test_nonzero_codes_scale_and_explicit_broadcast(self):
        with self.subTest(domain="extreme-scales"):
            self._extreme_scales_refuse_lossy_inspection()
        codes = np.array([[0x38, 0xb8, 0x7e], [8, 1, 0x80]], dtype=np.uint8).view(np.int8)
        got = ScaledFP8Spec((0,)).decode(codes, np.array([2, -1], dtype=np.int32))
        np.testing.assert_array_equal(got, [[4, -4, 1792], [2**-7, 2**-10, -0.]])
        self.assertTrue(np.signbit(got[1,2]))
        np.testing.assert_array_equal(ScaledFP8Spec(()).decode(codes, np.array(0, dtype=np.int32)),
                                     [[1,-1,448],[2**-6,2**-9,-0.]])
        with self.assertRaises(ValueError):
            ScaledFP8Spec((1,)).decode(codes, np.array([2,-1], dtype=np.int32))
        with self.assertRaises(ValueError):
            ScaledFP8Spec((0,)).decode(codes, np.array([2,-1], dtype=np.int8))
        with self.assertRaises(ValueError):
            ScaledFP8Spec(()).decode(np.array([127], dtype=np.int8), np.array(0, dtype=np.int32))

    def _extreme_scales_refuse_lossy_inspection(self):
        spec = ScaledFP8Spec(())
        code = np.array([3], dtype=np.int8)  # 3 * 2**-9
        # 3 * 2**-1075 rounds to a nonzero binary64 value: still lossy.
        with self.assertRaisesRegex(ValueError, "exact finite binary64"):
            spec.decode(code, np.array(-1066, dtype=np.int32))
        np.testing.assert_array_equal(
            spec.decode(code, np.array(-1065, dtype=np.int32)), [3 * 2.0**-1074])
        for exponent in (-2**31, 2**31 - 1):
            with self.assertRaises(ValueError):
                spec.decode(code, np.array(exponent, dtype=np.int32))
            zero = spec.decode(np.array([-128], dtype=np.int8), np.array(exponent, dtype=np.int32))
            self.assertTrue(np.signbit(zero[0]))
            self.assertEqual(zero[0], 0)

    def test_numerical_route_refuses_silent_fusion(self):
        source = ContractionNumerics('block32-anchor-truncate-wrap-i32', 'bf16-rne-per-instruction',
            ('contract','restore_scale','cast_bf16','bias','activation'), 'bf16-rne', 'ordered-select')
        require_same_numerics(source, replace(source))
        with self.assertRaises(ValueError):
            require_same_numerics(source, replace(source, bias_rounding='bf16-chop'))
        with self.assertRaises(ValueError):
            require_same_numerics(source, replace(source, activation='sign-clamp'))
        with self.assertRaises(ValueError):
            require_same_numerics(source, replace(source, accumulation='exact-dot'))
        with self.assertRaises(ValueError):
            require_same_numerics(source, replace(source, stages=('bias','contract','restore_scale')))
        # Nonzero BF16 witness: 1 + 3/512 = 1.005859375, above midpoint.
        value = np.array([1 + 3/512], dtype=np.float32).view(np.uint32)
        chop = value >> 16
        rne = (value + 0x7fff + ((value >> 16) & 1)) >> 16
        self.assertEqual(int(chop[0]), 0x3f80)
        self.assertEqual(int(rne[0]), 0x3f81)

class PrequantizedBiasPreparation(unittest.TestCase):
    def source(self, *, dtype="bf16", numerics=True):
        import json
        from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType
        raw = lambda shape, kind: TensorType(shape, kind, "raw-bit-pattern")
        types = (raw((2, 64), "i8"), raw((2, 1), "i32"),
                 raw((3, 64), "i8"), raw((), "i32"), raw((3,), dtype))
        inputs = tuple(SemanticNode(key, "input", (), typ, effect="input")
                       for key, typ in zip(("a", "ae", "w", "we", "b"), types))
        attrs = (("semantic_kind", "prequantized_contraction"),
                 ("bias_order", "after_output_cast"), ("prov.region_id", "original"))
        if numerics:
            contract = ContractionNumerics("ordered-block32", "bf16-rne-per-instruction",
                ("contract", "restore_scale", "cast_bf16", "source-bias-after-output-cast"), "bf16-rne")
            attrs += (("numerical.contraction", json.dumps(contract.record(), sort_keys=True)),)
        result = SemanticNode("result", "quantized_nt", tuple(n.id for n in inputs),
                              raw((2, 3), dtype), attrs)
        return KernelRequest((*inputs, result), ("result",), ("external",),
                             tuple((n.id, "external") for n in inputs), "test-target", "diagnostic")

    def test_bias_preparation_preserves_source_and_column_rne_order(self):
        import json
        from merlin.compiler.semantic.source_preparation import (
            prepare_source, verify_source_preparation, preparation_counts, preparation_filename,
            is_prequantized_bias_add,
        )
        from merlin.compiler.semantic.prequantized_bias import split_prequantized_bias
        source = self.source()
        prepared, receipt = prepare_source(source)
        self.assertEqual(verify_source_preparation(prepared, receipt).digest(), source.digest())
        self.assertEqual(prepared.outputs, source.outputs)
        self.assertEqual(prepared.source_identity, source.source_identity)
        self.assertEqual(preparation_counts(receipt)["source_prequantized_bias_splits"], 1)
        self.assertEqual(preparation_filename(receipt), "source-preparation.json")
        add = prepared.node("result")
        self.assertTrue(is_prequantized_bias_add(prepared, add))
        readout = prepared.node(add.inputs[0])
        self.assertEqual(readout.inputs, ("a", "ae", "w", "we"))
        numerical = ContractionNumerics.from_record(json.loads(dict(readout.attrs)["numerical.contraction"]))
        self.assertEqual(numerical.stages, ("contract", "restore_scale", "cast_bf16"))
        self.assertEqual(numerical.bias_rounding, "none")
        broadcast = prepared.node(add.inputs[1])
        self.assertEqual(broadcast.index_maps[0].apply((1, 2)), (2,))
        self.assertEqual(broadcast.index_maps[-1].apply((1, 2)), (1, 2))
        # The split feeds original BF16 bias into one source RNE addition.
        readout_values = np.ones((2, 3), dtype=np.float32)
        bias_values = np.array([3/512, -3/512, 0], dtype=np.float32)
        summed = (readout_values + bias_values).view(np.uint32)
        rounded = (summed + 0x7fff + ((summed >> 16) & 1)) >> 16
        np.testing.assert_array_equal(rounded, [[0x3f81, 0x3f7e, 0x3f80]] * 2)
        legacy, _ = split_prequantized_bias(self.source(numerics=False))
        self.assertTrue(is_prequantized_bias_add(legacy, legacy.node("result")))
        f32 = self.source(dtype="f32", numerics=False)
        self.assertEqual(split_prequantized_bias(f32)[0], f32)

    def test_bias_preparation_refuses_forged_structure_and_numerics(self):
        import copy
        import json
        from merlin.compiler.semantic.model import IndexMap
        from merlin.compiler.semantic.prequantized_bias import (
            split_prequantized_bias, verify_prequantized_bias, is_prequantized_bias_add,
        )
        source = self.source()
        prepared, receipt = split_prequantized_bias(source)
        add, broadcast = prepared.node("result"), prepared.node("result.bias")
        def changed(node):
            return replace(prepared, nodes=tuple(node if n.id == node.id else n for n in prepared.nodes))
        chopped = replace(add, attrs=(("numeric_semantics", "bf16-chop"),))
        self.assertFalse(is_prequantized_bias_add(changed(chopped), chopped))
        shifted = replace(broadcast, index_maps=(IndexMap(2, ((1, 0),), (0,)), broadcast.index_maps[-1]))
        self.assertFalse(is_prequantized_bias_add(changed(shifted), add))
        with self.assertRaises(ValueError):
            verify_prequantized_bias(changed(shifted), receipt)
        forged = copy.deepcopy(receipt)
        forged["changes"][0]["original"]["attrs"]["bias_order"] = "before_output_cast"
        with self.assertRaises(ValueError):
            verify_prequantized_bias(prepared, forged)
        original = source.node("result")
        numerical = ContractionNumerics.from_record(json.loads(dict(original.attrs)["numerical.contraction"]))
        for contract in (replace(numerical, bias_rounding="bf16-chop"),
                         replace(numerical, stages=tuple(reversed(numerical.stages))),
                         replace(numerical, activation="ordered-select")):
            with self.subTest(contract=contract), self.assertRaises(ValueError):
                attrs = dict(original.attrs)
                attrs["numerical.contraction"] = json.dumps(contract.record())
                bad = replace(original, attrs=tuple(attrs.items()))
                split_prequantized_bias(replace(source, nodes=(*source.nodes[:-1], bad)))
        with self.assertRaises(ValueError):
            ContractionNumerics.from_record({**numerical.record(), "extra": "untrusted"})
        with self.assertRaises(ValueError):
            ContractionNumerics.from_record(None)
        stale = replace(prepared.node("result.readout"), attrs=original.attrs)
        self.assertFalse(is_prequantized_bias_add(changed(stale), add))
        collision = replace(source.nodes[0], id="result.readout")
        with self.assertRaisesRegex(ValueError, "collision"):
            split_prequantized_bias(replace(source, nodes=(*source.nodes[:-1], collision,
                replace(original, inputs=("a", "ae", "w", "we", "b"))),
                input_storages=(*source.input_storages, (collision.id, "external"))))

if __name__ == '__main__':
    unittest.main()
