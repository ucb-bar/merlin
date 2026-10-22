"""Direct unittest entry point for the explicit ResNet development recipe."""

import importlib.util
import sys
import unittest

import numpy as np
from merlin.common.paths import repo_root


MODULE_PATH = repo_root() / "examples/gemmini/comparisons/tvm/resnet_quantized.py"
sys.path.insert(0, str(MODULE_PATH.parent))
spec = importlib.util.spec_from_file_location("resnet_quantized", MODULE_PATH)
recipe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recipe
spec.loader.exec_module(recipe)


class IntegerPolicies(unittest.TestCase):
    def test_top1_gate_exact_two_in_one_hundred_boundary(self):
        import verify_resnet
        accuracy = {"samples": 100, "torch": {"top1": {"correct": 89}}, "relax": {"top1": {"correct": 87}}}
        self.assertGreater(89 / 100 - 87 / 100, 0.02)
        self.assertEqual(verify_resnet.top1_rate_drop(accuracy), 0.02)
        self.assertLessEqual(verify_resnet.top1_rate_drop(accuracy), 0.02)
        accuracy["relax"]["top1"]["correct"] = 86
        self.assertGreater(verify_resnet.top1_rate_drop(accuracy), 0.02)
        accuracy["relax"]["top1"]["correct"] = 90
        self.assertEqual(verify_resnet.top1_rate_drop(accuracy), -0.01)

    def test_joint_residual_cancellation_signed_ties_and_independent_fraction_oracle(self):
        from fractions import Fraction
        left = np.array([500, -500, 1, -1, recipe.LIMIT, -recipe.LIMIT], dtype=np.int32)
        right = np.array([-497, 497, 1, -1, -recipe.LIMIT, recipe.LIMIT], dtype=np.int32)
        np.testing.assert_array_equal(recipe.residual_add(left, right, 0, 0, 0, recipe.LIMIT, recipe.LIMIT), [3, -3, 2, -2, 0, 0])
        self.assertEqual(int(recipe.saturate(left)[0]) + int(recipe.saturate(right)[0]), 0)
        for le, re, oe in ((-1, -1, 0), (-3, -1, 0), (30, 30, 0), (0, 0, 30)):
            with self.subTest(exponents=(le, re, oe)):
                expected = []
                for l, r in zip(left, right):
                    value = Fraction(int(l)) * Fraction(2) ** (le - oe) + Fraction(int(r)) * Fraction(2) ** (re - oe)
                    magnitude = abs(value)
                    rounded = (2 * magnitude.numerator + magnitude.denominator) // (2 * magnitude.denominator)
                    expected.append(max(-127, min(127, -rounded if value < 0 else rounded)))
                np.testing.assert_array_equal(recipe.residual_add(left, right, le, re, oe, recipe.LIMIT, recipe.LIMIT), expected)
        a, b, den, bound = recipe.residual_add_factors(30, 30, 0, recipe.LIMIT, recipe.LIMIT)
        self.assertEqual(int(bound), 2 * recipe.LIMIT * 2**30)
        self.assertLess(int(bound), np.iinfo(np.int64).max)

    def test_joint_residual_rejects_unsafe_scales_bounds_and_operands(self):
        for args in ((31, 0, 0, 127, 127), (-31, 0, 0, 127, 127), (0, 0, 31, 127, 127),
                     (0, 0, 0, recipe.LIMIT + 1, 127), (0, 0, 0, -1, 127),
                     (np.iinfo(np.int64).min, 0, 0, 127, 127), (0.5, 0, 0, 127, 127)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                recipe.residual_add_factors(*args)
        for left in (np.array([-128], dtype=np.int8), np.array([np.iinfo(np.int32).min], dtype=np.int32), np.array([1], dtype=np.int64)):
            with self.assertRaises(ValueError):
                recipe.residual_add(left, np.array([0], dtype=np.int8), 0, 0, 0, 127, 127)

    def test_accumulator_chain_proof_refuses_fanout_and_preadd_relu(self):
        import copy
        shape = (1, 4, 2, 2)
        sites = [recipe.Site('conv', 'conv2d', ('image',), shape, 0, {}),
                 recipe.Site('bn', 'identity', ('conv',), shape, 0, {}),
                 recipe.Site('sum', 'add', ('bn', 'skip'), shape, 0, {}),
                 recipe.Site('relu', 'relu', ('sum',), shape, 0, {})]
        self.assertEqual(recipe.residual_accumulator_targets(sites), {'conv': {'add': 'sum', 'identities': ['bn'], 'relu': 'relu', 'add_identities': []}})
        self.assertEqual(recipe.residual_accumulator_targets(sites + [recipe.Site('side', 'identity', ('conv',), shape, 0, {})]), {})
        self.assertEqual(recipe.residual_accumulator_targets(sites + [recipe.Site('side', 'identity', ('sum',), shape, 0, {})]), {})
        altered = copy.deepcopy(sites)
        altered[1].kind = 'relu'
        self.assertEqual(recipe.residual_accumulator_targets(altered), {})

    def test_activation_mse_accounts_for_signed_clipping_and_ties(self):
        values = np.array([-100.0, -1.5, -0.5, 0.5, 1.5, 100.0], dtype=np.float32)
        count, errors = recipe.activation_scale_errors(values, (0, -1))
        self.assertEqual(count, 6)
        # Independent scalar oracle includes signed ties and both saturated tails.
        import math
        expected = []
        for exponent in (0, -1):
            scale = 2.0 ** exponent
            total = 0.0
            for value in values:
                rounded = math.copysign(math.floor(abs(float(value) / scale) + 0.5), value)
                dequantized = max(-127, min(127, rounded)) * scale
                total += (dequantized - float(value)) ** 2
            expected.append(total)
        self.assertEqual(errors, expected)
        self.assertLess(errors[0], errors[1])
        count, errors = recipe.activation_scale_errors(np.arange(10, dtype=np.float32), (0,), sample_limit=3)
        self.assertEqual(count, 3)
        self.assertEqual(errors, [0.0])
        for values in (np.zeros(0, dtype=np.float32), np.array([np.nan], dtype=np.float32), np.array([1.0], dtype=np.float64)):
            with self.assertRaises(ValueError):
                recipe.activation_scale_errors(values, (0,))

    def test_signed_rounding_and_shifts(self):
        values = np.arange(-9, 10, dtype=np.int64)
        np.testing.assert_array_equal(recipe.round_divide(values, 2), [-5, -4, -4, -3, -3, -2, -2, -1, -1, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
        np.testing.assert_array_equal(recipe.rescale(values, -1), recipe.round_divide(values, 2))
        np.testing.assert_array_equal(recipe.rescale(values, 2), values * 4)
        with self.assertRaises(ValueError):
            recipe.rescale(values, 31)
        with self.assertRaises(ValueError):
            recipe.round_divide(np.array([np.iinfo(np.int64).min]), 2)
        with self.assertRaises(ValueError):
            recipe.rescale(np.array([np.iinfo(np.int64).min]), 1)
        with self.assertRaises(ValueError):
            recipe.round_divide(values, 1.5)
        with self.assertRaises(ValueError):
            recipe.rescale(values.astype(np.float64), 1)
        np.testing.assert_array_equal(recipe.round_divide(values, 3),
                                      [-3, -3, -2, -2, -2, -1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3])
        for operation in (lambda: recipe.round_divide(np.array([np.iinfo(np.int64).max]), 2),
                          lambda: recipe.rescale(np.array([np.iinfo(np.int64).max]), 1),
                          lambda: recipe.rescale(values, np.iinfo(np.int64).min)):
            with self.assertRaises(ValueError):
                operation()

    def test_symmetric_quantization(self):
        np.testing.assert_array_equal(recipe.quantize(np.array([-200., -1.5, -.5, 0., .5, 1.5, 200.]), 0), [-127, -2, -1, 0, 1, 2, 127])
        self.assertEqual(recipe.exponent_for(127.), 0)
        self.assertEqual(recipe.exponent_for(128.), 1)
        with self.assertRaises(ValueError):
            recipe.quantize(np.array([np.nan]), 0)

    def test_contraction_against_direct_int64(self):
        x = np.arange(-18, 18, dtype=np.int8).reshape(1, 2, 3, 6)
        w = np.arange(-16, 16, dtype=np.int8).reshape(2, 2, 2, 4)
        attrs = {"stride": [1, 2], "padding": [1, 1], "dilation": [1, 1]}
        actual = recipe.integer_contraction(x, w, attrs)
        padded = np.pad(x.astype(np.int64), ((0, 0), (0, 0), (1, 1), (1, 1)))
        expected = np.zeros((1, 2, 4, 3), dtype=np.int64)
        for o in range(2):
            for y in range(4):
                for z in range(3):
                    expected[0, o, y, z] = (padded[0, :, y:y + 2, 2 * z:2 * z + 4] * w[o].astype(np.int64)).sum()
        np.testing.assert_array_equal(actual, expected)
        a, b = x.reshape(1, -1), w.reshape(2, -1)
        np.testing.assert_array_equal(recipe.integer_contraction(a[:, :16], b[:, :16]), a[:, :16].astype(np.int64) @ b[:, :16].astype(np.int64).T)

    def test_bias_aware_scaling_independent_power_of_two_witness(self):
        # Tiny folded weights can coexist with a substantial folded bias.
        raw_weight = np.array([[2. ** -24, -2. ** -24], [-2. ** -24, 2. ** -24], [0., 0.], [1., -1.]], dtype=np.float32)
        raw_bias = np.array([4., -4., 64., 2.], dtype=np.float32)
        original_weight, original_bias = raw_weight.copy(), raw_bias.copy()
        weight, exponents, bias, changes = recipe.quantize_parameters(raw_weight, raw_bias, -1)
        # At e=-28 the first two biases have magnitude 2**31; e=-27 is
        # the first admissible scale, with exact qweights +/-8 and bias +/-2**30.
        np.testing.assert_array_equal(exponents, [-27, -27, -23, -6])
        np.testing.assert_array_equal(weight, [[8, -8], [-8, 8], [0, 0], [64, -64]])
        np.testing.assert_array_equal(bias, [2 ** 30, -(2 ** 30), 2 ** 30, 256])
        self.assertEqual([change["channel"] for change in changes], [0, 1, 2])
        for channel, change in enumerate(changes):
            self.assertEqual(change["raw_weight_sha256"], recipe.array_digest(raw_weight[channel]))
            self.assertGreater(change["initial_bound_with_bias"], recipe.LIMIT)
            self.assertEqual(change["bound_with_bias"], 127 * sum(abs(int(v)) for v in weight[channel]) + abs(int(bias[channel])))
            self.assertLessEqual(change["bound_with_bias"], recipe.LIMIT)
        for channel in (0, 1):
            self.assertEqual(changes[channel]["initial_exponent"], -30)
            self.assertEqual(changes[channel]["initial_bias"], (1 if channel == 0 else -1) * 2 ** 33)
            self.assertEqual(changes[channel]["initial_accumulation_bound"], 127 * 128)
        # Independent Python integer contraction+bias witness at both endpoints.
        for channel in range(4):
            for inputs in ((127, -127), (-127, 127)):
                result = sum(x * int(w) for x, w in zip(inputs, weight[channel])) + int(bias[channel])
                self.assertLessEqual(abs(result), recipe.LIMIT)
        np.testing.assert_array_equal(raw_weight, original_weight)
        np.testing.assert_array_equal(raw_bias, original_bias)

    def test_bias_aware_scaling_exact_int32_boundary(self):
        for sign in (-1, 1):
            for extra, expected_exponent in ((0, -24), (1, -23)):
                with self.subTest(sign=sign, extra=extra):
                    raw_bias = np.array([sign * (recipe.LIMIT + extra) * 2. ** -24])
                    weight, exponents, bias, changes = recipe.quantize_parameters(np.zeros((1, 1)), raw_bias, 0)
                    self.assertEqual(int(exponents[0]), expected_exponent)
                    self.assertEqual(int(bias[0]), sign * (recipe.LIMIT if extra == 0 else 2 ** 30))
                    self.assertEqual(len(changes), extra)
        # A representable bias also needs headroom for the contraction itself.
        weight, exponents, bias, changes = recipe.quantize_parameters(np.array([[64.]]), np.array([float(recipe.LIMIT - 8127)]), 0)
        self.assertEqual(int(exponents[0]), 1)
        self.assertEqual(changes[0]["initial_bound_with_bias"], recipe.LIMIT + 1)
        self.assertEqual(changes[0]["initial_accumulation_bound"], 8128)
        self.assertEqual(int(weight[0, 0]), 32)

    def test_bias_aware_scaling_retains_fail_closed_limits(self):
        cases = [
            (np.array([[127. * 2. ** 30]]), np.array([float(recipe.LIMIT) * 2. ** 60]), 30),
            (np.zeros((1, 131072)), np.zeros(1), 0),
            (np.array([[2. ** -40]]), np.zeros(1), 0),
            (np.array([[np.nan]]), np.zeros(1), 0),
            (np.zeros((1, 1)), np.array([np.inf]), 0),
            (np.zeros((1, 1)), np.zeros(1), 31),
        ]
        for weight, bias, exponent in cases:
            with self.subTest(shape=weight.shape, exponent=exponent), self.assertRaises(ValueError):
                recipe.quantize_parameters(weight, bias, exponent)

    def test_finer_activation_scale_recomputes_bias_headroom(self):
        weight = np.array([[2. ** -24, -2. ** -24]], dtype=np.float32)
        bias = np.array([4.], dtype=np.float32)
        coarse = recipe.quantize_parameters(weight, bias, -1)
        fine = recipe.quantize_parameters(weight, bias, -2)
        self.assertEqual(int(coarse[1][0]), -27)
        self.assertEqual(int(fine[1][0]), -26)
        self.assertEqual(int(fine[2][0]), 2 ** 30)
        self.assertLessEqual(127 * abs(fine[0].astype(np.int64)).sum() + abs(int(fine[2][0])), recipe.LIMIT)
        with self.assertRaises(ValueError):
            recipe.quantize_parameters(np.zeros((1, 1)), np.array([float(recipe.LIMIT) * 2.]), -30)

    def test_post_relu_range_proof_and_signed_residual_cancellation(self):
        import operator
        import torch

        modules = torch.nn.Module()
        modules.add_module("arbitrary_left", torch.nn.Conv2d(1, 1, 1))
        modules.add_module("arbitrary_right", torch.nn.Conv2d(1, 1, 1))
        modules.add_module("pass_through", torch.nn.Identity())
        modules.add_module("discard_negative", torch.nn.ReLU(inplace=True))
        for branch in ("none", "producer", "identity", "residual"):
            with self.subTest(branch=branch):
                graph = torch.fx.Graph()
                image = graph.placeholder("pixels")
                left = graph.call_module("arbitrary_left", (image,))
                producer = left
                if branch == "residual":
                    right = graph.call_module("arbitrary_right", (image,))
                    producer = graph.call_function(operator.add, (left, right))
                identity = graph.call_module("pass_through", (producer,))
                relu = graph.call_module("discard_negative", (identity,))
                extra = producer if branch == "producer" else identity
                graph.output((relu, extra) if branch in ("producer", "identity") else relu)
                targets = recipe.post_relu_calibration_targets(torch.fx.GraphModule(modules, graph))
                if branch in ("producer", "identity"):
                    self.assertEqual(targets, {})
                else:
                    self.assertEqual(targets, {producer.name: {"relu": relu.name, "identities": [identity.name]}})
        # Signed pre-Add arms must survive unchanged: +128 and -126 cancel
        # to +2. Clipping the positive arm before addition would give +1.
        arms = [recipe.rescale(np.array([64], dtype=np.int8), 1),
                recipe.rescale(np.array([-63], dtype=np.int8), 1)]
        np.testing.assert_array_equal(recipe.saturate(sum(arms)), [2])
        # Large discarded negatives cannot consume the positive capacity.
        values = np.array([-512., -1., 0., .125, 1.], dtype=np.float32)
        exponent = recipe.exponent_for(float(np.maximum(values, 0).max()))
        np.testing.assert_array_equal(np.maximum(recipe.quantize(values, exponent), 0),
                                      recipe.quantize(np.maximum(values, 0), exponent))
        self.assertEqual(recipe.exponent_for(0.), -24)

    def test_averagepool_single_rounding_independent_witnesses(self):
        cases = [
            ([1, 0, 0], 0, -1, 1),
            ([-1, 0, 0], 0, -1, -1),
            ([1, 0, 0, 0], 0, -1, 1),
            ([-1, 0, 0, 0], 0, -1, -1),
            ([1, 1, 0], 0, 1, 0),
            ([-1, -1, 0], 0, 1, 0),
            ([3, 0, 0], 0, 1, 1),
            ([-3, 0, 0], 0, 1, -1),
            ([1, 2, 3], -2, -2, 2),
            ([127, 127], 0, -1, 127),
            ([-127, -127], 0, -1, -127),
        ]
        for values, input_exponent, output_exponent, expected in cases:
            with self.subTest(values=values, scales=(input_exponent, output_exponent)):
                actual = recipe.averagepool(np.array(values, dtype=np.int8).reshape(1, 1, 1, -1), input_exponent, output_exponent)
                self.assertEqual(actual.dtype, np.int8)
                self.assertEqual(actual.shape, (1, 1, 1, 1))
                self.assertEqual(int(actual.item()), expected)

    def test_averagepool_overflow_and_shift_proofs(self):
        limit64 = int(np.iinfo(np.int64).max)
        cases = [
            (limit64 // 128 + 1, 0, 0),
            (limit64 // (127 * 2 ** 30) + 1, 0, -30),
            (limit64 // 2 ** 30 + 1, -30, 0),
            (limit64 // 128, -2, 0),  # Sum and denominator fit; rounded numerator does not.
            (1, 30, -1),
            (1, 31, 1),
            (0, 0, 0),
        ]
        for area, input_exponent, output_exponent in cases:
            with self.subTest(area=area, scales=(input_exponent, output_exponent)), self.assertRaises(ValueError):
                recipe.averagepool_scale_factors(area, input_exponent, output_exponent)
        self.assertEqual(recipe.averagepool_scale_factors(3, 0, -1), (2, 3))
        self.assertEqual(recipe.averagepool_scale_factors(3, 0, 1), (1, 6))
        self.assertEqual(recipe.averagepool_scale_factors(limit64 // 128, 0, 0), (1, limit64 // 128))
        with self.assertRaises(ValueError):
            recipe.averagepool(np.full((1, 1, 1, 1), -128, dtype=np.int8), 0, 0)


class FullResNetDiagnostic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torchvision
        torch.set_num_threads(2)
        torch.manual_seed(811)
        cls.model = torchvision.models.resnet50(weights=None).cpu().eval()
        rng = np.random.default_rng(815)
        cls.calibration = rng.standard_normal((1, 3, 64, 64), dtype=np.float32)
        cls.evaluation = rng.standard_normal((1, 3, 64, 64), dtype=np.float32)
        cls.original_hash = recipe.state_digest(cls.model)
        cls.plan = recipe.calibrate_resnet50(cls.model, [cls.calibration], calibration_ids=["synthetic-calibration-815"], input_shape=cls.calibration.shape,
                                            checkpoint_declaration="random diagnostic; not pretrained")

    def test_complete_inventory_and_source_preserved(self):
        manifest = self.plan.manifest()
        self.assertFalse(manifest["paper_quality_approved"])
        self.assertEqual(manifest["recipe"], "resnet50-v1.5-symmetric-pow2-development-v4")
        for site in manifest["sites"]:
            if "bound_with_bias" in site:
                self.assertLessEqual(max(site["bound_with_bias"]), recipe.LIMIT)
                self.assertLessEqual(site["contraction_dimension_bound"], recipe.LIMIT)
            if "mean_requantization" in site:
                self.assertLessEqual(site["mean_requantization"]["rounded_numerator_bound"], np.iinfo(np.int64).max)
        self.assertEqual(sum(site.kind == "conv2d" for site in self.plan.sites), 53)
        self.assertEqual(sum(site.kind == "linear" for site in self.plan.sites), 1)
        self.assertEqual(sum(site.kind == "add" for site in self.plan.sites), 16)
        self.assertEqual(self.original_hash, recipe.state_digest(self.model))
        with self.assertRaises(ValueError):
            recipe.reference(self.plan, self.calibration)

    def assert_llvm_reference(self, plan, modes=("baseline", "optimized")):
        import tvm
        from verify_resnet import prepare_graph
        mod = recipe.export_relax(plan, return_integer_logits=True)
        image = recipe.quantize(self.evaluation, plan.input_exponent)
        expected = recipe.reference(plan, self.evaluation, return_integer_logits=True)
        for mode in modes:
            with self.subTest(mode=mode):
                _, lowered = prepare_graph(mod, mode, tvm)
                vm = tvm.relax.VirtualMachine(tvm.relax.build(lowered, target="llvm", pipeline=None), tvm.cpu())
                actual = vm["main"](tvm.nd.array(image))
                self.assertEqual(len(actual), len(expected))
                for value, oracle, dtype in zip(actual, expected, (np.int32, np.float32)):
                    self.assertEqual(value.numpy().dtype, dtype)
                    self.assertEqual(value.numpy().shape, (1, 1000))
                    np.testing.assert_array_equal(value.numpy(), oracle)

    def test_full_semantic_cpu_graph_matches_independent_reference(self):
        self.assert_llvm_reference(self.plan)

    def test_all_contractions_reach_device_graph(self):
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                _, _, coverage = recipe.prepare_device_graph(self.plan, optimize=optimize)
                self.assertEqual(coverage["device_invocations"], 54)

    def test_train_mse_plan_signed_observations_and_exact_cpu_graph(self):
        source = recipe.state_digest(self.model)
        plan = recipe.calibrate_resnet50(self.model, [self.calibration], calibration_ids=["synthetic-mse-training"],
                                        input_shape=self.calibration.shape, activation_estimator="mse")
        self.assertEqual(source, recipe.state_digest(self.model))
        self.assertEqual(plan.input_exponent, self.plan.input_exponent)
        self.assertEqual(plan.manifest()["recipe"], "resnet50-v1.5-symmetric-pow2-train-mse-v5")
        observed = {row["observed_site"] for row in plan.provenance["activation_scale_checks"]}
        projected = {row["site"]: row["relu"] for row in plan.provenance["post_relu_calibration"]}
        self.assertTrue(any(site.kind == "conv2d" and site.name not in projected for site in plan.sites))
        for site in plan.sites:
            if site.kind in ("conv2d", "add", "averagepool"):
                self.assertIn(projected.get(site.name, site.name), observed)
        for row in plan.provenance["activation_scale_checks"]:
            index = row["candidate_exponents"].index(row["selected_exponent"])
            self.assertEqual(row["mean_squared_errors"][index], min(row["mean_squared_errors"]))
        self.assert_llvm_reference(plan, ("optimized",))

    def test_classifier_logits_do_not_require_an_activation_exponent(self):
        import copy
        import torch
        model = copy.deepcopy(self.model)
        with torch.no_grad():
            model.fc.weight.fill_(1e9)
            model.fc.bias.zero_()
            self.assertGreater(float(model(torch.from_numpy(self.calibration)).abs().max()), 127 * 2.0**30)
        plan = recipe.calibrate_resnet50(model, [self.calibration], calibration_ids=["large-classifier-logits"],
                                        input_shape=self.calibration.shape, activation_estimator="mse")
        self.assertIsNone(plan.sites[-1].exponent)
        plan.validate()

    def test_accumulator_residual_full_reference_and_both_llvm_modes(self):
        import copy
        source = recipe.state_digest(self.model)
        plan = recipe.calibrate_resnet50(self.model, [self.calibration], calibration_ids=["synthetic-accumulator-training"],
                                        input_shape=self.calibration.shape, activation_estimator="mse", residual_policy="accumulator")
        self.assertEqual(source, recipe.state_digest(self.model))
        self.assertEqual(sum(site.retain_accumulator for site in plan.sites), 20)
        manifest = plan.manifest()
        self.assertEqual(len(manifest['residual_integer_proofs']), 16)
        self.assert_llvm_reference(plan)
        for optimize in (False, True):
            _, _, coverage = recipe.prepare_device_graph(plan, optimize=optimize)
            self.assertEqual(coverage["device_invocations"], 54)
        invalid = copy.deepcopy(plan)
        next(site for site in invalid.sites if site.kind == 'conv2d').retain_accumulator = True
        with self.assertRaisesRegex(ValueError, "unique Identity chain"):
            invalid.validate()

    def test_zero_positive_range_preserves_source_and_rebuilds_biases(self):
        import copy
        import torch
        model = copy.deepcopy(self.model)
        with torch.no_grad():
            model.conv1.weight.zero_()
            model.bn1.bias.fill_(-64.)
        source_hash = recipe.state_digest(model)
        plan = recipe.calibrate_resnet50(model, [self.calibration], calibration_ids=["negative-stem"], input_shape=self.calibration.shape)
        projection = next(row for row in plan.provenance["post_relu_calibration"] if row["site"] == plan.sites[0].name)
        self.assertEqual(projection["observed_max_abs"], 64.)
        self.assertEqual(projection["observed_post_relu_maximum"], 0.)
        self.assertEqual(plan.sites[0].exponent, -24)
        self.assertEqual(recipe.state_digest(model), source_hash)
        plan.validate()
        _, values = recipe.reference(plan, self.evaluation, return_intermediates=True)
        self.assertTrue((values[projection["relu"]] == 0).all())

    def test_averagepool_preserves_fractional_spatial_mean(self):
        _, values = recipe.reference(self.plan, self.evaluation, return_intermediates=True)
        site = next(site for site in self.plan.sites if site.kind == "averagepool")
        data = values[site.inputs[0]]
        self.assertEqual(data.shape[2:], (2, 2))
        self.assertTrue((data >= 0).all())
        source = next(source for source in self.plan.sites if source.name == site.inputs[0])
        shift = source.exponent - site.exponent
        numerator = data.astype(np.int64).sum(axis=(2, 3), keepdims=True) * 2 ** max(shift, 0)
        denominator = 4 * 2 ** max(-shift, 0)
        expected = np.clip((numerator + denominator // 2) // denominator, -127, 127).astype(np.int8)
        np.testing.assert_array_equal(values[site.name], expected)

    def test_unsafe_modified_plan_refused(self):
        import copy
        modified = copy.copy(self.plan)
        modified.sites = list(self.plan.sites)
        modified.sites[0] = copy.copy(modified.sites[0])
        modified.sites[0].bias = np.full_like(modified.sites[0].bias, recipe.LIMIT)
        with self.assertRaises(ValueError):
            recipe.reference(modified, self.evaluation)
        with self.assertRaises(ValueError):
            recipe.export_relax(modified)

    def test_invalid_calibration_refused(self):
        with self.assertRaises(ValueError):
            recipe.calibrate_resnet50(self.model, [self.calibration, self.calibration], calibration_ids=["one", "two"], input_shape=self.calibration.shape)

    def test_invalid_plan_parameters_and_geometry_refused(self):
        import copy
        def modify_site(kind, update):
            plan = copy.copy(self.plan)
            plan.sites = list(self.plan.sites)
            index = next(i for i, site in enumerate(plan.sites) if site.kind == kind)
            plan.sites[index] = copy.copy(plan.sites[index])
            update(plan.sites[index])
            return plan
        cases = [
            ("conv2d", lambda site: setattr(site, "weight_exponents", np.full_like(site.weight_exponents, np.iinfo(np.int64).min))),
            ("linear", lambda site: setattr(site, "weight_exponents", np.full_like(site.weight_exponents, np.iinfo(np.int64).min))),
            ("conv2d", lambda site: setattr(site, "inputs", site.inputs * 2)),
            ("conv2d", lambda site: setattr(site, "weight", site.weight[:, :2].copy())),
            ("conv2d", lambda site: setattr(site, "shape", (*site.shape[:-1], site.shape[-1] + 1))),
            ("conv2d", lambda site: setattr(site, "attrs", {**site.attrs, "stride": [0, 1]})),
            ("conv2d", lambda site: setattr(site, "exponent", 30)),
            ("relu", lambda site: setattr(site, "exponent", site.exponent + 1)),
            ("add", lambda site: setattr(site, "inputs", site.inputs[:1])),
            ("flatten", lambda site: setattr(site, "shape", (1, 1))),
            ("averagepool", lambda site: setattr(site, "shape", (*site.shape[:2], 2, 1))),
            ("averagepool", lambda site: setattr(site, "exponent", 30)),
            ("maxpool", lambda site: setattr(site, "attrs", {**site.attrs, "padding": [100, 100]})),
            ("linear", lambda site: setattr(site, "shape", (1, 999))),
            ("linear", lambda site: setattr(site, "exponent", 0)),
        ]
        for kind, update in cases:
            with self.subTest(kind=kind, update=update):
                plan = modify_site(kind, update)
                with self.assertRaises(ValueError):
                    plan.validate()
                with self.assertRaises(ValueError):
                    recipe.reference(plan, self.evaluation)
        self.assertEqual(recipe.state_digest(self.model), self.original_hash)

    def test_modified_calibration_provenance_refused(self):
        import copy
        for hashes in ((), ("bad-hash",), self.plan.calibration_hashes * 2):
            plan = copy.copy(self.plan)
            plan.calibration_hashes = hashes
            with self.subTest(hashes=hashes), self.assertRaises(ValueError):
                plan.validate()
        plan = copy.copy(self.plan)
        plan.provenance = {**plan.provenance, "calibration_ids": []}
        with self.assertRaises(ValueError):
            plan.validate()


if __name__ == "__main__":
    unittest.main()
