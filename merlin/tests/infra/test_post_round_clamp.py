import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _post_round_clamp_fixture import BASE

from merlin.llvmlower.post_round_clamp import rewrite

SOURCE = """define i8 @quant(float %x) {
entry:
  %producer = fdiv float %x, 1.250000e-01
  %r = call float @roundevenf(float %producer)
  %lc = fcmp olt float %r, -1.270000e+02
  %lo = select i1 %lc, float -1.270000e+02, float %r
  %hc = fcmp ogt float %lo, 1.270000e+02
  %hi = select i1 %hc, float 1.270000e+02, float %lo
  %q = fptosi float %hi to i8
  ret i8 %q
}
declare float @roundevenf(float)
"""


class Tests(unittest.TestCase):
    def test_integer_terminal_selected(self):
        s, r = rewrite(SOURCE, BASE)
        self.assertEqual(r["rewrites"], 1)
        self.assertNotIn("%r = call float @roundevenf", s)
        self.assertIn("fcvt.w.s", s)
        self.assertIn("%producer = fdiv float %x, 1.250000e-01", s)
        self.assertIn("%q = fptosi float %hi to i8", s)

    def test_portable_proof_route(self):
        s, r = rewrite(SOURCE, dataclasses.replace(BASE, host_isa="portable"))
        self.assertEqual(r["rewrites"], 1)
        self.assertIn("declare float @llvm.roundeven.f32(float)", s)

    def test_live_unclamped_round_refused(self):
        s = SOURCE.replace("  ret i8 %q", "  %other = fadd float %r, %x\n  ret i8 %q")
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_live_final_float_refused(self):
        s = (
            SOURCE.replace("  ret i8 %q", "  call void @observe(float %hi)\n  ret i8 %q")
            + "declare void @observe(float)\n"
        )
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_float_store_refused(self):
        s = SOURCE.replace("  ret i8 %q", "  store float %hi, ptr null\n  ret i8 %q")
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_wrong_endpoint_refused(self):
        s = SOURCE.replace("1.270000e+02", "1.275000e+02")
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_wider_than_float_exact_integer_refused(self):
        s = SOURCE.replace("i8", "i32")
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_foreign_round_function_refused(self):
        s = SOURCE.replace("@roundevenf", "@roundf")
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_local_roundeven_definition_refused(self):
        s = SOURCE.replace("declare float @roundevenf(float)", "define float @roundevenf(float %z) { ret float %z }")
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_fast_math_refused(self):
        s = SOURCE.replace("call float @roundevenf", "call fast float @roundevenf")
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_strict_fp_refused(self):
        s = SOURCE.replace("@quant(float %x) {", "@quant(float %x) strictfp {")
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_constrained_fp_refused(self):
        s = SOURCE + "declare float @llvm.experimental.constrained.fadd.f32(float,float,metadata,metadata)\n"
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_fenv_caller_refused(self):
        s = SOURCE + "declare i32 @fetestexcept(i32)\n"
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_errno_observed_refused(self):
        with self.assertRaises(ValueError):
            rewrite(SOURCE, dataclasses.replace(BASE, errno_observed=True))

    def test_fenv_observed_refused(self):
        with self.assertRaises(ValueError):
            rewrite(SOURCE, dataclasses.replace(BASE, fenv_observed=True))

    def test_value_only_not_boolean_refused(self):
        with self.assertRaises(ValueError):
            rewrite(SOURCE, dataclasses.replace(BASE, value_only=1))

    def test_supplier_changed_refused(self):
        with self.assertRaises(ValueError):
            rewrite(SOURCE, dataclasses.replace(BASE, supplier_sha256="0" * 64))

    def test_basic_block_crossing_refused(self):
        s = SOURCE.replace("  %lc =", "  br label %next\nnext:\n  %lc =")
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_post_round_clamp_not_alone_live_predicate(self):
        s = SOURCE.replace("  ret i8 %q", "  %use = select i1 %lc, i8 %q, i8 0\n  ret i8 %use")
        try:
            self.assertEqual(rewrite(s, BASE)[0], s)
        except ValueError:
            pass

    def test_unknown_supplier_source_refused(self):
        with self.assertRaises(ValueError):
            rewrite(SOURCE, dataclasses.replace(BASE, supplier_source_sha256="0" * 64))

    def test_supplier_recipe_changed_refused(self):
        with self.assertRaises(ValueError):
            rewrite(SOURCE, dataclasses.replace(BASE, supplier_recipe_sha256="0" * 64))

    def test_named_temporary_collision_avoided(self):
        s = SOURCE.replace("  %producer =", "  %merlin.post_rne.0.lowcmp = fcmp olt float %x, 0.0\n  %producer =")
        selected, r = rewrite(s, BASE)
        self.assertEqual(r["rewrites"], 1)
        self.assertIn("%merlin.post_rne.1.lowcmp =", selected)

    def test_float_bound_counterexample(self):
        import numpy as np

        x = np.float32(0.6)
        self.assertNotEqual(
            np.clip(np.rint(x), np.float32(-0.5), np.float32(0.5)),
            np.rint(np.clip(x, np.float32(-0.5), np.float32(0.5))),
        )

    def test_unclamped_live_counterexample(self):
        import numpy as np

        self.assertNotEqual(np.rint(np.float32(1000)), np.rint(np.clip(np.float32(1000), -127, 127)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
