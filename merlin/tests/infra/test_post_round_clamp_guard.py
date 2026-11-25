import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_post_round_clamp import BASE, SOURCE

from merlin.llvmlower.post_round_clamp import rewrite
from merlin.llvmlower.post_round_clamp_guard import CalleePolicy


class GuardTests(unittest.TestCase):
    def refused(self, s):
        with self.assertRaises((ValueError, TypeError)):
            rewrite(s, BASE)

    def test_alias_returning_zero(self):
        self.refused(
            SOURCE.replace(
                "declare float @roundevenf(float)",
                (
                    "@roundevenf = alias float (float), ptr @wrong\ndefine float @wrong(float "
                    "%x) {\nentry:\n ret float 0.000000e+00\n}"
                ),
            )
        )

    def test_ifunc(self):
        self.refused(
            SOURCE.replace("declare float @roundevenf(float)", "@roundevenf = ifunc float (float), ptr @resolver")
        )

    def test_global_rebinding(self):
        self.refused(SOURCE.replace("declare float @roundevenf(float)", "@roundevenf = external global ptr"))

    def test_double_declaration(self):
        self.refused(SOURCE + "declare float @roundevenf(float)\n")

    def test_missing_declaration(self):
        self.refused(SOURCE.replace("declare float @roundevenf(float)", ""))

    def test_wrong_signature(self):
        self.refused(SOURCE.replace("declare float @roundevenf(float)", "declare double @roundevenf(double)"))

    def test_unknown_direct_observer(self):
        self.refused(SOURCE.replace("  ret i8 %q", "  call void @observe()\n  ret i8 %q") + "declare void @observe()\n")

    def test_unused_unknown_declaration(self):
        self.refused(SOURCE + "declare i32 @fetestexcept(i32)\n")

    def test_inline_csr_read(self):
        self.refused(SOURCE.replace("  ret i8 %q", '  %flags = call i32 asm "csrr $0, fflags", "=r"()\n  ret i8 %q'))

    def test_inline_frm_read(self):
        self.refused(SOURCE.replace("  ret i8 %q", '  %mode = call i32 asm "csrr $0, frm", "=r"()\n  ret i8 %q'))

    def test_unknown_indirect_observer(self):
        self.refused(
            SOURCE.replace("@quant(float %x)", "@quant(float %x, ptr %observer)").replace(
                "  ret i8 %q", "  call void %observer()\n  ret i8 %q"
            )
        )

    def test_function_address_taken(self):
        self.refused(SOURCE + "@p = global ptr @roundevenf\n")

    def test_unknown_llvm_intrinsic(self):
        self.refused(SOURCE + "declare i32 @llvm.read_register.i32(metadata)\n")

    def test_source_policy_digest_changed(self):
        with self.assertRaises(ValueError):
            rewrite(SOURCE, dataclasses.replace(BASE, callee_policy=CalleePolicy(BASE.callee_policy.path, "0" * 64)))

    def test_untyped_policy(self):
        with self.assertRaises(TypeError):
            rewrite(SOURCE, dataclasses.replace(BASE, callee_policy=object()))

    def test_local_direct_helper_reachable(self):
        s = SOURCE + "define i8 @outer(float %x) {\nentry:\n %q = call i8 @quant(float %x)\n ret i8 %q\n}\n"
        out, r = rewrite(s, BASE)
        self.assertEqual(r["rewrites"], 1)

    def test_actual_module_has_separately_retained_replay(self):
        self.skipTest(
            "Actual linked-module evidence uses the independent artifact replay, not synthetic fixture semantics"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
