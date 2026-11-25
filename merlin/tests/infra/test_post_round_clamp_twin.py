import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from _post_round_clamp_fixture import ROOT as _FIXTURE_ROOT

from merlin.llvmlower import post_round_clamp_stage as T
from merlin.llvmlower.compilation_recipe import CompilationRecipe

ROOT = _FIXTURE_ROOT / "twin-fixtures"
ROOT.mkdir(exist_ok=True)


class TwinTests(unittest.TestCase):
    def test_same_size_changed_executable_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            d = Path(tmp)
            a = d / "a"
            b = d / "b"
            a.write_bytes(b"123")
            b.write_bytes(b"124")
            with self.assertRaises(ValueError):
                T.identity_equal(a, b)

    def test_actual_native_link_identical_and_supplier_traced(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            d = Path(tmp)
            src = d / "source.c"
            src.write_text("int shared_value(void){return 3;} int main(void){return shared_value()-3;}\n")
            cc = Path(shutil.which("cc")).resolve()
            obj = d / "model.o"
            elf = d / "model.elf"
            recipe = CompilationRecipe(d, producer=Path(__file__))

            def run(argv, **kw):
                return subprocess.run(argv, capture_output=True, text=True, check=True, **kw)

            recipe.run([cc, "-c", src, "-o", obj], runner=run, inputs=[src], output=obj)
            recipe.run([cc, obj, "-o", elf], runner=run, inputs=[obj], output=elf)
            recipe.completed(elf)
            observed, result, command = T.trace_completed_link(
                recipe.path, d / "twin", expected_suppliers={"shared_value": obj}
            )
            self.assertTrue(observed["same_executable_bytes"])
            self.assertIn("definition of shared_value", result.stderr)
            self.assertTrue(elf.is_file())
            self.assertEqual(subprocess.run([elf]).returncode, 0)

    def test_wrong_actual_supplier_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            d = Path(tmp)
            src = d / "source.c"
            src.write_text("int shared_value(void){return 3;} int main(void){return shared_value()-3;}\n")
            extra = d / "extra.c"
            extra.write_text("int unrelated(void){return 7;}\n")
            cc = Path(shutil.which("cc")).resolve()
            obj = d / "model.o"
            other = d / "other.o"
            elf = d / "model.elf"
            recipe = CompilationRecipe(d, producer=Path(__file__))

            def run(argv, **kw):
                return subprocess.run(argv, capture_output=True, text=True, check=True, **kw)

            for source, target in [(src, obj), (extra, other)]:
                recipe.run([cc, "-c", source, "-o", target], runner=run, inputs=[source], output=target)
            recipe.run([cc, obj, other, "-o", elf], runner=run, inputs=[obj, other], output=elf)
            recipe.completed(elf)
            with self.assertRaises(ValueError):
                T.trace_completed_link(recipe.path, d / "twin", expected_suppliers={"shared_value": other})
            self.assertEqual(subprocess.run([elf]).returncode, 0)

    def test_timeout_bool_refused_before_io(self):
        with self.assertRaises(ValueError):
            T.trace_completed_link(Path("/nonexistent"), Path("/nonexistent"), expected_suppliers={}, timeout_s=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
