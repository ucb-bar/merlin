"""The atlas-kernels hook resolves sources as tools/ did and its standalone stand-ins match atlas-mlir."""
import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kernel_library  # noqa: E402
import atlas_program  # noqa: E402
import run_model_tile  # noqa: E402
from chipyard import arena  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KernelLibraryTest(unittest.TestCase):
    def test_program_source_names_match_atlas_program(self):
        tree = ast.parse(kernel_library.module_path("atlas_program.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "code_sources")
        names = next(node.value for node in function.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "names" for target in node.targets))
        self.assertEqual(ast.literal_eval(names), kernel_library.PROGRAM_SOURCES)

    def test_sources_resolve_to_atlas_mlir_or_the_submodule(self):
        paths = kernel_library.source_paths(traffic_reuse=True)
        self.assertEqual(paths["run_model_tile.py"], ROOT / "tools/run_model_tile.py")
        self.assertEqual(paths["atlas_program.py"], kernel_library.LIBRARY / "kernels/projection_gemm/atlas_program.py")
        for name, path in paths.items():
            self.assertEqual(Path(sys.modules[name[:-3]].__file__ if name[:-3] in sys.modules else path), path)

    def test_real_helpers_win_over_standalone_fallbacks(self):
        self.assertEqual(Path(run_model_tile.__file__), ROOT / "tools/run_model_tile.py")
        self.assertEqual(Path(arena.__file__), ROOT / "tools/chipyard/arena.py")

    def test_fallback_archive_admission_is_a_verbatim_copy(self):
        fallback = load("fallback_tile_admission", kernel_library.LIBRARY / "lib/fallback/tile_admission.py")
        self.assertEqual(inspect.getsource(fallback.validate_archive_members),
                         inspect.getsource(run_model_tile.validate_archive_members))

    def test_fallback_relocation_matches_chipyard_arena(self):
        fallback = load("fallback_arena", kernel_library.LIBRARY / "lib/fallback/chipyard/arena.py")
        for m, k, n in ((32, 32, 32), (128, 768, 768), (128, 3072, 768), (64, 12288, 960), (128, 768, 3072)):
            source = atlas_program.assembly(m, k, n, m * k + k * n + 256, -15, traffic_reuse=(m, k) != (32, 32))
            with self.subTest(shape=(m, k, n)):
                self.assertEqual(fallback.relocate_assembly(source, 0x100000000, 0x90000000),
                                 arena.relocate_assembly(source, 0x100000000, 0x90000000))


if __name__ == "__main__":
    unittest.main()
