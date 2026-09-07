"""Pure build declarations preserve the runtime API without loading execution backends."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import pytest

from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe


def test_runtime_reexports_exact_pure_objects():
    from merlin.runtime.backends import base
    from merlin.runtime.fp8_formats import float_format_of
    assert base.HarnessBuildRecipe is HarnessBuildRecipe
    assert base.float_format_of is float_format_of
    for token, expected in (("f32", "f32"), ("bfloat16", "bf16"), ("i8", None), ("unknown", None)):
        assert float_format_of(token) == expected


def test_recipe_defaults_and_order_are_unchanged():
    recipe = HarnessBuildRecipe(Path("/cc"), (Path("/inc"),), (Path("/support.c"),), Path("/link.ld"), 0)
    assert recipe.command(sources=[Path("/source.c")], output=Path("/out")) == [
        "/cc", "-I", "/inc", "-T", "/link.ld", "-o", "/out", "/source.c", "/support.c"]
    assert recipe.error_cls is RuntimeError
    assert recipe.cflags == recipe.ldflags == ()
    with pytest.raises(FrozenInstanceError):
        recipe.load_address = 1
    with pytest.raises(RuntimeError, match="no -march"):
        recipe.march()


def test_pure_recipe_and_format_import_with_runtime_backends_masked():
    script = '''
import sys
class Deny:
    def find_spec(self, fullname, *args):
        if fullname.startswith("merlin.runtime.backends") or fullname in {"merlin.runtime.simulator", "merlin.runtime.reference"}:
            raise ImportError("masked execution/oracle: " + fullname)
sys.meta_path.insert(0, Deny())
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.runtime.fp8_formats import float_format_of
assert float_format_of("f32") == "f32"
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
