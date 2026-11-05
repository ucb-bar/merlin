"""A reviewed leaf fixes split-library shadowing without widening any namespace."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from merlin.common.paths import repo_root

ROOT = repo_root()
SPEC = importlib.util.spec_from_file_location(
    "prepare_author_library_example", ROOT / "examples/gemmini/phase1/launch/prepare_author_library.py"
)
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def test_reviewed_dependency_restores_staged_rtl_discovery_import(tmp_path):
    generic, sdk = tmp_path / "generic", tmp_path / "base-sdk"

    def copy(root, module):
        relative = Path(*module.split(".")).with_suffix(".py")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / "src" / relative).read_bytes())

    # Model the actual selected ordering: root package extends across both
    # releases, but generic merlin.common is a regular, closed subpackage.
    root = generic / "merlin"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n")
    for name in ("merlin.common.__init__", "merlin.common.digest", "merlin.common.paths"):
        copy(generic, name)
    for name in ("merlin.targetgen.__init__", "merlin.targetgen.rtl.__init__",
                 "merlin.targetgen.rtl.facts", "merlin.targetgen.rtl.gen_isa_module",
                 "merlin.common.__init__", "merlin.common.facts_view"):
        copy(sdk, name)
    code = "\n".join([
        "import sys,json",
        "sys.path[:0] = " + repr([str(generic), str(sdk)]),
        "from merlin.targetgen.rtl import gen_isa_module",
        "from merlin.common import facts_view",
        "assert callable(gen_isa_module.generate)",
        "assert facts_view.interface({'interfaces':[{'name':'probe','width':7}]}, 'probe')['width'] == 7",
        "print(json.dumps({'generator':gen_isa_module.__file__, 'dependency':facts_view.__file__}))",
    ])
    before = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert before.returncode != 0 and "No module named 'merlin.common.facts_view'" in before.stderr
    assert PREP.reviewed_discovery_modules((), existing=()) == ()
    modules = PREP.reviewed_discovery_modules((), existing=(), enabled=True)
    additions = PREP.reviewed_modules(modules, existing=())
    assert additions == {"merlin/common/facts_view.py": "merlin.common.facts_view"}
    for name in additions.values():
        copy(generic, name)
    after = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert after.returncode == 0, after.stderr
    result = json.loads(after.stdout)
    assert Path(result["generator"]).is_relative_to(sdk)
    assert Path(result["dependency"]).is_relative_to(generic)
    assert PREP.reviewed_discovery_modules((), existing=modules, enabled=True) == ()
    assert PREP.reviewed_discovery_modules(modules, existing=(), enabled=True) == modules
