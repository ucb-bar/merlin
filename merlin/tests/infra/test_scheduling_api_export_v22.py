"""V22 keeps the V21 scheduling roster and stages no author-withheld package in its SDK release."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase2.common_transforms import supports_check

from merlin.common.access import author_inaccessible
from merlin.common.paths import repo_root
from merlin.compile.scheduling.api_catalog import scheduling_capabilities
from merlin.compile.scheduling.export import public_modules
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.generate_bundles import _select_python_grants
from merlin.targetgen.tool_profile import (
    COMMON_SUBSTRATE_V14,
    COMMON_SUBSTRATE_V15,
    attach_common_substrate,
    tool_profile_prompt,
    verify_tool_profile,
)
from merlin.targetgen.tool_profile_shared import (
    prepare_shared_source_release,
    reviewed_shared_manifest,
    sdk_support_sources,
    staged_module,
)

TOOLS = TR.ARM_TOOLS["merlin_rtlchecks"]


@pytest.fixture(scope="module")
def releases(tmp_path_factory):
    out = {}
    for version in (21, 22):
        root = tmp_path_factory.mktemp(f"sdk{version}")
        out[version] = (
            root,
            prepare_shared_source_release(
                root,
                python_source_root=repo_root() / "src",
                license_directory=repo_root(),
                tools=TOOLS,
                version=version,
                source_binding=True,
            ),
        )
    return out


def _modules(root: Path) -> dict[str, Path]:
    return {staged_module(p.relative_to(root).as_posix()): p for p in (root / "merlin").rglob("*.py")}


def _imports(path: Path, module: str):
    """Absolute dotted targets of every import statement in one staged file (lazy ones included)."""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*parts, base] if base else parts)
            yield base
            yield from (f"{base}.{alias.name}" for alias in node.names)


def test_v22_roster_and_catalog_are_v21_under_a_new_inventory_name():
    assert library_modules(22) == library_modules(21)
    assert public_modules(22) == public_modules(21)
    v21, v22 = scheduling_capabilities(21), scheduling_capabilities(22)
    assert v22["inventory_version"] == 22 and v21["inventory_version"] == 21
    assert v22["capabilities"] == v21["capabilities"]
    assert sdk_support_sources(21) == () and sdk_support_sources(22)
    for invalid in (23, True, "22"):
        with pytest.raises(ValueError):
            library_modules(invalid)
        with pytest.raises(ValueError):
            scheduling_capabilities(invalid)


def test_v22_release_stages_no_withheld_package_and_v21_bytes_are_unchanged(releases):
    (root21, manifest21), (root22, manifest22) = releases[21], releases[22]
    assert manifest21["schema"] == "merlin.shared_scheduling_library.v15"
    assert manifest22["schema"] == "merlin.shared_scheduling_library.v16"
    assert reviewed_shared_manifest(root22, version=22) == manifest22
    assert [r["module"] for r in manifest22["members"]] == [r["module"] for r in manifest21["members"]]
    staged21, staged22 = _modules(root21), _modules(root22)
    withheld21 = sorted(m for m in staged21 if author_inaccessible(m))
    # The historical release still carries the protected spine; it is frozen evidence, not a fix target.
    assert {"merlin.kernels.cca", "merlin.kernels.action_catalog", "merlin.kernels.microkernel"} <= set(withheld21)
    # ... and the RTL bridge's bit-exact cosim, a module-level access-policy oracle under a directory grant.
    assert "merlin.targetgen.rtl.mlc_bridge" in withheld21
    assert not [m for m in staged22 if author_inaccessible(m)]
    assert set(staged21) - set(staged22) == set(withheld21)
    assert set(staged22) - set(staged21) == {staged_module("merlin/" + r) for r in sdk_support_sources(22)} - set(
        staged21
    )
    for module in set(staged21) & set(staged22):
        assert staged21[module].read_bytes() == staged22[module].read_bytes(), module


def test_v22_generic_analysis_closure_is_staged_and_old_shim_paths_are_unused(releases):
    root, _ = releases[22]
    staged = _modules(root)
    analysis = "merlin.compile.analysis"
    wanted, missing, old_paths = set(), set(), set()
    for module, path in staged.items():
        for target in _imports(path, module):
            if target == analysis or target.startswith(analysis + "."):
                wanted.add(target)
            if target in ("merlin.kernels.endpoints", "merlin.kernels.roles"):
                old_paths.add(module)
    for target in wanted:
        if target not in staged and target.rpartition(".")[0] not in staged:
            missing.add(target)
    assert not missing, sorted(missing)
    assert not old_paths, sorted(old_paths)
    # The explicit list is exactly the derived closure beyond what arm tool grants already stage (plus
    # the package initializer, for arms whose tools do not grant it).
    granted = {
        staged_module("merlin/" + p.removeprefix("merlin/python/merlin/"))
        for t in TOOLS
        for p in TR.spec(t).bundle_paths
    }
    support = {staged_module("merlin/" + r) for r in sdk_support_sources(22)}
    assert support - {analysis} == {t for t in wanted if t in staged} - granted - {analysis}


def test_v22_isolated_imports_of_tools_that_need_moved_analyses(releases):
    root, _ = releases[22]
    code = "\n".join(
        [
            "import importlib, sys",
            "sys.path.insert(0, " + repr(str(root)) + ")",
            "for name in ('merlin.xdsl_dialects.lowering.feature_axis_trace', 'merlin.compile.analysis.rocc_commands',",
            "             'merlin.compile.analysis.intrinsics_header', 'merlin.targetgen.rtl_backend'):",
            "    module = importlib.import_module(name)",
            "    assert module.__file__.startswith(" + repr(str(root)) + "), (name, module.__file__)",
            "import importlib.util",
            "spec = importlib.util.find_spec('merlin.kernels')",
            "assert spec is None or not any(p.startswith(" + repr(str(root)) + ")",
            "                               for p in spec.submodule_search_locations or ()), spec",
            "print('V22-SDK-IMPORTS-PASS')",
        ]
    )
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip().endswith("V22-SDK-IMPORTS-PASS")


def _arm_bundle(root: Path, substrate: str) -> dict:
    allowed = [{"path": p, "mode": "ro", "note": "tool"} for t in TOOLS for p in TR.spec(t).bundle_paths]
    bundle = dict(arm="merlin_assisted", tools=list(TOOLS), allowed=allowed, denied=[], host_inputs=[])
    _select_python_grants(bundle, root)
    attach_common_substrate(bundle, version=substrate, python_source_root=root, host_index_bits=64)
    _select_python_grants(bundle, root)
    return bundle


def test_v15_bundle_denies_withheld_grants_and_grants_staged_analyses(releases):
    root22, _ = releases[22]
    bundle = _arm_bundle(root22, COMMON_SUBSTRATE_V15)
    sdk = str(root22 / "merlin") + "/"
    allowed = [row["path"] for row in bundle["allowed"]]
    assert not [
        p for p in allowed if p.startswith(sdk) and author_inaccessible(staged_module("merlin/" + p[len(sdk) :]))
    ]
    missing = [p for p in allowed if p.startswith(sdk) and not Path(p).exists()]
    assert not missing, missing
    assert {sdk + r for r in sdk_support_sources(22)} <= set(allowed)
    denied = {row["path"] for row in bundle["denied"]}
    assert sdk + "kernels/cca.py" in denied and sdk + "kernels/endpoints.py" in denied
    # The rtl/ directory grant stays; the oracle module below it is a recorded denial.
    assert sdk + "targetgen/rtl/" in allowed and sdk + "targetgen/rtl/mlc_bridge.py" in denied
    assert not (root22 / "merlin/targetgen/rtl/mlc_bridge.py").exists()
    profile = verify_tool_profile(bundle)
    assert profile["common_substrate"] == COMMON_SUBSTRATE_V15 and supports_check(profile)
    prompt = tool_profile_prompt(bundle)
    assert "scheduling_capabilities(22)" in prompt and "scheduling_capabilities(21)" not in prompt
    # The V14 bundle keeps its historical grants byte for byte.
    root21, _ = releases[21]
    old = _arm_bundle(root21, COMMON_SUBSTRATE_V14)
    assert str(root21 / "merlin/kernels/cca.py") in {row["path"] for row in old["allowed"]}
    assert "scheduling_capabilities(21)" in tool_profile_prompt(old)
    assert not supports_check({"common_substrate": "merlin.common_compiler_substrate.v16"})
