"""Private certificate IO closure without changing historical/public grants."""

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from test_compiler_tool_profile import selected_sources as selected_sources  # noqa: F401

from merlin.common.paths import repo_root
from merlin.targetgen.tool_profile import (
    COMMON_SUBSTRATE_V1,
    COMMON_SUBSTRATE_V2,
    COMMON_SUBSTRATE_V3,
    COMMON_SUBSTRATE_V4,
    COMMON_SUBSTRATE_V5,
    _digest,
    attach_common_substrate,
    inherited_tool_profile,
    substrate_data,
    substrate_inventory,
    verify_tool_profile,
)

VERSIONS = (COMMON_SUBSTRATE_V1, COMMON_SUBSTRATE_V2, COMMON_SUBSTRATE_V3, COMMON_SUBSTRATE_V4, COMMON_SUBSTRATE_V5)
PRIVATE = ("merlin.common.__init__", "merlin.common.dense_float_ir")
PREFIX = "merlin/python/merlin/"


def bundle(root, version):
    result = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(result, version=version, python_source_root=root, host_index_bits=64)
    return result


def extend_fixture(root, version):
    for module in sum(substrate_inventory(version), ()):
        path = root / (module.replace(".", "/") + ".py")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('"""Independent historical source inventory fixture."""\n')
    for relative in substrate_data(version):
        path = root / "merlin" / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("-- Independent historical data inventory fixture.\n")


def add_dense_dependency(root, statement="from merlin.common.dense_float_ir import DenseFloatParser\n"):
    (root / "merlin/transforms/certificates.py").write_text(statement)
    for module in PRIVATE:
        path = root / (module.replace(".", "/") + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repo_root() / "src" / (module.replace(".", "/") + ".py")).read_bytes())


@pytest.mark.parametrize("version", VERSIONS)
def test_new_dense_dependency_preserves_historical_frozen_inventory(selected_sources, version):
    extend_fixture(selected_sources, version)
    historical = bundle(selected_sources, version)
    frozen = copy.deepcopy(historical)
    old = verify_tool_profile(historical)
    assert "common_implementation_modules" not in old
    add_dense_dependency(selected_sources)
    current = bundle(selected_sources, version)
    profile = verify_tool_profile(current)
    assert profile["common_implementation_modules"] == list(PRIVATE)
    assert profile["common_public_modules"] == old["common_public_modules"] == list(substrate_inventory(version)[1])
    assert not any(
        name == "merlin.common" or name.startswith("merlin.common.") for name in profile["common_public_modules"]
    )
    assert verify_tool_profile(historical) == old
    assert historical == frozen
    assert inherited_tool_profile(historical, copy.deepcopy(historical)) == old
    with pytest.raises(ValueError, match="Phase 2 changed"):
        inherited_tool_profile(historical, current)


@pytest.mark.parametrize(
    "statement",
    [
        "from merlin.common.dense_float_ir import DenseFloatParser\n",
        "from ..common.dense_float_ir import DenseFloatParser\n",
        "from merlin.common import dense_float_ir\n",
        "import merlin.common.dense_float_ir as dense\n",
    ],
)
def test_selected_certificate_import_adds_only_closed_private_pair(selected_sources, statement):
    add_dense_dependency(selected_sources, statement)
    profile = verify_tool_profile(bundle(selected_sources, COMMON_SUBSTRATE_V1))
    assert profile["common_implementation_modules"] == list(PRIVATE)
    assert [row["module"] for row in profile["common_sources"]][-2:] == list(PRIVATE)


def test_unreferenced_helper_or_import_spelling_in_text_grants_nothing(selected_sources):
    add_dense_dependency(selected_sources)
    (selected_sources / "merlin/transforms/certificates.py").write_text(
        '"""from merlin.common.dense_float_ir import DenseFloatParser"""\n# import merlin.common.dense_float_ir\n'
    )
    profile = verify_tool_profile(bundle(selected_sources, COMMON_SUBSTRATE_V1))
    assert "common_implementation_modules" not in profile
    assert not any(row["module"] in PRIVATE for row in profile["common_sources"])


@pytest.mark.parametrize("missing", PRIVATE)
def test_missing_selected_private_namespace_or_helper_refuses(selected_sources, missing):
    add_dense_dependency(selected_sources)
    (selected_sources / (missing.replace(".", "/") + ".py")).unlink()
    with pytest.raises(ValueError, match="common compiler source is absent"):
        bundle(selected_sources, COMMON_SUBSTRATE_V1)


@pytest.mark.parametrize(
    "members",
    [
        ["merlin.common.dense_float_ir"],
        ["merlin.common.__init__"],
        [*PRIVATE, "merlin.runtime.reference"],
        list(reversed(PRIVATE)),
    ],
)
def test_private_extension_cannot_admit_partial_or_arbitrary_source_grants(selected_sources, members):
    add_dense_dependency(selected_sources)
    selected = bundle(selected_sources, COMMON_SUBSTRATE_V1)
    profile = selected["compiler_tool_profile"]
    profile["common_implementation_modules"] = members
    profile["sha256"] = _digest({k: v for k, v in profile.items() if k != "sha256"})
    with pytest.raises(ValueError, match="implementation dependency changed"):
        verify_tool_profile(selected)


@pytest.mark.parametrize("version", VERSIONS)
def test_fresh_profiles_execute_common_pipeline_from_declared_release_only(tmp_path, version):
    selected = bundle(repo_root() / "src", version)
    profile = verify_tool_profile(selected)
    assert profile["common_implementation_modules"] == ["merlin.transforms.pipeline_state", *PRIVATE]
    paths = [row["path"] for row in profile["common_sources"]]
    assert len(paths) == len(set(paths))
    assert [row["path"] for row in selected["allowed"] if row["path"].startswith(PREFIX + "common/")] == [
        PREFIX + "common/__init__.py",
        PREFIX + "common/dense_float_ir.py",
    ]
    root = tmp_path / "frozen"
    for row in profile["common_sources"]:
        relative = "merlin/" + row["path"].removeprefix(PREFIX)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repo_root() / "src" / relative).read_bytes())
    (root / "profile.json").write_text(json.dumps(profile))
    dependencies = set()
    for name in ("xdsl", "typing_extensions", "immutabledict", "ordered_set"):
        spec = importlib.util.find_spec(name)
        assert spec is not None and spec.origin is not None
        path = Path(spec.origin)
        dependencies.add(str(path.parent.parent if spec.submodule_search_locations else path.parent))
    script = r"""import hashlib, importlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path[:0] = [str(root), *json.loads(sys.argv[2])]
profile = json.loads((root/'profile.json').read_text())
for name in profile['common_public_modules']:
    importlib.import_module(name)
from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.pipeline import PipelineOptions, apply_common_pipeline
raw = bytes.fromhex('0000807F00000080')
source = 'builtin.module { %x = arith.constant dense<"0x0000807F00000080"> : tensor<2xf32> }'
module = parse_ir(source)
result = apply_common_pipeline(module, PipelineOptions(index_bits=64), plan_only=True)
parsed = parse_ir(ir_text(result.module))
assert parsed.body.block.first_op.value.data.data == raw
pins = {root/('merlin/'+r['path'].removeprefix('merlin/python/merlin/')): r['sha256']
        for r in profile['common_sources']}
for name, module in tuple(sys.modules.items()):
    if name == 'merlin' or name.startswith('merlin.'):
        if module.__file__ is not None:
            path = pathlib.Path(module.__file__)
            assert path in pins, (name,path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == pins[path]
        assert all(pathlib.Path(p).is_relative_to(root) for p in getattr(module,'__path__',()))
try:
    import merlin.runtime.backends
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('withheld backend present')
"""
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", script, str(root), json.dumps(sorted(dependencies))],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr


def test_private_helper_remains_unavailable_as_a_direct_candidate_api(tmp_path):
    from merlin.targetgen.common_transforms import CommonTransformService
    from merlin.targetgen.package_runtime import _py_imports_merlin
    from merlin.targetgen.sandbox import bwrap

    root = repo_root() / "src"
    selected = bundle(root, COMMON_SUBSTRATE_V1)
    for row in selected["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(root / row["path"].removeprefix("merlin/python/"))
    archive = tmp_path / "archive.yaml"
    archive.write_text(yaml.safe_dump(selected))
    workspace = tmp_path / "original"
    workspace.mkdir()
    bwrap.materialize_bundle_inputs(workspace, selected, repo=repo_root())
    owner = CommonTransformService(workspace=workspace, bundle_manifest=archive, repo=repo_root())
    with owner.compiler_imports() as admission:
        assert _py_imports_merlin("import merlin.transforms.certificates", compiler_api=admission) is None
        for module in ("merlin.common", "merlin.common.dense_float_ir", "merlin.runtime.reference"):
            assert _py_imports_merlin("import " + module, compiler_api=admission) is not None
