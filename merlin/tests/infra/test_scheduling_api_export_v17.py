"""One additive source export; existing V16 discovery and rosters stay fixed."""

import hashlib
import json
import subprocess
import sys

import pytest

from merlin.common.paths import repo_root
from merlin.compile.scheduling.api_catalog import scheduling_capabilities
from merlin.compile.scheduling.export import public_modules
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V10, attach_common_substrate, verify_tool_profile
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release, reviewed_shared_manifest


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_v17_adds_only_reviewed_spatial_owner_and_closes_public_imports(tmp_path):
    assert _digest(library_modules(16)) == "59488f02aca3507417cd20ac309e88737c4390fd41c2271a4c6f51a5c6549b46"
    assert _digest(public_modules(16)) == "a52f692fe60ced031c47372f16fa943e9962d8138b3cda24e35c99581d417ce6"
    assert _digest(scheduling_capabilities()) == "67b63dfde34f9e80719d5bce0f19f0cd591c7cc749835cb8fe08624660dc6905"
    new = {"merlin.compile.scheduling." + name for name in (
        "spatial_readout_cache", "rolling_window", "rolling_spatial",
    )} | {"merlin.xdsl_dialects.rolling_window"}
    assert set(library_modules(17)) - set(library_modules(16)) == new
    assert set(public_modules(17)) - set(public_modules(16)) == new
    with pytest.raises(ValueError, match="explicit inventory"):
        scheduling_capabilities(20)
    root = tmp_path / "prospective"
    manifest = prepare_shared_source_release(
        root, python_source_root=repo_root() / "src", license_directory=repo_root(), version=17, source_binding=True,
    )
    assert reviewed_shared_manifest(root, version=17) == manifest
    assert manifest["schema"] == "merlin.shared_scheduling_library.v11"
    assert manifest["scheduling_capabilities"] == scheduling_capabilities(17)
    bundle = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V10, python_source_root=root, host_index_bits=64)
    profile = verify_tool_profile(bundle)
    from merlin_experiments.phase2.common_transforms import supports_check

    assert supports_check(profile)
    assert not supports_check({"common_substrate": "merlin.common_compiler_substrate.v13"})
    code = "\n".join([
        "import importlib,json,pathlib,sys",
        "root=pathlib.Path(" + repr(str(root)) + ")",
        "sys.path.insert(0,str(root))",
        "from merlin.compile.scheduling.api_catalog import scheduling_capabilities",
        "catalog=scheduling_capabilities(17)",
        "manifest=json.loads((root/'shared-library.json').read_text())",
        "assert catalog==manifest['scheduling_capabilities']",
        "members={m['module']:m for m in manifest['members'] if m['module']}",
        "for row in catalog['capabilities']:",
        " for symbol in row['symbols']:",
        "  module,name=symbol.rsplit('.',1);owner=importlib.import_module(module)",
        "  assert pathlib.Path(owner.__file__)==root/members[module]['path']",
        "  assert callable(getattr(owner,name))",
        "print(sum(len(row['symbols'])for row in catalog['capabilities']))",
    ])
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == 24
