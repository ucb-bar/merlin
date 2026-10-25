"""Completed APIs enter a new closed export/profile, with real isolated emission."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase2.common_transforms import supports_check

from merlin.common.paths import repo_root
from merlin.compile.scheduling.api_catalog import scheduling_capabilities
from merlin.compile.scheduling.export import public_modules
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen.tool_profile import (
    COMMON_SUBSTRATE_V11,
    attach_common_substrate,
    tool_profile_prompt,
    verify_tool_profile,
)
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release, reviewed_shared_manifest

NEW = {
    "merlin.compile.scheduling.dense_readout_overlap",
    "merlin.llvmlower.host_builder",
    "merlin.compile.scheduling.projection_views",
    "merlin.transforms.tensor_views",
    "merlin.llvmlower.source_region",
    "merlin.llvmlower.source_region_host",
    "merlin.compile.composition_catalog",
}


def test_v18_additive_closed_inventory_and_original_version_defaults():
    assert set(library_modules(18)) - set(library_modules(17)) == NEW | {"merlin.llvmlower.llvm_loop_metadata"}
    assert set(public_modules(18)) - set(public_modules(17)) == NEW
    assert library_modules(18)[:len(library_modules(17))] == library_modules(17)
    assert public_modules(18)[:len(public_modules(17))] == public_modules(17)
    assert library_modules() == library_modules(6)
    assert public_modules() == public_modules(6)
    assert scheduling_capabilities() == scheduling_capabilities(16)
    assert scheduling_capabilities(18)["capabilities"][:5] == scheduling_capabilities(17)["capabilities"]
    assert "composition" not in scheduling_capabilities(17)
    assert not any("ready_readout" in name for name in library_modules(18))
    for invalid in (20, True, "18"):
        with pytest.raises(ValueError):
            library_modules(invalid)
        with pytest.raises(ValueError):
            scheduling_capabilities(invalid)


def test_v18_normal_profile_isolated_imports_and_actual_calls(tmp_path):
    root = tmp_path / "prospective"
    manifest = prepare_shared_source_release(
        root, python_source_root=repo_root() / "src", license_directory=repo_root(), version=18, source_binding=True,
    )
    assert reviewed_shared_manifest(root, version=18) == manifest
    assert manifest["schema"] == "merlin.shared_scheduling_library.v12"
    bundle = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V11, python_source_root=root, host_index_bits=64)
    profile = verify_tool_profile(bundle)
    assert profile["shared_scheduling_library"] == manifest
    assert supports_check(profile)
    assert "scheduling_capabilities(18)" in tool_profile_prompt(bundle)
    assert not supports_check({"common_substrate": "merlin.common_compiler_substrate.v13"})
    with pytest.raises(ValueError, match="unknown common compiler substrate"):
        attach_common_substrate(dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[]),
                                version="merlin.common_compiler_substrate.v13", python_source_root=root,
                                host_index_bits=64)
    probe = Path(__file__).with_name("scheduling_export_v18_probe.py")
    code = "\n".join([
        "import importlib,json,pathlib,runpy,sys",
        "root=pathlib.Path(" + repr(str(root)) + ")",
        "sys.path.insert(0,str(root))",
        "from merlin.compile.scheduling.api_catalog import scheduling_capabilities",
        "catalog=scheduling_capabilities(18)",
        "manifest=json.loads((root/'shared-library.json').read_text())",
        "assert catalog==manifest['scheduling_capabilities']",
        "members={m['module']:m for m in manifest['members'] if m['module']}",
        "rows=catalog['capabilities']",
        "for row in rows:",
        " if row.get('access','author_library')!='author_library': continue",
        " for symbol in row['symbols']:",
        "  module,name=symbol.rsplit('.',1);owner=importlib.import_module(module)",
        "  assert module in manifest['public_modules'],module",
        "  assert pathlib.Path(owner.__file__)==root/members[module]['path']",
        "  assert callable(getattr(owner,name)),symbol",
        "from merlin.compile.composition_catalog import composition_capabilities",
        "from merlin.compile.scheduling.library_sources import library_modules",
        "assert composition_capabilities(selected_modules=library_modules(18))==catalog['composition']",
        "assert catalog['composition']['selected_source_contract_required']",
        "receipt=runpy.run_path(" + repr(str(probe)) + ")['run']()",
        "print(json.dumps(receipt,sort_keys=True))",
    ])
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["readout_source_products"] == 300
    assert receipt["readout_dynamic_drains_before"] > receipt["readout_dynamic_drains_after"]
    assert receipt["readout_llvm_operations"] > 0 and receipt["persistent_llvm_operations"] > 0
    assert len(receipt["readout_actual_kernel_sha256"]) == hashlib.sha256().digest_size * 2


def test_v11_enters_original_check_validation_and_cli_unknown_versions_refuse(tmp_path):
    from merlin.targetgen.common_transforms import CommonTransformService

    service = object.__new__(CommonTransformService)
    service.profile = {"common_substrate": COMMON_SUBSTRATE_V11}
    with pytest.raises(ValueError, match="cannot select semantics"):
        service.check({"schema": "not-a-proof"}, work_root=tmp_path)
    service.profile = {"common_substrate": "merlin.common_compiler_substrate.v13"}
    with pytest.raises(ValueError, match="outside the original frozen substrate"):
        service.check({}, work_root=tmp_path)
    help_result = subprocess.run([sys.executable, "-B", "-m", "merlin.targetgen.generate_bundles", "--help"],
                                 text=True, capture_output=True)
    assert help_result.returncode == 0, help_result.stderr
    assert COMMON_SUBSTRATE_V11 in help_result.stdout
