"""Manifest inspection works from only the two explicitly selected public leaves."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_shared_pass_contract import manifest

from merlin.targetgen import capsule_common, shared_pass_contract
from merlin.targetgen.compiler_library import freeze_compiler_library
from merlin.targetgen.package_runtime import CertFailure


def test_actual_public_inspector_sources_form_closed_grant_and_run_in_isolation(tmp_path):
    public = ("merlin.targetgen.shared_pass_contract", "merlin.targetgen.oot_starterkit.passes")
    root = tmp_path / "selected"
    sources = {}
    source_root = Path(shared_pass_contract.__file__).parents[2]
    for module in public:
        parts = module.split(".")
        for index in range(1, len(parts)):
            relative = "/".join(parts[:index]) + "/__init__.py"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Selected public namespace; no implicit sibling grants.\n")
            sources[relative] = ".".join(parts[:index])
        relative = module.replace(".", "/") + ".py"
        shutil.copyfile(source_root / relative, root / relative)
        sources[relative] = module
    selected = freeze_compiler_library(
        root, review_id="neutral-public-pass-inspector", public_modules=public, sources=tuple(sources.items())
    )
    assert all(selected.permits(module) for module in public)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()))
    script = """
import importlib.abc,json,sys
sys.path.insert(0, sys.argv[1])
class NoEnforcement(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('merlin.targetgen.capsule_common','merlin.targetgen.package_runtime','aet')):
            raise AssertionError('protected import: '+fullname)
sys.meta_path.insert(0,NoEnforcement())
from merlin.targetgen.oot_starterkit.passes import inspect_manifest
value=inspect_manifest(sys.argv[2])
assert value['changed_ir_consumption']=='NOT_RUN'
assert value['numerical_agreement']=='NOT_RUN'
assert value['performance']=='NOT_MEASURED'
print(json.dumps(value))
"""
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(root), str(path)],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "declared_interfaces"
    assert report["contract"]["target_consumer"]["target"] == "neutral"
    broken = manifest()
    broken["commands"]["emit_target_bundle"]["argv"] = ["{tool}", "{input_mlir}"]
    path.write_text(json.dumps(broken))
    refused = subprocess.run(
        [sys.executable, "-I", "-c", script, str(root), str(path)],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert refused.returncode != 0 and "producing {output_json}" in refused.stderr
    assert not refused.stdout


@pytest.mark.parametrize("broken", [False, True])
def test_protected_adapter_reuses_public_structure_but_retains_failure_category(broken):
    value = manifest()
    if broken:
        value["abi_version"] = "99"
        with pytest.raises(ValueError, match="ABI 0.1"):
            shared_pass_contract.validate_target_ir_consumer_declaration(value)
        with pytest.raises(CertFailure, match="ABI 0.1") as refused:
            capsule_common._target_consumer_record(SimpleNamespace(manifest=value))
        assert refused.value.plane == "contract"
    else:
        expected = shared_pass_contract.validate_target_ir_consumer_declaration(value)
        assert capsule_common._target_consumer_record(SimpleNamespace(manifest=value)) == expected
        value["target_ir_consumer"]["input"] = "changed"
        assert expected["declaration"]["input"] == "selected_target_mlir"
