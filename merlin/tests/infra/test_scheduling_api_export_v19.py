"""One new causal frontier owner; unchanged historical inventories and guards."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.compile.scheduling.api_catalog import scheduling_capabilities
from merlin.compile.scheduling.export import public_modules
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen.compiler_library import CompilerLibraryError, freeze_compiler_library
from merlin.targetgen.tool_profile import COMMON_SUBSTRATE_V13, attach_common_substrate, verify_tool_profile
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release, reviewed_shared_manifest


def test_v19_adds_only_ready_owner_and_old_refresh_cannot_acquire_it(tmp_path):
    new = ("merlin.compile.scheduling.dense_ready_readout",)
    assert library_modules(19) == (*library_modules(18), *new)
    assert public_modules(19) == (*public_modules(18), *new)
    assert library_modules() == library_modules(6)
    assert "final_k_causal_accumulator_readout_frontier" not in {
        r["name"] for r in scheduling_capabilities(18)["capabilities"]
    }
    assert scheduling_capabilities(19)["capabilities"][:-1] == scheduling_capabilities(18)["capabilities"]
    assert scheduling_capabilities(19)["capabilities"][-1]["symbols"]
    old = tmp_path / "legacy-refresh"
    m = prepare_shared_source_release(
        old, python_source_root=repo_root() / "src", license_directory=repo_root(), version=18, source_binding=True
    )
    # A new HEAD is not the old frozen source epoch. No implicit roster growth.
    with pytest.raises(CompilerLibraryError, match="unreviewed library dependency"):
        freeze_compiler_library(old, review_id="legacy-refresh-must-refuse", public_modules=tuple(m["public_modules"]),
                                sources=tuple((r["path"], r["module"]) for r in m["members"])
                                + tuple((r, None) for r in m["review_resources"]))
    for version in (23, True, "19"):
        with pytest.raises(ValueError):
            library_modules(version)


def test_v19_actual_exported_frontier_scf_llvm_and_original_check_guard(tmp_path):
    from merlin_experiments.phase2.common_transforms import supports_check

    from merlin.targetgen.common_transforms import CommonTransformService

    root = tmp_path / "prospective"
    manifest = prepare_shared_source_release(
        root, python_source_root=repo_root() / "src", license_directory=repo_root(), version=20, source_binding=True,
    )
    assert reviewed_shared_manifest(root, version=20) == manifest
    assert manifest["schema"] == "merlin.shared_scheduling_library.v14"
    bundle = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(bundle, version=COMMON_SUBSTRATE_V13, python_source_root=root, host_index_bits=64)
    profile = verify_tool_profile(bundle)
    assert supports_check(profile)
    assert not supports_check({"common_substrate": "merlin.common_compiler_substrate.v16"})
    service = object.__new__(CommonTransformService)
    service.profile = profile
    with pytest.raises(ValueError, match="cannot select semantics"):
        service.check({"schema": "no-author-proof-answers"}, work_root=tmp_path)
    code = "\n".join([
        "import importlib,json,pathlib,sys",
        "root=pathlib.Path(" + repr(str(root)) + ")",
        "sys.path.insert(0,str(root))",
        "sys.path.append(" + repr(str(Path(__file__).parent)) + ")",
        "from merlin.compile.scheduling.api_catalog import scheduling_capabilities",
        "catalog=scheduling_capabilities(20)",
        "for row in catalog['capabilities']:",
        " for symbol in row['symbols']:",
        "  module,name=symbol.rsplit('.',1)",
        "  assert callable(getattr(importlib.import_module(module),name)),symbol",
        "import test_dense_ready_readout as focused",
        "focused.test_actual_scf_and_llvm_consume_frontier_not_conservative_source_projection()",
        "manifest=json.loads((root/'shared-library.json').read_text())",
        "members={r['module']:r for r in manifest['members'] if r['module']}",
        "for name,owner in tuple(sys.modules.items()):",
        " if name=='merlin' or name.startswith('merlin.'):",
        "  path=getattr(owner,'__file__',None)",
        "  if path: assert pathlib.Path(path)==root/members[name]['path'],(name,path)",
        "print('actual-final-K-source-SCF-LLVM-PASS')",
    ])
    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code], text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "actual-final-K-source-SCF-LLVM-PASS"
    assert json.loads((root / "shared-library.json").read_text())["inventory_version"] == 20
