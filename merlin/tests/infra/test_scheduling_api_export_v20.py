"""The ordinary selected SDK must contain executable carried-event dependencies."""

import json
import subprocess
import sys
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.compile.scheduling.api_catalog import scheduling_capabilities
from merlin.compile.scheduling.export import public_modules
from merlin.compile.scheduling.library_sources import library_modules
from merlin.targetgen.compiler_library import freeze_compiler_library
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release, reviewed_shared_manifest


def test_v20_membership_and_current_v21_isolated_real_scf_llvm_callbacks(tmp_path):
    additions = (
        "merlin.sched.ir.events",
        "merlin.compile.scheduling.scf_carried_events",
        "merlin.sched.primitives.event_loops",
    )
    assert library_modules(20)[:len(library_modules(19)) + 3] == (*library_modules(19), *additions)
    assert public_modules(20)[:len(public_modules(19)) + 3] == (*public_modules(19), *additions)
    assert "merlin.xdsl_dialects.lowering.outline" in library_modules(20)
    assert "merlin.runtime.reference" not in library_modules(20)
    assert "merlin.runtime.simulator" not in library_modules(20)
    assert scheduling_capabilities(20)["inventory_version"] == 20
    assert scheduling_capabilities(20)["capabilities"][-1]["name"] == "loop_carried_completion_events"
    assert "merlin.common.stat_memo" not in library_modules(20)
    assert "merlin.common.stat_memo" in library_modules(21)
    root = tmp_path / "sdk"
    manifest = prepare_shared_source_release(
        root, python_source_root=repo_root() / "src", license_directory=repo_root(),
        version=21, source_binding=True,
    )
    assert reviewed_shared_manifest(root, version=21) == manifest
    contract = freeze_compiler_library(
        root, review_id="explicit-carried-event-sdk",
        public_modules=tuple(manifest["public_modules"]),
        sources=tuple((row["path"], row["module"]) for row in manifest["members"])
        + tuple((path, None) for path in manifest["review_resources"]),
    )
    contract.verify(root)
    code = "\n".join([
        "import pathlib,sys",
        f"root=pathlib.Path({str(root)!r})",
        "sys.path.insert(0,str(root))",
        f"sys.path.append({str(Path(__file__).parent)!r})",
        "import test_scf_carried_events as check",
        "check.test_startup_steady_drain_and_nested_real_loop_results(7,False)",
        "check.test_startup_steady_drain_and_nested_real_loop_results(5,True)",
        "check.test_actual_llvm_cfg_consumes_carried_event_handles_and_completion_calls()",
        "for name,owner in tuple(sys.modules.items()):",
        " if name=='merlin' or name.startswith('merlin.'):",
        "  path=getattr(owner,'__file__',None)",
        "  if path: assert pathlib.Path(path).is_relative_to(root),(name,path)",
        "print('carried-events-export-SCF-LLVM-PASS')",
    ])
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "carried-events-export-SCF-LLVM-PASS"
    assert json.loads((root / "shared-library.json").read_text())["inventory_version"] == 21
