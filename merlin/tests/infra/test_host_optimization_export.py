"""The selected host library executes without the compiler or target checkout."""

import hashlib
import json
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.compile.host_optimization_export import PUBLIC_MODULES, export_host_optimizations
from merlin.llvmlower.toolchain import m2m_python
from merlin.targetgen.compiler_library import freeze_compiler_library
from merlin.tests.ir.test_scalar_contraction import source


def test_export_is_reviewable_and_retains_exact_algorithm_sources(tmp_path):
    output = tmp_path / "host-library"
    manifest = export_host_optimizations(output)
    assert manifest["schema"] == "merlin.shared_host_library.v1"
    assert manifest["public_modules"] == list(PUBLIC_MODULES)
    for row in manifest["members"]:
        assert hashlib.sha256((output / row["path"]).read_bytes()).hexdigest() == row["sha256"]
        if row["module"] and not row["path"].endswith("__init__.py"):
            assert (output / row["path"]).read_bytes() == (repo_root() / "src" / row["path"]).read_bytes()
    freeze_compiler_library(
        output,
        review_id="shared-host-source-test",
        public_modules=PUBLIC_MODULES,
        sources=tuple((row["path"], row["module"]) for row in manifest["members"])
        + (("shared-host-library.json", None),),
    )
    assert not any("targetgen" in row["path"] or "phase1_generated" in row["path"] for row in manifest["members"])
    with pytest.raises(ValueError, match="empty"):
        export_host_optimizations(output)


def test_exported_native_rewrite_and_cpu_emission_execute_in_fresh_interpreter(tmp_path):
    output = tmp_path / "host-library"
    export_host_optimizations(output)
    text = source((3, 7), (7, 5), (3, 5))
    program = """
import sys, json
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from torch_mlir import ir
from merlin.compile.host_optimizations import rewrite_tensor_module, legalize_bounded_rounding
from merlin.runtime.host_rounding import roundeven_f32_assembly
import merlin.compile.host_optimizations as owner
assert Path(owner.__file__).is_relative_to(Path(sys.argv[1]))
with ir.Context() as ctx:
    text = sys.stdin.read()
    module = ir.Module.parse(text)
    counts = rewrite_tensor_module(ctx, module)
    assert counts['contraction'] == 1, counts
    module.operation.verify()
    assert not any(rewrite_tensor_module(ctx, module).values())
    for attribute in ('llvm.strictfp', 'merlin.fenv = true'):
        scoped = ir.Module.parse(text.replace('module {', 'module attributes {' + attribute + '} {', 1))
        before = str(scoped)
        assert not any(rewrite_tensor_module(ctx, scoped).values())
        assert str(scoped) == before
    unsupported = ir.Module.parse(text.replace('f32', 'f64'))
    before = str(unsupported)
    assert not any(rewrite_tensor_module(ctx, unsupported).values())
    assert str(unsupported) == before
assert 'fcvt.w.s' in roundeven_f32_assembly(isa='rv64gc', abi='lp64d')
assert legalize_bounded_rounding('source', host_isa=None)[0] == 'source'
try:
    import merlin.targetgen
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('private compiler orchestration leaked into the host source library')
print(json.dumps(counts))
"""
    run = subprocess.run(
        [str(m2m_python()), "-c", program, str(output)],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert json.loads(run.stdout)["contraction"] == 1
