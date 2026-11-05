"""Explicit generated instruction inputs reuse normal reviewed-library isolation."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase1 import generated_isa_library as G
from merlin_experiments.phase2 import compiler_libraries as overlays
from merlin_experiments.phase2 import compiler_library_state as state
from test_compiler_libraries import policy

from merlin.targetgen import isa_dialect_gen, isa_spec
from merlin.targetgen.compiler_library import CompilerLibrarySelection, freeze_compiler_library, scoped_compiler_library


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "base"
    (root / "merlin/targetgen").mkdir(parents=True)
    sources = [("merlin/__init__.py", "merlin"), ("merlin/targetgen/__init__.py", "merlin.targetgen")]
    for path, _ in sources:
        (root / path).write_text("")
    for module in (isa_spec, isa_dialect_gen):
        name = "merlin/targetgen/" + Path(module.__file__).name
        (root / name).write_bytes(Path(module.__file__).read_bytes())
        sources.append((name, module.__name__))
    (root / "private_backend.py").write_text('raise RuntimeError("not granted")')
    contract = freeze_compiler_library(
        root,
        review_id="test-base",
        public_modules=(isa_spec.__name__, isa_dialect_gen.__name__),
        sources=tuple(sources),
    )
    base = CompilerLibrarySelection(contract, root)
    model = isa_spec.ResolvedISA(
        "synthetic",
        "example",
        8,
        "a" * 64,
        "b" * 64,
        (
            isa_spec.Instruction(
                "step",
                (),
                (),
                (),
                (isa_spec.EncodingField(0, 8, 7, None),),
                '{"status":"unknown"}',
                '{"status":"unknown"}',
            ),
        ),
    )
    package = tmp_path / "package"
    isa_dialect_gen.generate_package(model, package)
    (package / "private_kernel.py").write_text('raise RuntimeError("not granted")')
    kwargs = dict(
        review_id="test-with-scaffold",
        manifest_sha256=hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest(),
        facts_sha256=model.facts_sha256,
        declaration_sha256=model.declaration_sha256,
    )
    return base, package, tmp_path / "selected", kwargs


def test_exact_extension_import_and_normal_readonly_overlay(inputs, tmp_path):
    base, package, dest, kwargs = inputs
    selected = G.prepare_library(base, package, dest, **kwargs)
    assert len(selected.contract.members) == len(base.contract.members) + 5
    assert not (dest / "private_backend.py").exists()
    assert not (dest / "reviewed_isa/private_kernel.py").exists()
    assert "fresh optimized backend atop generated scaffolding" in G.prompt(selected)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            f"import sys; sys.path.insert(0, {str(dest)!r}); from reviewed_isa.instruction_dialect import encode; "
            'assert encode("step", {}) == 7',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    binding = state.freeze_selection(selected, tmp_path / "snapshot")
    frozen = state.verify_binding(binding, selected)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    with scoped_compiler_library(frozen):
        mounted = overlays.apply(policy(candidate))
        assert str(frozen.root / "reviewed_isa/instruction_dialect.py") in mounted.argv
        assert str(package) not in mounted.argv
        assert "private_kernel.py" not in str(mounted.argv)
    assert not (frozen.root / "reviewed_isa/instruction_dialect.py").stat().st_mode & 0o222
    with pytest.raises(state.StageGateError):
        state.verify_binding(binding, base)


@pytest.mark.parametrize("change", ["manifest", "facts", "declaration", "product", "rerender", "symlink", "generator"])
def test_refuse_unreviewed_or_inconsistent_products(inputs, change):
    base, package, dest, kwargs = inputs
    if change in ("manifest", "facts", "declaration"):
        kwargs[change + "_sha256"] = "f" * 64
    elif change == "symlink":
        (package / "instruction_dialect.py").unlink()
        (package / "instruction_dialect.py").symlink_to(package / "isa-resolved.json")
    else:
        path = package / "instruction_dialect.py"
        if change != "generator":
            path.write_text(path.read_text() + '\nraise RuntimeError("injected")\n')
        if change in ("rerender", "generator"):
            manifest_path = package / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            if change == "rerender":
                manifest["files"]["instruction_dialect.py"] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                manifest["generator"]["modules"]["merlin.targetgen.isa_spec"] = "f" * 64
            manifest_path.write_text(json.dumps(manifest))
            kwargs["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        G.prepare_library(base, package, dest, **kwargs)
    assert not dest.exists()


def test_legacy_library_has_no_scaffold_guidance(inputs):
    assert G.prompt(inputs[0]) == ""
