"""Atlas kernel-library example: routing, synthetic artifact integrity, and selection schema.

Mirrors test_radiance_kernel_library_example.py. Nothing here contacts a network or runs hardware. The
two local-checkout tests run only when MERLIN_ATLAS_KERNELS names an atlas-kernels checkout that has the
pinned commit; they export from it and regenerate selection.yaml from it.
"""

import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root

EXAMPLE = "examples/atlas/phase1/kernel_library"
RADIANCE = "examples/radiance/phase1/kernel_library"
PINNED = "9f63c532d4aad8e1f6a178109edba042528ca33c"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, repo_root() / path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules while the module executes.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[name]
    return module


def load_materializer():
    return _load(f"{EXAMPLE}/materialize.py", "atlas_kernel_library_example")


def load_generator():
    return _load(f"{EXAMPLE}/generate_selection.py", "atlas_kernel_library_selection_generator")


def load_registry():
    # The only implementation of semantic_kernel_family_registry_v1, the schema both selections use.
    return _load("examples/muon/support/backend/muon_kernel_selection.py", "semantic_kernel_family_registry")


def _selection():
    return yaml.safe_load(load_materializer().SELECTION.read_bytes())


def test_selection_bytes_and_pinned_semantics_preserved():
    module = load_materializer()
    payload = module.SELECTION.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == "5238697eeb582c2ab9673f272700e28463551fbfbca8c7e296096b2920a3c641"
    selection = yaml.safe_load(payload)
    assert selection["source_commit"] == PINNED
    assert selection["package"] == "out/artifacts/targets/atlas/kernel_library_v1"
    assert load_generator().SOURCE_COMMIT == PINNED


def test_selection_uses_the_radiance_field_names():
    radiance = yaml.safe_load((repo_root() / RADIANCE / "selection.yaml").read_bytes())
    atlas = _selection()
    assert list(atlas) == list(radiance)
    assert atlas["compiler_selection"]["schema"] == radiance["compiler_selection"]["schema"]
    radiance_keys = {key for rule in radiance["compiler_selection"]["rules"] for key in rule}
    for rule in atlas["compiler_selection"]["rules"]:
        assert set(rule) == radiance_keys


def test_every_declared_family_has_one_compiler_rule():
    registry = load_registry()
    census = registry.validate_selection_contract(_selection())
    assert census["qualified_families"] == 9
    assert census["experimental_families"] == 0
    assert census["compiler_rules"] == 9
    assert census["missing_rules"] == []
    assert census["selection_is_numeric_qualification"] is False


MXU = ["systolic_array", "fp8_mxu", "bf16_accumulate"]
VPU = ["vector_unit", "exp", "reciprocal", "sqrt", "tanh"]


@pytest.mark.parametrize(
    ("request_", "selected"),
    [
        ({"op": "matmul", "dtype": "fp8_e4m3", "shape": {"M": 128, "K": 2048, "N": 2048}}, "kernels/projection_gemm"),
        ({"op": "matmul", "dtype": "fp8_e4m3", "shape": {"M": 64, "K": 736, "N": 2048}}, "kernels/projection_gemm"),
        ({"op": "matmul", "dtype": "fp8_e4m3", "shape": {"M": 64, "K": 720, "N": 2048}}, None),
        ({"op": "matmul_silu", "dtype": "fp8_e4m3", "shape": {"M": 128, "K": 2048, "N": 2048}}, "kernels/epilogue"),
        ({"op": "matmul_add", "dtype": "fp8_e4m3", "shape": {"M": 128, "K": 2048, "N": 2048}}, "kernels/rne_add"),
        (
            {"op": "matmul_rope", "dtype": "fp8_e4m3", "shape": {"M": 32, "K": 2048, "N": 256, "head_dim": 64}},
            "kernels/rope",
        ),
        ({"op": "rmsnorm", "dtype": "bf16", "shape": {"rows": 128, "cols": 2048}}, "kernels/rmsnorm"),
        ({"op": "rmsnorm", "dtype": "bf16", "shape": {"rows": 128, "cols": 960}}, None),
        (
            {"op": "attention_gqa_causal", "dtype": "bf16", "shape": {"Sq": 128, "Sk": 128, "D": 64, "QH": 32, "KVH": 4}},
            "kernels/attention",
        ),
        ({"op": "attention_gqa", "dtype": "bf16", "shape": {"Sq": 1024, "Sk": 4096, "D": 64, "QH": 12, "KVH": 12}}, None),
        (
            {"op": "matmul_bias_gelu_tanh", "dtype": "fp8_e4m3", "shape": {"M": 128, "K": 768, "N": 3072}},
            "kernels/smol_fusion",
        ),
    ],
)
def test_selection_routes_model_requests(request_, selected):
    registry = load_registry()
    report = registry.select_kernel_family(request_, {"features": MXU + VPU}, _selection())
    assert report.selected_family == selected
    assert len(report.decisions) == 9


def test_exact_add_wins_but_truncating_epilogue_add_remains_auditable():
    registry = load_registry()
    report = registry.select_kernel_family(
        {"op": "matmul_add", "dtype": "fp8_e4m3", "shape": {"M": 128, "K": 2048, "N": 2048}},
        {"features": MXU + VPU},
        _selection(),
    )
    decisions = {item.family: item.status for item in report.decisions}
    assert decisions["kernels/rne_add"] == "selected"
    assert decisions["kernels/epilogue"] == "eligible_not_selected"


def test_missing_hardware_capability_is_a_refusal():
    registry = load_registry()
    report = registry.select_kernel_family(
        {"op": "rmsnorm", "dtype": "bf16", "shape": {"rows": 32, "cols": 2048}},
        {"features": ["vector_unit", "reciprocal"]},
        _selection(),
    )
    assert report.selected_family is None


def test_fused_stage_families_are_selected_through_compound_ops():
    generator = load_generator()
    rules = {rule["family"]: rule for rule in _selection()["compiler_selection"]["rules"]}
    fused = sorted(name for name, rule in generator.FAMILY_RULES.items() if rule["launch"] == "fused")
    assert fused == ["epilogue", "resnet_patch_gather", "rne_add", "rope"]
    for name in fused:
        assert all("_" in op for op in rules[f"kernels/{name}"]["semantic_ops"])
    text = generator.SELECTION.read_text(encoding="utf-8")
    assert text.count("# fused into projection_gemm;") == 4


def test_materializer_differs_from_radiance_only_in_names():
    def body(path):
        text = (repo_root() / path).read_text(encoding="utf-8")
        return text[text.index("from __future__") :]

    renamed = body(f"{RADIANCE}/materialize.py")
    for old, new in [
        ("radiance_kernel_library_materialization_v1", "atlas_kernel_library_materialization_v1"),
        ("# Radiance PR#1 kernel-library treatment", "# atlas-kernels kernel-library treatment"),
        ("MERLIN_RADIANCE_KERNELS", "MERLIN_ATLAS_KERNELS"),
        ("radiance-kernels checkout", "atlas-kernels checkout"),
        ("targets/radiance/kernel_library_pr1_v1", "targets/atlas/kernel_library_v1"),
    ]:
        assert old in renamed
        renamed = renamed.replace(old, new)
    assert renamed == body(f"{EXAMPLE}/materialize.py")


def test_cli_preserves_default_output_without_running_materialization(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_OUT_ROOT", raising=False)
    module = load_materializer()
    observed = []
    monkeypatch.setattr(module, "materialize", lambda checkout, output: observed.append((checkout, output)) or {})

    def forbidden(*args, **kwargs):
        raise AssertionError("routing test must not execute git or generate kernel payloads")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.chdir(tmp_path)
    assert module.main(["--checkout", str(tmp_path)]) == 0
    assert observed == [(tmp_path, repo_root() / "out/artifacts/targets/atlas/kernel_library_v1")]


def test_cli_reads_checkout_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_ATLAS_KERNELS", str(tmp_path))
    module = load_materializer()
    observed = []
    monkeypatch.setattr(module, "materialize", lambda checkout, output: observed.append(checkout) or {})
    assert module.main([]) == 0
    assert observed == [tmp_path]


def test_cli_requires_a_checkout(monkeypatch):
    monkeypatch.delenv("MERLIN_ATLAS_KERNELS", raising=False)
    module = load_materializer()
    with pytest.raises(SystemExit):
        module.main([])


def test_cli_preserves_explicit_output(tmp_path, monkeypatch):
    module = load_materializer()
    observed = []
    monkeypatch.setattr(module, "materialize", lambda checkout, output: observed.append((checkout, output)) or {})
    output = tmp_path / "selected-output"
    assert module.main(["--checkout", str(tmp_path), "--output", str(output)]) == 0
    assert observed == [(tmp_path, output)]


def test_cli_honors_output_root(tmp_path, monkeypatch):
    module = load_materializer()
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "products"))
    observed = []
    monkeypatch.setattr(module, "materialize", lambda checkout, output: observed.append(output) or {})
    assert module.main(["--checkout", str(tmp_path)]) == 0
    assert observed == [tmp_path / "products/artifacts/targets/atlas/kernel_library_v1"]


def test_cli_does_not_hide_output_symlink(tmp_path, monkeypatch):
    module = load_materializer()
    output = tmp_path / "alias"
    output.symlink_to(tmp_path / "missing", target_is_directory=True)
    observed = []
    monkeypatch.setattr(module, "materialize", lambda checkout, output: observed.append(output) or {})
    assert module.main(["--checkout", str(tmp_path), "--output", str(output)]) == 0
    assert observed == [output]


@pytest.fixture
def synthetic_export(tmp_path, monkeypatch):
    module = load_materializer()
    selection = tmp_path / "selection.yaml"
    selection.write_text(
        yaml.safe_dump(
            {
                "source_commit": "a" * 40,
                "source_pin": "synthetic",
                "shared_paths": ["kernel.py"],
            }
        )
    )
    monkeypatch.setattr(module, "SELECTION", selection)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        member = tarfile.TarInfo("kernel.py")
        member.size = len(b"fixture")
        stream.addfile(member, io.BytesIO(b"fixture"))
    monkeypatch.setattr(module, "_git", lambda *args: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=archive.getvalue()))
    output = tmp_path / "export"
    manifest = module.materialize(tmp_path, output)
    return module, output, manifest


def test_synthetic_manifest_names_the_atlas_schema(synthetic_export):
    _, output, manifest = synthetic_export
    assert manifest["schema"] == "atlas_kernel_library_materialization_v1"
    assert yaml.safe_load((output / "manifest.yaml").read_text()) == manifest
    assert (output / "README.md").read_text().startswith("# atlas-kernels kernel-library treatment")


def test_unchanged_export_is_reused(synthetic_export, tmp_path):
    module, output, manifest = synthetic_export
    before = (output / "kernel.py").stat().st_ino
    assert module.materialize(tmp_path, output) == manifest
    assert (output / "kernel.py").stat().st_ino == before


@pytest.mark.parametrize("member", ["kernel.py", "selection.yaml", "README.md", "manifest.yaml", "extra.txt"])
def test_changed_export_is_refused_without_overwrite(synthetic_export, tmp_path, member):
    module, output, _ = synthetic_export
    (output / member).write_text("changed")
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        module.materialize(tmp_path, output)
    assert (output / member).read_text() == "changed"


def test_missing_export_member_is_refused(synthetic_export, tmp_path):
    module, output, _ = synthetic_export
    (output / "kernel.py").unlink()
    with pytest.raises(FileExistsError):
        module.materialize(tmp_path, output)
    assert not (output / "kernel.py").exists()


@pytest.mark.parametrize("directory", [False, True])
def test_export_symlinks_are_refused(synthetic_export, tmp_path, directory):
    module, output, _ = synthetic_export
    if directory:
        alias = tmp_path / "alias"
        alias.symlink_to(output, target_is_directory=True)
        output = alias
    else:
        payload = output / "kernel.py"
        external = tmp_path / "external.py"
        payload.rename(external)
        payload.symlink_to(external)
    with pytest.raises(ValueError, match="artifact"):
        module.materialize(tmp_path, output)


def _local_checkout() -> Path:
    raw = os.environ.get("MERLIN_ATLAS_KERNELS", "")
    if not raw:
        pytest.skip("MERLIN_ATLAS_KERNELS is not set")
    checkout = Path(raw)
    probe = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", f"{PINNED}^{{commit}}"], capture_output=True, check=False
    )
    if probe.returncode:
        pytest.skip(f"MERLIN_ATLAS_KERNELS does not contain {PINNED}")
    return checkout


def test_local_checkout_exports_every_family(tmp_path):
    checkout = _local_checkout()
    module = load_materializer()
    output = tmp_path / "kernel_library_v1"
    manifest = module.materialize(checkout, output)
    assert manifest["source_commit"] == PINNED
    assert manifest["qualified_families"] == 9
    assert manifest["experimental_families"] == 0
    for family in _selection()["qualified_families"]:
        assert (output / family / "kernel.yaml").is_file()
    assert (output / "lib" / "golden").is_dir()
    assert not (output / "coverage").exists()
    assert module.materialize(checkout, output) == manifest


def test_local_checkout_regenerates_selection_exactly():
    checkout = _local_checkout()
    generator = load_generator()
    assert generator.main(["--checkout", str(checkout), "--check"]) == 0


def test_local_checkout_check_mode_reports_drift(tmp_path):
    checkout = _local_checkout()
    generator = load_generator()
    drifted = tmp_path / "selection.yaml"
    drifted.write_text(generator.SELECTION.read_text(encoding="utf-8").replace("priority: 40", "priority: 41"))
    assert generator.main(["--checkout", str(checkout), "--output", str(drifted), "--check"]) == 1


@pytest.mark.parametrize(
    ("family", "field", "value", "message"),
    [
        ("rmsnorm", "requires_hardware", ["vector_unit", "reciprocal"], "emits vsqrt"),
        ("projection_gemm", "shape", {"M": {"multiple_of": 32}, "K": {"multiple_of": 64}}, "variant"),
        ("rope", "dtypes", ["mxfp8"], "dtype mxfp8"),
        ("epilogue", "semantic_ops", ["silu"], "compound semantic ops"),
    ],
)
def test_local_checkout_refuses_rules_that_disagree_with_the_library(monkeypatch, family, field, value, message):
    checkout = _local_checkout()
    generator = load_generator()
    rules = {name: dict(rule) for name, rule in generator.FAMILY_RULES.items()}
    rules[family][field] = value
    monkeypatch.setattr(generator, "FAMILY_RULES", rules)
    with pytest.raises(generator.SelectionSourceError, match=message):
        generator.collect(checkout, PINNED)
