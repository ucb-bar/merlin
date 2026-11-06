"""Atlas kernel-library example: routing, synthetic artifact integrity, and selection schema.

Mirrors test_radiance_kernel_library_example.py. Nothing here contacts a network or runs hardware. The
local-checkout tests skip only when MERLIN_ATLAS_KERNELS is absent. An explicitly configured checkout
must contain the pinned commit; the tests export from it and regenerate selection.yaml from it.
"""

import ast
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
PINNED = "ce465827a977ba8451b585f71910726e7f12b885"


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
    from merlin.targetgen import kernel_family_selection
    return kernel_family_selection


def _selection():
    return yaml.safe_load(load_materializer().SELECTION.read_bytes())


def test_selection_bytes_and_pinned_semantics_preserved():
    module = load_materializer()
    payload = module.SELECTION.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == "c2a0710640d6f993d5ff1a188bfa92b003c332582918fc760ce462d50ce9a5f2"
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
        ({"op": "rmsnorm", "dtype": "bf16", "shape": {"rows": 128, "cols": 960}}, "kernels/rmsnorm"),
        (
            {
                "op": "attention_gqa_causal",
                "dtype": "bf16",
                "shape": {"Sq": 128, "Sk": 128, "D": 64, "QH": 32, "KVH": 4},
            },
            "kernels/attention",
        ),
        (
            {"op": "attention_gqa", "dtype": "bf16", "shape": {"Sq": 1024, "Sk": 4096, "D": 64, "QH": 12, "KVH": 12}},
            None,
        ),
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


def test_materializer_preserves_radiance_cli_and_git_wrapper():
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
    atlas = body(f"{EXAMPLE}/materialize.py")
    # Export integrity is exercised below. Atlas additionally prunes locally restricted paths.
    for function in ("_git", "main"):
        def source(text):
            module = ast.parse(text)
            node = next(row for row in module.body if isinstance(row, ast.FunctionDef) and row.name == function)
            return ast.dump(node, include_attributes=False)
        assert source(renamed) == source(atlas)


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
    commands = []
    def run(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(stdout=archive.getvalue())
    monkeypatch.setattr(module, "_test_archive_commands", commands, raising=False)
    monkeypatch.setattr(module.subprocess, "run", run)
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
    raw = os.environ.get("MERLIN_ATLAS_KERNELS")
    if raw is None:
        pytest.skip("MERLIN_ATLAS_KERNELS is not set")
    if not raw.strip():
        pytest.fail("MERLIN_ATLAS_KERNELS is explicitly empty")
    checkout = Path(raw)
    if not checkout.is_dir():
        pytest.fail(f"MERLIN_ATLAS_KERNELS checkout does not exist: {checkout}")
    probe = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "-e", f"{PINNED}^{{commit}}"], capture_output=True, check=False
    )
    if probe.returncode:
        pytest.fail(f"MERLIN_ATLAS_KERNELS checkout does not contain pinned commit {PINNED}: {checkout}")
    return checkout


def test_local_checkout_skips_only_when_environment_is_absent(monkeypatch):
    monkeypatch.delenv("MERLIN_ATLAS_KERNELS", raising=False)
    with pytest.raises(pytest.skip.Exception, match="is not set"):
        _local_checkout()


@pytest.mark.parametrize("value", ["", " "])
def test_local_checkout_rejects_explicit_empty_environment(monkeypatch, value):
    monkeypatch.setenv("MERLIN_ATLAS_KERNELS", value)
    with pytest.raises(pytest.fail.Exception, match="explicitly empty"):
        _local_checkout()


def test_local_checkout_rejects_missing_path(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_ATLAS_KERNELS", str(tmp_path / "missing"))
    with pytest.raises(pytest.fail.Exception, match="checkout does not exist"):
        _local_checkout()


def test_local_checkout_rejects_missing_pinned_commit(tmp_path, monkeypatch):
    subprocess.run(["git", "-C", str(tmp_path), "init", "--quiet"], check=True)
    monkeypatch.setenv("MERLIN_ATLAS_KERNELS", str(tmp_path))
    with pytest.raises(pytest.fail.Exception, match=f"does not contain pinned commit {PINNED}"):
        _local_checkout()


@pytest.fixture
def atlas_mlir_checkout(tmp_path):
    """Build a real index and HEAD without needing the linked repository's objects."""

    def create(gitlink_commit=None):
        subprocess.run(["git", "-C", str(tmp_path), "init", "--quiet"], check=True)
        if gitlink_commit is not None:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(tmp_path),
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    f"160000,{gitlink_commit},third_party/atlas-kernels",
                ],
                check=True,
            )
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Nicolas Reed",
                "-c",
                "user.email=reednicolas@berkeley.edu",
                "commit",
                "--quiet",
                "--allow-empty",
                "--author=Nicolas Reed <reednicolas@berkeley.edu>",
                "-m",
                "test(atlas): create pin fixture",
            ],
            check=True,
        )
        return tmp_path

    return create


def test_atlas_mlir_pin_matches_real_gitlink(atlas_mlir_checkout):
    generator = load_generator()
    checkout = atlas_mlir_checkout(PINNED)
    assert generator.check_atlas_mlir_pin(PINNED, checkout) == {
        "status": "checked",
        "source_commit": PINNED,
        "gitlink_commit": PINNED,
    }


def test_atlas_mlir_pin_rejects_mismatched_gitlink(atlas_mlir_checkout):
    generator = load_generator()
    checkout = atlas_mlir_checkout("a" * 40)
    with pytest.raises(generator.SelectionSourceError, match="atlas-mlir"):
        generator.check_atlas_mlir_pin(PINNED, checkout)


def test_atlas_mlir_pin_rejects_absent_gitlink(atlas_mlir_checkout):
    generator = load_generator()
    checkout = atlas_mlir_checkout()
    with pytest.raises(generator.SelectionSourceError, match="atlas-mlir"):
        generator.check_atlas_mlir_pin(PINNED, checkout)


def test_atlas_mlir_pin_rejects_invalid_checkout(tmp_path):
    generator = load_generator()
    with pytest.raises(generator.SelectionSourceError, match="atlas-mlir"):
        generator.check_atlas_mlir_pin(PINNED, tmp_path / "missing")


def test_atlas_mlir_pin_reports_no_checkout():
    assert load_generator().check_atlas_mlir_pin(PINNED, None) == {"status": "not-checked"}


def test_collect_enforces_provided_atlas_mlir_pin(atlas_mlir_checkout, monkeypatch):
    generator = load_generator()
    checkout = atlas_mlir_checkout("a" * 40)
    monkeypatch.setenv("MERLIN_ATLAS_MLIR", str(checkout))
    with pytest.raises(generator.SelectionSourceError, match="differs from selection pin"):
        generator.collect(checkout, PINNED)


def test_local_checkout_exports_every_family(tmp_path):
    checkout = _local_checkout()
    module = load_materializer()
    output = tmp_path / "kernel_library_v1"
    manifest = module.materialize(checkout, output)
    assert manifest["source_commit"] == PINNED
    assert manifest["qualified_families"] == 9
    assert manifest["experimental_families"] == 0
    for family in _selection()["qualified_families"]:
        family_root = output / family
        assert (family_root / "kernel.yaml").is_file()
        recipe = subprocess.run(
            ["make", "--no-print-directory", "-n", "-C", str(family_root), "gen"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "lib/family_interface.py" in recipe
        assert f'--family "{family_root.name}"' in recipe
    assert (output / "kernels/common.mk").is_file()
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
        ("projection_gemm", "semantic_ops", ["nonsense"], "semantic_ops"),
        ("projection_gemm", "requires_hardware", [], "requires_hardware"),
        (
            "resnet_pool",
            "shape",
            {
                "kernel": {"eq": 2},
                "stride": {"eq": 2},
                "padding": {"eq": 1},
                "channels": {"multiple_of": 32},
                "height": {"min": 1},
                "width": {"min": 1},
            },
            "pool geometry",
        ),
        (
            "resnet_pool",
            "shape",
            {
                "kernel": {"eq": 3},
                "stride": {"eq": 2},
                "padding": {"eq": 1},
                "channels": {"multiple_of": 7},
                "height": {"min": 1},
                "width": {"min": 1},
            },
            "pool geometry",
        ),
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


@pytest.mark.parametrize("rows", [32, 128])
@pytest.mark.parametrize("cols", [32, 960, 2048, 4096, 7136])
def test_norm_manifest_width_domain(rows, cols):
    report = load_registry().select_kernel_family(
        {"op": "rmsnorm", "dtype": "bf16", "shape": {"rows": rows, "cols": cols}},
        {"features": MXU + VPU}, _selection(),
    )
    assert report.selected_family == "kernels/rmsnorm"


@pytest.mark.parametrize(
    ("op", "dtype", "shape"),
    [
        ("rmsnorm", "bf16", {"rows": 64, "cols": 2048}),
        ("rmsnorm", "bf16", {"rows": 96, "cols": 2048}),
        ("rmsnorm", "bf16", {"rows": 32, "cols": 7168}),
        ("rmsnorm", "bf16", {"rows": 32, "cols": 7135}),
        ("matmul_rope", "fp8_e4m3", {"M": 64, "K": 2048, "N": 256, "head_dim": 64}),
        ("matmul_rope", "fp8_e4m3", {"M": 96, "K": 2048, "N": 256, "head_dim": 64}),
        ("matmul_rope", "fp8_e4m3", {"M": 32, "K": 2048, "N": 512, "head_dim": 64}),
        ("attention_gqa", "bf16", {"Sq": 128, "Sk": 128, "D": 64, "QH": 15, "KVH": 4}),
        ("attention_gqa", "bf16", {"Sq": 128, "Sk": 128, "D": 64, "QH": 4, "KVH": 8}),
        ("attention_gqa", "bf16", {"Sq": 128, "Sk": 128, "D": 64, "QH": 4}),
        ("attention_gqa", "bf16", {"Sq": 4097, "Sk": 128, "D": 64, "QH": 4, "KVH": 2}),
        ("attention_gqa", "bf16", {"Sq": 128, "Sk": 2049, "D": 64, "QH": 4, "KVH": 2}),
        ("attention_gqa", "bf16", {"Sq": 32, "Sk": 64, "D": 128, "QH": 4, "KVH": 2}),
        ("conv2d_im2col_matmul", "fp8_e4m3", {"channels": 96, "kernel": 3, "stride": 1, "out_channels": 64}),
        ("conv2d_im2col_matmul", "fp8_e4m3", {"channels": 64, "kernel": 2, "stride": 1, "out_channels": 64}),
        ("max_pool2d", "fp8_e4m3", {"kernel": 3, "stride": 2, "padding": 1,
                                    "channels": 96, "height": 112, "width": 112}),
        ("max_pool2d", "fp8_e4m3", {"kernel": 3, "stride": 2, "padding": 1,
                                    "channels": 64, "height": 112, "width": 111}),
    ],
)
def test_catalog_refuses_outside_reviewed_shape_domains(op, dtype, shape):
    report = load_registry().select_kernel_family(
        {"op": op, "dtype": dtype, "shape": shape}, {"features": MXU + VPU}, _selection(),
    )
    assert report.selected_family is None


@pytest.mark.parametrize(
    ("op", "shape", "family"),
    [
        ("max_pool2d", {"kernel": 3, "stride": 2, "padding": 1, "channels": 64,
                        "height": 112, "width": 112}, "resnet_pool"),
        ("conv2d_im2col_matmul", {"channels": 64, "kernel": 3, "stride": 2,
                                 "out_channels": 128}, "resnet_patch_gather"),
    ],
)
def test_pool_and_patch_reviewed_shapes(op, shape, family):
    report = load_registry().select_kernel_family(
        {"op": op, "dtype": "fp8_e4m3", "shape": shape}, {"features": MXU + VPU}, _selection(),
    )
    assert report.selected_family == f"kernels/{family}"


def test_local_pin_matches_current_bringup_and_public_tracking_branch():
    checkout = _local_checkout()
    runner = repo_root() / "examples/atlas/model_bringup/qwen/full_model/run.py"
    module = ast.parse(runner.read_text())
    current = next(ast.literal_eval(row.value) for row in module.body
                   if isinstance(row, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "KERNELS" for t in row.targets))
    assert current == PINNED
    subprocess.run(["git", "-C", str(checkout), "merge-base", "--is-ancestor",
                    PINNED, "origin/feat/kernel-library"], check=True)


def test_local_manifest_drift_requires_review():
    generator = load_generator()
    data = generator.collect(_local_checkout(), PINNED)
    data["manifests"]["rmsnorm"]["constraints"]["rows"] = "32 through 128"
    with pytest.raises(generator.SelectionSourceError, match="unreviewed row/width"):
        generator.manifest_shapes(data["manifests"])


def test_archive_applies_local_source_access_exclusions(synthetic_export):
    module, _, _ = synthetic_export
    assert module._test_archive_commands[0][-2:] == [
        ":(exclude,icase)*hammer*", ":(exclude,icase)*vlsi*",
    ]


def test_digest_prunes_before_inspecting_restricted_members(tmp_path, monkeypatch):
    module = load_materializer()
    # Mock names only: no restricted file or directory is created or inspected.
    def walk(*args, **kwargs):
        dirs = ["restricted-hammer-directory", "safe"]
        yield str(tmp_path), dirs, []
        raise AssertionError("digest must reject before traversal resumes")
    monkeypatch.setattr(module.os, "walk", walk)
    with pytest.raises(ValueError, match="restricted member"):
        module._payload_digest(tmp_path)


def test_catalog_remains_disabled_authoring_evidence():
    selection = _selection()
    assert "authoring evidence" in selection["purpose"]
    assert tuple(selection["prohibited_uses"]) == load_generator().PROHIBITED_USES
    descriptor = yaml.safe_load((repo_root() / "examples/atlas/target/descriptor.yaml").read_text())
    assert not descriptor.get("information_sets")


@pytest.mark.parametrize(("sq", "sk", "selected"), [(1024, 2048, True), (1025, 2048, False),
                                                    (4096, 512, True), (4096, 513, False)])
def test_attention_descriptor_capacity(sq, sk, selected):
    report = load_registry().select_kernel_family(
        {"op": "attention_gqa", "dtype": "bf16", "shape": {"Sq": sq, "Sk": sk, "D": 64, "QH": 4, "KVH": 2}},
        {"features": MXU + VPU}, _selection(),
    )
    assert (report.selected_family == "kernels/attention") == selected


def test_pool_requires_mxu_capabilities():
    report = load_registry().select_kernel_family(
        {"op": "max_pool2d", "dtype": "fp8_e4m3", "shape": {"kernel": 3, "stride": 2, "padding": 1,
         "channels": 64, "height": 112, "width": 112}}, {"features": VPU}, _selection(),
    )
    assert report.selected_family is None
