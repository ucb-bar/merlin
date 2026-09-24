"""Explicit Phase0 inputs never acquire ambient public or private siblings."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import __main__ as cli
from merlin_experiments.phase0 import generation, profiles


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document))
    return path


@pytest.fixture
def inputs(tmp_path):
    return {
        "recipe": _write(tmp_path / "public/profile.yaml", {"capsules": [{"name": "public"}]}),
        "performance_template": _write(tmp_path / "shared/performance.yaml", {"sweeps": []}),
        "synth_profile": _write(tmp_path / "generated/synth.yaml", {"capsules": [{"name": "synth"}]}),
        "smt_profile": _write(tmp_path / "generated/smt.yaml", {"capsules": [{"name": "smt"}]}),
        "hidden_profile": _write(
            tmp_path / "private/holdout.yaml", {"capsules": [{"name": "hidden"}], "sweeps": [{"id": "hidden_sweep"}]}
        ),
    }


def test_explicit_merge_order_and_template_identity(inputs, monkeypatch):
    observed = []
    original = Path.read_text

    def read(path, *args, **kwargs):
        observed.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(profiles, "_profiles_root", lambda *args: pytest.fail("ambient profile discovery"))
    result = profiles.load_profile("unrelated-selector", **inputs)
    assert observed == list(inputs.values())
    assert [row["name"] for row in result["capsules"]] == ["public", "synth", "smt", "hidden"]
    assert result["capsules"][0]["label"] == "public"
    assert all("label" not in row for row in result["capsules"][1:])
    assert result["sweeps"] == [{"id": "hidden_sweep"}]
    assert result["_performance_template"]["path"] == str(inputs["performance_template"])
    assert result["_performance_template"]["sha256"] == profiles._document_digest({"sweeps": []})


def test_public_load_does_not_read_or_stat_private_path(inputs, monkeypatch):
    original_stat, original_read = Path.stat, Path.read_text

    def stat(path, *args, **kwargs):
        assert path != inputs["hidden_profile"], "public load statted private input"
        return original_stat(path, *args, **kwargs)

    def read(path, *args, **kwargs):
        assert path != inputs["hidden_profile"], "public load read private input"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "read_text", read)
    result = profiles.load_profile("fixture", include_holdouts=False, **inputs)
    assert [row["name"] for row in result["capsules"]] == ["public", "synth", "smt"]
    assert result["sweeps"] == []


def test_omitted_sidecars_do_not_discover_siblings(inputs):
    for suffix in ("synth", "smt", "hidden"):
        _write(inputs["recipe"].parent / f"fixture.{suffix}.yaml", {"capsules": [{"name": "ambient"}]})
    result = profiles.load_profile(
        "fixture", recipe=inputs["recipe"], performance_template=inputs["performance_template"]
    )
    assert [row["name"] for row in result["capsules"]] == ["public"]


def test_explicit_missing_optional_sidecars_are_omitted(inputs, tmp_path):
    for name in ("synth_profile", "smt_profile", "hidden_profile"):
        inputs[name] = tmp_path / "absent" / name
    assert [row["name"] for row in profiles.load_profile("fixture", **inputs)["capsules"]] == ["public"]


@pytest.mark.parametrize("name", ["synth_profile", "smt_profile", "hidden_profile"])
def test_optional_directory_is_not_misreported_as_absence(inputs, tmp_path, name):
    inputs[name] = tmp_path
    with pytest.raises(ValueError, match="optional profile is not a file"):
        profiles.load_profile("fixture", **inputs)


@pytest.mark.parametrize("name", ["synth_profile", "smt_profile", "hidden_profile"])
def test_broken_optional_alias_is_not_misreported_as_absence(inputs, tmp_path, name):
    alias = tmp_path / "broken.yaml"
    alias.symlink_to(tmp_path / "missing.yaml")
    inputs[name] = alias
    with pytest.raises(ValueError, match="optional profile is not a file"):
        profiles.load_profile("fixture", **inputs)


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"recipe": "public.yaml"}, "requires performance_template"),
        (
            {"recipe": "public.yaml", "performance_template": "shared.yaml", "profiles_root": "old"},
            "mutually exclusive",
        ),
        ({"performance_template": "shared.yaml"}, "require recipe"),
        ({"synth_profile": "synth.yaml"}, "require recipe"),
        ({"smt_profile": "smt.yaml"}, "require recipe"),
        ({"hidden_profile": "private.yaml"}, "require recipe"),
    ],
)
def test_invalid_input_modes_refuse_before_read_or_generation(options, reason, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: pytest.fail("invalid mode read input"))
    monkeypatch.setattr(generation, "_ensure_contract_on_path", lambda *args: pytest.fail("invalid mode mutated setup"))
    with pytest.raises(ValueError, match=reason):
        profiles.load_profile("fixture", **options)
    with pytest.raises(ValueError, match=reason):
        generation.generate_target("fixture", descriptor="target.yaml", output_root=tmp_path / "out", **options)


def test_explicit_template_validation_is_not_bypassed(inputs):
    _write(inputs["performance_template"], {"capsules": [{"name": "hand_written_performance"}]})
    with pytest.raises(ValueError, match="must generate entries through `sweeps`"):
        profiles.load_profile("fixture", **inputs)


def test_cli_forwards_all_explicit_inputs_without_legacy_discovery(inputs, tmp_path, monkeypatch):
    from merlin.common import paths

    monkeypatch.setattr(paths, "checkout_root", lambda: None)
    calls = []
    monkeypatch.setattr(cli, "generate_target", lambda target, **kwargs: calls.append((target, kwargs)) or [])
    monkeypatch.setattr(cli, "profile_targets", lambda **kwargs: pytest.fail("explicit recipe enumerated targets"))
    descriptor, output = tmp_path / "target.yaml", tmp_path / "output"
    argv = ["--target", "fixture", "--descriptor", str(descriptor), "--output-root", str(output)]
    for name, path in inputs.items():
        argv.extend(["--" + name.replace("_", "-"), str(path)])
    assert cli.main(argv) == 0
    assert calls == [("fixture", {"descriptor": descriptor, "output_root": output, **inputs})]
    assert not output.exists()


def test_generation_forwards_paths_before_numerical_work(inputs, tmp_path, monkeypatch):
    class InputsReached(Exception):
        pass

    monkeypatch.setattr(generation, "_ensure_contract_on_path", lambda *args: None)
    monkeypatch.setattr(generation, "load_target_experiment", lambda *args: SimpleNamespace(target="hardware"))

    def load(target, **kwargs):
        assert target == "profile-id"
        assert kwargs == inputs
        raise InputsReached

    monkeypatch.setattr(generation, "load_profile", load)
    with pytest.raises(InputsReached):
        generation.generate_target(
            "profile-id", descriptor=tmp_path / "target.yaml", output_root=tmp_path / "output", **inputs
        )
    assert not (tmp_path / "output").exists()


def test_single_explicit_recipe_cannot_discover_comparison_roster(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "generate_target", lambda *args, **kwargs: pytest.fail("must refuse before generation"))
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "--target",
                "fixture",
                "--output-root",
                str(tmp_path / "output"),
                "--recipe",
                str(inputs["recipe"]),
                "--performance-template",
                str(inputs["performance_template"]),
                "--comparison-manifest",
            ]
        )
    assert raised.value.code == 2


def test_cli_does_not_accept_abbreviated_input_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "generate_target", lambda *args, **kwargs: pytest.fail("abbreviated flag executed"))
    with pytest.raises(SystemExit) as raised:
        cli.main(["--target", "fixture", "--output-root", str(tmp_path / "output"), "--rec", "ambiguous.yaml"])
    assert raised.value.code == 2


def test_resident_reuse_nested_extent_reaches_capsule_builder_without_freezing_tile_size():
    from merlin_experiments.phase0.sweeps import _resolve_flat_extents

    from merlin.targetgen.corpus_spec import CorpusBinding, build_resident_reuse

    entry = {
        "name": "carried_reuse",
        "kind": "program",
        "source_role": "derived_sweep",
        "source_reference": "carried-state axis",
        "op": "resident_reuse",
        "K": "2*tile",
        "N": "tile",
        "matmuls": [
            {"lhs": "A0", "out": "O0", "M": "tile-1", "epilogue": []},
            {"lhs": "A1", "out": "O1", "M": "2*tile", "epilogue": ["relu"]},
        ],
    }
    for edge in (16, 32):
        binding = CorpusBinding(
            target="synthetic",
            tile_dim=edge,
            operand_dtype="int8",
            accum_dtype="i32",
            integer=True,
            tiers=["L1"],
            compare="exact_int",
        )
        resolved = _resolve_flat_extents(entry, binding)
        capsule, mlir = build_resident_reuse(resolved, binding)
        assert capsule["inputs"][0]["shape"] == [2 * edge, edge]
        assert [row["shape"] for row in capsule["inputs"][1:]] == [
            [edge - 1, 2 * edge],
            [2 * edge, 2 * edge],
        ]
        assert f"tensor<{edge - 1}x{2 * edge}xi8>" in mlir
        assert f"tensor<{2 * edge}x{2 * edge}xi8>" in mlir
    assert entry["matmuls"][0]["M"] == "tile-1"
