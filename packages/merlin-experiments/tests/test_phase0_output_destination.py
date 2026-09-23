"""Generation cannot implicitly overwrite the descriptor's source corpus."""

import pytest
from merlin_experiments.phase0 import __main__ as cli
from merlin_experiments.phase0 import generation


def test_api_requires_destination_before_loading_or_mutating_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("input resolution must not start without an explicit output")

    monkeypatch.setattr(generation, "_descriptor_for", forbidden)
    monkeypatch.setattr(generation, "_ensure_contract_on_path", forbidden)
    with pytest.raises(ValueError, match="explicit output_root"):
        generation.generate_target("fixture")


def test_cli_requires_destination_before_generation(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("generation must not start without an explicit output")

    monkeypatch.setattr(cli, "generate_target", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--target", "fixture"])
    assert raised.value.code == 2
    assert "--output-root" in capsys.readouterr().err


def test_cli_forwards_explicit_artifact_destination(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "generate_target", lambda target, **options: calls.append((target, options)) or [])
    profiles, descriptor, output = tmp_path / "recipes", tmp_path / "target.yaml", tmp_path / "artifacts"
    assert (
        cli.main(
            [
                "--target",
                "fixture",
                "--profiles-root",
                str(profiles),
                "--descriptor",
                str(descriptor),
                "--output-root",
                str(output),
            ]
        )
        == 0
    )
    assert calls == [("fixture", {"profiles_root": profiles, "descriptor": descriptor, "output_root": output})]
    assert not output.exists()


def test_cli_refuses_implicit_recipe_discovery_before_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "generate_target", lambda *a, **k: pytest.fail("implicit input reached generator"))
    with pytest.raises(SystemExit) as raised:
        cli.main(["--target", "fixture", "--output-root", str(tmp_path / "output")])
    assert raised.value.code == 2
    assert not (tmp_path / "output").exists()


def test_api_refuses_implicit_inputs_before_descriptor_environment_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(generation, "_descriptor_for", lambda *a: pytest.fail("descriptor discovery ran"))
    monkeypatch.setattr(generation, "_ensure_contract_on_path", lambda *a: pytest.fail("provider setup ran"))
    with pytest.raises(ValueError, match="explicit recipe inputs"):
        generation.generate_target("fixture", output_root=tmp_path / "output")
    assert not (tmp_path / "output").exists()
