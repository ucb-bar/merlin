"""Prepared paired CLI restores original owners; these fixtures issue no timing."""

import json

import pytest
from merlin_experiments.phase2 import paired_cli as CLI
from merlin_experiments.phase2.compiler_pass_preparation import CompilerPassPreparation
from test_compiler_pass_state import original

from merlin.common.digest import sha256_file


@pytest.fixture
def prepared_arguments(tmp_path, monkeypatch):
    inputs, passes = original.__wrapped__(tmp_path)
    library = inputs["library"]
    contract = tmp_path / "reviewed.json"
    contract.write_text(json.dumps(library.contract.record()))
    preparation = CompilerPassPreparation(
        "configuration_cleanup", baseline_policy={"enabled": False}, candidate_policy={"enabled": True}
    )
    policy = tmp_path / "original-policies.json"
    policy.write_text(json.dumps(preparation.record()))
    arguments = []
    for name in (
        "source-root",
        "functional-runs-root",
        "measurement-root",
        "contract-root",
        "candidate-record",
        "descriptor",
        "corpus-root",
        "corpus-manifest",
        "gsim-certificate",
        "rtl-facts",
    ):
        arguments += ["--" + name, str(tmp_path / name)]
    for name in (
        "functional-submission-sha256",
        "corpus-manifest-sha256",
        "corpus-capsules-sha256",
        "gsim-certificate-sha256",
    ):
        arguments += ["--" + name, "a" * 64]
    for name, value in (
        ("functional-run-id", "base"),
        ("run-id", "pair"),
        ("phase", "tuning"),
        ("compiler-library-root", library.root),
        ("compiler-library-contract", contract),
        ("compiler-library-contract-sha256", library.contract.sha256),
        ("compiler-pass-selection", inputs["path"]),
        ("compiler-pass-selection-sha256", inputs["sha256"]),
        ("compiler-pass-admission-root", inputs["work_root"]),
        ("compiler-pass-preparation", policy),
        ("compiler-pass-preparation-sha256", sha256_file(policy)),
        ("compiler-preparation-verification-root", tmp_path / "fresh-verification"),
    ):
        arguments += ["--" + name, str(value)]
    monkeypatch.setattr(CLI, "load_target_experiment", lambda *a, **kw: object())
    return arguments, inputs, passes, preparation


def test_prepared_pair_reopens_original_selection_before_independent_final_consumer(prepared_arguments, monkeypatch):
    arguments, inputs, passes, preparation = prepared_arguments

    def consume(*args, **kwargs):
        assert kwargs["compiler_pass_selection"].record() == passes.record()
        assert kwargs["compiler_pass_preparation"].record() == preparation.record()
        assert kwargs["compiler_library_selection"].record() == inputs["library"].record()
        assert kwargs["compiler_preparation_verification_root"] == inputs["path"].parent / "fresh-verification"
        raise RuntimeError("independent final consumer reached")

    monkeypatch.setattr(CLI.PI, "load_paired_inputs", consume)
    with pytest.raises(RuntimeError, match="independent final consumer reached"):
        CLI.main(arguments)
    assert (inputs["work_root"] / "original-check-ran").is_file()
    assert not (inputs["path"].parent / "measurement-root").exists()


@pytest.mark.parametrize("damage", ["missing", "existing", "relative", "library_child", "campaign_child", "alias"])
def test_invalid_replay_destination_refuses_before_original_factory(prepared_arguments, monkeypatch, damage):
    arguments, inputs, _, _ = prepared_arguments
    index = arguments.index("--compiler-preparation-verification-root")
    root = inputs["path"].parent
    if damage == "missing":
        del arguments[index : index + 2]
    elif damage == "existing":
        (root / "fresh-verification").mkdir()
    elif damage == "relative":
        arguments[index + 1] = "relative-verification"
    elif damage == "library_child":
        arguments[index + 1] = str(inputs["library"].root / "verification")
    elif damage == "campaign_child":
        arguments[index + 1] = str(root / "measurement-root/pair/verification")
    else:
        (root / "canonical").mkdir()
        (root / "alias").symlink_to(root / "canonical", target_is_directory=True)
        arguments[index + 1] = str(root / "alias/verification")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid replay destination reached an original factory")

    monkeypatch.setattr(CLI.PI, "load_paired_inputs", forbidden)
    from merlin_experiments.phase2 import compiler_pass_state as PASSES

    monkeypatch.setattr(PASSES, "restore_original_selection", forbidden)
    with pytest.raises(CLI.PC.CampaignGateError):
        CLI.main(arguments)
    assert not inputs["work_root"].exists()
