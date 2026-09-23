"""The transcript audit's answer tokens must not match the contract the agent is required to read.

A reference backend evicted to its own package lives in a directory literally named ``backend``, so
tokenising the route by bare stem puts the generic word "backend" on the answer list. It then
substring-matches granted, public paths — ``merlin/contract/mlir_oot_backend_contract.yaml`` above all,
which every OOT arm must read — and the run reports ``answer_access_clean: false`` for doing exactly what
it was told to do. That is worse than no audit: a signal that cries wolf on the contract makes the next
real leak read as more noise. Tokens are therefore path-qualified (``gemmini/backend``), which still
matches a genuine read of the evicted route.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox.answer_surfaces import audit_tokens
from merlin.targetgen.target_experiment import load_target_experiment

DESCRIPTOR = repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"

# Paths a compliant agent reads. None may trip the answer audit.
GRANTED = [
    "merlin/contract/mlir_oot_backend_contract.yaml",
    "merlin/contract/target_dialect_contract.yaml",
    "merlin/contract/command_buffer_abi.yaml",
    "merlin/contract/interface_grammar.md",
    "merlin/contract/schemas/command_buffer.schema.json",
    "merlin/contract/capsules/profiles/gemmini.yaml",
    "examples/gemmini/phase0/recipe.yaml",
]

# Real answer surfaces. Each MUST trip it.
ANSWERS = [
    "merlin/contract/capsules/isa/A2_single_tile_matmul/golden.yaml",
    "merlin/contract/capsules/isa/A2_single_tile_matmul/expected_instruction_coverage.yaml",
    "merlin/contract/capsules/model/M2_microvit_gemmini/capsule.weights.safetensors",
    # A holdout path. The token under test is the trailing "capsules/hidden" pair, so the capsule
    # name is immaterial to the assertion — and naming a real one would publish a holdout.
    "merlin/contract/capsules/hidden/PLACEHOLDER_hidden/capsule.yaml",
    "merlin/python/merlin/runtime/reference.py",
    "merlin/targets/gemmini/backend/gemmini.py",
]


@pytest.fixture(scope="module")
def tokens():
    return audit_tokens(load_target_experiment(DESCRIPTOR))["answer"]


def test_no_answer_token_is_a_bare_generic_word(tokens):
    """A single generic path component ("backend", "sim_oracle") is too coarse to be a token."""
    generic = {"backend", "backends", "sim_oracle", "oracle", "contract", "target", "targets"}
    bad = sorted(t for t in tokens if t in generic)
    assert not bad, f"answer tokens {bad} are bare generic words and will match granted paths"


@pytest.mark.parametrize("path", GRANTED)
def test_granted_contract_paths_do_not_trip_the_audit(path, tokens):
    hit = [t for t in tokens if t in path]
    assert not hit, f"granted path {path} matched answer token(s) {hit}"


def test_public_weight_filename_declaration_does_not_trip_the_audit(tokens):
    """Capsule YAML/MLIR must name the private file without that public declaration reading its bytes."""
    declaration = 'prov.weights_file = "capsule.weights.safetensors"'
    assert not [token for token in tokens if token in declaration]


@pytest.mark.parametrize("path", ANSWERS)
def test_real_answer_surfaces_still_trip_the_audit(path, tokens):
    fixture = path.split("/hidden/", 1)[0] + "/hidden" if "/hidden/" in path else path
    if (".safetensors" in path or "/hidden/" in path) and not (repo_root() / fixture).exists():
        pytest.skip("private corpus fixture absent; synthetic private surfaces are tested independently below")
    assert any(t in path for t in tokens), f"answer surface {path} matched NO token — the audit is blind"


def test_private_resource_tokens_are_nonvacuous_without_real_answer_data(tmp_path, monkeypatch):
    surfaces = importlib.import_module("merlin.targetgen.sandbox.answer_surfaces")
    monkeypatch.setattr(surfaces, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(surfaces, "artifacts_dir", lambda: tmp_path / "out/artifacts")
    monkeypatch.setattr(surfaces, "_evicted_oracle_modules", lambda: [])
    monkeypatch.setattr(surfaces, "_support_package_dirs", lambda: [])
    hidden = "merlin/contract/capsules/hidden/"
    te = SimpleNamespace(hidden_corpus=lambda: hidden, prior_backends=(), backend_package=None)
    # Declared identity remains suspicious even before the optional private files exist locally.
    before = surfaces.audit_tokens(te)["answer"]
    assert "capsules/hidden" in before
    assert "merlin/runtime/reference" in before
    private = tmp_path / "merlin/contract/capsules/model/synthetic/capsule.weights.safetensors"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"synthetic fixture, not model weights")
    after = surfaces.audit_tokens(te)["answer"]
    assert any(token in str(private) for token in after)
    assert not any(token in 'weights_file = "capsule.weights.safetensors"' for token in after)


def test_renamed_explicit_support_oracle_tokens_follow_current_owner(tmp_path, monkeypatch):
    """Current audit tokens follow the selected package, not its retired in-tree spelling."""
    import subprocess

    from merlin.common.yaml import write_yaml

    root = tmp_path / "unrelated-provider-directory"
    write_yaml(
        root / "contracts/target_contract.yaml",
        {"name": "synthetic_device", "plugin": {"sim_oracle": "sim_oracle.py"}},
    )
    oracle = root / "sim_oracle.py"
    oracle.write_text("raise AssertionError('metadata audit must not execute oracle')\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(tmp_path / "empty-references"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("token discovery launched a process"))
    descriptor = SimpleNamespace(
        target="synthetic_device", hidden_corpus=lambda: None, prior_backends=(), backend_package=None
    )
    selected = audit_tokens(descriptor)["answer"]
    assert any(token in str(oracle) for token in selected)
    assert not {"backend", "sim_oracle", "oracle", "contract", "target", "targets"}.intersection(selected)
    assert not any(token in path for path in GRANTED for token in selected)
