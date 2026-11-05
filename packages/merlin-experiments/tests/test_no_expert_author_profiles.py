"""Fresh EL profiles remove expert material without changing authoring capabilities."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import author_profiles as A


@pytest.mark.parametrize("level", (1, 2, 3))
def test_no_expert_profiles_preserve_the_original_tooling_level(level):
    historical = A.resolve(f"xs-el-v1/el{level}")
    clean = A.resolve(f"xs-el-no-expert-v1/el{level}")
    assert historical.raw_kernel_references is True
    assert clean.raw_kernel_references is False
    assert (clean.level, clean.entries, clean.tools, clean.modern) == (
        historical.level, historical.entries, historical.tools, historical.modern)
    assert clean.record()["raw_kernel_references"] is False
    assert clean.record()["live_cca"] is False
    assert "raw_kernel_references" not in historical.record()
    assert "Expert kernel sources, ELFs, disassemblies" in A.task(clean)
    assert "reference evidence" not in A.task(clean)
    assert "numerical/ISA/timing judge" in A.task(clean)
    options = SimpleNamespace(author_profile=clean.name, sandbox="bwrap",
        compiler_execution="snapshot_external_v1", fresh_backend=True,
        seed_submission="", qualify_submission="", joint_epoch="", with_tool=[], without_tool=[],
        compiler_access="selected_library" if level==1 else "full_compiler", full_compiler_root="",
        reference_experiments="", reference_experiments_tasks="off")
    assert A.validate_options(options) == clean
    options.reference_experiments = "old-expert-capture.json"
    with pytest.raises(ValueError, match="reference analysis"):
        A.validate_options(options)


@pytest.mark.parametrize("leaf", ("reference-kernels", "public-references", "REFERENCE_MAP.json",
                                   "reference-experiments", "performance-baselines", "expert-targets"))
def test_expert_containers_cannot_be_granted(tmp_path, leaf):
    with pytest.raises(ValueError, match="reference material"):
        A._verify_no_expert_grants({"allowed": [{"path": str(tmp_path/leaf)}]})


def test_renamed_alias_resolved_symlink_and_broad_parent_cannot_reexpose_bank(tmp_path):
    bank = tmp_path/"reference-kernels"
    bank.mkdir()
    (bank/"reference-manifest.json").write_text("{}")
    alias = tmp_path/"ordinary-examples"
    alias.symlink_to(bank, target_is_directory=True)
    for row in ({"path": str(tmp_path/"renamed"), "as": "reference-kernels"},
                {"path": str(alias)}):
        with pytest.raises(ValueError, match="reference material"):
            A._verify_no_expert_grants({"allowed": [row]})
    with pytest.raises(ValueError, match="ancestor grant"):
        A._verify_no_expert_grants({"allowed": [{"path": str(tmp_path)}],
                                   "host_inputs": [{"path": str(bank)}]})


def test_regular_public_data_survives_a_no_reference_run_name(tmp_path):
    path = tmp_path/"no-reference-kernels-study"/"payload/corpus/isa"
    assert not A.expert_reference_grant({"path": str(path), "note": "capsule corpus"})
    A._verify_no_expert_grants({"allowed": [{"path": str(path)}]})


def test_renamed_directory_still_carries_the_origin_manifest(tmp_path):
    bank = tmp_path/"innocent"
    bank.mkdir()
    (bank/"reference-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="reference material"):
        A._verify_no_expert_grants({"allowed": [{"path": str(bank)}]})


def test_clean_release_replaces_inherited_cca_addendum(tmp_path, monkeypatch):
    import yaml

    directory = tmp_path / "input_bundles/public_v0"
    directory.mkdir(parents=True)
    manifest = directory / "input_bundle_manifest.yaml"
    manifest.write_text(yaml.safe_dump({"bundle_id": "public_v0", "allowed": [], "denied": []}))
    (directory / "TASK_ADDENDUM.md").write_text("Read reference-kernels/expert.c and use CCA.")
    monkeypatch.setattr(A, "canonical_common", lambda *args: [{"path": "/public/isa"}])
    # Bundle admission has separate negative-container tests; this exercises the
    # source-release document writer, including an actual stale inherited file.
    monkeypatch.setattr(A, "materialize_bundle", lambda bundle, **kwargs: bundle)
    A.stage_release_bundles(tmp_path, None, profile=A.NO_EXPERT_NAMES[0],
                            source_selection=None, common_inputs="canonical_common")
    text = (directory / "TASK_ADDENDUM.md").read_text()
    assert text == A.task(A.resolve(A.NO_EXPERT_NAMES[0]))
    assert "Read reference-kernels/expert.c" not in text
    assert "Expert kernel sources, ELFs, disassemblies" in text
