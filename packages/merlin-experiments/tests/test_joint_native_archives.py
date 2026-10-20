"""Neutral original-owner transport checks, never actual target qualification."""

import copy
import json
from dataclasses import replace

import pytest
from merlin_experiments.phase2 import archive_evidence as J
from merlin_experiments.phase2 import archive_screen_review as R
from merlin_experiments.phase2 import archive_screen_selection as S
from merlin_experiments.phase2 import public_native_exploration as N
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from merlin_timing import archive as A
from merlin_timing import contracts as C
from test_ordinary_native_archive_screen import native_source
from test_ordinary_native_archive_screen import ordinary_inputs as ordinary_inputs
from test_ordinary_native_archive_screen import source as source


def variant(document):
    result = copy.deepcopy(document)
    cases = {}
    for case in result["cases"]:
        previous = case["case_id"]
        case["case_id"] = "new-" + previous
        case["elf_sha256"] = C.digest(["new", previous])
        cases[previous] = case
    for row in result["records"]:
        case = cases[row["case_id"]]
        row["record_id"], row["case_id"] = "new-" + row["record_id"], case["case_id"]
        for proof in (row["feature_provenance"], row["label"]):
            proof["execution_sha256"] = C.execution_identity(case)
        row["event_report"]["report"]["elf_sha256"] = case["elf_sha256"]
        row["event_report"]["sha256"] = C.digest(row["event_report"]["report"])
    return C.dataset(result)


@pytest.fixture
def joint(source, tmp_path, monkeypatch):
    selected = S.NativeArchiveScreenSelection(**native_source(source))
    second = variant(source["dataset"])
    files = [tmp_path / name for name in ("extra.tar", "extra-binding.json", "extra-reports.json")]
    files[0].write_text("second neutral original archive")
    files[1].write_bytes(source["binding_path"].read_bytes())
    files[2].write_text(json.dumps({"reports": {}}))
    additional = J.AdditionalArchiveEvidence(files[0], sha256_file(files[0]), files[1], files[2])
    selected = replace(selected, additional_archives=(additional,))

    def inspect(path, digest):
        assert sha256_file(path) == digest
        return {
            "export_scope": "training_eligible_public",
            "dataset": second if path == files[0] else source["dataset"],
            "records": [],
        }

    monkeypatch.setattr(A, "inspect_archive", inspect)
    monkeypatch.setattr(A, "prepare", lambda inventory, binding: {"observations": [], "dataset": inventory["dataset"]})
    monkeypatch.setattr(A, "attach", lambda plan, observations: copy.deepcopy(plan["dataset"]))
    monkeypatch.setattr(
        N.HISTORY,
        "prepare",
        lambda **kw: {
            "schema": N.HISTORY.SCHEMA,
            "archive_sha256": kw["archive_sha256"],
            "sources": {},
            "cases": [{"case": c, "records": []} for c in kw["dataset"]["cases"]],
        },
    )
    return selected, files, second


def test_explicit_union_reopens_both_original_datasets_and_keeps_default_formal_gate(joint):
    selected, _, second = joint
    reviewed = selected.preflight()
    assert reviewed["report"]["qualified"] and reviewed["report"]["metrics"]["declared_held_cases"] == 48
    assert len(reviewed["archive_components"]) == 2
    assert reviewed["dataset"]["cases"][-len(second["cases"]) :] == second["cases"]
    with pytest.raises(StageGateError, match="public diagnostic"):
        selected.install(None)


def test_original_native_owner_pins_every_additional_file_and_declaration(joint):
    selected, files, _ = joint
    owner = N.prepare_public_native_exploration(selected)
    assert owner.record()["schema"] == "merlin.public_native_exploration.v4"
    assert owner.record()["additional_reviewed_archive_sha256s"] == [selected.additional_archives[0].archive_sha256]
    assert owner.record()["measured_cycles"] is None and owner.record()["promotion_authority"] == "NONE"
    files[2].write_text("changed original report")
    with pytest.raises(StageGateError, match="evidence changed"):
        owner.verify()


def test_original_native_owner_refuses_rebound_selected_union(joint):
    selected, _, _ = joint
    owner = N.prepare_public_native_exploration(selected)
    object.__setattr__(selected, "additional_archives", ())
    with pytest.raises(StageGateError, match="issued owners"):
        owner.verify()


def test_changed_inputs_during_original_catalog_preparation_cannot_be_adopted(joint, monkeypatch):
    selected, files, _ = joint
    original = N.HISTORY.prepare

    def changed(**kwargs):
        result = original(**kwargs)
        files[2].write_text("changed while original catalog is being prepared")
        return result

    monkeypatch.setattr(N.HISTORY, "prepare", changed)
    with pytest.raises(StageGateError, match="input changed"):
        N.prepare_public_native_exploration(selected)


def test_joint_function_replacement_is_not_saved_qualification(joint, monkeypatch):
    selected, _, _ = joint
    owner = N.prepare_public_native_exploration(selected)
    monkeypatch.setattr(J.UNION, "join", lambda *_: {"qualified": True})
    with pytest.raises(StageGateError, match="issued owners"):
        owner.verify()


@pytest.mark.parametrize("damage", ["domain", "timing", "source_split", "duplicate_execution", "bad_native"])
def test_union_cannot_mask_foreign_or_incompatible_original_evidence(joint, damage):
    selected, files, second = joint
    if damage in {"domain", "timing"}:
        binding = json.loads(files[1].read_text())
        if damage == "domain":
            binding["domain"]["device_sha256"] = C.digest("other hardware")
        else:
            binding["review"]["timing_selection"] = {"copied": True}
        files[1].write_text(json.dumps(binding))
    elif damage == "source_split":
        group = second["cases"][0]["source_group_sha256"]
        for case in second["cases"]:
            if case["source_group_sha256"] == group:
                case["split"] = "calibration"
    elif damage == "duplicate_execution":
        second["cases"][0]["elf_sha256"] = C.digest("0-0")
        for row in second["records"][:1]:
            case = second["cases"][0]
            row["label"]["execution_sha256"] = C.execution_identity(case)
            row["feature_provenance"]["execution_sha256"] = C.execution_identity(case)
    else:
        row = second["records"][-1]
        row["event_report"]["report"]["model_sha256"] = C.digest("other model")
        row["event_report"]["sha256"] = C.digest(row["event_report"]["report"])
    with pytest.raises((C.ContractError, StageGateError)):
        selected.preflight()


def test_additional_declarations_never_accept_saved_owner_json_or_symlinks(tmp_path):
    with pytest.raises(StageGateError):
        J.declarations(({"qualified": True},))
    original = tmp_path / "original"
    original.write_text("neutral")
    link = tmp_path / "linked"
    link.symlink_to(original)
    with pytest.raises(StageGateError, match="canonical"):
        J.declarations((J.AdditionalArchiveEvidence(link, sha256_file(original), original, original),))


def test_union_does_not_enable_empirical_or_untyped_archive_transport(joint):
    selected, _, _ = joint
    with pytest.raises(StageGateError, match="immutable"):
        R.review_archive(
            runtime=selected.runtime,
            archive=selected.archive,
            archive_sha256=selected.archive_sha256,
            binding_path=selected.binding_path,
            observations_path=selected.observations_path,
            screen_path=None,
            additional_archives=[],
        )
    with pytest.raises(StageGateError, match="explicit native"):
        R.review_archive(
            runtime=selected.runtime,
            archive=selected.archive,
            archive_sha256=selected.archive_sha256,
            binding_path=selected.binding_path,
            observations_path=selected.observations_path,
            screen_path=selected.binding_path,
            additional_archives=selected.additional_archives,
        )
