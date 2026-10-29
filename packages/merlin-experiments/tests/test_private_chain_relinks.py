"""Historical relink audits allow only the exact deliberately removed ELF input."""

import json

import pytest
from merlin_experiments.phase1.feedback import private_chain_relinks as R
from merlin_experiments.phase2 import component_instruction_audit as A
from merlin_experiments.phase2.contracts import document_sha256

from merlin.common import invocation_record as IR
from merlin.model_check.full_output_audit import Files


@pytest.fixture
def observation(tmp_path):
    elf, policy, policy_receipt = (tmp_path / name for name in ("model.elf", "policy.json", "policy-receipt.json"))
    for path in (elf, policy, policy_receipt):
        path.write_bytes(b"metadata fixture; no target code")
    report_path = tmp_path / "report.json"
    report = {"policy_sha256": "a" * 64, "accessor_intake_sha256": "b" * 64,
              "decoder_selection": {"selector": "source_bound"}}
    args = {"policy_sha256": report["policy_sha256"], "accessor_intake_sha256": report["accessor_intake_sha256"],
            "decoder_selection": report["decoder_selection"]}
    with IR.observe_call(tmp_path, stage="whole_linked_instruction_policy", function=A._scan,
                         arguments=args, inputs=(elf, policy, policy_receipt), outputs=(report_path,)) as call:
        report_path.write_text(json.dumps(report))
        call.returned()
    selected_elf = {"path": str(elf), "sha256": IR._pin(elf)["sha256"]}
    check = document_sha256({"policy_sha256": "a" * 64, "accessor_sha256": "b" * 64,
                             "selection": report["decoder_selection"]})
    return call.path, report_path, report, selected_elf, check, policy


def test_original_completed_issuer_survives_only_its_exact_elf_cleanup(observation):
    path, report_path, report, elf, check, _policy = observation
    R._historical_issuer(Files(), path, report_path=report_path, report=report, elf=elf, expected_check=check)
    from pathlib import Path

    Path(elf["path"]).unlink()
    R._historical_issuer(Files(), path, report_path=report_path, report=report, elf=elf, expected_check=check)
    with pytest.raises(ValueError, match="another original policy"):
        R._historical_issuer(Files(), path, report_path=report_path, report=report, elf=elf, expected_check="f" * 64)
    with pytest.raises(ValueError, match="actual relink ELF"):
        R._historical_issuer(Files(), path, report_path=report_path, report=report,
                             elf={**elf, "sha256": "f" * 64}, expected_check=check)


def test_changed_or_unavailable_other_policy_inputs_do_not_get_cleanup_exception(observation):
    path, report_path, report, elf, check, policy = observation
    policy.write_bytes(b"changed")
    with pytest.raises(ValueError):
        R._historical_issuer(Files(), path, report_path=report_path, report=report, elf=elf, expected_check=check)
    policy.unlink()
    with pytest.raises(ValueError):
        R._historical_issuer(Files(), path, report_path=report_path, report=report, elf=elf, expected_check=check)


def test_failed_or_foreign_issuer_and_changed_output_refuse(observation):
    path, report_path, report, elf, check, _policy = observation
    original = json.loads(path.read_bytes())
    for key, value in (("status", "interrupted"), ("outcome", "raised"), ("inputs_unchanged", False),
                       ("callable", "foreign.accept"), ("outputs", [])):
        path.write_text(json.dumps({**original, key: value}))
        with pytest.raises(ValueError):
            R._historical_issuer(Files(), path, report_path=report_path, report=report, elf=elf, expected_check=check)
    path.write_text(json.dumps(original))
    report_path.write_text("changed")
    with pytest.raises(ValueError):
        R._historical_issuer(Files(), path, report_path=report_path, report=report, elf=elf, expected_check=check)
