"""Native admission of one pinned precomputed CCA bank for a coverage bank arm.

Uses real ``reference_subsets`` condition payloads (the two smallest published banks) as fixtures,
copied into a temporary repository, frozen through the real bundle snapshot and reached through
``author_profiles.check_broker_action`` and the broker's reference-analysis response.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1 import author_profiles as A
from merlin_experiments.phase1 import fresh10_profiles as P
from merlin_experiments.phase1.brokers import selfcheck as broker
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_analysis_probe as probe
from merlin_experiments.phase1.feedback import reference_bank as bank_mode
from merlin_experiments.phase1.tools import selfcheck as client

from merlin.common.paths import artifacts_dir
from merlin.targetgen.sandbox import bwrap as BW

pytest.importorskip("merlin.mining.reference_subsets")


def _conditions():
    """The two smallest CCA-enabled conditions of the selected subsets product (local fixture names)."""
    selected = os.environ.get("MERLIN_CCA_BANK_CONDITIONS")
    root = Path(selected) if selected else (
        artifacts_dir() / "kernel-mining/gemmini/expert-kernel-subsets-v3/conditions")
    sized = []
    for bank in sorted(root.glob("*/bank.json")) if root.is_dir() else ():
        units = len(json.loads(bank.read_bytes()).get("units", ()))
        if units:
            sized.append((units, bank.parent))
    if len(sized) < 2:
        pytest.skip("published CCA bank conditions are unavailable in this checkout")
    sized.sort()
    return {"cca-4": sized[0][1], "cca-8": sized[1][1]}


BANK4, BANK8 = P.COVERAGE_BANK_PREFIX + "fixture-4", P.COVERAGE_BANK_PREFIX + "fixture-8"
RAW = P.COVERAGE_NAMES[0]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _request(operation, path="", **fields):
    return dict(action=analysis.ACTION, reference_analysis=dict(operation=operation, path=path, **fields))


@pytest.fixture
def world(tmp_path, monkeypatch):
    conditions = _conditions()
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    repo = tmp_path / "repo"
    for label in ("cca-4", "cca-8"):
        shutil.copytree(conditions[label], repo / "banks" / label)
    references = repo / "refs/reference-kernels"
    (references / "models").mkdir(parents=True)
    (references / "models/raw.elf").write_bytes(b"\x7fELF" + b"\0" * 60)
    pins = {name: _sha(repo / "banks" / label / "bank.json") for name, label in ((BANK4, "cca-4"), (BANK8, "cca-8"))}
    monkeypatch.setattr(P, "COVERAGE_BANKS", pins)
    monkeypatch.setenv(A.ENV, BANK4)

    def make(label="cca-4", *, alias=None, name="run"):
        bundle = dict(allowed=[dict(path="refs/reference-kernels/", mode="ro", **{"as": "reference-kernels"}),
                               dict(path="banks/" + label + "/", mode="ro", **{"as": "cca-evidence"})],
                      host_inputs=[])
        workspace = tmp_path / name / "workspace"
        workspace.mkdir(parents=True)
        BW.materialize_bundle_inputs(workspace, bundle, repo=repo)
        (workspace / "cca-evidence").symlink_to(alias or BW.resolve_grant("banks/" + label + "/", repo).absolute())
        (workspace / "reference-kernels").symlink_to(references)
        manifest = tmp_path / name / "input_bundle_manifest.yaml"
        manifest.write_text(yaml.safe_dump(bundle))
        return dict(workspace=workspace, bundle_manifest=manifest, repo=repo)

    selected = make()
    yield dict(make=make, repo=repo, run=selected, banks={label: json.loads(
        (repo / "banks" / label / "bank.json").read_bytes()) for label in ("cca-4", "cca-8")})
    for directory in tmp_path.glob("*/workspace"):
        BW.remove_bundle_snapshot(directory)


def _handle(world, request, run=None):
    return analysis.handle(request, **(run or world["run"]))


def _occurrence(bank, index=0):
    unit = bank["units"][index]
    report = unit["reports"][0]
    return unit["kernel_id"], Path(report).parent.name, Path(report).parent.as_posix()


def _report(response, workspace):
    if response.get("report_in_detail"):
        from merlin_experiments.phase1.feedback.result_details import load_row

        return load_row(response, workspace=workspace)["report"]
    return response["report"]


def test_bank_arm_profile_admits_reference_analysis_only(world, monkeypatch):
    profile = A.resolve(BANK4)
    assert profile.coverage and profile.seeded and profile.bank_sha256 == P.COVERAGE_BANKS[BANK4]
    assert profile.allows("reference_analysis") and not profile.allows("automatic_kernel_learning")
    A.check_broker_action("reference-analysis")
    with pytest.raises(FileNotFoundError):
        A.check_broker_action("kernel-learning")
    monkeypatch.setenv(A.ENV, RAW)
    with pytest.raises(FileNotFoundError):
        A.check_broker_action("reference-analysis")


def test_inventory_lists_exactly_the_admitted_units_from_the_verified_bank(world):
    bank = world["banks"]["cca-4"]
    response = _handle(world, _request("inventory", "reference-kernels", limit=16))
    assert response["status"] == "analyzed", response
    report = _report(response, world["run"]["workspace"])
    assert report["kind"] == "cca_bank_inventory" and report["ordering"] == "unit_id"
    expected = sorted(unit["kernel_id"] for unit in bank["units"])
    assert [row["unit_id"] for row in report["units"]] == expected[:16]
    assert report["bank"]["sha256"] == P.COVERAGE_BANKS[BANK4]
    assert report["bank"]["on_demand_analysis"] == "not_performed"
    for row in report["units"]:
        assert row["summary"]["instruction_classes"] and isinstance(row["summary"]["observation_fields"], list)
        assert row["summary"]["originals"] and len(row["summary"]["accelerator_sites"]) == len(row["card_occurrences"])
        assert row["card_occurrences"] and all(path.startswith("cca-evidence/")
                                               for occurrence in row["occurrences"]
                                               for path in occurrence["files"].values())
    # The compact transport keeps every unit and occurrence locator.
    if response.get("report_in_detail"):
        assert [row["unit_id"] for row in response["report"]["units"]] == expected[:16]


def test_lift_serves_precomputed_records_by_unit_or_path(world, monkeypatch):
    monkeypatch.setattr(analysis, "_lift", lambda *a, **k: pytest.fail("on-demand lift in bank mode"))
    bank = world["banks"]["cca-4"]
    unit, region, folder = _occurrence(bank)
    expected = json.loads((world["repo"] / "banks/cca-4" / folder / "cca.json").read_bytes())
    for request in (_request("lift", entry=unit), _request("lift", "cca-evidence/" + folder + "/analysis.json"),
                    _request("lift", entry=unit, region=region)):
        response = _handle(world, request)
        assert response["status"] == "analyzed", response
        report = _report(response, world["run"]["workspace"])
        assert report["source"] == "precomputed_cca_bank" and report["reference_entry"]["id"] == unit
        served = [row for row in report["regions"] if row["bank_occurrence"]["region_id"] == region]
        assert served and served[0]["cca"] == expected


def test_compare_uses_precomputed_reference_against_owned_candidate(world, monkeypatch):
    monkeypatch.setattr(analysis, "_lift", analysis._lift)
    bank = world["banks"]["cca-4"]
    unit, region, folder = _occurrence(bank)
    workspace = world["run"]["workspace"]
    (workspace / "artifacts").mkdir()
    shutil.copyfile(world["repo"] / "banks/cca-4" / folder / "instructions.dis", workspace / 'artifacts' / 'kernel.dis')
    response = _handle(world, _request("compare", entry=unit, region=region,
                                       candidate=dict(path=(Path('artifacts') / 'kernel.dis').as_posix(), format="disassembly")))
    assert response["status"] == "analyzed", response
    report = _report(response, workspace)
    assert report["kind"] == "reference_candidate_comparison"
    assert report["reference"]["source"] == "precomputed_cca_bank"
    assert report["candidate"]["input_role"] == "candidate_artifact"
    # A raw reference on the left is refused, not lifted.
    refused = _handle(world, _request("compare", "reference-kernels/models/raw.elf",
                                      candidate=dict(path=(Path('artifacts') / 'kernel.dis').as_posix())))
    assert refused["reason"] == bank_mode.OUTSIDE


@pytest.mark.parametrize("request_of", [
    lambda b4, b8: _request("lift", entry=next(u["kernel_id"] for u in b8["units"]
                                               if u["kernel_id"] not in {v["kernel_id"] for v in b4["units"]})),
    lambda b4, b8: _request("lift", "cca-evidence/" + next(u["card"] for u in b8["units"]
                                                         if u not in b4["units"])),
    lambda b4, b8: _request("lift", entry=b4["units"][0]["kernel_id"], region="region-unknown"),
    lambda b4, b8: _request("lift", entry="machine-" + "0" * 64),
    lambda b4, b8: _request("lift", "reference-kernels/models/raw.elf"),
    lambda b4, b8: _request("lift", "reference-kernels"),
    lambda b4, b8: _request("lift", entry=b4["units"][0]["kernel_id"], symbol="kernel_entry"),
    lambda b4, b8: _request("inventory", "reference-kernels/models/raw.elf"),
])
def test_excluded_unit_and_raw_on_demand_cca_are_refused(world, monkeypatch, request_of):
    monkeypatch.setattr(analysis, "_lift", lambda *a, **k: pytest.fail("on-demand lift in bank mode"))
    response = _handle(world, request_of(world["banks"]["cca-4"], world["banks"]["cca-8"]))
    assert response["status"] == "unknown" and response["reason"] == bank_mode.OUTSIDE, response
    assert "inventory" in response["next_action"]


def test_another_conditions_bank_is_refused(world):
    other = world["make"]("cca-8", name="other")
    response = _handle(world, _request("inventory"), run=other)
    assert response["reason"] == "cca_bank_pin_mismatch"


def test_global_corpus_or_ancestor_grant_is_refused(world, tmp_path):
    corpus = world["repo"] / "corpus"
    corpus.mkdir()
    (corpus / "database.sqlite").write_bytes(b"SQLite format 3\0")
    with pytest.raises(bank_mode._Refused, match="cca_bank_unverified"):
        bank_mode.verify(corpus, P.COVERAGE_BANKS[BANK4])
    with pytest.raises(bank_mode._Refused, match="cca_bank_unverified"):
        bank_mode.verify(world["repo"] / "banks", P.COVERAGE_BANKS[BANK4])


@pytest.mark.parametrize("change", ["card", "bank_roster", "extra_file", "symlink"])
def test_mutated_bank_or_hand_edited_map_is_refused(world, change):
    root = world["repo"] / "banks/cca-4"
    bank = world["banks"]["cca-4"]
    for path in root.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    if change == "card":
        card = root / bank["units"][0]["card"]
        document = json.loads(card.read_bytes())
        document["occurrences"][0]["accelerator_sites"] += 1
        card.write_text(json.dumps(document))
        expected = "cca_bank_unverified"
    elif change == "bank_roster":
        document = dict(bank, units=bank["units"][:1])
        (root / "bank.json").write_text(json.dumps(document))
        expected = "cca_bank_pin_mismatch"
    elif change == "extra_file":
        (root / "REFERENCE_MAP.json").write_text("{}")
        expected = "cca_bank_unverified"
    else:
        (root / "units/link").symlink_to(world["repo"] / "banks/cca-8")
        expected = "cca_bank_unverified"
    response = _handle(world, _request("inventory"), run=world["make"]("cca-4", name="mutated"))
    assert response["status"] == "unknown" and response["reason"] == expected, response


def test_alias_escape_is_refused(world):
    other = world["make"]("cca-4", alias=world["repo"] / "banks/cca-8", name="escape")
    assert _handle(world, _request("inventory"), run=other)["reason"] == "cca_bank_alias_does_not_match_original_grant"


def test_candidate_side_analysis_is_unchanged_in_bank_mode(world, monkeypatch):
    workspace = world["run"]["workspace"]
    (workspace / "artifacts").mkdir()
    (workspace / 'artifacts' / 'kernel.dis').write_text("00000000 <kernel>:\n 0: 00008067 ret\n")
    monkeypatch.setattr(bank_mode, "open_bank", lambda *a, **k: pytest.fail("candidate-only reads the bank"))
    response = _handle(world, _request("candidate", (Path('artifacts') / 'kernel.dis').as_posix(), symbol="kernel"))
    assert response["status"] == "analyzed", response
    before_after = _handle(world, _request("compare", (Path('artifacts') / 'kernel.dis').as_posix(), symbol="kernel",
                                           candidate=dict(path=(Path('artifacts') / 'kernel.dis').as_posix(), symbol="kernel")))
    assert before_after["status"] == "analyzed", (before_after.get("reason"), before_after.get("gap"))


def test_raw_arm_refuses_reference_analysis_before_reading_anything(world, monkeypatch):
    monkeypatch.setenv(A.ENV, RAW)
    monkeypatch.setattr(bank_mode, "open_bank", lambda *a, **k: pytest.fail("raw arm read a bank"))
    monkeypatch.setattr(analysis, "_reference_root", lambda *a, **k: pytest.fail("raw arm read a grant"))
    response = _handle(world, _request("lift", entry=world["banks"]["cca-4"]["units"][0]["kernel_id"]))
    assert response["status"] == "denied" and response["reason"] == "reference_guidance_disabled"


def test_broker_response_serves_the_bank_through_the_real_action_gate(world):
    A.check_broker_action("reference-analysis")
    unit, region, _ = _occurrence(world["banks"]["cca-4"])
    response = broker._reference_analysis_response(
        _request("lift", entry=unit, region=region), workspace=world["run"]["workspace"], facts=None,
        compiler_sha256="0" * 64, rid="r1", started_ns=time.time_ns(),
        bundle_manifest=world["run"]["bundle_manifest"], repo=world["repo"], descriptor=None, target=None)
    assert response["status"] == "analyzed" and response["official"] is False
    assert response["schema"] == analysis.SCHEMA and response["qualification_issued"] is False


def test_preflight_probe_checks_inventory_unit_lift_and_raw_refusal(world):
    record = probe.probe(**world["run"])
    assert record["status"] == "analyzed" and record["mode"] == "cca_bank"
    assert record["raw_reference_refusal"] == bank_mode.OUTSIDE
    assert record["unit"] in {unit["kernel_id"] for unit in world["banks"]["cca-4"]["units"]}


def test_release_staging_copies_exactly_the_pinned_bank_and_declares_one_grant(world, tmp_path):
    from merlin_experiments.corpus import public_references as R
    from merlin_experiments.spec import SpecError

    source, pin = world["repo"] / "banks/cca-4", P.COVERAGE_BANKS[BANK4]
    destination = world["repo"] / "release/experiment/cca-evidence"
    destination.parent.mkdir(parents=True)
    record = R.stage_bank(source, pin, destination)
    assert record["sha256"] == pin and record["units"] == [u["kernel_id"] for u in world["banks"]["cca-4"]["units"]]
    assert bank_mode.verify(destination, pin)["label"] == world["banks"]["cca-4"]["label"]
    assert not any(path.stat().st_mode & 0o222 for path in (destination, *destination.rglob("*")))
    document = {"information_sets": {"public_v0": {"allowed": []}}}
    R.declare_bank(document, destination=destination, variants=["public_v0"], sha256=pin, repo=world["repo"])
    assert document["information_sets"]["public_v0"]["allowed"] == [
        {"path": "release/experiment/cca-evidence/", "as": "cca-evidence", "mode": "ro",
         "note": "pinned precomputed CCA bank; static evidence only"}]
    with pytest.raises(SpecError, match="differs from its declared pin"):
        R.stage_bank(world["repo"] / "banks/cca-8", pin, tmp_path / "other")
    with pytest.raises(SpecError):
        R.stage_bank(world["repo"] / "banks", pin, tmp_path / "ancestor")


def test_copied_client_reaches_bank_over_the_qa_channel(world, tmp_path):
    workspace = world["run"]["workspace"]
    shim = workspace / "agent_selfcheck.py"
    shutil.copyfile(client.__file__, shim)
    assert shim.read_bytes() == Path(client.__file__).read_bytes()
    channel = workspace / ".qa_channel"
    channel.mkdir()
    unit, region, _ = _occurrence(world["banks"]["cca-4"])
    argv = [sys.executable, str(shim), "--reference-analysis", "lift", "--reference-entry", unit,
            "--reference-region", region, "--wait-budget", "10"]
    process = subprocess.Popen(argv, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline, requests = time.monotonic() + 10, []
        while not requests and time.monotonic() < deadline:
            requests = list(channel.glob("req_*.json"))
            time.sleep(0.01)
        assert requests
        request = json.loads(requests[0].read_bytes())
        A.check_broker_action(request["action"])
        rid = request["request_id"]
        response = broker._reference_analysis_response(
            request, workspace=workspace, facts=None, compiler_sha256="0" * 64, rid=rid,
            started_ns=time.time_ns(), bundle_manifest=world["run"]["bundle_manifest"], repo=world["repo"])
        (channel / f"resp_{rid}.json").write_text(json.dumps(response))
        (channel / f"done_{rid}").write_text("ok")
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stdout + stderr
        result, _ = json.JSONDecoder().raw_decode(stdout)
        assert result["status"] == "analyzed" and result["report"]["reference_entry"]["id"] == unit
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
