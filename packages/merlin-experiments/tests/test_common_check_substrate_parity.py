"""Phase2 uses the original frozen host checker without issuing new proof grants.

Real source/profile snapshot and protocol admission execute. No theorem, native
compiler, numerical executor or simulator is started; missing Lean stays refused.
"""

import copy
import json
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import common_transforms as P2
from merlin_experiments.phase2 import functional_inputs as FI
from test_phase2_broker import make_broker

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.tool_profile import attach_common_substrate, inherited_tool_profile
from merlin.targetgen.tool_profile_shared import prepare_shared_source_release
from merlin.transforms.client import CHECK_SCHEMA
from merlin.verify.lean_contract import coverage_policy_record


@pytest.mark.parametrize("profile", [None, [], {}, {"common_substrate": []}, {"common_substrate": "unknown"}])
def test_unknown_or_malformed_profile_cannot_advertise_a_checker(profile):
    assert P2.supports_check(profile) is False


def original(tmp_path, version, monkeypatch):
    root = repo_root() / "src"
    if version in {"v6", "v7"}:
        root = tmp_path / "release"
        prepare_shared_source_release(
            root, python_source_root=repo_root() / "src", license_directory=repo_root(), version=int(version[-1])
        )
    bundle = dict(arm="raw_baseline", tools=[], allowed=[], denied=[], host_inputs=[])
    attach_common_substrate(
        bundle, version="merlin.common_compiler_substrate." + version, python_source_root=root, host_index_bits=64
    )
    for row in bundle["allowed"]:
        if row["path"].startswith("merlin/python/"):
            row["path"] = str(root / row["path"].removeprefix("merlin/python/"))
    archived = tmp_path / "original.yaml"
    archived.write_text(yaml.safe_dump(bundle))
    workspace = tmp_path / "original_workspace"
    workspace.mkdir()
    bwrap.materialize_bundle_inputs(workspace, bundle, repo=repo_root())
    selected = dict(bundle=bundle, workspace=workspace, bundle_path=archived, repo=repo_root())
    # Only the pre-existing private Phase1 provenance reconstruction is simulated.
    # P2 constructs the actual CommonTransformService from the sealed source view.
    monkeypatch.setattr(FI, "_verify_private_functional_provenance", lambda _pin: selected)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    before = copy.deepcopy(bundle)
    apply, check = P2.services(
        SimpleNamespace(host_provenance=object()), candidate=candidate, work_root=tmp_path / "host_work"
    )
    assert callable(apply) and callable(check)
    assert inherited_tool_profile(before, bundle) == before["compiler_tool_profile"]
    assert bundle["compiler_tool_profile"]["proof_policy"] == coverage_policy_record(version="3")
    return bundle, before, archived, candidate, check


@pytest.mark.parametrize("version", ["v3", "v4", "v5", "v6", "v7"])
def test_fixed_broker_uses_original_checker_and_refuses_candidate_authority(tmp_path, monkeypatch, version):
    bundle, before, archived, candidate, check = original(tmp_path, version, monkeypatch)
    action = P2.check_action()
    broker = make_broker(tmp_path, actions=(action,), services=BP.BrokerServices(common_check=check), max_calls=12)
    request = dict(schema=CHECK_SCHEMA, before_ir="actual original IR", after_ir="actual changed IR", entry="forward")
    for key in ("decoder", "callback", "semantics", "effects", "coverage", "toolchain", "proof_result"):
        path = candidate / (key + ".json")
        path.write_text(json.dumps({**request, key: "candidate must not choose this"}))
        answer = broker.execute({"action": action.name, "bindings": {"request": path.name}})
        assert answer["returncode"] == 125
        assert "cannot select semantics" in answer["stderr"]
    path = candidate / "valid.json"
    path.write_text(json.dumps(request))
    answer = broker.execute({"action": action.name, "bindings": {"request": path.name}})
    assert answer["returncode"] == 125 and "no explicitly selected common formal toolchain" in answer["stderr"]
    assert bundle == before
    assert json.loads(json.dumps(yaml.safe_load(archived.read_text()))) == json.loads(json.dumps(before))
    receipts = [json.loads(line) for line in broker.receipt_path.read_text().splitlines()]
    assert len(receipts) == 8 and all(row["returncode"] == 125 for row in receipts)
    assert all(row["action"] == "common-check" for row in receipts)


@pytest.mark.parametrize("version", ["v3", "v4", "v5", "v6", "v7"])
def test_original_profile_cannot_change_after_phase2_callback_is_issued(tmp_path, monkeypatch, version):
    bundle, _, archived, candidate, check = original(tmp_path, version, monkeypatch)
    changed = copy.deepcopy(bundle)
    changed["compiler_tool_profile"]["proof_policy"]["mode"] = "candidate-selected"
    archived.write_text(yaml.safe_dump(changed))
    path = candidate / "check.json"
    path.write_text(json.dumps(dict(schema=CHECK_SCHEMA, before_ir="before", after_ir="after", entry="forward")))
    with pytest.raises(ValueError, match="authority changed"):
        check(path.name, 5)
