"""Synthetic role custody; none of these fixtures grants target execution."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import development_cohort as F
from merlin_experiments.phase0 import development_sources as D
from merlin_experiments.phase0 import public_programs as P
from merlin_experiments.phase0 import public_source_roles as R
from merlin_experiments.phase0 import writer
from test_operator_private_exclusion import selected
from test_public_integer_capture import capture

from merlin.targetgen import capsule_source as S
from merlin.targetgen.corpus_spec import CorpusBinding


def identity(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, *, role="functional_guard", chosen="functional_guard"):
    source = {
        "schema": "merlin.phase0.public_development_source.v2",
        "scope": "public_development",
        "name": "synthetic_public",
        "pattern": "matmul",
        "geometry": {"M": 1, "K": 2, "N": 1},
        "arithmetic": {"contract": D.CONTRACT, "readout_shift": 3},
        "max_logical_elements": 1000,
    }
    request = {"declaration": source, **({"role": role} if role is not None else {})}
    root = tmp_path / "frozen"
    manifest = F.freeze(
        [request],
        root,
        provenance={"purpose": "synthetic role test"},
        budgets={
            "max_members": 4,
            "max_source_bytes": 10000,
            "max_reference_elements": 100,
            "max_reference_work": 100,
            "max_total_reference_work": 100,
            "max_total_reference_elements": 100,
        },
    )
    private, private_path = selected(tmp_path)
    profiles = tmp_path / "operator"
    profiles.mkdir()
    profile = profiles / "roles.json"
    row = {
        "identity": "original:0",
        "request_sha256": manifest["members"][0]["request_sha256"],
        "declaration_sha256": D._digest(source),
        "role": chosen,
        "origin": "original_request" if role is not None else "prospective_operator_profile",
        "reason": "Explicit setup obligation; no timing or success-based selection",
    }
    document = {
        "schema": R.SCHEMA,
        "scope": "prospective_public_development_before_performance_evaluation",
        "inventories": {"original": {"path": str(root / "manifest.json"), "sha256": identity(root / "manifest.json")}},
        "members": [row],
    }
    profile.write_text(json.dumps(document))
    profile.chmod(0o600)
    return root, manifest, private, private_path, profile, document


def owner(tmp_path, **kwargs):
    values = fixture(tmp_path, **kwargs)
    return R.select(values[4], identity(values[4]), operator_exclusion=values[2]), values


def bound_capture(tmp_path, selection, values):
    root, manifest, *_ = values
    captured = capture(tmp_path / "capture")
    loader = root / manifest["members"][0]["loader"]
    # Transport-only fixture: original target/sealed admission remains absent.
    Path(captured["loader_path"]).write_bytes(loader.read_bytes())
    captured["loader_sha256"] = identity(loader)
    return captured, selection.bind("original:0", captured)


def test_original_guard_cannot_be_selected_as_objective(tmp_path):
    values = fixture(tmp_path, chosen="objective")
    with pytest.raises(ValueError, match="contradicts explicit"):
        R.select(values[4], identity(values[4]), operator_exclusion=values[2])


@pytest.mark.parametrize("substitute", [None, {}, {"status": "private_exclusion_checked"}])
def test_new_role_owner_needs_original_live_private_selection(tmp_path, substitute):
    values = fixture(tmp_path)
    with pytest.raises(ValueError, match="original live private-exclusion"):
        R.select(values[4], identity(values[4]), operator_exclusion=substitute)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "request", "declaration", "origin", "unknown_role"])
def test_profile_keeps_complete_original_roster_and_intent(tmp_path, mutation):
    values = fixture(tmp_path)
    profile, document = values[4:]
    if mutation == "missing":
        document["members"] = []
    elif mutation == "duplicate":
        document["members"] *= 2
    elif mutation in {"request", "declaration"}:
        document["members"][0][mutation + "_sha256"] = "f" * 64
    elif mutation == "origin":
        document["members"][0]["origin"] = "prospective_operator_profile"
    else:
        document["members"][0]["role"] = "caller_selected_fast_case"
    profile.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        R.select(profile, identity(profile), operator_exclusion=values[2])


def test_undeclared_role_requires_new_operator_profile_and_reason(tmp_path):
    selection, values = owner(tmp_path, role=None, chosen="objective")
    assert selection.verify()[2]["original:0"]["role"] == "objective"
    before = values[5]
    before["members"][0]["reason"] = ""
    values[4].write_text(json.dumps(before))
    with pytest.raises(ValueError):
        R.select(values[4], identity(values[4]), operator_exclusion=values[2])


@pytest.mark.parametrize("change", ["loader", "label", "capture", "receipt", "profile"])
def test_exact_capture_and_selected_role_binding_cannot_be_substituted(tmp_path, change):
    selection, values = owner(tmp_path)
    captured, binding = bound_capture(tmp_path, selection, values)
    entry = {"materialized_capture": captured, "public_source_role": binding, "cat": "model"}
    entry = copy.deepcopy(entry)
    if change in {"loader", "label"}:
        key = "loader_sha256" if change == "loader" else "workload_id"
        entry["materialized_capture"][key] = "different"
    elif change == "profile":
        entry["materialized_capture"]["profile_sha256"] = "f" * 64
    else:
        entry["materialized_capture"][change + "_sha256"] = "f" * 64
    with R.scoped_selection(selection), pytest.raises(ValueError):
        R.require_entry(entry)


def test_serialized_role_is_not_a_writer_grant(tmp_path, monkeypatch):
    selection, values = owner(tmp_path)
    captured, binding = bound_capture(tmp_path, selection, values)
    entry = {"materialized_capture": captured, "public_source_role": binding, "cat": "model"}
    monkeypatch.setattr(
        writer, "_write_capsule_inner", lambda *a, **k: pytest.fail("writer entered without live owner")
    )
    with pytest.raises(ValueError, match="live operator"):
        writer._write_capsule(entry, None, tmp_path / "public")
    with pytest.raises(ValueError, match="live operator"):
        with R.scoped_selection(dict(binding)):
            pass


def test_guard_complete_reference_packaging_and_exclusion_without_score(tmp_path):
    selection, values = owner(tmp_path)
    captured, role = bound_capture(tmp_path, selection, values)
    entry = {
        "name": "guard",
        "cat": "model",
        "kind": "model",
        "label": "public",
        "materialized_capture": captured,
        "public_source_role": role,
        "capture_quantization": "already_materialized",
        "output_numeric_policy": "captured_per_output_v1",
    }
    with pytest.raises(ValueError, match="cannot become"):
        P.performance(entry, trace={}, forms={}, facts_sha256="a" * 64)
    binding = CorpusBinding("fixture", 2, "int8", "i32", True, ["L0"], "exact_int", atol=0, rtol=0)
    written = S.write_model_capsule(
        entry, binding, tmp_path / "publication", artifact=S.materialized_model_artifacts(captured)
    )
    cap = yaml.safe_load((written / "capsule.yaml").read_bytes())
    before_policy = copy.deepcopy(cap["numeric_policy"])
    with R.scoped_selection(selection):
        R.stamp_written(entry, cap, written)
        (written / "capsule.yaml").write_text(yaml.safe_dump(cap))
        assert R.require_publication(cap, written) == "functional_guard"
    assert cap["numeric_policy"] == before_policy and "performance" not in cap
    assert P.verify_written(cap, written)["status"] == "source_prepared_execution_unknown"
    values[2].check_member(written)
    with pytest.raises(ValueError, match="live operator"):
        R.require_publication(cap, written)
    changed = copy.deepcopy(cap)
    changed["performance"] = {"member_class": "OBJECTIVE"}
    with pytest.raises(ValueError, match="cannot become"):
        R.verify_written(changed, written)


def test_profile_and_private_selection_drift_refuse_existing_owner(tmp_path):
    selection, values = owner(tmp_path)
    values[4].chmod(0o644)
    with pytest.raises(ValueError):
        selection.verify()
    values[4].chmod(0o600)
    values[3].write_text("{}")
    with pytest.raises(ValueError):
        selection.verify()


def test_complete_objective_roster_cannot_drop_unpublished_members(tmp_path):
    selection, _ = owner(tmp_path, role=None, chosen="objective")
    with R.scoped_selection(selection), pytest.raises(ValueError, match="omitted"):
        R.require_objective_roster([])


def test_legacy_performance_projection_has_no_implicit_role_reinterpretation(tmp_path):
    captured = capture(tmp_path / "capture")
    trace = json.loads((tmp_path / "capture/frontend-trace.json").read_bytes())
    assert (
        P.performance({"materialized_capture": captured}, trace=trace, forms={}, facts_sha256="a" * 64)["member_class"]
        == "OBJECTIVE"
    )


def test_original_requirements_transports_guard_and_keeps_sealed_numeric_gate(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import capture_execution_attestation as attest
    from merlin_experiments.phase0 import evidence, requirements
    from test_public_integer_capture import POLICY

    selection, values = owner(tmp_path)
    captured, _ = bound_capture(tmp_path, selection, values)
    label = captured["workload_id"]
    calls = []
    # Synthetic transport observation only. The real writer below still needs
    # its original independent realization owner and therefore must refuse.
    monkeypatch.setattr(attest, "require_verified_execution", lambda value: calls.append(value))
    receipt = {"synthetic transport fixture": True}
    full = {
        "applications": {
            label: {
                "capture_source_path": captured["path"],
                "capture_sha256": captured["capture_sha256"],
                "capture_receipt": {"receipt_sha256": captured["receipt_sha256"]},
                "n_operations": 1,
            }
        }
    }
    entries, payloads = requirements._materialized_iteration_capsules(
        full,
        "a" * 64,
        loader_snapshots={label: (Path(captured["loader_path"]), captured["loader_sha256"])},
        performance_selection=POLICY,
        capture_attestations={label: receipt},
        source_role_selection=selection,
        source_role_members={label: "original:0"},
    )
    assert calls == [receipt]
    entry = entries[0]
    assert entry["cat"] == "model" and entry["label"] == "public" and "performance" not in entry
    assert entry["capture_execution_attestations"] == [receipt]
    prepared = tmp_path / "prepared"
    evidence._materialize_evidence(prepared, payloads)
    for key in ("path", "loader_path", "profile_path"):
        if key in entry["materialized_capture"]:
            entry["materialized_capture"][key] = str(prepared / entry["materialized_capture"][key])
    binding = CorpusBinding("fixture", 2, "int8", "i32", True, ["L0"], "exact_int", atol=0, rtol=0)
    with R.scoped_selection(selection), pytest.raises(ValueError, match="lacks a verified isolated"):
        writer._write_capsule(entry, binding, tmp_path / "full-writer")


def test_duplicate_operator_profile_keys_are_not_reinterpreted(tmp_path):
    values = fixture(tmp_path)
    raw = values[4].read_text().replace('"schema":', '"schema":"forged", "schema":', 1)
    values[4].write_text(raw)
    with pytest.raises(ValueError):
        R.select(values[4], identity(values[4]), operator_exclusion=values[2])
