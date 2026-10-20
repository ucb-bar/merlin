"""Synthetic decision plumbing; these fixtures do not qualify target timing."""

import pytest
from merlin_experiments.phase2 import _public_native_feedback as FEEDBACK
from merlin_experiments.phase2 import native_development_decision as D
from merlin_experiments.phase2 import public_diagnostic_launch as L
from merlin_experiments.phase2 import public_native_exploration as N
from merlin_experiments.phase2.contracts import StageGateError
from test_ordinary_native_archive_screen import ordinary_inputs as ordinary_inputs
from test_ordinary_native_archive_screen import source as source
from test_public_native_exploration import projection, selection


def evidence(owner, interval):
    arms = []
    for digest, limits in (("b", [100, 120]), ("c", interval)):
        arms.append(
            dict(
                functional_status="PASS",
                native_exploration=projection(owner, limits),
                compiled=dict(status="observed", elf_sha256=digest * 64, llvm_mlir_sha256=digest * 64),
            )
        )
    document = dict(
        selected_members=[["family", "one"]],
        cells=[dict(family="family", capsule="one", baseline=arms[0], candidate=arms[1])],
    )
    document["native_exploration"] = FEEDBACK.join(document, owner=owner)
    return document


@pytest.mark.parametrize(
    "interval,expected", [([50, 60], "candidate"), ([99.5, 99.9], None), ([95, 125], None), ([130, 160], "baseline")]
)
def test_qualified_priority_never_promotes_or_establishes_plateau(source, interval, expected):
    owner = N.prepare_public_native_exploration(selection(source))
    result = D.decide(evidence(owner, interval), owner=owner, required_members=(("family", "one"),))
    assert result["next_exploration_parent"] == expected
    assert result["promotion_authority"] == "NONE" and result["measured_cycles"] is None
    assert result["plateau_evidence"] is False and result["full_search_ready"] is False


@pytest.mark.parametrize("fault", ["same_ir", "missing_compilation", "bad_hash", "incorrect"])
def test_no_priority_from_unchanged_unidentified_or_incorrect_products(source, fault):
    owner = N.prepare_public_native_exploration(selection(source))
    doc = evidence(owner, [50, 60])
    arm = doc["cells"][0]["candidate"]
    if fault == "same_ir":
        arm["compiled"]["llvm_mlir_sha256"] = "b" * 64
    elif fault == "missing_compilation":
        arm["compiled"] = {}
    elif fault == "bad_hash":
        arm["compiled"]["elf_sha256"] = "z" * 64
    else:
        arm["functional_status"] = "FAIL"
    result = D.decide(doc, owner=owner, required_members=(("family", "one"),))
    assert result["next_exploration_parent"] is None


@pytest.mark.parametrize("fault", ["drop", "replace", "duplicate", "malformed", "forged_projection", "drift"])
def test_original_roster_projection_and_owner_remain_required(source, fault):
    owner = N.prepare_public_native_exploration(selection(source))
    doc = evidence(owner, [50, 60])
    if fault == "drop":
        doc["selected_members"] = []
    elif fault == "replace":
        doc["selected_members"] = [["family", "other"]]
    elif fault == "duplicate":
        doc["cells"].append(doc["cells"][0])
    elif fault == "malformed":
        doc["selected_members"] = [[{}, "one"]]
    elif fault == "forged_projection":
        doc["native_exploration"]["members"][0]["recommendation"] = "forged"
    else:
        source["observations_path"].write_text("changed evidence")
    with pytest.raises(StageGateError):
        D.decide(doc, owner=owner, required_members=(("family", "one"),))


@pytest.mark.parametrize("policy,owner", [(D.POLICY, None), ("implicit", object()), (True, object())])
def test_launcher_rejects_unissued_or_implicit_native_policy_before_jobs(tmp_path, policy, owner):
    with pytest.raises(StageGateError, match="explicit policy"):
        L.run_plan({}, {}, output=tmp_path / "never", native_exploration=owner, native_decision_policy=policy)
    assert not (tmp_path / "never").exists()


@pytest.mark.parametrize("pair_status,expected", [("UNKNOWN", None), ("PASS", "candidate")])
def test_explicit_pair_must_pass_before_native_priority(source, pair_status, expected):
    owner = N.prepare_public_native_exploration(selection(source))
    doc = evidence(owner, [50, 60])
    doc["cells"][0]["diagnostic_pair_check"] = {"status": pair_status}
    doc["native_exploration"] = FEEDBACK.join(doc, owner=owner)
    assert doc["native_exploration"]["members"][0]["recommendation"] == expected
    result = D.decide(doc, owner=owner, required_members=(("family", "one"),))
    assert result["next_exploration_parent"] == expected


@pytest.mark.parametrize("protocol", [True, "implicit", "merlin.diagnostic_source_pair.v1"])
def test_pair_protocol_cannot_execute_without_original_preparation(tmp_path, protocol):
    with pytest.raises(StageGateError, match="explicit protocol"):
        L.run_plan({}, {}, output=tmp_path / "never", diagnostic_pair_protocol=protocol)
    assert not (tmp_path / "never").exists()
