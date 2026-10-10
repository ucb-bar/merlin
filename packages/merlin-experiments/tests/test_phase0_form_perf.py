"""The form-perf tier: classes from iteration forms, a member per class, a vendor bar, a coverage report."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import form_perf as FP
from merlin_experiments.phase0 import profiles

from merlin.common.paths import repo_root
from merlin.targetgen import corpus_spec as CS


def _template() -> dict:
    return yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text())


def _pw(*, stratify: bool = False) -> dict:
    """The shared PW family; geometry stratification off unless a test is about it."""
    sweep = copy.deepcopy(next(s for s in _template()["sweeps"] if s["id"] == "PW"))
    sweep["requires_form_scope"]["stratify_geometry"] = stratify
    return sweep


def _member(group, key, entry, cycles, placement="systolic"):
    return {
        "group": group,
        "placement": placement,
        "key": key,
        "entry": entry,
        "price": {"macs": 1, "predicted_cycles": cycles},
    }


_STEM = {"placement": "device", "op": "matmul", "activation_source": "model_input"}
_BODY = {"placement": "device", "op": "matmul", "activation_source": "intermediate"}
_POOL = {"placement": "host_region", "op": "window_mean", "input_rank": 4, "reduced_dims": 2}


def _scope() -> dict:
    applications = {
        "residual_cnn": {
            "members": [
                _member(4, _STEM, {"op": "matmul", "M": 256, "K": 27, "N": 8, "epilogue": []}, 512),
                _member(10, _BODY, {"op": "matmul", "M": 256, "K": 72, "N": 8, "epilogue": []}, 1280),
                _member(15, _BODY, {"op": "matmul", "M": 256, "K": 72, "N": 8, "epilogue": []}, 1280),
                _member(
                    16,
                    _POOL,
                    {"op": "matmul", "M": 8, "K": 256, "N": 1, "epilogue": ["acc_scale"], "acc_scale": 1 / 256},
                    128,
                    placement="host",
                ),
            ]
        },
        "coverage_mlp": {"members": [_member(1, _STEM, {"op": "matmul", "M": 2, "K": 8, "N": 4, "epilogue": []}, 2)]},
    }
    summary = FP.aggregate(applications)
    return {"schema": FP.SCHEMA, "classes": summary["classes"], "applications": summary["application_totals"]}


def test_classes_carry_per_application_shares_and_the_costliest_representative():
    scope = _scope()
    by_key = {
        row["key"]["activation_source" if "activation_source" in row["key"] else "op"]: row for row in scope["classes"]
    }
    stem, body, pool = by_key["model_input"], by_key["intermediate"], by_key["window_mean"]
    assert stem["share_by_application"] == {"coverage_mlp": 1.0, "residual_cnn": pytest.approx(512 / 3200)}
    assert body["occurrences"] == 2 and body["max_share"] == pytest.approx(2560 / 3200)
    assert pool["max_share"] == pytest.approx(128 / 3200)
    assert stem["members"][stem["representative"]]["application"] == "residual_cnn"
    assert [row["class_id"] for row in scope["classes"]] == [stem["class_id"], body["class_id"], pool["class_id"]]


def test_non_array_forms_are_unpriced_and_require_conservative_coverage():
    machine = SimpleNamespace(array_rows=16, array_cols=16, refusals={})
    matmul = {"op": "matmul", "M": 16, "K": 32, "N": 16}
    residual = {"op": "residual_add", "M": 16, "N": 16}
    assert FP.predict(matmul, machine)["predicted_cycles"] is not None
    assert FP.predict(residual, machine)["predicted_cycles"] is None
    assert FP.predict(matmul, machine, placement="host")["predicted_cycles"] is None

    summary = FP.aggregate(
        {
            "iteration": {
                "members": [
                    _member(1, _STEM, matmul, 100),
                    _member(2, {"placement": "device", "op": "residual_add"}, residual, None),
                ]
            }
        }
    )
    assert summary["application_totals"]["iteration"]["basis"] == "macs"
    assert all(row["unpriced_applications"] == ["iteration"] for row in summary["classes"])
    requirement = {"scope": {"performance": {"forms": {"schema": FP.SCHEMA, "classes": summary["classes"]}}}}
    coverage = FP.form_perf_coverage(requirement, [], threshold=0.99)
    assert coverage["status"] == "incomplete"
    assert len(coverage["missing"]) == 2
    assert all(row["required_reason"] == "unpriced_application_requires_all_forms" for row in coverage["classes"])


def test_batch_price_multiplies_per_slice_issue_cycles_without_flattening_rows(monkeypatch):
    machine = SimpleNamespace(array_rows=4, array_cols=4, refusals={})
    entry = {"op": "matmul", "M": 3, "K": 5, "N": 6}
    one = FP.predict(entry, machine)
    five = FP.predict(entry, machine, batch_slices=5)
    assert (five["rows"], five["depth"], five["cols"], five["batch_slices"]) == (3, 5, 6, 5)
    assert five["macs"] == 5 * one["macs"]
    assert five["predicted_cycles"] == 5 * one["predicted_cycles"]
    # A nonlinear neutral cost witness distinguishes B separate invocations from flattened M.
    from merlin.perf import mesh_occupancy

    monkeypatch.setattr(mesh_occupancy, "tile_issue_cycles", lambda rows, *_args, **_kwargs: rows**2)
    assert FP.predict(entry, machine, batch_slices=5)["predicted_cycles"] == 5 * 3**2
    assert FP.predict({**entry, "M": 15}, machine)["predicted_cycles"] == 15**2
    with pytest.raises(ValueError, match="positive integer"):
        FP.predict(entry, machine, batch_slices=0)


def test_application_members_retain_each_batch_shape_without_changing_slice_form(monkeypatch):
    from merlin.targetgen import group_capsule_entries as capsules
    from merlin.xdsl_dialects.lowering import compute_groups as groups_module
    from merlin.xdsl_dialects.lowering import group_command as command

    groups = [SimpleNamespace(index=i, placement="device", root=object()) for i in (0, 1)]
    monkeypatch.setattr(groups_module, "form_groups", lambda *_args, **_kwargs: groups)
    monkeypatch.setattr(capsules, "activation_source", lambda *_args: "intermediate")
    monkeypatch.setattr(capsules, "host_reduction_forms", lambda *_args: [])
    monkeypatch.setattr(
        command,
        "program",
        lambda group, **_kwargs: command.GroupProgram(
            entry={"op": "matmul", "M": 3, "K": 5, "N": 6, "epilogue": []},
            stored_operand=1,
            transposed=False,
            batch_shape=(2,) if group.index == 0 else (5,),
        ),
    )
    machine = SimpleNamespace(array_rows=4, array_cols=4, refusals={})
    found = FP.application_members("synthetic", None, SimpleNamespace(operand_dtype="int8"), machine=machine)
    first, second = found["members"]
    assert (first["batch_shape"], second["batch_shape"]) == ([2], [5])
    assert (first["batch_slices"], second["batch_slices"]) == (2, 5)
    assert first["entry"] == second["entry"]
    assert second["price"]["predicted_cycles"] * 2 == first["price"]["predicted_cycles"] * 5
    (form,) = FP.aggregate({"iteration": found})["classes"]
    assert [row["batch_shape"] for row in form["members"]] == [[2], [5]]


def test_every_class_gets_a_model_shaped_member_with_a_vendor_bar():
    requirement = {"scope": {"performance": {"forms": _scope()}}}
    entries = FP.form_perf_entries(_pw(), requirement, "f" * 64)
    assert len(entries) == 3
    pool = next(e for e in entries if e["performance"]["form"]["key"]["op"] == "window_mean")
    assert (pool["M"], pool["K"], pool["N"], pool["epilogue"]) == (8, 256, 1, ["acc_scale"])
    assert pool["cat"] == "_perf" and pool["source_role"] == "derived_sweep" and pool["label"] == "dev"
    arms = pool["performance"]["arms"]
    assert arms["vendor_reference"]["instruction_policy"] == "unrestricted"
    assert arms["candidate"]["instruction_policy"] == "experiment_declared_prohibited_instruction_roles"
    assert pool["performance"]["acceptance"]["analyzer"].startswith("perf_vendor_reference_claim.")
    assert pool["performance"]["form"]["requirement_basis"] == {"sha256": "f" * 64, "axis": "scope.performance.forms"}
    assert entries[0]["name"].startswith("PW00_")


def test_bounded_residual_form_spans_the_operand_format():
    key = {"placement": "device", "op": "residual_add", "activation_source": "intermediate"}
    member = _member(
        16,
        key,
        {
            "op": "residual_add",
            "M": 256,
            "N": 8,
            "epilogue": ["relu"],
            "operand_dtype": "int8",
            "lhs_scale": 1.06,
            "rhs_scale": 0.31,
            "bound_lsb": 2,
        },
        256,
    )
    summary = FP.aggregate({"iteration_cnn": {"members": [member]}})
    requirement = {"scope": {"performance": {"forms": {"schema": FP.SCHEMA, "classes": summary["classes"]}}}}
    entries = FP.form_perf_entries(_pw(), requirement, "f" * 64)
    assert entries[0]["stimulus_range"] == [-128, 127]
    capsule, _ = CS.build(
        entries[0],
        CS.CorpusBinding(
            target="synthetic",
            tile_dim=16,
            operand_dtype="int8",
            accum_dtype="i32",
            integer=True,
            tiers=["L2"],
            compare="bounded_int",
        ),
    )
    assert capsule["stimulus_range"] == [-128, 127]
    assert capsule["numeric_policy"]["atol"] == capsule["operation"]["attributes"]["bound_lsb"] == 2


def test_a_requirement_without_a_form_scope_is_blocked_not_silent():
    blocked: list = []
    assert FP.form_perf_entries(_pw(), {"scope": {"performance": {}}}, "a" * 64, blocked=blocked) == []
    assert blocked and blocked[0]["status"] == "blocked_unimplemented" and blocked[0]["family"] == "PW"


def test_coverage_requires_a_member_and_a_vendor_bar_above_the_threshold():
    scope = _scope()
    requirement = {"scope": {"performance": {"forms": scope}}}
    entries = FP.form_perf_entries(_pw(), requirement, "f" * 64)
    threshold = _pw()["requires_form_scope"]["min_predicted_cycle_share"]
    complete = FP.form_perf_coverage(requirement, entries, threshold=threshold)
    assert complete["status"] == "complete" and not complete["missing"]
    assert all(row["ratio_status"] == "unmeasured" for row in complete["classes"])
    assert complete["claim_model_check"] == {
        "visibility": "owner_only_after_phase1_freeze",
        "mode": "statistics_only",
        "writes_capsules": False,
        "status": "deferred_until_phase1_freeze",
    }
    without_pool = [e for e in entries if e["performance"]["form"]["key"]["op"] != "window_mean"]
    incomplete = FP.form_perf_coverage(requirement, without_pool, threshold=threshold)
    assert incomplete["status"] == "incomplete" and len(incomplete["missing"]) == 1
    unbarred = copy.deepcopy(entries)
    del unbarred[0]["performance"]["arms"]["vendor_reference"]
    assert FP.form_perf_coverage(requirement, unbarred, threshold=threshold)["status"] == "incomplete"


def test_form_coverage_reports_joint_iteration_extent_gap_without_claim_shapes():
    key = {"placement": "device", "op": "matmul", "activation_source": "intermediate"}
    # The costliest representative has more MACs, yet it does not jointly bound
    # a deeper sibling. A form-level capsule is present; performance scale is not proved.
    summary = FP.aggregate(
        {
            "iteration": {
                "members": [
                    _member(1, key, {"op": "matmul", "M": 64, "K": 16, "N": 64}, 500),
                    _member(2, key, {"op": "matmul", "M": 16, "K": 128, "N": 16}, 100),
                ]
            }
        }
    )
    requirement = {"scope": {"performance": {"forms": {"schema": FP.SCHEMA, "classes": summary["classes"]}}}}
    entries = FP.form_perf_entries(_pw(), requirement, "f" * 64)
    report = FP.form_perf_coverage(requirement, entries, threshold=0.01)
    assert report["status"] == "complete"  # Existing form-presence gate is unchanged.
    diagnostic = report["classes"][0]["joint_extent_diagnostic"]
    assert diagnostic["status"] == "observed_all_bounded"
    assert (diagnostic["iteration_groups"], diagnostic["bounded_groups"]) == (2, 2)

    # Without the added mechanism witness, the diagnostic still reports the
    # source-derived gap instead of treating the primary form as scale proof.
    primary_only = FP.form_perf_coverage(requirement, entries[:1], threshold=0.01)
    diagnostic = primary_only["classes"][0]["joint_extent_diagnostic"]
    assert diagnostic["status"] == "observed_no_witness"
    assert (diagnostic["iteration_groups"], diagnostic["bounded_groups"]) == (2, 1)
    assert diagnostic["unbounded_groups"] == [{"application": "iteration", "group": 2}]

    emitted = copy.deepcopy(entries[0])
    emitted.pop("op")
    emitted["operation"] = {"op": "matmul", "attributes": {"lhs": "A0", "weight": "W"}}
    emitted["inputs"] = [
        {"name": "A0", "shape": [8, 16]},
        {"name": "W", "shape": [16, 8]},
    ]
    # The report reads emitted operands, not a stale copied representative price.
    narrowed = FP.form_perf_coverage(requirement, [emitted], threshold=0.01)
    assert narrowed["classes"][0]["joint_extent_diagnostic"]["bounded_groups"] == 0


def test_joint_extent_witnesses_use_observed_iteration_groups_and_keep_paired_arms():
    key = {"placement": "device", "op": "matmul", "activation_source": "intermediate"}
    summary = FP.aggregate(
        {
            "iteration": {
                "members": [
                    _member(1, key, {"op": "matmul", "M": 64, "K": 16, "N": 64}, 500),
                    _member(2, key, {"op": "matmul", "M": 16, "K": 128, "N": 16}, 100),
                    _member(3, key, {"op": "matmul", "M": 8, "K": 8, "N": 8}, 10),
                ]
            }
        }
    )
    requirement = {"scope": {"performance": {"forms": {"schema": FP.SCHEMA, "classes": summary["classes"]}}}}
    entries = FP.form_perf_entries(_pw(), requirement, "f" * 64)
    assert {(e["M"], e["K"], e["N"]) for e in entries} == {(64, 16, 64), (16, 128, 16)}
    assert entries[0]["name"].startswith("PW00_")
    assert entries[1]["performance"]["form"]["representative"]["group"] == 2
    assert all(set(e["performance"]["arms"]) == {"candidate", "vendor_reference"} for e in entries)
    assert entries[1]["performance"]["arms"]["vendor_reference"]["demand_equal_entry"] == {"M": 16, "K": 128, "N": 16}
    capsule, interface = CS.build(
        entries[1],
        CS.CorpusBinding(
            target="synthetic",
            tile_dim=16,
            operand_dtype="int8",
            accum_dtype="i32",
            integer=True,
            tiers=["L2"],
            compare="exact_int",
        ),
    )
    assert capsule["operation"]["op"] == "matmul" and "merlin_iface.matmul" in interface
    coverage = FP.form_perf_coverage(requirement, entries, threshold=0.01)
    assert coverage["classes"][0]["joint_extent_diagnostic"]["status"] == "observed_all_bounded"
    window = {
        "signature": "k3x3/s1x1/d1x1/pad1x1",
        "kernel": [3, 3],
        "stride": [1, 1],
        "dilation": [1, 1],
        "pad_before": [1, 1],
        "pad_after": [1, 1],
    }
    with_window = {**requirement, "conv_geometry": {"required": [window]}}
    # A matmul extent witness cannot discharge the independent source-window obligation.
    assert FP.form_perf_coverage(with_window, entries, threshold=0.01)["source_windows_without_form"] == [
        window["signature"]
    ]


def test_joint_extent_witness_is_capped_at_one_per_class_and_leaves_other_gaps_visible():
    key = {"placement": "device", "op": "matmul", "activation_source": "intermediate"}
    summary = FP.aggregate(
        {
            "iteration": {
                "members": [
                    _member(1, key, {"op": "matmul", "M": 64, "K": 16, "N": 64}, 500),
                    _member(2, key, {"op": "matmul", "M": 16, "K": 128, "N": 16}, 100),
                    _member(3, key, {"op": "matmul", "M": 128, "K": 8, "N": 8}, 50),
                ]
            }
        }
    )
    requirement = {"scope": {"performance": {"forms": {"schema": FP.SCHEMA, "classes": summary["classes"]}}}}
    entries = FP.form_perf_entries(_pw(), requirement, "f" * 64)
    assert len(entries) == 2
    assert entries[1]["performance"]["form"]["mechanism"] == "observed_joint_extent_witness"
    diagnostic = FP.form_perf_coverage(requirement, entries, threshold=0.01)["classes"][0]["joint_extent_diagnostic"]
    assert diagnostic["status"] == "observed_no_witness"
    assert diagnostic["unbounded_groups"] == [{"application": "iteration", "group": 3}]


def test_source_convolution_window_is_not_silently_covered_by_integer_matmul_forms():
    scope = _scope()
    requirement = {
        "scope": {"performance": {"forms": scope}},
        "conv_geometry": {
            "required": [
                {
                    "signature": "k3x3/s1x1/d1x1/pad1x1",
                    "kernel": [3, 3],
                    "stride": [1, 1],
                    "dilation": [1, 1],
                    "pad_before": [1, 1],
                    "pad_after": [1, 1],
                }
            ]
        },
    }
    entries = FP.form_perf_entries(_pw(), requirement, "f" * 64)
    coverage = FP.form_perf_coverage(requirement, entries, threshold=0.01)
    assert coverage["missing"] == []
    assert coverage["source_windows_without_form"] == ["k3x3/s1x1/d1x1/pad1x1"]
    assert coverage["status"] == "incomplete"
    phase2 = {"status": "complete", "blockers": []}
    FP.attach_to_phase2_report(phase2, coverage)
    assert phase2["status"] == "incomplete"
    assert phase2["blockers"][0]["source_windows_without_form"] == coverage["source_windows_without_form"]
    scope["classes"].append(
        {
            "class_id": "source-window",
            "label": "conv2d_k3x3s1_raw",
            "key": {
                "op": "conv2d",
                "geometry": {"kh": 3, "kw": 3, "stride": [1, 1], "padding": [1, 1, 1, 1]},
            },
            "max_share": 0.1,
        }
    )
    entries.append(
        {
            "name": "source-window-capsule",
            "performance": {"form": {"class_id": "source-window"}, "arms": {"vendor_reference": {}}},
        }
    )
    covered = FP.form_perf_coverage(requirement, entries, threshold=0.01)
    assert covered["status"] == "complete"
    assert covered["source_windows_without_form"] == []
    complete_phase2 = {"status": "complete", "blockers": []}
    FP.attach_to_phase2_report(complete_phase2, covered)
    assert complete_phase2["status"] == "complete" and complete_phase2["blockers"] == []


def test_source_window_performance_member_reuses_independent_functional_synthesis():
    window = {
        "signature": "k3x3/s1x1/d1x1/pad1x1",
        "kernel": [3, 3],
        "stride": [1, 1],
        "dilation": [1, 1],
        "pad_before": [1, 1],
        "pad_after": [1, 1],
    }
    functional = {
        "name": "SY_conv_from_iteration",
        "source_role": "derived_sweep",
        "generalization": {"generalization_axis": "conv_window", "conv_window": window["signature"]},
        "op": "conv2d",
        "operand_dtype": "i8",
        "kh": 3,
        "kw": 3,
        "stride": [1, 1],
        "dilation": [1, 1],
        "padding": [1, 1, 1, 1],
        "Himg": 4,
        "Wimg": 4,
        "ci": 4,
        "N": 16,
    }
    requirement = {"scope": {"performance": {"forms": _scope()}}, "conv_geometry": {"required": [window]}}
    entries = FP.form_perf_entries(_pw(), requirement, "f" * 64, source_window_entries=[functional])
    source = next(e for e in entries if e["performance"]["form"].get("source_window_signature"))
    assert source["op"] == "conv2d" and source["kh"] == 3 and source["padding"] == [1, 1, 1, 1]
    assert source["performance"]["form"]["source_member"] == functional["name"]
    assert source["performance"]["arms"]["vendor_reference"]["demand_equal_entry"]["dilation"] == [1, 1]
    capsule, interface = CS.build_conv2d(
        source,
        CS.CorpusBinding(
            target="synthetic",
            tile_dim=16,
            operand_dtype="int8",
            accum_dtype="i32",
            integer=True,
            tiers=["L2"],
            compare="exact_int",
        ),
    )
    assert capsule["operation"]["attributes"]["padding"] == window["pad_before"] + window["pad_after"]
    assert "merlin_iface.conv2d" in interface
    emitted = [*entries[:-1], {**capsule, "performance": source["performance"]}]
    assert FP.form_perf_coverage(requirement, emitted, threshold=0.01)["status"] == "complete"
    assert FP.form_perf_coverage(requirement, entries, threshold=0.01)["status"] == "complete"
    wrong = copy.deepcopy(functional)
    wrong["padding"] = [0, 0, 0, 0]
    blocked = []
    mismatched = FP.form_perf_entries(_pw(), requirement, "f" * 64, source_window_entries=[wrong], blocked=blocked)
    assert blocked and "no exact derived functional member" in blocked[0]["reason"]
    assert FP.form_perf_coverage(requirement, mismatched, threshold=0.01)["status"] == "incomplete"


def test_form_scope_declaration_and_held_out_refusal(tmp_path):
    bad = _pw()
    bad["requires_form_scope"]["min_predicted_cycle_share"] = 0
    with pytest.raises(ValueError, match="min_predicted_cycle_share"):
        FP.validate_form_scope_declaration(bad, owner="PW")
    with pytest.raises(ValueError, match="held-out"):
        FP.derive_form_scope(
            "synthetic",
            {"resnet34": tmp_path / "x.mlir"},
            SimpleNamespace(),
            iteration_roster=["resnet34"],
            held_out=["resnet34"],
        )


def test_the_claim_model_statistic_refuses_before_the_phase1_freeze(tmp_path):
    with pytest.raises(ValueError, match="after the Phase-1 freeze"):
        FP.claim_model_form_statistics(
            "synthetic",
            tmp_path / "m.mlir",
            _scope(),
            SimpleNamespace(),
            phase1_freeze_receipt=tmp_path / "missing.json",
        )


def test_claim_extent_audit_requires_one_observed_public_group_to_dominate_all_axes():
    scope = _scope()
    public = {row["class_id"]: row for row in scope["classes"]}
    claims = [
        _member(1, _STEM, {"op": "matmul", "M": 2, "K": 8, "N": 4}, 5),
        _member(2, _BODY, {"op": "matmul", "M": 512, "K": 72, "N": 8}, 10),
        _member(
            3,
            {"placement": "device", "op": "matmul", "activation_source": "unseen"},
            {"op": "matmul", "M": 8, "K": 8, "N": 8},
            20,
        ),
    ]
    assert FP._iteration_extent_shares(claims, public, "predicted_cycles") == (5.0, 10.0)

    # Matching per-slice M/K/N does not witness an unobserved batch multiplicity.
    larger_batch = [{**claims[0], "batch_slices": 3}]
    assert FP._iteration_extent_shares(larger_batch, public, "predicted_cycles") == (0.0, 5.0)

    # Maxima from different public groups must not combine into an unseen 3-D regime.
    key = _BODY
    cid = FP.class_id(key)
    split = {
        cid: {
            "members": [
                {"entry": {"op": "matmul", "M": 512, "K": 16, "N": 8}},
                {"entry": {"op": "matmul", "M": 16, "K": 72, "N": 8}},
            ]
        }
    }
    assert FP._iteration_extent_shares(claims[1:2], split, "predicted_cycles") == (0.0, 10.0)


def test_claim_form_statistics_require_the_actual_frozen_submission(tmp_path):
    from merlin.common import oot_repo

    run = tmp_path / "run"
    submission = run / "submission"
    submission.mkdir(parents=True)
    (submission / "compiler.py").write_text("pass\n")
    repo = oot_repo.init(run / "oot")
    committed = oot_repo.commit_candidate(repo, submission, label="freeze", when=1, run_id="run")
    oot_repo.tag(repo, oot_repo.FROZEN_TAG, committed.commit)
    receipt = run / "freeze.json"
    receipt.write_text(
        json.dumps(
            {
                "submission_sha256": committed.package_digest,
                "submission_files": committed.n_files,
                "oot": {
                    "repo": str(repo),
                    "frozen_commit": committed.commit,
                    "package_digest": committed.package_digest,
                },
            }
        )
    )
    assert FP._verified_phase1_freeze_digest(receipt) == hashlib.sha256(receipt.read_bytes()).hexdigest()

    (submission / "compiler.py").write_text("changed\n")
    with pytest.raises(ValueError, match="changed after its freeze"):
        FP._verified_phase1_freeze_digest(receipt)
    (submission / "compiler.py").write_text("pass\n")
    receipt.write_text(json.dumps({"submission_sha256": committed.package_digest}))
    with pytest.raises(ValueError, match="bound frozen OOT submission"):
        FP._verified_phase1_freeze_digest(receipt)


def test_the_shared_template_loads_the_form_family_and_the_stream_families(tmp_path):
    profile: dict = {"capsules": []}
    profiles._merge_shared_perf(
        profile,
        source=tmp_path / "recipe.yaml",
        performance_template=repo_root() / "experiments/templates/phase0/performance.yaml",
    )
    families = {row["family"]: row for row in profile["_performance_template"]["families"]}
    assert families["PW"]["claim"] == "DIFFERENTIAL"
    assert {"PD", "PA", "PJ"} <= set(families) and families["PD"]["claim"] == "EMITS"
    blocked = {row["family"] for row in profile["_performance_template"]["blocked_unimplemented"]}
    assert "PT" in blocked and blocked.isdisjoint({"PD", "PA", "PJ"})
    with pytest.raises(ValueError, match="performance.claim"):
        profiles._validate_performance_block(
            {**next(s for s in _template()["sweeps"] if s["id"] == "PK")["base"]["performance"], "claim": "CERTIFIES"},
            owner="PK",
        )


def test_a_group_statement_cannot_relabel_the_template_family():
    scope = _scope()
    for row in scope["classes"]:
        for member in row["members"]:
            member["entry"].update(kind="op", cat="layers", label="public", source_role="model_derived")
    entries = FP.form_perf_entries(_pw(), {"scope": {"performance": {"forms": scope}}}, "f" * 64)
    assert {(e["kind"], e["cat"], e["label"], e["source_role"]) for e in entries} == {
        ("model_slice", "_perf", "dev", "derived_sweep")
    }


_ROWS = {"placement": "host_region", "op": "row_sum", "input_rank": 3, "reduced_dims": 1}


def _tolerance_applications() -> dict:
    """A one-element reduction group (costliest) beside a two-element one, and a class of only 1x1s."""
    return {
        "multimodal_policy": {
            "members": [
                _member(22, _ROWS, {"op": "matmul", "M": 1, "K": 8, "N": 1, "epilogue": []}, 900, placement="host"),
                _member(23, _ROWS, {"op": "matmul", "M": 1, "K": 8, "N": 2, "epilogue": []}, 100, placement="host"),
                _member(4, _STEM, {"op": "matmul", "M": 1, "K": 4, "N": 1, "epilogue": []}, 50),
            ]
        }
    }


def test_a_tolerance_target_never_picks_a_one_element_representative():
    exact = {
        row["key"]["op"] if row["key"]["op"] != "matmul" else "stem": row
        for row in FP.aggregate(_tolerance_applications())["classes"]
    }
    graded = {
        row["key"]["op"] if row["key"]["op"] != "matmul" else "stem": row
        for row in FP.aggregate(_tolerance_applications(), tolerance_graded=True)["classes"]
    }
    # An exact-integer target keeps the costliest member, whatever its size.
    assert exact["row_sum"]["members"][exact["row_sum"]["representative"]]["group"] == 22
    assert exact["stem"]["representative"] == 0
    # Under a tolerance the costliest member writes one element, so the two-element one represents it.
    assert graded["row_sum"]["members"][graded["row_sum"]["representative"]]["group"] == 23
    # A class whose every member writes one element is skipped, with its reason, not minted.
    assert graded["stem"]["representative"] is None
    assert graded["stem"]["status"] == FP.SKIPPED_UNFALSIFIABLE and "fewer than 2" in graded["stem"]["reason"]


def test_an_unfalsifiable_class_is_skipped_and_reported_not_failed():
    summary = FP.aggregate(_tolerance_applications(), tolerance_graded=True)
    requirement = {"scope": {"performance": {"forms": {"schema": FP.SCHEMA, "classes": summary["classes"]}}}}
    skipped: list = []
    entries = FP.form_perf_entries(_pw(), requirement, "f" * 64, skipped=skipped)
    assert [e["name"] for e in entries] == [
        "PW00_" + next(r["label"] for r in summary["classes"] if r["key"]["op"] == "row_sum")
    ]
    assert len(skipped) == 1 and skipped[0]["status"] == FP.SKIPPED_UNFALSIFIABLE and skipped[0]["class_id"]
    coverage = FP.form_perf_coverage(requirement, entries, threshold=0.01)
    assert coverage["skipped_unfalsifiable"] == [skipped[0]["class_id"]]
    assert coverage["status"] == "complete" and not coverage["missing"]
    assert {r["status"] for r in coverage["classes"]} == {"covered", FP.SKIPPED_UNFALSIFIABLE}


def _strata_requirement():
    key = {"placement": "device", "op": "matmul", "activation_source": "intermediate"}
    summary = FP.aggregate(
        {
            "projection": {"members": [_member(1, key, {"op": "matmul", "M": 96, "K": 768, "N": 4096}, 900)]},
            "convolution": {
                "members": [
                    _member(2, key, {"op": "matmul", "M": 2048, "K": 72, "N": 16}, 800),
                    _member(3, key, {"op": "matmul", "M": 2048, "K": 72, "N": 16}, 800),
                ]
            },
            "step": {"members": [_member(4, key, {"op": "matmul", "M": 1, "K": 768, "N": 4096}, 50)]},
        }
    )
    return {"scope": {"performance": {"forms": {"schema": FP.SCHEMA, "classes": summary["classes"]}}}}


def test_the_shared_template_stratifies_form_members_by_geometry():
    assert next(s for s in _template()["sweeps"] if s["id"] == "PW")["requires_form_scope"]["stratify_geometry"]


def test_each_further_geometry_stratum_of_a_class_gets_one_observed_member():
    """One class can hold a wide projection, a tall convolution-as-GEMM and a one-row product; their
    cost is decided by different regimes, so each stratum gets its own observed member, once."""
    from merlin.capture.shape_taxonomy import classify_geometry

    requirement = _strata_requirement()
    plain = FP.form_perf_entries(_pw(), requirement, "f" * 64)
    stratified = FP.form_perf_entries(_pw(stratify=True), requirement, "f" * 64)
    assert len(stratified) > len(plain)
    extents = [(e["M"], e["K"], e["N"]) for e in stratified]
    assert len(extents) == len(set(extents)), "no work is measured twice"
    geometries = [classify_geometry(m, n, k) for m, k, n in extents]
    assert len(geometries) == len(set(geometries)), "at most one member per stratum"
    added = [e for e in stratified if e["performance"]["form"].get("mechanism") == FP.GEOMETRY_WITNESS]
    assert added and all(e["name"].startswith("PWG") for e in added)
    assert all(e["performance"]["form"]["geometry"] in geometries for e in added)
    assert all(set(e["performance"]["arms"]) == {"candidate", "vendor_reference"} for e in added)
    row = FP.form_perf_coverage(requirement, stratified, threshold=0.01)["classes"][0]
    assert row["geometry_strata"]["unrepresented"] == []
    row = FP.form_perf_coverage(requirement, plain, threshold=0.01)["classes"][0]
    assert row["geometry_strata"]["unrepresented"], "an unminted stratum stays visible"


def test_stratify_geometry_must_be_a_boolean():
    bad = _pw()
    bad["requires_form_scope"]["stratify_geometry"] = "yes"
    with pytest.raises(ValueError, match="stratify_geometry"):
        FP.validate_form_scope_declaration(bad, owner="PW")
