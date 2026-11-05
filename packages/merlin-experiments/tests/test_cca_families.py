"""CCA family feedback: grouping, nearest bank unit with basis, exceptions, budget skip, bank lane."""

from __future__ import annotations

import json
from pathlib import Path

import merlin.kernels.mechanism_applicability as MA
from merlin.mining.reference_subset_cards import family_key, family_signature, signature_similarity
from merlin_experiments.phase1.feedback import cca_families as F


def analysis(*, roles=("operand_load", "accumulate"), fields=("LOAD.rows", "LOAD.cols"), sites=10, depth=2,
             form="systolic", stride=64, tag=""):
    """A lifted region record with only the fields the signature reads (plus counts it must ignore)."""
    return dict(
        accelerator_sites=sites,
        cfg=dict(natural_loops=depth, trip_counts=dict(proved=[dict(value=sites)])),
        cca=dict(compute=dict(contraction_form=form, accumulator_dtype="i32"),
                 dispatch=dict(n_dispatches=sites * 7, loop_offloaded=False),
                 operand_fields=dict(stride=[["LOAD.stride", [stride]]])),
        mechanism_projection=dict(
            role_sites={role: sites for role in roles},
            instruction_classes={"LOAD": sites, "COMPUTE": sites * 2},
            field_fingerprints=[dict(instruction=f.split(".")[0], field=f.split(".")[1], sites=sites,
                                     distinct_values=sites) for f in fields],
            loop_nest=dict(value=dict(depth=depth, levels=[dict(command_sites=1, header_pc=sites)] * depth)),
            counted_control=dict(status="proved", commands=sites * 7),
        ),
        evidence=dict(evidence_id="e" + tag),
    )


def member(case, identity, record, region="r0"):
    return dict(case=case, identity=identity, region_id=region, accelerator_sites=1, analysis=record)


def test_signature_ignores_counts_and_shapes_but_not_structure():
    small, large = analysis(sites=4, stride=16), analysis(sites=4000, stride=1024)
    assert family_signature(small) == family_signature(large)
    assert family_key(family_signature(small)) == family_key(family_signature(large))
    other = analysis(fields=("LOAD.rows", "LOAD.cols", "LOAD.stride"))
    assert family_key(family_signature(other)) != family_key(family_signature(small))
    assert family_signature(analysis(form="other"))["dataflow"]["compute.contraction_form"] == "other"


def test_grouping_identical_family_and_representatives():
    members = [member("a", "machine-1", analysis(sites=3)), member("b", "machine-1", analysis(sites=3)),
               member("c", "machine-2", analysis(sites=900)), member("d", "machine-3", analysis(depth=4))]
    families = F.group(members)
    assert sorted(len(rows) for rows in families.values()) == [1, 3]
    record = F.build(members, [], cycles={"c": 50, "a": 10})
    big = record["families"][0]
    assert big["members"] == 3 and big["identical_bodies"] == 2 and big["cases"] == ["a", "b", "c"]
    assert "slowest member" in big["representative_basis"]
    assert [rep["role"] for rep in big["representatives"]] == ["most_frequent", "slowest"]
    assert big["representatives"][0]["case"] == "a" and big["representatives"][1]["case"] == "c"
    assert big["nearest_unit"] is None and big["representatives"][0]["status"] == "no bank unit"
    assert "frequent body only" in record["families"][1]["representative_basis"]


def test_nearest_unit_reports_agreeing_and_differing_fields():
    mine = family_signature(analysis())
    units = [dict(unit_id="u-far", region_id="r", original_id="o", operation=None,
                  signature=family_signature(analysis(roles=("readout",), fields=("STORE.rows",), depth=5))),
             dict(unit_id="u-near", region_id="r", original_id="o", operation=None,
                  signature=family_signature(analysis(fields=("LOAD.rows", "LOAD.cols", "LOAD.stride"))))]
    match = F.nearest(mine, units)
    assert match["unit_id"] == "u-near" and "roles" in match["agree"] and "loop_nest" in match["agree"]
    assert match["differ"] == {"configured_fields": {"only_yours": [], "only_unit": ["LOAD.stride"]}}
    assert match["semantic_equivalence"] == "unknown" and "no operation class" in match["shortlist"]
    assert signature_similarity(mine, mine)["score"] == 1.0


def test_family_feedback_lists_exceptions_and_shape_dependent_members(monkeypatch):
    def differences(reference, candidate, *, structure=None):
        # Member c already matches the unit's buffering; the representative a does not.
        matches = candidate["evidence"]["evidence_id"] == "ec"
        return [dict(effect="operand_buffering", role="operand_load", status="observed",
                     difference="same" if matches else "differs",
                     direction="candidate_matches" if matches else "candidate_below",
                     reference=dict(disjoint_windows=2), candidate=dict(disjoint_windows=2 if matches else 1)),
                dict(effect="sync_sites", status="observed", difference="differs", direction="candidate_below")]

    def applicability(reference, candidate, *, shape=None, **_):
        status = "satisfied" if (shape or {}).get("m", 0) < 100 else "violated"
        rows = [dict(row, routes=[dict(capability="double_buffer", family_match=True,
                                       admission=dict(status=status))])
                for row in differences(reference, candidate)]
        return dict(differences=rows, missing_capabilities=[])

    monkeypatch.setattr(MA, "mechanism_differences", differences)
    monkeypatch.setattr(MA, "applicability", applicability)
    unit = dict(unit_id="u", region_id="r", original_id="o", operation=None, analysis=analysis(tag="unit"),
                signature=family_signature(analysis()))
    members = [member("a", "machine-1", analysis(tag="a")), member("b", "machine-1", analysis(tag="a")),
               member("c", "machine-2", analysis(tag="c"))]
    record = F.build(members, [unit], shapes={"a": {"m": 8}, "b": {"m": 8}, "c": {"m": 512}})
    family = record["families"][0]
    rep = family["representatives"][0]
    assert rep["status"] == "compared" and rep["differences"][0]["effect"] == "operand_buffering"
    assert rep["differences"][0]["unit"] == {"disjoint_windows": 2} and rep["differences"][0]["yours"] == {
        "disjoint_windows": 1}
    assert all(row["effect"] != "sync_sites" for row in rep["differences"])  # outside FACT_KINDS
    assert family["exceptions"] == [dict(case="c", region_id="r0", differs_in=["operand_buffering operand_load"])]
    assert family["shape_dependent"][0]["case"] == "c"
    assert set(family["shape_dependent"][0]["admissions"].values()) == {"violated"}
    text = "\n".join(F.render(dict(record, grade_token="7")))
    assert "## CCA families" in text and "Nearest unit `u`" in text and "Exceptions" in text
    assert "--reference-entry u --reference-region r" in text and "not semantic equivalence" in text


def test_budget_exhaustion_skips_with_a_stated_reason():
    ticks = iter(range(100))
    record = F.build([member("a", "m", analysis())], [], deadline=0, clock=lambda: next(ticks) + 1)
    assert record["families"][0]["status"] == "skipped: budget exhausted"
    assert "budget exhausted" in "\n".join(F.render(dict(record, grade_token="1")))


def test_compute_states_why_it_is_skipped(tmp_path):
    record = F.compute(tmp_path, tmp_path, pin="0" * 64, target="t")
    assert record["status"] == "skipped" and "cca-evidence" in record["reason"]
    assert "Not computed" in "\n".join(F.render(record))
    assert F.compute(tmp_path, tmp_path, pin="0" * 64, target=None)["reason"].startswith("no selected target")


def test_brief_section_only_for_bank_profiles(monkeypatch, tmp_path):
    (tmp_path / "qa").mkdir()
    (tmp_path / "qa" / F.RECORD).write_text(json.dumps(dict(schema=F.SCHEMA, status="skipped", reason="x")))
    monkeypatch.setattr(F, "_bank_pin", lambda: None)
    assert F.brief_lines(tmp_path) == []
    assert F.refresh_async(tmp_path, tmp_path) is None
    monkeypatch.setattr(F, "_bank_pin", lambda: "p" * 64)
    assert F.brief_lines(tmp_path)[0].startswith("## CCA families")


def test_mechanism_summary_marks_each_fact_observed_or_unknown():
    record = analysis()
    record["derived_effects"] = dict(facts=dict(
        operand_buffering=dict(status="observed", scope=MA.STREAM_SCOPE,
                               value=dict(operand_load=dict(disjoint_windows=2, alternating=True))),
        issue_overlap=dict(status="unknown", reason="no_proved_command_stream")))
    summary = F.mechanism_summary(record)
    assert summary["roles"] == dict(status="observed", value=dict(A=10, W=0, output=0))
    assert summary["buffering"] == dict(status="observed", value=dict(A=dict(windows=2, alternating=True)))
    assert summary["overlap"] == dict(status="unknown", reason="no_proved_command_stream")
    assert summary["prefetch_distance"]["status"] == "unknown"
    assert "overlap unknown" in F._summary_text(summary)


def test_client_prints_reply_path_and_compact_summary():
    from merlin_experiments.phase1.tools import selfcheck as client

    reply = dict(status="analyzed", detail=dict(path="selfcheck_out/details/x.json"), report=dict(
        kind="cca_bank_inventory", total=3, units=[dict(unit_id="machine-abc", mechanisms=dict(
            buffering=dict(status="observed", value={}), overlap=dict(status="unknown", reason="r")))]))
    line = client._analysis_summary_line(json.dumps(reply), "7")
    assert "status=analyzed" in line and "kind=cca_bank_inventory" in line and "total=3" in line
    assert "machine-abc: observed buffering" in line and ".qa_channel/resp_7.json" in line
    assert "selfcheck_out/details/x.json" in line
    source = Path(client.__file__).read_text()
    assert "reply will be written to .qa_channel/resp_" in source and '"reply_path"' in source


def test_applicability_summary_leads_with_observed_and_groups_unknowns():
    from merlin_experiments.phase1.feedback.reference_applicability import summary

    rows = [dict(effect="operand_buffering", role="operand_load", difference="differs", reference=2, candidate=1,
                 routes=[dict(capability="double_buffer", family_match=True)]),
            dict(effect="issue_overlap", difference="unknown", reason="no_proved_command_stream"),
            dict(effect="prefetch_distance", difference="unknown", candidate=dict(reason="no_proved_command_stream")),
            dict(effect="sync_sites", difference="same")]
    out = summary(rows)
    assert out["observed_differences"] == [dict(effect="operand_buffering operand_load", unit=2, yours=1,
                                                routes=["double_buffer"])]
    assert out["same"] == 1
    assert out["unknown_by_reason"] == [dict(reason="no_proved_command_stream",
                                             effects=["issue_overlap", "prefetch_distance"], count=2)]


def test_unit_prefix_selects_one_unit_or_lists_candidates():
    import pytest

    from merlin_experiments.phase1.feedback import reference_analysis as A
    from merlin_experiments.phase1.feedback import reference_bank as RB

    bank = dict(units=[dict(kernel_id="machine-aa11", card="c", reports=["m/o/r1/analysis.json", "m/o/r1/cca.json",
                                                                        "m/o/r1/cca.mlir", "m/o/r1/instructions.dis"]),
                       dict(kernel_id="machine-aa22", card="d", reports=[])])
    unit, occurrences = RB._select(dict(entry="machine-aa1"), bank)
    assert unit["kernel_id"] == "machine-aa11" and occurrences[0]["region_id"] == "r1"
    with pytest.raises(A._AnalysisGap) as error:
        RB._select(dict(entry="machine-aa"), bank)
    assert str(error.value) == "unknown_or_ambiguous_unit_id"
    assert error.value.detail["candidates"] == ["machine-aa11", "machine-aa22"]
    assert "unknown_or_ambiguous_unit_id" in A._GAPS


#: Fake ELF bytes -> the lifted record each one stands for.
_BODIES = {b"a": dict(), b"a2": dict(sites=900), b"b": dict(depth=4)}


def _selfcheck_fixture(monkeypatch, tmp_path, cases):
    """A self-check raw runs root with one linked kernel per case, and the lifting/bank seams stubbed."""
    runs = tmp_path / "selfcheck_x"
    for case, body in cases.items():
        generated = runs / "runs" / "suite" / case / "generated"
        generated.mkdir(parents=True)
        (generated.parent / "capsule_result.json").write_text("{}")
        (generated / "package_kernel.elf").write_bytes(body)
    calls = dict(lift=0, bank=0)

    def lift(path, **_):
        calls["lift"] += 1
        body = Path(path).read_bytes()
        return [dict(region_id="region-" + body.decode(), identity="machine-" + body.decode(), accelerator_sites=1,
                     analysis=analysis(**_BODIES[body]))]

    def bank(workspace, pin, root=None):
        calls["bank"] += 1
        return Path(root), dict(units=[]), dict(contract="c", facts="f")

    near = dict(unit_id="machine-near", region_id="region-u", original_id="o", symbol="tiled_matmul_ws",
                operation=None, analysis=analysis(tag="unit"), signature=family_signature(analysis()))
    far = dict(unit_id="machine-far", region_id="region-v", original_id="o", symbol="conv_ws", operation=None,
               analysis=analysis(tag="far"), signature=family_signature(analysis(roles=("readout",), depth=5)))
    monkeypatch.setattr(F, "lift_kernel", lift)
    monkeypatch.setattr(F, "open_selected_bank", bank)
    monkeypatch.setattr(F, "_decoder_for", lambda files, target: ({}, None))
    monkeypatch.setattr(F, "_units_for", lambda root, bank, pin: [far, near])
    monkeypatch.setattr(F, "case_semantics", lambda workspace, cases: ({}, {}))
    monkeypatch.setattr(MA, "selected_geometry", lambda target, facts: (None, "no geometry in this test"))
    monkeypatch.setattr(MA, "applicability", lambda reference, candidate, **_: dict(differences=[
        dict(effect="issue_overlap", status="unknown", difference="unknown", reason="no_proved_command_stream"),
        dict(effect="operand_buffering", role="operand_load", difference="differs", reference=dict(
            disjoint_windows=2), candidate=dict(disjoint_windows=1)),
        dict(effect="operand_load_per_compute", difference="differs", reference=0.25, candidate=2.0,
             comparison="reference 0.25, candidate 2.0, candidate higher")], missing_capabilities=[]))
    monkeypatch.setattr(MA, "mechanism_differences", lambda reference, candidate, **_: [])
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace, runs, calls


def _section(workspace, runs, **kwargs):
    kwargs.setdefault("arm", dict(cca=True, bank_sha256="p" * 64))
    return F.selfcheck_section(workspace, runs, bank_root=runs.parent, target="t", facts={}, **kwargs)


def test_selfcheck_kernels_reads_the_runs_layout(tmp_path, monkeypatch):
    _, runs, _ = _selfcheck_fixture(monkeypatch, tmp_path, {"mm_a": b"a"})
    (runs / "runs" / "suite" / "no_elf").mkdir()
    (runs / "runs" / "suite" / "no_elf" / "capsule_result.json").write_text("{}")
    for root in (runs, runs / "runs", runs / "runs" / "suite"):
        kernels, missing = F.selfcheck_kernels(root)
        assert sorted(kernels) == ["mm_a"] and "no linked kernel ELF" in missing["no_elf"]
    kernels, missing = F.selfcheck_kernels(runs, cases=["mm_a", "absent"])
    assert list(kernels) == ["mm_a"] and missing == {"absent": "no result for this case in the self-check output"}


def test_selfcheck_section_groups_cases_and_names_the_nearest_unit(tmp_path, monkeypatch):
    workspace, runs, _ = _selfcheck_fixture(monkeypatch, tmp_path, {"mm_a": b"a", "mm_big": b"a2", "conv": b"b"})
    out = _section(workspace, runs, cases=["mm_a", "mm_big", "conv"])
    record, text = out["record"], out["text"]
    assert record["status"] == "compared" and record["cache"] == "miss"
    families = {family["label"]: family for family in record["families"]}
    # The two matmul kernels differ only in counts: one family; the deeper loop nest is another.
    assert families["F1"]["cases"] == ["mm_a", "mm_big"] and families["F2"]["cases"] == ["conv"]
    assert record["cases"]["mm_big"]["kernels"] == [dict(region_id="region-a2", family="F1")]
    nearest = families["F1"]["nearest_unit"]
    assert nearest["unit_id"] == "machine-near" and nearest["symbol"] == "tiled_matmul_ws"
    assert "roles" in nearest["agree"]
    assert "Nearest unit `tiled_matmul_ws` `machine-near`" in text and "-> F1" in text
    # Order facts (comparable numbers) lead the differences; unknowns are grouped by reason.
    assert [row["effect"] for row in families["F1"]["top_differences"]] == [
        "operand_load_per_compute", "operand_buffering"]
    assert text.index("operand_load_per_compute") < text.index("operand_buffering operand_load")
    assert "Unknown (no_proved_command_stream): issue_overlap" in text
    assert "Mechanisms, unit | `mm_a` region-a | `mm_big` region-a2:" in text and "    width:" in text
    assert "not semantic equivalence" in text and len(text.encode()) <= F.SELFCHECK_TEXT_BYTES
    detail = json.loads(Path(out["detail_path"]).read_text())
    assert out["detail_path"].endswith(".json") and record["detail"] in text
    assert detail["families"][0]["nearest_mechanisms"]["roles"]["status"] == "observed"


def test_selfcheck_section_text_is_bounded(tmp_path, monkeypatch):
    workspace, runs, _ = _selfcheck_fixture(monkeypatch, tmp_path, {"mm_a": b"a", "conv": b"b"})
    out = _section(workspace, runs, limit=700)
    assert len(out["text"].encode()) <= 700 and "more lines in the full record" in out["text"]
    assert out["record"]["detail"] in out["text"]


def test_selfcheck_section_caches_by_elf_digest(tmp_path, monkeypatch):
    workspace, runs, calls = _selfcheck_fixture(monkeypatch, tmp_path, {"mm_a": b"a", "conv": b"b"})
    first = _section(workspace, runs)
    assert calls == dict(lift=2, bank=1)
    again = _section(workspace, runs)
    assert calls == dict(lift=2, bank=1) and again["record"]["cache"] == "hit"
    assert "cached (kernels unchanged)" in again["text"]
    assert again["record"]["families"] == first["record"]["families"]
    # One changed kernel: a new record, but the unchanged kernel's lift comes from its digest's cache.
    (runs / "runs" / "suite" / "mm_a" / "generated" / "package_kernel.elf").write_bytes(b"a2")
    changed = _section(workspace, runs)
    assert changed["record"]["cache"] == "miss" and calls == dict(lift=3, bank=2)


def test_selfcheck_section_budget_skips_with_a_stated_reason(tmp_path, monkeypatch):
    workspace, runs, calls = _selfcheck_fixture(monkeypatch, tmp_path, {"mm_a": b"a"})
    ticks = iter(range(1000))
    out = _section(workspace, runs, budget_s=0, clock=lambda: next(ticks))
    assert out["record"]["status"] == "skipped" and "budget of 0s exhausted" in out["record"]["reason"]
    assert out["detail_path"] is None and "not computed for this self-check: budget" in out["text"]
    assert calls["lift"] == 0
    # Past the budget mid-way: partial, stated, and never cached as complete.
    ticks = iter(range(1000))
    out = _section(workspace, runs, budget_s=2, clock=lambda: next(ticks))
    assert out["record"]["status"] == "partial" and "budget was exhausted" in out["text"]
    assert _section(workspace, runs)["record"]["cache"] == "miss"


def test_selfcheck_section_disabled_arm_returns_nothing(tmp_path, monkeypatch):
    workspace, runs, calls = _selfcheck_fixture(monkeypatch, tmp_path, {"mm_a": b"a"})

    class Profile:
        bank_sha256 = None

    for arm in (False, None, dict(cca=False, bank_sha256="p" * 64), dict(), Profile()):
        assert _section(workspace, runs, arm=arm) is None
    assert calls == dict(lift=0, bank=0) and not (workspace / "selfcheck_out").exists()
    Profile.bank_sha256 = "p" * 64
    assert _section(workspace, runs, arm=Profile())["record"]["status"] == "compared"
    # An enabled arm that cannot compute says why; it never claims the service is off.
    skipped = F.selfcheck_section(workspace, tmp_path / "empty", arm=True, bank_root=runs.parent, target="t")
    assert skipped["record"]["status"] == "skipped" and "no linked kernel ELF" in skipped["text"]
    assert "disabled" not in skipped["text"] and "not enabled" not in skipped["text"]


def test_bank_units_name_each_occurrence_by_its_card_symbol(monkeypatch):
    from merlin_experiments.phase1.feedback import reference_bank as RB

    reports = [f"m/o/r1/{name}" for name in ("analysis.json", "cca.json", "cca.mlir", "instructions.dis")]
    files = {"u/card.json": json.dumps(dict(kernel_id="machine-1", occurrences=[
        dict(occurrence_id="o:r1", symbol="tiled_matmul_ws")])), reports[0]: json.dumps(analysis())}
    bank = dict(units=[dict(kernel_id="machine-1", card="u/card.json", reports=reports)],
                files={name: "x" for name in files})
    monkeypatch.setattr(RB, "_pinned", lambda root, bank, relative: (files[relative], {}))
    (unit,) = F.bank_units(Path("."), bank)
    assert unit["symbol"] == "tiled_matmul_ws" and unit["region_id"] == "r1"
    files["u/card.json"] = json.dumps(dict(kernel_id="machine-1", occurrences=[dict(occurrence_id="o:r1")]))
    assert F.bank_units(Path("."), bank)[0]["symbol"] is None


def test_inventory_extents_carry_trip_counts_transfer_shapes_and_symbol():
    from merlin_experiments.phase1.feedback import reference_bank as RB

    record = analysis()
    record.update(symbol="kernel", derived_effects=dict(facts=dict(role_transfer_width=dict(status="observed", value=dict(
        data_movement=dict(operand_load=dict(shapes=[dict(rows=16, cols=64, count=3)])))))))
    extents = RB._extents(record)
    assert extents["loop_trip_counts"] == [10] and extents["loop_trip_counts_status"] == "proved"
    assert extents["transfer_shapes"] == dict(operand_load=[dict(rows=16, cols=64, count=3)])
    assert extents["entry_symbol"] == "kernel"
