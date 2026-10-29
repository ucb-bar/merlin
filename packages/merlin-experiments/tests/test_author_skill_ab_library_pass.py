"""The ab-library-pass author skill: its comparison reads real response shapes and its steps stay current.

``scripts/ab_compare.py`` compares two self-check responses (or a variant with the newest submission/
rows) by cycles of one kind and identity, with each row's numerics; ``scripts/mechanism_api.py`` finds
a catalogue mechanism's API in the editable library. The SKILL.md steps name self-check flags and brief
fields; these tests fail when those drift from the shim and the brief that define them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from author_skill_fixtures import assert_generic_skill, front_matter, load_script

from merlin.common.paths import module_source_path

SKILL = "ab-library-pass"
GSIM = {"engine": "sim-a", "simulator_sha256": "1" * 64}
MODEL = {key: str(index) * 64 for index, key in enumerate((
    "selection_sha256", "model_sha256", "facts_sha256", "comparison_domain_sha256",
    "scope_sha256", "capsule_sha256"), start=1)}
MODEL.update(timing_mode="untuned", timing_epoch=0)


@pytest.fixture(scope="module")
def ab():
    return load_script(SKILL, "ab_compare")


@pytest.fixture(scope="module")
def api():
    return load_script(SKILL, "mechanism_api")


def _core_row(name, cycles, *, passed=True, modeled=None, identity=GSIM, model=MODEL):
    """A core self-check ``per_capsule`` row as the broker returns it (fields the comparison reads)."""
    return {"capsule": name, "pass": passed, "numerical_execution_passed": passed,
            "barrier_tier": "L3", "barrier_status": "pass" if passed else "fail",
            "barrier_cycles": cycles if passed else None, "barrier_timing_identity": identity,
            "performance": {"status": "PROVISIONAL", "complete_event_accounting": True,
                            "estimated_whole_soc_cycles": modeled, **model}}


def _scalability_row(name, cycles, *, passed=True, model=MODEL):
    return {"capsule": name, "status": "pass" if passed else "fail",
            "tiers": {"L0": "pass", "L1": "pass", "L2": "pass" if passed else "fail"},
            "performance": {"status": "PROVISIONAL", "complete_event_accounting": True,
                            "estimated_whole_soc_cycles": cycles, **model}}


def _response(rows, *, tree="submission", digest="a" * 64, at=1, key="per_capsule", overrides=None):
    document = {"selfcheck_submission": tree, "submission_sha256": digest, "graded_at_unix_ns": at,
                "all_pass": all(r.get("pass", True) for r in rows), key: rows}
    if tree != "submission":
        document["variant"] = {"name": tree.partition("/")[2], "path": tree, "sha256": "b" * 64, "official": False}
    if overrides is not None:
        document["variant"] = {"name": None, "path": "submission", "kind": "knob_overrides",
                               "tuning_overrides": overrides, "sha256": "c" * 64, "official": False}
    return document


def _channel(tmp_path: Path, **responses) -> Path:
    channel = tmp_path / ".qa_channel"
    channel.mkdir(exist_ok=True)
    for rid, document in responses.items():
        (channel / f"resp_{rid}.json").write_text(json.dumps(document))
    return channel


def test_skill_is_generic_bounded_and_self_describing():
    assert_generic_skill(SKILL, allowed_imports=frozenset({'merlin'}))
    fields, body = front_matter(SKILL)
    assert "suggested_mechanisms" in fields["description"] and "ab_compare.py" in fields["description"]
    for script in ("ab_compare.py", "mechanism_api.py", "apply_library_pass.py"):
        assert f"$SKILL/scripts/{script}" in body


def test_named_selfcheck_flags_and_brief_fields_exist():
    _, body = front_matter(SKILL)
    shim = module_source_path("merlin_experiments.phase1.tools.selfcheck").read_text()
    flags = {word for line in body.splitlines() if "agent_selfcheck.py" in line
             for word in line.replace("`", " ").split() if word.startswith("--")}
    assert {"--capsules", "--submission", "--wait-budget", "--attach", "--out"} <= flags
    flags |= {"--suite", "--capsules"}  # named in prose: `--suite scalability`, `--capsules changed`
    assert "`--suite scalability`" in body and "`--capsules changed`" in body
    assert all(f'"{flag}"' in shim for flag in flags), sorted(f for f in flags if f'"{f}"' not in shim)
    brief = module_source_path("merlin_experiments.phase1.feedback.performance_brief").read_text()
    tuning = module_source_path("merlin_experiments.phase1.feedback.idle_tuning").read_text()
    tuning += module_source_path("merlin_experiments.phase1.feedback.idle_tuning_results").read_text()
    for field in ("largest_headroom", "lowest_utilization", "suggested_mechanisms", "workload_check",
                  "dominant_modeled_resource", "test_case"):
        assert f'"{field}"' in brief and field in body, field
    for field in ("idle_tuning", "delta_pct", "numerics", "how_to_apply"):
        assert f'"{field}"' in tuning and field in body, field
    assert "qa/idle_tuning.json" in body and 'Path("qa") / "idle_tuning.json"' in tuning


def test_measured_cycles_compare_first_and_a_faster_variant_is_kept(tmp_path, ab):
    base = _response([_core_row("a", 1000, modeled=900), _core_row("b", 500, modeled=480)])
    fast = _response([_core_row("a", 800, modeled=950), _core_row("b", 450, modeled=470)], tree="variants/fast")
    result = ab.compare(fast, base, channel=_channel(tmp_path))
    rows = {e["test_case"]: e for e in result["cases"]}
    assert rows["a"]["kind"] == "measured" and rows["a"]["cycles_before"] == 1000 and rows["a"]["delta"] == -200
    assert rows["a"]["numerics"] == "unchanged (both pass)"
    assert result["verdict"]["decision"] == "keep" and result["verdict"]["exit"] == 0
    assert result["after"] == "variants/fast" and result["after_sha256"] == "b" * 64


def test_changed_numerics_mixed_and_slower_are_not_kept(tmp_path, ab):
    channel = _channel(tmp_path)
    base = _response([_core_row("a", 1000), _core_row("b", 500)])
    broken = _response([_core_row("a", 700), _core_row("b", 0, passed=False)], tree="variants/x")
    assert ab.compare(broken, base, channel=channel)["verdict"]["decision"] == "do not keep"
    mixed = _response([_core_row("a", 700), _core_row("b", 550)], tree="variants/x")
    verdict = ab.compare(mixed, base, channel=channel)["verdict"]
    assert verdict["decision"] == "mixed" and "['b']" in verdict["why"]
    slower = _response([_core_row("a", 1100), _core_row("b", 500)], tree="variants/x")
    assert ab.compare(slower, base, channel=channel)["verdict"]["exit"] == 1


def test_cycles_of_other_simulators_or_models_are_never_compared(tmp_path, ab):
    channel = _channel(tmp_path)
    base = _response([_core_row("a", 1000, modeled=900)])
    other_sim = _response([_core_row("a", 10, modeled=850, identity={"engine": "sim-b"})], tree="variants/x")
    row = ab.compare(other_sim, base, channel=channel)["cases"][0]
    assert row["kind"] == "modeled" and row["cycles_before"] == 900 and row["cycles_after"] == 850
    other_model = _response([_core_row("a", 10, modeled=5, identity={"engine": "sim-b"},
                                       model={**MODEL, "timing_epoch": 1})], tree="variants/x")
    result = ab.compare(other_model, base, channel=channel)
    assert "different" in result["cases"][0]["status"] and result["verdict"]["decision"] == "inconclusive"


@pytest.mark.parametrize('field,value', [
    ('status', 'UNKNOWN'), ('status', 'FAILED'), ('status', 'PARTIAL'),
    ('complete_event_accounting', False), ('complete_event_accounting', None),
    ('model_sha256', None), ('facts_sha256', None), ('scope_sha256', 'not-a-digest'),
    ('selection_sha256', None), ('comparison_domain_sha256', None), ('capsule_sha256', None),
    ('estimated_whole_soc_cycles', float('inf')), ('estimated_whole_soc_cycles', float('nan')),
])
def test_unknown_partial_or_unidentified_model_never_keeps_variant(tmp_path, ab, field, value):
    base = _response([_scalability_row('case', 100)])
    row = _scalability_row('case', 50)
    row['performance'][field] = value
    result = ab.compare(_response([row]), base, channel=_channel(tmp_path))
    assert result['verdict']['decision'] == 'inconclusive'
    assert result['verdict']['exit'] == 2


@pytest.mark.parametrize('pin', tuple(key for key in MODEL if key.endswith('_sha256')))
def test_model_domain_changes_are_not_compared(tmp_path, ab, pin):
    base = _response([_scalability_row('case', 100)])
    after = _response([_scalability_row('case', 50, model={**MODEL, pin: 'f' * 64})])
    result = ab.compare(after, base, channel=_channel(tmp_path))
    assert result['verdict']['decision'] == 'inconclusive'


@pytest.mark.parametrize('base_pass,after_pass', [(False, False), (False, True), (True, False)])
def test_keep_requires_both_numerical_checks_to_pass(tmp_path, ab, base_pass, after_pass):
    base = _response([_scalability_row('case', 100, passed=base_pass)])
    after = _response([_scalability_row('case', 50, passed=after_pass)])
    assert ab.compare(after, base, channel=_channel(tmp_path))['verdict']['decision'] != 'keep'
    assert not ab.passed({'status': 'pass', 'pass': True, 'numerical_execution_passed': False})


def test_one_winner_cannot_hide_another_cases_unknown_timing(tmp_path, ab):
    base = _response([_scalability_row('win', 100), _scalability_row('pending', 100)])
    pending = _scalability_row('pending', 50)
    pending['performance']['status'] = 'UNKNOWN'
    after = _response([_scalability_row('win', 50), pending])
    result = ab.compare(after, base, channel=_channel(tmp_path))
    assert result['cases'][0]['status'] == 'compared'
    assert result['verdict']['decision'] == 'inconclusive'


def test_explicit_requested_missing_candidate_case_is_unranked_but_subsets_remain_valid(tmp_path, ab):
    base = _response([_scalability_row('a', 100), _scalability_row('b', 100)])
    after = _response([_scalability_row('a', 50)])
    channel = _channel(tmp_path)
    result = ab.compare(after, base, channel=channel, cases={'a', 'b'})
    assert result['cases'][1] == {
        'test_case': 'b', 'after_passed': None, 'status': 'no candidate row for requested case'}
    assert result['verdict']['decision'] == 'inconclusive'
    assert ab.compare(after, base, channel=channel, cases={'a'})['verdict']['decision'] == 'keep'
    assert ab.compare(after, base, channel=channel)['verdict']['decision'] == 'keep'


def test_missing_simulator_identity_is_not_matching_measured_evidence(tmp_path, ab):
    base = _response([_core_row('case', 100, identity={})])
    after = _response([_core_row('case', 50, identity={})])
    assert ab.compare(after, base, channel=_channel(tmp_path))['verdict']['decision'] == 'inconclusive'


def test_two_missing_model_identities_do_not_match(tmp_path, ab):
    base = _response([_scalability_row('case', 100, model={})])
    after = _response([_scalability_row('case', 50, model={})])
    assert ab.compare(after, base, channel=_channel(tmp_path))['verdict']['decision'] == 'inconclusive'


def test_scalability_rows_and_knob_copies_compare_by_the_timing_model(tmp_path, ab):
    base = _response([_scalability_row("big", 10_000)], key="cases")
    knob = _response([_scalability_row("big", 9_000)], key="cases", overrides={"overlap": "on"})
    result = ab.compare(knob, base, channel=_channel(tmp_path))
    assert result["after"] == 'knobs {"overlap": "on"}'
    assert result["cases"][0]["kind"] == "modeled" and result["cases"][0]["delta_pct"] == -10.0
    assert result["cases"][0]["evidence"] == "timing model (provisional)"
    assert result["verdict"]["decision"] == "keep"


def test_without_before_each_case_uses_the_newest_submission_rows(tmp_path, ab, capsys):
    channel = _channel(
        tmp_path,
        **{"3_1_old": _response([_core_row("a", 1200)], at=1, digest="1" * 64),
           "3_2_new": _response([_core_row("a", 1000), _core_row("b", 400)], at=2, digest="2" * 64),
           "3_3_model_only": _response([_scalability_row("a", 999)], key="cases", at=3, digest="3" * 64),
           "3_4_variant": _response([_core_row("a", 900), _core_row("b", 380)], tree="variants/v", at=4),
           "3_5_knob": _response([_core_row("a", 1)], overrides={"k": 1}, at=5)})
    after = ab.resolve("variants/v", channel)
    result = ab.compare(after, None, channel=channel)
    rows = {e["test_case"]: e for e in result["cases"]}
    assert rows["a"]["cycles_before"] == 1000 and rows["a"]["baseline_sha256"] == "2" * 64
    assert ab.resolve("submission", channel)["submission_sha256"] == "3" * 64  # knob copy is not submission/
    assert ab.resolve("3_2_new", channel)["submission_sha256"] == "2" * 64
    assert ab.main(["variants/v", "--channel", str(channel)]) == 0
    out = capsys.readouterr().out
    assert "verdict: KEEP" in out and "900" in out and "-10.0%" in out
    assert ab.main(["variants/v", "3_1_old", "--channel", str(channel), "--cases", "a", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cases"][0]["cycles_before"] == 1200


def test_errors_and_unknown_inputs_are_reported_not_compared(tmp_path, ab):
    channel = _channel(tmp_path, **{"3_9_err": {"all_pass": False, "error": "budget exhausted"}})
    with pytest.raises(SystemExit, match="error"):
        ab.resolve("3_9_err", channel)
    with pytest.raises(SystemExit, match="no completed response"):
        ab.resolve("variants/none", channel)
    with pytest.raises(SystemExit, match="not a file"):
        ab.resolve("nonsense", channel)
    lonely = _response([_core_row("z", 10)], tree="variants/v")
    result = ab.compare(lonely, None, channel=channel)
    assert result["cases"][0]["status"] == "no baseline row for this case"
    assert result["verdict"]["exit"] == 2


def test_mechanism_api_finds_the_symbol_and_its_library_callers(tmp_path, api, capsys):
    ws = tmp_path / "ws"
    library = ws / "submission" / "compiler_library" / "pkg" / "sub"
    library.mkdir(parents=True)
    (library.parent / "__init__.py").write_text("")
    (library / "__init__.py").write_text("")
    (library / "plan.py").write_text('def plan_tiles(shape, *, capacity: int) -> dict:\n    """Fit tiles."""\n')
    (library / "user.py").write_text("from .plan import plan_tiles\n")
    (library.parent / "other.py").write_text("import pkg.sub.plan\n")
    (ws / "OPTIMIZATION_CATALOG.json").write_text(json.dumps({"mechanisms": [
        {"id": "tiles", "facet": "storage", "mechanism": "fit tiles",
         "compiler": {"module": "pkg.sub.plan", "symbol": "plan_tiles", "path": "src/pkg/sub/plan.py"},
         "applicability": "static shapes", "remaining_adapter_or_feedback": "capacities from the target",
         "evidence_to_request": "chosen tile"}]}))
    (ws / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps({"all_mechanisms": [
        {"id": "tiles", "family": "storage", "api": "pkg.sub.plan", "symbol": "plan_tiles", "status": "supported"},
        {"id": "only_suggested", "family": "overlap", "api": "pkg.missing", "symbol": "f", "status": "unknown"}]}))
    assert api.main(["tiles", "--workspace", str(ws)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["signature"] == "def plan_tiles(shape, *, capacity: int) -> dict"
    assert shown["docstring"] == "Fit tiles." and shown["source"].endswith("plan.py:1")
    assert shown["library_callers"] == ["pkg/other.py:1", "pkg/sub/user.py:1"]
    assert shown["remaining_adapter_or_feedback"] == "capacities from the target"
    assert shown["status_for_this_target"] == "supported"
    assert api.main(["only_suggested", "--workspace", str(ws)]) == 0
    assert "not found" in json.loads(capsys.readouterr().out)["source"]
    assert api.main(["--list", "--facet", "overlap", "--workspace", str(ws)]) == 0
    assert capsys.readouterr().out.split()[0] == "only_suggested"
    assert api.main(["absent", "--workspace", str(ws)]) == 2
