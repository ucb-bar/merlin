"""Self-check responses carry a bounded performance brief for their own passing test cases."""

from __future__ import annotations

import json

from merlin_experiments.phase1.feedback import performance_brief as PB
from merlin_experiments.phase1.feedback import selfcheck_notes


def _roles(workspace, *names):
    (workspace / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {name: {'workload_role': 'large_kernel'} for name in names}}))


def _suggestions(workspace):
    rows = [
        {"id": "tiles", "family": "storage", "api": "lib.capacity", "symbol": "fit", "status": "supported",
         "requirements": {"memory.scratchpad": "supported"}, "evidence": "tile and bytes"},
        {"id": "coalesce", "family": "movement", "api": "lib.copies", "symbol": "merge", "status": "supported",
         "requirements": {"memory.explicit_dma": "supported"}, "evidence": "transfer count"},
        {"id": "loops", "family": "issue", "api": "lib.loops", "symbol": "compact", "status": "supported",
         "requirements": {"submission.cpu_issued": "supported"}, "evidence": "command count"},
        {"id": "partials", "family": "numeric_formats", "api": "lib.partials", "symbol": "seed",
         "status": "supported", "requirements": {"numeric.int32_accumulation": "supported"},
         "mechanism": "Initialize partial sums once", "evidence": "accumulation epochs"},
        {"id": "async", "family": "overlap", "api": "lib.async", "symbol": "go", "status": "unknown",
         "requirements": {"scheduling.async_issue": "unknown"}, "evidence": "queue"},
    ]
    (workspace / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps({"all_mechanisms": rows}))


def _performance(cycles, resource):
    return {
        "status": "PROVISIONAL", "complete_event_accounting": True, "estimated_whole_soc_cycles": cycles,
        "structural_roofline": {"useful_macs": 4096, "structural_peak_macs_per_cycle": 256,
                                "structural_compute_floor_cycles": 16},
        "scoped_resource_feedback": {"resources": [{"name": resource, "capacity_normalized_activity": 0.8},
                                                   {"name": "cpu.issue", "capacity_normalized_activity": 0.1}]},
    }


def test_core_brief_ranks_passing_cases_and_names_matching_mechanisms(tmp_path):
    _suggestions(tmp_path)
    _roles(tmp_path, 'small', 'big', 'untimed', 'known', 'unknown')
    (tmp_path / "selfcheck_out" / "big").mkdir(parents=True)
    (tmp_path / "selfcheck_out" / "big" / "command_buffer.json").write_text("{}")
    rows = [
        {"capsule": "small", "pass": True, "performance": _performance(64, "cpu.issue")},
        {"capsule": "big", "pass": True, "performance": _performance(1600, "accelerator.storage.accumulator.all")},
        {"capsule": "broken", "pass": False, "performance": _performance(9999, "cpu.issue")},
        {"capsule": "untimed", "pass": True},
    ]
    brief = PB.build(rows, workspace=tmp_path, artifact_root="selfcheck_out")
    assert brief["counts"] == {"passing": 3, "with_evidence": 2, "without_evidence": 1}
    top, second = brief["largest_headroom"]
    assert (top["test_case"], second["test_case"]) == ("big", "small")
    assert top["gap_ratio"] == 100.0 and top["array_utilization_pct"] == 1.0
    assert [m["id"] for m in top["suggested_mechanisms"]] == ["partials", "tiles", "coalesce"]
    assert "accumulator" in top["mechanism_basis"] and "useful compute-ceiling attainment is 1.0%" in top["reading"]
    assert [e["test_case"] for e in brief["lowest_utilization"]] == ["big", "small"]
    assert [m["id"] for m in second["suggested_mechanisms"]] == ["loops"]
    assert top["inspect"] == "python isa_tools.py debug selfcheck_out/big/command_buffer.json --capsule big"
    assert "inspect" not in second  # no copied command buffer, no command
    assert brief["without_evidence"] == ["untimed"] and "latest_completed_grade.json" in brief["note"]


def test_measured_gsim_cycles_are_labelled_and_preferred(tmp_path):
    _roles(tmp_path, 'm')
    row = {"capsule": "m", "pass": True, "barrier_tier": "L3", "barrier_cycles": 320,
           "performance": _performance(999, "cpu.issue")}
    (entry,) = PB.build([row], workspace=tmp_path)["largest_headroom"]
    assert entry["cycles"] == 320 and entry["evidence"].startswith("measured")


def _source_activity(performance):
    pins = dict(model_sha256='a'*64, elf_sha256='b'*64, scope_sha256='c'*64)
    performance.update(pins)
    performance['scoped_resource_feedback'].update(pins, source_actor_activity={
        **pins, 'schema': 'merlin.scoped_source_actor_feedback.v1', 'status': 'modeled',
        'estimated_cycles_same_window': performance['estimated_whole_soc_cycles'],
        'actors': [{'name': name, 'availability': 'modeled', 'active_fraction': value}
                   for name, value in [('execute.feed', .2), ('reader.slot_residency', .3)]],
        'overlaps': [{'left': 'reader.slot_residency', 'right': 'execute.feed',
                      'availability': 'modeled', 'fraction_of_window': 0.0}]})
    return performance


def test_source_activity_reaches_case_advice_without_static_overlap_or_instruction_ranking(tmp_path):
    _roles(tmp_path, 'region')
    row = {'capsule': 'region', 'pass': True,
           'performance': _source_activity(_performance(1600, 'accelerator.controller.execute'))}
    brief = PB.build([row], workspace=tmp_path)
    signals = brief['per_case_advice']['region']['signals']['schedule_activity']
    assert signals['active_fraction']['execute.feed'] == .2
    assert signals['overlap_fraction'][0]['fraction_of_window'] == 0.0
    assert signals['additive'] is False and 'not physical' in signals['basis']
    assert 'legality nor profitability' in signals['next_action']
    assert brief['largest_headroom'][0]['cycles'] == 1600


def test_source_activity_rejects_other_identity_scope_or_measured_denominator(tmp_path):
    _roles(tmp_path, 'region')
    for field, value in [('scope_sha256', 'd'*64), ('model_sha256', 'd'*64),
                         ('elf_sha256', 'd'*64), ('estimated_cycles_same_window', 999)]:
        performance = _source_activity(_performance(1600, 'accelerator.controller.execute'))
        performance['scoped_resource_feedback']['source_actor_activity'][field] = value
        entry, = PB.build([{'capsule': 'region', 'pass': True, 'performance': performance}],
                          workspace=tmp_path)['largest_headroom']
        assert 'schedule_activity' not in entry
    performance = _source_activity(_performance(1600, 'accelerator.controller.execute'))
    entry, = PB.build([{'capsule': 'region', 'pass': True, 'barrier_tier': 'L3',
                       'barrier_cycles': 320, 'performance': performance}],
                     workspace=tmp_path)['largest_headroom']
    assert 'schedule_activity' not in entry


def test_scalability_rows_use_l2_tier_and_carry_no_debugger_command(tmp_path):
    _roles(tmp_path, 's')
    (tmp_path / "selfcheck_out" / "s").mkdir(parents=True)
    (tmp_path / "selfcheck_out" / "s" / "command_buffer.json").write_text("{}")
    cases = [{"capsule": "s", "tiers": {"L0": "pass", "L1": "pass", "L2": "pass"},
              "performance": _performance(500, "cpu.issue")},
             {"capsule": "f", "tiers": {"L2": "fail"}, "performance": _performance(500, "cpu.issue")}]
    brief = PB.build(cases, workspace=tmp_path, debuggable=False)
    assert brief["counts"]["passing"] == 1 and "inspect" not in brief["largest_headroom"][0]
    assert "public roster" in brief["inspect_note"]


def test_attach_is_advisory_and_never_fails(tmp_path):
    document = {"all_pass": False}
    assert PB.attach(document, [{"capsule": "x", "pass": False}], workspace=tmp_path) == {"all_pass": False}
    assert PB.attach(document, "not rows", workspace=tmp_path) == {"all_pass": False}
    PB.attach(document, [{"capsule": "y", "pass": True, "performance": _performance(64, "cpu.issue")}],
              workspace=tmp_path)
    assert document["performance_brief"]["advisory_only"] is True


def test_package_failure_note_follows_the_failure_plane():
    build = selfcheck_notes.package_failure_note({"plane": "build", "detail": "cmake error"})
    assert "cmake" in build
    refused = selfcheck_notes.package_failure_note(
        {"plane": "integrity", "detail": "editable compiler source outside public namespace: merlin/frontend/x.py"})
    assert "cmake" not in refused and "merlin/frontend/x.py" in refused and "namespace" in refused
    other = selfcheck_notes.package_failure_note({"plane": "integrity", "detail": "forbidden pattern"})
    assert "cmake" not in other and "integrity scan" in other


def test_unknown_timing_keeps_primary_structural_work_visible_without_ranking(tmp_path):
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps(dict(
        schema='merlin.public_capability_inventory.v1', members={
            'region': dict(workload_role='connected'),
            'control': dict(workload_role='diagnostic_control'),
            'broken': dict(workload_role='large_kernel')})))
    rows = [dict(capsule=name, tiers={'L2': status}, performance=dict(
        status='UNKNOWN', estimated_whole_soc_cycles=1,
        complete_event_accounting=False, reason='unmodeled memory interaction'))
        for name, status in [('region', 'pass'), ('control', 'pass'), ('broken', 'fail')]]
    brief = PB.build(rows, workspace=tmp_path)
    assert brief['largest_headroom'] == []
    assert brief.get('lowest_utilization', []) == []
    assert brief['counts']['with_evidence'] == 0
    structural = brief['structural_exploration']
    assert structural['cases'] == ['region']
    assert structural['performance_qualification'] is False
    assert 'Spike numerics' in structural['action'] and 'fast_estimate' in structural['action']


def test_missing_timing_on_controls_does_not_request_performance_exploration(tmp_path):
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps(dict(
        schema='merlin.public_capability_inventory.v1', members={
            'control': dict(workload_role='correctness')})))
    brief = PB.build([dict(capsule='control', tiers={'L2': 'pass'})], workspace=tmp_path)
    assert 'structural_exploration' not in brief


def test_advice_survives_filtering_to_case_rows_without_promoting_unknown(tmp_path):
    _suggestions(tmp_path)
    _roles(tmp_path, 'small', 'big', 'untimed', 'known', 'unknown')
    rows = [dict(capsule='known', pass_=True), dict(capsule='unknown', pass_=True)]
    for row in rows:
        row['pass'] = row.pop('pass_')
    rows[0]['performance'] = _performance(1600, 'accelerator.storage.accumulator.all')
    rows[1]['performance'] = {'status': 'UNKNOWN', 'estimated_whole_soc_cycles': None}
    document = {'per_capsule': rows, 'all_pass': True}
    PB.attach(document, rows, workspace=tmp_path)
    filtered = {row['capsule']: row for row in document['per_capsule']}
    assert filtered['known']['optimization_next']['mechanisms'][0]['id'] == 'partials'
    assert 'incumbent' in filtered['unknown']['optimization_next']['action']
    assert filtered['unknown']['performance']['estimated_whole_soc_cycles'] is None
    assert document['all_pass'] is True
    assert all(entry['test_case'] != 'unknown' for entry in document['performance_brief']['largest_headroom'])


def test_reply_is_lean_and_keeps_advice_on_its_case_rows(tmp_path):
    _suggestions(tmp_path)
    _roles(tmp_path, 'known')
    rows = [{'capsule': 'known', 'pass': True, 'performance': _performance(1600, 'cpu.issue')}]
    document = {'per_capsule': rows, 'next_steps': {'lines': ['a', 'b'], 'steps': [{'choice': 'x'}]}}
    PB.attach(document, rows, workspace=tmp_path)
    brief = document['performance_brief']
    assert 'per_case_advice' not in brief and 'family_convergence' not in brief
    assert brief['largest_headroom'][0]['test_case'] == 'known'
    assert 'optimization_next' in rows[0] and brief['note'] == PB.LEAN_NOTE and len(brief['note']) < 400
    assert 'steps' not in document['next_steps'] and document['next_steps']['lines']


def test_structural_capacity_reaches_unknown_case_without_cycle_utilization(tmp_path, monkeypatch):
    _suggestions(tmp_path)
    _roles(tmp_path, 'small', 'big', 'untimed', 'known', 'unknown')
    use = {"scratchpad": {"pct": 12.5, "lower_bound": True},
           "operand_reads": {"capacity_bound_ratio": 1.5},
           "basis": "static decoded trace"}
    monkeypatch.setattr(PB.capacity_use, "for_case", lambda *a, **kw: use)
    rows = [{"capsule": "unknown", "pass": True, "performance": {"status": "UNKNOWN"}},
            {"capsule": "known", "pass": True, "performance": _performance(1600, "cpu.issue")}]
    document = {"per_capsule": rows}
    PB.attach(document, rows, workspace=tmp_path)
    unknown = rows[0]["optimization_next"]["signals"]
    known = rows[1]["optimization_next"]["signals"]
    assert unknown["capacity_use"] == use
    assert unknown["array_utilization_pct"] is None
    assert unknown["timing_status"] == "unavailable"
    assert known["array_utilization_pct"] == 1.0
    assert known["evidence_kind"] == "provisional_model"
    assert known["gap_ratio"] == 100.0
    assert known["dominant_modeled_resource"]["name"] == "cpu.issue"
    assert len(document["performance_brief"]["largest_headroom"]) == 1


def test_role_summary_separates_controls_models_and_measurements_before_truncation(tmp_path):
    roles = {f'control{i}': {'workload_role': 'correctness'} for i in range(20)}
    roles.update({name: {'workload_role': 'large_kernel'} for name in ('modeled', 'measured', 'unknown', 'failed')})
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1', 'members': roles}))
    rows = [{'capsule': f'control{i}', 'pass': True, 'performance': _performance(1600, 'cpu.issue')}
            for i in range(20)]
    rows += [
        {'capsule': 'modeled', 'pass': True, 'performance': _performance(32, 'cpu.issue')},
        {'capsule': 'measured', 'pass': True, 'barrier_tier': 'L3', 'barrier_cycles': 64,
         'performance': _performance(32, 'cpu.issue')},
        {'capsule': 'unknown', 'pass': True, 'performance': {'status': 'UNKNOWN'}},
        {'capsule': 'failed', 'pass': False, 'performance': _performance(16, 'cpu.issue')},
    ]
    brief = PB.build(rows, workspace=tmp_path)
    groups = {(g['workload_role'], g['evidence_kind']): g
              for g in brief['workload_role_summary']['groups']}
    assert groups['correctness', 'provisional_model']['passing_cases'] == 20
    assert groups['correctness', 'provisional_model']['median_array_utilization_pct'] == 1
    # A response mixing gSIM and model rows ranks every row with a model estimate on model cycles.
    assert groups['large_kernel', 'provisional_model']['median_array_utilization_pct'] == 50
    assert groups['large_kernel', 'provisional_model']['passing_cases'] == 2
    assert 'gsim_measured' not in {kind for _, kind in groups}
    by_case = {e['test_case']: e for e in brief['largest_headroom']}
    assert by_case['measured']['cycles'] == 32 and by_case['measured']['gsim_cycles'] == 64
    assert 'gsim_cycles' not in by_case['modeled']
    assert groups['large_kernel', 'unavailable'] == {
        'workload_role': 'large_kernel', 'evidence_kind': 'unavailable', 'passing_cases': 1}
    assert sum(g['passing_cases'] for g in groups.values()) == 23
    assert brief['counts'] == {'passing': 23, 'with_evidence': 22, 'without_evidence': 1}
    assert {e['test_case'] for e in brief['largest_headroom']} == {'modeled', 'measured'}


def test_a_response_of_gsim_rows_only_keeps_ranking_on_gsim_cycles(tmp_path):
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {name: {'workload_role': 'large_kernel'} for name in ('a', 'b')}}))
    rows = [{'capsule': name, 'pass': True, 'barrier_tier': 'L3', 'barrier_cycles': 64,
             'performance': _performance(32, 'cpu.issue')} for name in ('a', 'b')]
    entries = PB.build(rows, workspace=tmp_path)['largest_headroom']
    assert {e['cycles'] for e in entries} == {64}
    assert all(e['evidence'].startswith('measured: gSIM') and 'gsim_cycles' not in e for e in entries)
