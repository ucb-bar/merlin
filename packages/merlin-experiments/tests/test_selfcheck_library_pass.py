"""library_pass: the library pass to try next, inside the per-case ``optimization_next`` the author reads.

The fixture holds rows recorded by one fresh-compiler run (a Spike scalability response of five model
GEMMs and a core gSIM response of one small case), trimmed to the fields the performance brief reads,
and that run's staged ``OPTIMIZATION_SUGGESTIONS.json`` mechanisms.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import library_pass as LP
from merlin_experiments.phase1.feedback import performance_brief as PB

FIXTURE = json.loads(Path(__file__).with_name("library_pass_fresh4.json").read_text(encoding="utf-8"))


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(FIXTURE["suggestions"]))
    submission = tmp_path / "submission"
    (submission / "compiler_library" / "merlin").mkdir(parents=True)
    (submission / "dense_lowering.py").write_text(
        "from merlin.compile.scheduling.dense_rhs_residency import plan_dense_rhs_residency\n"
    )
    # The library copy is never read as the author's own imports.
    (submission / "compiler_library" / "merlin" / "x.py").write_text(
        "from merlin.compile.scheduling.dense_segmented_pipeline import schedule_dense_segmented_pipeline\n"
    )
    (tmp_path / ".agents" / "skills" / "ab-library-pass").mkdir(parents=True)
    names = ['case'] + [r['capsule'] for part, key in [('scalability', 'cases'), ('core_gsim', 'per_capsule')]
                        for r in FIXTURE[part][key]]
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {name: {'workload_role': 'large_kernel'} for name in names}}))
    return tmp_path


def _passes(document, key):
    return {row["capsule"]: (row.get("optimization_next") or {}).get(LP.KEY) for row in document[key]}


def test_recorded_scalability_rows_carry_the_pass_in_their_optimization_next(workspace):
    document = copy.deepcopy(FIXTURE["scalability"])
    PB.attach(document, document["cases"], workspace=workspace)
    written = {name: found for name, found in _passes(document, "cases").items() if found}
    assert written
    for name, found in written.items():
        line = found["line"]
        assert "\n" not in line
        assert line.startswith("provisional cycle-model estimate (not measured hardware cycles): ")
        assert "% compute-peak attainment" in line and "utilization" not in line
        assert f"--capsules {name} --suite scalability --wait-budget 5" in line and "--sim gsim" not in line
        assert "| bottleneck: " in line and "| library: " in line and "apply_library_pass.py" in line
        assert "cp -r" not in line and "--adapter tools/library_pass_adapter.py" in line
        assert found["ab"]["application_status"] == "not_run"
        assert found["ab"]["compare"].startswith("python .agents/skills/ab-library-pass/scripts/ab_compare.py ")
        assert found["skill"] == "ab-library-pass" and found["reference"].startswith("pass-library reference/")
        assert found["evidence_kind"] == "provisional_model"
    found = written["PD_resnet50_model_m196k2304n256_raw"]
    assert found["bottleneck"].startswith("local storage (busiest modeled resource accelerator.storage.scratchpad.0")
    first = found["mechanisms"][0]
    assert first["opportunity_status"] == "missing_adapter"  # confirmed workload before unchecked split-K
    assert first["imported_by_submission"] is False
    assert first["recipe"] == "ab-library-pass/reference/adapter.md"
    assert first["application_status"].startswith("unknown")
    # the lean reply keeps the advice on the case row only, not a second copy in the brief
    assert "per_case_advice" not in document["performance_brief"]


def test_recorded_gsim_row_is_checked_on_gsim_with_a_gsim_domain_request_bound(workspace):
    document = copy.deepcopy(FIXTURE["core_gsim"])
    PB.attach(document, document["per_capsule"], workspace=workspace)
    (found,) = [f for f in _passes(document, "per_capsule").values() if f]
    assert found["line"].startswith(
        "gSIM-measured: 799 cycles, x4.1 over the read requests bound (gSIM-domain memory path, 16 B/cycle)"
    )
    assert "--capsules G_matmul_m1k48n64_raw --sim spike --wait-budget 5" in found["ab"]["check"]
    # A diagnostic control gets no per-case advice from the brief, so no pass either.
    (workspace / "PUBLIC_CAPABILITIES.json").write_text(
        json.dumps(
            {
                "schema": "merlin.public_capability_inventory.v1",
                "members": {"G_matmul_m1k48n64_raw": {"workload_role": "diagnostic_control"}},
            }
        )
    )
    document = copy.deepcopy(FIXTURE["core_gsim"])
    PB.attach(document, document["per_capsule"], workspace=workspace)
    assert not any(_passes(document, "per_capsule").values())


def _entry(**extra):
    base = {
        "test_case": "case",
        "cycles": 1000,
        "evidence": "measured: gSIM cycles of this request",
        "evidence_kind": "gsim_measured",
        "roofline_bound_cycles": 250,
        "bound_by": "read requests",
        "gap_ratio": 4.0,
        "headroom_cycles": 750,
    }
    return {**base, **extra}


def test_bottleneck_classes_follow_fence_restreaming_resource_then_bound():
    selection = FIXTURE["suggestions"]
    fence = LP.bottleneck(
        _entry(fence_overlap={"fence_per_tile": True, "fences": 9, "tiles_per_fence": 1.0}), selection
    )
    assert fence[0].startswith("fence-bound (9 fences") and fence[1]
    restream = LP.bottleneck(
        _entry(capacity_use={"operand_reads": {"ratio": 4.46, "capacity_bound_ratio": 1.5}}), selection
    )
    assert restream[0].startswith("operand re-streaming (reads 4.46x")
    assert restream[1][0]["opportunity_status"] == "candidate"  # unchecked resident prerequisites do not lead
    close = LP.bottleneck(
        _entry(capacity_use={"operand_reads": {"ratio": 1.6, "capacity_bound_ratio": 1.5}}), selection
    )
    assert close[0].startswith("read requests roofline bound")  # re-reads within reach of the capacity bound
    busy = LP.bottleneck(_entry(dominant_modeled_resource={"name": "cpu.issue", "modeled_activity": 0.4}), selection)
    assert busy[0].startswith("instruction issue (busiest modeled resource cpu.issue, 40% modeled activity")
    assert LP.bottleneck(_entry(bound_by=None), selection) is None


def test_nothing_without_headroom_bottleneck_mechanism_or_a_graded_submission(workspace):
    selection = FIXTURE["suggestions"]
    imports = LP.inventory_script().imported_modules(workspace / "submission")

    def found(entry, suite="core"):
        return LP.block(entry, selection=selection, suite=suite, imports=imports, workspace=workspace)

    restream = found(_entry(capacity_use={"operand_reads": {"ratio": 4.46, "capacity_bound_ratio": 1.5}}))
    assert restream["line"].startswith("gSIM-measured: 1000 cycles, x4.0 over the read requests bound (gSIM-domain")
    assert all(m["id"] != "resident_rhs" for m in restream["mechanisms"])  # offer new applicable routes first
    resident_only = {"all_mechanisms": [r for r in selection["all_mechanisms"] if r["id"] == "resident_rhs"]}
    again = LP.block(_entry(), selection=resident_only, suite="core", imports=imports, workspace=workspace)
    resident = again["mechanisms"][0]
    assert resident["imported_by_submission"] is True
    assert resident["application_status"].startswith("unknown")  # imports are not execution
    assert found(_entry(headroom_cycles=150)) is None  # within 20% of the bound
    assert found(_entry(headroom_cycles=300, model_confidence={"spread": 0.4})) is None  # within the model spread
    assert found(_entry(bound_by=None)) is None  # unknown bottleneck
    assert found(_entry(), suite="unknown") is None
    assert LP.block(_entry(), selection={"all_mechanisms": []}, suite="core", imports={}, workspace=workspace) is None
    assert LP.suite_of({"variant": {"name": "v"}}) is None and LP.suite_of({}) == "core"
    assert LP.suite_of({"schema": LP.SCALABILITY_SCHEMA}) == "scalability"
    assert LP.bound_text("compute", None) == "the compute bound"


def test_annotate_never_fails_and_needs_the_staged_selection(tmp_path):
    brief = {"largest_headroom": [_entry()], "per_case_advice": {"case": {}}}
    assert LP.annotate([], brief, workspace=tmp_path, suite="core") == 0
    assert brief["per_case_advice"] == {"case": {}}
    assert LP.annotate([], "not a brief", workspace=tmp_path, suite="core") == 0
    assert LP.annotate([], brief, workspace=tmp_path, suite=None) == 0


def test_actual_catalog_fence_actions_and_per_case_workload_filter(monkeypatch, tmp_path):
    from merlin_experiments.phase1 import optimization_suggestions as suggestions
    from merlin_experiments.phase1.feedback import case_declarations
    from merlin_experiments.phase1.feedback import mechanism_hints as hints

    catalog_path = Path(__file__).resolve().parents[3] / 'docs/design/joint_optimization_catalog.json'
    catalog = json.loads(catalog_path.read_text())
    # All catalogue declarations explicitly supported here; production retains real unknowns.
    capabilities = {suggestions.ALIASES[t]: 'supported'
                    for entry in catalog['mechanisms'] for t in entry['required_traits']
                    if not suggestions.ALIASES[t].startswith('workload.')}
    modules = tuple(entry['compiler']['module'] for entry in catalog['mechanisms'])
    selection = suggestions.select(catalog, capabilities=capabilities, public_modules=modules)
    available, _ = hints.fence_hints(selection, why='fence density')
    assert [r['id'] for r in available[:2]] == ['barrier_cleanup', 'copy_pipeline']
    blocked = copy.deepcopy(selection)
    barrier = next(r for r in blocked['all_mechanisms'] if r['id'] == 'barrier_cleanup')
    barrier['requirements']['scheduling.dependency_tokens'] = 'unknown'
    barrier['status'] = 'unknown'
    available, unknown = hints.fence_hints(blocked, why='fence density')
    assert 'barrier_cleanup' not in {r['id'] for r in available}
    assert unknown[0]['id'] == 'barrier_cleanup'
    assert 'scheduling.dependency_tokens' in unknown[0]['capability_unconfirmed']

    dense = {'semantic': {'semantic_family': 'contraction'},
             'operation': {'op': 'matmul', 'attributes': {'lhs': 'a', 'weight': 'b'}},
             'inputs': [{'name': 'a', 'role': 'input', 'shape': [32, 64]},
                        {'name': 'b', 'role': 'weight', 'shape': [64, 128]}]}
    monkeypatch.setattr(case_declarations, 'load', lambda name, workspace: dense)
    scoped = hints.for_case(selection, 'not_a_model_or_shape_name', tmp_path)
    spatial = next(r for r in scoped['all_mechanisms'] if r['id'] == 'quantized_spatial_readout_cache')
    assert spatial['requirements']['workload.convolution'] == 'unsupported'
    assert spatial['status'] == 'unsupported'
    offered = hints.hints(scoped, dominant={'name': 'accelerator.storage.accumulator.all'}, limit=64)
    assert 'quantized_spatial_readout_cache' not in {r['id'] for r in offered}
    assert next(r for r in selection['all_mechanisms'] if r['id'] == spatial['id'])['status'] != 'unsupported'
    monkeypatch.setattr(case_declarations, 'load', lambda name, workspace: None)
    assert hints.for_case(selection, 'unclassified', tmp_path) is selection


def test_fence_bound_cases_name_their_own_epoch_mechanism_first(monkeypatch, workspace):
    from merlin_experiments.phase1 import optimization_suggestions as suggestions
    from merlin_experiments.phase1.feedback import case_declarations

    catalog_path = Path(__file__).resolve().parents[3] / 'docs/design/joint_optimization_catalog.json'
    catalog = json.loads(catalog_path.read_text())
    capabilities = {suggestions.ALIASES[t]: 'supported'
                    for entry in catalog['mechanisms'] for t in entry['required_traits']
                    if not suggestions.ALIASES[t].startswith('workload.')}
    modules = tuple(entry['compiler']['module'] for entry in catalog['mechanisms'])
    selection = suggestions.select(catalog, capabilities=capabilities, public_modules=modules)
    conv = {'semantic': {'semantic_family': 'contraction'},
            'operation': {'op': 'conv2d', 'attributes': {'ifm': 'x', 'weight': 'w', 'layout': 'NHWC',
                                                         'kernel': [3, 3, 64, 64], 'stride': [1, 1],
                                                         'padding': [1, 1, 1, 1]}},
            'inputs': [{'name': 'x', 'role': 'input', 'shape': [4, 14, 14, 64]},
                       {'name': 'w', 'role': 'weight', 'shape': [3, 3, 64, 64]}]}
    dense = {'semantic': {'semantic_family': 'contraction'},
             'operation': {'op': 'matmul', 'attributes': {'lhs': 'a', 'weight': 'b'}},
             'inputs': [{'name': 'a', 'role': 'input', 'shape': [32, 64]},
                        {'name': 'b', 'role': 'weight', 'shape': [64, 128]}]}
    fenced = _entry(fence_overlap={'fence_per_tile': True, 'fences': 9, 'tiles_per_fence': 1.0})

    monkeypatch.setattr(case_declarations, 'load', lambda name, ws: conv)
    found = LP.block(fenced, selection=selection, suite='core', imports={}, workspace=workspace)
    assert found['bottleneck'].startswith('fence-bound')
    assert found['mechanisms'][0]['id'] == 'convolution_compute_epochs' and not found['mechanisms'][0]['applies_if']
    assert found['ab']['mechanism'] == 'convolution_compute_epochs'
    assert '| library: convolution_compute_epochs (' in found['line']

    monkeypatch.setattr(case_declarations, 'load', lambda name, ws: dense)
    found = LP.block(fenced, selection=selection, suite='core', imports={}, workspace=workspace)
    named = {row['id'] for row in found['mechanisms']} | {row['id'] for row in found['related_capability_unconfirmed']}
    assert 'convolution_compute_epochs' not in named
    assert found['mechanisms'][0]['id'] == 'dense_panel_epochs' and not found['mechanisms'][0]['applies_if']
    assert found['mechanisms'][1]['id'] == 'barrier_cleanup'
    assert found['ab']['mechanism'] == 'dense_panel_epochs'

    monkeypatch.setattr(case_declarations, 'load', lambda name, ws: conv)
    found = LP.block(fenced, selection=selection, suite='core', imports={}, workspace=workspace)
    named = {row['id'] for row in found['mechanisms']} | {row['id'] for row in found['related_capability_unconfirmed']}
    assert 'dense_panel_epochs' not in named


def test_connected_program_ab_command_keeps_the_programs_suite(workspace, monkeypatch):
    import argparse
    import shlex

    from merlin_experiments.phase1.tools import selfcheck

    command = LP.ab_command('case', 'lib-case', 'programs', mechanism='barrier_cleanup', workspace=workspace)
    arguments = shlex.split(command.split(' && ', 1)[1])[2:]
    parse = argparse.ArgumentParser.parse_args
    observed = []
    class Parsed(Exception):
        pass
    def capture(parser, argv=None, namespace=None):
        observed.append(parse(parser, argv, namespace))
        raise Parsed
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', capture)
    with pytest.raises(Parsed):
        selfcheck.main(arguments)
    assert observed[0].suite == 'programs' and observed[0].sim == 'spike'
    assert observed[0].capsules == 'case'
    found = LP.block(_entry(), selection=FIXTURE['suggestions'], suite='programs', imports={}, workspace=workspace)
    assert found['ab']['check'].endswith('--capsules case --suite programs --sim spike --wait-budget 5')
