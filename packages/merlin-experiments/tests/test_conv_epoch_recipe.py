"""The convolution compute-epoch recipe: every dropped drain is backed by an ordering fact.

A synthetic ``command_ordering`` record shaped like the RTL producer's output (three queues; a load
queue that releases at issue and may complete writes out of order) drives the recipe. With every
ordering fact present, compute epochs remove most drains, every route reproduces the source
convolution and the completion-order check finds no hazard. Removing one fact removes exactly the
routes whose law needs it, and the command adapter shows the checks catch the races they exist for.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from author_skill_fixtures import load_script

SKILL = "heavy-pass-recipes"
SELECT = 1 << 31
SPACES = ("scratchpad", "accumulator")


def _slot(operand, offset, width):
    return {"operand": operand, "offset": offset, "width": width}


def ordering_record():
    """Load (funct 2, writes), readout (3, reads), compute (4, reads), descriptor (6, reads + writes)."""
    accesses = {
        "MVIN.spad_addr": ("MVIN", "write", "load", 2, "rs2"),
        "MVOUT.acc_addr": ("MVOUT", "read", "store", 3, "rs2"),
        "COMPUTE_PRELOADED.a_spad": ("COMPUTE_PRELOADED", "read", "execute", 4, "rs1"),
        "PRELOAD.weight_spad": ("PRELOAD", "read", "execute", 6, "rs1"),
        "PRELOAD.c_addr": ("PRELOAD", "write", "execute", 6, "rs2"),
    }
    blocks = {"cols": _slot("rs2", 32, 7), "block": 16, "stride": {"constant": 16}}
    extents = {
        "MVIN.spad_addr": {"rows": _slot("rs2", 48, 5), "blocks": blocks},
        "MVOUT.acc_addr": {"rows": _slot("rs2", 48, 5), "blocks": blocks},
        "COMPUTE_PRELOADED.a_spad": {"rows": _slot("rs1", 48, 5), "scale": {"code": 0, **_slot("rs1", 16, 16)}},
        "PRELOAD.weight_spad": {"rows": _slot("rs1", 48, 5)},
        "PRELOAD.c_addr": {"rows": _slot("rs2", 48, 5), "scale": {"code": 0, **_slot("rs2", 48, 16)}},
    }
    cross = []
    for older, (_, d_old, q_old, _, _) in accesses.items():
        for younger, (_, d_young, q_young, _, _) in accesses.items():
            if q_old == q_young:
                continue
            ordered = "write" in (d_old, d_young)
            for space in SPACES:
                cross.append(
                    {
                        "older": older,
                        "younger": younger,
                        "space": space,
                        "overlap": "completion" if ordered else "issue",
                        "disjoint": "issue",
                        "tracked": ordered,
                    }
                )
    return {
        "status": "derived",
        "funct_queue": [
            {"funct": 0, "selector": _slot("rs1", 0, 2), "by_code": {"0": "execute", "1": "load", "2": "store"}},
            *({"funct": f, "queue": q} for f, q in ((2, "load"), (3, "store"), (4, "execute"), (6, "execute"))),
        ],
        "accesses": {
            key: {"class": c, "field": key.split(".")[1], **_slot(op, 0, 32), "direction": d, "queue": q, "funct": f,
                  "code": None}
            for key, (c, d, q, f, op) in accesses.items()
        },
        "extents": {key: {"status": "derived", "scale": None, "blocks": None, "conditions": [], **value}
                    for key, value in extents.items()},
        "same_queue": {"execute": {"release": "issue"}, "load": {"release": "issue"}, "store": {"release": "completion"}},
        "completion_order": {"execute": {"order": "in_order"}, "load": {"order": "out_of_order"},
                             "store": {"order": "in_order"}},
        "cross_queue": cross,
        "classes": {
            "by_funct": {"0": "CONFIG", "2": "MVIN", "3": "MVOUT", "4": "COMPUTE_PRELOADED", "6": "PRELOAD"},
            "config": {"funct": 0, "selector": _slot("rs1", 0, 2),
                       "by_code": {"0": "CONFIG_EX", "1": "CONFIG_LD", "2": "CONFIG_ST"}},
        },
        "local_address": {"accumulator_select": SELECT, "sentinel": (1 << 32) - 1},
        "unknown": [],
    }


def facts_file(tmp_path, edge=4, operand_rows=128, accumulator_rows=32):
    path = tmp_path / "facts.json"
    entries = {q: {"entries": {"status": "derived", "value": v}} for q, v in (("load", 8), ("execute", 16),
                                                                              ("store", 4))}
    path.write_text(
        json.dumps(
            {
                "arrays": [{"name": "array", "rows": edge, "cols": edge}],
                "memories": [
                    {"name": "operand_sram", "banks": 1, "depth": operand_rows, "bytes": operand_rows * edge},
                    {"name": "acc_sram", "banks": 1, "depth": accumulator_rows, "bytes": accumulator_rows * edge * 4},
                ],
                "datapaths": [{"name": "operand_sram", "dtype": "i8"}, {"name": "acc_sram", "dtype": "i32"}],
                "timing_structure": {"command_classes": {"classes": entries}},
            }
        )
    )
    return str(path)


def ordering_file(tmp_path, record, name="ordering.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"command_ordering": record}))
    return str(path)


@pytest.fixture(scope="module")
def epochs():
    return load_script(SKILL, "conv_epoch_recipe")


def _by_route(out):
    rows = {}
    for row in out["routes"]:
        rows.setdefault(row["route"], []).append(row)
    return rows


@pytest.mark.parametrize(
    "geometry",
    [
        None,  # derived: padded, ragged output width and output channels
        dict(shape=(1, 7, 7, 6, 9), kernel=(3, 3), stride=(2, 2), pad=(1, 1, 1, 1)),  # stride 2, channel tails
        dict(shape=(2, 5, 6, 5, 7), kernel=(3, 2), stride=(1, 1), pad=(0, 1, 2, 0)),  # asymmetric padding
    ],
)
def test_epochs_drop_drains_only_with_facts_and_every_route_verifies(epochs, tmp_path, geometry):
    out = epochs.recipe(facts_file(tmp_path), geometry, ordering=ordering_file(tmp_path, ordering_record()))
    assert out["issue_law"] == "admitted" and out["epoch_law"] == "admitted"
    for table in out["law_flags"].values():
        assert all(row["status"] in ("fact", "program") and row["basis"] for row in table.values())
    assert out["law_flags"]["epoch"]["accumulator_result_visibility"]["status"] == "fact"
    assert out["law_flags"]["epoch"]["complete_device_drain"]["status"] == "program"
    routes = _by_route(out)
    assert set(routes) == {"synchronous", "drain_per_compute", "epochs"}
    computes = {row["effects"]["compute"] for rows in routes.values() for row in rows}
    commands = {row["commands"] for rows in routes.values() for row in rows}
    assert len(computes) == 1 and len(commands) == 1  # same products and commands; only drains move
    base = routes["drain_per_compute"][0]
    assert base["drains"] >= base["effects"]["compute"] + base["effects"]["store"]
    assert routes["synchronous"][0]["drains"] > base["drains"]
    assert min(row["drains"] for row in routes["epochs"]) * 3 < base["drains"]
    for rows in routes.values():
        for row in rows:
            assert row["order_check"]["hazard_pairs"] == 0 and row["read_then_write"]["hazard"] == 0
            assert row["order_check"]["verdict"] == "clear", row["order_check"]["uncertain"]
    assert "every route reproduced the source convolution" in out["verified"]
    assert out["comparison"]["drain_per_compute"]["drains"] == base["drains"]


def test_a_missing_fact_keeps_the_drains_its_law_would_drop(epochs, tmp_path):
    facts = facts_file(tmp_path)
    record = ordering_record()
    record["cross_queue"] = [
        c for c in record["cross_queue"] if (c["older"], c["younger"]) != ("PRELOAD.c_addr", "MVOUT.acc_addr")
    ]
    out = epochs.recipe(facts, ordering=ordering_file(tmp_path, record, "no-readout-wait.json"))
    assert out["law_flags"]["epoch"]["accumulator_result_visibility"]["status"] == "UNKNOWN"
    assert out["issue_law"] == "admitted" and out["epoch_law"].startswith("not admitted")
    assert set(_by_route(out)) == {"synchronous", "drain_per_compute"}

    record = ordering_record()
    for c in record["cross_queue"]:  # a load no longer waits for the compute that read its rows
        if (c["older"], c["younger"]) == ("COMPUTE_PRELOADED.a_spad", "MVIN.spad_addr"):
            c.update(overlap="issue", tracked=False)
    out = epochs.recipe(facts, ordering=ordering_file(tmp_path, record, "no-war.json"))
    flags = out["law_flags"]
    assert flags["epoch"]["local_write_after_read"]["status"] == "UNKNOWN"
    assert flags["issue"]["local_range_dependencies"]["status"] == "UNKNOWN"
    assert set(_by_route(out)) == {"synchronous"}

    out = epochs.recipe(facts)  # no ordering facts at all
    assert out["law_flags"] is None and set(_by_route(out)) == {"synchronous"}


def _stream(epochs, record, effects, caps):
    masks = {"scratchpad": caps["operand"]["rows"] - 1, "accumulator": caps["accumulator"]["rows"] - 1}
    cmds = epochs.Commands(record, epochs.roles(record), masks)
    for kind, fields in effects:
        cmds(kind, fields) if kind != "drain" else cmds.drain()
    return cmds, masks


def test_the_checks_catch_zero_fill_then_overwrite_and_an_unordered_reload(epochs):
    from merlin.targetgen.completion_order import OrderingModel, check_commands

    caps = {"operand": {"rows": 128}, "accumulator": {"rows": 32}}
    record = ordering_record()
    load = dict(local_row=0, rows=4, columns=4, first_element=0, row_stride_elements=4)
    compute = dict(lhs_row=0, rhs_row=8, depth=4, rows=4, columns=4, accumulator_row=0)
    # A zero fill and the data rows into the same rows, nothing consuming them in between: a race.
    cmds, masks = _stream(epochs, record, [("load_input", dict(load, zero=True)), ("load_input", load)], caps)
    result = check_commands(cmds.items, OrderingModel.from_facts(record, row_masks=masks))
    assert result["verdict"] == "hazard" and result["by_kind"][0]["kind"] == "write-write"
    # The same pair with a drain between them is ordered.
    cmds, _ = _stream(epochs, record, [("load_input", dict(load, zero=True)), ("drain", None), ("load_input", load)],
                      caps)
    assert check_commands(cmds.items, OrderingModel.from_facts(record, row_masks=masks))["verdict"] == "clear"
    # A reload of rows a compute read: ordered by the load -> compute -> load waits the facts derive ...
    effects = [("load_input", load), ("compute", compute), ("load_input", load), ("compute", compute)]
    cmds, _ = _stream(epochs, record, effects, caps)
    assert check_commands(cmds.items, OrderingModel.from_facts(record, row_masks=masks))["verdict"] == "clear"
    assert cmds.war == {"ordered": 4, "hazard": 0, "unknown": 0, "same_queue": 0}
    # ... and a hazard in both checks when the load does not wait for the compute that read the rows.
    broken = copy.deepcopy(record)
    for c in broken["cross_queue"]:
        if c["older"] in ("COMPUTE_PRELOADED.a_spad", "PRELOAD.weight_spad") and c["younger"] == "MVIN.spad_addr":
            c.update(overlap="issue", tracked=False)
    cmds, _ = _stream(epochs, broken, effects, caps)
    assert cmds.war["hazard"] == 4
    assert check_commands(cmds.items, OrderingModel.from_facts(broken, row_masks=masks))["verdict"] == "hazard"


def test_command_roles_come_from_access_shapes_not_names(epochs):
    record = ordering_record()
    roles = epochs.roles(record)
    assert {k: v["funct"] for k, v in roles.items()} == {"transfer": 2, "readout": 3, "descriptor": 6, "compute": 4}
    del record["extents"]["PRELOAD.c_addr"]
    with pytest.raises(epochs.Unavailable, match="descriptor"):
        epochs.roles(record)


def test_large_layer_counts_only(epochs, tmp_path):
    geometry = dict(shape=(1, 12, 12, 16, 16), kernel=(3, 3), stride=(1, 1), pad=(1, 1, 1, 1))
    out = epochs.recipe(facts_file(tmp_path, edge=8, operand_rows=1024, accumulator_rows=256), geometry,
                        numerics=False, ordering=ordering_file(tmp_path, ordering_record()))
    routes = _by_route(out)
    assert out["verified"].startswith("numerics skipped")
    best = min(routes["epochs"], key=lambda row: row["drains"])
    assert best["drains_per_compute"] < 0.25 < routes["drain_per_compute"][0]["drains_per_compute"]


def test_readout_adapter_checks_actual_consumer_and_preserves_full_source(epochs, tmp_path):
    from dataclasses import replace

    from merlin.compile.scheduling.convolution_exploration import (
        ConvolutionRhsPhaseLaw,
        ConvolutionRhsPhases,
        ConvolutionScheduleChoice,
        emit_convolution_schedule,
        plan_convolution_rhs_phases,
        plan_convolution_schedule,
    )
    from merlin.compile.scheduling.window_transfer import ConvolutionReadoutEpochLaw

    S, W = epochs._load("recipe_support"), epochs._load("conv_window_recipe")
    facts = facts_file(tmp_path)
    caps = S.capacities(facts)
    source = W.plan(caps, dict(shape=(1, 4, 5, 3, 9), kernel=(1, 1),
        stride=(1, 1), pad=(0, 0, 0, 0)))
    record = ordering_record()
    flags = epochs.law_flags(record, epochs._structure(facts), epochs.roles(record))
    retained, rhs_law, issue, compute = epochs._laws(source, flags, "selected-source", (4,))
    schedule = plan_convolution_schedule(source, ConvolutionScheduleChoice(None, True, 4),
        retention_law=retained, issue_law=issue, epoch_law=compute)
    # Explicit synthetic target premises; not inferred from capacity or a name.
    law = ConvolutionReadoutEpochLaw(compute.hardware_identity, compute.numerical_policy_identity,
        "synthetic-store-snapshot-source", "conditional-test-binding", 3, 2,
        local_read_after_write=True, accumulator_result_visibility=True,
        ordered_configuration_capture=True, finite_queue_progress=True,
        complete_device_drain=True, exclusive_private_state=True)
    image = [n % 13 - 6 for n in range(60)]
    weights = [[(k+c) % 7 - 3 for c in range(9)] for k in range(3)]
    expected = W.reference(source, image, weights)
    machine = W.Machine(source, image, weights, True)
    effects, drains = [], []

    def lower(kind, fields):
        effects.append((kind, fields))
        machine(kind, fields)

    applied = epochs.emit_checked_readout_epoch(schedule, S.Integers(), lower,
        complete=lambda k, f: drains.append(k), readout_law=law)
    assert machine.out == expected and not machine.hazards
    assert applied["application"]["status"] == "consumer_receipt_matches"
    assert applied["application"]["numerical_qualification"] == "UNKNOWN"
    assert applied["application"]["cycles"] == "UNKNOWN"
    assert 0 < drains.count("store") < sum(k == "store" for k, _ in effects)

    default = emit_convolution_schedule(schedule, S.Integers(), lambda *_: None, complete=lambda *_: None)
    for bad in (default, {**applied["emission"], "source_sha256": "changed"},
                {**applied["emission"], "readout_epoch": None},
                {**applied["emission"], "completion": "no-op"}):
        with pytest.raises(ValueError, match="was not applied"):
            epochs.verify_readout_application(schedule, law, bad)
    with pytest.raises(ValueError, match="was not applied"):
        epochs.verify_readout_application(schedule, replace(law, pending_readouts=1), applied["emission"])
    for missing in (None, replace(law, ordered_configuration_capture=False)):
        seen = []
        with pytest.raises(ValueError):
            epochs.emit_checked_readout_epoch(schedule, S.Integers(), lambda *a: seen.append(a),
                complete=lambda *a: seen.append(a), readout_law=missing)
        assert not seen

    phase_law = ConvolutionRhsPhaseLaw(source.identity, "synthetic-independent-output-proof",
        independent_output_columns=True, immutable_inputs=True, no_intermediate_observers=True,
        full_completion_before_reprepare=True)
    rhs = plan_convolution_rhs_phases(schedule, operand_rows=12, rhs_base=6,
        law=rhs_law, phase_law=phase_law)
    assert type(rhs) is ConvolutionRhsPhases and len(rhs.phases) == 2
    machine = W.Machine(source, image, weights, True)
    phase_drains = []
    phased = epochs.emit_checked_readout_epoch(schedule, S.Integers(), machine,
        complete=lambda k, f: phase_drains.append(k), readout_law=law, rhs=rhs)
    assert machine.out == expected and not machine.hazards
    assert phased["application"]["complete_rhs_phases"] == 2
    assert phase_drains.count("rhs_phase_last_use") == 2
    saved = json.loads(json.dumps(phased["emission"]))
    epochs.verify_readout_application(schedule, law, saved, rhs=rhs)
    saved["phase_emissions"].pop()
    with pytest.raises(ValueError, match="phases were not applied completely"):
        epochs.verify_readout_application(schedule, law, saved, rhs=rhs)


@pytest.mark.parametrize('mode', ['spatial', 'wide', 'combined', 'phased-combined'])
def test_readout_adapter_forwards_and_verifies_optional_actual_consumers(epochs, tmp_path, mode):
    from dataclasses import replace

    from merlin.compile.scheduling.convolution_exploration import (
        ConvolutionRhsPhaseLaw,
        ConvolutionScheduleChoice,
        plan_convolution_rhs_phases,
        plan_convolution_rhs_residency,
        plan_convolution_schedule,
    )
    from merlin.compile.scheduling.convolution_output_groups import plan_rolling_convolution_output_group
    from merlin.compile.scheduling.dense_array_retention import DenseArrayRetentionAdmission
    from merlin.compile.scheduling.window_transfer import (
        ConvolutionReadoutEpochLaw,
        ConvolutionReadoutTransferLaw,
        ConvolutionResidentInputRows,
    )

    S, W = epochs._load('recipe_support'), epochs._load('conv_window_recipe')
    facts = facts_file(tmp_path, operand_rows=1024, accumulator_rows=128)
    source = W.plan(S.capacities(facts), dict(shape=(1, 3, 9, 4, 9), kernel=(1, 1),
        stride=(1, 1), pad=(0, 0, 0, 0)))
    record = ordering_record()
    flags = epochs.law_flags(record, epochs._structure(facts), epochs.roles(record))
    retained, rhs_law, issue, compute = epochs._laws(source, flags, 'TEST-ONLY-source', (128,))
    schedule = plan_convolution_schedule(source, ConvolutionScheduleChoice(None, True, 128),
        retention_law=replace(retained, independent_output_rows=True, no_intermediate_observers=True),
        issue_law=replace(issue, load_issue_budget=64), epoch_law=compute)
    rhs = plan_convolution_rhs_residency(schedule, operand_rows=1024, rhs_base=512, law=rhs_law)
    inputs = ConvolutionResidentInputRows(source.plan.window, source.identity, 4, 32, 12, 2, 32, 480, 4)
    readout = ConvolutionReadoutEpochLaw(compute.hardware_identity, compute.numerical_policy_identity,
        'TEST-ONLY-readout-source', 'synthetic-not-target-qualification', 32, 2,
        local_read_after_write=True, accumulator_result_visibility=True,
        ordered_configuration_capture=True, finite_queue_progress=True,
        complete_device_drain=True, exclusive_private_state=True)
    if mode == 'phased-combined':
        rhs = plan_convolution_rhs_phases(schedule, operand_rows=520, rhs_base=512, law=rhs_law,
            input_rows=inputs, phase_law=ConvolutionRhsPhaseLaw(source.identity, 'TEST-ONLY-column-effects',
                independent_output_columns=True, immutable_inputs=True, no_intermediate_observers=True,
                full_completion_before_reprepare=True))
    selected = {}
    if mode != 'wide':
        group = plan_rolling_convolution_output_group(schedule, rhs, inputs, max_output_rows=2)
        selected = dict(output_group=group, array_law=DenseArrayRetentionAdmission(
            compute.hardware_identity, compute.numerical_policy_identity, 'TEST-ONLY-fresh-stay'))
        field, changed = 'spatial_output_group', 'group_sha256'
    if mode != 'spatial':
        selected.update(readout_transfer_law=ConvolutionReadoutTransferLaw(
            compute.hardware_identity, compute.numerical_policy_identity, source.plan.identity,
            'TEST-ONLY-narrow-block-map', 4, 4, 12, ((0, 9, 'a'*64),),
            scaled_i8=True, exact_block_column_map=True, original_numerical_state=True))
        if mode == 'wide':
            field, changed = 'readout_transfers', 'law_sha256'
    if 'combined' in mode:
        selected['grouped_transport'] = True
    effects = []
    applied = epochs.emit_checked_readout_epoch(schedule, S.Integers(), lambda *a: effects.append(a),
        complete=lambda *a: effects.append(a), readout_law=readout, rhs=rhs, input_rows=inputs, **selected)
    assert effects and applied['application']['spatial_group_applied' if mode == 'spatial' else 'wide_readouts_applied']
    if 'combined' in mode:
        assert applied['application']['grouped_transport_applied']
    assert applied['application']['cycles'] == 'UNKNOWN'
    saved = json.loads(json.dumps(applied['emission']))
    epochs.verify_readout_application(schedule, readout, saved, rhs=rhs, input_rows=inputs, **selected)
    child = saved['phase_emissions'][0] if mode == 'phased-combined' else saved
    child[field][changed] = 'changed'
    with pytest.raises(ValueError, match='not applied'):
        epochs.verify_readout_application(schedule, readout, saved, rhs=rhs, input_rows=inputs, **selected)
    with pytest.raises(ValueError, match='needs its actual requested'):
        epochs.verify_readout_application(schedule, readout, applied['emission'], rhs=rhs, input_rows=inputs)

    if 'combined' in mode:
        saved = json.loads(json.dumps(applied['emission']))
        child = saved['phase_emissions'][0] if mode == 'phased-combined' else saved
        child['rolling_transport']['output_rows_per_group'] += 1
        with pytest.raises(ValueError, match='grouped transport was not applied'):
            epochs.verify_readout_application(schedule, readout, saved, rhs=rhs, input_rows=inputs, **selected)


def test_recipe_discovers_checked_public_ordering_without_operator_environment(epochs, tmp_path, monkeypatch):
    import hashlib

    (tmp_path / "rtl_facts").mkdir()
    (tmp_path / "submission").mkdir()
    facts = facts_file(tmp_path / "rtl_facts")
    record = ordering_record()
    staged = tmp_path / "COMMAND_ORDERING_SELECTION.json"
    staged.write_text(json.dumps({
        "schema": "merlin.rtl_command_ordering_selection.v1",
        "base_facts": {"sha256": hashlib.sha256(Path(facts).read_bytes()).hexdigest()},
        "ordering_sha256": epochs._digest(record), "ordering": record,
    }))
    monkeypatch.delenv(epochs.SELECTION_ENV, raising=False)
    monkeypatch.chdir(tmp_path / "submission")
    found, provenance = epochs.load_ordering(facts_path=facts)
    assert found == record and provenance["bound_to_facts"] is True
    assert provenance["discovery"] == "staged public command ordering selection"
    out = epochs.recipe(facts)
    assert out["issue_law"] == "admitted" and out["epoch_law"] == "admitted"
    assert "epochs" in {row["route"] for row in out["routes"]}
    # A nearby public record for another source cannot grant an epoch.
    doc = json.loads(staged.read_text())
    doc["base_facts"]["sha256"] = "0" * 64
    staged.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="different RTL facts"):
        epochs.load_ordering(facts_path=facts)


def test_generated_adapter_from_one_staged_sdk_and_full_compiler_bundle(tmp_path):
    """Actual generated-ISA/SCF execution; no imports from a companion checkout."""
    import shutil
    import subprocess
    import sys

    from author_skill_fixtures import skill_dir
    from merlin_experiments.phase1 import full_compiler_workspace as F

    from merlin.common.paths import repo_root
    from merlin.targetgen import tool_registry as TR
    from merlin.targetgen.tool_profile_shared import prepare_shared_source_release

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    scripts = bundle / 'scripts'
    scripts.mkdir()
    for name in ('conv_epoch_recipe', 'conv_adapter_example', 'conv_window_recipe', 'recipe_support'):
        shutil.copyfile(skill_dir(SKILL) / 'scripts' / (name + '.py'), scripts / (name + '.py'))
    sdk = bundle / 'sdk'
    prepare_shared_source_release(sdk, python_source_root=repo_root() / 'src', license_directory=repo_root(),
        tools=TR.ARM_TOOLS['merlin_rtlchecks'], version=23, source_binding=True)
    full = F.prepare(repo_root(), bundle / 'full-compiler', review_id='toy generated-effect consumer dependency test')
    code = '''
import importlib.util, json, pathlib, sys
library, scripts, output = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(library))
spec = importlib.util.spec_from_file_location('conv_epoch_recipe', scripts / 'conv_epoch_recipe.py')
recipe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recipe
spec.loader.exec_module(recipe)
contract = output.with_suffix('.contract.json')
assert recipe.main(['--write-toy-contract', str(contract), '--library', str(library)]) == 0
assert recipe.main(['--adapter-contract', str(contract), '--library', str(library), '--output', str(output)]) == 0
for name, module in tuple(sys.modules.items()):
 if name == 'merlin' or name.startswith('merlin.'):
  path = getattr(module, '__file__', None)
  if path: assert pathlib.Path(path).is_relative_to(library), (name, path)
'''
    for mode, library in (('sdk', sdk), ('full-compiler', full.python_root)):
        output = bundle / (mode + '-example')
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(library), str(scripts), str(output)],
            cwd=bundle, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr[-8000:]
        receipt = json.loads((output / 'receipt.json').read_text())
        rows = receipt['alternatives']
        assert len(rows) == 3 and all(row['status'] == 'applied' for row in rows)
        assert all(row['exact_source_check'] == dict(status='PASS', elements=75,
            scope='toy generated IR execution') for row in rows)
        assert len({row['artifact_sha256'] for row in rows}) == 3
        assert rows[-1]['application']['grouped_transport_applied']
        assert rows[-1]['application']['wide_readouts_applied']
        assert rows[-1]['consumer']['spatial_output_group']['layout']['output_rows'] == 2
        assert rows[-1]['consumer']['rolling_transport']['output_rows_per_group'] == 2
        for row in rows:
            text = (output / row['artifact']).read_text()
            assert 'scf.for' in text and 'toy_effects.' in text
            assert row['cycles'] == 'UNKNOWN'
        assert receipt['selected_alternative'] is None and not receipt['target_qualification']
        assert not receipt['performance_claim']
