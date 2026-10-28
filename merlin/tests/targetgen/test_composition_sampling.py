import itertools
import json

import pytest

from merlin.targetgen.corpus_synth import SynthesisError
from merlin_experiments.phase0 import composition_generation as CG

CHAINS = ("c0", "c1", "c2", "c3", "c4")
SCHEMES = ("s0", "s1", "s2", "s3")
SHAPES = ("aligned", "tail", "other")


def _entry(chain, scheme, shape, producer="matmul", runtime=False):
    instance = dict(producer=producer)
    if runtime:
        instance["runtime_scale_input"] = "S"
    return dict(
        name=f"SY_composition_{chain}_{scheme}_{shape}",
        cat="layers",
        op="composition_program",
        generalization=dict(
            composition_chain_id=chain,
            quantization_scheme_id=scheme,
            shape_class=shape,
            composition_instance=instance,
        ),
    )


def _declared():
    return [
        _entry(chain, scheme, shape, producer="operand_sum" if chain == "c4" else "matmul", runtime=scheme == "s3")
        for chain, scheme, shape in itertools.product(CHAINS, SCHEMES, SHAPES)
    ]


def test_every_declared_composition_is_kept_when_it_fits():
    chosen, record = CG.sample_declared(_declared(), budget=200, workload_count=109)
    assert len(chosen) == 60
    assert record["policy"] == "all_declared_fit" and record["unselected"] == []
    assert record["selected_plus_workload"] == 169 and record["obligations_uncovered"] == []


def test_a_shortfall_is_a_recorded_sample_that_spreads_across_chains():
    declared = _declared()
    chosen, record = CG.sample_declared(declared, budget=160, workload_count=109)
    assert len(chosen) == 51 and record["selected_plus_workload"] == 160
    assert record["policy"] == "sample_to_budget" and record["declared_count"] == 60
    assert len(record["unselected"]) == 9 and not set(record["unselected"]) & {e["name"] for e in chosen}
    assert {row["chain"] for row in record["selected"]} == set(CHAINS)
    assert all(row["selected"] >= 9 for row in record["chains"].values())
    assert record["obligations_uncovered"] == []


def test_the_sample_is_independent_of_input_order_and_repeatable():
    declared = _declared()
    first = CG.sample_declared(declared, budget=160, workload_count=109)
    second = CG.sample_declared(list(reversed(declared)), budget=160, workload_count=109)
    assert [e["name"] for e in first[0]] == [e["name"] for e in second[0]]
    assert first[1] == second[1]


def test_a_tight_budget_covers_each_obligation_before_repeating_one():
    chosen, record = CG.sample_declared(_declared(), budget=120, workload_count=109)
    assert len(chosen) == 11
    covered = {key for row in record["selected"] for key in row["new_obligations"]}
    assert {f"chain:{chain}" for chain in CHAINS} <= covered
    assert {f"scheme:{scheme}" for scheme in SCHEMES} <= covered
    assert {f"shape_class:{shape}" for shape in SHAPES} <= covered
    assert "producer:operand_sum" in covered and "scale_source:runtime_input" in covered


def test_workloads_that_fill_the_budget_leave_no_room_and_refuse():
    with pytest.raises(SynthesisError, match="leave no room"):
        CG.sample_declared(_declared(), budget=160, workload_count=160)


def test_append_declared_counts_workloads_not_performance_companions(tmp_path, monkeypatch):
    declared = _declared()
    monkeypatch.setattr("merlin.targetgen.composition_corpus.entries", lambda spec, facts: declared)
    monkeypatch.setattr(CG, "prepare_entries", lambda entries: entries)
    workloads = [dict(name=f"w{i}", cat="layers", op="matmul") for i in range(109)]
    perf = [dict(name=f"w{i}_perf", cat="_perf", op="matmul") for i in range(159)]
    path = tmp_path / "composition-selection.json"
    sink = {}
    result = CG.append_declared(
        workloads + perf, dict(compositions={"target_count": 60}), {}, budget=160, selection_path=path,
        selection_sink=sink,
    )
    compositions = [e for e in result if e.get("op") == "composition_program"]
    assert len(compositions) == 51 and len(result) == 109 + 159 + 51
    assert sum(1 for e in result if e["cat"] != "_perf") == 160
    record = json.loads(path.read_text())
    assert record["workload_count"] == 109 and record["selected_count"] == 51
    assert record["schema"] == "merlin.composition_selection.v1"
    assert [row["name"] for row in record["selected"]] == sorted(e["name"] for e in compositions)
    assert sink == record


def test_append_declared_without_declared_compositions_changes_nothing(tmp_path):
    entries = [dict(name="w", cat="layers", op="matmul")]
    assert CG.append_declared(entries, dict(), {}, budget=160, selection_path=tmp_path / "x.json") is entries
    assert not (tmp_path / "x.json").exists()
