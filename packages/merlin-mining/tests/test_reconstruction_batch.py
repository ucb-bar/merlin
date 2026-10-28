"""All regions survive batching; source hypotheses cannot qualify themselves."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.kernels.reference_inventory import SCHEMA, digest
from merlin.mining.reconstruction_batch import execute_batch, experiment_packets
from merlin.mining.reconstruction_cli import _execute_job, main
from merlin.mining.reconstruction_experiments import build_experiment_packet
from merlin.mining.reconstruction_reference import dense_source_region


def base_result():
    return dict(schema="merlin.kernel_reconstruction_run.v1", mode="reference_guided", command="reconstruct",
                inventory=dict(manifest_sha256="1" * 64, references=[]), gaps=[], performance_qualification=False)


def run_batch(tmp_path, jobs, command="reconstruct"):
    return execute_batch(command, dict(regions=jobs), base_result(), pin=dict(sha256="2" * 64),
                         runner=_execute_job, root=tmp_path, workspace=tmp_path / "work",
                         max_candidates=1, max_search_steps=256, proposal_attempts=1)


def test_batch_uses_real_library_and_consumed_schedule_on_multiple_geometries(tmp_path):
    pytest.importorskip("z3")
    path = repo_root() / "merlin/tests/kernels/test_reconstruction.py"
    spec = importlib.util.spec_from_file_location("batch_checked_library_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    jobs, executed = [], []
    for ordinal, shape in enumerate(((8, 6, 8), (5, 7, 9))):
        selected = fixture.adapter(shape)
        original = selected.observe

        def observe(candidate, *, shape=shape, original=original):
            checked = candidate.evaluation.result.artifact
            checked.revalidate()
            fixture.execute(checked.kernel, shape)
            executed.append(shape)
            return original(candidate)

        region = dense_source_region(selected.source)
        jobs.append(dict(region_id=f"source-{ordinal}", reference_ids=[],
                         source_region=region, reference_region=region,
                         adapter=replace(selected, observe=observe), goals=["schedule.consumed"]))
    jobs.insert(1, dict(region_id="unsupported", reference_ids=[], gaps=[dict(
        kind="representation_gap", goals=["source.complete_fused_region"],
        reason="complete fused output has no selected representation adapter",
        next_action="bind_complete_fused_source")]))
    result = run_batch(tmp_path, jobs)
    assert result["batch"]["complete_selection"]
    assert [row["status"] for row in result["regions"]] == ["reconstructed", "unresolved", "reconstructed"]
    assert executed == [(8, 6, 8), (5, 7, 9)]
    for row in (result["regions"][0], result["regions"][2]):
        candidate = row["reconstruction"]["candidates"][0]
        assert candidate["satisfied"]
        assert "schedule.consumed" in candidate["observation"]["established"]
        assert any("schedule.plan" in intermediate["text"] for intermediate in candidate["intermediates"])
    assert not result["performance_qualification"]


def source_job(tmp_path, *, analysis=True, mutate=False):
    path = tmp_path / "source.mlir"
    path.write_text("module {}\n")

    def provider(*, source, workspace):
        program = workspace / "program.bin"
        program.write_bytes(b"explicit synthetic protocol fixture, no machine execution")
        pin = dict(path=str(program), sha256=digest(program.read_bytes()), kind="synthetic_fixture")
        if mutate:
            path.write_text("module { /* changed */ }\n")
        return dict(program=pin, artifacts=[pin], status="reconstructed", performance_qualification=True,
                    established_mechanisms=["reuse.lhs"], reanalysis=(dict(
                        program_sha256=pin["sha256"], observations=dict(static_bytes=program.stat().st_size))
                        if analysis else None))

    return dict(region_id="hypothesis", reference_ids=["listing-leaf"],
                source=dict(path=path.name, sha256=digest(path.read_bytes())), hypothesis=provider,
                reference_bindings=dict(status="unknown"), goals=["reuse.lhs"])


def test_source_only_receipt_cannot_close_semantics_numerics_or_mechanisms(tmp_path):
    result = run_batch(tmp_path, [source_job(tmp_path)])
    row = result["regions"][0]
    assert row["status"] == "compiled_hypothesis"
    assert row["reference_bindings"]["status"] == "unknown"
    assert row["established_mechanisms"] == []
    assert not row["performance_qualification"] and not row["reference_equivalence"]
    assert {gap["goals"][0] for gap in row["gaps"]} == {
        "source.reference_region_equivalence", "numeric.reference_boundary_equivalence", "performance.complete_region"}
    assert row["reanalysis"]["program_sha256"] == row["program"]["sha256"]
    assert all(task["context"]["region_id"] == "hypothesis" for task in row["learning_tasks"])


@pytest.mark.parametrize("provider_claim", [None, True])
def test_source_only_schedule_goal_requires_independent_executable_binding(tmp_path, provider_claim):
    job = source_job(tmp_path)
    compile_source = job["hypothesis"]

    def provider(**kwargs):
        receipt = compile_source(**kwargs)
        if provider_claim is not None:
            receipt["reanalysis"]["observations"]["scheduling_dialect_consumption"] = provider_claim
            receipt["established_mechanisms"] = ["schedule.consumed"]
        return receipt

    job.update(hypothesis=provider, goals=["schedule.consumed"])
    row = run_batch(tmp_path, [job])["regions"][0]
    gap = next(gap for gap in row["gaps"] if gap["goals"] == ["schedule.consumed"])
    assert gap["kind"] == "adapter_gap"
    assert "schedule.plan" in gap["reason"] and "actual emitted-program binding" in gap["reason"]
    assert gap["next_action"] == "bind_checked_schedule_plan_consumer_and_actual_program_with_independent_owner"
    task = next(task for task in row["learning_tasks"] if task["goals"] == ("schedule.consumed",))
    assert task["gap_kind"] == "adapter_gap" and task["route"] == "adapter_or_preservation"
    assert row["status"] == "compiled_hypothesis" and row["established_mechanisms"] == []
    assert not row["reference_equivalence"] and not row["performance_qualification"]


def test_program_requires_actual_hash_bound_reanalysis(tmp_path):
    result = run_batch(tmp_path, [source_job(tmp_path, analysis=False)])
    row = result["regions"][0]
    assert row["status"] == "unresolved"
    assert any(gap["goals"] == ["emission.actual_program_reanalysis"] for gap in row["gaps"])


def test_analysis_retains_qualification_gaps_without_running_source_compiler(tmp_path):
    job = source_job(tmp_path)
    job["hypothesis"] = lambda **kwargs: pytest.fail("analysis reached source compiler")
    row = run_batch(tmp_path, [job], command="analyze")["regions"][0]
    assert len(row["learning_tasks"]) == 3
    assert not row["reference_equivalence"]


def test_whole_graph_module_cannot_be_reported_as_compiled_individual_region(tmp_path):
    job = source_job(tmp_path)
    job["source_binding"] = dict(standalone=False)
    job["hypothesis"] = lambda **kwargs: pytest.fail("whole graph reached per-region compiler")
    result = run_batch(tmp_path, [job])
    assert result["regions"][0]["status"] == "unresolved"
    assert result["regions"][0]["gaps"][0]["goals"] == ["source.standalone_region"]


@pytest.mark.parametrize("when", ["before", "after"])
def test_source_pin_is_checked_before_and_after_compilation(tmp_path, when):
    job = source_job(tmp_path, mutate=when == "after")
    if when == "before":
        job["source"]["sha256"] = "0" * 64
        job["hypothesis"] = lambda **kwargs: pytest.fail("changed source reached compiler")
    with pytest.raises(ValueError, match="SHA256"):
        run_batch(tmp_path, [job])


def test_intake_gap_kinds_keep_region_identity_and_action(tmp_path):
    jobs = [dict(region_id=f"leaf-{i}", gaps=[dict(kind="reference_alignment_gap",
            reason="operand values are unavailable", next_action="recover_runtime_operand_values")]) for i in range(2)]
    result = run_batch(tmp_path, jobs)
    for i, row in enumerate(result["regions"]):
        assert row["gaps"][0]["original_kind"] == "reference_alignment_gap"
        assert row["learning_tasks"][0]["gap_kind"] == "missing_evidence"
        assert row["learning_tasks"][0]["context"]["region_id"] == f"leaf-{i}"
    assert result["regions"][0]["learning_tasks"][0]["task_id"] != result["regions"][1]["learning_tasks"][0]["task_id"]


def test_all_sixty_regions_are_diagnosed_and_sharded_without_drops():
    index = dict(regions=[dict(region_id=f"leaf-{i:02}", gaps=[dict(kind="missing_evidence",
        goals=["source.reference_equivalence"], reason="selected operand values unavailable",
        next_action="recover_selected_operands")]) for i in range(60)])
    packets = experiment_packets(index, context=dict(visibility="reference_guided"))
    assert len(packets) == 4
    assert [len(packet["tasks"]) for packet in packets] == [16, 16, 16, 12]
    assert {row["region_id"] for packet in packets for row in packet["tasks"]} == {
        row["region_id"] for row in index["regions"]}
    assert len({row["task_id"] for packet in packets for row in packet["tasks"]}) == 60
    assert packets[-1]["coverage"]["next_offset"] is None
    assert packets == experiment_packets(index, context=dict(visibility="reference_guided"))
    with pytest.raises(ValueError, match="offset"):
        build_experiment_packet(index, context={}, diagnosis_offset=-1)


def test_cli_publishes_all_shards_from_one_selected_batch_factory(tmp_path, monkeypatch, capsys):
    source = tmp_path / "reference.py"
    source.write_text("def selected(a): return a\n")
    factory = tmp_path / "selected_region_batch.py"
    factory.write_text("""def factory(document,index,*,root,workspace):
    return {'regions':[{'region_id':f'leaf-{i}', 'gaps':[{'kind':'missing_evidence',
      'goals':['source.reference_equivalence'],'reason':'selected operands unavailable',
      'next_action':'recover_selected_operands'}]} for i in range(60)]}
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    context = tmp_path / "context.json"
    context.write_text(json.dumps(dict(visibility="reference_guided")))
    document = dict(schema=SCHEMA, mode="reference_guided", references=[dict(id="listing",
        family="selected", visibility="operator_only", format="python_schedule",
        artifact=dict(path=source.name, sha256=digest(source.read_bytes())))],
        region_binding=dict(factory="selected_region_batch:factory", sha256=digest(factory.read_bytes())),
        experiment_context=dict(path=context.name, sha256=digest(context.read_bytes())))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document))
    output = tmp_path / "batch"
    assert main(["reconstruct", "--manifest", str(manifest), "--out", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    shards = json.loads((output / "packet-shards.json").read_text())
    result = json.loads((output / "index.json").read_text())
    assert len(result["regions"]) == 60 and shards["complete"]
    assert printed["packet_shards"] == str(output / "packet-shards.json")
    seen = set()
    for shard in shards["shards"]:
        selection = json.loads(Path(shard["selection"]).read_text())
        packet = json.loads(Path(selection["packet"]["path"]).read_text())
        seen.update(row["region_id"] for row in packet["tasks"] if "region_id" in row)
    assert seen == {f"leaf-{i}" for i in range(60)}


def test_duplicate_region_identity_refuses_instead_of_overwriting(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        run_batch(tmp_path, [dict(region_id="same"), dict(region_id="same")])
