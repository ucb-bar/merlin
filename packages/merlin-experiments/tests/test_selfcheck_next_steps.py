"""next_steps: one ranked, workload-scoped choice and its apply command where the author actually reads.

The self-check client prints the block after the reply (on stderr, so a script parsing the JSON on
stdout keeps working); the grade summary's and round brief's "Largest headroom" lines show the same
choice instead of a bare "try X, Y". A fence-bound product is never pointed at a convolution readout
cache, imports alone do not prove an applied mechanism, and a catalogue
mechanism described for fence-bound convolutions ranks first for one once it exists.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import composition_fixtures as F
import pytest
from merlin_experiments.phase1.feedback import next_steps as N
from merlin_experiments.phase1.feedback import performance_brief as PB
from merlin_experiments.phase1.feedback import performance_headline as H

FIXTURE = json.loads(Path(__file__).with_name("library_pass_fresh4.json").read_text(encoding="utf-8"))
SPATIAL = "quantized_spatial_readout_cache"
#: A catalogue row of the shape a fence-bound convolution mechanism takes (synchronization facet,
#: names drains and epochs, requires a convolution workload).
EPOCHS = {
    "id": "convolution_epoch_like",
    "status": "unknown",
    "family": "synchronization",
    "api": "merlin.compile.scheduling.convolution_exploration",
    "symbol": "convolution_pass_space",
    "mechanism": "Group convolution computes into epochs that drain the device only at epoch boundaries",
    "legality": "ordering facts per flag; UNKNOWN keeps the drain",
    "evidence": "Fences per compute",
    "requirements": {
        "compiler.api:merlin.compile.scheduling.convolution_exploration": "supported",
        "memory.scratchpad": "supported",
        "workload.convolution": "unknown",
    },
}


def conv(name="CV_case"):
    return {
        "name": name,
        "kind": "layer",
        "label": "public",
        "semantic": {"semantic_family": "contraction"},
        "inputs": [
            {"name": "IFM", "role": "input", "shape": [1, 8, 8, 4], "dtype": "i8"},
            {"name": "W", "role": "weight", "shape": [36, 16], "dtype": "i8"},
        ],
        "operation": {
            "op": "conv2d",
            "attributes": {"ifm": "IFM", "weight": "W", "kh": 3, "kw": 3, "stride": [1, 1], "padding": [1, 1, 1, 1]},
        },
    }


def entry(name, *, fence=True, headroom=9000):
    out = {
        "test_case": name,
        "cycles": 10000,
        "roofline_bound_cycles": 1000,
        "headroom_cycles": headroom,
        "gap_ratio": 10.0,
        "bound_by": "compute",
        "evidence": "modeled: Spike timing model (provisional)",
        "evidence_kind": "provisional_model",
        "dominant_modeled_resource": {"name": "accelerator.storage.scratchpad.0", "modeled_activity": 0.6},
    }
    if fence:
        out["fence_overlap"] = {
            "fence_per_tile": True,
            "fences": 4096,
            "output_tiles": 64,
            "tiles_per_fence": 0.02,
            "reading": "a full-device fence after every output tile",
            "dynamic": {"fence_wait_share_of_cycles": 0.8},
            "suggested_mechanisms": [{"id": SPATIAL}, {"id": "final_k_accumulator_readout_frontier"}],
        }
    return out


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(FIXTURE["suggestions"]))
    F.declare(tmp_path, "layers", F.matmul("MM_case"))
    F.declare(tmp_path, "layers", conv())
    F.declare(tmp_path, "layers", F.conv_pool("CHAIN_case"))
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "backend.py").write_text("import json\n")
    (tmp_path / ".agents" / "skills" / "ab-library-pass").mkdir(parents=True)
    (tmp_path / 'PUBLIC_CAPABILITIES.json').write_text(json.dumps({
        'schema': 'merlin.public_capability_inventory.v1',
        'members': {name: {'workload_role': 'connected'} for name in ('MM_case', 'CV_case', 'CHAIN_case', 'MD_case')}}))
    return tmp_path


def _choose(workspace, item, **kw):
    return N.choose(item, N.Context(workspace), **kw)


def test_a_fence_bound_product_is_not_pointed_at_a_convolution_readout_cache(workspace):
    step = _choose(workspace, entry("MM_case"))
    assert step["kind"] == "mechanism" and step["choice"] not in (SPATIAL, "final_k_accumulator_readout_frontier")
    assert step["choice"] == "barrier_cleanup" and "capability unconfirmed" in step["choice_text"]
    assert step["bottleneck"].startswith("the device drains after nearly every output tile (4096 fences")
    assert "apply_library_pass.py" in step["apply"] and "--mechanism barrier_cleanup --case MM_case" in step["apply"]
    assert step["apply"].endswith("--capsules MM_case --sim spike --wait-budget 5")
    assert step["read"] == ".agents/skills/pass-library/reference/synchronization.md"
    # The same holds for an undeclared case whose public inventory row shows two-dimensional operands only.
    (workspace / "PUBLIC_CAPABILITIES.json").write_text(
        json.dumps(
            {
                "schema": "merlin.public_capability_inventory.v1",
                "members": {
                    "JT_case": {
                        "workload_role": "large_kernel",
                        "selfcheck_suite": "scalability",
                        "input_tensors": [
                            {"role": "weight", "shape": [2048, 32]},
                            {"role": "input", "shape": [1, 2048]},
                        ],
                    }
                }
            }
        )
    )
    step = _choose(workspace, entry("JT_case"))
    assert step["choice"] != SPATIAL and "--suite scalability" in step["apply"]
    selection = FIXTURE["suggestions"]
    names = [c["row"]["id"] for c in N.mechanism_candidates(selection, N.FENCE_FACETS, N.FENCE_TERMS, "fence", {})]
    assert SPATIAL not in names and names[-1] == "command_loops"  # stream compaction keeps every fence


def test_a_catalogue_mechanism_for_fence_bound_convolutions_ranks_first_only_for_convolutions(workspace):
    suggestions = copy.deepcopy(FIXTURE["suggestions"])
    suggestions["all_mechanisms"].append(EPOCHS)
    (workspace / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(suggestions))
    assert _choose(workspace, entry("CV_case"))["choice"] == EPOCHS["id"]
    assert _choose(workspace, entry("MM_case"))["choice"] != EPOCHS["id"]


def test_import_is_discovery_only_and_declared_knob_tuning_remains_available(workspace):
    submission = workspace / "submission"
    (submission / "fences.py").write_text("from merlin.compile.scheduling.effect_cleanup import cleanup_indices\n")
    step = _choose(workspace, entry("MM_case"))
    assert step["choice"] == "barrier_cleanup"
    assert "runtime use unconfirmed" in step["choice_text"]
    (submission / "TUNING_KNOBS.json").write_text(
        json.dumps(
            {
                "schema": "merlin.tuning_knobs.v1",
                "settings_file": "tuning_settings.json",
                "knobs": {"fence_cleanup": {"values": [False, True], "mechanism": "barrier_cleanup"}},
            }
        )
    )
    (submission / "tuning_settings.json").write_text(json.dumps({"fence_cleanup": False}))
    step = _choose(workspace, entry("MM_case"))
    assert step["choice"] == "barrier_cleanup"
    assert "imported, but knob fence_cleanup=false keeps it at its baseline" in step["choice_text"]
    assert step["apply"].startswith("set fence_cleanup to true in variants/")
    (submission / "tuning_settings.json").write_text(json.dumps({"fence_cleanup": True}))
    assert _choose(workspace, entry("MM_case"))["choice"] == "barrier_cleanup"


def test_a_host_flagged_chain_case_leads_with_its_composition_route(workspace):
    rows = [{"capsule": "CHAIN_case", "pass": True, "failure": {"category": "host_compute_in_accelerator_group"}}]
    step = N.choose(entry("CHAIN_case", fence=False), N.Context(workspace, {}, rows))
    assert step["kind"] == "composition" and step["choice"] == "producer_pool"
    assert "--mechanism producer_pool" in step["apply"] and step["read"].endswith("spatial_chain_recipe.py")


def test_a_model_case_leads_with_its_source_composition_request(workspace):
    F.declare(workspace, "model", F.model("MD_case"), ir=F.PROGRAM_IR)
    step = _choose(workspace, entry("MD_case", fence=False))
    assert step["kind"] == "source_composition" and step["choice"].startswith("prepack_weights")
    assert '"schema":"merlin.source-composition-preferences.v1"' in step["apply"]
    assert "source_composition.json in a copy of submission/, then python agent_selfcheck.py" in step["apply"]


def _response(workspace):
    rows = [{"capsule": name, "pass": True, "barrier_tier": "L2"} for name in ("MM_case", "CV_case")]
    brief = {
        "largest_headroom": [
            entry("MM_case", headroom=9000),
            entry("CV_case", headroom=8000),
            entry("CHAIN_case", headroom=7000),
            entry("unknown_case", headroom=6000),
        ]
    }
    return {"per_capsule": rows}, rows, brief


def test_the_block_holds_the_top_cases_within_its_budget(workspace):
    document, rows, brief = _response(workspace)
    found = N.block(document, rows, brief, workspace=workspace)
    lines = found["lines"]
    assert lines[0] == N.HEADER and 2 <= len(lines) <= N.LIMIT + 1
    assert lines[1].startswith("1. MM_case: the device drains") and "| apply: " in lines[1]
    assert all(len(text.encode()) <= N.LINE_BYTES for text in lines)
    assert sum(len(text.encode()) + 1 for text in lines) <= N.BLOCK_BYTES
    # attach() puts it on the response; a brief with no headroom puts nothing.
    N.attach(document, rows, brief, workspace=workspace)
    assert document[N.KEY]["lines"] == lines
    assert N.block({}, rows, {"largest_headroom": []}, workspace=workspace) is None
    N.attach(document, rows, None, workspace=workspace)  # never raises


def test_the_client_prints_the_block_after_the_reply_and_keeps_stdout_json(workspace, tmp_path_factory):
    from merlin_experiments.phase1.tools import selfcheck as client

    document, rows, brief = _response(workspace)
    N.attach(document, rows, brief, workspace=workspace)
    ws = tmp_path_factory.mktemp("client")
    shutil.copy(client.__file__, ws / "agent_selfcheck.py")
    (ws / "submission").mkdir()
    (ws / "submission" / "x.py").write_text("x = 1\n")
    rid = "1_2_abc"
    sha = client._submission_digest(ws / "submission")
    channel = ws / ".qa_channel"
    channel.mkdir()
    (channel / f"req_{rid}.json").write_text(json.dumps({"submission": "submission", "submission_sha256": sha}))
    reply = {
        **document,
        "all_pass": True,
        "selfcheck_protocol": client.PROTOCOL_VERSION,
        "selfcheck_request_id": rid,
        "submission_sha256": sha,
    }
    (channel / f"resp_{rid}.json").write_text(json.dumps(reply))
    (channel / f"done_{rid}").write_text("")
    proc = subprocess.run(
        [sys.executable, "-B", "-I", str(ws / "agent_selfcheck.py"), "--attach", rid],
        cwd=ws,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["all_pass"] is True  # stdout stays one JSON document
    block = proc.stderr.split("[self-check] ", 1)[1].strip()
    assert block.splitlines()[0] == N.HEADER and "1. MM_case: " in block
    assert len(block.encode()) <= client._NEXT_STEPS_BYTES
    # A reply without next steps prints nothing extra; oversized input is cut to the client's budget.
    assert client._next_steps_text({}) == "" and client._next_steps_text({"next_steps": {"lines": [N.HEADER]}}) == ""
    huge = {"next_steps": {"lines": [N.HEADER] + ["x" * 2000] * 5}}
    assert len(client._next_steps_text(huge).encode()) <= client._NEXT_STEPS_BYTES


def test_brief_attach_adds_next_steps_to_a_self_check_response(workspace):
    document = copy.deepcopy(FIXTURE["core_gsim"])
    PB.attach(document, document["per_capsule"], workspace=workspace)
    found = document.get(N.KEY)
    assert found is None or found["lines"][0] == N.HEADER


def test_headroom_lines_show_one_choice_with_its_apply_command(workspace):
    verdict = {"per_capsule": [{"capsule": "MM_case", "pass": True}]}
    block = {"largest_headroom": [{**entry("MM_case"), "suggested_mechanisms": [{"id": "dma_coalescing"}]}]}
    top = H._top(block, H._next_steps(workspace, verdict))
    assert top[0]["mechanisms"] == [SPATIAL, "final_k_accumulator_readout_frontier"]  # JSON field unchanged
    lines = H.render({"schema": H.SCHEMA, "grade": {}, "correctness": "ok", "top_headroom": top})
    (line,) = [text for text in lines if text.startswith("1. `MM_case`")]
    assert "; next: barrier_cleanup (" in line and "python .agents/skills/ab-library-pass/scripts/" in line
    assert "try " not in line and SPATIAL not in line
    # Without a workspace (or a ranked step) the line keeps the catalogue ids it always had.
    lines = H.render({"schema": H.SCHEMA, "grade": {}, "correctness": "ok", "top_headroom": H._top(block)})
    assert any("; try quantized_spatial_readout_cache" in text for text in lines)


#: The dense counterpart: drains once per output block where ordering facts allow; dense workloads only.
DENSE_EPOCHS = {
    "id": "dense_panel_epochs",
    "status": "unknown",
    "family": "synchronization",
    "api": "merlin.compile.scheduling.dense_panel_epochs",
    "symbol": "DensePanelEpochLaw",
    "mechanism": "Merge the K steps of each output block into one epoch, so the device drains once per output "
    "block instead of a full fence after every K step",
    "legality": "ordering facts per flag; UNKNOWN keeps the drain",
    "evidence": "Fences per output tile",
    "requirements": {
        "compiler.api:merlin.compile.scheduling.dense_panel_epochs": "supported",
        "memory.scratchpad": "supported",
        "memory.accumulator": "supported",
        "workload.dense_contraction": "unknown",
    },
}


def test_fence_bound_products_get_dense_epochs_and_convolutions_keep_their_own(workspace):
    suggestions = copy.deepcopy(FIXTURE["suggestions"])
    suggestions["all_mechanisms"] += [EPOCHS, DENSE_EPOCHS]
    (workspace / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(suggestions))
    F.declare(
        workspace,
        "layers",
        {
            **F.matmul("DECODE_case"),
            "inputs": [{"name": "A0", "role": "input", "shape": [1, 2048], "dtype": "i8"}, *F.weights((2048, 64))],
        },
    )
    inventory_path = workspace / "PUBLIC_CAPABILITIES.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["members"]["DECODE_case"] = {"workload_role": "connected"}
    inventory_path.write_text(json.dumps(inventory))
    for case in ("MM_case", "DECODE_case"):
        step = _choose(workspace, entry(case))
        assert step["choice"] == "dense_panel_epochs", case  # ahead of barrier_cleanup
        assert "--mechanism dense_panel_epochs" in step["apply"] and "applies if" not in step["choice_text"]
        assert step["read"].endswith("pass-library/reference/synchronization.md")
    assert _choose(workspace, entry("CV_case"))["choice"] == EPOCHS["id"]
    # without the fence bottleneck the epoch mechanism is not what a product is pointed at
    assert _choose(workspace, entry("MM_case", fence=False))["choice"] != "dense_panel_epochs"


def test_an_imported_epoch_law_without_calls_remains_an_untried_alternative(workspace):
    suggestions = copy.deepcopy(FIXTURE["suggestions"])
    suggestions["all_mechanisms"] += [DENSE_EPOCHS]
    (workspace / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(suggestions))
    (workspace / "submission" / "backend.py").write_text(
        "from merlin.compile.scheduling.dense_panel_epochs import DensePanelEpochLaw\n"
    )
    step = _choose(workspace, entry("MM_case"))
    assert step["choice"] == "dense_panel_epochs"
    assert step["source_call_sites"] == []
    assert "runtime use unconfirmed" in step["choice_text"]


def test_the_catalogue_describes_both_epoch_mechanisms_as_fence_removal():
    from merlin.common.paths import repo_root

    root = repo_root()
    catalog = json.loads((root / "docs/design/joint_optimization_catalog.json").read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in catalog["mechanisms"]}
    for name in ("convolution_compute_epochs", "dense_panel_epochs"):
        text = " ".join(str(rows[name].get(key, "")) for key in ("mechanism", "applicability")).lower()
        assert rows[name]["facet"] == "synchronization" and any(term in text for term in N.FENCE_TERMS)
    dense = rows["dense_panel_epochs"]
    assert all(word in dense["mechanism"].lower() for word in ("drain", "epoch", "fence"))
    assert dense["compiler"]["symbol"] == "DensePanelEpochLaw" and "epoch_law" in dense["compiler"]["source_summary"]
    operations = json.loads((root / "docs/design/operation_optimization_catalog.json").read_text(encoding="utf-8"))
    by_family = {row["id"]: row for row in operations["families"]}
    for family in ("contraction", "matrix_vector"):
        ids = {m["id"] for m in by_family[family]["mechanisms"]}
        assert {"dense_panel_epochs", "heavy-pass-recipes/dense_epoch_recipe.py"} <= ids, family
    assert "dense_panel_epochs" not in {m["id"] for m in by_family["convolution"]["mechanisms"]}


def test_missing_adapter_is_a_concrete_construction_task_not_an_optional_catalogue(workspace):
    step = _choose(workspace, entry('MM_case'))
    assert step['construction_task']['kind'] == 'bind_source_effects'
    assert 'construct shared source/effect adapter' in step['choice_text']
    assert 'parallel_schedule_recipe.py --adapter AUTHORED_OWNER.py' in step['construction_task']['command']
    (workspace/'submission'/'tools').mkdir()
    (workspace/'submission'/'tools'/'library_pass_adapter.py').write_text('not_imported = True')
    step = _choose(workspace, entry('MM_case'))
    assert step['construction_task']['kind'] == 'declare_alternatives'
    assert 'declare finite legal alternatives' in step['choice_text']


def test_printed_progress_is_only_for_the_reply_source_and_model(workspace):
    from merlin_experiments.phase1.feedback import idle_tuning as U

    model = dict(selection_sha256='s', model_sha256='m', timing_mode='untuned', timing_epoch=0)
    progress = dict(source_sha256='current', ready_baselines=2, declared_candidates=3,
        outstanding_candidate_cases=4, requests_in_flight=2, status='running', model_domains=[model])
    (workspace/'qa').mkdir()
    (workspace/U.LEDGER).write_text(json.dumps(dict(source_sha256='current', progress=progress)))
    rows = [dict(capsule='MM_case', performance=model)]
    document = dict(submission_sha256='current')
    brief = dict(largest_headroom=[entry('MM_case')])
    result = N.block(document, rows, brief, workspace=workspace)
    assert '2 ready baselines, 3 candidates, 4 candidate/case checks outstanding, 2 requests in flight' in result['lines'][0]
    assert result['idle_sweep_progress'] == progress
    rows[0]['performance'] = {**model, 'model_sha256': 'older'}
    result = N.block(document, rows, brief, workspace=workspace)
    assert result['lines'][0] == N.HEADER and 'idle_sweep_progress' not in result
