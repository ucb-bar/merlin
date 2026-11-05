"""``--reference-analysis verify`` over synthetic owned before/after selfcheck outputs.

The selected decoder is replaced by a synthetic role roster and operand protocol, and
``lowered.llvm.mlir`` holds a synthetic decoded trace, so these tests exercise the verdict,
storage, sweep, path and transport logic without target facts.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments.phase1.brokers import selfcheck as broker
from merlin_experiments.phase1.feedback import candidate_verification as V
from merlin_experiments.phase1.feedback import reference_bank
from merlin_experiments.phase1.tools import selfcheck as client

ROLES = {"LD": ["operand_load"], "LD2": ["operand_load"], "W": ["weight_load"], "MM": ["accumulate"],
         "ST": ["readout", "commit"]}
PROTOCOL = dict(address_field=dict(field="addr"), roles=dict(classes={
    "LD": dict(fields=[dict(name="dst", bundle="b", field="addr", operand="rs2")]),
    "LD2": dict(fields=[dict(name="dst", bundle="b", field="addr", operand="rs2")]),
    "W": dict(fields=[dict(name="w", bundle="b", field="addr", operand="rs1")]),
}))
DECODER = dict(target="synthetic", semantics=object(), structure=dict(instruction_roles=ROLES),
               decoder_inputs=[dict(path="synthetic", sha256="0" * 64)])
STRUCTURE = dict(instruction_roles=ROLES, operand_protocol=PROTOCOL)


def site(klass, executions=1, **fields):
    return {"class": klass, "decoded": fields, "executions": executions}


def trace(*rows):
    return {"instructions": list(rows), "summary": {"counted_loops": {"basis": "synthetic", "functions": []}}}


RELOADING = trace(site("LD", 4, dst=0), site("W", 1, w=64), site("MM", 2), site("W", 1, w=80), site("MM", 2),
                  site("ST", 1))
RETAINING = trace(site("LD", 4, dst=0), site("W", 1, w=64), site("MM", 2), site("MM", 2), site("ST", 1))


@pytest.fixture
def selected(monkeypatch):
    monkeypatch.setattr(V, "_selected", lambda *a, **k: (DECODER, STRUCTURE))
    monkeypatch.setattr(V, "decode_lowered", lambda text, decoder: json.loads(text))
    monkeypatch.setattr(reference_bank, "selected_pin", lambda: "f" * 64)


def case(workspace, relative, value, *, receipts=None, tensors=None):
    path = workspace / relative
    path.mkdir(parents=True, exist_ok=True)
    (path / V.LOWERED).write_text(json.dumps(value))
    params = {"compiler_lowering": {"dense_schedules": receipts}} if receipts else {}
    (path / V.COMMANDS).write_text(json.dumps(dict(tensors=tensors or {}, commands=[], params=params)))
    return relative


def run(workspace, before, after, **fields):
    value = dict(operation="verify", path=before, candidate=dict(path=after), **fields)
    return V.handle(value, base=dict(schema="merlin.reference_analysis.v1"), workspace=workspace)


def full(workspace, reply):
    return json.loads((workspace / reply["detail"]["path"]).read_text())["report"]


def test_applied_retention_reduces_proved_loads_per_compute(tmp_path, selected):
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RELOADING)
    after = case(tmp_path, "selfcheck_out/requests/b/G_mm", RETAINING)
    reply = run(tmp_path, before, after, mechanism="operand_retention")
    assert reply["status"] == "analyzed" and reply["report_in_detail"] is True
    row = full(tmp_path, reply)["pairs"][0]
    assert row["intervention_applied"]["status"] == V.APPLIED
    assert row["intervention_applied"]["evidence"]["after"]["value"]["retained"] == 1
    assert row["traffic_reduced"] == dict(status="reduced", before=1.5, after=1.25,
                                          basis="proved dynamic loads per compute")
    assert row["traffic_delta"]["per_role"]["weight_load"]["dynamic_delta"] == -1
    assert row["numerically_correct"]["status"] == "unknown"


def test_identical_stream_is_an_emitted_fallback_even_when_correct(tmp_path, selected):
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RELOADING)
    after = case(tmp_path, "selfcheck_out/requests/b/G_mm", RELOADING)
    saved = tmp_path / 'artifacts' / 'after.json'
    saved.parent.mkdir()
    saved.write_text(json.dumps(dict(submission_sha256="1" * 64, per_capsule=[dict(
        capsule="G_mm", **{"pass": True}, numerical_execution_passed=True, numeric_consistency_passed=True)])))
    row = full(tmp_path, run(tmp_path, before, after, mechanism="operand_retention",
                             verdicts=[(Path('artifacts') / 'after.json').as_posix()]))["pairs"][0]
    assert row["numerically_correct"]["status"] == "correct"
    assert row["intervention_applied"]["status"] == V.FALLBACK
    assert row["intervention_applied"]["reason"] == "emitted_stream_identical_to_before"
    assert row["traffic_reduced"]["status"] == "not_reduced"


def test_unchanged_declared_effect_is_a_fallback_and_undeclared_change_is_unknown(tmp_path, selected):
    changed = trace(*RELOADING["instructions"], site("ST", 1))
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RELOADING)
    after = case(tmp_path, "selfcheck_out/requests/b/G_mm", changed)
    row = full(tmp_path, run(tmp_path, before, after, mechanism="operand_retention"))["pairs"][0]
    assert row["intervention_applied"]["reason"] == "declared_mechanism_effect_unchanged"
    row = full(tmp_path, run(tmp_path, before, after))["pairs"][0]
    assert row["intervention_applied"]["status"] == V.UNKNOWN


def test_unproved_trip_count_keeps_dynamic_traffic_unknown(tmp_path, selected):
    unproved = trace(site("LD", None, dst=0), site("W", 1, w=64), site("MM", 2), site("MM", 2))
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RELOADING)
    after = case(tmp_path, "selfcheck_out/requests/b/G_mm", unproved)
    row = full(tmp_path, run(tmp_path, before, after))["pairs"][0]
    assert row["traffic_reduced"]["status"] == "unknown"
    assert row["traffic_reduced"]["static_after"] == 1.0
    assert row["traffic_delta"]["per_role"]["operand_load"]["dynamic_delta"] is None
    assert row["traffic_delta"]["per_role"]["weight_load"]["dynamic_delta"] == -1


def test_overlapping_operand_load_windows_are_a_storage_conflict(tmp_path, selected):
    clobber = trace(site("LD", 1, dst=0), site("LD2", 1, dst=0), site("W", 1, w=64), site("MM", 1))
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RELOADING)
    after = case(tmp_path, "selfcheck_out/requests/b/G_mm", clobber)
    row = full(tmp_path, run(tmp_path, before, after))["pairs"][0]
    assert row["storage_valid"]["before"] == "unknown"  # verifier not run on either; windows alone are lexical
    assert row["storage_valid"]["after"] == "operand_window_overlap_observed"
    overlap = row["storage_valid"]["after_detail"]["local_address_windows"]["overlaps"][0]
    assert {overlap["first"]["instruction"], overlap["second"]["instruction"]} == {"LD", "LD2"}


def test_sweep_table_pairs_by_capsule_and_reports_blockers_without_extrapolation(tmp_path, selected):
    capsules = tmp_path / "public"
    for name, (m, k, n) in dict(G_a=(4, 8, 8), G_b=(1, 8, 8), G_c=(64, 4096, 64)).items():
        (capsules / name).mkdir(parents=True)
        (capsules / name / "capsule.yaml").write_text(json.dumps(dict(
            operation=dict(op="matmul"), inputs=[dict(name="A", role="input", shape=[m, k], dtype="i8"),
                                                 dict(name="W", role="weight", shape=[k, n], dtype="i8")])))
    refusal = [dict(schema="s.v1", requested_pipeline="retained", selected_pipeline="compact",
                    pipeline_refusal="single product has no retained operand reuse")]
    for name in ("G_a", "G_b"):
        case(tmp_path, "selfcheck_out/requests/old/" + name, RELOADING)
    case(tmp_path, "selfcheck_out/requests/new/G_a", RETAINING)
    case(tmp_path, "selfcheck_out/requests/new/G_b", RELOADING, receipts=refusal)
    case(tmp_path, "selfcheck_out/requests/new2/G_c", RETAINING)
    value = dict(operation="verify", path="selfcheck_out/requests/old",
                 candidate=dict(path="selfcheck_out/requests/new"), mechanism="operand_retention",
                 sweep=["selfcheck_out/requests/new", "selfcheck_out/requests/new2"])
    reply = V.handle(value, base={}, workspace=tmp_path, capsules_root=capsules,
                     facts=_facts(tmp_path, capacity=4096))
    sweep = full(tmp_path, reply)["sweep"]
    rows = {row["capsule"]: row for row in sweep["rows"]}
    assert rows["G_a"]["intervention_applied"]["status"] == V.APPLIED
    assert rows["G_b"]["intervention_applied"]["status"] == V.FALLBACK
    assert rows["G_b"]["blockers"]["legality"][0]["value"] == "single product has no retained operand reuse"
    assert rows["G_c"]["intervention_applied"]["status"] == V.UNKNOWN and rows["G_c"]["baseline"] is None
    assert [t["tensor"] for t in rows["G_c"]["blockers"]["capacity"]] == ["A", "W"]
    assert rows["G_a"]["shape"]["basis"] == "public_capsule_declaration"
    region = sweep["applicability_region"]
    assert region["counts"] == {V.APPLIED: 1, V.FALLBACK: 1, V.UNKNOWN: 1}
    assert region["applied"][0]["shapes"] == {"A": [4, 8], "W": [8, 8]}
    assert region["extrapolation"].startswith("none")
    assert reply["report"]["sweep"]["counts"] == region["counts"]


def _facts(root, *, capacity):
    path = root / "facts.json"
    path.write_text(json.dumps(dict(facts=dict(memory_address_spaces=[dict(name="local", banks=1, depth=1,
                                                                             bytes=capacity)]))))
    return path


@pytest.mark.parametrize("path", ["reference-kernels/x", "cca-evidence/bank.json", "../selfcheck_out/x",
                                  "/etc/passwd", "variants", "qa/cases", "selfcheck_out/link/G_mm"])
def test_non_owned_paths_are_refused_before_reading(tmp_path, selected, path):
    (tmp_path / "selfcheck_out").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "selfcheck_out/link").symlink_to(tmp_path / "elsewhere")
    after = case(tmp_path, "selfcheck_out/requests/b/G_mm", RELOADING)
    request = dict(action="reference-analysis", reference_analysis=dict(
        operation="verify", path=path, candidate=dict(path=after)))
    reply = V.broker_response(request, workspace=tmp_path, facts=None, compiler_sha256="0" * 64, rid="r",
                              started_ns=time.time_ns())
    assert reply["status"] == "unknown" and reply["reason"] in ("verify_requires_owned_path", "verify_input_missing")


def test_verify_is_served_only_in_a_bank_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_bank, "selected_pin", lambda: None)
    request = dict(action="reference-analysis", reference_analysis=dict(
        operation="verify", path="selfcheck_out/a", candidate=dict(path="selfcheck_out/b")))
    reply = broker._reference_analysis_response(request, workspace=tmp_path, facts=None, compiler_sha256="0" * 64,
                                                rid="r", started_ns=time.time_ns())
    # Outside a bank arm verify is refused like every other reference-analysis operation.
    assert reply["status"] == "denied" and reply["reason"] == "reference_guidance_disabled"
    assert reply["official"] is False


def test_unknown_mechanism_names_are_refused(tmp_path, selected):
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RELOADING)
    with pytest.raises(ValueError, match="verify_mechanism_not_a_derived_effect"):
        run(tmp_path, before, before, mechanism="something_else")
    assert "operand_retention" in V.mechanisms() and "transfer_tiles" in V.mechanisms()


def test_copied_client_reaches_verify_through_the_broker(tmp_path, selected):
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RELOADING)
    after = case(tmp_path, "selfcheck_out/requests/b/G_mm", RETAINING)
    shim = tmp_path / "agent_selfcheck.py"
    shutil.copyfile(client.__file__, shim)
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    argv = [sys.executable, str(shim), "--reference-analysis", "verify", "--before", before, "--after", after,
            "--mechanism", "operand_retention", "--sweep", "selfcheck_out/requests/b", "--wait-budget", "10",
            "--baseline", "selfcheck_out/requests/a"]
    process = subprocess.Popen(argv, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline, requests = time.monotonic() + 10, []
        while not requests and time.monotonic() < deadline:
            requests = list(channel.glob("req_*.json"))
            time.sleep(0.01)
        assert requests
        request = json.loads(requests[0].read_bytes())
        assert request["action"] == "reference-analysis"
        assert request["reference_analysis"]["candidate"] == dict(path=after)
        assert request["reference_analysis"]["sweep"] == ["selfcheck_out/requests/b"]
        assert request["reference_analysis"]["baseline"] == "selfcheck_out/requests/a"
        rid = request["request_id"]
        response = broker._reference_analysis_response(request, workspace=tmp_path, facts=None,
                                                       compiler_sha256="0" * 64, rid=rid, started_ns=time.time_ns())
        (channel / f"resp_{rid}.json").write_text(json.dumps(response))
        (channel / f"done_{rid}").write_text("ok")
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stdout + stderr
        result, _ = json.JSONDecoder().raw_decode(stdout)
        assert result["status"] == "analyzed"
        assert result["report"]["pairs"][0]["intervention_applied"]["status"] == V.APPLIED
        assert result["report"]["sweep"]["counts"][V.APPLIED] == 1
        assert result["report"]["pairs"][0]["regressions_vs_baseline"]["traffic_regressed"] is False
        assert result["report"]["sweep"]["rows"][0]["fell_back_to_seed_or_before"] is False
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_client_refuses_verify_flags_without_verify(tmp_path):
    shim = tmp_path / "agent_selfcheck.py"
    shutil.copyfile(client.__file__, shim)
    done = subprocess.run([sys.executable, str(shim), "--before", "selfcheck_out/a"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 2 and "require --reference-analysis verify" in done.stderr


# --- stream facts, regression flags, seed baseline and identical emitted code ----------------------

def _observed(value):
    return dict(status="observed", value=value, scope="proved_dynamic_command_stream", evidence=[])


def _stream(input_cols, weight_overlap, band_run):
    return dict(
        role_transfer_width=_observed(dict(by_role=dict(
            input=[dict(rows=4, cols=input_cols, count=64 // input_cols, elements=4 * input_cols)],
            weight=[dict(rows=4, cols=16, count=8, elements=64)]))),
        operand_buffering=_observed(dict(by_role=dict(weight=dict(disjoint=2, alternating=True)))),
        issue_overlap=_observed(dict(loads_with_compute_before_barrier=dict(weight=dict(count=weight_overlap,
                                                                                       total=8)),
                                     compute_run_median=band_run, output_group_extent=dict(rows=4, cols=16),
                                     compute_run_between_load_bands=band_run, loads_per_compute_run=0.25)))


@pytest.fixture
def streams(monkeypatch):
    """Stream facts per build directory (the dynamic stepper is another module's job)."""
    table = {}

    def facts(commands, structure, evidence_id, reason=None):
        for key, value in table.items():
            if key in evidence_id:
                return value
        return {name: dict(status="unknown", reason="no_proved_command_stream") for name in
                ("role_transfer_width", "operand_buffering", "issue_overlap")}

    monkeypatch.setattr(V, "stream_facts", facts)
    return table


def test_preload_is_counted_apart_from_data_movement(tmp_path, selected):
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RELOADING)
    after = case(tmp_path, "selfcheck_out/requests/b/G_mm", RETAINING)
    row = full(tmp_path, run(tmp_path, before, after))["pairs"][0]
    dynamic = row["traffic_delta"]["loads_per_compute"]["dynamic"]
    assert dynamic["data_movement_loads_per_compute"] == dict(before=1.0, after=1.0)
    assert dynamic["preloads_per_compute"] == dict(before=0.5, after=0.25)
    flags = row["regressions"]
    assert flags["traffic_regressed"]["status"] is False
    assert flags["overlap_regressed"]["status"] == "unknown" and flags["width_narrowed"]["status"] == "unknown"


def test_regression_flags_against_before_and_seed_baseline(tmp_path, selected, streams):
    case(tmp_path, "selfcheck_out/requests/seed/G_mm", RELOADING)
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RETAINING)
    after = case(tmp_path, "selfcheck_out/requests/b/G_mm", trace(site("LD", 6, dst=0), site("W", 1, w=64),
                                                                    site("MM", 4), site("ST", 1)))
    streams.update({"requests/seed": _stream(16, 6, 64), "requests/a": _stream(16, 6, 64),
                    "requests/b": _stream(4, 0, 256)})
    reply = run(tmp_path, before, after, baseline="selfcheck_out/requests/seed")
    row = full(tmp_path, reply)["pairs"][0]
    for flags in (row["regressions"], row["against_baseline"]["regressions"]):
        assert flags["traffic_regressed"]["status"] is True
        assert flags["overlap_regressed"]["status"] is True
        assert [r["role"] for r in flags["overlap_regressed"]["roles"]] == ["weight"]
        assert flags["width_narrowed"]["status"] is True
        assert [r["role"] for r in flags["width_narrowed"]["roles"]] == ["input"]
        assert flags["prefetch_coarsened"]["status"] is True
        assert flags["output_group_narrowed"]["status"] is False
    assert row["against_baseline"]["baseline"].endswith("requests/seed/G_mm")
    summary = reply["report"]["pairs"][0]
    assert summary["regressions_vs_baseline"]["width_narrowed"] is True
    assert row["fell_back_to_seed_or_before"]["status"] is False


def test_sweep_marks_rows_whose_emitted_code_is_the_seed_or_before(tmp_path, selected):
    tensors = lambda m: {"A": dict(shape=[m, 16], dtype="int8", role="input")}  # noqa: E731
    seed_dir, after_dir = "selfcheck_out/requests/seed", "selfcheck_out/requests/b"
    for m in (16, 48):
        case(tmp_path, f"{seed_dir}/G_m{m}", RELOADING, tensors=tensors(m))
    case(tmp_path, f"{after_dir}/G_m16", RETAINING, tensors=tensors(16))
    case(tmp_path, f"{after_dir}/G_m48", RELOADING, tensors=tensors(48))  # the same emitted bytes as the seed
    before = case(tmp_path, "selfcheck_out/requests/a/G_m16", RELOADING, tensors=tensors(16))
    reply = run(tmp_path, before, f"{after_dir}/G_m16", sweep=[after_dir], baseline=seed_dir)
    sweep = full(tmp_path, reply)["sweep"]
    rows = {r["capsule"]: r for r in sweep["rows"]}
    assert rows["G_m48"]["fell_back_to_seed_or_before"]["status"] is True
    assert rows["G_m48"]["fell_back_to_seed_or_before"]["identical_to"] == ["seed_baseline"]
    assert rows["G_m16"]["fell_back_to_seed_or_before"]["status"] is False
    region = sweep["applicability_region"]["identical_emitted_code"]
    assert region["counts"] == dict(fell_back=1, emitted_new_code=1, unknown=0)
    assert region["observed_values_by_axis"]["A[0]"] == dict(fell_back=[48], emitted_new_code=[16])
    assert "none" in region["extrapolation"]
    assert {r["capsule"]: r["fell_back_to_seed_or_before"] for r in reply["report"]["sweep"]["rows"]} == \
        dict(G_m16=False, G_m48=True)


def test_baseline_must_be_an_owned_path(tmp_path, selected):
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RELOADING)
    with pytest.raises(ValueError, match="verify_requires_owned_path"):
        run(tmp_path, before, before, baseline="../elsewhere")
    with pytest.raises(ValueError, match="invalid_analysis_request"):
        V.validate(dict(operation="verify", path=before, candidate=dict(path=before), baseline=3))


def test_copied_client_sends_the_baseline(tmp_path):
    shim = tmp_path / "agent_selfcheck.py"
    shutil.copyfile(client.__file__, shim)
    done = subprocess.run([sys.executable, str(shim), "--baseline", "selfcheck_out/seed"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 2 and "--baseline" in done.stderr


def test_further_regression_flags_are_feature_detected(tmp_path, selected, streams):
    before = case(tmp_path, "selfcheck_out/requests/a/G_mm", RETAINING)
    after = case(tmp_path, "selfcheck_out/requests/b/G_mm", RELOADING)
    further = lambda rate, config, same, contiguous, on: dict(  # noqa: E731
        host_issue_rate=_observed(dict(rate=rate)),
        reconfiguration=_observed(dict(config_per_block=config, placement="hoisted")),
        local_capacity_use=_observed(dict(rows=32, capacity=64, same_bank_halves=same)),
        dram_contiguity=_observed(dict(contiguous=contiguous)), epilogue_placement=_observed(dict(on_accelerator=on)))
    streams.update({"requests/a": dict(_stream(16, 6, 64), **further(0.5, 1, False, True, True)),
                    "requests/b": dict(_stream(16, 6, 64), **further(0.25, 2, True, False, False))})
    flags = full(tmp_path, run(tmp_path, before, after))["pairs"][0]["regressions"]
    for name in ("issue_rate_regressed", "reconfiguration_increased", "bank_conflict_introduced", "contiguity_lost",
                 "epilogue_moved_off_accelerator"):
        assert flags[name]["status"] is True, name
    assert flags["capacity_underused"]["status"] is False
    streams.clear()
    plain = full(tmp_path, run(tmp_path, before, after))["pairs"][0]["regressions"]
    assert plain["contiguity_lost"]["status"] == "unknown"


def _attributed(operands, hit):
    """Deriver layout: one movement role whose loads are attributed to ``operands`` ({operand: cols})."""
    def width(cols):
        return dict(transfers=4, shapes=[dict(rows=4, cols=cols, count=4, elements=4 * cols)])

    return dict(
        role_transfer_width=_observed(dict(data_movement=dict(operand_load=dict(
            transfers=4 * len(operands), shapes=[s for c in operands.values() for s in width(c)["shapes"]],
            by_operand={k: width(c) for k, c in operands.items()})))),
        operand_buffering=dict(status="unknown", reason="no_proved_command_stream"),
        issue_overlap=_observed(dict(loads=4 * len(operands), loads_with_compute_before_barrier=hit,
                                     by_operand={k: dict(loads=4, loads_with_compute_before_barrier=hit)
                                                 for k in operands})))


def test_verify_compares_at_role_level_when_operand_attribution_differs():
    before = dict(evidence_id="before", stream_facts=_attributed(dict(a0=16, a1=16), 1.0))
    after = dict(evidence_id="after", stream_facts=_attributed(dict(unattributed=4), 0.0))
    changes = V.stream_changes(before, after)
    fallback = {(r["effect"], r["role"]) for r in changes if r.get("attribution") == "role_level_fallback"}
    assert fallback == {("role_transfer_width", "operand_load"), ("issue_overlap", "all")}
    flags = V.regressions(before, after, changes)
    assert flags["width_narrowed"]["status"] is True and flags["overlap_regressed"]["status"] is True
    same = V.stream_changes(before, dict(before, evidence_id="after"))
    assert not any(r.get("attribution") == "role_level_fallback" for r in same)
    assert {r.get("attribution") for r in same if r["role"] in ("a0", "a1", "operand_load:a0", "operand_load:a1")} \
        == {"argument_register"}


# --- verify through the real client: entry symbol, the author's own object, host compile, stream mechanisms ---

def test_a_declared_stream_mechanism_is_judged_from_the_proved_streams():
    def build(emitted, proved, overlap):
        # The static IR site list is the same on every build: loop structure alone differs.
        return dict(stream=dict(status="decoded", sha256="same-sites"), effects={}, emitted_sha256=emitted,
                    proved_stream_sha256=proved,
                    stream_facts=dict(issue_overlap=_observed(dict(loads_with_compute_before_barrier=overlap))))

    applied = V._applied(build("e1", "p1", 0.25), build("e2", "p2", 1.0), "issue_overlap")
    assert applied["status"] == V.APPLIED and applied["evidence"]["basis"] == "stream_facts"
    unchanged = V._applied(build("e1", "p1", 0.25), build("e2", "p2", 0.25), "issue_overlap")
    assert unchanged["status"] == V.FALLBACK and unchanged["reason"] == "declared_mechanism_effect_unchanged"
    same = V._applied(build("e1", "p1", 0.25), build("e2", "p1", 0.25), "issue_overlap")
    assert same["reason"] == "proved_command_stream_identical_to_before"
    assert V._applied(build("e1", "p1", 0.25), build("e1", "p2", 1.0), "issue_overlap")["status"] == V.FALLBACK
    unproved = V._applied(build("e1", "p1", 0.25), build("e2", None, 1.0), "issue_overlap")
    assert unproved["status"] == V.UNKNOWN and "not established on both builds" in unproved["reason"]
    # Static mechanisms keep the IR-order identity check.
    assert V._applied(build("e1", "p1", 0.25), build("e2", "p2", 1.0), "operand_retention")["reason"] == \
        "emitted_stream_identical_to_before"
    assert {"prefetch_distance", "issue_overlap", "output_group_extent"} <= set(V.mechanisms())


def test_object_recipe_reads_the_declared_entry_and_kernel_flags():
    document = dict(harness_abi=dict(entry_symbol="entry_fn"), toolchain=dict(build=dict(
        kernel_stack_frame=dict(entry_symbol="entry_fn"), cflags=["-march=rv64x", "-mabi=lp64x", "-O2", "-g"])))
    recipe = V.object_recipe(document, "synthetic")
    assert recipe["entry_symbol"] == "entry_fn" and recipe["flags"] == ["-march=rv64x", "-mabi=lp64x"]
    assert V.object_recipe({})["entry_symbol"] is None


def test_build_stream_prefers_the_elf_then_one_owned_object_then_the_emitted_ir(tmp_path, monkeypatch):
    seen = []
    def elf_stream(entry, decoder, symbol):
        seen.append((entry[1], symbol))
        return None, "r"

    monkeypatch.setattr(V, "_elf_stream", elf_stream)
    monkeypatch.setattr(V, "_compiled_stream", lambda lowered, decoder, symbol, build: (None, "c", dict(kind="host")))
    case_dir = tmp_path / "selfcheck_out" / "G_mm"
    case_dir.mkdir(parents=True)
    (case_dir / V.LOWERED).write_text("{}")
    (case_dir / "kernel.norelax.o").write_bytes(b"\x7fELF")
    files = V._files(case_dir, "selfcheck_out/G_mm")
    assert files[V.OBJECT][1] == "selfcheck_out/G_mm/kernel.norelax.o"
    _, _, source = V._build_stream(files, b"{}", {}, "entry_fn", None)
    assert source["kind"] == "owned_relocatable_object" and seen[-1] == ("selfcheck_out/G_mm/kernel.norelax.o",
                                                                          "entry_fn")
    (case_dir / "other.o").write_bytes(b"\x7fELF")
    files = V._files(case_dir, "selfcheck_out/G_mm")
    assert V.OBJECT not in files and files[V.OBJECTS_AMBIGUOUS] == ["kernel.norelax.o", "other.o"]
    assert V._build_stream(files, b"{}", {}, "entry_fn", None)[2] == dict(kind="host")
    assert "several relocatable objects" in V._build_stream(files, None, {}, "entry_fn", None)[1]
    assert "no kernel entry symbol" in V._build_stream(files, b"{}", {}, None, None)[1]


def test_emitted_ir_is_compiled_host_side_without_relaxation(tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain

    log = tmp_path / "argv.log"
    tool = tmp_path / "tool"
    tool.write_text(f"#!/bin/sh\necho \"$@\" >> {log}\nfor last; do :; done\necho x > \"$last\"\n")
    tool.chmod(0o755)
    monkeypatch.setattr(toolchain, "clang", lambda: tool)
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: tool)
    monkeypatch.setattr(V, "_elf_stream", lambda entry, decoder, symbol: ([dict(seq=0)], None))
    stream, reason, source = V._compiled_stream(b"module {}", {}, "entry_fn",
                                                dict(flags=["-march=rv64x", "-mabi=lp64x"]))
    assert stream == [dict(seq=0)] and reason is None and source["kind"] == "host_compiled_lowered_mlir"
    translate, compile_line = log.read_text().splitlines()
    assert translate.startswith("--mlir-to-llvmir")
    assert "-mno-relax" in compile_line and "-march=rv64x" in compile_line and "-mabi=lp64x" in compile_line
    assert V._compiled_stream(b"", {}, "entry_fn", dict(flags=[]))[1].endswith("no kernel -march/-mabi recipe")


def test_client_sends_the_candidate_symbol_with_verify(tmp_path):
    shim = tmp_path / "agent_selfcheck.py"
    shutil.copyfile(client.__file__, shim)
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    (tmp_path / "submission").mkdir()
    argv = [sys.executable, str(shim), "--reference-analysis", "verify", "--before", "selfcheck_out/a",
            "--after", "selfcheck_out/b", "--candidate-symbol", "entry_fn", "--wait-budget", "5"]
    process = subprocess.Popen(argv, cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline, requests = time.monotonic() + 10, []
        while not requests and time.monotonic() < deadline:
            requests = list(channel.glob("req_*.json"))
            time.sleep(0.01)
        assert requests, process.communicate(timeout=5)
        request = json.loads(requests[0].read_bytes())
        assert request["reference_analysis"]["candidate"] == dict(path="selfcheck_out/b", symbol="entry_fn")
        V.validate(request["reference_analysis"])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_undeclared_change_is_judged_on_stream_mechanisms_measured_on_both_builds():
    def build(emitted, proved, facts):
        return dict(evidence_id=emitted, stream=dict(status="decoded", sha256=emitted), effects={},
                    emitted_sha256=emitted, proved_stream_sha256=proved, stream_facts=facts)

    seed = build("e0", "p0", _attributed(dict(a0=16, a1=16), 0.0))
    applied = V._applied(seed, build("e1", "p1", _attributed(dict(a0=16, a1=16), 1.0)), None)
    evidence = applied["evidence"]
    assert applied["status"] == V.APPLIED and evidence["changed_mechanisms"] == ["issue_overlap"]
    assert "role_transfer_width" in evidence["unchanged_mechanisms"]
    assert "operand_buffering" in evidence["unmeasured_mechanisms"]
    assert "mechanism" not in evidence  # the bounded reply reports no declared mechanism's effect status
    # Equal measures on a changed stream stay unknown and name what was not measured on both builds.
    unknown = V._applied(seed, build("e1", "p1", _attributed(dict(a0=16, a1=16), 0.0)), None)
    assert unknown["status"] == V.UNKNOWN and "not measured: " in unknown["reason"]
    assert "operand_buffering" in unknown["reason"]
    # An incidental count changing (more transfers of the same width) is not a changed mechanism.
    wider = _attributed(dict(a0=16, a1=16), 0.0)
    wider["role_transfer_width"]["value"]["data_movement"]["operand_load"]["transfers"] = 64
    assert V._applied(seed, build("e1", "p1", wider), None)["status"] == V.UNKNOWN
    blind = V._applied(seed, build("e1", None, {}), None)
    assert blind["status"] == V.UNKNOWN and "no proved command stream on after" in blind["reason"]
    assert V._applied(seed, build("e1", "p0", {}), None)["reason"] == "proved_command_stream_identical_to_before"
    assert V._applied(seed, build("e0", "p1", {}), None)["reason"] == "emitted_stream_identical_to_before"
