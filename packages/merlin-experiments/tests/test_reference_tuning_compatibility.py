"""Reference requests use current bundle planning and immutable result records.

Synthetic timing rows test identity and feedback only; no author or device runs.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from merlin_experiments.phase1.brokers import selfcheck
from merlin_experiments.phase1.feedback import idle_tuning as tuning
from merlin_experiments.phase1.feedback import idle_tuning_results as results
from merlin_experiments.phase1.feedback import reference_experiments as reference

from merlin.verify.lean_contract import document_digest


def _write(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _performance(cycles=200, *, model="model", elf="baseline-elf"):
    return dict(status="PROVISIONAL", complete_event_accounting=True,
                estimated_whole_soc_cycles=cycles, selection_sha256="selected-model",
                model_sha256=model, timing_mode="untuned", timing_epoch=0,
                scope_sha256="scope", comparison_domain_sha256="domain", elf_sha256=elf,
                structural_roofline=dict(structural_compute_floor_cycles=100))


@pytest.fixture
def selected(tmp_path, monkeypatch):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "settings.json").write_text(json.dumps(dict(retain=False)))
    (submission / "TUNING_KNOBS.json").write_text(json.dumps(dict(
        schema="merlin.tuning_knobs.v1", settings_file="settings.json",
        knobs=dict(retain=dict(values=[False, True], mechanism="schedule.loop_transformed")))))
    source = selfcheck._submission_digest(submission)
    context = dict(compiler_sha256=source, source_sha256="b" * 64, environment_sha256="c" * 64,
                   reference_sha256="d" * 64, visibility="reference_guided",
                   evaluator=dict(identity="synthetic-protocol-owner", sha256="e" * 64),
                   tuning=dict(current=dict(retain=False), domains=dict(retain=dict(values=[False, True]))))
    identity = document_digest(context)
    request = dict(request_id="selected-sibling-trial", role="variant", submission="submission",
                   argv=["python3", "agent_selfcheck.py", "--submission", "submission",
                         "--capsules", "sibling", "--suite", "scalability", "--sim", "spike"],
                   broker_payload=dict(suite="scalability"), settings=dict(retain=True),
                   settings_sha256=document_digest(dict(retain=True)), tuning_overrides=dict(retain=True),
                   changed_factors=["retain"], capsule="sibling", split="training", status="ready",
                   context_identity=identity, baseline_compiler_sha256=source, sim="spike", tiers="")
    task = dict(task_id="selected-region", kind="mechanism_experiment", goals=["schedule.loop_transformed"],
                status="ready", summary="Check the selected source region", requests=[request],
                evidence=[], hypotheses=[], prior_attempts=[], acceptance={},
                next_action=dict(route="existing_selfcheck_tuning_broker", depends_on=[], gaps=[]))
    packet = dict(schema=reference.PACKET_SCHEMA, context=context,
                  execution_context_identity=identity, tasks=[task])
    packet["identity"] = document_digest(packet)
    packet_path = tmp_path / "packet.json"
    packet_sha = _write(packet_path, packet)
    selection = tmp_path / "selection.json"
    sha = _write(selection, dict(schema=reference.SCHEMA,
        packet=dict(path=str(packet_path), sha256=packet_sha), current_context=context))
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    for name in (tuning.ENV, *tuning.KNOBS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MERLIN_IDLE_TUNE_MAX_REQUESTS", "8")
    monkeypatch.setenv("MERLIN_IDLE_TUNE_MAX_PER_SOURCE", "8")
    tuner = tuning.IdleTuner(tmp_path)
    # The external family roll-up supplies a real representative/sibling choice.
    monkeypatch.setattr(tuner, "_rollup_bundles", lambda *args: [dict(
        label="related source cases", representative="representative", siblings=["sibling", "unrelated"])])
    known = {name: [(1, source, _performance(), True)] for name in ("representative", "sibling", "unrelated")}
    with reference.invocation(selection, sha):
        yield dict(tuner=tuner, channel=channel, source=source, known=known,
                   cohort=list(known), l2=dict.fromkeys(known, "pass"), packet=packet)


def _plan(selected):
    return selected["tuner"].plan(selected["channel"], selected["source"], selected["cohort"],
                                  selected["l2"], selected["known"], selection="selected-model")


def _submit(selected):
    job, = _plan(selected)
    rid = selected["tuner"].submit(selected["channel"], job, job["cases"], selected["source"])
    request = json.loads((selected["channel"] / f"req_{rid}.json").read_text())
    return job, rid, request


def _response(selected, rid, request):
    path = selected["channel"] / f"resp_{rid}.json"
    _write(path, dict(selfcheck_request_id=rid, submission_sha256=selected["source"],
        selfcheck_submission="submission", graded_at_unix_ns=3,
        variant=dict(kind="knob_overrides", sha256="f" * 64,
                     tuning_overrides=dict(retain=True), settings_file="settings.json"),
        tuning_experiment=request["tuning_experiment"], cases=[dict(capsule="sibling", status="pass",
            tiers=dict(L2="pass"), performance=_performance(180, elf="candidate-elf"))]))
    return path


def test_guided_sibling_is_planned_alone_and_cannot_expand_to_unrelated_cases(selected):
    job, = _plan(selected)
    assert job["cases"] == ["sibling"]
    assert job["candidate"]["key"] == results.knob_key(dict(retain=True))
    assert job["candidate"]["reference_experiment"]["task_id"] == "selected-region"
    with pytest.raises(ValueError, match="exact selected cases"):
        selected["tuner"].submit(selected["channel"], job, ["representative", "sibling"], selected["source"])
    assert not list(selected["channel"].glob("req_*.json"))


def test_canonical_guided_overrides_are_not_replanned_while_live_or_after_completion(selected):
    job, rid, request = _submit(selected)
    canonical = results.knob_key(dict(retain=True))
    assert job["candidate"]["key"] == canonical
    assert (canonical, "sibling") in selected["tuner"]._tried(
        selected["channel"], selected["known"], "selected-model")[1]
    assert _plan(selected) == []
    _response(selected, rid, request)
    assert _plan(selected) == []
    assert len(list(selected["channel"].glob("req_*.json"))) == 1


@pytest.mark.parametrize("comparable", [True, False])
def test_reference_identity_survives_parse_record_and_compact_feedback(selected, comparable):
    _, rid, request = _submit(selected)
    path = _response(selected, rid, request)
    metadata = request["tuning_experiment"]["reference_experiment"]
    parsed = results.parse(path)
    assert parsed["reference_experiment"] == metadata
    known = selected["known"]
    if not comparable:
        known = {"sibling": [(1, selected["source"], _performance(model="another-model"), True)]}
    record, = results.records([parsed], known, selected["source"], variant_digests={}, now_ns=4)
    assert record["reference_experiment"] == metadata
    assert results.compact(record)["reference_experiment"] == metadata
    assert record["measured_cycles_before"] == (200 if comparable else None)
    assert record["origin"] == "idle_tuning"
    assert record["candidate"]["key"] == results.knob_key(dict(retain=True))
