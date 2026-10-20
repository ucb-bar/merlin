"""Existing lean dashboard shows broker timing without inventing native utilization."""

import json

from merlin_experiments.tracking import phase1_detail, phase1_views, records


def test_recorded_resource_timing_and_missing_records(tmp_path):
    (tmp_path / "simreq_one.json").write_text(json.dumps({"sim": "gsim"}))
    (tmp_path / "simrun_one").write_text("running")
    (tmp_path / "simtiming_one.json").write_text(
        json.dumps(
            {
                "schema": "simjob_resource_timing_v1",
                "job_id": "one",
                "queue_wait_s": 12.5,
                "reserved_verilator_slots": 1,
            }
        )
    )
    (tmp_path / "simreq_two.json").write_text("{}")
    jobs = phase1_detail.jobs([tmp_path], records.Inventory())
    html = phase1_views.timeline_section(None, {"jobs": jobs}, 100)
    assert "1 active, 1 queued" in html
    assert "legacy Verilator reservations: 1" in html
    assert "mean queue wait: 12.5s" in html
    assert "not measured emulator utilization" in html
    assert next(j for j in jobs if j["id"] == "one")["resource_timing"] == {
        "queue_wait_s": 12.5,
        "reserved_verilator_slots": 1,
    }
    assert next(j for j in jobs if j["id"] == "two")["resource_timing"] == {}
    # Wrong owner/schema cannot turn a record into a measured zero.
    (tmp_path / "simtiming_two.json").write_text(
        json.dumps(
            {
                "schema": "foreign",
                "job_id": "two",
                "queue_wait_s": 0,
            }
        )
    )
    assert (
        next(j for j in phase1_detail.jobs([tmp_path], records.Inventory()) if j["id"] == "two")["resource_timing"]
        == {}
    )
